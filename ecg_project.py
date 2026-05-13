import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, find_peaks, iirnotch, sosfiltfilt


# Default file is a normal sinus rhythm ECG fragment.
DEFAULT_ECG_FILE = Path("MLII") / "1 NSR" / "100m (0).mat"


def load_ecg_file(file_name):
    mat_file = Path(file_name)
    if not mat_file.exists():
        raise FileNotFoundError(f"Could not find ECG file: {mat_file}")

    mat_data = loadmat(mat_file)
    if "val" not in mat_data:
        available = [key for key in mat_data.keys() if not key.startswith("__")]
        raise ValueError(f"The .mat file does not contain 'val'. Found: {available}")

    ecg = np.asarray(mat_data["val"], dtype=float).flatten()

    ecg = ecg - np.median(ecg)
    return ecg


def add_noise(clean_ecg, fs, baseline_amp, powerline_amp, white_noise_std, powerline_freq, seed):
    np.random.seed(seed)
    t = np.arange(len(clean_ecg)) / fs

    # scale the noise from the signal range so the same settings work on different files.
    ecg_range = np.percentile(clean_ecg, 95) - np.percentile(clean_ecg, 5)

    baseline_wander = baseline_amp * ecg_range * np.sin(2 * np.pi * 0.3 * t)
    powerline_noise = powerline_amp * ecg_range * np.sin(2 * np.pi * powerline_freq * t)
    random_noise = np.random.normal(0, white_noise_std * ecg_range, len(clean_ecg))
    
    # Add the noise components to the clean ECG signal.
    noisy_ecg = clean_ecg + baseline_wander
    noisy_ecg = noisy_ecg + powerline_noise
    noisy_ecg = noisy_ecg + random_noise
    return noisy_ecg

# highpass filter "baseline_wander_removal"
def apply_highpass(ecg, fs):
    sos = butter(3, 0.5, btype="highpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, ecg)
    return filtered

# notch filter "artificial_noise_powerline"
def apply_notch(ecg, fs, powerline_freq):
    b, a = iirnotch(w0=powerline_freq, Q=30, fs=fs)
    filtered = filtfilt(b, a, ecg)
    return filtered

# lowpass filter "physiological_signal_band"
def apply_lowpass(ecg, fs):
    sos = butter(4, 40.0, btype="lowpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, ecg)
    return filtered


def filter_ecg(noisy_ecg, fs, powerline_freq):
    # Step-by-step on purpose so the filtering pipeline is easier to explain.
    after_highpass = apply_highpass(noisy_ecg, fs)
    after_notch = apply_notch(after_highpass, fs, powerline_freq)
    after_lowpass = apply_lowpass(after_notch, fs)
    return after_lowpass


def get_r_peaks(ecg, fs):
    centered = ecg - np.median(ecg)

    signal_range = np.percentile(centered, 99) - np.percentile(centered, 1)
    prominence_from_range = 0.25 * signal_range
    prominence_from_std = 0.8 * np.std(centered)
    prominence = max(prominence_from_range, prominence_from_std, 1.0)

    # 0.25 seconds keeps the detector from counting two peaks too close together.
    minimum_distance = int(0.25 * fs)
    peaks, _ = find_peaks(centered, distance=minimum_distance, prominence=prominence)

    duration = len(ecg) / fs
    heart_rate = len(peaks) / duration * 60
    return peaks, heart_rate


def calculate_snr(clean_ecg, test_ecg):
    signal_power = np.mean(clean_ecg**2)
    error_power = np.mean((clean_ecg - test_ecg) ** 2)
    if error_power == 0:
        return math.inf
    return 10 * math.log10(signal_power / error_power)


def calculate_rmse(clean_ecg, test_ecg):
    error = clean_ecg - test_ecg
    return float(np.sqrt(np.mean(error**2)))


def get_fft(ecg, fs):
    centered = ecg - np.mean(ecg)
    frequencies = np.fft.rfftfreq(len(centered), d=1 / fs)
    magnitude = np.abs(np.fft.rfft(centered)) / len(centered)
    return frequencies, magnitude

# plot 1: ECG comparison before and after filtering
def plot_ecg_comparison(t, clean_ecg, noisy_ecg, filtered_ecg, output_dir):
    output_file = output_dir / "ecg_clean_noisy_filtered.png"

    plt.figure(figsize=(12, 6))
    plt.plot(t, clean_ecg, color="#111827", linewidth=1.0, label="Clean reference ECG")
    plt.plot(t, noisy_ecg, color="#f97316", linewidth=0.8, alpha=0.75, label="Noisy ECG")
    plt.plot(t, filtered_ecg, color="#2563eb", linewidth=1.1, label="Filtered ECG")
    plt.title("ECG Before and After Digital Filtering")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Centered amplitude")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=160)
    plt.close()

    return output_file

# plot 2: zoomed-in ECG segment to show details of filtering effect
def plot_zoomed_ecg(t, clean_ecg, noisy_ecg, filtered_ecg, output_dir):
    output_file = output_dir / "ecg_zoomed_comparison.png"

    start_sec = 2.0
    end_sec = 5.0
    zoom_area = (t >= start_sec) & (t <= end_sec)

    plt.figure(figsize=(12, 6))
    plt.plot(t[zoom_area], clean_ecg[zoom_area], color="#111827", linewidth=1.3, label="Clean reference")
    plt.plot(t[zoom_area], noisy_ecg[zoom_area], color="#f97316", linewidth=0.9, alpha=0.75, label="Noisy")
    plt.plot(t[zoom_area], filtered_ecg[zoom_area], color="#2563eb", linewidth=1.2, label="Filtered")
    plt.title("Zoomed ECG Segment")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Centered amplitude")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=160)
    plt.close()

    return output_file

# plot 3: FFT magnitude spectrum before and after filtering to show frequency-domain effect of filters
def plot_fft(clean_ecg, noisy_ecg, filtered_ecg, fs, output_dir):
    output_file = output_dir / "ecg_fft_before_after.png"

    clean_freq, clean_mag = get_fft(clean_ecg, fs)
    noisy_freq, noisy_mag = get_fft(noisy_ecg, fs)
    filtered_freq, filtered_mag = get_fft(filtered_ecg, fs)

    max_freq_to_show = 90.0
    clean_part = clean_freq <= max_freq_to_show
    noisy_part = noisy_freq <= max_freq_to_show
    filtered_part = filtered_freq <= max_freq_to_show

    plt.figure(figsize=(12, 6))
    plt.plot(clean_freq[clean_part], clean_mag[clean_part], color="#111827", label="Clean")
    plt.plot(noisy_freq[noisy_part], noisy_mag[noisy_part], color="#f97316", alpha=0.8, label="Noisy")
    plt.plot(filtered_freq[filtered_part], filtered_mag[filtered_part], color="#2563eb", label="Filtered")
    plt.title("FFT Magnitude Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=160)
    plt.close()

    return output_file

# plot 4: detected R-peaks on noisy and filtered ECG to show improvement in heart-rate estimation after filtering
def plot_peaks(t, noisy_ecg, filtered_ecg, noisy_peaks, filtered_peaks, noisy_hr, filtered_hr, output_dir):
    output_file = output_dir / "ecg_r_peak_detection.png"
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(t, noisy_ecg, color="#f97316", linewidth=0.9)
    axes[0].scatter(t[noisy_peaks], noisy_ecg[noisy_peaks], color="#991b1b", s=24, label="Detected peaks")
    axes[0].set_title(f"Noisy ECG Peak Detection ({noisy_hr:.1f} bpm)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(t, filtered_ecg, color="#2563eb", linewidth=1.0)
    axes[1].scatter(t[filtered_peaks], filtered_ecg[filtered_peaks], color="#1d4ed8", s=24, label="Detected peaks")
    axes[1].set_title(f"Filtered ECG Peak Detection ({filtered_hr:.1f} bpm)")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=160)
    plt.close()

    return output_file


def list_sample_files(root_folder, limit):
    root = Path(root_folder)
    mat_files = sorted(root.glob("*/*.mat"))

    if len(mat_files) == 0:
        print(f"No .mat files found under {root}")
        return

    print(f"Found {len(mat_files)} ECG fragments under {root}:")
    for file_path in mat_files[:limit]:
        print(f"- {file_path}")

    remaining = len(mat_files) - limit
    if remaining > 0:
        print(f"... {remaining} more not shown")

# printing output summary
def print_summary(args, file_path, clean_ecg, noisy_ecg, filtered_ecg, peaks_info, plot_files):
    clean_peaks, clean_hr = peaks_info["clean"]
    noisy_peaks, noisy_hr = peaks_info["noisy"]
    filtered_peaks, filtered_hr = peaks_info["filtered"]

    snr_before = calculate_snr(clean_ecg, noisy_ecg)
    snr_after = calculate_snr(clean_ecg, filtered_ecg)
    rmse_before = calculate_rmse(clean_ecg, noisy_ecg)
    rmse_after = calculate_rmse(clean_ecg, filtered_ecg)

    noisy_hr_error = abs(noisy_hr - clean_hr)
    filtered_hr_error = abs(filtered_hr - clean_hr)
    rmse_reduction = 100 * (1 - rmse_after / rmse_before)

    print("\nECG Signal Denoising and Heart Rate Estimation")
    print(f"Selected file: {file_path}")
    print(f"Rhythm class: {file_path.parent.name}")
    print(f"Samples: {len(clean_ecg)}")
    print(f"Sampling rate: {args.fs:.1f} Hz")
    print(f"Duration: {len(clean_ecg) / args.fs:.2f} seconds")

    print("\nNoise model:")
    print(f"- Baseline wander: 0.3 Hz, amplitude factor {args.baseline_amp}")
    print(f"- Powerline interference: {args.powerline_freq:.1f} Hz, amplitude factor {args.powerline_amp}")
    print(f"- Gaussian white noise std factor: {args.white_noise_std}")

    print("\nFilter pipeline:")
    print("- High-pass Butterworth: 0.5 Hz cutoff")
    print(f"- Notch filter: {args.powerline_freq:.1f} Hz")
    print("- Low-pass Butterworth: 40 Hz cutoff")

    print("\nQuantitative before/after comparison:")
    print(f"SNR before filtering: {snr_before:.2f} dB")
    print(f"SNR after filtering:  {snr_after:.2f} dB")
    print(f"SNR improvement:      {snr_after - snr_before:.2f} dB")
    print(f"RMSE before filtering: {rmse_before:.2f}")
    print(f"RMSE after filtering:  {rmse_after:.2f}")
    print(f"RMSE reduction:        {rmse_reduction:.1f}%")

    print("\nHeart-rate estimation:")
    print(f"Clean reference: {len(clean_peaks)} peaks, {clean_hr:.1f} bpm")
    print(f"Noisy signal:    {len(noisy_peaks)} peaks, {noisy_hr:.1f} bpm")
    print(f"Filtered signal: {len(filtered_peaks)} peaks, {filtered_hr:.1f} bpm")
    print(f"Noisy HR error:    {noisy_hr_error:.1f} bpm")
    print(f"Filtered HR error: {filtered_hr_error:.1f} bpm")

    print("\nGenerated plots:")
    for plot_file in plot_files:
        print(f"- {plot_file}")

    print("\nSignals and Systems framing:")
    print("- The ECG fragment is modeled as a discrete-time signal x[n].")
    print("- Noise is added as baseline wander, sinusoidal interference, and random noise.")
    print("- Digital filters remove targeted frequency components from the noisy ECG.")
    print("- FFT plots show the frequency-domain effect of filtering.")
    print("- R-peak detection shows how denoising improves heart-rate estimation.")


def build_argument_parser():
    parser = argparse.ArgumentParser(description="ECG denoising and heart-rate estimation project.")
    parser.add_argument("--file", default=str(DEFAULT_ECG_FILE), help="Path to a .mat ECG file containing variable 'val'.")
    parser.add_argument("--fs", type=float, default=360.0, help="Sampling rate in Hz.")
    parser.add_argument("--output-dir", default="ecg_outputs", help="Folder for generated plots.")
    parser.add_argument("--powerline-freq", type=float, default=60.0, help="Powerline interference frequency in Hz.")

    parser.add_argument("--baseline-amp", type=float, default=0.25, help="Baseline wander amplitude factor.")
    parser.add_argument("--powerline-amp", type=float, default=0.25, help="Powerline interference amplitude factor.")
    parser.add_argument("--white-noise-std", type=float, default=0.12, help="White noise standard deviation factor.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for repeatable synthetic noise.")

    parser.add_argument("--list-samples", action="store_true", help="List ECG files and exit.")
    parser.add_argument("--list-root", default="MLII", help="Root folder for --list-samples.")
    parser.add_argument("--list-limit", type=int, default=25, help="Number of files to print with --list-samples.")
    return parser


def main():
    args = build_argument_parser().parse_args()

    if args.list_samples:
        list_sample_files(args.list_root, args.list_limit)
        return 0

    file_path = Path(args.file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        clean_ecg = load_ecg_file(file_path)
        noisy_ecg = add_noise(
            clean_ecg,
            args.fs,
            args.baseline_amp,
            args.powerline_amp,
            args.white_noise_std,
            args.powerline_freq,
            args.seed,
        )
        filtered_ecg = filter_ecg(noisy_ecg, args.fs, args.powerline_freq)

        t = np.arange(len(clean_ecg)) / args.fs

        clean_peaks, clean_hr = get_r_peaks(clean_ecg, args.fs)
        noisy_peaks, noisy_hr = get_r_peaks(noisy_ecg, args.fs)
        filtered_peaks, filtered_hr = get_r_peaks(filtered_ecg, args.fs)

        peaks_info = {
            "clean": (clean_peaks, clean_hr),
            "noisy": (noisy_peaks, noisy_hr),
            "filtered": (filtered_peaks, filtered_hr),
        }

        plot_files = []
        plot_files.append(plot_ecg_comparison(t, clean_ecg, noisy_ecg, filtered_ecg, output_dir))
        plot_files.append(plot_zoomed_ecg(t, clean_ecg, noisy_ecg, filtered_ecg, output_dir))
        plot_files.append(plot_fft(clean_ecg, noisy_ecg, filtered_ecg, args.fs, output_dir))
        plot_files.append(plot_peaks(t, noisy_ecg, filtered_ecg, noisy_peaks, filtered_peaks, noisy_hr, filtered_hr, output_dir))

        print_summary(args, file_path, clean_ecg, noisy_ecg, filtered_ecg, peaks_info, plot_files)

    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
