import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from types import SimpleNamespace

import numpy as np

from ecg_project import (
    DEFAULT_ECG_FILE,
    add_noise,
    calculate_rmse,
    calculate_snr,
    filter_ecg,
    get_r_peaks,
    load_ecg_file,
    plot_ecg_comparison,
    plot_fft,
    plot_peaks,
    plot_zoomed_ecg,
)


class ECGDemoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ECG Denoising Demo")
        self.root.geometry("780x520")

        self.selected_file = tk.StringVar(value=str(DEFAULT_ECG_FILE))
        self.output_dir = Path("ecg_gui_outputs")
        self.last_plots = []

        self.build_screen()

    def build_screen(self):
        main = ttk.Frame(self.root, padding=16)
        main.pack(fill="both", expand=True)

        title = ttk.Label(main, text="ECG Signal Denoising and Heart Rate Estimation", font=("Segoe UI", 15, "bold"))
        title.pack(anchor="w")

        file_row = ttk.Frame(main)
        file_row.pack(fill="x", pady=(18, 8))

        ttk.Label(file_row, text="ECG file:").pack(side="left")
        file_entry = ttk.Entry(file_row, textvariable=self.selected_file)
        file_entry.pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(file_row, text="Browse", command=self.browse_file).pack(side="left")

        sample_row = ttk.Frame(main)
        sample_row.pack(fill="x", pady=(0, 12))

        ttk.Button(sample_row, text="NSR Sample", command=lambda: self.use_sample("MLII/1 NSR/100m (0).mat")).pack(side="left")
        ttk.Button(sample_row, text="AFIB Sample", command=lambda: self.use_sample("MLII/4 AFIB/201m (0).mat")).pack(side="left", padx=8)
        ttk.Button(sample_row, text="PVC Sample", command=lambda: self.use_sample("MLII/7 PVC/105m (0).mat")).pack(side="left")

        controls = ttk.LabelFrame(main, text="Noise settings")
        controls.pack(fill="x", pady=8)

        self.baseline_var = tk.DoubleVar(value=0.25)
        self.powerline_var = tk.DoubleVar(value=0.25)
        self.white_var = tk.DoubleVar(value=0.12)

        self.add_slider(controls, "Baseline wander", self.baseline_var, 0.0, 0.5, 0)
        self.add_slider(controls, "60 Hz interference", self.powerline_var, 0.0, 0.5, 1)
        self.add_slider(controls, "White noise", self.white_var, 0.0, 0.25, 2)

        button_row = ttk.Frame(main)
        button_row.pack(fill="x", pady=10)

        ttk.Button(button_row, text="Run Analysis", command=self.run_analysis).pack(side="left")
        ttk.Button(button_row, text="Open Plot Folder", command=self.open_output_folder).pack(side="left", padx=8)
        ttk.Button(button_row, text="Open Main Plot", command=self.open_main_plot).pack(side="left")

        self.output_box = tk.Text(main, height=14, wrap="word")
        self.output_box.pack(fill="both", expand=True, pady=(8, 0))
        self.write_output("Choose an ECG sample, then click Run Analysis.\n")

    def add_slider(self, parent, label, variable, min_value, max_value, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=8, pady=6)
        slider = ttk.Scale(parent, from_=min_value, to=max_value, variable=variable)
        slider.grid(row=row, column=1, sticky="ew", padx=8, pady=6)
        value_label = ttk.Label(parent, textvariable=variable, width=8)
        value_label.grid(row=row, column=2, sticky="e", padx=8, pady=6)
        parent.columnconfigure(1, weight=1)

    def browse_file(self):
        file_name = filedialog.askopenfilename(
            title="Choose ECG .mat file",
            initialdir="MLII",
            filetypes=[("MAT files", "*.mat"), ("All files", "*.*")],
        )
        if file_name:
            self.selected_file.set(file_name)

    def use_sample(self, sample_path):
        self.selected_file.set(sample_path)

    def write_output(self, message):
        self.output_box.delete("1.0", "end")
        self.output_box.insert("end", message)

    def run_analysis(self):
        try:
            file_path = Path(self.selected_file.get())
            fs = 360.0
            powerline_freq = 60.0

            clean_ecg = load_ecg_file(file_path)
            noisy_ecg = add_noise(
                clean_ecg,
                fs,
                self.baseline_var.get(),
                self.powerline_var.get(),
                self.white_var.get(),
                powerline_freq,
                seed=7,
            )
            filtered_ecg = filter_ecg(noisy_ecg, fs, powerline_freq)
            t = np.arange(len(clean_ecg)) / fs

            clean_peaks, clean_hr = get_r_peaks(clean_ecg, fs)
            noisy_peaks, noisy_hr = get_r_peaks(noisy_ecg, fs)
            filtered_peaks, filtered_hr = get_r_peaks(filtered_ecg, fs)

            self.output_dir.mkdir(exist_ok=True)
            self.last_plots = [
                plot_ecg_comparison(t, clean_ecg, noisy_ecg, filtered_ecg, self.output_dir),
                plot_zoomed_ecg(t, clean_ecg, noisy_ecg, filtered_ecg, self.output_dir),
                plot_fft(clean_ecg, noisy_ecg, filtered_ecg, fs, self.output_dir),
                plot_peaks(t, noisy_ecg, filtered_ecg, noisy_peaks, filtered_peaks, noisy_hr, filtered_hr, self.output_dir),
            ]

            snr_before = calculate_snr(clean_ecg, noisy_ecg)
            snr_after = calculate_snr(clean_ecg, filtered_ecg)
            rmse_before = calculate_rmse(clean_ecg, noisy_ecg)
            rmse_after = calculate_rmse(clean_ecg, filtered_ecg)
            rmse_reduction = 100 * (1 - rmse_after / rmse_before)

            summary = []
            summary.append(f"File: {file_path}")
            summary.append(f"Rhythm class: {file_path.parent.name}")
            summary.append(f"Samples: {len(clean_ecg)}   Duration: {len(clean_ecg) / fs:.2f} seconds")
            summary.append("")
            summary.append("Before/after metrics:")
            summary.append(f"SNR:  {snr_before:.2f} dB -> {snr_after:.2f} dB")
            summary.append(f"RMSE: {rmse_before:.2f} -> {rmse_after:.2f}  ({rmse_reduction:.1f}% reduction)")
            summary.append("")
            summary.append("Heart-rate estimation:")
            summary.append(f"Clean:    {len(clean_peaks)} peaks, {clean_hr:.1f} bpm")
            summary.append(f"Noisy:    {len(noisy_peaks)} peaks, {noisy_hr:.1f} bpm")
            summary.append(f"Filtered: {len(filtered_peaks)} peaks, {filtered_hr:.1f} bpm")
            summary.append("")
            summary.append("Plots saved:")
            for plot in self.last_plots:
                summary.append(f"- {plot}")

            self.write_output("\n".join(summary))

        except Exception as error:
            messagebox.showerror("Analysis failed", str(error))

    def open_output_folder(self):
        self.output_dir.mkdir(exist_ok=True)
        os.startfile(self.output_dir.resolve())

    def open_main_plot(self):
        if not self.last_plots:
            messagebox.showinfo("No plot yet", "Run the analysis first.")
            return
        os.startfile(Path(self.last_plots[0]).resolve())


def main():
    root = tk.Tk()
    app = ECGDemoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
