# ECG Signal Denoising and Heart Rate Estimation

This is the main Signals and Systems project. It uses raw ECG waveform fragments from the `MLII` dataset, adds controlled realistic noise, applies digital filters, and estimates heart rate from detected R-peaks.

## Dataset

Use the ECG fragments in:

```text
MLII/
```

Each `.mat` file contains a `val` array with 3600 ECG samples. The script treats this as 10 seconds sampled at 360 Hz.

Useful examples:

```text
MLII\1 NSR\100m (0).mat
MLII\4 AFIB\201m (0).mat
MLII\7 PVC\105m (0).mat
```

## Setup

Install the required libraries:

```powershell
python -m pip install -r requirements.txt
```

## Usage

List sample ECG files:

```powershell
python ecg_project.py --list-samples
```

Run the default normal ECG sample:

```powershell
python ecg_project.py
```

Run the simple GUI demo:

```powershell
python ecg_gui.py
```

Run a different rhythm sample:

```powershell
python ecg_project.py --file "MLII\4 AFIB\201m (0).mat" --output-dir ecg_outputs_afib
```

Generated plots are saved in the selected output folder.

