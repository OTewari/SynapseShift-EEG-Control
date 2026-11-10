"""Render the README figures off-screen (matplotlib Agg backend).

    python docs/make_figures.py

Writes:
  docs/simulated_eeg.png       - the app's 8-channel, 10-second rolling view, filled from the built-in
                                 simulated EEG generator (same generator, filter and layout as visualize_eeg)
  docs/calibration_examples.png - one band-passed calibration window per class from data/calibration_data.pkl
(docs/confusion_matrix.png is written by evaluate.py.)
"""
import os
import sys
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
import ControlSys as app  # noqa: E402

FS = 250


def simulated_rolling_plot(path="docs/simulated_eeg.png", seed=0):
    np.random.seed(seed)
    window_duration = 10
    n_total = FS * window_duration
    buffer = np.zeros((8, n_total))
    # fill the 10 s buffer exactly as visualize_eeg does: 0.5 s windows, each filtered on its own
    for _ in range(n_total // 125):
        new = app.filter_eeg(app.get_raw_eeg())  # SIMULATED/no board -> generate_simulated_eeg
        buffer = np.roll(buffer, -new.shape[1], axis=1)
        buffer[:, -new.shape[1]:] = new
    t = np.linspace(-window_duration, 0, n_total)
    fig, axs = plt.subplots(8, 1, sharex=True, figsize=(10, 8), dpi=110)
    for i in range(8):
        axs[i].plot(t, buffer[i], lw=0.7)
        axs[i].set_ylim(-100, 100)
        axs[i].set_ylabel(app.electrode_names[i], rotation=0, labelpad=18)
    axs[-1].set_xlabel("Time (sec)")
    fig.suptitle("Rolling 8-channel view, simulated EEG (1-50 Hz band-passed, 0.5 s updates)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def calibration_examples(path="docs/calibration_examples.png", seed=0):
    with open("data/calibration_data.pkl", "rb") as f:
        data = pickle.load(f)
    X = np.array(data["features"])
    y = np.array(data["labels"])
    rng = np.random.default_rng(seed)
    names = list(app.ACTIONS.keys())
    t = np.arange(X.shape[2]) / FS
    fig, axs = plt.subplots(1, len(names), sharey=True, figsize=(13, 4.2), dpi=110)
    spacing = 150
    for k, name in enumerate(names):
        idx = rng.choice(np.where(y == k)[0])
        w = app.filter_eeg(X[idx])
        for c in range(8):
            axs[k].plot(t, w[c] - c * spacing, lw=0.7)
        axs[k].set_title(f"{name} (window {idx})", fontsize=9)
        axs[k].set_xlabel("s")
    axs[0].set_yticks([-c * spacing for c in range(8)], app.electrode_names)
    fig.suptitle("One recorded 0.5 s calibration window per class, 1-50 Hz band-passed "
                 f"(channels offset by {spacing} uV)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    app.SIMULATED = True
    simulated_rolling_plot()
    calibration_examples()
    print("Wrote docs/simulated_eeg.png and docs/calibration_examples.png")
