"""Held-out evaluation of SynapseShift's two classifiers on data/calibration_data.pkl.

Usage (from the repository root, inside the virtual environment):

    python evaluate.py              # prints the results table and writes docs/confusion_matrix.png

What it does
------------
* Features come from ControlSys.extract_all_features, the exact function the app uses at run time
  (1-50 Hz Butterworth band-pass -> per-channel alpha/beta/gamma Welch power -> frontal theta/alpha
  features -> CSP log-variance features).
* The committed csp_filters.pkl was fitted on the whole calibration file, so using it inside
  cross-validation would leak test labels into the features. Instead, CSP filters are re-fitted on
  the training fold only, with ComputeCSPFilter.compute_csp_filters (same settings as that script:
  target label 4 = "motor", n_filters=2), and applied to both folds.
* The two models are the app's own: SVC(probability=True, decision_function_shape='ovr') with
  default hyper-parameters, and the 9-layer Keras MLP from ControlSys.train_model (20 epochs, Adam).
  Like the app, no feature scaling is applied.
* Two splitting schemes are reported:
    1. Stratified 5-fold CV with shuffling (window-level). Adjacent windows of the same recording
       block can land in train and test, so this is the more optimistic of the two.
    2. Leave-one-session-out: every pass of "Start Collection" (one block per class) is held out
       in turn. This is closer to how the app is used (train on earlier recordings, run later).
* The app's own "Accuracy" (scored on the training data) is shown for comparison.
"""
import os
import sys
import pickle
import random
import argparse
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)  # sklearn 1.9 deprecates SVC(probability=True)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLBACKEND", "Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)  # the app resolves csp_filters.pkl and data/ relative to the working directory
sys.path.insert(0, HERE)

import ControlSys as app  # noqa: E402  (importing does not start the GUI or touch the board)
from ComputeCSPFilter import compute_csp_filters  # noqa: E402

from sklearn.svm import SVC  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix  # noqa: E402

app.tf.get_logger().setLevel("ERROR")  # silence tf.function retracing warnings

CLASS_NAMES = list(app.ACTIONS.keys())  # ["none", "blink", "jaw", "relax", "motor", "mental"]
MOTOR_LABEL = app.ACTION_TO_INDEX["motor"]
SEED = 0


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    app.tf.keras.utils.set_random_seed(seed)


def load_calibration(path="data/calibration_data.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    X_raw = np.array(data["features"])  # (n_windows, 8, 125)
    y = np.array(data["labels"]).astype(int)
    return X_raw, y


def session_ids(y):
    """Group windows into collection sessions.

    start_collection() records the classes in ACTIONS order (none, blink, jaw, relax, motor, mental)
    and appends them to the file, so a new session starts whenever the label goes back down
    (e.g. mental -> none). Classes the user chose not to keep in a session are simply absent.
    """
    groups = np.zeros(len(y), dtype=int)
    g = 0
    for i in range(1, len(y)):
        if y[i] < y[i - 1]:
            g += 1
        groups[i] = g
    return groups


def features_with_filters(X_raw, filters):
    """Run the app's extract_all_features with the given CSP filters instead of csp_filters.pkl."""
    original = app.extract_csp_features

    def csp_from_fold(raw):
        projected = np.dot(filters, raw)
        return np.log(np.var(projected, axis=1) + 1e-6)  # identical to ControlSys.extract_csp_features

    app.extract_csp_features = csp_from_fold
    try:
        return np.array([app.extract_all_features(x) for x in X_raw])
    finally:
        app.extract_csp_features = original


def build_mlp(n_features, n_classes):
    """Same architecture as ControlSys.train_model (keep in sync)."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Input
    return Sequential([
        Input(shape=(n_features,)),
        Dense(256, activation='gelu'),
        Dense(64, activation='elu'),
        Dense(128, activation='gelu'),
        Dense(64, activation='leaky_relu'),
        Dense(32, activation='relu'),
        Dense(64, activation='elu'),
        Dropout(0.1),
        Dense(64, activation='elu'),
        Dense(n_classes, activation='softmax'),
    ])


def fit_predict(kind, F_train, y_train, F_test):
    if kind == "svm":
        model = SVC(probability=True, decision_function_shape='ovr', random_state=SEED)
        model.fit(F_train, y_train)
        return model.predict(F_test)
    set_seeds()
    model = build_mlp(F_train.shape[1], len(CLASS_NAMES))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(F_train, y_train, epochs=20, verbose=0)
    return np.argmax(model.predict(F_test, verbose=0), axis=1)


def run_cv(X_raw, y, splits, kind):
    y_pred = np.full_like(y, -1)
    fold_acc = []
    for train_idx, test_idx in splits:
        filters = compute_csp_filters(X_raw[train_idx], y[train_idx], MOTOR_LABEL, n_filters=2)
        F_train = features_with_filters(X_raw[train_idx], filters)
        F_test = features_with_filters(X_raw[test_idx], filters)
        pred = fit_predict(kind, F_train, y[train_idx], F_test)
        y_pred[test_idx] = pred
        fold_acc.append(accuracy_score(y[test_idx], pred))
    return y_pred, np.array(fold_acc)


def training_set_accuracy(X_raw, y, kind):
    """What the app reports: fit on everything (with the committed csp_filters.pkl), score on the same data."""
    F = np.array([app.extract_all_features(x) for x in X_raw])
    return accuracy_score(y, fit_predict(kind, F, y, F))


def plot_confusion(cm, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.6, 4.8), dpi=130)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=30)
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalised rate")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--no-plot", action="store_true", help="do not write docs/confusion_matrix.png")
    args = parser.parse_args()

    X_raw, y = load_calibration()
    groups = session_ids(y)
    counts = np.bincount(y, minlength=len(CLASS_NAMES))
    print(f"Calibration data: {X_raw.shape[0]} windows of {X_raw.shape[1]} channels x {X_raw.shape[2]} samples")
    print("Windows per class: " + ", ".join(f"{n}={c}" for n, c in zip(CLASS_NAMES, counts)))
    print(f"Collection sessions detected: {groups.max() + 1}")
    print(f"Chance level (majority class): {counts.max() / counts.sum():.3f}\n")

    skf = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X_raw, y))
    loso = [(np.where(groups != g)[0], np.where(groups == g)[0]) for g in np.unique(groups)]

    rows = []
    preds = {}
    for kind, label in (("svm", "SVM (RBF, default C)"), ("mlp", "Keras MLP (20 epochs)")):
        train_acc = training_set_accuracy(X_raw, y, kind)
        p_skf, a_skf = run_cv(X_raw, y, skf, kind)
        p_loso, a_loso = run_cv(X_raw, y, loso, kind)
        preds[kind] = p_skf
        rows.append((label, train_acc,
                     accuracy_score(y, p_skf), a_skf.std(), f1_score(y, p_skf, average="macro"),
                     accuracy_score(y, p_loso), a_loso.min(), a_loso.max(), f1_score(y, p_loso, average="macro")))

    print("| Model | App-reported accuracy (training data) | Stratified 5-fold accuracy (+/- fold std) | 5-fold macro-F1 "
          "| Leave-one-session-out accuracy (fold min-max) | LOSO macro-F1 |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | {r[1]:.3f} | {r[2]:.3f} +/- {r[3]:.3f} | {r[4]:.3f} | {r[5]:.3f} ({r[6]:.2f}-{r[7]:.2f}) | {r[8]:.3f} |")

    # Folding "motor" into "none" (no key), as in the final live run-through. This is applied after
    # prediction to the same out-of-fold predictions; the models are not retrained on 5 classes.
    none_label = app.ACTION_TO_INDEX["none"]

    def fold_motor(a):
        a = a.copy()
        a[a == MOTOR_LABEL] = none_label
        return a

    def wrong_key_rate(true, pred):
        """Share of windows where a key is pressed that is not the intended one."""
        return np.mean((pred != none_label) & (pred != true))

    print("\nKey-press view (stratified 5-fold, out-of-fold predictions):")
    print("| Model | 6-class accuracy | Wrong-key rate, 6 classes | Accuracy, motor folded into none "
          "| Wrong-key rate, motor folded into none |")
    print("|---|---|---|---|---|")
    for kind, label in (("svm", "SVM (RBF, default C)"), ("mlp", "Keras MLP (20 epochs)")):
        p = preds[kind]
        print(f"| {label} | {accuracy_score(y, p):.3f} | {wrong_key_rate(y, p):.3f} "
              f"| {accuracy_score(fold_motor(y), fold_motor(p)):.3f} "
              f"| {wrong_key_rate(fold_motor(y), fold_motor(p)):.3f} |")

    best = max(("svm", "mlp"), key=lambda k: accuracy_score(y, preds[k]))
    cm = confusion_matrix(y, preds[best], labels=range(len(CLASS_NAMES)))
    print(f"\nOut-of-fold confusion matrix, stratified 5-fold, {best.upper()} (rows = true, cols = predicted):")
    print("        " + " ".join(f"{n:>7}" for n in CLASS_NAMES))
    for n, row in zip(CLASS_NAMES, cm):
        print(f"{n:>7} " + " ".join(f"{v:>7}" for v in row))
    if not args.no_plot:
        name = "SVM" if best == "svm" else "Keras MLP"
        plot_confusion(cm, f"{name}, out-of-fold predictions, stratified 5-fold CV", "docs/confusion_matrix.png")
        print("\nWrote docs/confusion_matrix.png")


if __name__ == "__main__":
    main()
