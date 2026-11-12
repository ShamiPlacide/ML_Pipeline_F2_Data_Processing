"""Audio pipeline: load, visualise, augment and extract voice features (Role C).

Run from the repo root:
    python src/audio_processing.py

Reads   media/audio/*.m4a                 raw phone recordings
Writes  media/audio/original/*.wav        members, 16 kHz mono, <member>_<phrase>_<take>.wav
        media/audio/unauthorized/*.wav    non-member voices, unknown_<name>.wav
        media/audio/augmented/*.wav       6 augmented copies of every clip
        data/processed/audio_features.csv one row per clip (original + augmented)
        reports/figures/audio_*.png       waveforms, spectrograms, augmentation plot

The same load_clean() and extract_features() are reused by src/predict.py, so a
clip is processed identically at training time and at prediction time.

Adding recordings: drop more .m4a files into media/audio/ and re-run. The member
is the first word of the file name and the phrase is detected from the words
"approve" / "confirm" / "transaction" (e.g. nick_approve_2.m4a). Several takes
of the same phrase are numbered automatically. Any file without a phrase word,
or starting with "unknown", is treated as a non-member (unauthorized) voice.
"""
import re
import warnings
from pathlib import Path

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "media" / "audio"
ORIGINAL_DIR = RAW_DIR / "original"
UNAUTHORIZED_DIR = RAW_DIR / "unauthorized"
AUGMENTED_DIR = RAW_DIR / "augmented"
FEATURES_PATH = ROOT / "data" / "processed" / "audio_features.csv"
FIG_DIR = ROOT / "reports" / "figures"

SR = 16_000          # speech carries little above 8 kHz, so 16 kHz is enough
N_MFCC = 13
N_FFT = 512          # 32 ms window
HOP = 160            # 10 ms hop
TRIM_DB = 30         # anything 30 dB below the peak counts as silence
SEED = 42

PHRASES = {"yes_approve": "Yes, approve", "confirm_transaction": "Confirm transaction"}
META_COLS = ["member", "phrase", "augmentation", "is_original", "source_audio", "audio_path"]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def parse_raw_name(path):
    """'Nick-approve.m4a' -> ('nick', 'yes_approve').

    A file whose name contains neither phrase (or starts with "unknown") is a
    non-member recording and becomes ('unknown', <file stem>).
    """
    stem = Path(path).stem.lower()
    member = re.split(r"[-_ ]", stem)[0]
    if member == "unknown":
        return "unknown", re.sub(r"^unknown[-_ ]*", "", stem) or "voice"
    if "approve" in stem:
        return member, "yes_approve"
    if "confirm" in stem or "transaction" in stem:
        return member, "confirm_transaction"
    return "unknown", stem


def load_clean(path):
    """Load any audio file as 16 kHz mono, trim silence, peak-normalise."""
    with warnings.catch_warnings():          # m4a goes through the audioread fallback
        warnings.simplefilter("ignore")
        y, _ = librosa.load(path, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=TRIM_DB)
    peak = np.abs(y).max()
    return (y / peak * 0.95).astype(np.float32) if peak > 0 else y


def convert_raw_recordings():
    """Convert the raw .m4a recordings to cleaned, consistently named .wav files."""
    for folder in (ORIGINAL_DIR, UNAUTHORIZED_DIR, AUGMENTED_DIR):   # generated folders
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*.wav"):
            old.unlink()
    def order(raw):   # a file without a number is take 1; "nick_approve_2" is take 2
        number = re.search(r"(\d+)$", raw.stem)
        return (*parse_raw_name(raw), int(number.group(1)) if number else 1, raw.name)

    rows, takes = [], {}
    for raw in sorted(RAW_DIR.glob("*.m4a"), key=order):
        member, phrase = parse_raw_name(raw)
        if member == "unknown":
            out = UNAUTHORIZED_DIR / f"unknown_{phrase}.wav"
        else:
            takes[member, phrase] = takes.get((member, phrase), 0) + 1
            out = ORIGINAL_DIR / f"{member}_{phrase}_{takes[member, phrase]}.wav"
        y = load_clean(raw)
        sf.write(out, y, SR, subtype="PCM_16")
        rows.append({"raw_file": raw.name, "member": member, "phrase": phrase,
                     "wav": str(out.relative_to(ROOT)), "seconds": round(len(y) / SR, 2)})
    return pd.DataFrame(rows)


def list_clips():
    """Every cleaned clip as (member, phrase, path). Members first, then unknowns."""
    clips = []
    for wav in sorted(ORIGINAL_DIR.glob("*.wav")):
        member, phrase = wav.stem.rsplit("_", 1)[0].split("_", 1)   # drop the take number
        clips.append((member, phrase, wav))
    for wav in sorted(UNAUTHORIZED_DIR.glob("*.wav")):
        clips.append(("unknown", "unauthorized", wav))
    return clips


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #
def add_noise(y, snr_db, rng):
    """Add white background noise at a given signal-to-noise ratio."""
    noise_power = np.mean(y ** 2) / (10 ** (snr_db / 10))
    return y + rng.normal(0, np.sqrt(noise_power), len(y))


def augment(y, rng):
    """Six augmented versions of a clip: {name: signal}."""
    out = {
        "pitch_up": librosa.effects.pitch_shift(y, sr=SR, n_steps=1),
        "pitch_down": librosa.effects.pitch_shift(y, sr=SR, n_steps=-1),
        "stretch_fast": librosa.effects.time_stretch(y, rate=1.15),
        "stretch_slow": librosa.effects.time_stretch(y, rate=0.85),
        "noise_20db": add_noise(y, 20, rng),
        "noise_10db": add_noise(y, 10, rng),
    }
    return {k: np.clip(v, -1, 1).astype(np.float32) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def extract_features(y):
    """One fixed-length feature vector (dict) for a clip of any duration.

    Frame-level features are computed every 10 ms and summarised by their mean
    and standard deviation over the *voiced* frames only (frames where a pitch
    was detected). Pauses and background noise say nothing about who is
    speaking, so leaving them out makes the features depend less on how much
    silence a recording happens to contain.

    MFCCs + delta MFCCs describe the vocal-tract shape, roll-off / centroid /
    bandwidth the brightness of the voice, RMS the energy, F0 the pitch.
    """
    kw = {"n_fft": N_FFT, "hop_length": HOP}
    f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=400, sr=SR, frame_length=1024, hop_length=HOP)
    voiced = voiced & ~np.isnan(f0)
    keep = voiced if voiced.sum() >= 5 else np.ones_like(voiced)   # fallback: all frames

    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, **kw)
    delta = librosa.feature.delta(mfcc)
    frame_feats = {f"mfcc_{i}": mfcc[i] for i in range(N_MFCC)}
    frame_feats.update({f"delta_mfcc_{i}": delta[i] for i in range(N_MFCC)})
    frame_feats.update({
        "rolloff": librosa.feature.spectral_rolloff(y=y, sr=SR, roll_percent=0.85, **kw)[0],
        "centroid": librosa.feature.spectral_centroid(y=y, sr=SR, **kw)[0],
        "bandwidth": librosa.feature.spectral_bandwidth(y=y, sr=SR, **kw)[0],
        "rms_energy": librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)[0],
        "zcr": librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP)[0],
    })

    feats = {}
    for name, values in frame_feats.items():
        values = values[keep]
        if not name.startswith("delta"):          # a delta averages to ~0 by construction
            feats[f"{name}_mean"] = values.mean()
        feats[f"{name}_std"] = values.std()
    feats["f0_mean"] = f0[voiced].mean() if voiced.any() else 0.0
    feats["f0_std"] = f0[voiced].std() if voiced.any() else 0.0
    feats["voiced_fraction"] = voiced.mean()
    feats["duration_s"] = len(y) / SR
    return {k: float(v) for k, v in feats.items()}


def features_from_file(path):
    """Features for one audio file, exactly as used in training (for predict.py)."""
    return extract_features(load_clean(path))


def build_feature_table():
    """Augment every clip, save the augmented audio, return the feature table."""
    AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows = []
    for member, phrase, wav in list_clips():
        y, _ = librosa.load(wav, sr=SR)
        versions = {"original": y, **augment(y, rng)}
        for aug_name, signal in versions.items():
            if aug_name == "original":
                path = wav
            else:
                path = AUGMENTED_DIR / f"{wav.stem}_{aug_name}.wav"
                sf.write(path, signal, SR, subtype="PCM_16")
            rows.append({
                "member": member, "phrase": phrase, "augmentation": aug_name,
                "is_original": int(aug_name == "original"), "source_audio": wav.name,
                "audio_path": str(path.relative_to(ROOT)), **extract_features(signal),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _waveform(ax, y, title):
    librosa.display.waveshow(y, sr=SR, ax=ax, color="#2a6f97")
    ax.set(title=title, xlabel="Time (s)", ylabel="Amplitude", ylim=(-1.05, 1.05))


def _spectrogram(ax, y, title):
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=64)
    img = librosa.display.specshow(librosa.power_to_db(mel, ref=np.max), sr=SR, hop_length=HOP,
                                   x_axis="time", y_axis="mel", ax=ax, cmap="magma",
                                   vmin=-80, vmax=0)
    ax.set(title=title, xlabel="Time (s)", ylabel="Frequency (Hz)")
    return img


def plot_waveforms_and_spectrograms():
    """One row per recording: waveform on the left, mel spectrogram on the right."""
    clips = list_clips()
    fig, axes = plt.subplots(len(clips), 2, figsize=(12, 2.4 * len(clips)), constrained_layout=True)
    for (member, phrase, wav), (ax_w, ax_s) in zip(clips, axes):
        y, _ = librosa.load(wav, sr=SR)
        if member == "unknown":
            label = f"Non-member ({wav.stem}): unauthorized sample"
        else:
            label = f'{member.title()}: "{PHRASES[phrase]}", take {wav.stem.rsplit("_", 1)[1]}'
        _waveform(ax_w, y, f"{label} (waveform)")
        img = _spectrogram(ax_s, y, f"{label} (mel spectrogram)")
    fig.colorbar(img, ax=axes[:, 1], format="%+2.0f dB", shrink=0.4, label="Power (dB)")
    path = FIG_DIR / "audio_waveforms_spectrograms.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_augmentations(wav=None):
    """Original clip and its six augmentations, waveform + spectrogram each."""
    wav = Path(wav) if wav else sorted(ORIGINAL_DIR.glob("*.wav"))[0]
    y, _ = librosa.load(wav, sr=SR)
    versions = {"original": y, **augment(y, np.random.default_rng(SEED))}
    fig, axes = plt.subplots(2, len(versions), figsize=(3.1 * len(versions), 5.2),
                             constrained_layout=True)
    for col, (name, signal) in enumerate(versions.items()):
        _waveform(axes[0, col], signal, name)
        _spectrogram(axes[1, col], signal, "")
        for ax in axes[:, col]:
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
            if col:
                ax.set_ylabel("")
    fig.suptitle(f"Augmentations of {wav.name}")
    path = FIG_DIR / "audio_augmentations.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_feature_separation(df):
    """Do the saved features separate speakers? Mean pitch vs. mean MFCC 1."""
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    markers = {"yes_approve": "o", "confirm_transaction": "s"}
    df = df[df["augmentation"].isin(["original", "stretch_fast", "stretch_slow", "noise_20db"])]
    for i, (member, grp) in enumerate(df.groupby("member", sort=True)):
        for phrase, sub in grp.groupby("phrase"):
            ax.scatter(sub["f0_mean"], sub["mfcc_1_mean"], color=f"C{i}", alpha=0.8,
                       marker=markers.get(phrase, "X"), s=45,
                       label=f"{member} / {phrase}")
    ax.set(xlabel="Mean pitch F0 (Hz)", ylabel="Mean MFCC 1",
           title="Speaker separation in two audio features (original, stretched and noisy clips)")
    ax.legend(fontsize=7, ncol=2)
    path = FIG_DIR / "audio_feature_separation.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)

    manifest = convert_raw_recordings()
    print("Recordings found:\n", manifest.to_string(index=False), sep="")

    df = build_feature_table()
    df.to_csv(FEATURES_PATH, index=False)
    n_feat = df.shape[1] - len(META_COLS)
    print(f"\nSaved {FEATURES_PATH.relative_to(ROOT)}: {len(df)} clips x {n_feat} features")
    print(df.groupby(["member", "phrase"]).size().rename("clips").to_string())

    for path in (plot_waveforms_and_spectrograms(), plot_augmentations(),
                 plot_feature_separation(df)):
        print("Saved", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
