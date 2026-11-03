import json
import os
import shutil
import subprocess
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "music_genre_cnn_4genres.keras"
CONFIG_PATH = MODEL_DIR / "inference_config.json"

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Missing model: {MODEL_PATH}. Copy music_genre_cnn_4genres.keras "
        "from the Colab export into the model/ folder."
    )
if not CONFIG_PATH.is_file():
    raise FileNotFoundError(
        f"Missing inference configuration: {CONFIG_PATH}. Copy inference_config.json "
        "from the SAME Colab export as the model. Both files are required."
    )
if shutil.which("ffmpeg") is None:
    raise RuntimeError("FFmpeg is missing. Install the system package listed in packages.txt.")

CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
GENRES = list(CONFIG["genres"])
SAMPLE_RATE = int(CONFIG["sample_rate"])
DURATION_SECONDS = int(CONFIG["duration_seconds"])
N_FFT = int(CONFIG["n_fft"])
HOP_LENGTH = int(CONFIG["hop_length"])
N_MELS = int(CONFIG["n_mels"])
SPEC_SHAPE = tuple(CONFIG["spectrogram_shape"])
POP_LOGIT_BIAS = float(CONFIG.get("pop_logit_bias", 0.0))
POP_INDEX = GENRES.index("Pop") if "Pop" in GENRES else None

# Load the saved CNN once when the app starts; uploads do not retrain it.
model = tf.keras.models.load_model(MODEL_PATH, compile=False)


def decode_audio(audio_path: str) -> np.ndarray:
    command = [
        shutil.which("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(audio_path), "-map", "0:a:0", "-t", str(DURATION_SECONDS),
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-acodec", "pcm_f32le",
        "-f", "f32le", "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0 or not result.stdout:
        details = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Could not decode this audio file. {details or 'No audio samples returned.'}")

    waveform = np.frombuffer(result.stdout, dtype="<f4").astype(np.float32, copy=True)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("The uploaded file did not produce valid audio samples.")

    target_samples = SAMPLE_RATE * DURATION_SECONDS
    if waveform.size < target_samples:
        waveform = np.pad(waveform, (0, target_samples - waveform.size))
    else:
        waveform = waveform[:target_samples]
    return waveform


def audio_to_logmel(audio_path: str) -> np.ndarray:
    waveform = decode_audio(audio_path)
    mel_power = librosa.feature.melspectrogram(
        y=waveform,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel_power, ref=np.max, top_db=80.0)
    mel_scaled = np.clip((mel_db + 80.0) / 80.0, 0.0, 1.0)
    spectrogram = np.round(mel_scaled * 255).astype(np.uint8)

    if tuple(spectrogram.shape) != SPEC_SHAPE:
        raise ValueError(
            f"Audio preprocessing produced {spectrogram.shape}; "
            f"the saved model expects {SPEC_SHAPE}. Check that the model and config match."
        )
    return spectrogram


def apply_pop_bias(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    logits = np.log(np.clip(values, 1e-8, 1.0))
    if POP_INDEX is not None:
        logits[:, POP_INDEX] += POP_LOGIT_BIAS
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return (exp_logits / exp_logits.sum(axis=1, keepdims=True)).astype(np.float32)


def predict_genre(audio_path):
    if not audio_path:
        raise gr.Error("Please upload an audio file first.")
    suffix = Path(audio_path).suffix.lower()
    if suffix not in {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}:
        raise gr.Error("Supported formats: MP3, WAV, FLAC, OGG, M4A and AAC.")

    try:
        spec = audio_to_logmel(audio_path)
        model_input = (spec.astype(np.float32) / 255.0)[None, :, :, None]
        raw_scores = model.predict(model_input, verbose=0)
        scores = apply_pop_bias(raw_scores)[0]
        return {genre: float(score) for genre, score in zip(GENRES, scores)}
    except gr.Error:
        raise
    except Exception as error:
        raise gr.Error(f"Prediction failed: {type(error).__name__}: {error}") from error


demo = gr.Interface(
    fn=predict_genre,
    inputs=gr.Audio(
        sources=["upload"],
        type="filepath",
        label="Upload a song (the first 30 seconds are analysed)",
    ),
    outputs=gr.Label(num_top_classes=4, label="Predicted genre and model scores"),
    title="Music Genre Prediction",
    description="Upload an audio clip to classify it as Electronic, Hip-Hop, Pop or Rock.",
    flagging_mode="never",
    delete_cache=(3600, 3600),
)

if __name__ == "__main__":
    # Set GRADIO_SHARE=true only when launching from a Colab notebook.
    make_share_link = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    demo.queue().launch(share=make_share_link)
