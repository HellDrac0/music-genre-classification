# Music Genre Prediction

A four-class music genre classifier built with a CNN and log-Mel spectrograms. Upload a short audio clip to predict **Electronic, Hip-Hop, Pop, or Rock**.

## Project result

The currently selected model run reported **74.22% test accuracy** and **0.7445 macro F1** on **799 test tracks**. Per-genre F1: Electronic 0.7778, Hip-Hop 0.8392, Pop 0.6233, Rock 0.7380. These are the user's reported evaluation results; reproduce them only with the matching saved model, inference configuration, and test split.

## Repository contents

- `app.py` — Gradio inference app; loads the saved model once at startup.
- `model/` — add the model and matching inference configuration exported by Colab.
- `notebook/` — the 4,000-track training notebook.
- `requirements.txt` — Python dependencies for the app.
- `packages.txt` — system package required by the app (FFmpeg).

## Important: add the trained model

The model weights are not included in this starter package because they must come from your completed Colab run. In the final Colab export, find `model/music_genre_cnn_4genres.keras` and `model/inference_config.json`. Copy **both files from the same export** into this repository's `model/` folder. Do not mix files from different model runs: the configuration contains the genre order and preprocessing settings, including the Pop score adjustment.

## Run the app locally

Use Python 3.10 or newer. Install FFmpeg on your operating system, then run:

```bash
pip install -r requirements.txt
python app.py
```

Open the local Gradio URL printed in the terminal.

## Run the app from Google Colab

1. Upload this project folder (with both model files already placed in `model/`) to the Colab runtime, or create the same folder structure under `/content`.
2. Install the Python dependencies and FFmpeg if they are not already available.
3. Run `GRADIO_SHARE=true python app.py` from the folder containing `app.py`.
4. Open the temporary public Gradio link printed by the cell. It works only while that Colab runtime and process remain active.

You do **not** run the training notebook for every upload. The app loads the saved model once when it starts. Each upload is decoded, converted into a spectrogram, and passed through the already-trained model for prediction.

## Deploy to Hugging Face Spaces

1. Create a new Space and select the **Gradio** SDK.
2. Upload `app.py`, `requirements.txt`, `packages.txt`, and the `model/` folder containing both required files.
3. Use `README_SPACE.md` as the Space's root `README.md` (it contains the required Space configuration header).
4. Wait for the Space build to finish, then open the App tab.

The app schedules Gradio cache cleanup every hour for cached files older than one hour. Server restarts also clear the temporary cache. The application code does not copy uploaded songs into the training dataset or save them as permanent project records. Uploaded files are still transmitted to and temporarily handled by the hosting service, so do not upload private audio.

## Dataset

The training notebook downloads FMA-Small and its metadata directly into the Colab runtime. The original audio dataset is intentionally not included in this repository. The notebook saves metadata and track-level split records locally for project documentation.

## Limitations

- This is an academic demonstration, not a production music-rights or genre-identification service.
- Pop is the most difficult class in the reported evaluation.
- The model analyses the first 30 seconds of an uploaded clip.
- Displayed scores are model scores, not calibrated guarantees of correctness.
