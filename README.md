# Brain Tumor Detection from MRI Scans

A deep learning comparative study using **Custom CNN**, **VGG16**, and **ResNet50** to classify brain MRI scans — with an interactive Streamlit demo app.

> ⚠️ **Disclaimer:** This project is for **educational and research purposes only**. It is **not** a medical diagnostic tool, has not been clinically validated, and must **not** be used to make or support real medical decisions. Always consult a qualified medical professional for diagnosis.

## Overview

The notebook covers two tasks:

1. **Binary classification** — Tumor (`yes`) vs. No Tumor (`no`)
2. **Multi-class classification** — Healthy · Glioma · Meningioma · Pituitary

For each task, three models are trained and compared:
- A custom CNN trained from scratch
- VGG16 (transfer learning, frozen ImageNet weights)
- ResNet50 (partial fine-tuning — last 10 layers unfrozen)

All 6 trained models classify an MRI image of **any input resolution** — no manual resizing needed.

## Project Structure

| File | Purpose |
|---|---|
| `brain_tumor_detection.ipynb` | Trains, evaluates, saves, and uploads all 6 models |
| `app.py` | Streamlit demo — upload a scan, pick a model, see a live prediction |
| `requirements.txt` | Dependencies for the notebook |
| `assets/` | Training-curve and confusion-matrix images used by the app's Model Comparison tab |

Trained models are hosted on Hugging Face Hub: **[mvdu/brain-tumor-models](https://huggingface.co/mvdu/brain-tumor-models)** — `app.py` downloads them automatically at runtime, so no large `.h5` files are bundled in this repo.

## Datasets

| Task | Dataset | Source |
|---|---|---|
| Binary | `akar49/MRI_Classification` (`notumor` / `tumor`, ~3.6k images) | [Hugging Face](https://huggingface.co/datasets/akar49/MRI_Classification) — downloaded automatically by the notebook, with a duplicate-check step between train/test |
| Multi-class | `brain-tumor-mri-dataset` (4 classes) | [Kaggle — Masoud Nickparvar](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) — classes dynamically balanced to the smallest available class |

The binary dataset needs no manual setup — the notebook downloads it directly via the `datasets` library. The multi-class dataset structure looks like this once attached:

```
brain-tumor-mri-dataset/
├── Training/
│   ├── notumor/
│   ├── glioma/
│   ├── meningioma/
│   └── pituitary/
└── Testing/
    ├── notumor/
    ├── glioma/
    ├── meningioma/
    └── pituitary/
```

The loader auto-detects its actual path (handling Kaggle's occasional extra nesting) — no manual path editing needed in most cases.

## Results

Validation-set accuracy from the final training run (see the notebook for full confusion matrices and classification reports):

| Model | Task | Accuracy | Val Loss |
|---|---|---|---|
| ResNet50 (fine-tuned) | Binary | **99.07%** | 0.04 |
| VGG16 (frozen) | Binary | 97.83% | 0.08 |
| Custom CNN | Binary | 96.43% | 0.10 |
| ResNet50 (fine-tuned) | Multi-class | **96.18%** | 0.15 |
| VGG16 (frozen) | Multi-class | 83.19% | 0.49 |
| Custom CNN | Multi-class | 71.04% | 0.72 |

**ResNet50 is the strongest model on both tasks.** The multi-class Custom CNN is noticeably weaker — in particular it struggles to distinguish **Meningioma** from the other three classes (see its confusion matrix in the notebook or the app's Comparison tab). This is a real capability gap for a simple from-scratch CNN on a harder 4-class task, not a bug.

## Known Limitations

- The multi-class Custom CNN performs meaningfully worse than the other two multi-class models, especially on Meningioma — a genuine model-capability limitation, not a data or pipeline bug.
- The multi-class dataset mixes different MRI scan orientations (axial, sagittal, coronal), which makes the 4-class task inherently harder than the binary one.
- Models were evaluated on a held-out validation split, not a fully separate external test set.
- No cross-validation was performed.

---

## Running the Notebook on Kaggle (Recommended)

1. Go to [kaggle.com](https://www.kaggle.com) and sign in.
2. Click **+ New Notebook**.
3. Upload `brain_tumor_detection.ipynb` via **File → Import Notebook**.
4. On the right panel, click **Add Data** and attach **Brain Tumor MRI Dataset** by Masoud Nickparvar (for the multi-class task — the binary dataset downloads automatically, no attachment needed).
5. Under **Settings → Accelerator**, select **GPU T4 x2** (free tier).
6. Click **Run All** (`▶▶`) or press `Shift+Enter` through each cell, in order, top to bottom.

## Running the Notebook Locally

```bash
pip install tensorflow opencv-python imutils scikit-learn \
            matplotlib seaborn pandas numpy datasets imagehash \
            huggingface_hub kagglehub

jupyter notebook brain_tumor_detection.ipynb
```

Requires Python 3.8–3.11, TensorFlow 2.10+. The binary dataset downloads automatically. For the multi-class dataset, either let `kagglehub.dataset_download(...)` fetch it (needs a local Kaggle API token) or download it manually — the auto-detect loader will find it as long as it matches one of the candidate paths in the "Load Multi-Class Dataset" cell. Then: **Kernel → Restart & Run All**.

## Notebook Walkthrough

| Section | What it does |
|---|---|
| Imports & Helpers | Loads every library; defines `crop_brain_roi`, `plot_accuracy_and_loss`, `plot_confusion_matrix`, and `predict_mri_scan` (classifies an MRI image of **any resolution**) |
| *Optional* — Load Pre-Trained Models | Skips training entirely by downloading the 6 already-trained `.h5` files from `mvdu/brain-tumor-models` on Hugging Face |
| Load Binary Dataset | Downloads `akar49/MRI_Classification`, checks for train/test duplicates, crops/resizes/normalises all images |
| Binary EDA, Split & Augmentation | Class-distribution chart, sample scans, 80/20 stratified split, rotation/flip augmentation |
| Binary VGG16 / Custom CNN / ResNet50 | Each model gets its **own fresh** `EarlyStopping`/`ReduceLROnPlateau` instances — important if you modify these cells: reusing one shared callback list across multiple `.fit()` calls causes it to compare a new model's loss against a *previous* model's best value, which can restore badly undertrained weights |
| Binary Evaluation & Demo | Confusion matrices, classification reports, and a demo that classifies an image resized to a random arbitrary resolution |
| Load Multi-Class Dataset | Auto-detects the Kaggle dataset path, balances all 4 classes dynamically, applies `crop_brain_roi()` |
| Crop Sanity Check | Before/after `crop_brain_roi()` samples for each class — run before training to confirm cropping isn't cutting off scan regions |
| Multi-Class Build & Train | Same fresh-callbacks-per-model pattern. ResNet50 uses partial fine-tuning (last 10 layers) with the same preprocessing `Lambda` layer as its binary counterpart |
| Multi-Class Evaluation & Demo | Confusion matrices, classification reports, arbitrary-size demo |
| Save All Models | Saves all 6 models as `.h5` files |
| Upload to Hugging Face Hub | Pushes all 6 `.h5` files to `mvdu/brain-tumor-models` (requires a Hugging Face **write** token) |

### Output files

```
binary_vgg16.h5
binary_custom_cnn.h5
binary_resnet50.h5
multiclass_custom_cnn.h5
multiclass_resnet50.h5
multiclass_vgg16.h5
```

```python
import tensorflow as tf
model = tf.keras.models.load_model('binary_resnet50.h5')
```

> `binary_resnet50.h5` and `multiclass_resnet50.h5` both contain a `Lambda` preprocessing layer. Loading them with plain `tf.keras.models.load_model(...)` can be unreliable across Python/Keras versions — `app.py` works around this with a manual architecture-rebuild + weight-loading step (see `build_binary_resnet50_architecture()` / `build_multiclass_resnet50_architecture()` and `load_weights_manually()` in `app.py`).

### Uploading models to Hugging Face Hub

1. Get a **write**-access token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (Create new token → Write role).
2. Run the upload cell — `login()` prompts you to paste the token (input hidden).
3. Wait for all 6 "✅ Uploaded: ..." confirmations.

Forking this project? Change `HF_REPO_ID` in that cell (and in `app.py`) to your own `username/repo-name`.

### Expected training times (GPU, Kaggle T4 x2)

| Model | Task | Approx. time |
|---|---|---|
| VGG16 | Binary | 5–10 min |
| Custom CNN | Binary | 3–6 min |
| ResNet50 | Binary | 8–15 min |
| Custom CNN | Multi-class | 5–10 min |
| ResNet50 | Multi-class | 10–20 min |
| VGG16 | Multi-class | 5–10 min |

`EarlyStopping` will often stop training earlier than the epoch limit.

---

## Running the Streamlit App

`app.py` downloads all 6 models from Hugging Face Hub automatically the first time it runs (and caches them locally afterward) — **the `.h5` files don't need to be local.**

### Folder structure

```
your-folder/
├── app.py
├── requirements.txt
└── assets/
    ├── vgg16_b.png            (+ vgg16_b1.png, cnn_b.png, cnn_b1.png, resnet50_b.png, resnet50_b1.png)
    ├── vgg16_m.png / vgg16_m1.png
    ├── cnn_mpng.png / cnn_m1.png
    └── resnet50_m.png / resnet50_m1.png
```

The `assets/` folder holds the training-curve and confusion-matrix images shown in the app's **Model Comparison** tab.

### Install & run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. Use the sidebar to pick a model, upload an MRI image of **any size** in the **Detection** tab, and check the **Model Comparison** tab for accuracy/loss curves and confusion matrices across all 6 models.

### What the app does automatically

- Downloads and caches each model from `mvdu/brain-tumor-models` on first use (`@st.cache_resource` + `hf_hub_download`).
- Resizes uploaded images to the correct input size per model (224×224 for Binary, 128×128 for MultiClass) — works with any input resolution.
- Applies the same `crop_brain_roi` preprocessing used during training.
- Rebuilds `binary_resnet50.h5` / `multiclass_resnet50.h5`'s architecture natively and loads weights by exact layer name, avoiding a Lambda-layer deserialization issue that can otherwise crash on load.

⚠️ Predictions on images that don't match the training data's format (wrong scan type, wrong orientation, non-MRI images) are not reliable.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `❌ Could not find the binary dataset` | Hugging Face download failed — check your connection / that `datasets` installed (`!pip install -q datasets`) |
| `❌ Could not find the multi-class dataset` | Make sure the Kaggle dataset is attached via **Add Data**, or that `kagglehub.dataset_download(...)` succeeded locally |
| `NameError: name 'X' is not defined` after a Kaggle session restart | The kernel lost all in-memory variables — re-run the **Imports** cell and the relevant **dataset-loading** cell before re-running any training/demo cell |
| A model's `EarlyStopping` restores oddly poor weights despite steady improvement in the log | Check its `.fit()` call uses its **own** fresh callback instances, not one shared across multiple models' training in the same session |
| `OOM / ResourceExhaustedError` | Reduce `batch_size` in the relevant training cell |
| `ModuleNotFoundError: <package>` | Notebook: `!pip install <package>` in a cell. App: `pip install <package>` in the same environment used to launch `streamlit run app.py` |
| `401 Unauthorized` when uploading to Hugging Face | Token wasn't pasted into `login()`, or lacks **Write** permission — generate a new one with the Write role |
| App shows a broken image icon in "Model Comparison" | The `assets/` folder is missing or not next to `app.py` |
| App error loading a model | Check `HF_REPO_ID` in `app.py` matches `mvdu/brain-tumor-models` and the repo is public (or you're authenticated) |

## License

Specify a license here (e.g. MIT) if you intend others to reuse this code.
