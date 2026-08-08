# Brain Tumor Detection — How to Run

## What This Project Does

Trains and compares three deep-learning models on brain MRI images to detect tumors:

| Task | Classes | Models |
|---|---|---|
| **Binary** | Tumor (`yes`) vs No Tumor (`no`) | VGG16 · Custom CNN · ResNet50 |
| **Multi-class** | Healthy · Glioma · Meningioma · Pituitary | VGG16 · Custom CNN · ResNet50 |

The project has two parts:
1. **`brain_tumor_detection.ipynb`** — the notebook that loads the datasets, trains all 6 models, evaluates them, and saves them as `.h5` files.
2. **`app.py`** — a Streamlit web app that loads the trained `.h5` files and lets you upload an MRI scan, pick a model, and see a live prediction, plus a model-comparison tab with accuracy/loss charts.

---

## Dataset Structure

### Part 1 — Binary Dataset
Your folder must look exactly like this:

```
brain_tumor_dataset/
├── no/          ← MRI scans without a tumour (.jpg / .jpeg / .png)
└── yes/         ← MRI scans with a tumour    (.jpg / .jpeg / .png)
```

The notebook auto-discovers this folder by scanning `/kaggle/input` for any
directory that contains both a `yes/` and a `no/` sub-folder.

### Part 2 — Multi-Class Dataset
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

---

## Running on Kaggle (Recommended)

1. Go to [kaggle.com](https://www.kaggle.com) and sign in.
2. Click **+ New Notebook**.
3. Upload `brain_tumor_detection.ipynb` via **File → Import Notebook**.
4. On the right panel, click **Add Data** and attach:
   - **Dataset 1 (Binary):** search `Brain MRI Images for Brain Tumor Detection`
     by Navoneel Chakrabarty — the folder with `yes/` and `no/`.
   - **Dataset 2 (Multi-class):** search `Brain Tumor MRI Dataset`
     by Masoud Nickparvar — the folder with `Training/` and `Testing/`.
5. Under **Settings → Accelerator**, select **GPU T4 x2** (free tier).
6. Click **Run All** (`▶▶`) or press `Shift+Enter` through each cell.

---

## Running Locally

### Step 1 — Install dependencies

```bash
pip install tensorflow opencv-python imutils scikit-learn \
            matplotlib seaborn pandas numpy
```

> Requires Python 3.8–3.11. TensorFlow 2.10+ recommended.

### Step 2 — Organise your data

Place the datasets anywhere on your machine, then update these two lines
inside the notebook:

| Cell | Variable | Change to |
|---|---|---|
| Cell 2 | `locate_binary_dataset('/kaggle/input')` | path containing `yes/` and `no/` |
| Cell 14 | `MULTICLASS_DATASET_PATH` | path to the 4-class dataset root |

**Example:**
```python
# Cell 2 — change the default base path
binary_dataset_root = locate_binary_dataset('/home/user/data')

# Cell 14 — update the constant
MULTICLASS_DATASET_PATH = '/home/user/data/brain-tumor-mri-dataset'
```

### Step 3 — Launch Jupyter

```bash
jupyter notebook brain_tumor_detection.ipynb
```

Then run all cells: **Kernel → Restart & Run All**.

---

## Cell-by-Cell Walkthrough

| Cell | What it does |
|---|---|
| **1** | Imports every library; defines `crop_brain_roi`, `plot_accuracy_and_loss`, `plot_confusion_matrix` |
| **2** | Auto-finds the binary dataset; loads, crops, resizes, and normalises all images |
| **3** | Shows a class-distribution bar chart and a 2×5 grid of random sample scans |
| **4** | Splits data 80/20 (train/val); sets up augmentation and shared training callbacks |
| **5** | Builds VGG16 model with frozen ImageNet base + custom classifier head |
| **6** | Trains VGG16; plots accuracy & loss curves |
| **7** | Builds the Custom CNN (3 conv blocks, trained from scratch) |
| **8** | Trains Custom CNN; plots accuracy & loss curves |
| **9** | Builds ResNet50 with partial fine-tuning (last 10 layers unfrozen) |
| **10** | Trains ResNet50; plots accuracy & loss curves |
| **11** | Evaluates all 3 binary models — classification reports + confusion matrices |
| **12** | Summary table: best val accuracy and loss for each binary model |
| **13** | Demo: picks a random scan, runs it through ResNet50, shows prediction + confidence |
| **14** | Loads the 4-class dataset (merges Training + Testing, then re-splits) |
| **15** | Builds and trains Custom CNN, ResNet50, and VGG16 for 4-class task |
| **16** | Evaluates all 3 multi-class models — reports + confusion matrices |
| **17** | Overlaid accuracy curves + summary table for multi-class results |
| **18** | Saves all 6 trained models as `.h5` files |

---

## Output Files

After Cell 18 runs you will find these files in the notebook's working directory:

```
binary_vgg16.h5              — VGG16 binary classifier
binary_custom_cnn.h5         — Custom CNN binary classifier
binary_resnet50.h5           — ResNet50 binary classifier
multiclass_custom_cnn.h5     — Custom CNN 4-class classifier
multiclass_resnet50.h5       — ResNet50 4-class classifier
multiclass_vgg16.h5          — VGG16 4-class classifier
```

Load any model later with:
```python
import tensorflow as tf
model = tf.keras.models.load_model('binary_resnet50.h5')
```

---

## Running the Streamlit App (Interactive Demo)

Once you have the 6 `.h5` files (either trained yourself via the notebook, or
provided to you directly), you can run the interactive web app instead of
using the notebook.

### Step 1 — Folder structure

Put everything in **one folder** like this:

```
your-folder/
├── app.py
├── assets/
│   ├── binary_vgg16_history.png
│   ├── binary_customcnn_history.png
│   ├── binary_resnet50_history.png
│   ├── multiclass_customcnn_history.png
│   ├── multiclass_resnet50_history.png
│   ├── multiclass_vgg16_history.png
│   └── multiclass_accuracy_all.png
├── binary_vgg16.h5
├── binary_custom_cnn.h5
├── binary_resnet50.h5
├── multiclass_custom_cnn.h5
├── multiclass_resnet50.h5
└── multiclass_vgg16.h5
```

The `assets/` folder holds the training-history chart images shown in the
app's **Model Comparison** tab — the app won't run correctly without it.

### Step 2 — Install dependencies

```bash
pip install streamlit tensorflow opencv-python imutils pandas numpy pillow
```

> If you already ran the notebook locally, you likely have everything except
> `streamlit`. Just run `pip install streamlit` in that case.

### Step 3 — Launch the app

From inside `your-folder`, run:

```bash
streamlit run app.py
```

This opens the app in your browser at `http://localhost:8501`. Use the
sidebar to pick a model, upload an MRI image in the **Detection** tab, and
check the **Model Comparison** tab for accuracy/loss charts across all 6
models.

### What the app does automatically

- Resizes images to the correct input size per model (224×224 for Binary,
  128×128 for MultiClass).
- Applies the same black-padding crop (`crop_brain_roi`) used during Binary
  model training — MultiClass images are used as-is, since that's how they
  were trained.
- Caches each loaded model so switching between them in the sidebar doesn't
  reload from disk every time.

⚠️ This app is for educational/demo purposes — predictions on images that
don't match the training data's format (wrong scan type, wrong orientation,
non-MRI images) are not reliable.

---

## Expected Training Times (GPU)

| Model | Task | Approx. time |
|---|---|---|
| VGG16 | Binary | 5–10 min |
| Custom CNN | Binary | 3–6 min |
| ResNet50 | Binary | 8–15 min |
| Custom CNN | Multi-class | 5–10 min |
| ResNet50 | Multi-class | 10–20 min |
| VGG16 | Multi-class | 10–20 min |

EarlyStopping will often stop training earlier than the epoch limit.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `❌ Could not find the binary dataset` | Check that `yes/` and `no/` folders are inside the Kaggle input path |
| `⚠️ Folder not found: Training` | Verify the multi-class dataset is attached and `MULTICLASS_DATASET_PATH` is correct |
| `OOM / ResourceExhaustedError` | Reduce `batch_size` in Cell 6/8 from 32 → 16, or Cell 10 from 16 → 8 |
| `ModuleNotFoundError: imutils` | Notebook: run `!pip install imutils` in a cell. App: run `pip install imutils` in your terminal, in the same Python environment you use to launch `streamlit run app.py` |
| `ModuleNotFoundError: streamlit` (or any other module) | Run `pip install <module-name>` in the same environment/terminal you use to run the app — a fresh install won't have packages until you install them there |
| App shows a broken image icon in "Model Comparison" | The `assets/` folder is missing or not next to `app.py` — re-check the folder structure above |
| App error loading a `.h5` model | Make sure all 6 `.h5` files are in the same folder as `app.py`, with the exact filenames listed in "Output Files" above |
| Validation accuracy stuck near 50% | Increase `epochs` slightly or check that data loaded correctly in Cell 3 |
