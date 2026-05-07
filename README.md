

# ScreenShot

Source code, pretrained models, evaluation data, and interactive dashboard for reproducing the experiments.

## Interactive Dashboard

Upload any drug screening dataset, run real-time inference, and explore dose-response predictions, hit detection analytics, and combination response surfaces.

<video src="https://github.com/user-attachments/assets/73565278-021e-4bc7-a764-af4437c6d038" autoplay loop muted playsinline></video>

See [Dashboard setup](#dashboard-setup) for instructions.

## Setup

```bash
# Requires Python 3.11 or higher

# Option 1: venv
python -m venv venv
source venv/bin/activate

# Option 2: conda
conda create -n screenshot python=3.11 -y
conda activate screenshot

# Install
pip install -e .
```

## Repository structure

```
configs/           Experiment and dataset configurations
data/              Evaluation datasets (.feather)
models/            Pretrained ScreenShot and MLP weights + drug libraries
screenshot/        ScreenShot model package
datascreen/        Drug library and data processing package
scripts/           Experiment scripts
dashboard/         Interactive web dashboard (optional)
```

## Few-shot experiments (Table 1)

```bash
cd scripts

# ScreenShot
python few_shot.py --method screenshot \
    --config ../configs/config_few_shot.yaml \
    --data-config ../configs/config_batchie.yaml \
    --input ../data/batchie.feather \
    --device cuda

# XGBoost
python few_shot.py --method xgb \
    --config ../configs/config_few_shot.yaml \
    --data-config ../configs/config_batchie.yaml \
    --input ../data/batchie.feather

# TabPFN
python few_shot.py --method tabpfn \
    --config ../configs/config_few_shot.yaml \
    --data-config ../configs/config_batchie.yaml \
    --input ../data/batchie.feather \
    --device cuda

# MLP (from scratch)
python few_shot.py --method mlp \
    --config ../configs/config_few_shot.yaml \
    --data-config ../configs/config_batchie.yaml \
    --input ../data/batchie.feather \
    --device cuda

# MLP (fine-tuned)
python few_shot.py --method mlp_finetuned \
    --config ../configs/config_few_shot.yaml \
    --data-config ../configs/config_batchie.yaml \
    --input ../data/batchie.feather \
    --device cuda
```

Replace `config_batchie.yaml` / `batchie.feather` with the corresponding dataset:

| Dataset | Config | Data |
|---------|--------|------|
| BATCHIE | `config_batchie.yaml` | `batchie.feather` |
| PDO-Breast | `config_pdo_breast.yaml` | `pdo_breast.feather` |
| NCI-ALMANAC | `config_nci_almanac.yaml` | `nci_almanac.feather` |
| GDSC-SQ | `config_gdsc_sq.yaml` | `gdsc_sq.feather` |

Results are saved to `results/few_shot/`.

## Active learning (Figure 3)

```bash
python active_learning_ablation.py \
    --config ../configs/config_active_learning.yaml \
    --data-config ../configs/config_batchie.yaml \
    --input ../data/batchie.feather \
    --device cuda
```

Runs random, cold-start (k-medoids), and adaptive selection (1 and 2 rounds).
Results are saved to `results/active_learning/`.

## Active learning ablation (Appendix)

```bash
python active_learning_ablation.py \
    --config ../configs/config_active_learning_ablation.yaml \
    --data-config ../configs/config_batchie.yaml \
    --input ../data/batchie.feather \
    --device cuda
```

Results are saved to `results/active_learning/ablation/`.

## Dashboard setup

The dashboard provides an interactive web interface for:
- Uploading drug screening data (CSV, Feather, Excel)
- Real-time streaming inference with device selection (CPU/MPS/CUDA)
- Dose-response curve visualization (overlay and per-sample views)
- Hit detection analytics (heatmap and bar plots)
- 3D combination response surface prediction

### 1. Install dependencies

```bash
pip install -e ".[dashboard]"
```

### 2. Install and build the frontend

```bash
cd dashboard/frontend
npm install
cd ..
```

### 3. Start the dashboard

```bash
make dev
```

This starts both the backend (FastAPI on port 8000) and the frontend (Vite on port 5173). Open http://localhost:5173 in your browser.

### 4. Usage

1. Upload a dataset from `data/` (e.g. `pdo_breast.feather` or `batchie.feather`)
2. Review the detected column mapping
3. Click "Generate" to create dose-response queries
4. Select a device (CPU, MPS, or CUDA) and click "Run Inference"
5. Explore predictions in overlay or per-sample view
6. Scroll to the analytics section for hit detection
7. Use the combination section to predict 3D drug pair response surfaces
