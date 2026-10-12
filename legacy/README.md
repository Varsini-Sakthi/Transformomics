> **Legacy (v1).** This is the original proof-of-concept, now named Transformomics (formerly SpatialFormer). The Hugging Face Space URLs below keep the old name until the Space is renamed.

# Transformomics - Spatially Aware Cell State Prediction

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Varsini-Sakthi/Spatialformer)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A spatially aware transformer that integrates **gene expression** and **spatial neighborhood context** to predict cell cluster identity in spatial transcriptomics data. Built on a 10x Genomics Visium mouse brain dataset.

🔗 **Live Demo**: [huggingface.co/spaces/Varsini-Sakthi/Spatialformer](https://huggingface.co/spaces/Varsini-Sakthi/Spatialformer)

---

## Overview

Standard single-cell models treat each cell independently. Transformomics extends this by giving each cell access to its spatial neighbors, encoding tissue microenvironment as a sequence of tokens fed into a transformer encoder.

**Key idea:** For each spot, a sequence `[CLS, query_cell, neighbor_1, ..., neighbor_6]` is constructed. The transformer learns which neighbors to attend to, and the CLS token embedding is used for classification.

---

## Model Architecture

```
Gene expression (2000 HVGs) ──► Linear projection ──┐
Spatial coordinates (x, y)  ──► Linear projection ──┼──► Token embeddings
Token type (CLS/query/nbr)  ──► Embedding lookup  ──┘
                                        │
                          Transformer Encoder (2 layers, 4 heads)
                                        │
                                   CLS token
                                        │
                            Classification head (10 classes)
```

| Hyperparameter | Value |
|---|---|
| Embed dim | 64 |
| Attention heads | 4 |
| Transformer layers | 2 |
| Feed-forward dim | 128 |
| Neighbors (k) | 6 |
| Sequence length | 8 (CLS + query + 6 neighbors) |
| Optimizer | AdamW (lr=1e-3) |
| Loss | Cross-entropy |
| Early stopping patience | 10 epochs |

---

## Dataset

- **Source**: 10x Genomics Visium - Mouse Brain Coronal Section
- **Spots**: 2,796 tissue-covered spots
- **Genes**: 32,285 → filtered to 2,000 highly variable genes
- **Labels**: Graph-based clustering (10 clusters) from Cell Ranger output
- **Spatial coordinates**: Pixel coordinates from `obsm['spatial']`

---

## Results

| Model | Val Accuracy | Val Macro-F1 |
|---|---|---|
| Logistic Regression | 0.9232 | 0.9176 |
| MLP (512, 256) | 0.9107 | 0.8992 |
| XGBoost | 0.8982 | 0.8858 |
| **SpatialTransformer** | **0.8911** | **0.8795** |

> The transformer achieves competitive performance while additionally providing spatial attention maps and neighborhood-aware embeddings - interpretability that purely expression-based models lack.

---

## Interpretation

**Attention analysis** reveals that the CLS token attends primarily to the query cell (~0.59) with distributed attention across neighbors (~0.045 each), indicating the model uses spatial context as a regularizer rather than a primary signal.

**Gene saliency** (gradient × input) identifies cluster-specific marker genes consistent with known mouse brain cell type markers.

**Spatial attention maps** show that neighbor reliance varies by tissue region, boundary spots between anatomical domains show higher neighbor attention than spots in homogeneous regions.

---

## Project Structure

```
transformomics/
├── Spatially Aware Cell State Prediction.ipynb   # Full training notebook
├── app.py                                         # Gradio inference app
├── requirements.txt                               # Dependencies
├── checkpoints/
│   ├── inference_pipeline.pt                      # Model + preprocessors
│   ├── label_encoder.pkl                          # Label encoder
│   ├── val_cls_attn.npy                           # Attention weights
│   ├── class_saliency.npy                         # Gene saliency scores
│   └── hvg_names.npy                              # HVG gene names
└── figures/
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── umap_cls.png
    ├── spatial_predictions.png
    ├── attn_mean_bar.png
    ├── attn_by_cluster.png
    ├── spatial_attn_map.png
    └── gene_saliency.png
```

---

## Quickstart

### Installation

```bash
conda create -n spaformer python=3.10
conda activate spaformer
pip install torch scanpy anndata scikit-learn xgboost umap-learn gradio matplotlib seaborn
```

### Run the app locally

```bash
git clone https://huggingface.co/spaces/Varsini-Sakthi/Spatialformer
cd Spatialformer
python app.py
```

Open `http://127.0.0.1:7860`, upload any preprocessed `.h5ad` file with `obsm['spatial']`, and click **Run inference**.

### Input format

The `.h5ad` file must have:
- `adata.X` - raw or normalized gene expression matrix
- `adata.var_names` - gene names (overlap with training HVGs is handled automatically)
- `adata.obsm['spatial']` - 2D spatial coordinates

---

## Reproducing Training

Open `Spatially Aware Cell State Prediction.ipynb` and run all cells in order. The notebook covers:

1. Data loading from 10x Visium format
2. QC, normalization, HVG selection
3. Spatial k-NN graph construction (k=6)
4. Token sequence assembly
5. SpatialTransformer training (30 epochs, early stopping)
6. Baseline comparisons
7. Attention weight extraction
8. Gene saliency analysis

---

## Dependencies

| Package | Purpose |
|---|---|
| `scanpy` | Single-cell data processing |
| `torch` | Model training |
| `scikit-learn` | Baselines + preprocessing |
| `umap-learn` | Embedding visualization |
| `gradio` | Inference app |
| `seaborn` | Figure generation |
| `xgboost` | XGBoost baseline |

---

## References

- Wolf, F. A., Angerer, P., & Theis, F. J. (2018). SCANPY: large-scale single-cell gene expression data analysis. Genome biology, 19(1), 15.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.

---

## Author

**Varsini Sakthivadivel Ramasamy**

**Ms Bioinformatics, Johns Hopkins University**

**B.Tech Biotechnology, VIT University**

Spatial transcriptomics + deep learning research  
🔗 [Hugging Face](https://huggingface.co/Varsini-Sakthi)
