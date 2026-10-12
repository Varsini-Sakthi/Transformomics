## Writefile app.py
import gradio as gr
import torch
import numpy as np
import pickle
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from torch import nn 
import umap
import pandas as pd

class SpatialTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, embed_dim=64,
                 num_heads=4, num_layers=2, ff_dim=128, 
                 dropout=0.1, max_token_types=3):
        super().__init__()
        self.expr_proj = nn.Linear(input_dim, embed_dim)
        self.spatial_proj = nn.Linear(2, embed_dim)
        self.token_embed = nn.Embedding(max_token_types, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, expression, spatial, token_type):
        x = (self.expr_proj(expression) +
             self.spatial_proj(spatial) +
             self.token_embed(token_type))
        x[:, 0:1, :] = self.cls_token.expand(x.size(0), -1, -1)
        x = self.transformer(x)
        return self.classifier(self.norm(x[:, 0, :]))

device = torch.device("cpu")
ckpt = torch.load("checkpoints/inference_pipeline.pt", map_location=device)
cfg = ckpt["config"]
model = SpatialTransformer(**cfg).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

with open("checkpoints/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
    
hvg_genes = ckpt["hvg_genes"]
coords_mean = np.array(ckpt["coords_mean"])
coords_std = np.array(ckpt["coords_std"])
label_names = [str(c) for c in ckpt["label_classes"]]
def run_inference(h5ad_file):
    adata = sc.read_h5ad(h5ad_file)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Align columns to the training HVG order; genes absent from the upload stay zero.
    # (The previous version appended zero columns at the END, which shifted every present
    # gene to the wrong input position whenever any HVG was missing.)
    var_pos = {g: i for i, g in enumerate(adata.var_names)}
    present = [(j, var_pos[g]) for j, g in enumerate(hvg_genes) if g in var_pos]
    missing = len(hvg_genes) - len(present)
    Xfull = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X = np.zeros((adata.n_obs, len(hvg_genes)), dtype=np.float32)
    cols_train, cols_up = zip(*present)
    X[:, list(cols_train)] = Xfull[:, list(cols_up)]

    coords = adata.obsm["spatial"].astype(np.float32)
    coords_norm = (coords - coords_mean) / coords_std

    k = 6
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
    _, indices        = nbrs.kneighbors(coords)
    neighbor_indices  = indices[:, 1:]

    n_spots, n_genes = X.shape
    seq_len = k + 2
    all_seq = np.zeros((n_spots, seq_len, n_genes), dtype=np.float32)
    all_sp = np.zeros((n_spots, seq_len, 2),      dtype=np.float32)
    all_tt = np.zeros((n_spots, seq_len),         dtype=np.int64)

    for i in range(n_spots):
        all_seq[i, 1] = X[i];           all_sp[i, 1] = coords_norm[i]; all_tt[i, 1] = 1
        all_seq[i, 2:] = X[neighbor_indices[i]]
        all_sp[i, 2:] = coords_norm[neighbor_indices[i]]; all_tt[i, 2:] = 2

    preds, probs, cls_embs = [], [], []
    bs = 64
    with torch.no_grad():
        for start in range(0, n_spots, bs):
            e = torch.tensor(all_seq[start:start+bs]).to(device)
            s = torch.tensor(all_sp[start:start+bs]).to(device)
            tt = torch.tensor(all_tt[start:start+bs]).to(device)
            logits = model(e, s, tt)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            x = model.expr_proj(e) + model.spatial_proj(s) + model.token_embed(tt)
            x[:, 0:1, :] = model.cls_token.expand(x.size(0), -1, -1)
            x = model.transformer(x)
            cls_embs.append(model.norm(x[:, 0, :]).cpu().numpy())

    preds = np.array(preds)
    probs = np.concatenate(probs, axis=0)
    cls_embs = np.concatenate(cls_embs, axis=0)
    pred_labels = le.inverse_transform(preds)

    fig1, ax1 = plt.subplots(figsize=(6, 5))
    sc1 = ax1.scatter(coords[:, 0], coords[:, 1], c=preds, cmap="tab10", s=10, alpha=0.9)
    ax1.invert_yaxis()
    plt.colorbar(sc1, ax=ax1, label="Predicted cluster")
    ax1.set_title("Predicted clusters on tissue")
    plt.tight_layout()

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=42)
    emb2d = reducer.fit_transform(cls_embs)
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    sc2 = ax2.scatter(emb2d[:, 0], emb2d[:, 1], c=preds, cmap="tab10", s=8, alpha=0.8)
    plt.colorbar(sc2, ax=ax2, label="Predicted cluster")
    ax2.set_title("UMAP of CLS embeddings")
    plt.tight_layout()

    df = pd.DataFrame({
        "spot": adata.obs_names,
        "x": coords[:, 0], "y": coords[:, 1],
        "predicted_cluster": pred_labels,
        **{f"prob_cluster_{label_names[i]}": probs[:, i] for i in range(len(label_names))}
    })
    csv_path = "predictions.csv"
    df.to_csv(csv_path, index=False)

    summary = (f"YES {n_spots} spots processed\n"
               f"YES {missing} HVGs zero-padded\n\n"
               f"Cluster counts:\n{pd.Series(pred_labels).value_counts().to_string()}")

    return fig1, fig2, csv_path, summary

with gr.Blocks(title="Transformomics") as demo:
    gr.Markdown("## Transformomics - Spatially Aware Cell State Prediction")
    gr.Markdown("Upload a preprocessed '.h5ad' file with 'obsm['spatial']'.")
    with gr.Row():
        file_input = gr.File(label="Upload .h5ad file", file_types=[".h5ad"])
        run_btn = gr.Button("Run inference", variant="primary")
    summary_box = gr.Textbox(label="Summary", lines=8)

    with gr.Row():
        spatial_plot = gr.Plot(label="Spatial predictions")
        umap_plot = gr.Plot(label="UMAP embeddings")
    csv_output = gr.File(label="Download predictions CSV")
    run_btn.click(fn=run_inference, inputs=[file_input],
                  outputs=[spatial_plot, umap_plot, csv_output, summary_box])

if __name__ == "__main__":
    demo.launch()
    
