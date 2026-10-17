# Real Visium datasets for Transformomics

Goal: replace "synthetic data only" with evidence on real tissue. Each dataset below is tagged with
what it can and cannot support. **Verified** = I confirmed it from the source page; **unverified** =
check before relying on it.

| # | Dataset | Design | Labels | Supports |
|---|---|---|---|---|
| 1 | Human DLPFC (Maynard et al. 2021, spatialLIBD) | 12 sections, 3 donors, adjacent replicate pairs | Manual layers L1-L6 + WM | Everything: donor-out CV, boundaries, saliency, faithfulness |
| 2 | 5xFAD / WT time course (GSE174321; Choi et al. 2023, Exp Mol Med) | 8 coronal sections (3 and 7 months, WT and 5xFAD, n=2) | Expression-derived clusters only | Microglia programme task; white-matter/grey-matter question; exploratory (n=2/group) |
| 3 | 10x mouse brain serial sections (sagittal anterior 1+2, posterior 1+2) | Adjacent serial sections | None | Section-out replication, programme task, continuity with v1 |
| 4 | 10x human lymph node | 1 section, 4,035 spots | None from 10x | Non-brain tissue with distinct structures; ablations only |
| 5 | 10x human breast cancer, Block A sections 1 and 2 | 2 adjacent sections | None from 10x | Tumour/stroma interfaces; replicate pair |

Only #1 has independent ground truth. Do not use expression-derived clusters (#2) as boundary truth
for claims about boundaries: they are circular for the same reason v1's labels were.

## 1. DLPFC (spatialLIBD)
```r
# R
BiocManager::install(c("spatialLIBD", "zellkonverter"))
```
```bash
Rscript scripts/export_dlpfc.R data/dlpfc            # fetch_data(type="spe") via ExperimentHub, split to per-section h5ad
python scripts/make_manifest.py data/dlpfc --meta data/dlpfc/sections.csv \
    --label-key layer_guess_reordered --xcol spatial_x --ycol spatial_y --gene-col gene_name
python -m transformomics.run_multi --config configs/dlpfc.yaml --out results_dlpfc
```
Check after export: spots per section (~3-4.7k), the label column has L1-L6/WM (some spots unlabelled; the
pipeline keeps them as neighbour tokens but never as query spots), `obsm["spatial"]` exists, var_names are
gene symbols (needed for marker enrichment). Column names (`sample_id`, `subject`, `layer_guess_reordered`)
are the usual spatialLIBD ones - the R script stops and lists the real names if they differ.
**Unverified:** that `writeH5AD` round-trips every section cleanly on your Bioconductor version; if it
fails, export counts + coordinates + labels to CSV/MTX from R and build the AnnData in Python.
A third-party Hugging Face mirror of 4 DLPFC slices exists (`Shaow/dlpfc_visium_4slice`) but is not vetted.

## 2. 5xFAD time course
GEO accession **GSE174321**: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174321
Paper: https://www.nature.com/articles/s12276-023-01123-9 (8 Visium samples, Space Ranger 1.1.0, mm10).
**Unverified:** whether the GEO supplementary files include the spatial coordinate files
(`tissue_positions`/`spatial/` folder). Transformomics cannot run without coordinates; if they are
missing, request them from the authors. Once you have per-sample Space Ranger outputs:
```python
import scanpy as sc
a = sc.read_visium("path/to/sample_dir"); a.var_names_make_unique(); a.write_h5ad("data/5xfad/S1.h5ad")
```
then list sections in a manifest (`donor_id` = animal ID if known) and use `configs/fivexfad_program.yaml`.

## 3-5. 10x datasets
```bash
pip install squidpy
python scripts/fetch_10x.py --out data/10x V1_Human_Lymph_Node V1_Breast_Cancer_Block_A_Section_1 V1_Mouse_Brain_Sagittal_Posterior
```
Verified sample ids: `V1_Human_Lymph_Node`, `V1_Breast_Cancer_Block_A_Section_1`, `V1_Mouse_Brain_Sagittal_Posterior`.
Scanpy's `sc.datasets.visium_sge` also works but is deprecated in favour of squidpy.
**Unverified:** the exact ids for the serial sections ("Serial Section 1/2"); if an id errors, download
"Feature / barcode matrix HDF5 (filtered)" and "Spatial imaging data" from the 10x dataset page
(e.g. https://www.10xgenomics.com/datasets/mouse-brain-serial-section-1-sagittal-anterior-1-standard-1-1-0)
and load with `scanpy.read_visium`.
Without labels these support the programme task and the real/shuffled/none ablations, not label metrics.
For lymph node and breast cancer you would define programmes yourself (e.g. B-cell, T-cell,
germinal-centre; tumour/stroma) in a programmes YAML.

## Splitting rules that matter
* DLPFC: split by **donor** (`split.by: donor`). Same-donor sections are near-duplicates.
* Never concatenate sections before building graphs; `load_sections` builds one graph per section.
* Run the null (`--null-coords 1`) on every dataset before interpreting a real-vs-shuffled gap.
