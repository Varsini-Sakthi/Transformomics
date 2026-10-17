# Export the 12 DLPFC sections (spatialLIBD) to per-section .h5ad files + a sections.csv.
# Run once in R:  Rscript scripts/export_dlpfc.R data/dlpfc
# Column names below are the usual spatialLIBD ones - the script stops with the available names if they differ.
suppressPackageStartupMessages({ library(spatialLIBD); library(zellkonverter); library(SpatialExperiment) })
out <- commandArgs(trailingOnly = TRUE)[1]; if (is.na(out)) out <- "data/dlpfc"
dir.create(out, recursive = TRUE, showWarnings = FALSE)
SAMPLE_COL <- "sample_id"; DONOR_COL <- "subject"; LABEL_COL <- "layer_guess_reordered"

spe <- fetch_data(type = "spe")                      # downloads via ExperimentHub (large, cached)
cd <- colData(spe)
need <- c(SAMPLE_COL, DONOR_COL, LABEL_COL)
if (!all(need %in% colnames(cd))) stop("missing columns; available: ", paste(colnames(cd), collapse = ", "))
xy <- spatialCoords(spe)
colData(spe)$spatial_x <- xy[, 1]; colData(spe)$spatial_y <- xy[, 2]
meta <- unique(data.frame(section_id = as.character(cd[[SAMPLE_COL]]), donor_id = as.character(cd[[DONOR_COL]])))
for (s in meta$section_id) {
  sub <- as(spe[, colData(spe)[[SAMPLE_COL]] == s], "SingleCellExperiment")
  writeH5AD(sub, file.path(out, paste0(s, ".h5ad")), X_name = "counts")
}
write.csv(meta, file.path(out, "sections.csv"), row.names = FALSE)
message("wrote ", nrow(meta), " sections to ", out, "\nNext: python scripts/make_manifest.py ", out,
        " --meta ", file.path(out, "sections.csv"), " --label-key ", LABEL_COL,
        " --xcol spatial_x --ycol spatial_y --gene-col gene_name")
