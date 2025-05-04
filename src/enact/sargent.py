"""Class for running Sargent algorithm for cell type annotation
"""

import os
import pandas as pd
import anndata
import scanpy as sc
import numpy as np
import subprocess
import json
import shutil
import scipy.sparse
from datetime import datetime

from .pipeline import ENACT


class SargentPipeline(ENACT):
    """Class for running Sargent algorithm"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run_sargent(self):
        """Runs Sargent"""
        bin_assign_results = self.merge_files_sparse(self.bin_assign_dir)
        cell_lookup_df = self.merge_files(self.cell_ix_lookup_dir, save=False)

        spatial_cols = ["cell_x", "cell_y"]
        stat_columns = ["num_shared_bins", "num_unique_bins", "num_transcripts"]
        cell_lookup_df.loc[:, "id"] = cell_lookup_df["id"].astype(str)
        cell_lookup_df = cell_lookup_df.set_index("id")
        cell_lookup_df["num_transcripts"] = cell_lookup_df["num_transcripts"].fillna(0)

        bin_assign_result_sparse, gene_columns = bin_assign_results
        # Convert gene names to uppercase
        gene_columns = [gene.upper() for gene in gene_columns]
        adata = anndata.AnnData(X=bin_assign_result_sparse, obs=cell_lookup_df.copy())
        adata.var_names = gene_columns

        adata.obsm["spatial"] = cell_lookup_df[spatial_cols].astype(int)
        adata.obsm["stats"] = cell_lookup_df[stat_columns].astype(int)

        lib_size = adata.X.sum(1)
        adata.obs["size_factor"] = lib_size / np.mean(lib_size)
        adata.obs["lib_size"] = lib_size

        # Normalize data: scale to 10,000 counts per cell and log-transform.
        # Add small pseudocount to avoid zero counts
        adata.X = scipy.sparse.csr_matrix(adata.X.toarray() + 1)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # Convert cell markers to the format expected by Sargent
        cell_markers = {}
        for cell_type, markers in self.cell_markers.items():
            cell_markers[cell_type] = [marker.upper() for marker in markers]  # Sargent expects uppercase gene names

        # Run Sargent annotation
        try:
            # Create temporary directory for intermediate files
            temp_dir = os.path.join(self.sargent_results_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Save expression matrix to CSV
            expr_file = os.path.join(temp_dir, "expression_matrix.csv")
            pd.DataFrame(adata.X.toarray(), 
                        index=adata.obs_names, 
                        columns=adata.var_names).to_csv(expr_file)
            
            # Save cell markers to JSON
            markers_file = os.path.join(temp_dir, "cell_markers.json")
            with open(markers_file, 'w') as f:
                json.dump(cell_markers, f)
            
            # Create R script to run Sargent
            r_script = os.path.join(temp_dir, "run_sargent.R")
            cell_types_file = os.path.join(temp_dir, "cell_types.csv")
            confidence_file = os.path.join(temp_dir, "confidence_scores.csv")
            
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            r_script_content = f"""
# Load required packages
library(data.table)
library(Matrix)
library(Seurat)
library(ggplot2)
library(dplyr)
library(sargent)
library(yaml)

# Read data
cat("\\nStep 1: Reading data...\\n")
gex <- read.csv('{expr_file}', row.names=1)
gene.sets <- fromJSON('{markers_file}')
cells <- rownames(gex)
cat("Data read successfully. Dimensions:", dim(gex)[1], "genes x", dim(gex)[2], "cells\\n")

# Filter out cells with zero counts
cat("\\nStep 2: Filtering cells...\\n")
gex_filtered <- gex[,which(colSums(gex)!=0)]
cat("Cells filtered. Number of cells:", ncol(gex_filtered), "\\n")

# Create Seurat object and process
cat("\\nStep 3: Creating Seurat object and processing...\\n")
seurat_obj <- CreateSeuratObject(counts=gex_filtered) %>%
    NormalizeData(., normalization.method="LogNormalize", scale.factor=1e6, verbose=FALSE) %>%
    FindVariableFeatures(., selection.method="vst", nfeatures=2000, verbose=FALSE) %>%
    ScaleData(., do.scale=TRUE, do.center=TRUE, verbose=FALSE) %>%
    RunPCA(., features=VariableFeatures(.), verbose=FALSE) %>%
    FindNeighbors(., reduction="pca", dims=1:30, k.param=20, verbose=FALSE)

# Get adjacency matrix
cat("\\nStep 4: Getting adjacency matrix...\\n")
adjacent.mtx <- attr(seurat_obj, "graphs")[["RNA_nn"]]
cat("Adjacency matrix obtained\\n")

# Run Sargent annotation
cat("\\nStep 5: Running Sargent annotation...\\n")
srgnt <- sargentAnnotation(gex=gex_filtered, 
                          gene.sets=gene.sets,
                          adjacent.mtx=adjacent.mtx)
cat("Sargent annotation completed\\n")

# Generate and save visualizations
cat("\\nStep 6: Generating visualizations...\\n")
# Density plot
density_plot <- fetchDensityPlot(srgnt)
ggsave(plot=density_plot,
       filename=file.path('{self.sargent_results_dir}', paste0("DensityPlot.", "{current_date}", ".jpg")))

# Dot plot
dot_plot <- fetchDotPlot(srgnt, min.pct=0.1) +
    theme(axis.text.y=element_text(size=17),
          legend.title=element_text(size=14),
          legend.text=element_text(size=13)) +
    scale_x_discrete(limits = names(gene.sets))
ggsave(plot=dot_plot,
       filename=file.path('{self.sargent_results_dir}', paste0("DotPlot.", "{current_date}", ".jpg")),
       width=15)

# Save results
cat("\\nStep 7: Saving results...\\n")
write.csv(srgnt@celltype, '{cell_types_file}')
write.csv(srgnt@score, '{confidence_file}')
write.csv(srgnt@celltype_summary, 
          file.path('{self.sargent_results_dir}', paste0("sargent.rez.", "{current_date}", ".csv")))

# Save R object
save(srgnt, file=file.path('{self.sargent_results_dir}', paste0("sargent.", "{current_date}", ".Rdata")))
cat("Results saved\\n")
"""
            
            with open(r_script, 'w') as f:
                f.write(r_script_content)
            
            # Run R script
            subprocess.run(['Rscript', r_script], check=True)
            
            # Read results
            cell_types = pd.read_csv(cell_types_file, index_col=0)
            confidence = pd.read_csv(confidence_file, index_col=0)
            
            # Add results to AnnData
            adata.obs["cell_type"] = cell_types.values
            adata.obs["confidence"] = confidence.values
            
            # Save results
            results_df = adata.obs.drop(columns=adata.obs["cell_type"].unique().tolist())
            results_df.to_csv(os.path.join(self.cellannotation_results_dir, "merged_results.csv"))
            
            # Clean up temporary files
            shutil.rmtree(temp_dir)
            
            self.logger.info("✅ Successfully ran Sargent annotation")
            
        except Exception as e:
            self.logger.error(f"🛑 Failed to run Sargent annotation: {e}")
            # Clean up temporary files even if there's an error
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise


if __name__ == "__main__":
    sargent_pipeline = SargentPipeline(configs_path="config/configs.yaml")
    sargent_pipeline.run_sargent() 
