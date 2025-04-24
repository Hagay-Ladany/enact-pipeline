"""Class for defining methods to package pipeline outputs into AnnData objects
"""

import os
import pandas as pd
import anndata
import scanpy as sc
import seaborn as sns
import numpy as np

import celltypist
from celltypist import models

from .pipeline import ENACT


class CellTypistPipeline(ENACT):
    """Class for running CellAssign algorithm"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run_cell_typist(self):
        """Runs CellTypist"""
        bin_assign_results = self.merge_files_sparse(self.bin_assign_dir)
        cell_lookup_df = self.merge_files(self.cell_ix_lookup_dir, save=False)

        spatial_cols = ["cell_x", "cell_y"]
        stat_columns = ["num_shared_bins", "num_unique_bins", "num_transcripts"]
        cell_lookup_df.loc[:, "id"] = cell_lookup_df["id"].astype(str)
        cell_lookup_df = cell_lookup_df.set_index("id")
        cell_lookup_df["num_transcripts"] = cell_lookup_df["num_transcripts"].fillna(0)

        bin_assign_result_sparse, gene_columns = bin_assign_results
        adata = anndata.AnnData(X=bin_assign_result_sparse, obs=cell_lookup_df.copy())
        adata.var_names = gene_columns

        adata.obsm["spatial"] = cell_lookup_df[spatial_cols].astype(int)
        adata.obsm["stats"] = cell_lookup_df[stat_columns].astype(int)

        lib_size = adata.X.sum(1)
        adata.obs["size_factor"] = lib_size / np.mean(lib_size)
        adata.obs["lib_size"] = lib_size

        # Normalize data: scale to 10,000 counts per cell and log-transform.
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # Ensure the model filename ends with ".pkl"
        if ".pkl" not in self.cell_typist_model:
            self.cell_typist_model = self.cell_typist_model + ".pkl"

        # If the model file exists locally, load it; otherwise, download it.
        if os.path.isfile(self.cell_typist_model):
            try:
                model_obj = models.Model.load(self.cell_typist_model)
                self.logger.info(f"✅ Loaded local CellTypist model from {self.cell_typist_model}")
            except Exception as e:
                self.logger.error(f"🛑 Failed to load local model from {self.cell_typist_model}: {e}")
                raise
        else:
            try:
                models.download_models(model=self.cell_typist_model)
                model_obj = models.Model.load(self.cell_typist_model)
                self.logger.info(f"✅ Downloaded and loaded model {self.cell_typist_model}")
            except Exception as e:
                self.logger.error(f"🛑 Failed to download model {self.cell_typist_model}: {e}")
                raise

        # Run annotation
        predictions = celltypist.annotate(adata, model=model_obj)
        adata = predictions.to_adata(insert_labels=True, insert_conf=True, insert_prob=True)

        adata.obs.rename(columns={"predicted_labels": "cell_type"}, inplace=True)
        adata.obs[adata.obsm["spatial"].columns] = adata.obsm["spatial"]
        adata.obs[adata.obsm["stats"].columns] = adata.obsm["stats"]
        adata.obs["chunk_name"] = cell_lookup_df["chunk_name"]
        results_df = adata.obs.drop(columns=adata.obs["cell_type"].unique().tolist())
        results_df.to_csv(os.path.join(self.cellannotation_results_dir, "merged_results.csv"))


if __name__ == "__main__":
    cell_typist = CellTypistPipeline(configs_path="config/configs.yaml")
    cell_typist.run_cell_typist()
