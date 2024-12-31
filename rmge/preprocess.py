#!/usr/bin/env python3
import scanpy as sc
import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def generate_noisy_data(st_data, sn_data, device):
    """
    Generate noisy data and plot the density of log_rg values.
    
    Parameters:
    - st_data: Input data containing the 'counts' layer (AnnData object)
    - sn_data: Input data containing the 'counts' layer (AnnData object)
    - device: The device for PyTorch (either 'cpu' or 'cuda')
    
    Returns:
    - noise_sn_data: Generated noisy data tensor
    """
    
    # Calculate log_rg
    p_mean = np.mean(st_data.layers['counts']) / np.mean(sn_data.layers['counts'])
    pg = np.array(st_data.layers['counts'].mean(axis=0) / sn_data.layers['counts'].mean(axis=0)).squeeze()
    rg = pg / pg.mean()
    kde = gaussian_kde(np.log(rg))
    x_range = np.linspace(min(np.log(rg)), max(np.log(rg)), 1000)

    # Calculate cell counts and percentiles
    st_cell_counts = st_data.layers['counts'].sum(axis=1)
    cell_counts = sn_data.layers['counts'].sum(axis=1).squeeze()
    st_percentiles = np.percentile(st_cell_counts, np.arange(0, 100.0, 100 / len(cell_counts)))

    # Adjust the cell counts based on percentiles
    poior_cell_counts = cell_counts.copy()
    poior_cell_counts[np.argsort(cell_counts)] = st_percentiles
    pn = poior_cell_counts / cell_counts

    # Calculate p matrix, which is used for noise generation
    p = np.expand_dims(pn, axis=1) * np.expand_dims(rg, axis=0)
    p[p > 1] = 1

    # Generate noisy data using binomial distribution
    cell_tensor = torch.tensor(sn_data.layers['counts']).to(device)
    noise_sn_data = torch.binomial(cell_tensor, torch.tensor(p).to(device))

    # Normalize the noisy data
    noise_sn_data = noise_sn_data / noise_sn_data.sum(axis=1, keepdims=True) * 10000
    noise_sn_data = torch.log(noise_sn_data + 1)  # Apply log transformation to smooth the data

    return noise_sn_data
    
def remove_zero_columns(data):
    """
    Removes columns with zero sums from the AnnData object.
    """
    column_sums = np.array(data.X.sum(axis=0)).flatten()
    non_zero_columns = column_sums != 0
    data = data[:, non_zero_columns]
    return data
def preprocess_data(st_data, sn_data, target_sum=1e4, n_top_genes=3000):
    """
    Preprocess the spatial transcriptomics (st_data) and single-nucleus RNA-seq (sn_data) datasets.
    - Removes zero-sum columns
    - Filters common genes
    - Normalizes and log-transforms the data
    - Identifies highly variable genes for sn_data
    
    Parameters:
    - st_data: AnnData object containing spatial transcriptomics data
    - sn_data: AnnData object containing single-nucleus RNA-seq data
    - target_sum: The target sum for normalization (default is 10,000)
    - n_top_genes: The number of top variable genes to select from sn_data (default is 3000)
    
    Returns:
    - st_data: Processed AnnData object for spatial transcriptomics
    - sn_data: Processed AnnData object for single-nucleus RNA-seq
    """
    st_data = remove_zero_columns(st_data)
    sn_data = remove_zero_columns(sn_data)

    st_genes = st_data.var_names
    sn_genes = sn_data.var_names

    # Find common genes between st_data and sn_data
    common_genes = st_genes.intersection(sn_genes)

    # Subset both datasets to include only common genes
    st_data = st_data[:, common_genes].copy()
    sn_data = sn_data[:, common_genes].copy()

    # Normalize the single-nucleus (sn) data to a target sum of 10,000 reads per cell
    sc.pp.normalize_total(sn_data, target_sum=target_sum)

    # Apply log transformation to the sn data
    sc.pp.log1p(sn_data)

    # Identify the top `n_top_genes` highly variable genes in sn_data
    sc.pp.highly_variable_genes(sn_data, n_top_genes=n_top_genes)

    # Subset the sn_data to keep only the highly variable genes
    sn_data = sn_data[:, sn_data.var['highly_variable']]

    # Subset the st_data to include only the genes present in sn_data (common highly variable genes)
    st_data = st_data[:, sn_data.var_names]

    # Normalize the spatial transcriptomics (st) data to a target sum of 10,000 reads per cell
    sc.pp.normalize_total(st_data, target_sum=target_sum)

    # Apply log transformation to the st data
    sc.pp.log1p(st_data)

    return st_data, sn_data