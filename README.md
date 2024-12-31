# RMGE
Robust MetaGene Extractor

Implementation Description

This repository provides a Python implementation for transferring annotations from single-nucleus ATAC-seq (snATAC-seq) or single-nucleus RNA-seq (snRNA-seq) data to spatial transcriptomics data (Stereo-seq). The primary goal is to leverage chromatin accessibility or RNA expression profiles to annotate spatial transcriptomics data with high accuracy and biological relevance.
The method employs an autoencoder-based model combined with adversarial learning to align features across modalities, enabling annotation transfer of cell types or functional labels from snATAC-seq or snRNA-seq data to the spatial transcriptomics domain. This pipeline ensures robust and biologically meaningful annotations even in sparse spatial data.

Core Components

Spatial Transcriptomics Data (Stereo-seq): Provides spatially resolved gene expression profiles.
snATAC-seq or snRNA-seq: Supplies single-cell level chromatin accessibility (snATAC-seq) or RNA expression (snRNA-seq) data, which are used as the reference modalities for annotation transfer.

Key Steps

Preprocessing: Both spatial transcriptomics data and reference modalities (snATAC-seq/snRNA-seq) are trimmed to a shared gene feature space, normalized, and transformed for alignment.
Annotation Transfer: Cell types or functional labels from the reference modality are transferred to spatial cells using cross-modal mapping and adversarial learning.
Mapping: Latent feature spaces from both modalities are aligned, and spatial transcriptomics data is mapped to the reference feature space to facilitate annotation transfer.

Tutorial

The flow.ipynb notebook provides a step-by-step guide to performing annotation transfer and mapping. It demonstrates:

Datasets

All datasets used for this annotation transfer are publicly available and can be downloaded from:
Stereo-seq Data: Zenodo Link
snATAC-seq Data: Zenodo Link
snRNA-seq Data: Zenodo Link
These datasets include both spatial transcriptomics and reference data for facilitating cross-modal annotation transfer.

License

This repository is licensed under the MIT License. See the LICENSE file for more details.
