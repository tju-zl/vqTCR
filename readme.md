vqTCR: A Product-Quantized Representation Framework Linking TCR Specificity to Transcriptomic Cell States
===

## Overview of vqTCR
![](https://github.com/tju-zl/vqTCR/blob/main/overview.png)
We present vqTCR, a product vector-quantized variational autoencoder that learns a combinatorial latent discrete representation linking TCR specificity to functional transcriptional programs. By utilizing product quantization, vqTCR decomposes this T cell latent space into multiple subspaces, yielding discrete, interpretable T cell prototypes that capture the relationship between receptor specificity and functional states while maintaining representational flexibility. To further disentangle global and context-dependent expression, vqTCR incorporates a prototype baseline transcription module together with FiLM-modulated residual gene expressions, allowing the model to separate functional expression prototype from context-specific transcriptional transition that may vary across clonotypes, samples, or cell types.

## Getting started
1. Tutorials of preprocessing the adaptive immune data: refer to folder `pre_data`.
2. Benchmark results of vqTCR and other methods: refer to folder `benchmark`.
3. Case study of SCC patient results: refer to folder `case_study`.

## Dataset
Please refer to the supplementarial materials to get the download links (or in the notebook of tutorials).

Note: we prepared the preprocessed dataset in the folder `pre_data\prepared_data`.

## Key software dependencies
- Scanpy 1.9.8
- Pytorch 2.0.1
- Scirpy 0.22.3 (must after 0.13.1)
- Srublet 0.2.3

