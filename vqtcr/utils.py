# -*- coding: utf-8 -*-
# TreeVQ-EMA with stable splitting
# Author: Lei Zhang, 2110610@tongji.edu.cn
# License: MIT
# Date: 2025-09-10

import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# init codebook
def faiss_kmeans(x_np, k, niter=20, seed=2025, use_gpu=False):
    """
    x_np: (N, D) float32 C-contiguous array
    k: number of clusters
    niter: number of iterations
    return: centroids (k, D), assignments (N,)
    """
    try:
        import faiss
        assert x_np.dtype == 'float32' and x_np.flags['C_CONTIGUOUS']
        d = x_np.shape[1]
        kmeans = faiss.Kmeans(d, k, niter=niter, verbose=False, seed=seed)
        if use_gpu and faiss.get_num_gpus() > 0:
            kmeans.train(x_np, faiss.GpuMultipleClonerOptions())
        else:
            kmeans.train(x_np)
        D, I = kmeans.index.search(x_np, 1)
        return kmeans.centroids, I.ravel()
    except Exception:
        return None, None


@torch.no_grad()
def torch_kmeans(x, k, niter=20, iters=0, esp=1e-9):
    """
    x: [N, D], return: [k, D]
    """
    N, D = x.shape
    device = x.device
    i0 = torch.randint(0, N, (1,), device=device)
    cents = [x[i0]]
    for _ in range(1, k):
        C = torch.cat(cents, 0)
        d2 = torch.cdist(x, C).pow(2).min(1).values
        p = (d2 / (d2.sum() + esp)).clamp_min(esp)
        nxt = torch.multinomial(p, 1)
        cents.append(x[nxt])
    C = torch.cat(cents, 0)
    for _ in range(iters):
        a = torch.cdist(x, C).argmin(1)
        one = F.one_hot(a, k).float()
        cnt = one.sum(0).clamp_min(esp)
        sm = one.T @ x
        C = sm / cnt[:, None]
    return C


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def aa_tokenizer(adata, esm_type, read_col, label_col=None, length_col=None, mask_col=None, pad=32, add_special_tokens=True):
    """
    use the esm2 vocab to tokenize the amino acid sequence of TCRs
    esm_type = "facebook/esm2_t6_8M_UR50D" as default
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(esm_type)

    if add_special_tokens:
        pad = pad + 2
    
    sequences = adata.obs[read_col].tolist()
    encoded_sequences = tokenizer(sequences, 
                                  padding='max_length',
                                  max_length=pad, 
                                  return_tensors=None, 
                                  add_special_tokens=add_special_tokens)
    adata.obsm[label_col] = np.stack(encoded_sequences["input_ids"])
    adata.obsm[mask_col] = np.stack(encoded_sequences["attention_mask"])
    adata.obs[length_col] = [len(x) for x in encoded_sequences["input_ids"]]

