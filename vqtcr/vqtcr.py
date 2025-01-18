import torch
import scanpy as sc
import anndata as ad
import tqdm.notebook as tq
import matplotlib.pyplot as plt
import numpy as np

from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.optim import Adam, SGD

from .data import *
from .model import *
from .utils import *


class VQTCR:
    def __init__(self, params, adata, train_mode, metadata, conditional, labels):
        self.params = params
        self.tcr_params = params['tcr']
        self.rna_params = params['rna']
        self.cvq_params = params['vq_layer']
        self.cls_params = params['cls']
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # data prepare
        self.aa_to_id = adata.uns['aa_to_id']
        self.tcr_params['max_tcr_length'] = adata.obsm['alpha_seq'].shape[1] # 26 tokens
        self.tcr_params['num_seq_labels'] = len(self.aa_to_id) # 24 vocab
        self.rna_params['x_dim'] = adata.X.shape[1]
        self.train_loader, self.val_loader = get_dataloader(adata, 
                                                            batch_size=512, 
                                                            train_mode='semi_sup', 
                                                            sample_mode='clonetype',
                                                            metadata = ['clonotype'], 
                                                            labels = None, 
                                                            conditional = None)
        
        # init training
        self.model = VQTCRModel(self.params, self.tcr_params, self.rna_params, 
                                self.cvq_params, self.cls_params).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.params['lr'])
        
    def train(self):
        self.model.train()
        losses = []
        for ep in tq.tqdm(range(1, self.params['epoch']+1)):
            pass
    
    def get_emb(self):
        pass
    
        