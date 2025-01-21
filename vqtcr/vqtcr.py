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
    def __init__(self, params, adata, 
                 train_mode='semi_sup', 
                 metadata=['clonotype'], 
                 sample_mode='clonetype',
                 conditional=None, 
                 labels='binding_name'):
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
                                                            train_mode=train_mode, 
                                                            sample_mode=sample_mode,
                                                            metadata = metadata, 
                                                            labels = labels, 
                                                            conditional = conditional)
        
        # init training
        self.model = VQTCRModel(self.params, self.tcr_params, 
                                self.rna_params, self.cvq_params, 
                                self.cls_params, self.aa_to_id).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.params['lr'])
        
    def train(self):
        
        losses = []
        for ep in tq.tqdm(range(1, self.params['epoch']+1)):
            self.model.train()
            data = self.train_loader
            running_loss = 0
            for rna, tcr, _, _, labels in data:
                self.optimizer.zero_grad()
                loss_rna, loss_tcr, loss_vq, loss_cls = self.model(rna.to(self.device), 
                                                                   tcr.to(self.device),
                                                                   labels.to(self.device),
                                                                   ep, train_mode='semi-sup')
                loss = loss_rna + loss_tcr + loss_vq + loss_cls
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()
                running_loss += loss.item()
            losses.append(running_loss)
            
            self.model.eval()
            with torch.no_grad():
                loss_rna, loss_tcr, loss_vq, loss_cls = self.model(rna.to(self.device), 
                                                                   tcr.to(self.device),
                                                                   labels.to(self.device),
                                                                   ep, train_mode='semi-sup')
                print(loss_rna.item(), loss_tcr.item(), loss_vq.item(), loss_cls.item())
                # todo predict metrics
                
        
        torch.cuda.empty_cache()
        x = range(1, len(losses) + 1)
        plt.plot(x, np.log(losses))
        plt.show()
    
    def eval_metric(self):
        pass
        
    
    def get_emb(self):
        pass
    
        