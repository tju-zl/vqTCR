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


class vqTCR:
    def __init__(self, params, adata, 
                 data_mode='eval',
                 model_mode='semi_sup',
                 metadata=['clonotype'], 
                 sample_mode='clonotype',
                 conditional=None, 
                 labels=None):
        self.params = params
        self.model_mode = model_mode
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
        if model_mode == 'semi_sup':
            assert labels is not None, 'Error: label must provide if use semi-sup model mode.'
        else:
            assert labels is None, 'Error: label must None if use self-sup model mode.'
        if data_mode == 'eval':
            self.train_loader = get_dataloader(adata,
                                            batch_size=params['batch_size'], 
                                            data_mode=data_mode, 
                                            sample_mode=sample_mode,
                                            metadata = metadata, 
                                            labels = labels,
                                            conditional = conditional)
        else:
            self.train_loader, self.eval_loader = get_dataloader(adata,
                                                                batch_size=params['batch_size'], 
                                                                data_mode=data_mode, 
                                                                sample_mode=sample_mode,
                                                                metadata = metadata, 
                                                                labels = labels,
                                                                conditional = conditional)
        
        # init training
        self.model = vqTCRModel(self.params, self.tcr_params, 
                                    self.rna_params, self.cvq_params, 
                                    self.cls_params, self.aa_to_id, model_mode).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.params['lr'])
    
    # training model
    def train(self):
        losses = []
        # e_losses = []
        for ep in tq.tqdm(range(1, self.params['epoch']+1)):
            self.model.train()
            data = self.train_loader
            running_loss = 0
            for rna, tcr, _, _, labels, _ in data:
                self.optimizer.zero_grad()
                packdata = self.model(rna.to(self.device), tcr.to(self.device), labels.to(self.device))
                if self.model_mode == 'semi_sup':
                    tcr_att, h_tcr, c_rna, z, h, loss_rna, loss_tcr, loss_vq, loss_cls = packdata
                    loss = loss_rna + loss_tcr + loss_vq + loss_cls
                else:
                    tcr_att, h_tcr, c_rna, z, h, loss_rna, loss_tcr, loss_vq = packdata
                    loss = loss_rna + loss_tcr + loss_vq
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()
                running_loss += loss.item()
            losses.append(running_loss)
            
            # !主要看损失大小以及克隆型预测指标
            # if ep % (self.params['epoch'] / 10) == 0:
            #     e_running_loss = 0
            #     self.model.eval()
            #     data = self.eval_loader
            #     for rna, tcr, _, _, labels, _ in data:
            #         with torch.no_grad():
            #             packdata = self.model(rna.to(self.device), tcr.to(self.device), labels.to(self.device))
            #             if self.model_mode == 'semi_sup':
            #                 tcr_att, h_tcr, c_rna, z, h, loss_rna, loss_tcr, loss_vq, loss_cls = packdata
            #                 loss = loss_rna + loss_tcr + loss_vq + loss_cls
            #             else:
            #                 tcr_att, h_tcr, c_rna, z, h, loss_rna, loss_tcr, loss_vq = packdata
            #                 loss = loss_rna + loss_tcr + loss_vq
                        
            #             # if self.model_mode == 'semi_sup':
            #             #     print(f'EP[%3d]: rna_loss=%.4f., tcr_loss=%.4f., vq_loss=%.4f., cls_loss=%.4f.' 
            #             #             % (ep, loss_rna.item(), loss_tcr.item(), loss_vq.item(), loss_cls.item()))
            #             # else:
            #             #     print(f'EP[%3d]: rna_loss=%.4f., tcr_loss=%.4f., vq_loss=%.4f.' 
            #                         # % (ep, loss_rna.item(), loss_tcr.item(), loss_vq.item()))
            #             e_running_loss += loss.item()
            #     print(e_running_loss)
            #     e_losses.append(e_running_loss)
                    
        
        torch.cuda.empty_cache()
        x = range(1, len(losses) + 1)
        plt.plot(x, losses)
        plt.show()
        # x = range(1, len(e_losses) + 1)
        # plt.plot(x, e_losses)
        # plt.show()
    
    # evaluating model
    def eval_metric(self, adata, 
                    data_mode='eval',
                    metadata=['clonotype'], 
                    sample_mode='clonotype',
                    conditional=None, 
                    labels=None):
        tcr_atts = []
        # ct_atts = []
        # h_atts = []
        h_tcrs = []
        c_rnas = []
        zs = []
        hs = []
        self.model.eval()
        if data_mode == 'eval':
            dataloader = get_dataloader(adata, 
                                        batch_size=self.params['batch_size'], 
                                        data_mode=data_mode, 
                                        sample_mode=sample_mode,
                                        metadata = metadata, 
                                        labels = labels,
                                        conditional = conditional)
        else:
            dataloader, _ = get_dataloader(adata, 
                                        batch_size=self.params['batch_size'], 
                                        data_mode=data_mode, 
                                        sample_mode=sample_mode,
                                        metadata = metadata, 
                                        labels = labels,
                                        conditional = conditional)
        
        for rna, tcr, _, _, labels, _ in dataloader:
            with torch.no_grad():
                packdata = self.model(rna.to(self.device), tcr.to(self.device), labels.to(self.device))
                if self.model_mode == 'semi_sup':
                    tcr_att, h_tcr, c_rna, z, h, loss_rna, loss_tcr, loss_vq, loss_cls = packdata
                else:
                    tcr_att, h_tcr, c_rna, z, h, loss_rna, loss_tcr, loss_vq = packdata
                tcr_atts.append(tcr_att.cpu().numpy())
                # ct_atts.append(ct_att.cpu().numpy())
                # h_atts.append(h_att.cpu().numpy())
                h_tcrs.append(h_tcr.cpu().numpy())
                c_rnas.append(c_rna.cpu().numpy())
                zs.append(z.cpu().numpy())
                hs.append(h.cpu().numpy())
                if self.model_mode == 'semi_sup':
                    print(f'rna_loss=%.4f., tcr_loss=%.4f., vq_loss=%.4f., cls_loss=%.4f.' 
                                % (loss_rna.item(), loss_tcr.item(), loss_vq.item(), loss_cls.item()))
                else:
                        print(f'rna_loss=%.4f., tcr_loss=%.4f., vq_loss=%.4f.' 
                                % (loss_rna.item(), loss_tcr.item(), loss_vq.item()))
        
        tcr_atts = np.concatenate(tcr_atts, axis=0)
        # ct_atts = np.concatenate(ct_atts, axis=0)
        # h_atts = np.concatenate(h_atts, axis=0)
        h_tcrs = np.concatenate(h_tcrs, axis=0)
        c_rnas = np.concatenate(c_rnas, axis=0)
        zs = np.concatenate(zs, axis=0)
        hs = np.concatenate(hs, axis=0)
        adata.obs['att_t'] = tcr_atts
        # adata.obs['att_c'] = ct_atts
        # adata.obs['att_h'] = h_atts
        adata.obsm['htcr'] = h_tcrs
        adata.obsm['crna'] = c_rnas
        adata.obsm['z'] = zs
        adata.obsm['h'] = hs
        
        return adata

        