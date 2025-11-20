# project: Mr.TCR
# description: multi-modal analysis integrating TCR and GEX data
# version: 1.0.0, release: 1.0.0
# maintainer: Lei Zhang 
# email: tju_zhanglei@outlook.com
# date: 2025-09-24 (1020 pred)

import torch
import torch.nn.utils as utils
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, classification_report

import tqdm.notebook as tq
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import faiss

from .predata import *
from .model import *
from .utils import *
from .metric import *
from .plotting import *


class vqTCR:
    def __init__(self, params, adata, labels=None):
        
        self.params = params
        self.tcr_params = params['tcr']
        self.gex_params = params['gex']
        self.fus_params = params['fus']
        self.pty_params = params['pty']
        self.cls_params = params['pred'] if labels else None
        self.labels = labels

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # data
        self.adata = adata
        self.tcr_params['max_tcr_length'] = adata.obsm['alpha_seq'].shape[1]
        self.tcr_params['num_seq_labels'] = len(TCR_VOCAB)
        self.gex_params['x_dim'] = adata.X.shape[1]
        
        # model
        self.model = vqTCRModel(self.params, self.tcr_params, self.gex_params, 
                                self.fus_params, self.pty_params).to(self.device)
        
        # optimizer
        self.optimizer = Adam(self.model.parameters(), 
                              lr=self.params['lr'], 
                              weight_decay=self.params['weight_decay'])
        
        # reports
        self.report = {}
        self.report['logo'] = 'LoClonotype'
        self.report['dataset'] = params['dataset']
    
    # -----------------pretraining----------------- #
    def pretrain(self, col_name='', set_name=[], sample_mode='clonotype', 
                 shuffle=False, labels=None, visual=True):
        warmloader = get_dataloader(adata=self.adata.copy(), 
                                     batch_size=self.params['batch_size'],
                                     col_name=col_name,
                                     set_name=set_name,
                                     sample_mode=sample_mode,
                                     shuffle=shuffle, 
                                     labels=labels)
        # warmup codebook
        train_losses, running_losses = [], []
        for ep in tq.tqdm(range(1, 10+1), 'warmup codebook'):
            self.model.train()
            running_loss = 0.0
            for gex, tcr, _ in warmloader:
                self.optimizer.zero_grad()
                gex = gex.to(self.device)
                tcr = tcr.to(self.device)
                packdata = self.model.warmup_code(gex, tcr)
                loss_gex, loss_tcr, loss_reg = packdata[0]
                loss = loss_gex + loss_tcr + loss_reg
                print(loss_gex.item(), loss_tcr.item(), loss_reg.item())
                loss.backward()
                utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()
                running_loss += loss.item()
                running_losses.append(loss.item())
            train_losses.append(running_loss/len(warmloader))
        torch.cuda.empty_cache()
        
        if visual:
            self._plot_losses(train_losses, running_losses)
        
        # initialize codebook
        self._init_codebook(col_name, set_name, sample_mode)
    
        # pretrain representation model
        trainloader = get_dataloader(adata=self.adata.copy(), 
                                     batch_size=self.params['batch_size'],
                                     col_name=col_name,
                                     set_name=set_name,
                                     sample_mode=sample_mode,
                                     shuffle=shuffle, 
                                     labels=labels)
        train_losses, running_losses = [], []
        for ep in tq.tqdm(range(1, self.params['epoch']+1), 'Pre-Training'):
            self.model.train()
            running_loss = 0.0
            for gex, tcr, _ in trainloader:
                self.optimizer.zero_grad()
                gex = gex.to(self.device)
                tcr = tcr.to(self.device)
                packdata = self.model(gex, tcr)
                loss_gex, loss_tcr, loss_reg, info = packdata[0]
                loss_commit = info['commitment_loss']
                loss_all = loss_gex + loss_tcr + loss_reg + loss_commit
                print(loss_gex.item(), loss_tcr.item(), loss_reg.item(), 
                      loss_commit.item())
                loss_all.backward()
                utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()
                running_loss += loss_all.item()
                running_losses.append(loss_all.item())
            train_losses.append(running_loss/len(trainloader))  
        torch.cuda.empty_cache()
        
        if visual:
            self._plot_losses(train_losses, running_losses)
    
    def _plot_losses(self, train_losses, running_losses):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
        ax1.plot(train_losses, 'b-', label='Epoch Avg Loss')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Epoch Loss'); ax1.legend()
        ax2.plot(running_losses, 'r-', alpha=0.3, label='Batch Loss')
        ax2.set_xlabel('Batch'); ax2.set_ylabel('Batch Loss'); ax2.legend()
        plt.tight_layout(); plt.show()
    
    def _init_codebook(self, col_name='', set_name=[], sample_mode='clonotype'):
        h_tcells = []
        dataloader = get_dataloader(adata=self.adata, 
                                    batch_size=self.params['batch_size'], 
                                    sample_mode=sample_mode, 
                                    shuffle=False, 
                                    labels=None,
                                    col_name=col_name,  
                                    set_name= set_name)
        with torch.no_grad():
            self.model.eval()
            for gex, tcr, _ in dataloader:
                gex = gex.to(self.device)
                tcr = tcr.to(self.device)
                packdata = self.model.warmup_code(gex, tcr)
                h_tcell = packdata[1][0]
                h_tcells.append(h_tcell.cpu().numpy())
        h_tcells = np.concatenate(h_tcells, axis=0)
        
        M = self.pty_params['num_subspace']
        K = self.pty_params['codewords_per_space']
        d = self.pty_params['dim_latent'] // self.pty_params['num_subspace']
        
        codebooks = []
        for m in range(M):
            h_subspace = h_tcells[:, m * d: (m + 1) * d]

            kmeans = faiss.Kmeans(d, K, niter=20, verbose=True)
            kmeans.train(h_subspace)
            centroids = torch.tensor(kmeans.centroids).to(self.device)  # (K, D_subspace)
            codebooks.append(centroids)

        codebooks = torch.cat(codebooks, dim=0).view(M, K, d)
        
        self.model.Tcell_pty._init_codebook(init_code=codebooks, device=self.device)
    
    # -----------------LOGO prediction----------------- #
    def predict_train(self, trainloader, validloader, metric='weighted'):
        if self.labels is not None:
            self.pred_head = LabelpMHC(self.cls_params).to(self.device)
            self.loss_func_cls = nn.CrossEntropyLoss()
        self.pred_optimizer = AdamW(self.pred_head.parameters(), 
                              lr=0.0001, 
                              weight_decay=0.0001)
        self.scheduler = ReduceLROnPlateau(self.pred_optimizer, mode=self.params['direction'], 
                                           patience=5, factor=0.5, verbose=True)
        
        self.model.eval()
        self.best_score = 0
        self.patience_counter = 0
        # train prediction head
        for ep in tq.tqdm(range(1, 21), 'Training prediction head'):
            self.pred_head.train()
            for gex, tcr, labels in trainloader:
                self.pred_optimizer.zero_grad()
                gex = gex.to(self.device)
                tcr = tcr.to(self.device)
                labels = labels.to(self.device)
                h_tcr = self.model(gex, tcr)[1][0]
                pred_outputs = self.pred_head(h_tcr)
                loss = self.loss_func_cls(pred_outputs, labels)
                loss.backward()
                self.pred_optimizer.step()
            
            # validation
            self.pred_head.eval()
            val_preds, val_labels, val_probs = self._evaluate(validloader)
            val_metrics = self._calculate_metrics(val_labels, val_probs, val_preds)
            
            print(f'Epoch {ep}/{20}:')
            print(f'  Train Loss: {loss:.4f}')
            print(f'Val F1: {val_metrics[f"f1_{metric}"]:.4f}, '
                  f'Val AUC-ROC: {val_metrics[f"auc_roc_{metric}"]:.4f}, '
                  f'Val AUC-PR: {val_metrics[f"auc_pr_{metric}"]:.4f}')
            
            self.scheduler.step(val_metrics[f'auc_roc_{metric}'])
            
            if val_metrics[f'auc_roc_{metric}'] > self.best_score:
                self.best_score = val_metrics[f'auc_roc_{metric}']
                self.patience_counter = 0
                torch.save({'pred_head': self.pred_head.state_dict(),
                            'config': self.params}, 'best_model.pth')
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= 10:
                print(f"early stop at {ep+1} epoch")
                break

        self.pred_head.load_state_dict(torch.load('best_model.pth')['pred_head'])
        return self.best_score
    
    def predict_test(self, testloader, metric='weighted'):
        self.model.eval()
        self.pred_head.eval()
        test_preds, test_labels, test_probs = self._evaluate(testloader)
        test_metrics = self._calculate_metrics(test_labels, test_probs, test_preds)
        print(f"Test results - f1: {test_metrics[f'f1_{metric}']:.4f}, "
                  f"AUC-ROC: {test_metrics[f'auc_roc_{metric}']:.4f}, "
                  f"AUC-PR: {test_metrics[f'auc_pr_{metric}']:.4f}")
        return test_metrics
    
    def _evaluate(self, dataloader):
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for gex, tcr, labels in dataloader:
                gex = gex.to(self.device)
                tcr = tcr.to(self.device)
                labels = labels.to(self.device)
                h_tcr = self.model(gex, tcr)[1][0]
                pred_outputs = self.pred_head(h_tcr)
                
                probs = torch.softmax(pred_outputs, dim=1)
                preds = torch.argmax(pred_outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        return np.array(all_preds), np.array(all_labels), np.array(all_probs)

    def _calculate_metrics(self, labels, probs, preds, metric='weighted'):
        """
        Calculate classification metrics robustly for multi-class or single-class cases.
        """
        n_classes = probs.shape[1]
        metrics = {}

        # Accuracy
        metrics['accuracy'] = (preds == labels).mean()

        # F1-score
        metrics[f'f1_{metric}'] = f1_score(labels, preds, average=metric, zero_division=0)

        # AUROC
        try:
            # only compute AUROC if there are at least two unique classes in labels
            if len(np.unique(labels)) > 1:
                y_true_onehot = np.eye(n_classes)[labels]
                metrics[f'auc_roc_{metric}'] = roc_auc_score(
                    y_true_onehot, probs, average=metric, multi_class='ovr'
                )
            else:
                metrics[f'auc_roc_{metric}'] = np.nan
        except ValueError:
            metrics[f'auc_roc_{metric}'] = np.nan

        # AUPRC
        ap_scores = []
        for i in range(n_classes):
            y_true_binary = (labels == i).astype(int)
            if y_true_binary.sum() > 0:
                try:
                    ap = average_precision_score(y_true_binary, probs[:, i])
                    ap_scores.append(ap)
                except ValueError:
                    ap_scores.append(np.nan)
            else:
                ap_scores.append(np.nan)
        metrics[f'auc_pr_{metric}'] = np.nanmean(ap_scores)

        metrics['n_classes'] = n_classes
        metrics['n_valid_classes'] = len(np.unique(labels))

        return metrics
        
    # -----------------benchmark----------------- #
    def get_latent(self, adata, col_name='', set_name=[], sample_mode=None, 
                 shuffle=False, gene_analys=False):
        h_tcells, h_tcrs, h_gexs = [], [], []
        base_tcells, bias_gexs, gammas, betas = [], [], [], []
        indices = []
        dataloader = get_dataloader(adata, 
                                    batch_size=self.params['batch_size'], 
                                    sample_mode=sample_mode, 
                                    shuffle=shuffle, 
                                    labels=None,
                                    col_name=col_name,  
                                    set_name=set_name)
        with torch.no_grad():
            self.model.eval()
            for gex, tcr, _ in dataloader:
                gex = gex.to(self.device)
                tcr = tcr.to(self.device)
                
                packdata = self.model(gex, tcr)
                h_tcell, h_tcr, h_gex, base_tcell, bias_gex, gamma, beta, idx = packdata[1]
                h_tcells.append(h_tcell.cpu().numpy())
                h_tcrs.append(h_tcr.cpu().numpy())
                h_gexs.append(h_gex.cpu().numpy())
                if gene_analys:
                    base_tcells.append(base_tcell.cpu().numpy())
                    bias_gexs.append(bias_gex.cpu().numpy())
                    gammas.append(gamma.cpu().numpy())
                    betas.append(beta.cpu().numpy())
                    indices.append(idx.cpu().numpy())
                
        h_tcells = np.concatenate(h_tcells, axis=0)    
        h_tcrs = np.concatenate(h_tcrs, axis=0)
        h_gexs = np.concatenate(h_gexs, axis=0)
        
        adata.obsm['htcell'] = h_tcells
        adata.obsm['htcr'] = h_tcrs
        adata.obsm['hgex'] = h_gexs
        
        if gene_analys:
            base_tcells = np.concatenate(base_tcells, axis=0)    
            bias_gexs = np.concatenate(bias_gexs, axis=0)
            gammas = np.concatenate(gammas, axis=0)
            betas = np.concatenate(betas, axis=0)
            indices = np.concatenate(indices, axis=0)
            adata.obsm['base_tcells'] = base_tcells
            adata.obsm['bias_gexs'] = bias_gexs
            adata.obsm['gammas'] = gammas
            adata.obsm['betas'] = betas
            adata.obsm['indices'] = indices
        
        return adata
        
    # only select the first metric to control the training process           
    def compute_metrics(self, metrics, eval_set='val'):
        record = {}
        adata = self.get_latent(self.adata)
        adata_train = adata[adata.obs['set']=='train']
        adata_valid = adata[adata.obs['set']==eval_set]
        if 'knn_prediction' in metrics:
            knn_scores = get_knn_cls(adata_train.obsm['h'], adata_valid.obsm['h'],
                                     adata_train.obs['binding_name'], 
                                     adata_valid.obs['binding_name'])
            record['knn_prediction'] = knn_scores['weighted avg']['f1-score']
        elif 'sup_pred' in metrics:
            pred_labels = np.argmax(adata_valid.obsm['pred_pmhc'], axis=1)
            y_true = adata_valid.obs['binding_name'].map(adata_valid.uns['specificity_to_label'])
            sup_scores = classification_report(y_true, pred_labels, output_dict=True)
            record['sup_pred'] = sup_scores['weighted avg']['f1-score']
        elif 'sup_count' in metrics:
            mlse = MSLE()
            pred_avit = torch.FloatTensor(adata_valid.obsm['pred_pmhc'])
            real_avit = torch.FloatTensor(adata_valid.obsm['binding_counts'])
            record['sup_count'] = mlse(pred_avit, real_avit).cpu().numpy()
        else:
            raise ValueError('please given a metric to evaluate.')
        return record

