import torch
import torch.nn as nn
from .module import *


class VQTCRModel(nn.Module):
    def __init__(self, params, tcr_params, rna_params,
                 cvq_params, cls_params, aa_to_id):
        super().__init__()
        # params setting
        l_dim = params['dim_latent']
        x_dim = rna_params['hvgs']
        n_codebook = cvq_params['num_codebook']
        commitment = cvq_params['commitment']
        decay = cvq_params['decay']
        num_seq_labels = tcr_params['num_seq_labels']
        
        # TCR module
        self.alpha_encoder = TCREncoder(tcr_params, l_dim, num_seq_labels)
        self.alpha_decoder = TCRDecoder(tcr_params, l_dim, num_seq_labels)
        self.beta_encoder = TCREncoder(tcr_params, l_dim, num_seq_labels)
        self.beta_decoder = TCRDecoder(tcr_params, l_dim, num_seq_labels)
        
        # RNA module
        self.rna_encoder = RNAEncoder(x_dim, l_dim)
        self.rna_decoder = RNADecoder(x_dim, l_dim)
        
        # Bilinear Attention module for TCR chains
        self.bilin = nn.Bilinear(l_dim, l_dim, 1)
        self.act = nn.Sigmoid()
        
        # Conditional VQ-VAE module
        self.vq = VQEMA(l_dim, n_codebook, commitment, decay)
        
        # Semi-supervised module for pMHC prediction
        self.label_cls = LabelCLS(cls_params)
        
        # loss func
        self.loss_func_rna = nn.MSELoss()
        self.loss_func_tcr = nn.CrossEntropyLoss(ignore_index=aa_to_id['_'])
        self.loss_func_cls = nn.CrossEntropyLoss()
    
    def forward(self, rna, tcr, labels, ep, train_mode='unsup'):
        alpha_seq = tcr[:, :tcr.shape[1]//2]
        beta_seq = tcr[:, tcr.shape[1]//2:]
        
        # encode
        c_rna = self.rna_encoder(rna)
        h_alpha = self.alpha_encoder(alpha_seq)
        h_beta = self.beta_encoder(beta_seq)
        
        # fusion TCR
        tcr_att = self.act(self.bilin(h_alpha, h_beta))
        h_tcr = tcr_att * h_alpha + (1 - tcr_att) * h_beta
        
        # conditional add
        h_t = h_tcr + c_rna
        
        # vq searching
        z, loss_vq = self.vq(h_t, ep)
        
        # conditional add
        h = z + c_rna
        
        # decoder
        rec_rna = self.rna_decoder(h)
        rec_alpha = self.alpha_decoder(h, alpha_seq)
        rec_beta = self.beta_decoder(h, beta_seq)
        rec_tcr = torch.cat([rec_alpha, rec_beta], dim=1)
        
        # calc loss
        loss_rna, loss_tcr = self.calc_loss(rna, rec_rna, tcr, rec_tcr)
        
        # supervised, constrain the discrete representation
        if train_mode == 'semi-sup':
            pred_labels = self.label_cls(z)
            loss_cls = self.loss_func_cls(pred_labels, labels)
            return loss_rna, loss_tcr, loss_vq, loss_cls
        else:
            return loss_rna, loss_tcr, loss_vq
    
    def calc_loss(self, rna, rec_rna, tcr, rec_tcr):
        loss_rna = self.loss_func_rna(rec_rna, rna)
        mask = torch.ones_like(tcr).bool()
        mask[:, [0, mask.shape[1] // 2]] = False
        loss_tcr = self.loss_func_tcr(rec_tcr.flatten(end_dim=1), tcr[mask].flatten())
        return loss_rna, loss_tcr
