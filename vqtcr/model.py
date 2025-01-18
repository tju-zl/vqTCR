import torch
import torch.nn as nn
from .module import *


class VQTCRModel(nn.Module):
    def __init__(self, params, tcr_params, rna_params,
                 cvq_params, cls_params):
        super().__init__()
        # params setting
        l_dim = params['dim_latent']
        x_dim = rna_params['dim_latent']
        t_dim = tcr_params['dim_emb']
        n_codebook = cvq_params['num_codebook']
        commitment = cvq_params['commitment']
        decay = cvq_params['decay']
        num_seq_labels = tcr_params['num_seq_labels']
        
        # TCR module
        self.alpha_encoder = TCREncoder(tcr_params, t_dim, num_seq_labels)
        self.alpha_decoder = TCRDecoder(tcr_params, t_dim, num_seq_labels)
        self.beta_encoder = TCREncoder(tcr_params, t_dim, num_seq_labels)
        self.beta_decoder = TCRDecoder(tcr_params, t_dim, num_seq_labels)
        
        # RNA module
        self.rna_encoder = RNAEncoder(x_dim, l_dim)
        self.rna_decoder = RNADecoder(x_dim, l_dim)
        
        # Bilinear Attention module for TCR chains
        self.bilin = nn.Bilinear(l_dim, l_dim, 1)
        self.act = nn.Sigmoid()
        
        # Conditional VQ-VAE module
        self.vq = VQEMA(l_dim, n_codebook, commitment, decay)
        
        # Semi_sup module
        
        # loss func
    
    def forward(self, rna, tcr, ep, train_mode='unsup'):
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
        z, vq_loss = self.vq(h_t, ep)
        
        # conditional add
        h = z + c_rna
        
        # sup
        if train_mode == 'semi-sup':
            pass
        
        # decoder
        rec_rna = self.rna_decoder(h)
        rec_alpha = self.alpha_decoder(h, alpha_seq)
        rec_beta = self.beta_decoder(h, beta_seq)
        rec_tcr = torch.cat([rec_alpha, rec_beta], dim=1)
        
        return rec_rna, rec_tcr, vq_loss
    
    def calc_loss(self):
        pass
