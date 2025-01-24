import torch
import torch.nn as nn
from .module import *


class vqTCRModel(nn.Module):
    def __init__(self, params, tcr_params, rna_params,
                 cvq_params, cls_params, aa_to_id, model_mode):
        super().__init__()
        # params setting
        l_dim = params['dim_latent']
        x_dim = rna_params['hvgs']
        n_codebook = cvq_params['num_codebook']
        commitment = cvq_params['commitment']
        decay = cvq_params['decay']
        num_seq_labels = tcr_params['num_seq_labels']
        self.model_mode = model_mode
        
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
        
        # Bilinear Attention module for TCR chains and RNA
        self.bilin1 = nn.Bilinear(l_dim, l_dim, 1)
        self.bilin2 = nn.Bilinear(l_dim, l_dim, 1)
           
        # Conditional VQ-VAE module
        self.vq = VQEMA(l_dim, n_codebook, commitment, decay)

        # loss func
        self.loss_func_rna = nn.MSELoss()
        self.loss_func_tcr = nn.CrossEntropyLoss(ignore_index=aa_to_id['_'])
        
        # Semi-supervised module for pMHC prediction
        if model_mode == 'semi_sup':
            self.label_cls = LabelCLS(cls_params)
            self.loss_func_cls = nn.CrossEntropyLoss()
    
    def forward(self, rna, tcr, labels=None):
        alpha_seq = tcr[:, :tcr.shape[1]//2]
        beta_seq = tcr[:, tcr.shape[1]//2:]
        
        # encode
        c_rna = self.rna_encoder(rna)
        h_alpha = self.alpha_encoder(alpha_seq)
        h_beta = self.beta_encoder(beta_seq)
        
        # fusion TCR
        h_alpha = h_alpha / torch.norm(h_alpha, dim=1, keepdim=True)
        h_beta = h_beta / torch.norm(h_beta, dim=1, keepdim=True)
        tcr_att = self.act(self.bilin(h_alpha, h_beta))
        h_tcr = tcr_att * h_alpha + (1 - tcr_att) * h_beta
        
        # conditional add
        # h_tcr = h_tcr / torch.norm(h_tcr, dim=1, keepdim=True)
        # c_rna = c_rna / torch.norm(c_rna, dim=1, keepdim=True)
        # ct_att = self.act(self.bilin1(h_tcr, c_rna))
        # c_tcr = ct_att * h_tcr + (1 - ct_att) * c_rna
        c_tcr = h_tcr +  c_rna
        
        # vq searching
        z, loss_vq = self.vq(c_tcr)
        
        # conditional add
        # z = z / torch.norm(z, dim=1, keepdim=True)
        # h_att = self.act(self.bilin2(z, c_rna))
        # # h_att = ct_att
        # h = h_att * z + (1 - h_att) * c_rna
        h = z + c_rna
        
        # decoder
        rec_rna = self.rna_decoder(h)
        rec_alpha = self.alpha_decoder(h, alpha_seq)
        rec_beta = self.beta_decoder(h, beta_seq)
        rec_tcr = torch.cat([rec_alpha, rec_beta], dim=1)
        
        # calc loss
        loss_rna, loss_tcr = self.calc_loss(rna, rec_rna, tcr, rec_tcr)
        
        # supervised, constrain the discrete representation
        if self.model_mode == 'semi_sup':
            pred_labels = self.label_cls(z)
            loss_cls = self.loss_func_cls(pred_labels, labels)
            return tcr_att, h_tcr, c_rna, z, h, loss_rna, loss_tcr, loss_vq, loss_cls
        else:
            return tcr_att, h_tcr, c_rna, z, h, loss_rna, loss_tcr, loss_vq
    
    def calc_loss(self, rna, rec_rna, tcr, rec_tcr):
        loss_rna = self.loss_func_rna(rec_rna, rna)
        mask = torch.ones_like(tcr).bool()
        mask[:, [0, mask.shape[1] // 2]] = False
        loss_tcr = self.loss_func_tcr(rec_tcr.flatten(end_dim=1), tcr[mask].flatten())
        return loss_rna, loss_tcr
