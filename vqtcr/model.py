import torch
import torch.nn as nn
from .module import *


class vqTCRModel(nn.Module):
    def __init__(self, params, tcr_params, gex_params,
                 fus_params, pty_params):
        super().__init__()
        # params setting
        l_dim = params['dim_latent']
        x_dim = gex_params['x_dim']
        self.beta_only = tcr_params['beta_only']
        
        # TCR module
        if not self.beta_only:
            self.alpha_encoder = TCREncoder(tcr_params, l_dim)
            self.alpha_decoder = TCRDecoder(tcr_params, 2*l_dim)
        self.beta_encoder = TCREncoder(tcr_params, l_dim)
        self.beta_decoder = TCRDecoder(tcr_params, 2*l_dim)
        
        self.tcr_base_decoder = TCRBaselineDecoder(x_dim, 2*l_dim)
        
        # GEX module
        self.gex_encoder = GEXEncoder(x_dim, l_dim)
        
        # # Bilinear Attention module for TCR chains
        self.bilin = nn.Bilinear(l_dim, l_dim, 1)
        self.act = nn.Sigmoid()
        
        # Fusion
        self.fusion_film = FusionFiLM(l_dim, x_dim, fus_params['film_reg'])
        
        # Prototype register module
        self.Tcell_pty = PQcodebook(pty_params)
        
        # Loss function
        self.loss_func_rna = nn.MSELoss()
        self.loss_func_tcr = nn.CrossEntropyLoss(ignore_index=-100)
        
        self.fus_lin = nn.Linear(2*l_dim, l_dim)
    
    def warmup_code(self, gex, tcr):
        if not self.beta_only:
            alpha_seq = tcr[:, :tcr.shape[1]//2]
            beta_seq = tcr[:, tcr.shape[1]//2:]
        else:
            beta_seq = tcr
        
        # Encode GEX
        h_gex = self.gex_encoder(gex)
        
        # Encode TCR
        if not self.beta_only:
            h_alpha, l_alpha = self.alpha_encoder(alpha_seq)
        h_beta, l_beta = self.beta_encoder(beta_seq)
        
        # Fusion TCR alpha and beta
        if not self.beta_only:
            h_alpha = h_alpha / torch.norm(h_alpha, dim=1, keepdim=True)
            h_beta = h_beta / torch.norm(h_beta, dim=1, keepdim=True)
            tcr_att = self.act(self.bilin(h_alpha, h_beta))
            h_tcr = tcr_att * h_alpha + (1 - tcr_att) * h_beta
        else:
            h_tcr = h_beta
    
        h_tcr = F.layer_norm(h_tcr, (h_tcr.size(-1),))
        h_gex = F.layer_norm(h_gex, (h_gex.size(-1),))
        
        z_tcell = torch.cat([h_tcr, h_gex], dim=1)
        
        # Decode TCR
        if not self.beta_only:
            logic_alpha = self.alpha_decoder(z_tcell)
        logic_beta = self.beta_decoder(z_tcell)

        
        base_tcr = self.tcr_base_decoder(z_tcell)
        
        # # FiLM fusion
        bias_gex, loss_reg, gamma, beta = self.fusion_film(h_gex, z_tcell)

        # Full GEX
        rec_gex = base_tcr + bias_gex

        # Calculate reconstruction loss
        if not self.beta_only:
            loss_gex, loss_tcr = self.calc_loss(gex, base_tcr, rec_gex, l_beta, logic_beta, l_alpha, logic_alpha)
        else:
            loss_gex, loss_tcr = self.calc_loss(gex, base_tcr, rec_gex, l_beta, logic_beta)

        return [[loss_gex, loss_tcr, loss_reg],[z_tcell, h_tcr, h_gex, base_tcr, bias_gex, gamma, beta]]
    
    def forward(self, gex, tcr):
        if not self.beta_only:
            alpha_seq = tcr[:, :tcr.shape[1]//2]
            beta_seq = tcr[:, tcr.shape[1]//2:]
        else:
            beta_seq = tcr
        
        # Encode GEX
        h_gex = self.gex_encoder(gex)
        
        # Encode TCR
        if not self.beta_only:
            h_alpha, l_alpha = self.alpha_encoder(alpha_seq)
        h_beta, l_beta = self.beta_encoder(beta_seq)

        
        # Fusion TCR alpha and beta
        if not self.beta_only:
            h_alpha = h_alpha / torch.norm(h_alpha, dim=1, keepdim=True)
            h_beta = h_beta / torch.norm(h_beta, dim=1, keepdim=True)
            tcr_att = self.act(self.bilin(h_alpha, h_beta))
            h_tcr = tcr_att * h_alpha + (1 - tcr_att) * h_beta
        else:
            h_tcr = h_beta
        
        h_tcr = F.layer_norm(h_tcr, (h_tcr.size(-1),))
        h_gex = F.layer_norm(h_gex, (h_gex.size(-1),))
        
        h_tcell = torch.cat([h_tcr, h_gex], dim=1)
        
        z_tcell, indices, info, stats = self.Tcell_pty(h_tcell)
        
        # Decode TCR
        if not self.beta_only:
            logic_alpha = self.alpha_decoder(z_tcell)
        logic_beta = self.beta_decoder(z_tcell)

        
        base_tcr = self.tcr_base_decoder(z_tcell)
        
        # # FiLM fusion
        bias_gex, loss_reg, gamma, beta = self.fusion_film(h_gex, z_tcell)
        
        # Full GEX
        rec_gex = base_tcr + bias_gex

        # Calculate reconstruction loss
        if not self.beta_only:
            loss_gex, loss_tcr = self.calc_loss(gex, base_tcr, rec_gex, l_beta, logic_beta, l_alpha, logic_alpha)
        else:
            loss_gex, loss_tcr = self.calc_loss(gex, base_tcr, rec_gex, l_beta, logic_beta)

        return [[loss_gex, loss_tcr, loss_reg, info],[z_tcell, h_tcr, h_gex, base_tcr, bias_gex, gamma, beta, indices]]
    
    # Calculate reconstruction loss
    def calc_loss(self, gex, tcr_base, rec_gex, l_beta, logic_beta, l_alpha=None, logic_alpha=None):
        loss_base = self.loss_func_rna(tcr_base, gex)
        loss_bias = self.loss_func_rna(rec_gex, gex)
        # loss_gex = loss_bias
        loss_gex = 0.5 * loss_bias  + 0.5 * loss_base
       
        vocab_size = self.beta_encoder.embedding.vocab_size
        if l_alpha is not None and logic_alpha is not None:
            loss_alpha = self.loss_func_tcr(logic_alpha.view(-1, vocab_size), l_alpha.view(-1))
            loss_beta = self.loss_func_tcr(logic_beta.view(-1, vocab_size), l_beta.view(-1))
            loss_tcr = 0.5 * loss_alpha + 0.5 * loss_beta
            return loss_gex, loss_tcr
        else:
            loss_beta = self.loss_func_tcr(logic_beta.view(-1, vocab_size), l_beta.view(-1))
            
            return loss_gex, loss_beta
