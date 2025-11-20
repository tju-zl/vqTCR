import torch
import torch.nn as nn
from .module import *


class TCRModel(nn.Module):
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
            # self.alpha_encoder = Attention(tcr_params, l_dim)
            self.alpha_decoder = TCRDecoder(tcr_params, l_dim)
        self.beta_encoder = TCREncoder(tcr_params, l_dim)
        # self.beta_encoder = Attention(tcr_params, l_dim)
        self.beta_decoder = TCRDecoder(tcr_params, l_dim)
        
        # GEX module
        self.gex_encoder = GEXEncoder(x_dim, l_dim)
        
        # # Bilinear Attention module for TCR chains
        self.bilin = nn.Bilinear(l_dim, l_dim, 1)
        self.act = nn.Sigmoid()
        
        # Loss function
        self.loss_func_rna = nn.MSELoss()
        self.loss_func_tcr = nn.CrossEntropyLoss(ignore_index=-100)

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
        
        z_tcell = h_tcr
        
        # Decode TCR
        if not self.beta_only:
            logic_alpha = self.alpha_decoder(z_tcell)
        logic_beta = self.beta_decoder(z_tcell)

        # Calculate reconstruction loss
        if not self.beta_only:
            loss_tcr = self.calc_loss(l_beta, logic_beta, l_alpha, logic_alpha)
        else:
            loss_tcr = self.calc_loss(l_beta, logic_beta)

        return [[loss_tcr],[z_tcell, h_tcr, h_gex]]
    
    # Calculate reconstruction loss
    def calc_loss(self, l_beta, logic_beta, l_alpha=None, logic_alpha=None):
        vocab_size = self.beta_encoder.embedding.vocab_size
        if l_alpha is not None and logic_alpha is not None:
            loss_alpha = self.loss_func_tcr(logic_alpha.view(-1, vocab_size), l_alpha.view(-1))
            loss_beta = self.loss_func_tcr(logic_beta.view(-1, vocab_size), l_beta.view(-1))
            loss_tcr = 0.5 * loss_alpha + 0.5 * loss_beta
            return loss_tcr
        else:
            loss_beta = self.loss_func_tcr(logic_beta.view(-1, vocab_size), l_beta.view(-1))
            return loss_beta
        

class GEXModel(nn.Module):
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
            self.alpha_decoder = TCRDecoder(tcr_params, l_dim)
        self.beta_encoder = TCREncoder(tcr_params, l_dim)
        self.beta_decoder = TCRDecoder(tcr_params, l_dim)
        
        self.tcr_base_decoder = TCRBaselineDecoder(x_dim, l_dim)
        
        # GEX module
        self.gex_encoder = GEXEncoder(x_dim, l_dim)
        
        # # Bilinear Attention module for TCR chains
        self.bilin = nn.Bilinear(l_dim, l_dim, 1)
        self.act = nn.Sigmoid()
        
        # Fusion
        self.fusion_film = FusionFiLM(l_dim, x_dim, fus_params['film_reg'])
        
        # Prototype register module
        # self.Tcell_pty = PQLayer(pty_params)
        self.Tcell_pty = PQcodebook(pty_params)
        
        # Loss function
        self.loss_func_rna = nn.MSELoss()
        self.loss_func_tcr = nn.CrossEntropyLoss(ignore_index=-100)
        
        self.fus_lin = nn.Linear(2*l_dim, l_dim)
    
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
        
        z_tcell = h_gex

        rec_gex  = self.tcr_base_decoder(z_tcell)
        
        # Calculate reconstruction loss
        if not self.beta_only:
            loss_gex = self.calc_loss(gex, rec_gex)
        else:
            loss_gex = self.calc_loss(gex, rec_gex)

        return [[loss_gex],[z_tcell, h_tcr, h_gex]]

    
    # Calculate reconstruction loss
    def calc_loss(self, gex, rec_gex):
        loss_gex = self.loss_func_rna(rec_gex, gex)
        return loss_gex


class ConcatModel(nn.Module):
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
            self.alpha_decoder = TCRDecoder(tcr_params, l_dim)
        self.beta_encoder = TCREncoder(tcr_params, l_dim)
        self.beta_decoder = TCRDecoder(tcr_params, l_dim)
        
        self.tcr_base_decoder = TCRBaselineDecoder(x_dim, l_dim)
        
        # GEX module
        self.gex_encoder = GEXEncoder(x_dim, l_dim)
        
        # # Bilinear Attention module for TCR chains
        self.bilin = nn.Bilinear(l_dim, l_dim, 1)
        self.act = nn.Sigmoid()
        
        # Prototype register module
        self.Tcell_pty = PQcodebook(pty_params)
        
        # Loss function
        self.loss_func_rna = nn.MSELoss()
        self.loss_func_tcr = nn.CrossEntropyLoss(ignore_index=-100)
        
        self.fus_lin = nn.Sequential(nn.Linear(2*l_dim, l_dim),
                                     nn.Dropout(0.2),
                                     nn.ELU(),
                                     nn.LayerNorm(l_dim),
                                     nn.Linear(l_dim, l_dim))
    
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
        
        z_tcell = self.fus_lin(h_tcell)
        
        # Decode TCR
        if not self.beta_only:
            logic_alpha = self.alpha_decoder(z_tcell)
        logic_beta = self.beta_decoder(z_tcell)
        
        rec_gex = self.tcr_base_decoder(z_tcell)
        

        # Calculate reconstruction loss
        if not self.beta_only:
            loss_gex, loss_tcr = self.calc_loss(gex, rec_gex, l_beta, logic_beta, l_alpha, logic_alpha)
        else:
            loss_gex, loss_tcr = self.calc_loss(gex, rec_gex, l_beta, logic_beta)

        return [[loss_gex, loss_tcr],[z_tcell, h_tcr, h_gex]]

    
    # Calculate reconstruction loss
    def calc_loss(self, gex, rec_gex, l_beta, logic_beta, l_alpha=None, logic_alpha=None):
        loss_bias = self.loss_func_rna(rec_gex, gex)
        loss_gex = loss_bias
       
        vocab_size = self.beta_encoder.embedding.vocab_size
        if l_alpha is not None and logic_alpha is not None:
            loss_alpha = self.loss_func_tcr(logic_alpha.view(-1, vocab_size), l_alpha.view(-1))
            loss_beta = self.loss_func_tcr(logic_beta.view(-1, vocab_size), l_beta.view(-1))
            loss_tcr = 0.5 * loss_alpha + 0.5 * loss_beta
            return loss_gex, loss_tcr
        else:
            loss_beta = self.loss_func_tcr(logic_beta.view(-1, vocab_size), l_beta.view(-1))
            
            return loss_gex, loss_beta


class PoEModel(nn.Module):
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
            self.alpha_decoder = TCRDecoder(tcr_params, int(1.5*l_dim))
        self.beta_encoder = TCREncoder(tcr_params, l_dim)
        self.beta_decoder = TCRDecoder(tcr_params, int(1.5*l_dim))
        
        self.tcr_base_decoder = TCRBaselineDecoder(x_dim, int(1.5*l_dim))
        
        # GEX module
        self.gex_encoder = GEXEncoder(x_dim, l_dim)
        
        # # Bilinear Attention module for TCR chains
        self.bilin = nn.Bilinear(l_dim, l_dim, 1)
        self.act = nn.Sigmoid()
        
        
        # Loss function
        self.loss_func_rna = nn.MSELoss()
        self.loss_func_tcr = nn.CrossEntropyLoss(ignore_index=-100)
        self.loss_function_kld = KLD()

    
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

        mu_gex, logvar_gex = h_gex[:, :h_gex.shape[1] // 2], h_gex[:, h_gex.shape[1] // 2:]
        z_gex = self.reparameterize(mu_gex, logvar_gex)
        
        mu_tcr, logvar_tcr = h_tcr[:, :h_tcr.shape[1] // 2], h_tcr[:, h_tcr.shape[1] // 2:]
        z_tcr = self.reparameterize(mu_tcr, logvar_tcr) 
        
        mu_joint, logvar_joint = self.product_of_experts(mu_gex, mu_tcr, logvar_gex, logvar_tcr)
        z_joint = self.reparameterize(mu_joint, logvar_joint)
        
        mu = [mu_gex, mu_tcr, mu_joint]
        logvar = [logvar_gex, logvar_tcr, logvar_joint]
        
        z_tcell = torch.cat([z_gex, z_tcr, z_joint], dim=1)
        
        loss_kld = self.calculate_kld_loss(mu, logvar)
        
        # Decode TCR
        if not self.beta_only:
            logic_alpha = self.alpha_decoder(z_tcell)
        logic_beta = self.beta_decoder(z_tcell)
        
        rec_gex = self.tcr_base_decoder(z_tcell)
        

        # Calculate reconstruction loss
        if not self.beta_only:
            loss_gex, loss_tcr = self.calc_loss(gex, rec_gex, l_beta, logic_beta, l_alpha, logic_alpha)
        else:
            loss_gex, loss_tcr = self.calc_loss(gex, rec_gex, l_beta, logic_beta)

        return [[loss_gex, loss_tcr, loss_kld],[z_tcell, z_tcr, z_gex]]


    def reparameterize(self, mu, log_var):
        """
        https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        z = mu + (eps * std)  # sampling as if coming from the input space
        return z

    def product_of_experts(self, mu_rna, mu_tcr, logvar_rna, logvar_tcr):
        # formula: var_joint = inv(inv(var_prior) + sum(inv(var_modalities)))
        logvar_joint = 1.0 / torch.exp(logvar_rna) + 1.0 / torch.exp(
            logvar_tcr) + 1.0  # sum up all inverse vars, logvars first needs to be converted to var, last 1.0 is coming from the prior
        logvar_joint = torch.log(1.0 / logvar_joint)  # inverse and convert to logvar

        # formula: mu_joint = (mu_prior*inv(var_prior) + sum(mu_modalities*inv(var_modalities))) * var_joint, where mu_prior = 0.0
        mu_joint = mu_rna * (1.0 / torch.exp(logvar_rna)) + mu_tcr * (1.0 / torch.exp(logvar_tcr))
        mu_joint = mu_joint * torch.exp(logvar_joint)

        return mu_joint, logvar_joint
    
    def calculate_kld_loss(self, mu, logvar):
        kld_loss = (self.loss_function_kld(mu[0], logvar[0])
                    + self.loss_function_kld(mu[1], logvar[1])
                    + self.loss_function_kld(mu[2], logvar[2]))
        return 0.01 * kld_loss / 3.0 
        
    
    # Calculate reconstruction loss
    def calc_loss(self, gex, rec_gex, l_beta, logic_beta, l_alpha=None, logic_alpha=None):
        loss_bias = self.loss_func_rna(rec_gex, gex)
        loss_gex = loss_bias
       
        vocab_size = self.beta_encoder.embedding.vocab_size
        if l_alpha is not None and logic_alpha is not None:
            loss_alpha = self.loss_func_tcr(logic_alpha.view(-1, vocab_size), l_alpha.view(-1))
            loss_beta = self.loss_func_tcr(logic_beta.view(-1, vocab_size), l_beta.view(-1))
            loss_tcr = 0.5 * loss_alpha + 0.5 * loss_beta
            return loss_gex, loss_tcr
        else:
            loss_beta = self.loss_func_tcr(logic_beta.view(-1, vocab_size), l_beta.view(-1))
            
            return loss_gex, loss_beta

class KLD(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(KLD, self).__init__()
        self.reduction = reduction

    def forward(self, mu, logvar, mu_2=None, logvar_2=None):
        """
        Calculate the Kullbach-Leibler-Divergence between two Gaussians
        :param mu: mean of the first Gaussian
        :param logvar: log(var) of the first Gaussian
        :param mu_2: mean of the second Gaussian (default: 0)
        :param logvar_2: log(var) of the second Gaussian (default: 1)
        :return: loss value
        """
        if mu_2 is None or logvar_2 is None:
            kl = self.univariate_kl_loss(mu, logvar)
        else:
            kl = self.general_kl_loss(mu, logvar, mu_2, logvar_2)
        if self.reduction == 'mean':
            kl = torch.mean(kl)
        elif self.reduction == 'sum':
            kl = torch.sum(kl)
        else:
            raise NotImplementedError(f'reduction method {self.reduction} is not implemented.')
        return kl

    def univariate_kl_loss(self, mu, logvar):
        """
        KL loss between the input and a 0 mean, uni-variance Gaussian
        :param mu: mean of the distribution
        :param logvar: log variance of the distribution
        :return: Kulbach Leibler divergence between distribution and Gaussian
        """
        kl = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar, dim=1)
        return kl

    def general_kl_loss(self, mu_1, logvar_1, mu_2, logvar_2):
        """
        KL loss between two distributions
        :param mu_1: mean of the first distribution
        :param logvar_1: log variance of the first distribution
        :param mu_2: mean of the second distribution
        :param logvar_2: log variane of the second distribution
        :return: Kulbach Leibler divergence loss between the two distributions
        """
        kl = logvar_2 - logvar_1 + torch.exp(logvar_1)/torch.exp(logvar_2) + (mu_1-mu_2)**2/torch.exp(logvar_2)-1
        kl = 0.5 * torch.sum(kl)
        return kl        
        
class MoEModel(nn.Module):
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
            self.alpha_decoder = TCRDecoder(tcr_params, l_dim)
        self.beta_encoder = TCREncoder(tcr_params, l_dim)
        self.beta_decoder = TCRDecoder(tcr_params, l_dim)
        
        self.tcr_base_decoder = TCRBaselineDecoder(x_dim, l_dim)
        
        # GEX module
        self.gex_encoder = GEXEncoder(x_dim, l_dim)
        
        # # Bilinear Attention module for TCR chains
        self.bilin = nn.Bilinear(l_dim, l_dim, 1)
        self.act = nn.Sigmoid()
        
        # Loss function
        self.loss_func_rna = nn.MSELoss()
        self.loss_func_tcr = nn.CrossEntropyLoss(ignore_index=-100)
        self.loss_function_kld = KLD()
    
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
    
        mu_gex, logvar_gex = h_gex[:, :h_gex.shape[1] // 2], h_gex[:, h_gex.shape[1] // 2:]
        z_gex = self.reparameterize(mu_gex, logvar_gex)
        
        mu_tcr, logvar_tcr = h_tcr[:, :h_tcr.shape[1] // 2], h_tcr[:, h_tcr.shape[1] // 2:]
        z_tcr = self.reparameterize(mu_tcr, logvar_tcr) 
        
        mu = [mu_gex, mu_tcr]
        logvar = [logvar_gex, logvar_tcr]
        
        z_tcell = torch.cat([z_gex, z_tcr], dim=1)
        
        loss_kld = self.calculate_kld_loss(mu, logvar)
        
        # Decode TCR
        if not self.beta_only:
            logic_alpha = self.alpha_decoder(z_tcell)
        logic_beta = self.beta_decoder(z_tcell)
        
        rec_gex = self.tcr_base_decoder(z_tcell)
        

        # Calculate reconstruction loss
        if not self.beta_only:
            loss_gex, loss_tcr = self.calc_loss(gex, rec_gex, l_beta, logic_beta, l_alpha, logic_alpha)
        else:
            loss_gex, loss_tcr = self.calc_loss(gex, rec_gex, l_beta, logic_beta)

        return [[loss_gex, loss_tcr, loss_kld],[z_tcell, z_tcr, z_gex]]

    def reparameterize(self, mu, log_var):
        """
		https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/
		:param mu: mean from the encoder's latent space
		:param log_var: log variance from the encoder's latent space
		"""
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        z = mu + (eps * std)  # sampling as if coming from the input space
        return z
    
    # Calculate reconstruction loss
    def calc_loss(self, gex, rec_gex, l_beta, logic_beta, l_alpha=None, logic_alpha=None):
        loss_bias = self.loss_func_rna(rec_gex, gex)
        loss_gex = loss_bias
       
        vocab_size = self.beta_encoder.embedding.vocab_size
        if l_alpha is not None and logic_alpha is not None:
            loss_alpha = self.loss_func_tcr(logic_alpha.view(-1, vocab_size), l_alpha.view(-1))
            loss_beta = self.loss_func_tcr(logic_beta.view(-1, vocab_size), l_beta.view(-1))
            loss_tcr = 0.5 * loss_alpha + 0.5 * loss_beta
            return loss_gex, loss_tcr
        else:
            loss_beta = self.loss_func_tcr(logic_beta.view(-1, vocab_size), l_beta.view(-1))
            
            return loss_gex, loss_beta
        
    def calculate_kld_loss(self, mu, logvar):
        kld_loss = (self.loss_function_kld(mu[0], logvar[0]) + self.loss_function_kld(mu[1], logvar[1]))
        return 0.01 * kld_loss * 0.5
