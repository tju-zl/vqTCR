import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, EsmModel
from .predefined import TCR_VOCAB


# Mask aa sequence of TCR CDR3, exlude special tokens pad, (start, end).
def mask_mlm(input_ids, pad_token_id, mask_token_id, vocab_size, mlm_prob=0.15):
    """
    input_ids: [B, S]
    return: masked input_ids, labels
    """
    device = input_ids.device
    labels = input_ids.clone()
    
    is_pad = input_ids.eq(pad_token_id)
    can_mask = ~is_pad  # [B, S]
    
    mask_selector = (torch.rand_like(input_ids.float()) < mlm_prob) & can_mask
    
    labels[~mask_selector] = -100
    
    masked_input_ids = input_ids.clone()
    
    probs = torch.rand_like(input_ids.float())
    mask80 = (probs < 0.8) & mask_selector
    masked_input_ids[mask80] = mask_token_id
    
    rand10 = (probs >= 0.8) & (probs < 0.9) & mask_selector
    random_tokens = torch.randint(low=0, high=vocab_size, 
                                  size=input_ids.shape, device=device)
    random_tokens[random_tokens == pad_token_id] = mask_token_id
    masked_input_ids[rand10] = random_tokens[rand10]
    
    return masked_input_ids, labels


# Initialize amino acid embedding layer
class AminoAcidEmbedding(nn.Module):
    def __init__(self, params, use_esm=True, freeze_esm=False):
        super().__init__()
        self.params = params
        
        if use_esm:
            self.tokenizer = AutoTokenizer.from_pretrained(params['esm_type'])
            esm_model = EsmModel.from_pretrained(params['esm_type'])
            # get ESM embedding
            self.embedding = esm_model.embeddings.word_embeddings
            self.token_dim = self.embedding.embedding_dim
            self.vocab = self.tokenizer.get_vocab()
            self.vocab_size = self.tokenizer.vocab_size
            # freeze ESM embedding (optional)
            if freeze_esm:
                for param in self.embedding.parameters():
                    param.requires_grad = False
        else:
            self.vocab = TCR_VOCAB
            self.vocab_size = len(self.vocab)
            self.embedding = nn.Embedding(self.vocab_size, 
                                        params['dim_emb'], 
                                        padding_idx=TCR_VOCAB['<pad>']) #
            self.token_dim = self.embedding.embedding_dim

    def forward(self, x):
        if self.training:
            mask_rate = self.params['mask_rate']
        else:
            mask_rate = 0.0
        masked_ids, labels = mask_mlm(x, self.vocab['<pad>'], 
                                        self.vocab['<mask>'], 
                                        self.vocab_size, 
                                        mask_rate)
        emb = self.embedding(masked_ids)   # batch_size x seq_len x embedding_dim
        return emb, labels


class KmerCNNExtractor(nn.Module):
    '''
    Extract length-k motif from TCR CDR3 amino acid sequence
    multiple k-mer kernals with CNN (Convd1d)
    '''
    def __init__(self, params, hdim, token_dim):
        '''
        params: hyperparameters as dict
        hdim: output feature dimension
        '''
        super().__init__()
        self.params = params
        k_mers = params['kmer_kernels']
        dim_emb = int(params['dim_emb']/len(k_mers))
        
        # multiple k-mer convolutional layers
        self.conv_layers = nn.ModuleList([
            nn.Sequential(nn.Conv1d(token_dim, dim_emb, k, padding=k//2),
                          nn.BatchNorm1d(dim_emb),
                          nn.ELU()
                          ) for k in k_mers
            ])
        
        # feature dimension fusion
        self.fc_fusion = nn.Sequential(nn.Linear(params['dim_emb'], hdim),
                                       nn.ELU(),
                                       nn.Dropout(params['dropout']),
                                       nn.Linear(hdim, hdim)
                                       )
        
    
    def forward(self, x):       # input shape: (batch_size, seq_len, dim_emb)
        x = x.transpose(1, 2)
        conv_out = torch.cat([conv(x) for conv in self.conv_layers], dim=1)
        conv_out = conv_out.transpose(1, 2)
        conv_out = self.fc_fusion(conv_out)
        return conv_out


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]      # seq_len x batch_size x embedding_dim
        return self.dropout(x)


class MotifAttention(nn.Module):
    '''
    Using Transformer encoder to learn the TCRs
    '''
    def __init__(self, params, hdim):
        '''
        params: hyperparameters as dict
        hdim: output feature dimension
        '''
        super().__init__()
        self.params = params
        self.positional_encoding = PositionalEncoding(params['dim_emb'],
                                                      params['dropout'],
                                                      params['max_tcr_length'])
        encoding_layers = nn.TransformerEncoderLayer(params['dim_emb'],
                                                     params['num_heads'],
                                                     params['dim_emb'] * params['forward_expansion'],
                                                     params['dropout'])
        self.transformer_encoder = nn.TransformerEncoder(encoding_layers, params['num_encoder'])
        
        self.attention_pooling = nn.Sequential(
            nn.Linear(hdim, hdim),
            nn.LayerNorm(hdim),
            nn.Tanh(),
            nn.Dropout(params['dropout']),
            nn.Linear(hdim, 1),
            nn.Flatten(start_dim=1),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = x.transpose(0, 1)                   # seq_len x batch_size x dim_latent
        x = x + self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)
        attn_weights = self.attention_pooling(x).unsqueeze(-1)  # [batch_size, seq_len, 1]
        out = torch.sum(attn_weights * x, dim=1)                # [batch_size, dim_latent]
        return out

class TCREncoder(nn.Module):
    def __init__(self, params, hdim) -> None:
        super().__init__()
        self.params = params
        self.embedding = AminoAcidEmbedding(params, use_esm=params['esm_tokens'])
        self.multi_kmer = KmerCNNExtractor(params, hdim, self.embedding.token_dim)
        self.attention = MotifAttention(params, hdim)
    
    def forward(self, x):
        x, labels = self.embedding(x)
        x = self.multi_kmer(x)
        x = self.attention(x)
        return x, labels


class TCRDecoder(nn.Module):
    def __init__(self, params, hdim):
        super().__init__()
        self.params = params
        self.mlm_mlp = nn.Sequential(
            nn.Linear(hdim, 256),
            nn.ReLU(),
            nn.Linear(256, params['max_tcr_length']*len(TCR_VOCAB))
        )
    
    def forward(self, x):
        x = self.mlm_mlp(x)
        shape = (x.shape[0], self.params['max_tcr_length'], len(TCR_VOCAB))
        x = torch.reshape(x, shape)
        return x

# TCR base decoder
class TCRBaselineDecoder(nn.Module):
    def __init__(self, raw_dim=3000, latent_dim=64):
        super().__init__()
        self.in_dim = raw_dim
        self.latent_dim = latent_dim
        
        self.dec0 = nn.Sequential(nn.Linear(latent_dim, 256), 
                                #  nn.BatchNorm1d(256), 
                                 nn.LayerNorm(256),
                                 nn.ELU())
        self.dec1 = nn.Sequential(nn.Linear(256, 512), 
                                #   nn.BatchNorm1d(512),
                                  nn.LayerNorm(512),
                                  nn.ELU())
        self.dec2 = nn.Sequential(nn.Linear(512, 1024), 
                                #   nn.BatchNorm1d(1024),
                                  nn.LayerNorm(1024),
                                  nn.ELU())
        self.dec3 = nn.Linear(1024, raw_dim)
        
        self.res0 = nn.Linear(latent_dim, raw_dim)
        self.res1 = nn.Linear(256, 1024)
        self.res2 = nn.Linear(512, raw_dim)
        
    def forward(self, x):
        x1 = self.dec0(x)
        x2 = self.dec1(x1)
        x3 = self.dec2(x2)
        x4 = x3 + self.res1(x1)
        x5 = self.dec3(x4) + self.res2(x2)
        return x5 + self.res0(x) 


# GEX encoder
class GEXEncoder(nn.Module):
    def __init__(self, raw_dim=3000, latent_dim=64):
        super().__init__()
        self.in_dim = raw_dim
        self.latent_dim = latent_dim
        
        self.enc0 = nn.Sequential(nn.Linear(raw_dim, 1024), 
                                 nn.LayerNorm(1024),
                                 nn.ELU())
        self.enc1 = nn.Sequential(nn.Linear(1024, 512), 
                                  nn.LayerNorm(512),
                                  nn.ELU())
        self.enc2 = nn.Sequential(nn.Linear(512, 256), 
                                  nn.LayerNorm(256),
                                  nn.ELU())
        self.enc3 = nn.Linear(256, latent_dim)
        
        self.res0 = nn.Linear(raw_dim, latent_dim)
        self.res1 = nn.Linear(1024, 256)
        self.res2 = nn.Linear(512, latent_dim)
        
    def forward(self, x):
        x1 = self.enc0(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = x3 + self.res1(x1)
        x5 = self.enc3(x4) + self.res2(x2)
        return x5 + self.res0(x)


# GEX decoder
class GEXDecoder(nn.Module):
    def __init__(self, raw_dim=3000, latent_dim=64):
        super().__init__()
        self.in_dim = raw_dim
        self.latent_dim = latent_dim
        
        self.dec0 = nn.Sequential(nn.Linear(latent_dim, 256), 
                                 nn.LayerNorm(256),
                                 nn.ELU())
        self.dec1 = nn.Sequential(nn.Linear(256, 512), 
                                  nn.LayerNorm(512),
                                  nn.ELU())
        self.dec2 = nn.Sequential(nn.Linear(512, 1024), 
                                  nn.LayerNorm(1024),
                                  nn.ELU())
        self.dec3 = nn.Linear(1024, raw_dim)
        
        self.res0 = nn.Linear(latent_dim, raw_dim)
        self.res1 = nn.Linear(256, 1024)
        self.res2 = nn.Linear(512, raw_dim)
        
    def forward(self, x):
        x1 = self.dec0(x)
        x2 = self.dec1(x1)
        x3 = self.dec2(x2)
        x4 = x3 + self.res1(x1)
        x5 = self.dec3(x4) + self.res2(x2)
        return x5 + self.res0(x)    


# FiLM concatenation fusion
class FusionFiLM(nn.Module):
    def __init__(self, in_feat, out_feat, lambda_reg=1e-3):
        super().__init__()
        self.lambda_reg = lambda_reg
        
        self.gamma_net = nn.Linear(in_feat, out_feat)
        self.beta_net = nn.Linear(in_feat, out_feat)
        self.adapter = nn.Linear(2*in_feat, out_feat)
        
    def forward(self, x, y):
        gamma = self.gamma_net(x) + 1.0
        beta = self.beta_net(x)
        y = self.adapter(y)
        
        f = gamma * y + beta
        reg_loss = self.lambda_reg * ((gamma - 1.0).pow(2).mean() + beta.pow(2).mean())
        
        return f, reg_loss, gamma, beta
    

class PQcodebook(nn.Module):
    def __init__(self, params) -> None:
        super().__init__()
        self.params = params
        self.use_cosine = params['use_cosine']
        self.ema_eps = params['eps']
        self.ema_decay = params['ema_decay']
        self.dead_code_interval = params['dead_code_interval']
        self.dead_code_threshold = params['dead_code_threshold']
        self.dead_code_topk = params['dead_code_topk']
        self.commitment_beta = params['commitment_beta']
        self.diversity_weight = params['diversity_weight']
        self.sub_dim = params['dim_latent'] // params['num_subspace']
        
        self.M, self.K, self.d = params['num_subspace'], params['codewords_per_space'], self.sub_dim
        
        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        self.register_buffer("recent_usage", torch.zeros(self.M, self.K))
    
    @ torch.no_grad()
    def _init_codebook(self, init_code, device):

        init_code = init_code.to(device)
        self.codebook = nn.Parameter(init_code, requires_grad=False)
        
        self.register_buffer("ema_count", torch.zeros(self.M, self.K, device=device))
        self.register_buffer("ema_sum", torch.zeros(self.M, self.K, self.d, device=device))

        self.ema_sum.copy_(init_code)
        self.ema_count.add_(1.0) # avoid div by zero later
    
    def _distance(self, z_sub: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        # z_sub: [B, d]; E: [K, d]; return distances or negative sims: [B, K]
        if self.use_cosine:
            z_n = F.normalize(z_sub, dim=-1)
            E_n = F.normalize(E, dim=-1)
            # cosine similarity -> higher is better; we want argmax
            sim = z_n @ E_n.t()  # [B*, K]
            # we return negative to use argmin-style downstream if needed
            return -sim, sim
        else:
            # squared L2 distance: (x - e)^2 = x^2 + e^2 - 2 x·e
            x2 = (z_sub ** 2).sum(dim=-1, keepdim=True)      # [B, 1]
            e2 = (E ** 2).sum(dim=-1).unsqueeze(0)           # [1, K]
            xe = z_sub @ E.t()                               # [B, K]
            dist = x2 + e2 - 2 * xe
            return dist, -dist
    
    @torch.no_grad()
    def _ema_update(self, z_sub: torch.Tensor, indices: torch.Tensor, m: int):
        # z_sub: [B, d]; indices: [B]

        one_hot = F.one_hot(indices, num_classes=self.K).type_as(z_sub)  # [B, K]
        counts = one_hot.sum(dim=0)                                      # [K]
        sums = one_hot.t() @ z_sub                                       # [K, d]

        self.ema_count[m].mul_(self.ema_decay).add_(counts * (1 - self.ema_decay))
        self.ema_sum[m].mul_(self.ema_decay).add_(sums * (1 - self.ema_decay))

        n = self.ema_count[m] + self.ema_eps
        new_E = self.ema_sum[m] / n.unsqueeze(-1)                        # [K, d]
        if self.use_cosine:
            new_E = F.normalize(new_E, dim=-1)
        self.codebook.data[m] = new_E

        # track recent usage for dead-code
        total = n.sum().clamp_min(self.ema_eps)
        self.recent_usage[m] = n / total
    
    @torch.no_grad()
    def _dead_code_resurrect(self, z_sub: torch.Tensor, err: torch.Tensor, m: int):
        """
        Replace rarely used codes with high-error samples.
        z_sub: [B, d]; err: [B] quantization error per sample (for this subspace)
        """
        if self.step.item() % self.dead_code_interval != 0:
            return

        usage = self.recent_usage[m]  # [K]
        # Dead if usage ratio below threshold
        dead_mask = usage < self.dead_code_threshold
        if not torch.any(dead_mask):
            return

        num_dead = int(dead_mask.sum().item())
        topk = min(self.dead_code_topk, z_sub.size(0))
        if topk == 0 or num_dead == 0:
            return

        # pick top-k error samples to seed
        _, idx = torch.topk(err, k=topk, largest=True)
        candidates = z_sub[idx]  # [topk, d]
        # randomly choose without replacement
        replace = candidates[torch.randperm(candidates.size(0))[:num_dead]]

        # replace in codebook and reset EMA stats
        self.codebook.data[m, dead_mask] = F.normalize(replace, dim=-1) if self.use_cosine else replace
        self.ema_count[m, dead_mask] = 0.0
        self.ema_sum[m, dead_mask] = 0.0
        # give a tiny warm-start usage to avoid immediate dead again
        self.recent_usage[m, dead_mask] = self.dead_code_threshold
        
    def forward(self, z):
        if self.training:
            self.step += 1
        B, D = z.shape
        M, K, d = self.M, self.K, self.d
        assert D == M * d
        z_sub = z.view(B, M, d)
        
        all_indices = []
        recon_parts = []
        commit_loss = 0.0
        
        perplexities = []
        dead_rates = []
        usage_hists = []
        
        for m in range(self.M):
            z_m = z_sub[:, m, :]
            E_m = self.codebook[m]  # [K, d]
            # distances / similarities
            dist, sim = self._distance(z_m, E_m)  # [B, K]
            # select nearest (for cosine we use argmax of sim == argmin of -sim/dist)
            k = torch.argmax(sim, dim=-1)  # [B]
            all_indices.append(k)

            # straight-through estimator:
            # one-hot select and add gradient copy
            z_q_m = F.embedding(k, E_m)                     # [B*, d]
            recon_parts.append(z_q_m)

            # commitment loss (use cosine or L2)
            if self.use_cosine:
                # 1 - cosine(z, z_q) ~ angle; use MSE on normalized vectors
                z_norm = F.normalize(z_m, dim=-1)
                q_norm = F.normalize(z_q_m, dim=-1)
                commit = F.mse_loss(q_norm.detach(), z_norm, reduction="mean")
            else:
                commit = F.mse_loss(z_q_m.detach(), z_m, reduction="mean")
            commit_loss = commit_loss + commit

            # diversity/entropy (batch usage)
            with torch.no_grad():
                usage = F.one_hot(k, num_classes=K).float().mean(dim=0)  # [K]
                # perplexity = exp(H(p))
                entropy = -(usage * (usage.clamp_min(1e-12).log())).sum()
                perplexity = torch.exp(entropy)
                dead_rate = float((usage < self.dead_code_threshold).float().mean().item())
                perplexities.append(float(perplexity.item()))
                dead_rates.append(dead_rate)
                usage_hists.append(usage.cpu())

            # EMA update (no grad)
            if self.training:
                self._ema_update(z_m, k, m)

            # Dead-code handling with error as criterion
            with torch.no_grad():
                if self.training:
                    # quantization error per sample for this subspace
                    if self.use_cosine:
                        # use 1 - cosine sim as error (>=0)
                        z_norm = F.normalize(z_m, dim=-1)
                        q_norm = F.normalize(z_q_m, dim=-1)
                        err = (1.0 - (z_norm * q_norm).sum(dim=-1)).clamp_min(0)
                    else:
                        err = ((z_q_m - z_m) ** 2).sum(dim=-1)
                    self._dead_code_resurrect(z_m, err, m)

        # concat parts and apply STE
        z_q = torch.stack(recon_parts, dim=1).reshape(B, D)   # [B, M, d]=>[B, D]

        # Straight-through: copy gradient to inputs
        z_st = z + (z_q - z).detach()

        # aggregate losses
        losses = {
            "commitment_loss": self.commitment_beta * commit_loss / self.M,
            #"diversity_loss": self.diversity_weight * (diversity_loss / self.M),
        }

        # indices restore to [B, M]
        indices = torch.stack(all_indices, dim=-1)  # [B, M]

        stats = {
            "perplexity_per_codebook": perplexities,   # list of length M (python floats)
            "dead_rate_per_codebook": dead_rates,      # list of length M
            "usage_hist_per_codebook": usage_hists,    # list of tensors [K]
        }
        print(f"perplexity_per_codebook {perplexities}", 
              f"dead_rate_per_codebook{dead_rates}")
        return z_st, indices, losses, stats



# pMHC binding labels data supervised head
class LabelpMHC(nn.Module):
    def __init__(self, cls_params):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(cls_params['dim_latent'], 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(256, cls_params['n_labels']) 
        )
    
    def forward(self, x):
        return self.classifier(x)


# GEX reconstruction loss, Mean Squared Logarithmic Error
class MSLE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, x_pred, x_true):
        log_pred = torch.log(x_pred+1)
        log_true = torch.log(x_true+1)
        loss = self.mse(log_pred, log_true)
        return loss
