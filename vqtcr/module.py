import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TCREncoder(nn.Module):
    def __init__(self, params, hdim, num_seq_labels):
        """
        :param params: hyperparameters as dict
        :param hdim: output feature dimension
        :param num_seq_labels: number of aa labels, input dim, 24个
        """
        super().__init__()
        self.params = params

        self.num_seq_labels = num_seq_labels

        self.embedding = nn.Embedding(num_seq_labels, 
                                      params['dim_emb'], 
                                      padding_idx=0)
        self.positional_encoding = PositionalEncoding(params['dim_emb'],
                                                      params['dropout'],
                                                      params['max_tcr_length'])

        encoding_layers = nn.TransformerEncoderLayer(params['dim_emb'],
                                                     params['num_heads'],
                                                     params['dim_emb'] * params['forward_expansion'],
                                                     params['dropout'])
        self.transformer_encoder = nn.TransformerEncoder(encoding_layers, params['num_encoder'])

        self.fc_reduction = nn.Linear(params['max_tcr_length'] * params['dim_emb'], hdim)

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.num_seq_labels)
        x = x.transpose(0, 1)
        x = x + self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)
        x = x.flatten(1)
        x = self.fc_reduction(x)
        return x


class TCRDecoder(nn.Module):
    def __init__(self, params, hdim, num_seq_labels):
        """
        :param params: hyperparameters as dict
        :param hdim: input feature dimension
        :param num_seq_labels: number of aa labels, output dim
        """
        super().__init__()
        self.params = params
        self.hdim = hdim
        self.num_seq_labels = num_seq_labels

        self.fc_upsample = nn.Linear(hdim, self.params['max_tcr_length'] * params['dim_emb'])

        # the embedding size remains constant over all layers
        self.embedding = nn.Embedding(num_seq_labels, 
                                      params['dim_emb'], 
                                      padding_idx=0)
        self.positional_encoding = PositionalEncoding(params['dim_emb'],
                                                      params['dropout'],
                                                      self.params['max_tcr_length'])

        decoding_layers = nn.TransformerDecoderLayer(params['dim_emb'],
                                                     params['num_heads'],
                                                     params['dim_emb'] * params['forward_expansion'],
                                                     params['dropout'])
        self.transformer_decoder = nn.TransformerDecoder(decoding_layers, params['num_decoder'])

        self.fc_out = nn.Linear(params['dim_emb'], num_seq_labels)

    def forward(self, hidden_state, target_sequence):
        """
        Forward pass of the Decoder module
        :param hidden_state: joint hidden state of the VAE
        :param target_sequence: Ground truth output
        :return:
        """
        hidden_state = self.fc_upsample(hidden_state)
        shape = (hidden_state.shape[0], self.params['max_tcr_length'], self.params['dim_emb'])
        hidden_state = torch.reshape(hidden_state, shape)

        hidden_state = hidden_state.transpose(0, 1)

        target_sequence = target_sequence[:, :-1]
        target_sequence = target_sequence.transpose(0, 1)

        target_sequence = self.embedding(target_sequence) * math.sqrt(self.num_seq_labels)
        target_sequence = target_sequence + self.positional_encoding(target_sequence)
        try:
            target_mask = nn.Transformer.generate_square_subsequent_mask(None, target_sequence.shape[0]).to(hidden_state.device)
        except:  # new version don't need the None
            target_mask = nn.Transformer.generate_square_subsequent_mask(target_sequence.shape[0]).to(hidden_state.device)
        x = self.transformer_decoder(target_sequence, hidden_state, tgt_mask=target_mask)
        x = self.fc_out(x)
        x = x.transpose(0, 1)
        return x


class RNAEncoder(nn.Module):
    def __init__(self, raw_dim=3000, latent_dim=32):
        super().__init__()
        self.in_dim = raw_dim
        self.latent_dim = latent_dim
        
        self.enc0 = nn.Sequential(nn.Linear(raw_dim, 1024), 
                                 nn.BatchNorm1d(1024), 
                                 nn.ELU())
        self.enc1 = nn.Sequential(nn.Linear(1024, 512), 
                                  nn.BatchNorm1d(512),
                                  nn.ELU())
        self.enc2 = nn.Sequential(nn.Linear(512, 256), 
                                  nn.BatchNorm1d(256),
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


class RNADecoder(nn.Module):
    def __init__(self, raw_dim=3000, latent_dim=32):
        super().__init__()
        self.in_dim = raw_dim
        self.latent_dim = latent_dim
        
        self.dec0 = nn.Sequential(nn.Linear(latent_dim, 256), 
                                 nn.BatchNorm1d(256), 
                                 nn.ELU())
        self.dec1 = nn.Sequential(nn.Linear(256, 512), 
                                  nn.BatchNorm1d(512),
                                  nn.ELU())
        self.dec2 = nn.Sequential(nn.Linear(512, 1024), 
                                  nn.BatchNorm1d(1024),
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


class ExponentialMovingAverage(nn.Module):
    """Maintains an exponential moving average for a value.
    
      This module keeps track of a hidden exponential moving average that is
      initialized as a vector of zeros which is then normalized to give the average.
      This gives us a moving average which isn't biased towards either zero or the
      initial value. Reference (https://arxiv.org/pdf/1412.6980.pdf)
      
      Initially:
          hidden_0 = 0
      Then iteratively:
          hidden_i = hidden_{i-1} - (hidden_{i-1} - value) * (1 - decay)
          average_i = hidden_i / (1 - decay^i)
    """
    
    def __init__(self, init_value, decay):
        super().__init__()
        
        self.decay = decay
        self.counter = 0
        self.register_buffer("hidden", torch.zeros_like(init_value))
        
    def forward(self, value):
        self.counter += 1
        self.hidden.sub_((self.hidden - value) * (1 - self.decay))
        average = self.hidden / (1 - self.decay ** self.counter)
        return average


class VQEMA(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, commitment_cost, decay,
               epsilon=1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.epsilon = epsilon
        
        # initialize embeddings as buffers
        embeddings = torch.empty(self.num_embeddings, self.embedding_dim)
        nn.init.kaiming_uniform_(embeddings)
        self.register_buffer("embeddings", embeddings)
        self.ema_dw = ExponentialMovingAverage(self.embeddings, decay)
        
        # also maintain ema_cluster_size， which record the size of each embedding
        self.ema_cluster_size = ExponentialMovingAverage(torch.zeros((self.num_embeddings,)), decay)
        
    def forward(self, x):
        flat_x = x.reshape(-1, self.embedding_dim)
        
        # Use index to find embeddings in the latent space
        encoding_indices = self.get_code_indices(flat_x)
        quantized = self.quantize(encoding_indices)
        quantized = quantized.view_as(x) 
        
        #EMA
        with torch.no_grad():
            encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
            updated_ema_cluster_size = self.ema_cluster_size(torch.sum(encodings, dim=0))
            n = torch.sum(updated_ema_cluster_size)
            updated_ema_cluster_size = ((updated_ema_cluster_size + self.epsilon) /
                                      (n + self.num_embeddings * self.epsilon) * n)
            dw = torch.matmul(encodings.t(), flat_x) # sum encoding vectors of each cluster
            updated_ema_dw = self.ema_dw(dw)
            normalised_updated_ema_w = (
              updated_ema_dw / updated_ema_cluster_size.reshape(-1, 1))
            self.embeddings.data = normalised_updated_ema_w
    
        # commitment loss
        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = x + (quantized - x).detach()
    
        return quantized, loss
    
    def get_code_indices(self, flat_x):
        # compute L2 distance
        distances = (
            torch.sum(flat_x ** 2, dim=1, keepdim=True) +
            torch.sum(self.embeddings ** 2, dim=1) -
            2. * torch.matmul(flat_x, self.embeddings.t())
        ) # [N, M]
        encoding_indices = torch.argmin(distances, dim=1) # [N,]
        return encoding_indices
    
    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return F.embedding(encoding_indices, self.embeddings)


class LabelCLS(nn.Module):
    def __init__(self, cls_params):
        super().__init__()
        self.lin1 = nn.Linear(cls_params['dim_latent'], 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.act = nn.ELU()
        self.dropout = nn.Dropout(cls_params['dropout'])
        self.lin2 = nn.Linear(16, cls_params['n_labels'])
    
    def forward(self, x):
        return self.lin2(self.dropout(self.act(self.bn1(self.lin1(x)))))
        