def get_pred_config(dataset='', 
                batch_size=512, 
                latent_dim=64,
                dropout=0.2,
                lr=1e-4,
                early_stopping=10,
                save_path='',
                direction='max',
                epoch=100,
                weight_decay=1e-4,
                loss_weight=[1.0, 1.0, 1.0, 1.0], 
                kmer_kernals=[1,3,5,7], 
                num_subspace=3,
                codewords_per_space=64, 
                beta_only=False,
                n_labels=8,
                dead_code_interval=1000):
    
    params = {
            'dim_latent': latent_dim,           # dim of latent space
            'batch_size': batch_size,           # batch size
            'lr': lr,                           # learning rate
            'dropout': dropout,                 # dropout rate
            'epoch': epoch,                     # number of epochs
            'early': early_stopping,            # waiting steps for early stopping
            'dataset': dataset,                 # dataset name
            'save_path': save_path,             # save path
            'direction': direction,             # maximize or minimize for optimization
            'weight_decay': weight_decay,       # weight decay
            'loss_weight': loss_weight,         # loss weight for each task
            
            
            'gex': {
                'dim_latent': latent_dim,       # dim of rna latent
                'act': 'elu',                   # activation function
                'norm': True,                   # normalization
            },
            
            'tcr': {
                'esm_type': "facebook/esm2_t6_8M_UR50D",    # esm model type (only use aa tokens)
                'kmer_kernels': kmer_kernals,   # kmer kernels for tcr (list)
                'dim_latent': latent_dim,       # dim of tcr latent
                'dim_emb': latent_dim,          # dim of aa embedding
                'num_heads': 8,                 # number of attention heads
                'forward_expansion': 4,         # expansion factor for feedforward layer
                'dropout': dropout,             # dropout rate
                'num_encoder': 2,               # number of encoder layers
                'num_decoder': 2,               # number of decoder layers
                'mask_rate': 0.15,              # mask rate for tcr
                'beta_only': beta_only,         # only use beta chains
                'esm_tokens': True,             # use ESM pretrained tokens
            },
            
            'fus':{
                'dim_latent': latent_dim,       # dim of fusion latent
                'dropout': dropout,             # dropout rate
                'film_reg': 1e-0,               # film regularization
            },
            
            'pty':{
                'dim_latent': 2 * latent_dim,        # total dim of latent
                'num_subspace': num_subspace,
                'codewords_per_space': codewords_per_space,
                'eps': 1e-5,
                'init_scale': 1.0,
                'ema_decay': 0.99,
                'commitment_beta': 0.25,
                'use_cosine': True,
                'dead_code_interval': dead_code_interval,
                'dead_code_threshold': 1e-4,
                'dead_code_topk': 16,
                'diversity_weight': 1e-3
            },
            
            'pred':{
                'dim_latent': latent_dim,       # dim of fusion latent
                'dropout': dropout,             # dropout rate
                'n_labels': n_labels,           # number of labels to predict
            }
            
            }
    return params