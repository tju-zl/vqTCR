def get_config():
    params = {
            'seed': 3407,
            'dim_latent': 64,
            'batch_size': 4096,
            'lr': 1e-3,
            'dropout': 0.3,
            'epoch': 100,
            
            'vq_layer': {
                'num_codebook': 128,
                'commitment': 0.25,
                'lambda': 10,
                'decay': 0.99,
            },
            
            'rna': {
                'dim_latent': 64,
                'hvgs': 3000,
                'act': 'elu',
                'batch_norm': True,
            },
            
            'tcr': {
                'dim_latent': 64,
                'dim_emb': 64,
                'num_heads': 8,
                'forward_expansion': 4,
                'dropout': 0.3,
                'num_encoder': 2,
                'num_decoder': 2, 
            },
            
            'cls': {
                'dim_latent': 64,
                'activation': 'elu',
                'batch_norm': True,
                'dropout': 0.3,
                'n_labels': 8,
            }
        }
    
    return params

# 10X high affinity binder
HIGH_COUNT_ANTIGENS = ['A0201_ELAGIGILTV_MART-1_Cancer_binder',
					   'A0201_GILGFVFTL_Flu-MP_Influenza_binder',
					   'A0201_GLCTLVAML_BMLF1_EBV_binder',
					   'A0301_KLGGALQAK_IE-1_CMV_binder',
					   'A0301_RLRAEAQVK_EMNA-3A_EBV_binder',
					   'A1101_IVTDFSVIK_EBNA-3B_EBV_binder',
					   'A1101_AVFDRKSDAK_EBNA-3B_EBV_binder',
					   'B0801_RAKFKQLL_BZLF1_EBV_binder']
