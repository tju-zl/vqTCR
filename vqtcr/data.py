import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
import random
from scipy import sparse


def get_dataloader(adata, batch_size=512, data_mode='train', sample_mode='clonotype',
                   metadata = None, labels = None, conditional = None):
    """
    mode: 'train' to splite train and eval dataloader, 'eval' return all dataloader.
    sample_mode: using 'clonotype' of other labels to make sampling balance.
    metadata: specific any data in adata.obs.
    labels: for semi-supervised prediction of pMHC (antigen binder information).
    conditional: saved property for atlas mapping.
    """
    
    if data_mode == 'train':
        train_mask = (adata.obs['set'] == 'train').values
    elif data_mode == 'eval':
        train_mask = np.ones(shape=(len(adata), ), dtype=bool)
    
    # RNA splite
    rna_train = adata.X[train_mask]
    rna_val = adata.X[~train_mask]
    
    # TCR splite
    tcr_seq = np.concatenate([adata.obsm['alpha_seq'], adata.obsm['beta_seq']], axis=1)
    tcr_length = np.vstack([adata.obs['alpha_len'], adata.obs['beta_len']]).T
    tcr_train = tcr_seq[train_mask]
    tcr_val = tcr_seq[~train_mask]
    tcr_length_train = tcr_length[train_mask].tolist()
    tcr_length_val = tcr_length[~train_mask].tolist()
    
    # metadata
    metadata_train = adata.obs[metadata][train_mask].to_numpy()
    metadata_val = adata.obs[metadata][~train_mask].to_numpy()
    
    # labels
    if labels is not None:
        labels = adata.obs[labels].cat.codes.to_numpy()
        labels_train = labels[train_mask]
        labels_val = labels[~train_mask]
    else:
        labels_train = None
        labels_val = None
    
    if conditional is not None:
        conditional_train = adata.obsm[conditional][train_mask]
        conditional_val = adata.obsm[conditional][~train_mask]
    else:
        conditional_train = None
        conditional_val = None

    # generate tensor dataset
    train_dataset = ImmuneDataset(rna_train, tcr_train, tcr_length_train, metadata_train,
                                labels_train, conditional_train)
    val_dataset = ImmuneDataset(rna_val, tcr_val, tcr_length_val, metadata_val,
                                labels_val, conditional_val)

    # all data evaluation
    if data_mode == 'eval':
        eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        # sample_weight = balance_sampling(adata, train_mask, key_name=sample_mode)
        # sampler = WeightedRandomSampler(sample_weight, len(sample_weight))
        # eval_loader = DataLoader(train_dataset, batch_size=batch_size, 
        #                           shuffle=False, sampler=sampler,
        #                           worker_init_fn=seed_worker)
        return eval_loader
    
    # balance sampling for clonotype
    if sample_mode is not None:
        sample_weight = balance_sampling(adata, train_mask, key_name=sample_mode)
        sampler = WeightedRandomSampler(sample_weight, len(sample_weight))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                  shuffle=False, sampler=sampler,
                                  worker_init_fn=seed_worker)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                  shuffle=True, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


class ImmuneDataset(Dataset):
    def __init__(self, rna, tcr, tcr_length, metadata, 
                 labels=None, conditional=None):
        self.rna_data = self.to_tensor(rna)
        self.tcr_data = torch.LongTensor(tcr)
        self.tcr_length = torch.LongTensor(tcr_length)
        self.metadata = metadata.tolist()

        if labels is not None:
            self.labels = torch.LongTensor(labels)
        else:
            self.labels = None

        if conditional is not None:
            self.conditional = torch.LongTensor(conditional.argmax(1))
        else:
            self.conditional = None
        
    def to_tensor(self, x):
       if sparse.issparse(x):
           return torch.FloatTensor(x.todense())
       else:
           return torch.FloatTensor(x)
       
    def __len__(self):
        return len(self.rna_data)
    
    def __getitem__(self, index):
        if self.labels is None:
            if self.conditional is None:
                return self.rna_data[index], self.tcr_data[index],\
                    self.tcr_length[index], self.metadata[index], False, False
            else:
                return self.rna_data[index], self.tcr_data[index],\
                    self.tcr_length[index], self.metadata[index], \
                    False, self.conditional[index]
        else:
            if self.conditional is None:
                return self.rna_data[index], self.tcr_data[index],\
                    self.tcr_length[index], self.metadata[index], \
                        self.labels[index], False
            else:
                return self.rna_data[index], self.tcr_data[index],\
                    self.tcr_length[index], self.metadata[index],\
                    self.labels[index]


def balance_sampling(adata, train_mask, key_name):
    key_counts = []
    key_count = adata[train_mask].obs[key_name].map(adata[train_mask].obs[key_name].value_counts())
    key_counts.append(key_count)
    
    key_counts = pd.concat(key_counts, ignore_index=True)
    key_counts = np.log(key_counts/10+1)
    key_counts = 1/key_counts
    weights = key_counts/sum(key_counts)
    return weights


def seed_worker():
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
