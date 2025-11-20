"""
dataset preprocessing
1. prepare the data for training and evaluation
2. split the dataset into train and eval set
3. load the dataset
"""
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
import random
from scipy import sparse


# --------------split train-valid-test and add .obs.set to adata--------------
from sklearn.model_selection import GroupShuffleSplit
def split_dataset(adata, split_ratio, split_col=None, random_seed=42):
    groups =  adata.obs[split_col]
    spliter = GroupShuffleSplit(test_size=split_ratio,
                                n_splits=5,
                                random_state=random_seed)
    best_value = 1
    train, val = None, None
    for train_tmp, val_tmp in spliter.split(adata, groups=groups):
        split_value = abs(len(val_tmp)/len(adata) - split_ratio)
        if split_value < best_value:
            train = train_tmp
            val = val_tmp
            best_value = split_value
    train = adata[train]
    val = adata[val]
    return train, val


# add train-valid-test label (3:1:1)
def split_adata(adata, split_col=None, random_seed=42):
    train_val, test = split_dataset(adata, 0.20, split_col, random_seed)
    train, val = split_dataset(train_val, 0.25, split_col, random_seed)
    
    adata.obs['set'] = None
    adata.obs.loc[train.obs.index, 'set'] = 'train'
    adata.obs.loc[val.obs.index, 'set'] = 'valid'
    adata.obs.loc[test.obs.index, 'set'] = 'test'
    adata = adata[adata.obs['set'].isin(['train', 'valid', 'test'])]
    return adata


# ---------------split train-valid-test within groups----------------
from tqdm import tqdm
def stratified_group_shuffle_split(df, stratify_col, group_col, val_split, random_seed=42):
	"""
	https://stackoverflow.com/a/63706321
	Split the dataset into train and test. To create a val set, execute this code twice to first split train+val and test
	and then split the train and val.

	The splitting tries to improve splitting by two properties:
	1) Stratified splitting, so the label distribution is roughly the same in both sets, e.g. antigen specificity
	2) Certain groups are only in one set, e.g. the same clonotypes are only in one set, so the model cannot peak into similar sample during training.

	If there is only one group to a label, the group is defined as training, else as test sample, the model never saw this label before.

	The outcome is not always ideal, i.e. the label distribution may not , as the labels within a group is heterogeneous (e.g. 2 cells from the same clonotype have different antigen labels)
	Also see here for the challenges: https://github.com/scikit-learn/scikit-learn/issues/12076

	:param df: pd.DataFrame containing the data to split
	:param stratify_col: str key for the column containing the classes to be stratified over all sets
	:param group_col: str key for the column containing the groups to be kept in the same set
	"""
	groups = df.groupby(stratify_col)
	all_train = []
	all_test = []
	for id, group in tqdm(groups):
		# if a group is already taken in test or train it must stay there
		group = group[~group[group_col].isin(all_train + all_test)]
		# if group is empty
		if group.shape[0] == 0:
			continue

		if len(group) > 1:
			train_inds, test_inds = next(
				GroupShuffleSplit(test_size=val_split, n_splits=1, random_state=random_seed).split(group, groups=group[
					group_col]))
			all_train += group.iloc[train_inds][group_col].tolist()
			all_test += group.iloc[test_inds][group_col].tolist()
		# if there is only one clonotype for this particular label
		else:
			all_train += group[group_col].tolist()

	train = df[df[group_col].isin(all_train)]
	test = df[df[group_col].isin(all_test)]

	return train, test


# --------------prepare dataset and dataloader for pytorch framework--------------
class TCellDataset(Dataset):
    def __init__(self, gex, tcr, labels=None):
        self.gex_data = self.to_tensor(gex)
        self.tcr_data = torch.LongTensor(tcr)

        if labels is not None:
            self.labels = torch.LongTensor(labels)
        else:
            self.labels = None
        
    def to_tensor(self, x):
       if sparse.issparse(x):
           return torch.FloatTensor(x.todense())
       else:
           return torch.FloatTensor(x)
       
    def __len__(self):
        return len(self.gex_data)
    
    def __getitem__(self, index):
        if self.labels is None:
            return self.gex_data[index], self.tcr_data[index], False
        else:
            return self.gex_data[index], self.tcr_data[index], self.labels[index]
                

def get_dataset(adata, col_name='set', set_name=['train'], label_name = None):
    """
    set_name: `train`, `valid` and `test`.
    labels: for supervised prediction of pMHC (antigen binder information).
    """
    mask = (adata.obs[col_name].isin(set_name)).values
    gex_data = adata.X[mask]
    tcr_seq = np.concatenate([adata.obsm['alpha_seq'], adata.obsm['beta_seq']], axis=1)
    tcr_data = tcr_seq[mask]
    
    if label_name is not None:
        all_labels = adata.obs[label_name].cat.codes.to_numpy()
        labels = all_labels[mask]
    else:
        labels = None
    
    dataset = TCellDataset(gex_data, tcr_data, labels)

    return dataset, mask


# we use balance sampling strategy from mvTCR (https://github.com/SchubertLab/mvTCR/tree/master) 
# to highlight the rare or non-activited clonotypes, while some methods don't use balance sampling.
def balance_sampling(adata, mask, key_name):
    key_counts = []
    key_count = adata[mask].obs[key_name].map(adata[mask].obs[key_name].value_counts())
    key_counts.append(key_count)
    
    key_counts = pd.concat(key_counts, ignore_index=True)
    key_counts = np.log(key_counts/10+1)
    key_counts = 1/key_counts
    weights = key_counts/sum(key_counts)
    return weights

# standard balance sampling for unbalanced data
def standard_balance_sampling(adata, train_mask, key_name, epsilon=1e-4):
    class_counts = adata[train_mask].obs[key_name].value_counts()
    
    # Smooth inverse frequency: 1 / (num_sample + epsilon)
    sample_weights = adata[train_mask].obs[key_name].map(1 / np.log1p(class_counts + epsilon))
    
    return sample_weights.to_numpy()


def seed_worker():
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# if use all data, set obs['set']='train' befor get dataloader; 
# if lo task, col_name='donor'or'patient', set_name='patient_id'
def get_dataloader(adata, batch_size=512, col_name='set', set_name=['train'], sample_mode='clonotype', 
                   shuffle=False, labels = None):
    dataset, mask = get_dataset(adata, col_name, set_name, labels)
    if sample_mode is not None:
        sample_weight = balance_sampling(adata, mask, key_name=sample_mode)
        sampler = WeightedRandomSampler(sample_weight, len(sample_weight))
        data_loader = DataLoader(dataset, batch_size=batch_size, 
                                  shuffle=False, sampler=sampler,
                                  worker_init_fn=seed_worker)
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, 
                                  shuffle=shuffle, sampler=None,
                                  worker_init_fn=None)
    return data_loader


