import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
from sklearn.metrics import davies_bouldin_score, silhouette_score, classification_report
from scipy import stats


# evaluate the bingding prediction performace with unsup representations
def get_knn_cls(data_train, data_eval, label_train, label_eval):
    clf = KNeighborsClassifier(n_neighbors=5, weights='distance')
    clf.fit(data_train, label_train)
    label_pred = clf.predict(data_eval)
    report = classification_report(label_eval, label_pred, output_dict=True)
    return report


# calculate the silhouette score as internal cluster evaluation
def get_silhouette_score(rep, labels):
    try:
        score = silhouette_score(rep, labels, metric='euclidean', random_state=0)
    except:
        score = -99
    return score


# calculates the AMI scores sa external cluster evaluation
def get_adjusted_mutual_information(labels_true, labels_pred):
    scores = adjusted_mutual_info_score(labels_true, labels_pred)
    return scores


# calculates the NMI scores sa external cluster evaluation
def get_normalized_mutual_information(labels_true, labels_pred):
    scores = normalized_mutual_info_score(labels_true, labels_pred, average_method='arithmetic')
    return scores


# calculates the ARI scores sa external cluster evaluation
def get_adjusted_random_score(labels_true, labels_pred):
    scores = adjusted_rand_score(labels_true, labels_pred)
    return scores


def get_knn_f1_within_set(latent, column_name):
    con = latent.obsp['connectivities'].A.astype(bool)
    nearest_neighbor_label = [latent.obs[column_name].values[row].tolist()[0] for row in con]
    labels_true = latent.obs[column_name].values
    if torch.is_tensor(nearest_neighbor_label[0]):
        nearest_neighbor_label = [str(el.item()) for el in nearest_neighbor_label]
        labels_true = [str(el.item()) for el in labels_true]
    result = classification_report(labels_true, nearest_neighbor_label, output_dict=True)
    result = result['weighted avg']['f1-score']
    return result


def get_model_prediction_func(model, do_adata=False, metadata=None):
    def prediction_function(data):
        metadata_tmp = metadata if metadata is not None else []
        latent_space = model.get_latent(data, metadata=metadata_tmp, return_mean=True)
        if do_adata:
            return latent_space
        latent_space = latent_space.X
        return latent_space
    return prediction_function

