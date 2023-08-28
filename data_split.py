import numpy as np
import scipy.io as sio
import random
import os
from sampling_methods import init_category

def data_split(dataset_str, num_val, num_test):
    # Load dataset
    data = sio.loadmat(f'dataset/{dataset_str}.mat')
    labels = data['Label'] if ('Label' in data) else data['gnd']

    num_nodes = len(labels)

    # Get abnormal and normal indices
    idx_ab, idx_nor = np.where(labels==1)[0].tolist(), np.where(labels==0)[0].tolist()

    # Shuffle indices 
    for list_ in [idx_ab, idx_nor]: random.shuffle(list_)

    # Calculate the abnormal ratio
    ab_ratio = labels.sum() / num_nodes

    # Number of validation and test data
    num_val_ab = int(num_val * ab_ratio)
    num_test_ab = int(num_test * ab_ratio)

    # Obtain validation indices and test
    idx_val = idx_ab[:num_val_ab] + idx_nor[:num_val-num_val_ab]
    idx_test = idx_ab[num_val_ab:num_val_ab+num_test_ab] + idx_nor[num_val-num_val_ab:num_val-num_val_ab+num_test-num_test_ab]
    idx_train = idx_ab[num_val_ab+num_test_ab:] + idx_nor[num_val-num_val_ab+num_test-num_test_ab:]

    # idx_init = init_category(2, np.array(idx_train), labels).tolist()
    # idx_train = list(set(idx_train) - set(idx_init))
    # random.shuffle(idx_train)

    return idx_train, idx_val, idx_test


def init_category_nc(dataset_str, idx_train, number):
    # Load dataset
    data = sio.loadmat(f'dataset/{dataset_str}.mat')
    labels = np.squeeze(np.array(data['Class'],dtype=np.int64))[idx_train]
    ano_labels = data['Label'] if ('Label' in data) else data['gnd']
    ano_labels = ano_labels[idx_train]

    label_positions = {}

    for i, label in enumerate(labels):
        if ano_labels[i]==0:
            if label.item() not in label_positions:
                label_positions[label.item()] = []
            label_positions[label.item()].append(i)

    random_positions_list = []
    for key, val in label_positions.items():
        if len(val) >= number:
            random_positions_list.extend(random.sample(val, number))
    
    random.shuffle(random_positions_list)

    return np.array(idx_train)[random_positions_list]


if __name__ == '__main__':
    seed = 2
    np.random.seed(seed)
    random.seed(seed)

    num_val = 500
    num_test = 1000
    
    for dataset in ['cora', 'citeseer', 'pubmed', 'BlogCatalog', 'Flickr', 'AmazonComputers', 'AmazonPhoto']:
        # Load dataset
        data = sio.loadmat(f'dataset/{dataset}.mat')
        labels = data['Label'] if ('Label' in data) else data['gnd']

        idx_train, idx_val, idx_test = data_split(dataset, num_val, num_test)

        directory = "splited_data/" + dataset + "/"

        if not os.path.exists(directory):
            os.makedirs(directory)

        np.savetxt(directory+'traincand', idx_train, fmt='%d', delimiter=' ', newline='\n')
        np.savetxt(directory+'val', idx_val, fmt='%d', delimiter=' ', newline='\n')
        np.savetxt(directory+'test', idx_test, fmt='%d', delimiter=' ', newline='\n')


        nc_num = 20 # sample 20k nodes for node classification

        idx_nc = init_category_nc(dataset, idx_train, nc_num)

        np.savetxt(directory+'nc', idx_nc, fmt='%d', delimiter=' ', newline='\n')

        idx_traincand = np.setdiff1d(idx_train,idx_nc) # no duplicate elements between idx_nc and ad


        ad_num = 20 # sample 20k nodes for anomaly detection
        anomaly_ratio = labels[idx_train].sum() / labels[idx_train].shape[0]

        # selected_a = np.random.choice(np.where(labels[idx_traincand]==1)[0], size=int(anomaly_ratio*ad_num*2), replace=False)
        # selected_n = np.random.choice(np.where(labels[idx_traincand]==0)[0], size=ad_num*2-int(anomaly_ratio*ad_num*2), replace=False)
        # idx_ad = idx_traincand[np.hstack((selected_a, selected_n))].tolist()

        idx_ad = np.random.choice(idx_traincand, size=ad_num*2, replace=False)

        random.shuffle(idx_ad)
        print(labels[idx_ad].sum())

        np.savetxt(directory+'ad', idx_ad, fmt='%d', delimiter=' ', newline='\n')