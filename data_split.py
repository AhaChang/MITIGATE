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

    idx_init = init_category(2, np.array(idx_train), labels).tolist()
    idx_train = list(set(idx_train) - set(idx_init))
    random.shuffle(idx_train)

    
    
    return idx_init, idx_train, idx_val, idx_test


def init_category_nc(dataset_str, number):
    # Load dataset
    data = sio.loadmat(f'dataset/{dataset_str}.mat')
    labels = np.squeeze(np.array(data['Class'],dtype=np.int64))
    ano_labels = data['Label'] if ('Label' in data) else data['gnd']

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

    return random_positions_list


if __name__ == '__main__':
    seed = 1
    np.random.seed(seed)
    random.seed(seed)

    num_val = 500
    num_test = 1000
    
    for dataset in ['cora', 'citeseer', 'pubmed', 'BlogCatalog', 'Flickr', 'AmazonComputers', 'AmazonPhoto']:
        idx_init, idx_train, idx_val, idx_test = data_split(dataset, num_val, num_test)

        directory = "splited_data/" + dataset + "/"

        if not os.path.exists(directory):
            os.makedirs(directory)

        np.savetxt(directory+'init', idx_init, fmt='%d', delimiter=' ', newline='\n')
        np.savetxt(directory+'traincand', idx_train, fmt='%d', delimiter=' ', newline='\n')
        np.savetxt(directory+'val', idx_val, fmt='%d', delimiter=' ', newline='\n')
        np.savetxt(directory+'test', idx_test, fmt='%d', delimiter=' ', newline='\n')


    nc_num = 20 # sample 20k nodes

    for dataset in ['cora', 'citeseer', 'pubmed', 'BlogCatalog', 'Flickr', 'AmazonComputers', 'AmazonPhoto']:
        idx_nc = init_category_nc(dataset, nc_num)

        directory = "splited_data/" + dataset + "/"

        if not os.path.exists(directory):
            os.makedirs(directory)

        np.savetxt(directory+'nc', idx_nc, fmt='%d', delimiter=' ', newline='\n')

    
    k=20
    init_num = 2
    for dataset in ['cora', 'citeseer', 'pubmed', 'BlogCatalog', 'Flickr', 'AmazonComputers', 'AmazonPhoto']:

        idx_init_ab = np.loadtxt("splited_data/"+dataset+"/init", dtype=int)
        idx_train = np.loadtxt("splited_data/"+dataset+"/traincand", dtype=int)
        idx_selected_ab = np.random.choice(idx_train, size=2*(k-init_num), replace=False)
        idx_ab = np.hstack((idx_selected_ab, idx_init_ab)).tolist()
        random.shuffle(idx_ab)

        directory = "splited_data/" + dataset + "/"

        if not os.path.exists(directory):
            os.makedirs(directory)

        np.savetxt(directory+'ad', idx_ab, fmt='%d', delimiter=' ', newline='\n')