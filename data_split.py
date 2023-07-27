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


    return idx_train, idx_val, idx_test





if __name__ == '__main__':
    seed = 1
    np.random.seed(seed)
    random.seed(seed)

    num_val = 500
    num_test = 1000
    
    for dataset in ['cora', 'citeseer', 'pubmed', 'BlogCatalog', 'Flickr']:
        idx_train, idx_val, idx_test = data_split(dataset, num_val, num_test)

        directory = "splited_data/" + dataset + "/"

        if not os.path.exists(directory):
            os.makedirs(directory)

        np.savetxt(directory+'traincand', idx_train, fmt='%d', delimiter=' ', newline='\n')
        np.savetxt(directory+'val', idx_val, fmt='%d', delimiter=' ', newline='\n')
        np.savetxt(directory+'test', idx_test, fmt='%d', delimiter=' ', newline='\n')