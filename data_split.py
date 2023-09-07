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

def split_cvt(dataset, directory, num_val, num_test):
    idx_train, idx_val, idx_test = data_split(dataset, num_val, num_test)

    np.savetxt(directory+'traincand', idx_train, fmt='%d', delimiter=' ', newline='\n')
    np.savetxt(directory+'val', idx_val, fmt='%d', delimiter=' ', newline='\n')
    np.savetxt(directory+'test', idx_test, fmt='%d', delimiter=' ', newline='\n')


def select_nc(dataset, directory, nc_num):
    idx_train = np.loadtxt("splited_data/"+dataset+"/traincand", dtype=int)
    idx_nc = init_category_nc(dataset, idx_train, nc_num)
    np.savetxt(directory+'nc', idx_nc, fmt='%d', delimiter=' ', newline='\n')

def select_ad(dataset, directory, ad_num):
    idx_train = np.loadtxt("splited_data/"+dataset+"/traincand", dtype=int)
    idx_nc = np.loadtxt("splited_data/"+dataset+"/nc", dtype=int)
    idx_traincand = np.setdiff1d(idx_train,idx_nc) # no duplicate elements between idx_nc and ad

    idx_ad = np.random.choice(idx_traincand, size=ad_num*2, replace=False)
    random.shuffle(idx_ad)
    np.savetxt(directory+'ad_'+str(ad_num), idx_ad, fmt='%d', delimiter=' ', newline='\n')

    return idx_ad

# if __name__ == '__main__':
#     seed = 1
#     np.random.seed(seed)
#     random.seed(seed)

#     num_val = 500
#     num_test = 1000
    
#     for dataset in ['cora', 'citeseer', 'pubmed', 'BlogCatalog', 'Flickr', 'AmazonComputers', 'AmazonPhoto']:

#         directory = "splited_data/" + dataset + "/"

#         if not os.path.exists(directory):
#             os.makedirs(directory)

#         split_cvt(dataset, directory, num_val, num_test)

#         # sample 20k nodes for node classification
#         nc_num = 20
#         select_nc(dataset, directory, nc_num)


if __name__ == '__main__':
    seed = 1
    np.random.seed(seed)
    random.seed(seed)

    for dataset in ['cora', 'citeseer', 'pubmed', 'BlogCatalog', 'Flickr', 'AmazonComputers', 'AmazonPhoto']:
        directory = "splited_data/" + dataset + "/"

        data = sio.loadmat("./dataset/{}.mat".format(dataset))
        label = data['Label'] if ('Label' in data) else data['gnd']
        
        print(dataset)
        
        # sample nodes for anomaly detection
        for ad_num in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            idx_ad = select_ad(dataset, directory, ad_num)
            
            while label[idx_ad].sum() == 0:
                idx_ad = select_ad(dataset, directory, ad_num)
            
            print(label[idx_ad].sum())