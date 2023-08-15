import numpy as np
import random 
from heapq import nlargest, nsmallest
import torch
import torch.nn.functional as F
from time import perf_counter
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

def init_category(number, nodes_idx, labels):
    label_positions = {}

    for i, label in enumerate(labels[nodes_idx]):
        if label.item() not in label_positions:
            label_positions[label.item()] = []
        label_positions[label.item()].append(i)

    random_positions_list = []
    for key, val in label_positions.items():
        if len(val) >= number:
            random_positions_list.extend(random.sample(val, number))
    
    random.shuffle(random_positions_list)

    return nodes_idx[random_positions_list]

def init_category_nc(number, nodes_idx, labels, ano_labels):
    label_positions = {}

    for i, label in enumerate(labels[nodes_idx]):
        if ano_labels[nodes_idx[i]]==0:
            if label.item() not in label_positions:
                label_positions[label.item()] = []
            label_positions[label.item()].append(i)

    random_positions_list = []
    for key, val in label_positions.items():
        if len(val) >= number:
            random_positions_list.extend(random.sample(val, number))
    
    random.shuffle(random_positions_list)

    return nodes_idx[random_positions_list]

#calculate the percentage of elements larger than the k-th element
def percd(input,k): return sum([1 if i else 0 for i in input>input[k]])/float(len(input))

def get_entropy_score(output):
    prob_output = F.softmax(output, dim=1).detach()
    log_prob_output = F.log_softmax(output, dim=1).detach()
    entropy = -torch.sum(prob_output*log_prob_output, dim=1)
    return entropy

def get_density_score(features, nb_cluster):
    kmeans = KMeans(n_clusters=nb_cluster, random_state=0, n_init='auto').fit(features)
    ed=euclidean_distances(features,kmeans.cluster_centers_)
    ed_score = np.min(ed,axis=1)	#the larger ed_score is, the far that node is away from cluster centers, the less representativeness the node is
    edprec = torch.FloatTensor([percd(ed_score,i) for i in range(len(ed_score))]).cuda()
    return edprec


def query_random(number, nodes_idx):
    return np.random.choice(nodes_idx, size=number, replace=False)

def query_largest_degree(nx_graph, number, nodes_idx):
    degree_dict = dict(nx_graph.degree(nodes_idx))
    idx_topk = nlargest(number, degree_dict, key=degree_dict.get)
    # print(idx_topk)
    return idx_topk

def query_featprop(features, number, nodes_idx):
    features = features.cpu().numpy()
    X = features[nodes_idx]
    # print('X: ', X)
    t1 = perf_counter()
    distances = pairwise_distances(X, X)
    print('computer pairwise_distances: {}s'.format(perf_counter() - t1))
    clusters, medoids = k_medoids(distances, k=number)
    return np.array(nodes_idx)[medoids]

def query_entropy(prob, number, nodes_idx):
    output = prob[nodes_idx]
    entropy = get_entropy_score(output)
    indices = torch.topk(entropy, number, largest=True)[1]
    indices = list(indices.cpu().numpy())
    return np.array(nodes_idx)[indices]

def query_nc_minmax(prob, number, nodes_idx):
    max_scores = F.softmax(prob, dim=1)[nodes_idx].max(1)[0]
    indices = torch.topk(max_scores, number, largest=False)[1] # 最小的最大概率
    indices = list(indices.cpu().numpy())
    return np.array(nodes_idx)[indices]

def query_nc_entropy(prob, number, nodes_idx):
    output = prob[nodes_idx]
    entropy = get_entropy_score(output)
    indices = torch.topk(entropy, number, largest=True)[1]
    indices = list(indices.cpu().numpy())
    return np.array(nodes_idx)[indices]


def query_nc_ano(prob_nc, prob_ad, number, nodes_idx, val_res):
    # node classification label distribution
    output_nc = prob_nc[nodes_idx]
    entropy_nc = get_entropy_score(output_nc)
    entropy_nc_nor = (entropy_nc - entropy_nc.min())/(entropy_nc.max()-entropy_nc.min())
    # anomay detection anomaly score
    output_ad = prob_ad[nodes_idx]
    score_ad = F.softmax(output_ad, dim=1).detach()[:,-1]
    score_ad_nor = (score_ad - score_ad.min())/(score_ad.max()-score_ad.min())
    # 
    # if val_res<0.5:
    #     total_score = entropy_nc_nor
    # else:
    #     total_score = (1-val_res) * entropy_nc_nor + val_res * score_ad_nor
    total_score =(1-val_res) * entropy_nc_nor + val_res * score_ad_nor
    indices = torch.topk(total_score, number, largest=True)[1]
    indices = list(indices.cpu().numpy())
    return np.array(nodes_idx)[indices]


def query_density(embeds, number, nodes_idx, labels):
    unique_labels = torch.unique(labels)
    edprec = get_density_score(embeds[nodes_idx].cpu(), unique_labels.shape[0])
    indices = torch.topk(edprec, number, largest=True)[1]
    indices = list(indices.cpu().numpy())
    return np.array(nodes_idx)[indices]

def query_entropy_density(embeds, prob, number, nodes_idx, labels):
    unique_labels = torch.unique(labels)
    edprec = get_density_score(embeds[nodes_idx].cpu(), unique_labels.shape[0])
    entropy = get_entropy_score(prob[nodes_idx])
    finalweight = edprec + entropy
    indices = torch.topk(finalweight, number, largest=True)[1]
    indices = list(indices.cpu().numpy())
    return np.array(nodes_idx)[indices]

def query_featprop(features, number, nodes_idx):
    features = features.cpu().numpy()
    X = features[nodes_idx]
    distances = pairwise_distances(X, X)
    clusters, medoids = k_medoids(distances, k=number)
    return np.array(nodes_idx)[medoids]


def query_topk_anomaly(prob, number, nodes_idx):
    output = prob[nodes_idx]
    prob_output = F.softmax(output, dim=1).detach()
    indices = torch.topk(prob_output[:,-1], number, largest=True)[1]
    indices = list(indices.cpu().numpy())
    return np.array(nodes_idx)[indices]

def query_topk_medoids(embed, prob_ad, number, nodes_idx, nb_classes):
    embed = embed[nodes_idx].cpu().numpy()
    distances = pairwise_distances(embed, embed)
    clusters, medoids = k_medoids(distances, k=nb_classes*2)
    indices = torch.topk(prob_ad[:,-1][medoids], number, largest=True)[1]
    indices = list(indices.cpu().numpy())
    return medoids[indices]

def query_topk_medoids_m(embed, prob_ad, number, nodes_idx, nb_classes):
    embed = embed[nodes_idx].cpu().numpy()
    distances = pairwise_distances(embed, embed)
    clusters, medoids = k_medoids(distances, k=nb_classes+1)
    medoids_ano = medoids[prob_ad[:,-1][medoids].argmax()]
    indices = np.where(clusters==medoids_ano)[0]
    e = np.random.choice(indices, size=number, replace=False)
    # e = indices[torch.topk(prob_ad[np.array(nodes_idx)[indices]][:,1], number, largest=True)[1].cpu()]
    return np.array(nodes_idx)[e]

def return_community(number, nodes_idx, labels):
    unique_labels = torch.unique(labels)
    res = np.array([])
    k = number // unique_labels.shape[0]
    for label in unique_labels:
        label_node_idx = torch.LongTensor(nodes_idx)[torch.where(labels[nodes_idx]==label)[0]]
        res = np.hstack((res, np.random.choice(label_node_idx, size=k, replace=False)))
    return res

def return_anomaly(number, nodes_idx, ano_labels):
    ano_node_idx = torch.LongTensor(nodes_idx)[torch.where(ano_labels[nodes_idx]==1)[0]]
    nor_node_idx = torch.LongTensor(nodes_idx)[torch.where(ano_labels[nodes_idx]==0)[0]]
    return np.hstack((np.random.choice(ano_node_idx, size=number//2, replace=False),np.random.choice(nor_node_idx, size=number//2, replace=False)))

def k_medoids(distances, k=3):
    # From https://github.com/salspaugh/machine_learning/blob/master/clustering/kmedoids.py

    m = distances.shape[0] # number of points

    # Pick k random medoids.
    print('k: {}'.format(k))
    # curr_medoids = np.array([-1]*k)
    # while not len(np.unique(curr_medoids)) == k:
    #     curr_medoids = np.array([random.randint(0, m - 1) for _ in range(k)])
    curr_medoids = np.arange(m)
    np.random.shuffle(curr_medoids)
    curr_medoids = curr_medoids[:k]
    old_medoids = np.array([-1]*k) # Doesn't matter what we initialize these to.
    new_medoids = np.array([-1]*k)

    # Until the medoids stop updating, do the following:
    num_iter = 0
    while not ((old_medoids == curr_medoids).all()):
        num_iter += 1
        # print('curr_medoids: ', curr_medoids)
        # print('old_medoids: ', old_medoids)
        # Assign each point to cluster with closest medoid.
        t1 = perf_counter()
        clusters = assign_points_to_clusters(curr_medoids, distances)
        # print(f'clusters: {clusters}')
        # print('time assign point ot clusters: {}s'.format(perf_counter() - t1))
        # Update cluster medoids to be lowest cost point.
        t1 = perf_counter()
        for idx, curr_medoid in enumerate(curr_medoids):
            # print(f'idx: {idx}')
            cluster = np.where(clusters == curr_medoid)[0]
            # cluster = np.asarray(clusters == curr_medoid)
            # print(f'curr_medoid: {curr_medoid}')
            # print(f'np.where(clusters == curr_medoid): {np.where(clusters == curr_medoid)}')
            # print(f'cluster: {cluster}')
            new_medoids[curr_medoids == curr_medoid] = compute_new_medoid(cluster, distances)
            del cluster
        # print('time update medoids: {}s'.format(perf_counter() - t1))
        old_medoids[:] = curr_medoids[:]
        curr_medoids[:] = new_medoids[:]
        if num_iter >= 50:
            print(f'Stop as reach {num_iter} iterations')
            break
    print('total num_iter is {}'.format(num_iter))
    print('-----------------------------')
    clusters = assign_points_to_clusters(curr_medoids, distances)
    return clusters, curr_medoids

def assign_points_to_clusters(medoids, distances):
    distances_to_medoids = distances[:,medoids]
    clusters = medoids[np.argmin(distances_to_medoids, axis=1)]
    clusters[medoids] = medoids
    return clusters

def compute_new_medoid(cluster, distances):
    # mask = np.ones(distances.shape)
    # print(f'distance[10,10]: {distances[10,10]}')
    # t1 = perf_counter()
    # mask[np.ix_(cluster,cluster)] = 0.
    # print(f'np.ix_(cluster,cluster): {np.ix_(cluster,cluster)}')
    # print(f'mask: {mask}')
    # print('time creating mask: {}s'.format(perf_counter()-t1))
    # input('before')
    # cluster_distances = np.ma.masked_array(data=distances, mask=mask, fill_value=10e9)
    # print(f'cluster_distances: {cluster_distances}')
    # t1 = perf_counter()
    # print('cluster_distances.shape: {}'.format(cluster_distances.shape))
    # costs = cluster_distances.sum(axis=1)
    # print(f'costs: {costs}')
    # print('time counting costs: {}s'.format(perf_counter()-t1))
    # print(f'medoid: {costs.argmin(axis=0, fill_value=10e9)}')
    # return costs.argmin(axis=0, fill_value=10e9)
    cluster_distances = distances[cluster,:][:,cluster]
    costs = cluster_distances.sum(axis=1)
    min_idx = costs.argmin(axis=0)
    # print(f'new_costs: {costs}')
    # print(f'new_medoid: {cluster[min_idx]}')
    return cluster[min_idx]



def query_t3(adj, prob_nc, prob_ad, number, nodes_idx):
    '''
    The sum of entropy in the neighborhood
    '''
    comm_score = F.softmax(prob_nc,dim=1)
    ano_score = F.softmax(prob_ad, dim=1)[:,1]

    log_comm_score = torch.log_softmax(comm_score, dim=1).detach()
    stru_comm_entropy = -torch.sum(comm_score*log_comm_score, dim=1)
    nb_comm_entropy = torch.mm(adj, stru_comm_entropy.view(-1,1))

    final_score = nb_comm_entropy.view(-1) * (ano_score)

    indices = torch.topk(final_score[nodes_idx], number, largest=True)[1]
    indices = list(indices.cpu().numpy())

    return np.array(nodes_idx)[indices]


def query_t1(embed, prob_nc, prob_ad, number, nodes_idx, labels, idx_train_nc):
    '''
    Embedding center -- the mean of embeddings of labeled nodes
    Pseudo anomaly labels -- to remove predicted ood nodes
    Target: Identify the predicted in-distribution nodes that are farthest from the embedding center
    '''
    unique_labels = torch.unique(labels[idx_train_nc])
    anomaly_label  = F.softmax(prob_ad, dim=1).argmax(1)

    k = number // unique_labels.shape[0]
    grouped_indices = {label.item(): idx_train_nc[labels[idx_train_nc] == label] for label in unique_labels}
    centers = {label: embed[indices].mean(dim=0) for label, indices in grouped_indices.items()}

    argmax_result = prob_nc.argmax(dim=1)[nodes_idx]
    embeds_id0 = {label.item(): torch.LongTensor(nodes_idx)[labels[nodes_idx] == label] for label in unique_labels} # devide the community with actual labels 
    embeds_id = {label.item(): torch.LongTensor(nodes_idx)[(anomaly_label[nodes_idx]==0) & (argmax_result==label)] for label in unique_labels} # remove predicted anomalies

    selected = torch.LongTensor()
    for label in unique_labels:
        if embeds_id[label.item()].shape[0]<k:
            if embeds_id[label.item()].shape[0]>0:
                distances = torch.sum((embed[embeds_id[label.item()]] - centers[label.item()])**2, dim=1)
                indices = torch.topk(distances, embeds_id[label.item()].shape[0], largest=True)[1]
                indices = embeds_id[label.item()][indices]
                selected = torch.cat((selected,indices))

            distances = torch.sum((embed[embeds_id0[label.item()]] - centers[label.item()])**2, dim=1)
            indices = torch.topk(distances, k-embeds_id[label.item()].shape[0], largest=True)[1]
            indices = embeds_id0[label.item()][indices]
            selected = torch.cat((selected,indices))
        else:  
            distances = torch.sum((embed[embeds_id[label.item()]] - centers[label.item()])**2, dim=1)
            indices = torch.topk(distances, k, largest=True)[1]
            indices = embeds_id[label.item()][indices]
            selected = torch.cat((selected,indices))

    return selected


def query_t5(embed, prob_nc, prob_ad, number, nodes_idx, labels, idx_train_nc):
    '''
    Embedding center -- the mean of embeddings of labeled nodes
    Anomaly scores
    Target: Identify the predicted in-distribution nodes that are farthest from the embedding center
    '''
    unique_labels = torch.unique(labels[idx_train_nc])
    anomaly_scores  = F.softmax(prob_ad, dim=1)[:,1]

    k = number // unique_labels.shape[0]
    grouped_indices = {label.item(): idx_train_nc[labels[idx_train_nc] == label] for label in unique_labels}
    centers = {label: embed[indices].mean(dim=0) for label, indices in grouped_indices.items()}

    argmax_result = prob_nc.argmax(dim=1)[nodes_idx]
    embeds_id = {label.item(): torch.LongTensor(nodes_idx)[argmax_result == label] for label in unique_labels}

    selected = torch.LongTensor()
    for label in unique_labels:
        distances = torch.sum((embed[embeds_id[label.item()]] - centers[label.item()])**2, dim=1)
        distances_ano = distances * (1 - anomaly_scores[embeds_id[label.item()]])
        if distances_ano.shape[0]>k:
            indices = torch.topk(distances_ano, k, largest=True)[1]
        else:
            indices = torch.topk(distances_ano, distances_ano.shape[0], largest=True)[1]

        indices = embeds_id[label.item()][indices]
        selected = torch.cat((selected,indices))

    return selected

def query_t7(adj, prob_nc, prob_ad, number, nodes_idx):
    '''
    The sum of entropy in the neighborhood
    '''
    ano_entropy = get_entropy_score(prob_ad)
    comm_entropy = get_entropy_score(prob_nc)

    total_entropy = ano_entropy * comm_entropy

    nb_comm_entropy = torch.mm(adj, total_entropy.view(-1,1))

    final_score = nb_comm_entropy.view(-1) 

    indices = torch.topk(final_score[nodes_idx], number, largest=True)[1]
    indices = list(indices.cpu().numpy())

    return np.array(nodes_idx)[indices]