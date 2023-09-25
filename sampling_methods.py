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
    prob_output = F.softmax(output, dim=1)
    log_prob_output = F.log_softmax(output, dim=1)
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
    return idx_topk

def query_featprop(features, number, nodes_idx):
    features = features.cpu().numpy()
    X = features[nodes_idx]
    t1 = perf_counter()
    distances = pairwise_distances(X, X)
    print('computer pairwise_distances: {}s'.format(perf_counter() - t1))
    clusters, medoids = k_medoids(distances, k=number)
    return np.array(nodes_idx)[medoids]

def query_entropy(prob, number, nodes_idx):
    output = prob[nodes_idx]
    entropy = get_entropy_score(output).detach()
    indices = torch.topk(entropy, number, largest=True)[1]
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
    entropy = get_entropy_score(prob[nodes_idx]).detach()
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

def query_medoids(embed, prob_ad, number, nodes_idx, cluster_n):
    embed = embed[nodes_idx].cpu().numpy()
    distances = pairwise_distances(embed, embed)
    clusters, medoids = k_medoids(distances, k=cluster_n)
    indices = torch.topk(prob_ad[:,-1][medoids], number, largest=True)[1]
    indices = list(indices.cpu().numpy())
    return nodes_idx[medoids][indices]


def query_medoids_spec_nent_diff(adj, embed, prob_nc, prob_ad, number, nodes_idx, cluster_n, weight=0.5):
    n_entropy = get_entropy_score(prob_nc).detach()
    prob_ad = torch.softmax(prob_ad, dim=1)
    a_scores = prob_ad[:,1]

    scores_diff = torch.abs((n_entropy-n_entropy.mean())/(n_entropy.std()) - (a_scores-a_scores.mean())/(a_scores.std()))

    scores = weight * (n_entropy-n_entropy.mean())/(n_entropy.std()) + (1-weight) * scores_diff

    nodes_idx = np.array(nodes_idx)
    embed = torch.mm(adj[nodes_idx][:,nodes_idx],embed[nodes_idx])
    embed = embed.cpu().numpy()
    distances = pairwise_distances(embed, embed)

    clusters, medoids = k_medoids(distances, k=cluster_n)
    indices = torch.topk(scores[nodes_idx[medoids]], number, largest=True)[1]
    indices = list(indices.cpu().numpy())
    return nodes_idx[medoids][indices]


def query_medoids_spec_nent(adj, embed, prob_nc, number, nodes_idx, cluster_n):
    n_entropy = get_entropy_score(prob_nc).detach()

    scores = (n_entropy-n_entropy.mean())/(n_entropy.std())

    nodes_idx = np.array(nodes_idx)
    embed = torch.mm(adj[nodes_idx][:,nodes_idx],embed[nodes_idx])
    embed = embed.cpu().numpy()
    distances = pairwise_distances(embed, embed)

    clusters, medoids = k_medoids(distances, k=cluster_n)
    indices = torch.topk(scores[nodes_idx[medoids]], number, largest=True)[1]
    indices = list(indices.cpu().numpy())
    return nodes_idx[medoids][indices]

def query_medoids_spec_diff(adj, embed, prob_nc, prob_ad, number, nodes_idx, cluster_n):
    n_entropy = get_entropy_score(prob_nc).detach()
    prob_ad = torch.softmax(prob_ad, dim=1)
    a_scores = prob_ad[:,1]

    scores_diff = torch.abs((n_entropy-n_entropy.mean())/(n_entropy.std()) - (a_scores-a_scores.mean())/(a_scores.std()))

    scores = scores_diff 

    nodes_idx = np.array(nodes_idx)
    embed = torch.mm(adj[nodes_idx][:,nodes_idx],embed[nodes_idx])
    embed = embed.cpu().numpy()
    distances = pairwise_distances(embed, embed)

    clusters, medoids = k_medoids(distances, k=cluster_n)
    indices = torch.topk(scores[nodes_idx[medoids]], number, largest=True)[1]
    indices = list(indices.cpu().numpy())
    return nodes_idx[medoids][indices]

def query_medoids_diff(embed, prob_nc, prob_ad, number, nodes_idx, cluster_n):
    n_entropy = get_entropy_score(prob_nc).detach()
    prob_ad = torch.softmax(prob_ad, dim=1)
    a_scores = prob_ad[:,1]

    scores_diff = torch.abs((n_entropy-n_entropy.mean())/(n_entropy.std()) - (a_scores-a_scores.mean())/(a_scores.std()))

    scores = scores_diff 

    nodes_idx = np.array(nodes_idx)
    embed = embed.cpu().numpy()
    distances = pairwise_distances(embed, embed)

    clusters, medoids = k_medoids(distances[nodes_idx], k=cluster_n)
    indices = torch.topk(scores[nodes_idx[medoids]], number, largest=True)[1]
    indices = list(indices.cpu().numpy())
    return nodes_idx[medoids][indices]

def query_medoids_nent_diff(embed, prob_nc, prob_ad, number, nodes_idx, cluster_n, weight=0.5):
    n_entropy = get_entropy_score(prob_nc).detach()
    prob_ad = torch.softmax(prob_ad, dim=1)
    a_entropy = get_entropy_score(prob_ad).detach()

    scores_diff = torch.abs((n_entropy-n_entropy.mean())/(n_entropy.std()) - (a_entropy-a_entropy.mean())/(a_entropy.std()))

    scores = weight * (n_entropy-n_entropy.mean())/(n_entropy.std()) + (1-weight) * scores_diff

    nodes_idx = np.array(nodes_idx)
    embed = embed.cpu().numpy()
    distances = pairwise_distances(embed, embed)

    clusters, medoids = k_medoids(distances[nodes_idx], k=cluster_n)
    indices = torch.topk(scores[nodes_idx[medoids]], number, largest=True)[1]
    indices = list(indices.cpu().numpy())
    return nodes_idx[medoids][indices]


def query_nent_ascore(prob_nc, prob_ad, number, nodes_idx, weight):
    entropy = get_entropy_score(prob_nc).detach()
    prob_ad = torch.softmax(prob_ad, dim=1)
    pred_ascores = prob_ad[:,1]

    scores = weight * (entropy-entropy.mean())/(entropy.std()) + (1-weight) * (pred_ascores-pred_ascores.mean())/(pred_ascores.std())

    indices = torch.topk(scores[nodes_idx], number, largest=True)[1]
    indices = list(indices.cpu().numpy())
    return np.array(nodes_idx)[indices]


def query_topk_nent_aent(prob_nc, prob_ad, number, nodes_idx, weight):
    entropy = get_entropy_score(prob_nc).detach()
    prob_ad = torch.softmax(prob_ad, dim=1)
    a_entropy = get_entropy_score(prob_ad).detach()

    scores = weight * (entropy-entropy.min())/(entropy.max()-entropy.min()) + (1-weight) * (a_entropy-a_entropy.min())/(a_entropy.max()-a_entropy.min())
    indices = torch.topk(scores[nodes_idx], number, largest=True)[1]
    indices = list(indices.cpu().numpy())
    return np.array(nodes_idx)[indices]


def query_diff(prob_nc, prob_ad, number, nodes_idx):
    n_entropy = get_entropy_score(prob_nc).detach()
    prob_ad = torch.softmax(prob_ad, dim=1)
    pred_ascores = prob_ad[:,1]

    scores = torch.abs((n_entropy-n_entropy.mean())/(n_entropy.std()) - (pred_ascores-pred_ascores.mean())/(pred_ascores.std()))
    indices = torch.topk(scores[nodes_idx], number, largest=True)[1]
    indices = list(indices.cpu().numpy())
    return np.array(nodes_idx)[indices]

def query_nent_diff(prob_nc, prob_ad, number, nodes_idx, weight):
    n_entropy = get_entropy_score(prob_nc).detach()
    prob_ad = torch.softmax(prob_ad, dim=1)
    a_scores = prob_ad[:,1]

    scores_diff = torch.abs((n_entropy-n_entropy.mean())/(n_entropy.std()) - (a_scores-a_scores.mean())/(a_scores.std()))

    scores = weight * (n_entropy-n_entropy.mean())/(n_entropy.std()) + (1-weight) * scores_diff

    indices = torch.topk(scores[nodes_idx], number, largest=True)[1]
    indices = list(indices.cpu().numpy())
    return np.array(nodes_idx)[indices]

def return_community(number, nodes_idx, labels):
    unique_labels = torch.unique(labels)
    res = np.array([])
    k = number // unique_labels.shape[0]
    for label in unique_labels:
        label_node_idx = torch.LongTensor(nodes_idx)[torch.where(labels[nodes_idx]==label)[0]]
        res = np.hstack((res, np.random.choice(label_node_idx, size=k, replace=False)))
    return res

def return_anomaly(number, nodes_idx, ano_labels):
    ano_node_idx = np.array(nodes_idx)[np.where(ano_labels[nodes_idx]==1)[0]]
    nor_node_idx = np.array(nodes_idx)[np.where(ano_labels[nodes_idx]==0)[0]]
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

