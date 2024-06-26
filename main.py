import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import random
import os
import dgl
import argparse
import time
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

from utils import *
from models import *
from sampling_methods import *


def train_model(args, model, opt, features, adj, labels, ano_labels, idx_train_nc, idx_train_ad, idx_val, filename):
    xent = nn.CrossEntropyLoss()
    weight = (1-ano_labels[idx_train_ad]).sum()/ano_labels[idx_train_ad].sum()

    patience = 20
    early_stopping = 20
    best_val = 0
    cur_p = 0

    for epoch in range(args.max_epoch):
        model.train()
        opt.zero_grad()

        embed, prob_nc, prob_ad = model(features, adj)

        loss_comm = xent(prob_nc[idx_train_nc], labels[idx_train_nc]) # node classification
        loss_an = F.cross_entropy(prob_ad[idx_train_ad], ano_labels[idx_train_ad], weight=torch.tensor([1., weight]).cuda())

        loss_un = 0.
        if torch.where(ano_labels[idx_train_ad]==0)[0].shape[0] != 0:
            loss_un = loss_un + get_entropy_score(prob_nc[idx_train_ad][torch.where(ano_labels[idx_train_ad]==0)[0]]).mean() 
        if torch.where(ano_labels[idx_train_ad]==1)[0].shape[0] != 0:
            loss_un = loss_un - get_entropy_score(prob_nc[idx_train_ad][torch.where(ano_labels[idx_train_ad]==1)[0]]).mean() 

        loss_total = args.alpha * loss_comm + args.beta * loss_an + args.gamma * loss_un

        loss_total.backward()
        opt.step()

        with torch.no_grad():
            model.eval()
            embed, prob_nc, prob_ad  = model(features, adj)

            idx_val_id = idx_val[ano_labels[idx_val]==0]
            nc_acc_val = accuracy_score(labels[idx_val].cpu().detach().numpy(), prob_nc[idx_val].max(1)[1].cpu().detach().numpy())
            nc_idacc_val = accuracy_score(labels[idx_val_id].cpu().detach().numpy(), prob_nc[idx_val_id].max(1)[1].cpu().detach().numpy())

            print('Train Loss', "{:.5f}".format(loss_total.item()),
                    'Train Comm Loss', "{:.5f}".format(loss_comm.item()),
                    'Train Ano Loss', "{:.5f}".format(loss_an.item()),
                    'NC-ACC:', "{:.5f}".format(nc_acc_val),
                    'NC-IDACC: ', "{:.5f}".format(nc_idacc_val))

            
            pred_nscores = get_entropy_score(prob_nc)
            pred_ascores = torch.softmax(prob_ad, dim=1)[:,1]
            scores = (pred_nscores-pred_nscores.mean())/(pred_nscores.std()) + args.phi * (pred_ascores-pred_ascores.mean())/(pred_ascores.std())
            mix_auc_val = roc_auc_score(ano_labels[idx_val].cpu().detach().numpy(),scores[idx_val].cpu().detach().numpy())

            # Save model untill loss does not decrease
            if epoch > early_stopping: 
                if mix_auc_val > best_val:
                    best_iter = epoch
                    best_val = mix_auc_val
                    cur_p = 0
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                    }, filename+'_checkpoint.pt')
                else:
                    cur_p += 1

            if cur_p > patience or epoch+1 >= args.max_epoch:
                print('epoch: {}, best_val: {}'.format(epoch, best_val))
                checkpoint = torch.load(filename+'_checkpoint.pt')
                model.load_state_dict(checkpoint['model_state_dict'])
                opt.load_state_dict(checkpoint['optimizer_state_dict'])
                break

    return model, opt


def test_model(model, features, adj, labels, ano_labels, idx_train_ad, idx_test):
    with torch.no_grad():
        model.eval()
        embed, prob_nc, prob_ad = model(features, adj)

        abnormal_num = int(ano_labels[idx_train_ad].sum().item())
        normal_num = len(idx_train_ad) - abnormal_num - 20*len(set(labels.tolist()))

        pred_nscores = get_entropy_score(prob_nc)
        pred_ascores = torch.softmax(prob_ad, dim=1)[:,1]

        # mixed_scores based
        scores = (pred_nscores-pred_nscores.mean())/(pred_nscores.std()) + args.phi * (pred_ascores-pred_ascores.mean())/(pred_ascores.std())
        mix_auc = roc_auc_score(ano_labels[idx_test].cpu().detach().numpy(),scores[idx_test].cpu().detach().numpy())
        mix_ap = average_precision_score(ano_labels[idx_test].cpu().detach().numpy(),scores[idx_test].cpu().detach().numpy())

        # a_scores based
        ano_auc = roc_auc_score(ano_labels[idx_test].cpu().detach().numpy(),pred_ascores[idx_test].cpu().detach().numpy())
        ano_ap = average_precision_score(ano_labels[idx_test].cpu().detach().numpy(),pred_ascores[idx_test].cpu().detach().numpy())

        # n_scores based
        ent_auc = roc_auc_score(ano_labels[idx_test].cpu().detach().numpy(),pred_nscores[idx_test].cpu().detach().numpy())
        ent_ap = average_precision_score(ano_labels[idx_test].cpu().detach().numpy(),pred_nscores[idx_test].cpu().detach().numpy())

        print('Anomaly Detection Results')
        print('MIX-AUC', "{:.5f}".format(mix_auc), 'MIX-PR', "{:.5f}".format(mix_ap),
              'ANO-AUC', "{:.5f}".format(ano_auc), 'ANO-PR', "{:.5f}".format(ano_ap),
              'ENT-AUC', "{:.5f}".format(ent_auc), 'ENT-PR', "{:.5f}".format(ent_ap),
              'N_num', normal_num, 'A_num', abnormal_num)

        test_acc_nc = accuracy_score(labels[idx_test].cpu().detach().numpy(), prob_nc[idx_test].max(1)[1].cpu().detach().numpy())
        idx_test_id = idx_test[ano_labels[idx_test]==0]
        test_acc_nc_id = accuracy_score(labels[idx_test_id].cpu().detach().numpy(), prob_nc[idx_test_id].max(1)[1].cpu().detach().numpy())

        print('Node Classification Results')
        print('ACC', "{:.5f}".format(test_acc_nc), 
              'ID-ACC', "{:.5f}".format(test_acc_nc_id))
        
    return embed, prob_nc, prob_ad


def main(args):
    # Load and preprocess data
    adj, features, labels, idx_train, idx_val, idx_test, ano_label = load_mat_f(args.dataset)

    features, _ = preprocess_features(features)
    nb_classes = labels.max() + 1

    adj_n = normalize_adj(adj)
    adj = (adj_n + sp.eye(adj_n.shape[0])).todense()
    
    # Init Selection
    idx_train_nc = np.loadtxt("splited_data/"+args.dataset+"/nc", dtype=int)
    idx_train_ad = np.loadtxt("splited_data/"+args.dataset+"/nc", dtype=int)

    # Init annotation state
    state_an = torch.zeros(features.shape[0]) - 1
    state_an = state_an.long()
    state_an[idx_train_ad] = 1
    state_an[idx_train_nc] = 1
    state_an[idx_val] = 2
    state_an[idx_test] = 2

    timestamp = time.time()
    prefix = 'saved_models/' + args.dataset + args.strategy_ad + '/'
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    filename = prefix + str(args.seed)+ '_'+ str(timestamp)

    features = torch.FloatTensor(features)
    adj = torch.FloatTensor(adj)
    labels = torch.LongTensor(labels)
    ano_label = torch.LongTensor(ano_label)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    idx_train_nc = torch.LongTensor(idx_train_nc)
    idx_train_ad = torch.LongTensor(idx_train_ad)


    model = Model(features.shape[1], args.embedding_dim, nb_classes, dropout=args.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if torch.cuda.is_available():
        print('Using CUDA')
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        ano_label = ano_label.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
        idx_train_ad = idx_train_ad.cuda()
        idx_train_nc = idx_train_nc.cuda()


    # Train model
    budget_ad = args.iter_budget * 2
    iter_num = args.max_budget // args.iter_budget

    for iter in range(iter_num + 1):
        
        model, opt = train_model(args, model, opt, features, adj, labels, ano_label, idx_train_nc, idx_train_ad, idx_val, filename)
        embed, prob_nc, prob_ad = test_model(model, features, adj, labels, ano_label, idx_train_ad, idx_test)

        # Node Selection
        if len(idx_train_ad) < args.max_budget * 2 + idx_train_nc.shape[0]:
            weight = args.tau ** (len(idx_train_ad)-len(idx_train_nc))
            idx_cand_an = torch.where(state_an==-1)[0]

            if args.strategy_ad == 'nent_diff': # MITIGATE w/o clustering
                idx_selected_ad = query_nent_diff(prob_nc, prob_ad, budget_ad, idx_cand_an.tolist(), weight)
            elif args.strategy_ad == 'medoids_spec_diff': # MITIGATE w/o entropy score
                idx_selected_ad = query_medoids_spec_diff(adj, embed, prob_nc, prob_ad, budget_ad, idx_cand_an.tolist(), args.cluster_num)
            elif args.strategy_ad == 'medoids_spec_nent': # MITIGATE w/o confidence difference
                idx_selected_ad = query_medoids_spec_nent(adj, embed, prob_nc, budget_ad, idx_cand_an.tolist(), args.cluster_num)
            elif args.strategy_ad == 'medoids_nent_diff': # MITIGATE w/o masked aggregation
                idx_selected_ad = query_medoids_nent_diff(embed, prob_nc, prob_ad, budget_ad, idx_cand_an.tolist(), args.cluster_num, weight)
            elif args.strategy_ad == 'medoids_spec_nent_diff': # MITIGATE
                idx_selected_ad = query_medoids_spec_nent_diff(adj, embed, prob_nc, prob_ad, budget_ad, idx_cand_an.tolist(), args.cluster_num, weight)
            else:
                raise ValueError("AD Strategy is not defined")

            print('Selected! ')

            # Update state
            state_an[idx_selected_ad] = 1
            idx_train_ad = torch.cat((idx_train_ad, torch.tensor(idx_selected_ad).cuda()))



if __name__ == '__main__':
    # Set argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')  # 'BlogCatalog'  'Flickr'  'cora'  'citeseer' 
    parser.add_argument('--lr', type=float, default =0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--seed', type=int, default=800)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--max_epoch', type=int, default=300)
    parser.add_argument('--max_budget', type=int, default=40)
    parser.add_argument('--iter_budget', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=1.25, help='node classification loss weight')
    parser.add_argument('--beta', type=float, default=0.5, help='anomaly detection loss weight')
    parser.add_argument('--gamma', type=float, default=1, help='uncertainty loss weight') # gamma=0 MITIGATE w/o uncertainty loss
    parser.add_argument('--phi', type=float, default=1.25, help='anoamly score weight')
    parser.add_argument('--tau', type=float, default=0.95)
    parser.add_argument('--strategy_ad', type=str, default='medoids_spec_nent_diff')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--cluster_num', type=int, default=24)

    args = parser.parse_args()

    torch.cuda.set_device(args.device)

    # Set random seed
    dgl.random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    
    main(args)