import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import random
import os
import dgl
import argparse
import copy

from utils import *
from models import *
from sampling_methods import *

def main(args):
    # Load and preprocess data
    adj, features, labels, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label = load_mat(args.dataset)

    features, _ = preprocess_features(features)
    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = labels.max() + 1

    adj = normalize_adj(adj)
    adj = (adj + sp.eye(adj.shape[0])).todense()

    features = torch.FloatTensor(features)
    adj = torch.FloatTensor(adj)
    labels = torch.LongTensor(labels)
    ano_label = torch.LongTensor(ano_label)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    model = MultiTask(ft_size, args.embedding_dim, nb_classes, dropout=0.6)
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

    xent = nn.CrossEntropyLoss()

    # Init Selection
    idx_train_nc = init_category(args.init_num, idx_train, labels)
    idx_train_ad = init_category(args.init_num, idx_train, ano_label)

    # Budget for each task in each iteration
    budget_nc = int(nb_classes * (args.max_budget - args.init_num) / args.iter_num) 
    budget_ad = int(2 * (args.max_budget - args.init_num) / args.iter_num)

    # Init annotation state
    state_nc = torch.zeros(features.shape[0]) - 1
    state_nc = state_nc.long()
    state_nc[idx_train_nc] = 1
    state_an = torch.zeros(features.shape[0]) - 1
    state_an = state_an.long()
    state_an[idx_train_ad] = 1

    # Train model
    best_model = None
    best_opt = None
    best_val = 0
    for iter in range(args.iter_num + 1):
        for epoch in range(args.max_epoch):
            model.train()
            opt.zero_grad()

            embed, prob_nc, prob_ad = model(features, adj)
            
            idx_train_id = idx_train_nc[torch.nonzero(~torch.isin(idx_train_nc, idx_train_ad[torch.where(ano_label[idx_train_ad]==1)[0]])).squeeze()]
            loss_comm = xent(prob_nc[idx_train_id], labels[idx_train_id]) # In-distribution
            loss_an = xent(prob_ad[idx_train_ad], ano_label[idx_train_ad])

            loss_total = args.alpha * loss_comm + (1-args.alpha) * loss_an

            loss_total.backward()
            opt.step()

            with torch.no_grad():
                model.eval()
                embed, prob_nc, prob_ad  = model(features, adj)
                
                ad_auc_val = auc(prob_ad[idx_val], ano_label[idx_val])
                idx_val_id = idx_val[ano_label[idx_val]==0]
                nc_acc_val = accuracy(prob_nc[idx_val], labels[idx_val])
                nc_idacc_val = accuracy(prob_nc[idx_val_id], labels[idx_val_id])

                print('AD-AUC:', "{:.5f}".format(ad_auc_val), ' NC-ACC:', "{:.5f}".format(nc_acc_val), ' NC-IDACC: ', "{:.5f}".format(nc_idacc_val))

                if ad_auc_val * nc_idacc_val > best_val:
                    best_val = ad_auc_val * nc_idacc_val
                    best_model = copy.deepcopy(model.state_dict())
                    best_opt = copy.deepcopy(opt.state_dict())

        with torch.no_grad():
            model.load_state_dict(best_model)
            opt.load_state_dict(best_opt)

            model.eval()
            embed, prob_nc, prob_ad  = model(features, adj)

        # Node Selection
        if len(idx_train_nc) < args.max_budget * nb_classes:
            idx_cand_nc = torch.where(state_nc==-1)[0]
            idx_cand_an = torch.where(state_an==-1)[0]

            if args.strategy_nc == 'random':
                idx_selected_nc = query_random(budget_nc, idx_cand_nc.tolist())
            elif args.strategy_nc == 'largest_degree':
                idx_selected_nc = query_largest_degree(nx.from_numpy_array(np.array(adj.cpu())), budget_nc, idx_cand_nc.tolist())
            elif args.strategy_nc == 'uncertainty':
                idx_selected_nc = query_uncertainty(prob_nc, budget_nc, idx_cand_nc.tolist())
            elif args.strategy_nc == 't2':
                idx_selected_nc = query_t2(adj, prob_nc, prob_ad, budget_nc, idx_cand_nc.tolist())
            elif args.strategy_nc == 't3':
                idx_selected_nc = query_t3(adj, prob_nc, prob_ad, budget_nc, idx_cand_nc.tolist())
            elif args.strategy_nc == 't4':
                idx_selected_nc = query_t4(adj, prob_nc, prob_ad, budget_nc, idx_cand_nc.tolist())
            elif args.strategy_nc == 't1':
                idx_selected_nc = query_t1(embed, prob_nc, prob_ad, budget_nc, idx_cand_nc.tolist(), labels, idx_train_nc)
            else:
                raise ValueError("NC Strategy is not defined")
            
            # Update state
            state_nc[idx_selected_nc] = 1
            idx_train_nc = torch.cat((idx_train_nc, torch.LongTensor(idx_selected_nc).cuda()))

            if args.strategy_ad == 'random':
                idx_selected_ad = query_random(budget_ad, idx_cand_an.tolist())
            elif args.strategy_ad == 'uncertainty':
                idx_selected_ad = query_uncertainty(prob_ad, budget_ad, idx_cand_an.tolist())
            elif args.strategy_ad == 'topk_anomaly':
                idx_selected_ad = query_topk_anomaly(prob_ad, budget_ad, idx_cand_an.tolist())
            else:
                raise ValueError("AD Strategy is not defined")

            # Update state
            state_an[idx_selected_ad] = 1
            idx_train_ad = torch.cat((idx_train_ad, torch.tensor(idx_selected_ad).cuda()))


    # Test model
    with torch.no_grad():
        model.eval()
        embed, prob_nc, prob_ad = model(features, adj)

        test_auc_ano = auc(prob_ad[idx_test], ano_label[idx_test])
        test_acc_ano = accuracy(prob_ad[idx_test], ano_label[idx_test])
        test_f1micro_ano, test_f1macro_ano = f1(prob_ad[idx_test], ano_label[idx_test])

        abnormal_num = int(ano_label[idx_train_ad].sum().item())
        normal_num = len(idx_train_ad) - abnormal_num

        print('Anomaly Detection Results')
        print('ACC', "{:.5f}".format(test_acc_ano), 
            'F1-Micro', "{:.5f}".format(test_f1micro_ano), 
            'F1-Macro', "{:.5f}".format(test_f1macro_ano), 
            'AUC', "{:.5f}".format(test_auc_ano),
            'N_num', normal_num,
            'A_num', abnormal_num)
            
        test_acc_comm = accuracy(prob_nc[idx_test], labels[idx_test])
        idx_test_id = idx_test[ano_label[idx_test]==0]
        test_acc_nc_id = accuracy(prob_nc[idx_test_id], labels[idx_test_id])
        test_f1micro_comm, test_f1macro_comm = f1(prob_nc[idx_test], labels[idx_test])

        print('Community Detection Results')
        print('ACC', "{:.5f}".format(test_acc_comm), 
            'ID-ACC', "{:.5f}".format(test_acc_nc_id), 
            'F1-Micro', "{:.5f}".format(test_f1micro_comm), 
            'F1-Macro', "{:.5f}".format(test_f1macro_comm))

    # Save results
    import csv
    des_path = args.result_path + '.csv'

    if not os.path.exists(des_path):
        with open(des_path,'w+') as f:
            csv_write = csv.writer(f)
            csv_head = ["model", "seed", "dataset", "init_num", "num_epochs", "strategy_ad", "strategy_nc", "alpha", "nc-acc", "nc-idacc", "nc-f1-micro", "nc-f1-macro", "ad-auc", "ad-f1-macro", "A-num", "N-num"]
            csv_write.writerow(csv_head)

    with open(des_path, 'a+') as f:
        csv_write = csv.writer(f)
        data_row = ['m1', args.seed, args.dataset, args.init_num, args.max_epoch, args.strategy_ad, args.strategy_nc, args.alpha, test_acc_comm, test_acc_nc_id, test_f1micro_comm, test_f1macro_comm, test_auc_ano, test_f1macro_ano, abnormal_num, normal_num]
        csv_write.writerow(data_row)




if __name__ == '__main__':
    # Set argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')  # 'BlogCatalog'  'Flickr'  'ACM'  'cora'  'citeseer'  'pubmed'
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--seed', type=int, default=255)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--init_num', type=int, default=2)
    parser.add_argument('--max_epoch', type=int, default=300)
    parser.add_argument('--max_budget', type=int, default=20)
    parser.add_argument('--iter_num', type=int, default=9)
    parser.add_argument('--alpha', type=float, default=0.5, help='balance parameter')
    parser.add_argument('--strategy_nc', type=str, default='random') # random uncertainty largest_degree featprop
    parser.add_argument('--strategy_ad', type=str, default='random') # random uncertainty topk_anomaly
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--result_path', type=str, default='results/multitask')

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