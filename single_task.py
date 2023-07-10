import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import random
import dgl
import argparse
import os
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

    if args.task_type == 'ad':
        labels = ano_label

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

    model = SingleTask(ft_size, args.embedding_dim, nb_classes, dropout=args.dropout)
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

    # Init selection
    idx_train_nc = init_category(args.init_num, idx_train, labels)

    # Budget in each iteration
    budget_nc = int(nb_classes * (args.max_budget - args.init_num) / args.iter_num)

    # Init annotation state
    state_nc = torch.zeros(features.shape[0]) - 1
    state_nc = state_nc.long()
    state_nc[idx_train_nc] = 1

    # Train model
    best_model = None
    best_opt = None
    best_val = 0
    for iter in range(args.iter_num + 1):
        

        for epoch in range(args.max_epoch):
            model.train()
            opt.zero_grad()

            embed, prob_nc = model(features, adj)

            if args.task_type == 'ad':
                loss_comm = xent(prob_nc[idx_train_nc], labels[idx_train_nc]) 
            else:
                # idx_train_id = idx_train_nc[ano_label[idx_train_nc]==0]
                loss_comm = xent(prob_nc[idx_train_nc], labels[idx_train_nc])  # no anomaly labels, overall accuracy

            loss_comm.backward()
            opt.step()

            with torch.no_grad():
                model.eval()
                embed, prob_nc = model(features, adj)
                
                if args.task_type == 'ad':
                    acc_val = auc(prob_nc[idx_val], labels[idx_val])
                    print('AUC:', "{:.5f}".format(acc_val))
                else:
                    # idx_val_id = idx_val[ano_label[idx_val]==0]
                    acc_val = accuracy(prob_nc[idx_val], labels[idx_val])
                    print('ACC:', "{:.5f}".format(acc_val))

                if acc_val > best_val:
                    best_val = acc_val
                    best_model = copy.deepcopy(model.state_dict())
                    best_opt = copy.deepcopy(opt.state_dict())

        with torch.no_grad():
            model.load_state_dict(best_model)
            opt.load_state_dict(best_opt)

            model.eval()
            embed, prob_nc = model(features, adj)

        # Node Selection
        if len(idx_train_nc) < args.max_budget * nb_classes:
            idx_cand_nc = torch.where(state_nc==-1)[0]

            if args.strategy_nc == 'random':
                idx_selected_nc = query_random(budget_nc, idx_cand_nc.tolist())
            elif args.strategy_nc == 'largest_degree':
                idx_selected_nc = query_largest_degree(nx.from_numpy_array(np.array(adj.cpu())), budget_nc, idx_cand_nc.tolist())
            elif args.strategy_nc == 'uncertainty':
                idx_selected_nc = query_uncertainty(prob_nc, budget_nc, idx_cand_nc.tolist())
            elif args.strategy_nc == 'topk_anomaly':
                idx_selected_nc = query_topk_anomaly(prob_nc, budget_nc, idx_cand_nc.tolist())
            else:
                raise ValueError("Strategy is not defined")
            
            # Update state
            state_nc[idx_selected_nc] = 1
            idx_train_nc = torch.cat((idx_train_nc, torch.LongTensor(idx_selected_nc).cuda()))


    # Test model
    with torch.no_grad():
        model.eval()
        embed, prob_nc = model(features, adj)

        if args.task_type == 'ad':
            test_auc_ano = auc(prob_nc[idx_test], labels[idx_test])
            test_acc_ano = accuracy(prob_nc[idx_test], labels[idx_test])
            test_f1micro_ano, test_f1macro_ano = f1(prob_nc[idx_test], labels[idx_test])

            abnormal_num = int(labels[idx_train_nc].sum().item())
            normal_num = len(idx_train_nc) - abnormal_num

            print('Anomaly Detection Results')
            print('ACC', "{:.5f}".format(test_acc_ano), 
                'F1-Micro', "{:.5f}".format(test_f1micro_ano), 
                'F1-Macro', "{:.5f}".format(test_f1macro_ano), 
                'AUC', "{:.5f}".format(test_auc_ano),
                'N_num', normal_num,
                'A_num', abnormal_num)
            
        else:
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
    des_path = args.result_path + args.task_type + '.csv'
    if args.task_type == 'ad':
        if not os.path.exists(des_path):
            with open(des_path,'w+') as f:
                csv_write = csv.writer(f)
                csv_head = ["model", "seed", "dataset", "init_num", "num_epochs", "strategy_ad", "ad-auc", "ad-f1-macro", "A-num", "N-num"]
                csv_write.writerow(csv_head)

        with open(des_path, 'a+') as f:
            csv_write = csv.writer(f)
            data_row = [args.task_type, args.seed, args.dataset, args.init_num, args.max_epoch, args.strategy_nc, test_auc_ano, test_f1macro_ano, abnormal_num, normal_num]
            csv_write.writerow(data_row)

    else:
        if not os.path.exists(des_path):
            with open(des_path,'w+') as f:
                csv_write = csv.writer(f)
                csv_head = ["model", "seed", "dataset", "init_num", "num_epochs", "strategy_nc", "nc-acc", "nc-idacc", "nc-f1-micro", "nc-f1-macro"]
                csv_write.writerow(csv_head)

        with open(des_path, 'a+') as f:
            csv_write = csv.writer(f)
            data_row = [args.task_type, args.seed, args.dataset, args.init_num, args.max_epoch, args.strategy_nc, test_acc_comm, test_acc_nc_id, test_f1micro_comm, test_f1macro_comm]
            csv_write.writerow(data_row)



if __name__ == '__main__':
    # Set argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')  # 'cora'  'citeseer'  'pubmed'
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--init_num', type=int, default=2)
    parser.add_argument('--max_epoch', type=int, default=300)
    parser.add_argument('--max_budget', type=int, default=20)
    parser.add_argument('--iter_num', type=int, default=9)
    parser.add_argument('--strategy_nc', type=str, default='largest_degree') # random uncertainty largest_degree topk_anomaly
    parser.add_argument('--task_type', type=str, default='nc')
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--result_path', type=str, default='results/singletask_')

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