import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import random
import dgl
import argparse
import os
import copy
import time

from utils import *
from models import *
from sampling_methods import *

def main(args):
    # Load and preprocess data
    adj, features, labels, idx_train, idx_val, idx_test, ano_labels, str_ano_label, attr_ano_label = load_mat(args.dataset)

    features, _ = preprocess_features(features)
    nb_nodes = features.shape[0]
    ft_size = features.shape[1]


    nb_classes = labels.max() + 1

    adj = normalize_adj(adj)
    adj = (adj + sp.eye(adj.shape[0])).todense()

    features = torch.FloatTensor(features)
    adj = torch.FloatTensor(adj)
    labels = torch.LongTensor(labels)
    ano_labels = torch.LongTensor(ano_labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if args.info_type == 'labels':
        labels = labels.unsqueeze(1)
        features = torch.cat((features, labels), dim=1)
        model = SingleTask(ft_size+1, args.embedding_dim, 2, dropout=args.dropout)
    elif args.info_type == 'onehot_labels':
        labels = F.one_hot(labels, num_classes = nb_classes)
        features = torch.cat((features, labels), dim=1)
        model = SingleTask(ft_size+nb_classes, args.embedding_dim, 2, dropout=args.dropout)
    else:
        model = SingleTask(ft_size, args.embedding_dim, 2, dropout=args.dropout)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if torch.cuda.is_available():
        print('Using CUDA')
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        ano_labels = ano_labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    xent = nn.CrossEntropyLoss()

    timestamp = time.time()
    prefix = 'saved_models/SingleTask/' + args.dataset + str(args.device) + '/'
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    filename = prefix + str(args.seed)+ '_single_'+ str(timestamp)

    # Init selection
    idx_train_ad = init_category(args.init_num, idx_train, ano_labels)

    #
    budget_factor = 2 * (args.max_budget - args.init_num) / (args.iter_num*(args.iter_num + 1) / 2)
    max_budget_ad = 2 * (args.max_budget - args.init_num) 

    # Init annotation state
    state_ad = torch.zeros(features.shape[0]) - 1
    state_ad = state_ad.long()
    state_ad[idx_train_ad] = 1


    # Train model
    patience = 20
    early_stopping = 20
    best_val = 0
    total_spent_budget = 0
    for iter in range(args.iter_num + 1):
        cur_p = 0
        best_loss = 1e9

        # Budget in each iteration
        if args.ad_budget_type == 'equal':
            budget_ad = int(2 * (args.max_budget - args.init_num) / args.iter_num)
        elif args.ad_budget_type == 'decrease':
            budget_ad += iter * budget_factor
            if iter != args.iter_num:
                budget_ad = round(iter * budget_factor)
                total_spent_budget += budget_ad
            else:
                budget_ad = max_budget_ad - total_spent_budget

        for epoch in range(args.max_epoch):
            model.train()
            opt.zero_grad()

            embed, prob_ad = model(features, adj)

            loss = xent(prob_ad[idx_train_ad], ano_labels[idx_train_ad]) 
            loss.backward()
            opt.step()

            with torch.no_grad():
                model.eval()
                embed, prob_ad = model(features, adj)
                
                val_loss = xent(prob_ad[idx_val], ano_labels[idx_val]).item()

                auc_val = auc(prob_ad[idx_val], ano_labels[idx_val])
                val_f1micro, val_f1macro = f1(prob_ad[idx_val], ano_labels[idx_val])
                print('Train Loss',"{:.5f}".format(loss.item()),' AUC:', "{:.5f}".format(auc_val),' F1-Macro:', "{:.5f}".format(val_f1macro))

                # Save model untill loss does not decrease
                if epoch > early_stopping:
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_iter = epoch
                        cur_p = 0
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': opt.state_dict(),
                        }, filename+'_checkpoint.pt')
                    else:
                        cur_p += 1


                if cur_p > patience or epoch+1 >= args.max_epoch:
                    print('epoch: {}, auc_val: {}, best_auc_val: {}'.format(epoch, auc_val, best_val))
                    # load best model
                    checkpoint = torch.load(filename+'_checkpoint.pt')
                    model.load_state_dict(checkpoint['model_state_dict'])
                    opt.load_state_dict(checkpoint['optimizer_state_dict'])
                    break

        with torch.no_grad():
            model.eval()
            embed, prob_ad = model(features, adj)

            test_auc_ano = auc(prob_ad[idx_test], ano_labels[idx_test])
            test_f1micro_ano, test_f1macro_ano = f1(prob_ad[idx_test], ano_labels[idx_test])

            abnormal_num = int(ano_labels[idx_train_ad].sum().item())
            normal_num = len(idx_train_ad) - abnormal_num

            print('Anomaly Detection Results')
            print('F1-Macro', "{:.5f}".format(test_f1macro_ano), 
                  'AUC', "{:.5f}".format(test_auc_ano),
                  'N_num', normal_num,
                  'A_num', abnormal_num)
            
        # Node Selection
        if len(idx_train_ad) < args.max_budget * 2:
            idx_cand_ad = torch.where(state_ad==-1)[0]

            if args.strategy_ad == 'random':
                idx_selected_ad = query_random(budget_ad, idx_cand_ad.tolist())
            elif args.strategy_ad == 'largest_degree':
                idx_selected_ad = query_largest_degree(nx.from_numpy_array(np.array(adj.cpu())), budget_ad, idx_cand_ad.tolist())
            elif args.strategy_ad == 'featprop':
                idx_selected_ad = query_featprop(embed, budget_ad, idx_cand_ad.tolist())
            elif args.strategy_ad == 'entropy':
                idx_selected_ad = query_entropy(prob_ad, budget_ad, idx_cand_ad.tolist())
            elif args.strategy_ad == 'topk_anomaly':
                idx_selected_ad = query_topk_anomaly(prob_ad, budget_ad, idx_cand_ad.tolist())
            elif args.strategy_ad == 'density':
                idx_selected_ad = query_density(embed, budget_ad, idx_cand_ad.tolist(), labels)
            elif args.strategy_ad == 'entropy_density':
                idx_selected_ad = query_entropy_density(embed, prob_ad, budget_ad, idx_cand_ad.tolist(), labels)
            elif args.strategy_ad == 'topk_medoids':
                idx_selected_ad = query_topk_medoids(embed, prob_ad, budget_ad, idx_cand_ad.tolist(), nb_classes)
            else:
                raise ValueError("Strategy is not defined")
            
            # Update state
            state_ad[idx_selected_ad] = 1
            idx_train_ad = torch.cat((idx_train_ad, torch.LongTensor(idx_selected_ad).cuda()))




    # Save results
    import csv
    des_path = args.result_path + '.csv'

    if not os.path.exists(des_path):
        with open(des_path,'w+') as f:
            csv_write = csv.writer(f)
            csv_head = ["model", "seed", "dataset", "init_num", "num_epochs", "strategy_ad", "ad-auc", "ad-f1-macro", "A-num", "N-num"]
            csv_write.writerow(csv_head)

    with open(des_path, 'a+') as f:
        csv_write = csv.writer(f)
        data_row = [args.info_type, args.seed, args.dataset, args.init_num, args.max_epoch, args.strategy_ad, test_auc_ano, test_f1macro_ano, abnormal_num, normal_num]
        csv_write.writerow(data_row)


if __name__ == '__main__':
    # Set argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')  # 'cora'  'citeseer'  'pubmed'
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--seed', type=int, default=255)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--init_num', type=int, default=2)
    parser.add_argument('--max_epoch', type=int, default=300)
    parser.add_argument('--max_budget', type=int, default=20)
    parser.add_argument('--iter_num', type=int, default=9)
    parser.add_argument('--strategy_ad', type=str, default='entropy_density') # random entropy largest_degree topk_anomaly
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--info_type', type=str, default='onehot_labels') # non labels onehot_labels
    parser.add_argument('--result_path', type=str, default='results/label_info')
    parser.add_argument('--ad_budget_type', type=str, default='equal') # decrease/equal

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