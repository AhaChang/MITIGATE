import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import random
import os
import dgl
import argparse
import copy
import time

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

    timestamp = time.time()
    prefix = 'saved_models/FixNCBudegt/' + args.dataset + str(args.device) + '/'
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    filename = prefix + str(args.seed)+ '_fixncb_'+ str(timestamp)

    # Init Selection
    idx_train_nc = init_category_nc(args.nc_num, idx_train, labels, ano_label)
    idx_train_ad = init_category(args.init_num, idx_train, ano_label)

    # 
    budget_factor = 2 * (args.max_budget - args.init_num) / (args.iter_num*(args.iter_num + 1) / 2)
    max_budget_ad = 2 * (args.max_budget - args.init_num) 

    # Init annotation state
    state_an = torch.zeros(features.shape[0]) - 1
    state_an = state_an.long()
    state_an[idx_train_ad] = 1
    state_an[idx_val] = 2
    state_an[idx_train] = 2


    # Train model
    patience = 20
    early_stopping = 20
    best_val = 0
    budget_ad = 0
    total_spent_budget = 0
    for iter in range(args.iter_num + 1):
        cur_p = 0
        best_loss = 1e9

        # time sensitive parameters
        budget_ad += iter * budget_factor

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

            embed, prob_nc, prob_ad = model(features, adj)
            
            loss_comm = xent(prob_nc[idx_train_nc], labels[idx_train_nc]) # In-distribution
            loss_an = xent(prob_ad[idx_train_ad], ano_label[idx_train_ad])

            loss_total = args.alpha * loss_comm + (1-args.alpha) * loss_an

            loss_total.backward()
            opt.step()

            with torch.no_grad():
                model.eval()
                embed, prob_nc, prob_ad  = model(features, adj)
                idx_val_id = idx_val[ano_label[idx_val]==0]
                
                val_loss = xent(prob_ad[idx_val], ano_label[idx_val]).item()
                
                ad_auc_val = auc(prob_ad[idx_val], ano_label[idx_val])
                ad_f1micro_val, ad_f1macro_val = f1(prob_ad[idx_val], ano_label[idx_val])
                
                nc_acc_val = accuracy(prob_nc[idx_val], labels[idx_val])
                nc_idacc_val = accuracy(prob_nc[idx_val_id], labels[idx_val_id])

                print('Train Loss', "{:.5f}".format(loss_total.item()),
                      'Train Comm Loss', "{:.5f}".format(loss_comm.item()),
                      'Train Ano Loss', "{:.5f}".format(loss_an.item()),
                      'AD-AUC:', "{:.5f}".format(ad_auc_val),
                      'AD-F1:', "{:.5f}".format(ad_f1macro_val),
                      'NC-ACC:', "{:.5f}".format(nc_acc_val),
                      'NC-IDACC: ', "{:.5f}".format(nc_idacc_val))


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
                    print('epoch: {}, acc_val: {}, best_acc_val: {}'.format(epoch, ad_auc_val, best_val))
                    # load best model
                    checkpoint = torch.load(filename+'_checkpoint.pt')
                    model.load_state_dict(checkpoint['model_state_dict'])
                    opt.load_state_dict(checkpoint['optimizer_state_dict'])
                    break

        with torch.no_grad():
            model.eval()
            embed, prob_nc, prob_ad  = model(features, adj)

        # Node Selection
        if len(idx_train_ad) < args.max_budget * 2:
            idx_cand_an = torch.where(state_an==-1)[0]            

            if budget_ad != 0:
                if args.strategy_ad == 'random':
                    idx_selected_ad = query_random(budget_ad, idx_cand_an.tolist())
                elif args.strategy_ad == 'entropy':
                    idx_selected_ad = query_entropy(prob_ad, budget_ad, idx_cand_an.tolist())
                elif args.strategy_ad == 'topk_anomaly':
                    idx_selected_ad = query_topk_anomaly(prob_ad, budget_ad, idx_cand_an.tolist())
                elif args.strategy_ad == 'topk_medoids':
                    idx_selected_ad = query_topk_medoids(embed, prob_ad, budget_ad, idx_cand_an.tolist(), nb_classes)
                elif args.strategy_ad == 'return_anomaly':
                    idx_selected_ad = return_anomaly(budget_ad, idx_cand_an.tolist(), ano_label)
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
            csv_head = ["model", "seed", "dataset", "lr","embedding_dim", "init_num", "num_epochs", "nc_num", "strategy_ad", "alpha", "nc-acc", "nc-idacc", "nc-f1-micro", "nc-f1-macro", "ad-auc", "ad-f1-macro", "A-num", "N-num"]
            csv_write.writerow(csv_head)

    with open(des_path, 'a+') as f:
        csv_write = csv.writer(f)
        data_row = ['m1', args.seed, args.dataset, args.lr, args.embedding_dim, args.init_num, args.max_epoch, args.nc_num, args.strategy_ad, args.alpha, test_acc_comm, test_acc_nc_id, test_f1micro_comm, test_f1macro_comm, test_auc_ano, test_f1macro_ano, abnormal_num, normal_num]
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
    parser.add_argument('--alpha', type=float, default=0.9, help='balance parameter')
    parser.add_argument('--strategy_ad', type=str, default='topk_anomaly') # random uncertainty topk_anomaly
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--result_path', type=str, default='results/multitask')
    parser.add_argument('--nc_num', type=int, default=20)
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