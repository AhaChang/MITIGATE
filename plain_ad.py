import numpy as np
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

def train_ad_model(args, model_ad, opt_ad, features, adj, ano_labels, idx_train_ad, idx_val, filename):
    xent = nn.CrossEntropyLoss()
    # Train anomaly detection model
    patience = 20
    early_stopping = 20
    cur_p = 0
    best_loss = 1e9
    best_val = 0
    for epoch in range(args.max_epoch):
        model_ad.train()
        opt_ad.zero_grad()

        embed, prob_ad = model_ad(features, adj)
        
        loss_ad = xent(prob_ad[idx_train_ad], ano_labels[idx_train_ad]) 

        loss_ad.backward()
        opt_ad.step()

        with torch.no_grad():
            model_ad.eval()
            embed, prob_ad = model_ad(features, adj)
            
            val_loss = xent(prob_ad[idx_val], ano_labels[idx_val])
            
            ad_auc_val = auc(prob_ad[idx_val], ano_labels[idx_val])

            print('Train Loss', "{:.5f}".format(val_loss.item()),
                    'AD-AUC:', "{:.5f}".format(ad_auc_val))

            # Save model untill loss does not decrease
            if epoch > early_stopping:
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_iter = epoch
                    best_val = ad_auc_val
                    cur_p = 0
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model_ad.state_dict(),
                        'optimizer_state_dict': opt_ad.state_dict(),
                    }, filename+'_checkpoint_ad.pt')
                else:
                    cur_p += 1

            if cur_p > patience or epoch+1 >= args.max_epoch:
                print('epoch: {}, acc_val: {}, best_acc_val: {}'.format(epoch, ad_auc_val, best_val))
                # load best model
                checkpoint = torch.load(filename+'_checkpoint_ad.pt')
                model_ad.load_state_dict(checkpoint['model_state_dict'])
                opt_ad.load_state_dict(checkpoint['optimizer_state_dict'])
                break

    return model_ad, opt_ad


def test_ad_model(model_ad, features, adj, ano_labels, idx_train_ad, idx_test):
    with torch.no_grad():
        model_ad.eval()
        embed_ad, prob_ad = model_ad(features, adj)

        test_auc_ano = auc(prob_ad[idx_test], ano_labels[idx_test])
        test_acc_ano = accuracy(prob_ad[idx_test], ano_labels[idx_test])
        test_f1micro_ano, test_f1macro_ano = f1(prob_ad[idx_test], ano_labels[idx_test])

        abnormal_num = int(ano_labels[idx_train_ad].sum().item())
        normal_num = len(idx_train_ad) - abnormal_num

        print('Anomaly Detection Results')
        print('ACC', "{:.5f}".format(test_acc_ano), 
            'F1-Micro', "{:.5f}".format(test_f1micro_ano), 
            'F1-Macro', "{:.5f}".format(test_f1macro_ano), 
            'AUC', "{:.5f}".format(test_auc_ano),
            'N_num', normal_num,
            'A_num', abnormal_num)
        
    return embed_ad, prob_ad, test_auc_ano, test_f1macro_ano, test_f1micro_ano, abnormal_num, normal_num

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

    model_ad = ADModel(ft_size, args.embedding_dim, 2, dropout=0.6)
    opt_ad = torch.optim.Adam(model_ad.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    if torch.cuda.is_available():
        print('Using CUDA')
        model_ad.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        ano_labels = ano_labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    timestamp = time.time()
    prefix = 'saved_models/new/' + args.dataset + str(args.device) + '/'
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    filename = prefix + str(args.seed)+ '_fixncb_'+ str(timestamp)

    # Init Selection
    idx_train_ad = init_category(args.init_num, idx_train, ano_labels)

    # Init annotation state
    state_an = torch.zeros(features.shape[0]) - 1
    state_an = state_an.long()
    state_an[idx_train_ad] = 1
    state_an[idx_val] = 2
    state_an[idx_train] = 2

    # Train model
    budget_ad = int(2 * (args.max_budget - args.init_num) / args.iter_num)

    for iter in range(args.iter_num + 1):
        # Train anomaly detection model
        model_ad, opt_ad = train_ad_model(args, model_ad, opt_ad, features, adj, ano_labels, idx_train_ad, idx_val, filename)
        embed_ad, prob_ad, test_auc_ano, test_f1macro_ano, test_f1micro_ano, abnormal_num, normal_num = test_ad_model(model_ad, features, adj, ano_labels, idx_train_ad, idx_test)
                    

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
                elif args.strategy_ad == 'return_anomaly':
                    idx_selected_ad = return_anomaly(budget_ad, idx_cand_an.tolist(), ano_labels)
                else:
                    raise ValueError("AD Strategy is not defined")

                # Update state
                state_an[idx_selected_ad] = 1
                idx_train_ad = torch.cat((idx_train_ad, torch.tensor(idx_selected_ad).cuda()))

    # Test model
    print('Final Results:')
    embed_ad, prob_ad, test_auc_ano, test_f1macro_ano, test_f1micro_ano, abnormal_num, normal_num = test_ad_model(model_ad, features, adj, ano_labels, idx_train_ad, idx_test)
            

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
        data_row = ['plain_ad', args.seed, args.dataset, args.init_num, args.max_epoch, args.strategy_ad, test_auc_ano, test_f1macro_ano, abnormal_num, normal_num]
        csv_write.writerow(data_row)




if __name__ == '__main__':
    # Set argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pubmed')  # 'BlogCatalog'  'Flickr'  'ACM'  'cora'  'citeseer'  'pubmed'
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--seed', type=int, default=255)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--init_num', type=int, default=2)
    parser.add_argument('--max_epoch', type=int, default=300)
    parser.add_argument('--max_budget', type=int, default=20)
    parser.add_argument('--iter_num', type=int, default=9)
    parser.add_argument('--strategy_ad', type=str, default='random') # random uncertainty topk_anomaly
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--result_path', type=str, default='results/plain_ad')

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