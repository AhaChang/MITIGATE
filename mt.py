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

def entropy_loss(p, epsilon=1e-10):
    """
    Compute the entropy of a distribution for each sample in a batch.
    :param p: Tensor representing the predicted distribution for each sample
    :param epsilon: Small value to ensure numerical stability
    :return: Entropy values for each sample in the batch
    """
    p = p + epsilon
    batch_entropy = -torch.sum(p * torch.log(p), dim=-1)
    return batch_entropy.mean()


def train_nc_model(args, model_nc, opt_nc, features, adj, labels, ano_labels, model_ad, idx_train_nc, idx_train_ad, idx_val, filename):
    xent = nn.CrossEntropyLoss()
    # Train node classification model
    patience = 20
    early_stopping = 20
    best_val = 0
    cur_p = 0
    best_loss = 1e9
    for epoch in range(args.max_epoch):
        model_nc.train()
        opt_nc.zero_grad()

        embed, prob_nc = model_nc(features, adj)
        
        loss_sup = xent(prob_nc[idx_train_nc], labels[idx_train_nc]) 

        with torch.no_grad():
            model_ad.eval()
            embed_ad, prob_ad = model_ad(torch.cat((features, prob_nc), dim=1), adj)
            pred_ad = prob_ad.argmax(1)

        if args.loss == 'div':
            loss_un = entropy_loss(F.softmax(prob_nc,dim=1)[torch.where(pred_ad==0)[0]]) + 1/entropy_loss(F.softmax(prob_nc,dim=1)[torch.where(pred_ad==1)[0]])
            # loss_un = entropy_loss(F.softmax(prob_nc,dim=1)[torch.where(ano_labels[idx_train_ad]==0)[0]])/entropy_loss(F.softmax(prob_nc,dim=1)[torch.where(ano_labels[idx_train_ad]==1)[0]])
        elif args.loss == 'sum_div':
            loss_un = entropy_loss(F.softmax(prob_nc,dim=1)[torch.where(ano_labels[idx_train_ad]==0)[0]])+ 1/entropy_loss(F.softmax(prob_nc,dim=1)[torch.where(ano_labels[idx_train_ad]==1)[0]])
        elif args.loss == 'sum':
            loss_un = entropy_loss(F.softmax(prob_nc,dim=1)[torch.where(ano_labels[idx_train_ad]==0)[0]]) - entropy_loss(F.softmax(prob_nc,dim=1)[torch.where(ano_labels[idx_train_ad]==1)[0]])
        else:
            loss_un = 0

        loss = loss_sup + args.w1 * loss_un

        loss.backward()
        opt_nc.step()

        with torch.no_grad():
            model_nc.eval()
            model_ad.eval()
            embed, prob_nc = model_nc(features, adj)
            embed_ad, prob_ad = model_ad(torch.cat((features, prob_nc), dim=1), adj)
            pred_ad = prob_ad.argmax(1)
            
            val_loss_sup = xent(prob_nc[idx_val], labels[idx_val])

            if args.loss == 'div':
                val_loss_un = entropy_loss(F.softmax(prob_nc,dim=1)[torch.where(pred_ad==0)[0]]) + 1/entropy_loss(F.softmax(prob_nc,dim=1)[torch.where(pred_ad==1)[0]])
                # val_loss_un = entropy_loss(F.softmax(prob_nc,dim=1)[torch.where(ano_labels[idx_val]==0)[0]])/entropy_loss(F.softmax(prob_nc,dim=1)[torch.where(ano_labels[idx_val]==1)[0]])
            elif args.loss == 'sum_div':
                val_loss_un = entropy_loss(F.softmax(prob_nc,dim=1)[torch.where(ano_labels[idx_val]==0)[0]])+ 1/entropy_loss(F.softmax(prob_nc,dim=1)[torch.where(ano_labels[idx_val]==1)[0]])
            elif args.loss == 'sum':
                val_loss_un = entropy_loss(F.softmax(prob_nc,dim=1)[torch.where(ano_labels[idx_val]==0)[0]]) - entropy_loss(F.softmax(prob_nc,dim=1)[torch.where(ano_labels[idx_val]==1)[0]])
            else:
                val_loss_un = 0
            
            val_loss = val_loss_sup + args.w1 * val_loss_un

            nc_acc_val = accuracy(prob_nc[idx_val], labels[idx_val])

            print('Train Loss', "{:.5f}".format(loss.item()),
                    'NC-ACC:', "{:.5f}".format(nc_acc_val))

            # Save model untill loss does not decrease
            if epoch > early_stopping:
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_iter = epoch
                    best_val = nc_acc_val
                    cur_p = 0
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model_nc.state_dict(),
                        'optimizer_state_dict': opt_nc.state_dict(),
                    }, filename+'_checkpoint.pt')
                else:
                    cur_p += 1

            if cur_p > patience or epoch+1 >= args.max_epoch:
                print('epoch: {}, acc_val: {}, best_acc_val: {}'.format(epoch, nc_acc_val, best_val))
                # load best model
                checkpoint = torch.load(filename+'_checkpoint.pt')
                model_nc.load_state_dict(checkpoint['model_state_dict'])
                opt_nc.load_state_dict(checkpoint['optimizer_state_dict'])
                break

    return model_nc, opt_nc

def train_ad_model(args, model_ad, opt_ad, prob_nc, features, adj, ano_labels, idx_train_ad, idx_val, filename):
    xent = nn.CrossEntropyLoss()
    weight = (1-ano_labels[idx_train_ad]).sum()/ano_labels[idx_train_ad].sum()
    # Train anomaly detection model
    patience = 20
    early_stopping = 20
    cur_p = 0
    best_loss = 1e9
    best_val = 0
    for epoch in range(args.max_epoch):
        model_ad.train()
        opt_ad.zero_grad()

        embed, prob_ad = model_ad(torch.cat((features, prob_nc), dim=1), adj)
        
        # loss_ad = xent(prob_ad[idx_train_ad], ano_labels[idx_train_ad]) 
        loss_ad = F.cross_entropy(prob_ad[idx_train_ad], ano_labels[idx_train_ad], weight=torch.tensor([1., weight]).cuda())

        loss_ad.backward()
        opt_ad.step()

        with torch.no_grad():
            model_ad.eval()
            embed, prob_ad = model_ad(torch.cat((features, prob_nc), dim=1), adj)
            
            ad_auc_val = auc(prob_ad[idx_val], ano_labels[idx_val])
            ad_f1micro_val, ad_f1macro_val = f1(prob_ad[idx_val], ano_labels[idx_val])

            print('AD-AUC:', "{:.5f}".format(ad_auc_val),
                  'AD-F1-Macro:', "{:.5f}".format(ad_f1macro_val))

            # Save model untill loss does not decrease
            if epoch > early_stopping:
                if ad_f1macro_val > best_val:
                    best_iter = epoch
                    best_val = ad_f1macro_val
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

def test_nc_model(model_nc, features, adj, labels, ano_labels, idx_test):
    with torch.no_grad():
        model_nc.eval()
        embed_nc, prob_nc = model_nc(features, adj)

        test_acc_nc = accuracy(prob_nc[idx_test], labels[idx_test])
        idx_test_id = idx_test[ano_labels[idx_test]==0]
        test_acc_nc_id = accuracy(prob_nc[idx_test_id], labels[idx_test_id])
        test_f1micro_nc, test_f1macro_nc = f1(prob_nc[idx_test], labels[idx_test])

        print('Node Classification Results')
        print('ACC', "{:.5f}".format(test_acc_nc), 
            'ID-ACC', "{:.5f}".format(test_acc_nc_id), 
            'F1-Micro', "{:.5f}".format(test_f1micro_nc), 
            'F1-Macro', "{:.5f}".format(test_f1macro_nc))
        
    return embed_nc, prob_nc, test_acc_nc, test_acc_nc_id, test_f1macro_nc, test_f1micro_nc

def test_ad_model(model_ad, features, adj, ano_labels, prob_nc, idx_train_ad, idx_test):
    with torch.no_grad():
        model_ad.eval()
        embed_ad, prob_ad = model_ad(torch.cat((features, prob_nc), dim=1), adj)

        test_auc_ano = auc(prob_ad[idx_test], ano_labels[idx_test])
        test_acc_ano = accuracy(prob_ad[idx_test], ano_labels[idx_test])
        test_f1micro_ano, test_f1macro_ano = f1(prob_ad[idx_test], ano_labels[idx_test])
        test_pre = precision(prob_ad[idx_test], ano_labels[idx_test])
        test_rec = recall(prob_ad[idx_test], ano_labels[idx_test])

        abnormal_num = int(ano_labels[idx_train_ad].sum().item())
        normal_num = len(idx_train_ad) - abnormal_num

        print('Anomaly Detection Results')
        print('ACC', "{:.5f}".format(test_acc_ano), 
            'F1-Micro', "{:.5f}".format(test_f1micro_ano), 
            'F1-Macro', "{:.5f}".format(test_f1macro_ano), 
            'AUC', "{:.5f}".format(test_auc_ano),
            'N_num', normal_num,
            'A_num', abnormal_num)
        
    return embed_ad, prob_ad, test_auc_ano, test_f1macro_ano, test_f1micro_ano, test_pre, test_rec, abnormal_num, normal_num

def main(args):
    # Load and preprocess data
    adj, features, labels, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels = load_mat_f(args.dataset)

    features, _ = preprocess_features(features)
    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = labels.max() + 1

    adj = normalize_adj(adj)
    adj = (adj + sp.eye(adj.shape[0])).todense()

    # Init Selection
    idx_train_nc = np.loadtxt("splited_data/"+args.dataset+"/nc", dtype=int)
    idx_train_ad = np.loadtxt("splited_data/"+args.dataset+"/init", dtype=int)

    features = torch.FloatTensor(features)
    adj = torch.FloatTensor(adj)
    labels = torch.LongTensor(labels)
    ano_labels = torch.LongTensor(ano_labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    idx_train_nc = torch.LongTensor(idx_train_nc)
    idx_train_ad = torch.LongTensor(idx_train_ad)

    model_nc = NCModel(ft_size, args.embedding_dim, nb_classes, dropout=0.6)
    opt_nc = torch.optim.Adam(model_nc.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model_ad = ADModel(ft_size+nb_classes, args.embedding_dim, 2, dropout=0.6)
    opt_ad = torch.optim.Adam(model_ad.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    if torch.cuda.is_available():
        print('Using CUDA')
        model_nc.cuda()
        model_ad.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        ano_labels = ano_labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
        idx_train_ad = idx_train_ad.cuda()
        idx_train_nc = idx_train_nc.cuda()


    timestamp = time.time()
    prefix = 'saved_models/new/' + args.dataset + str(args.device) + args.loss + '/'
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    filename = prefix + str(args.seed)+ '_fixncb_'+ str(timestamp)

    

    # Init annotation state
    state_an = torch.zeros(features.shape[0]) - 1
    state_an = state_an.long()
    state_an[idx_train_ad] = 1
    state_an[idx_val] = 2
    state_an[idx_test] = 2

    # Train model    
    # pred_ad = None
    budget_ad = int(2 * (args.max_budget - args.init_num) / args.iter_num)
    for iter in range(args.iter_num + 1):

        # Train node classification model
        model_nc, opt_nc = train_nc_model(args, model_nc, opt_nc, features, adj, labels, ano_labels, model_ad, idx_train_nc, idx_train_ad, idx_val, filename)
        embed_nc, prob_nc, test_acc_nc, test_acc_nc_id, test_f1macro_nc, test_f1micro_nc = test_nc_model(model_nc, features, adj, labels, ano_labels, idx_test)

        # Train anomaly detection model
        model_ad, opt_ad = train_ad_model(args, model_ad, opt_ad, prob_nc, features, adj, ano_labels, idx_train_ad, idx_val, filename)
        embed_ad, prob_ad, test_auc_ano, test_f1macro_ano, test_f1micro_ano, test_pre, test_rec, abnormal_num, normal_num = test_ad_model(model_ad, features, adj, ano_labels, prob_nc, idx_train_ad, idx_test)
        
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
                    idx_selected_ad = query_topk_medoids(embed_nc+embed_ad, prob_ad, budget_ad, idx_cand_an.tolist(), nb_classes)
                elif args.strategy_ad == 'topk_medoids_1':
                    idx_selected_ad = query_topk_medoids(torch.cat((embed_nc,embed_ad),dim=1), prob_ad, budget_ad, idx_cand_an.tolist(), nb_classes)
                elif args.strategy_ad == 'topk_medoids_2':
                    idx_selected_ad = query_topk_medoids_m(embed_nc+embed_ad, prob_ad, budget_ad, idx_cand_an.tolist(), nb_classes)
                elif args.strategy_ad == 'topk_medoids_3':
                    idx_selected_ad = query_topk_medoids_m(torch.cat((embed_nc,embed_ad),dim=1), prob_ad, budget_ad, idx_cand_an.tolist(), nb_classes)    
                elif args.strategy_ad == 'nc_entropy':
                    idx_selected_ad = query_nc_entropy(prob_nc, budget_ad, idx_cand_an.tolist())
                elif args.strategy_ad == 'nc_minmax':
                    idx_selected_ad = query_nc_minmax(prob_nc, budget_ad, idx_cand_an.tolist())
                elif args.strategy_ad == 'return_anomaly':
                    idx_selected_ad = return_anomaly(budget_ad, idx_cand_an.tolist(), ano_labels)
                else:
                    raise ValueError("AD Strategy is not defined")

                # Update state
                state_an[idx_selected_ad] = 1
                idx_train_ad = torch.cat((idx_train_ad, torch.tensor(idx_selected_ad).cuda()))
            


    # Test model
    print('Final Results:')
    embed_nc, prob_nc, test_acc_nc, test_acc_nc_id, test_f1macro_nc, test_f1micro_nc = test_nc_model(model_nc, features, adj, labels, ano_labels, idx_test)
    embed_ad, prob_ad, test_auc_ano, test_f1macro_ano, test_f1micro_ano, test_pre, test_rec, abnormal_num, normal_num = test_ad_model(model_ad, features, adj, ano_labels, prob_nc, idx_train_ad, idx_test)

    # Save results
    import csv
    des_path = args.result_path + '.csv'

    if not os.path.exists(des_path):
        with open(des_path,'w+') as f:
            csv_write = csv.writer(f)
            csv_head = ["model", "seed", "dataset", "init_num", "num_epochs", "loss_type", "strategy_ad", "w1", "nc-acc", "nc-idacc", "nc-f1-micro", "nc-f1-macro", "ad-auc", "ad-f1-macro", "ad-pre-macro", "ad-rec-macro", "A-num", "N-num"]
            csv_write.writerow(csv_head)

    with open(des_path, 'a+') as f:
        csv_write = csv.writer(f)
        data_row = ['m1', args.seed, args.dataset, args.init_num, args.max_epoch, args.loss, args.strategy_ad, args.w1, test_acc_nc, test_acc_nc_id, test_f1micro_nc, test_f1macro_nc, test_auc_ano, test_f1macro_ano, test_pre, test_rec, abnormal_num, normal_num]
        csv_write.writerow(data_row)




if __name__ == '__main__':
    # Set argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='BlogCatalog')  # 'BlogCatalog'  'Flickr'  'cora'  'citeseer'  'pubmed'
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--seed', type=int, default=255)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--init_num', type=int, default=2)
    parser.add_argument('--max_epoch', type=int, default=300)
    parser.add_argument('--max_budget', type=int, default=20)
    parser.add_argument('--iter_num', type=int, default=9)
    parser.add_argument('--strategy_ad', type=str, default='topk_medoids') # random uncertainty topk_anomaly
    parser.add_argument('--nc_num', type=int, default=20)
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--result_path', type=str, default='results/multitask')

    parser.add_argument('--w1', type=float, default=0.1)
    parser.add_argument('--loss', type=str, default='div')

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