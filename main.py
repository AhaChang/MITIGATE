import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import random
import os
import dgl
import argparse
import time
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score

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
        mix_auc_f = roc_auc_score(ano_labels.cpu().detach().numpy(),scores.cpu().detach().numpy())
        mix_ap = average_precision_score(ano_labels[idx_test].cpu().detach().numpy(),scores[idx_test].cpu().detach().numpy())

        # a_scores based
        test_auc_ad = roc_auc_score(ano_labels[idx_test].cpu().detach().numpy(),pred_ascores[idx_test].cpu().detach().numpy())
        test_auc_ad_f = roc_auc_score(ano_labels.cpu().detach().numpy(),pred_ascores.cpu().detach().numpy())
        test_ap_ad = average_precision_score(ano_labels[idx_test].cpu().detach().numpy(),pred_ascores[idx_test].cpu().detach().numpy())

        # n_scores based
        ent_auc = roc_auc_score(ano_labels[idx_test].cpu().detach().numpy(),pred_nscores[idx_test].cpu().detach().numpy())
        ent_auc_f = roc_auc_score(ano_labels.cpu().detach().numpy(),pred_nscores.cpu().detach().numpy())
        ent_ap = average_precision_score(ano_labels[idx_test].cpu().detach().numpy(),pred_nscores[idx_test].cpu().detach().numpy())

        print('Anomaly Detection Results')
        print('MIX-T-AUC', "{:.5f}".format(mix_auc),
              'MIX-F-AUC', "{:.5f}".format(mix_auc_f),
              'ANO-T-AUC', "{:.5f}".format(test_auc_ad),
              'ANO-F-AUC', "{:.5f}".format(test_auc_ad_f),
              'ENT-T-AUC', "{:.5f}".format(ent_auc),
              'ENT-F-AUC', "{:.5f}".format(ent_auc_f),
              'N_num', normal_num, 'A_num', abnormal_num)

        test_acc_nc = accuracy_score(labels[idx_test].cpu().detach().numpy(), prob_nc[idx_test].max(1)[1].cpu().detach().numpy())
        idx_test_id = idx_test[ano_labels[idx_test]==0]
        test_acc_nc_id = accuracy_score(labels[idx_test_id].cpu().detach().numpy(), prob_nc[idx_test_id].max(1)[1].cpu().detach().numpy())
        test_f1micro_nc = f1_score(labels[idx_test].cpu().detach().numpy(), prob_nc[idx_test].max(1)[1].cpu().detach().numpy(), average='micro')
        test_f1macro_nc = f1_score(labels[idx_test].cpu().detach().numpy(), prob_nc[idx_test].max(1)[1].cpu().detach().numpy(), average='macro')

        print('Community Detection Results')
        print('ACC', "{:.5f}".format(test_acc_nc), 
            'ID-ACC', "{:.5f}".format(test_acc_nc_id), 
            'F1-Micro', "{:.5f}".format(test_f1micro_nc), 
            'F1-Macro', "{:.5f}".format(test_f1macro_nc))
        
    return embed, prob_nc, prob_ad, test_acc_nc, test_acc_nc_id, test_f1micro_nc, test_f1macro_nc,\
        test_auc_ad, test_ap_ad, mix_auc,mix_auc_f, mix_ap, ent_auc, ent_auc_f,ent_ap, abnormal_num, normal_num 



def test_save(model, features, adj, labels, ano_labels, idx_train_ad, idx_train_nc, idx_test):

    embed, prob_nc, prob_ad, test_acc_nc, test_acc_nc_id, test_f1micro_nc, test_f1macro_nc,\
        test_auc_ad, test_ap_ad, mix_auc,mix_auc_f, mix_ap, ent_auc, ent_auc_f,ent_ap, abnormal_num, normal_num = test_model(model, features, adj, labels, ano_labels, idx_train_ad, idx_test)

    # Save results
    import csv
    des_path = args.result_path + '.csv'

    if not os.path.exists(des_path):
        with open(des_path,'w+') as f:
            csv_write = csv.writer(f)
            csv_head = ["model", "seed", "dataset", "lr","embedding_dim", "dropout", "max_budget", "num_epochs", "cluster_num", "strategy_ad","weight_tmp", "alpha", "beta","gamma","phi", "nc-acc", "nc-idacc", "nc-f1-micro", "nc-f1-macro", "ad-auc", "ad-ap", "ad-auc-m", "ad-auc-m-full", "ad-ap-m", "ad-auc-e", "ad-auc-e-full", "ad-ap-e", "A-num", "N-num", "NC-num"]
            csv_write.writerow(csv_head)

    with open(des_path, 'a+') as f:
        csv_write = csv.writer(f)
        data_row = ['MultiAAD', args.seed, args.dataset, args.lr, args.embedding_dim, args.dropout, (len(idx_train_ad)-len(idx_train_nc))//2, args.max_epoch, args.cluster_num, args.strategy_ad, args.weight_tmp, args.alpha, args.beta, args.gamma,args.phi, test_acc_nc, test_acc_nc_id, test_f1micro_nc, test_f1macro_nc, test_auc_ad,test_ap_ad, mix_auc, mix_auc_f,mix_ap, ent_auc, ent_auc_f,ent_ap, abnormal_num, normal_num, len(idx_train_nc)]
        csv_write.writerow(data_row)

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
    prefix = 'saved_models/MultiTask/' + args.dataset + str(args.device) + args.strategy_ad + str(args.cluster_num) +str(args.weight_tmp) +str(args.seed)+ '/'
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
    results_iter = {}

    for iter in range(iter_num + 1):
        
        model, opt = train_model(args, model, opt, features, adj, labels, ano_label, idx_train_nc, idx_train_ad, idx_val, filename)
        embed, prob_nc, prob_ad  = test_save(model, features, adj, labels, ano_label, idx_train_ad, idx_train_nc, idx_test)

        # Node Selection
        if len(idx_train_ad) < args.max_budget * 2 + idx_train_nc.shape[0]:
            weight = args.weight_tmp ** (len(idx_train_ad)-len(idx_train_nc))
            idx_cand_an = torch.where(state_an==-1)[0]

            if args.strategy_ad == 'random':
                idx_selected_ad = query_random(budget_ad, idx_cand_an.tolist())
            elif args.strategy_ad == 'entropy_nc':
                idx_selected_ad = query_entropy(prob_nc, budget_ad, idx_cand_an.tolist())
            elif args.strategy_ad == 'entropy_ad':
                idx_selected_ad = query_entropy(prob_ad, budget_ad, idx_cand_an.tolist())
            elif args.strategy_ad == 'topk_anomaly':
                idx_selected_ad = query_topk_anomaly(prob_ad, budget_ad, idx_cand_an.tolist())
            elif args.strategy_ad == 'diff':
                idx_selected_ad = query_diff(prob_nc, prob_ad, budget_ad, idx_cand_an.tolist())
            elif args.strategy_ad == 'nent_diff':
                idx_selected_ad = query_nent_diff(prob_nc, prob_ad, budget_ad, idx_cand_an.tolist(), weight)
            elif args.strategy_ad == 'nent_ascore':
                idx_selected_ad = query_nent_ascore(prob_nc, prob_ad, budget_ad, idx_cand_an.tolist(), weight)
            
            elif args.strategy_ad == 'medoids_diff':
                idx_selected_ad = query_medoids_diff(embed, prob_nc, prob_ad, budget_ad, idx_cand_an.tolist(), args.cluster_num)
            elif args.strategy_ad == 'medoids_spec_diff':
                idx_selected_ad = query_medoids_spec_diff(adj, embed, prob_nc, prob_ad, budget_ad, idx_cand_an.tolist(), args.cluster_num)
            elif args.strategy_ad == 'medoids_spec_nent_diff':
                idx_selected_ad = query_medoids_spec_nent_diff(adj, embed, prob_nc, prob_ad, budget_ad, idx_cand_an.tolist(), args.cluster_num, weight)
            elif args.strategy_ad == 'medoids_spec_nent':
                idx_selected_ad = query_medoids_spec_nent(adj, embed, prob_nc, budget_ad, idx_cand_an.tolist(), args.cluster_num)
            elif args.strategy_ad == 'medoids_nent_diff':
                idx_selected_ad = query_medoids_nent_diff(embed, prob_nc, prob_ad, budget_ad, idx_cand_an.tolist(), args.cluster_num, weight)
            
            elif args.strategy_ad == 'return_anomaly':
                idx_selected_ad = return_anomaly(budget_ad, idx_cand_an.tolist(), ano_label.cpu().numpy())
            else:
                raise ValueError("AD Strategy is not defined")

            print('Selected! ')

            # Save embeddings and selected node index
            results_iter[iter]={'embeds':embed.detach().cpu().numpy(),'cur_selected':idx_selected_ad,'pre_selected':idx_train_ad.detach().cpu().numpy()}

            # Update state
            state_an[idx_selected_ad] = 1
            idx_train_ad = torch.cat((idx_train_ad, torch.tensor(idx_selected_ad).cuda()))

    # 保存字典到 Pickle 文件
    import pickle
    with open('results_iter/'+args.strategy_ad + '_' + args.dataset+'.pkl', 'wb') as pickle_file:
        pickle.dump(results_iter, pickle_file)


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
    parser.add_argument('--gamma', type=float, default=1, help='unsupervised loss weight')
    parser.add_argument('--phi', type=float, default=1.25, help='anoamly score weight')
    parser.add_argument('--weight_tmp', type=float, default=0.95)
    parser.add_argument('--strategy_ad', type=str, default='medoids_spec_nent_diff')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--result_path', type=str, default='results/multitask')
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