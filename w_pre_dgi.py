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

def train_model(args, model, opt, features, adj, labels, ano_labels, idx_train_nc, idx_train_ad, idx_val, filename):
    xent = nn.CrossEntropyLoss()
    if ano_labels[idx_train_ad].sum() == 0:
        weight = 1.
    else:
        weight = (1-ano_labels[idx_train_ad]).sum()/ano_labels[idx_train_ad].sum()

    patience = 20
    early_stopping = 20
    best_val = 0
    cur_p = 0

    for epoch in range(args.max_epoch):
        model.train()
        opt.zero_grad()

        embed, prob_nc, prob_ad = model(features, adj)
        pred_ad = prob_ad.argmax(1)

        loss_comm = xent(prob_nc[idx_train_nc], labels[idx_train_nc]) # node classification
        loss_an = F.cross_entropy(prob_ad[idx_train_ad], ano_labels[idx_train_ad], weight=torch.tensor([1., weight]).cuda())

        # only use selected samples as unsupervised loss
        mask = ~torch.isin(idx_train_ad, idx_train_nc)
        idx_s_ad = idx_train_ad[mask]

        if args.un_loss_type == 'div':
            loss_un = entropy_loss(F.softmax(prob_nc,dim=1)[torch.where(pred_ad==0)[0]]) 
            if torch.where(pred_ad==1)[0].shape[0] != 0:
                loss_un = loss_un + 1/entropy_loss(F.softmax(prob_nc,dim=1)[torch.where(pred_ad==1)[0]])
        elif args.un_loss_type == 'sum':
            loss_un = entropy_loss(F.softmax(prob_nc,dim=1)[torch.where(pred_ad==0)[0]])
            if torch.where(pred_ad==1)[0].shape[0] != 0:
                loss_un = loss_un - entropy_loss(F.softmax(prob_nc,dim=1)[torch.where(pred_ad==1)[0]])
        elif args.un_loss_type == 'sup_div':
            loss_un = 0.
            if torch.where(ano_labels[idx_s_ad]==0)[0].shape[0] != 0:
                loss_un = loss_un + entropy_loss(F.softmax(prob_nc,dim=1)[idx_s_ad][torch.where(ano_labels[idx_s_ad]==0)[0]]) 
            if torch.where(ano_labels[idx_s_ad]==1)[0].shape[0] != 0:
                loss_un = loss_un + 1/entropy_loss(F.softmax(prob_nc,dim=1)[idx_s_ad][torch.where(ano_labels[idx_s_ad]==1)[0]])
        elif args.un_loss_type == 'sup_sum':
            loss_un = 0.
            if torch.where(ano_labels[idx_s_ad]==0)[0].shape[0] != 0:
                loss_un = loss_un + entropy_loss(F.softmax(prob_nc,dim=1)[idx_s_ad][torch.where(ano_labels[idx_s_ad]==0)[0]]) 
            if torch.where(ano_labels[idx_s_ad]==1)[0].shape[0] != 0:
                loss_un = loss_un - entropy_loss(F.softmax(prob_nc,dim=1)[idx_s_ad][torch.where(ano_labels[idx_s_ad]==1)[0]])
        else:
            loss_un = 0.

        loss_total = args.alpha * loss_comm + args.beta * loss_un + args.gamma * loss_an

        loss_total.backward()
        opt.step()

        with torch.no_grad():
            model.eval()
            embed, prob_nc, prob_ad  = model(features, adj)
            pred_ad = prob_ad.argmax(1)

            idx_val_id = idx_val[ano_labels[idx_val]==0]

            ad_auc_val = auc(prob_ad[idx_val], ano_labels[idx_val])
            ad_f1micro_val, ad_f1macro_val = f1(prob_ad[idx_val], ano_labels[idx_val])

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
                if ad_auc_val + nc_idacc_val > best_val:
                    best_iter = epoch
                    best_val = ad_auc_val + nc_idacc_val
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
                checkpoint = torch.load(filename+'_checkpoint.pt')
                model.load_state_dict(checkpoint['model_state_dict'])
                opt.load_state_dict(checkpoint['optimizer_state_dict'])
                break
    return model, opt


def test_model(model, features, adj, labels, ano_labels, idx_train_ad, idx_test):
    with torch.no_grad():
        model.eval()
        embed, prob_nc, prob_ad = model(features, adj)

        test_auc_ad = auc(prob_ad[idx_test], ano_labels[idx_test])
        test_acc_ad = accuracy(prob_ad[idx_test], ano_labels[idx_test])
        test_f1micro_ad, test_f1macro_ad = f1(prob_ad[idx_test], ano_labels[idx_test])
        test_pre_ad = precision(prob_ad[idx_test], ano_labels[idx_test])
        test_rec_ad = recall(prob_ad[idx_test], ano_labels[idx_test])

        abnormal_num = int(ano_labels[idx_train_ad].sum().item())
        normal_num = len(idx_train_ad) - abnormal_num - 20*len(set(labels.tolist()))

        print('Anomaly Detection Results')
        print('ACC', "{:.5f}".format(test_acc_ad), 
            'F1-Micro', "{:.5f}".format(test_f1micro_ad), 
            'F1-Macro', "{:.5f}".format(test_f1macro_ad), 
            'AUC', "{:.5f}".format(test_auc_ad),
            'Recall', "{:.5f}".format(test_rec_ad),
            'Precision', "{:.5f}".format(test_pre_ad),
            'N_num', normal_num,
            'A_num', abnormal_num)

        

        from sklearn.metrics import roc_auc_score
        output = prob_nc
        entropy = get_entropy_score(output)
        pred_ascores = torch.softmax(prob_ad, dim=1)[:,1]
        scores = (entropy-entropy.min())/(entropy.max()-entropy.min()) + (pred_ascores-pred_ascores.min())/(pred_ascores.max()-pred_ascores.min())
        mix_auc = roc_auc_score(ano_labels[idx_test].cpu().detach().numpy(),scores[idx_test].cpu().detach().numpy())
        mix_auc_f = roc_auc_score(ano_labels.cpu().detach().numpy(),scores.cpu().detach().numpy())


        print('Anomaly Detection Results')
        print('F-AUC', "{:.5f}".format(mix_auc),
              'T-AUC', "{:.5f}".format(mix_auc_f))
            
        test_acc_nc = accuracy(prob_nc[idx_test], labels[idx_test])
        idx_test_id = idx_test[ano_labels[idx_test]==0]
        test_acc_nc_id = accuracy(prob_nc[idx_test_id], labels[idx_test_id])
        test_f1micro_nc, test_f1macro_nc = f1(prob_nc[idx_test], labels[idx_test])

        print('Community Detection Results')
        print('ACC', "{:.5f}".format(test_acc_nc), 
            'ID-ACC', "{:.5f}".format(test_acc_nc_id), 
            'F1-Micro', "{:.5f}".format(test_f1micro_nc), 
            'F1-Macro', "{:.5f}".format(test_f1macro_nc))
        
    return embed, prob_nc, prob_ad, test_acc_nc, test_acc_nc_id, test_f1micro_nc, test_f1macro_nc, test_auc_ad, test_f1macro_ad, test_pre_ad, test_rec_ad, abnormal_num, normal_num 

def pre_dgi(adj, features, hidden, dgi_epoch, dgi_lr, dgi_weight_decay, filename):
    cnt_wait = 0
    patience = 20
    batch_size = 1
    best_loss = 1e9
    nb_nodes = adj.shape[0]

    b_xent = torch.nn.BCEWithLogitsLoss()

    DGI_model = DGI(features.shape[-1], hidden, 'prelu')
        
    opt = torch.optim.Adam(DGI_model.parameters(), lr=dgi_lr,
                            weight_decay=dgi_weight_decay)
    DGI_model.train()
    print('Training unsupervised model.....')
    for i in range(dgi_epoch):
        opt.zero_grad()

        perm_idx = np.random.permutation(nb_nodes)
        shuf_fts = features[:, perm_idx, :]

        lbl_1 = torch.ones(batch_size, nb_nodes)
        lbl_2 = torch.zeros(batch_size, nb_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1)
        if torch.cuda.is_available():
            DGI_model.cuda()
            shuf_fts = shuf_fts.cuda()
            lbl = lbl.cuda()

        logits = DGI_model(features, shuf_fts, adj, True, None, None, None)

        loss = b_xent(logits, lbl)

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_iter = i
            cnt_wait = 0
            torch.save(DGI_model.state_dict(), filename + 'best_dgi_inc11.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            print('Early Stopping')
            break

        loss.backward()
        opt.step()

    print(f'Finished training unsupervised model, Loading {best_iter}th epoch')
    DGI_model.load_state_dict(torch.load(filename + 'best_dgi_inc11.pkl'))

    return DGI_model
    
def main(args):
    # Load and preprocess data
    adj, features, labels, idx_train, idx_val, idx_test, ano_label = load_mat_f(args.dataset)

    features, _ = preprocess_features(features)
    nb_classes = labels.max() + 1

    adj = normalize_adj(adj)
    adj = (adj + sp.eye(adj.shape[0])).todense()
    
    # Init Selection
    idx_train_nc = np.loadtxt("splited_data/"+args.dataset+"/nc", dtype=int)
    idx_train_ad = np.loadtxt("splited_data/"+args.dataset+"/nc", dtype=int)

    # Init annotation state
    state_an = torch.zeros(features.shape[0]) - 1
    state_an = state_an.long()
    state_an[idx_train_ad] = 1
    state_an[idx_val] = 2
    state_an[idx_test] = 2

    timestamp = time.time()
    prefix = 'saved_models/MultiTask/' + args.dataset + str(args.device) + args.strategy_ad + '1/'
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    filename = prefix + str(args.seed)+ '_multi_'+ str(timestamp)

    features = torch.FloatTensor(features)
    adj = torch.FloatTensor(adj)
    labels = torch.LongTensor(labels)
    ano_label = torch.LongTensor(ano_label)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    idx_train_nc = torch.LongTensor(idx_train_nc)
    idx_train_ad = torch.LongTensor(idx_train_ad)


    if torch.cuda.is_available():
        print('Using CUDA')
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        ano_label = ano_label.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
        idx_train_ad = idx_train_ad.cuda()
        idx_train_nc = idx_train_nc.cuda()

    if args.dataset in ['AmazonComputers', 'Flickr', 'AmazonPhoto']:
        hidden = 256
    else: # BlogCatalog , 'cora'
        hidden = 512

    # pre-train DGI
    dgi_epoch = 1000
    dgi_lr = 0.001
    dgi_weight_decay = 0.0
    dgi_model = pre_dgi(adj, features[np.newaxis], hidden, dgi_epoch, dgi_lr, dgi_weight_decay, filename)
    dgi_embeds, _ = dgi_model.embed(features[np.newaxis], adj, True, None)
    dgi_embeds = torch.squeeze(dgi_embeds, 0)

    if args.model == 'Model1':
        features = torch.cat((dgi_embeds,features),dim=1)
        model = Model1(features.shape[1], args.embedding_dim, nb_classes, dropout=args.dropout)
    elif args.model == 'Model2':
        features = dgi_embeds
        model = Model2(features.shape[1], args.embedding_dim, nb_classes, dropout=args.dropout)
    elif args.model == 'Model0':
        model = Model1(features.shape[1], args.embedding_dim, nb_classes, dropout=args.dropout)
    else:
        raise ValueError("Model is not defined")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if torch.cuda.is_available():
        model.cuda()


    # Train model
    budget_ad = args.iter_budget * 2
    iter_num = args.max_budget // args.iter_budget

    for iter in range(iter_num + 1):

        model, opt = train_model(args, model, opt, features, adj, labels, ano_label, idx_train_nc, idx_train_ad, idx_val, filename)


        with torch.no_grad():
            model.eval()
            embed, prob_nc, prob_ad  = model(features, adj)

        # Node Selection
        if len(idx_train_ad) < args.max_budget * 2 + idx_train_nc.shape[0]:

            idx_cand_an = torch.where(state_an==-1)[0]

            if args.strategy_ad == 'random':
                idx_selected_ad = query_random(budget_ad, idx_cand_an.tolist())
            elif args.strategy_ad == 'entropy_nc':
                idx_selected_ad = query_entropy(prob_nc, budget_ad, idx_cand_an.tolist())
            elif args.strategy_ad == 'entropy_ad':
                idx_selected_ad = query_entropy(prob_ad, budget_ad, idx_cand_an.tolist())
            elif args.strategy_ad == 'topk_anomaly':
                idx_selected_ad = query_topk_anomaly(prob_ad, budget_ad, idx_cand_an.tolist())
            elif args.strategy_ad == 'k_medoids':
                weight = args.weight_tmp ** (len(idx_train_ad)-len(idx_train_nc))
                idx_selected_ad = query_top2k_medoids_s(features, prob_nc, prob_ad, budget_ad, idx_cand_an.tolist(), nb_classes, weight)
            elif args.strategy_ad == 'topk_mix':
                weight = 0.5
                idx_selected_ad = query_topk_nent_ascore(prob_nc, prob_ad, budget_ad, idx_cand_an.tolist(), weight)
            elif args.strategy_ad == 'topk_w_mix':
                weight = args.weight_tmp ** (len(idx_train_ad)-len(idx_train_nc))
                idx_selected_ad = query_topk_nent_ascore(prob_nc, prob_ad, budget_ad, idx_cand_an.tolist(), weight)
            elif args.strategy_ad == 'topk_c_mix':
                weight = args.weight_tmp ** (len(idx_train_ad)-len(idx_train_nc))
                idx_selected_ad = query_topk_nent_aent(prob_nc, prob_ad, budget_ad, idx_cand_an.tolist(), weight)
            elif args.strategy_ad == 'return_anomaly':
                idx_selected_ad = return_anomaly(budget_ad, idx_cand_an.tolist(), ano_label.cpu().numpy())
            else:
                raise ValueError("AD Strategy is not defined")

            # Update state
            state_an[idx_selected_ad] = 1
            idx_train_ad = torch.cat((idx_train_ad, torch.tensor(idx_selected_ad).cuda()))

            print('Selected! ')

        
    # Test model
    embed, prob_nc, prob_ad, test_acc_nc, test_acc_nc_id, test_f1micro_nc, test_f1macro_nc, test_auc_ad, test_f1macro_ad, test_pre_ad, test_rec_ad, abnormal_num, normal_num = test_model(model, features, adj, labels, ano_label, idx_train_ad, idx_test)

    # Save results
    import csv
    des_path = args.result_path + '.csv'

    if not os.path.exists(des_path):
        with open(des_path,'w+') as f:
            csv_write = csv.writer(f)
            csv_head = ["model", "seed", "dataset", "lr","embedding_dim", "dropout", "max_budget", "num_epochs", "strategy_ad","weight_tmp", "un_loss_type", "alpha", "beta","gamma", "nc-acc", "nc-idacc", "nc-f1-micro", "nc-f1-macro", "ad-auc", "ad-f1-macro", "ad-pre-macro", "ad-rec-macro", "A-num", "N-num", "nc_num"]
            csv_write.writerow(csv_head)

    with open(des_path, 'a+') as f:
        csv_write = csv.writer(f)
        data_row = [args.model, args.seed, args.dataset, args.lr, args.embedding_dim, args.dropout, args.max_budget, args.max_epoch, args.strategy_ad, args.weight_tmp, args.un_loss_type, args.alpha, args.beta,args.gamma, test_acc_nc, test_acc_nc_id, test_f1micro_nc, test_f1macro_nc, test_auc_ad, test_f1macro_ad, test_pre_ad, test_rec_ad, abnormal_num, normal_num, len(idx_train_nc)]
        csv_write.writerow(data_row)




if __name__ == '__main__':
    # Set argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Flickr')  # 'BlogCatalog'  'Flickr'  'ACM'  'cora'  'citeseer'  'pubmed'
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--seed', type=int, default=255)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--max_epoch', type=int, default=300)
    parser.add_argument('--max_budget', type=int, default=20)
    parser.add_argument('--iter_budget', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=.8, help='node classification loss weight')
    parser.add_argument('--beta', type=float, default=1., help='unsupervised loss weight')
    parser.add_argument('--gamma', type=float, default=.2, help='anomaly detection loss weight')
    parser.add_argument('--weight_tmp', type=float, default=.96, help='anomaly detection loss weight')
    parser.add_argument('--strategy_ad', type=str, default='k_medoids') # random uncertainty topk_anomaly
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--result_path', type=str, default='results/multitask')
    parser.add_argument('--un_loss_type', type=str, default='sup_sum') # sum div sup_sum sup_div
    parser.add_argument('--model', type=str, default='Model1') # Model0 Model1 Model2
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