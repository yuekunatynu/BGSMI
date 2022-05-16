from BCGNNMI import BBGDCGCN_MI
import torch_geometric as tg
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from scipy import stats
import itertools
import time
import random

# Constructing Laplacian matrix
def todense(edge_index,node_num):
    adj=np.zeros([node_num,node_num],dtype=np.float32)
    for x,y in zip(edge_index[0],edge_index[1]):
        adj[x][y]=1
        adj[y][x]=1
    adj=torch.FloatTensor(adj)

    def normalize_torch(mx):
        rowsum = torch.sum(mx, 1)
        r_inv = torch.pow(rowsum, -1)
        r_inv[torch.isinf(r_inv)] = 0.
        r_inv[torch.isnan(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = torch.matmul(r_mat_inv, mx)
        return mx
    adj_normt = normalize_torch(adj + torch.eye(adj.shape[0]))
    return adj_normt

# Calculating accuracy 
def accuracy_mrun_np(outputs, labels_np, inds):
    preds_np = stats.mode(np.argmax(outputs, axis=2), axis=0)[0].reshape(-1).astype(np.int32)
    correct = np.equal(labels_np[inds], preds_np[inds]).astype(np.float32)
    correct = correct.sum()
    return correct / len(inds)

"""
load_by_geodata(): Data loading based on 'torch_geometric.data'
name: the name of dataset
root: the root of dataset
"""
def load_by_geodata(name,root):
    if name=='cora':
        datasets=tg.datasets.Planetoid(root=root,name=name.capitalize())
    elif name=='citeseer':
        datasets=tg.datasets.Planetoid(root=root,name=name.capitalize())

    datasets=datasets.data
    features, labels=datasets.x,datasets.y  
    try:
        train_mask,val_mask,test_mask=datasets.train_mask,datasets.val_mask,datasets.test_mask
        idx_all=torch.tensor(list(range(features.shape[0])))
        idx_train,idx_val,idx_test=idx_all[train_mask],idx_all[val_mask],idx_all[test_mask]
    except:
        idx_all=list(range(features.shape[0]))
        idx_train,idx_test=train_test_split(idx_all)
        idx_train,idx_val=train_test_split(idx_train)
        idx_train=torch.LongTensor(idx_train)
        idx_val=torch.LongTensor(idx_val)
        idx_test=torch.LongTensor(idx_test)
    return datasets.edge_index, features, labels, idx_train, idx_val, idx_test

"""
class experiment: Experimental implementation class
dataset: The name of dataset
lay: The number of layers
config: The setting of convolution kernel size
tern: Hyperparameter of drop_out
block: Hyperparameter of BGNN
"""
class experiment:
    def __init__(self,dataset='cora',lay=4,config=1,term=0.3,block=5,nways=5,kshots=1):
        seed = 5
        self.collect={}
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Load data
        edge_index,features,labels, idx_train, idx_val, idx_test = load_by_geodata(name=dataset,root='./GDC-master/truedataset')
        
        # Hyperparameters
        nfeat = features.shape[1]
        nclass = labels.max().item() + 1
        adj_nor=todense(edge_index, features.shape[0])
        if lay==4:
            if config==1:
                nfeat_list=[nfeat, 256, 128, 64,nclass]
            elif config==2:
                nfeat_list=[nfeat,128,64,32,nclass]
            elif config==3:
                nfeat_list=[nfeat,128,256,128,nclass]
            elif config==4:
                nfeat_list=[nfeat,256,512,128,nclass]
            elif config==5:
                nfeat_list=[nfeat,64,128,128,nclass]
            elif config==6:
                nfeat_list=[nfeat,64,32,nclass,nclass]
            elif config==7:
                nfeat_list=[nfeat,256,256,256,nclass]
            elif config==9:
                nfeat_list = [nfeat, 128, 128, 128,nclass] #[nfeat, 256, 128, 64,nclass]
        elif lay==2:
            nfeat_list = [nfeat, 256,nclass]
        elif lay==6:
            nfeat_list= [nfeat, 256, 256, 128,64,64,nclass]
        nlay = lay
        nblock = block
        num_edges = int(adj_nor.shape[0] ** 2)
        dropout = 0.5
        lr = 0.005
        mul_type='norm_first'

        # Defining model
        model = BBGDCGCN_MI(nfeat_list=nfeat_list
                        , dropout=dropout
                        , nblock=nblock
                        , nlay=nlay
                        , num_edges=num_edges
                        , regular_term=term
                        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        print("Model Summary:")
        print(model)
        print('----------------------')
        if torch.cuda.is_available():
            model.cuda()
        # N-way K-shot
        if dataset=='cora':
            class_label=[0,1,2,3,4,5,6]
            combination=list(itertools.combinations(class_label,nways))
        elif dataset=='citeseer':
            class_label=[0,1,2,3,4,5]
            combination=list(itertools.combinations(class_label,nways))
        for i in range(len(combination)):
            train_label = list(combination[i])
            test_label = [n for n in class_label if n not in train_label]

        # Training
        nepochs = 20
        num_run = 20
        epo_collectors=[]
        loss_collectors=[]
        accuracy_collectors=[]
        execution_collectors=[]

        if torch.cuda.is_available():
            edge_index=edge_index.cuda()
            features=features.cuda()
            labels=labels.cuda()
            idx_train=idx_train.cuda()
            idx_val=idx_val.cuda()
            idx_test=idx_test.cuda()
            adj_nor=adj_nor.cuda()
        labels_local = labels.clone().detach()
        select_class = random.sample(set(labels.cpu().tolist()), nways)
        class_idx=[[] for _ in range(nways)]
        cur_time=time.time()

        for i in range(features.shape[0]):
            for j in range(nways):
                if labels_local[i]==select_class[j]:
                    class_idx[j].append(i)
                    labels_local[i]=j
        for i in range(nepochs):
            class_train=[]
            class_all=[]
            for j in range(nways):
                class_train+=random.sample(class_idx[j],kshots)
                class_all+=class_idx[j]
            class_test=[n1 for n1 in class_all if n1 not in class_train]
            optimizer.zero_grad()
            wup = np.min([1.0, (i+1)/20])
                
            output, tot_loss, nll_loss, kld_loss, drop_rates,MI = model(x=features
                                                                    , labels=labels
                                                                    , edge_index=edge_index
                                                                    , obs_idx=idx_train
                                                                    , warm_up=wup
                                                                    , adj_normt=adj_nor
                                                                    , training=True
                                                                    , mul_type=mul_type)

            main_loss = tot_loss+MI
            main_loss.backward()
            optimizer.step()
            outs = [None]*num_run
            
            # Testing
            for j in range(num_run):
                outstmp, _, _, _,_,_ = model(x=features
                                        , labels=labels
                                        , edge_index=edge_index
                                        , obs_idx=idx_train
                                        , warm_up=wup
                                        , adj_normt=adj_nor
                                        , training=False
                                        , samp_type='rel_ber'
                                        , mul_type=mul_type)

                outs[j] = outstmp.cpu().data.numpy()
            outs = np.stack(outs)
            labels_np=labels.cpu().numpy()
            idx_val_np=idx_val.cpu().numpy()
            idx_test_np=idx_test.cpu().numpy()
            acc_val_tr = accuracy_mrun_np(outs, labels_np, idx_val_np)
            acc_test_tr = accuracy_mrun_np(outs, labels_np, idx_test_np)
            epo_collectors.append(i)
            loss_collectors.append(main_loss.item())
            execution_collectors.append(time.time()-cur_time)
            accuracy_collectors.append(acc_test_tr)
            
            # Recording
            print('Epoch: {:04d}'.format(i+1)
                , 'nll: {:.4f}'.format(nll_loss.item())
                , 'kld: {:.4f}'.format(kld_loss.item())
                ,'MI: {:.4f}'.format(MI.item())
                , 'acc_val: {:.4f}'.format(acc_val_tr)
                ,'time: {:.4f}'.format(time.time()-cur_time)
                , 'acc_test: {:.4f}'.format(acc_test_tr))
            print('----------------------')
            self.collect['x']=epo_collectors
            self.collect['acc']=accuracy_collectors
            self.collect['execution']=execution_collectors
            self.collect['loss']=loss_collectors
    def get_collect(self):
        return self.collect

if __name__ == "__main__":    

    result = experiment(dataset='cora',lay=4,block=2,config=9).get_collect()
    result_excel = pd.DataFrame()
    result_excel["x"] = result.get("x")
    result_excel["acc"] = result.get("acc")
    result_excel["execution"] = result.get("execution")
    result_excel["loss"] = result.get("loss")

    # Saving training result
    file_path=pd.ExcelWriter('result.xlsx')
    result_excel.fillna(' ', inplace=True)
    result_excel.to_excel(file_path, encoding='utf-8', index=False)
    file_path.save()
    
