import torch
#import torch_geometric as tg
import numpy as np
from numpy import math
#from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

    
# Convolution layer
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        torch.nn.init.normal_(tensor=self.weight, std=stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, inp, adj):
        support = torch.mm(inp, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

# Droput sampling layer
class BBGDC(nn.Module):
    def __init__(self, num_pars, alpha=0.8, kl_scale=1.0):
        super(BBGDC, self).__init__()
        self.num_pars = num_pars
        self.alpha = alpha
        self.kl_scale = kl_scale
        self.a_uc = nn.Parameter(torch.FloatTensor(self.num_pars))
        self.b_uc = nn.Parameter(torch.FloatTensor(self.num_pars))
        self.a_uc.data.uniform_(1.0, 1.5)
        self.b_uc.data.uniform_(0.49, 0.51)
    
    def get_params(self):
        a = F.softplus(self.a_uc.clamp(min=-10.))
        b = F.softplus(self.b_uc.clamp(min=-10., max=50.))
        return a, b
    
    def sample_pi(self):
        a, b = self.get_params()
        u = torch.rand(self.num_pars).clamp(1e-6, 1-1e-6)
        if torch.cuda.is_available():
            u = u.cuda()
        return (1 - u.pow_(1./b)).pow_(1./a)
    
    def get_weight(self, num_samps, training, samp_type='rel_ber'):
        temp = torch.Tensor([0.67])
        if torch.cuda.is_available():
            temp = temp.cuda()
        
        if training:
            pi = self.sample_pi()
            p_z = RelaxedBernoulli(temp, probs=pi)
            z = p_z.rsample(torch.Size([num_samps]))
        else:
            if samp_type=='rel_ber':
                pi = self.sample_pi()
                p_z = RelaxedBernoulli(temp, probs=pi)
                z = p_z.rsample(torch.Size([num_samps]))
            elif samp_type=='ber':
                pi = self.sample_pi()
                p_z = torch.distributions.Bernoulli(probs=pi)            
                z = p_z.sample(torch.Size([num_samps]))
        return z, pi
    
    def get_reg(self):
        a, b = self.get_params()
        kld = (1 - self.alpha/a)*(-0.577215664901532 - torch.digamma(b) - 1./b) + torch.log(a*b + 1e-10) - math.log(self.alpha) - (b-1)/b
        kld = (self.kl_scale) * kld.sum()
        return kld


# Our model
class BBGDCGCN_MI(nn.Module):
    def __init__(self, nfeat_list, dropout, nblock, nlay, num_edges,regular_term,pattern=None):
        super(BBGDCGCN_MI, self).__init__()
        
        assert len(nfeat_list)==nlay+1
        self.nlay = nlay
        self.regular_term=regular_term
        self.nblock = nblock
        self.norm_lays=[]
        self.num_edges = num_edges
        self.num_nodes = int(np.sqrt(num_edges))
        self.drpcon_list = []
        self.dropout = dropout
        gcs_list = []
        idx = 0
        for i in range(nlay):
            if i==0:
                self.drpcon_list.append(BBGDC(1))
                gcs_list.append([str(idx), GraphConvolution(nfeat_list[i], nfeat_list[i+1])])
                self.norm_lays.append(nn.BatchNorm1d(nfeat_list[i+1]).cuda())
                idx += 1
            else:
                self.drpcon_list.append(BBGDC(1))
                self.norm_lays.append(nn.BatchNorm1d(nfeat_list[i+1]).cuda())
                for j in range(self.nblock):
                    gcs_list.append([str(idx), GraphConvolution(int(nfeat_list[i]/self.nblock), nfeat_list[i+1])])
                    idx += 1
        
        self.drpcons = nn.ModuleList(self.drpcon_list)
        self.gcs = nn.ModuleDict(gcs_list)
        self.nfeat_list = nfeat_list
        self.partner=pattern
    
    def forward(self, x, labels, edge_index, obs_idx, warm_up, adj_normt, training=True
                , mul_type='norm_first', samp_type='rel_ber'):
        h_perv = x
        kld_loss = 0.0
        drop_rates = []
        edge_num=self.num_edges
        node_num=edge_index[0].shape[0]
        for i in range(self.nlay):
            mask_vec, drop_prob = self.drpcons[i].get_weight(self.nblock*edge_num, training, samp_type)
            mask_vec = torch.squeeze(mask_vec)
            drop_rates.append(drop_prob)
            if i==0:
                mask_mat = torch.reshape(mask_vec[:self.num_edges], (self.num_nodes, self.num_nodes)).cuda()
                adj_lay = torch.mul(mask_mat, adj_normt).cuda()
                x = F.relu(self.gcs[str(i)](x, adj_lay))
                x= self.norm_lays[i](x)
                x = F.dropout(x, self.dropout, training=training)
            
            else:
                feat_pblock = int(self.nfeat_list[i]/self.nblock)
                for j in range(self.nblock):
                    mask_mat = torch.reshape(mask_vec[j*self.num_edges:(j+1)*self.num_edges]
                                             , (self.num_nodes, self.num_nodes)).cuda()
                    adj_lay = torch.mul(mask_mat, adj_normt).cuda()
                    if i<(self.nlay-1):
                        if j==0:
                            x_out = self.gcs[str((i-1)*self.nblock+j+1)](x[:,j*feat_pblock:(j+1)*feat_pblock], adj_lay)
                        else:
                            x_out = x_out + self.gcs[str((i-1)*self.nblock+j+1)](x[:,j*feat_pblock:(j+1)*feat_pblock], adj_lay)
                    else:
                        if j==0:
                            out = self.gcs[str((i-1)*self.nblock+j+1)](x[:,j*feat_pblock:(j+1)*feat_pblock], adj_lay)
                        else:
                            out = out + self.gcs[str((i-1)*self.nblock+j+1)](x[:,j*feat_pblock:(j+1)*feat_pblock], adj_lay)
                
                if i<(self.nlay-1):
                    x = x_out
                    x = F.dropout(F.relu(x), self.dropout, training=training)
                    x=self.norm_lays[i](x)
            
            
            kld_loss += self.drpcons[i].get_reg()
            
        
        output = F.log_softmax(out,dim=1)
        MI=0.0
        for i in range(len(self.drpcons)-1):
            MI+=self.MI(self.drpcons[i],self.drpcons[i+1],50,50)/len(self.drpcons)
        nll_loss = self.loss(labels, output, obs_idx)
        re=self.regular_term
        tot_loss = nll_loss + warm_up * kld_loss+torch.exp(-torch.log(MI)/torch.log(torch.Tensor([0.5]).cuda()))
        drop_rates = torch.stack(drop_rates)
        return output, tot_loss, nll_loss, kld_loss, drop_rates,MI
    def MI(self,x,y,step_size,sample_size):  # x is the signle distribution and y is the join distribtuion y list joint distribution need to recalculate
        H_x=self.Hx(x,y,step_size,sample_size)[1]
        H_y=self.Hy(y.get_weight(sample_size,training=False)[0])
        return H_x+H_y-max(H_x,H_y)
    def Hy(self,y:list):
        return -torch.mean(torch.log2_(y))
    def sample_x_one_step(self,x:list):  # sample from  distribution one step
        return torch.mean(x)
    def Hx(self,x,y,step_size,sample_size):  # x and y is the distribution
        px=[]
        for i in range (sample_size):
            sam_x=x.get_weight(step_size,training=False)[0]
            px.append(self.sample_x_one_step(sam_x))
        px=torch.Tensor(px)
        hx=-torch.mean(torch.log2_(px))
        return px,hx 
    def Hxy(self,xy):  # sample from distribution list
        H_xy=-torch.mean(torch.log2_(xy))
        return H_xy
    def loss(self, labels, preds, obs_idx):
        loss=nn.NLLLoss()
        preds=preds[obs_idx]
        labels=labels[obs_idx]
        return loss(preds,labels)
