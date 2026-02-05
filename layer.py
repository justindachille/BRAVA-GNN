import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch_scatter import scatter_max
import sys

class GNN_Layer(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GNN_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GNN_Layer_Init(Module):
    def __init__(self, in_features, out_features, bias=True, init_type="AW"):
        super(GNN_Layer_Init, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.init_type = init_type

        self.input_dim = 1
        
        if self.init_type in ["degree_mix_sum", "degree_mix_max"]: 
            self.input_dim = 3
        elif self.init_type == "degree_mix_sum_2":
            self.input_dim = 2
        elif self.init_type == "degree_mix_max_sum": 
            self.input_dim = 6
        
        if self.init_type.startswith("degree_mix_mass_") or self.init_type.startswith("degree_mix_independent_"):
            try:
                self.input_dim = int(self.init_type.split("_")[-1])
            except ValueError:
                self.input_dim = 1

        if self.init_type.startswith("degree"):
            self.weight = Parameter(torch.FloatTensor(self.input_dim, out_features))
        else:
            self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj):
        if self.init_type.startswith("degree"):
            
            if self.init_type in ["degree_mix_max", "degree_mix_max_sum"]:
                adj = adj.coalesce()

            ones = torch.ones((adj.shape[0], 1), device=adj.device)
            d1 = torch.spmm(adj, ones)
            x = d1
            
            if self.init_type == "degree0":
                x = ones
            elif self.init_type == "degree1" or self.init_type == "degree_pure":
                x = d1
            elif self.init_type == "degree2" or self.init_type == "degree":
                d2 = torch.spmm(adj, d1)
                x = d2
            elif self.init_type == "degree3":
                d2 = torch.spmm(adj, d1)
                d3 = torch.spmm(adj, d2)
                x = d3
            elif self.init_type == "degree_mix_sum":
                d2 = torch.spmm(adj, d1)
                d3 = torch.spmm(adj, d2)
                x = torch.cat([d1, d2, d3], dim=1)
            elif self.init_type == "degree_mix_sum_2":
                d2 = torch.spmm(adj, d1)
                x = torch.cat([d1, d2], dim=1)
            elif self.init_type == "degree_mix_max":
                row, col = adj.indices()
                val = adj.values()

                d2 = F.relu(scatter_max(d1[col] * val.unsqueeze(1), row, dim=0, dim_size=adj.size(0))[0])
                d3 = F.relu(scatter_max(d2[col] * val.unsqueeze(1), row, dim=0, dim_size=adj.size(0))[0])
                x = torch.cat([d1, d2, d3], dim=1)
            elif self.init_type == "degree_mix_max_sum":
                d2_sum = torch.spmm(adj, d1)
                d3_sum = torch.spmm(adj, d2_sum)

                row, col = adj.indices()
                val = adj.values()
                d2_max = F.relu(scatter_max(d1[col] * val.unsqueeze(1), row, dim=0, dim_size=adj.size(0))[0])
                d3_max = F.relu(scatter_max(d2_max[col] * val.unsqueeze(1), row, dim=0, dim_size=adj.size(0))[0])

                x = torch.cat([d1, d2_sum, d3_sum, d1, d2_max, d3_max], dim=1)
            
            elif self.init_type.startswith("degree_mix_mass_") or self.init_type.startswith("degree_mix_independent_"):
                k = int(self.init_type.split("_")[-1])
                
                current_d = d1
                deg_list = [d1]
                
                for _ in range(k - 1):
                    current_d = torch.spmm(adj, current_d)
                    deg_list.append(current_d)
                
                if self.init_type.startswith("degree_mix_mass_"):
                    mass_list = []
                    current_mass = torch.zeros_like(d1)
                    for d in deg_list:
                        current_mass = current_mass + d
                        mass_list.append(current_mass)
                    x = torch.cat(mass_list, dim=1)
                else:
                    x = torch.cat(deg_list, dim=1)

            if self.init_type != "degree0":
                x = torch.log(x + 1)
            
            x_in = x
            output = torch.mm(x_in, self.weight)
        else:
            support = self.weight
            
            if adj.shape[0] < support.shape[0]:
                support = support[:adj.shape[0], :]

            output = torch.spmm(adj, support)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class MLP(Module):
    def __init__(self, nhid,dropout):
        super(MLP,self).__init__()
        self.dropout = dropout
        self.linear1 = torch.nn.Linear(nhid,2*nhid)
        self.linear2 = torch.nn.Linear(2*nhid,2*nhid)
        self.linear3 = torch.nn.Linear(2*nhid,1)


    def forward(self,input_vec,dropout):

        score_temp = F.relu(self.linear1(input_vec))
        score_temp = F.dropout(score_temp,self.dropout,self.training)
        score_temp = F.relu(self.linear2(score_temp))
        score_temp = F.dropout(score_temp,self.dropout,self.training)
        score_temp = self.linear3(score_temp)

        return score_temp