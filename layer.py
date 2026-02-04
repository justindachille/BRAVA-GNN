import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch_sparse import spspmm
from torch_scatter import scatter_max
import sys

from positional_encodings.torch_encodings import PositionalEncoding1D

class GNN_Layer(Module):
    """
    Layer defined for GNN-Bet
    """

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


class MasterUpdate(Module):
    def __init__(self, nhid):
        super(MasterUpdate, self).__init__()
        self.linear_update = torch.nn.Linear(2*nhid, nhid)
        self.linear_broadcast = torch.nn.Linear(nhid, nhid)
    
    def forward(self, master_state, node_embeddings):
        node_avg = torch.mean(node_embeddings, dim=0, keepdim=True)
        inp = torch.cat([master_state, node_avg], dim=1)
        new_master = torch.tanh(self.linear_update(inp))

        broadcast = self.linear_broadcast(new_master)

        return new_master, broadcast



class GNN_Layer_Init(Module):
    """
    First layer of GNN_Init, for embedding lookup
    """

    def __init__(self, in_features, out_features, bias=True, init_type="AW", leverage=False, lcc=False, normalize=False):
        super(GNN_Layer_Init, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.init_type = init_type
        self.leverage = leverage
        self.lcc = lcc
        self.normalize = normalize

        self.input_dim = 1
        
        # --- DIMENSION LOGIC ---
        if self.init_type == "degree_embedding": 
            self.input_dim = out_features
        elif self.init_type == "positional_encoding": 
            # 1D PE (uses degree1)
            self.input_dim = out_features
        elif self.init_type == "positional_encoding_3d":
            # 3D PE (uses d1, d2, d3 concatenated)
            # The result is 3 * out_features wide, so we project it back to out_features later
            self.input_dim = out_features * 3
        elif self.init_type in ["degree_mix_sum", "degree_mix_max"]: 
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

        if self.leverage: self.input_dim += 1
        if self.lcc: self.input_dim += 1

        if self.init_type == "degree_embedding":
            self.emb = torch.nn.Embedding(in_features, out_features)
            self.weight = Parameter(torch.FloatTensor(self.input_dim, out_features))
        
        elif self.init_type == "positional_encoding":
             self.weight = Parameter(torch.FloatTensor(self.input_dim, out_features))
             # Standard 1D PE
             self.p_enc = PositionalEncoding1D(out_features)
        
        elif self.init_type == "positional_encoding_3d":
             self.weight = Parameter(torch.FloatTensor(self.input_dim, out_features))
             # We use manual calculation now to avoid OOM
             
        elif self.init_type.startswith("degree"):
            # [H=DEGREE]
            self.weight = Parameter(torch.FloatTensor(self.input_dim, out_features))
        else:
            # [H=A*W]
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

    def get_pe_for_mass(self, mass_vector):
        """
        Analytically compute Sinusoidal Encoding for mass values.
        Using the package directly creates a lookup table of size [MaxMass, Dim].
        Since d3 masses can reach >1e8, this causes OOM.
        This function implements the exact same formula efficiently.
        
        Formula:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        """
        # Ensure input is (N, 1)
        if mass_vector.dim() == 1:
            mass_vector = mass_vector.unsqueeze(1)
            
        positions = mass_vector.float()
        D = self.out_features
        device = positions.device
        
        # div_term = 1 / 10000^(2i/D)
        # 2i/D ranges from 0 to (D-2)/D
        # We calculate half the dimensions (for sin/cos pairs)
        i = torch.arange(0, D, 2, device=device).float()
        div_term = torch.exp(i * (-math.log(10000.0) / D))
        
        # phase = pos * div_term
        # Shape: (N, 1) * (1, D/2) = (N, D/2)
        phase = positions * div_term.unsqueeze(0)
        
        pe = torch.zeros(positions.shape[0], D, device=device)
        pe[:, 0::2] = torch.sin(phase)
        pe[:, 1::2] = torch.cos(phase)
        
        return pe

    def forward(self, adj):
        if self.init_type.startswith("degree") or "positional_encoding" in self.init_type:
            
            if self.leverage or self.lcc or self.init_type in ["degree_mix_max", "degree_mix_max_sum"]:
                adj = adj.coalesce()

            # Calculate d1 (A*1)
            ones = torch.ones((adj.shape[0], 1), device=adj.device)
            d1 = torch.spmm(adj, ones) # Degree (A*1)
            x = d1
            
            if self.init_type == "degree0":
                x = ones
            elif self.init_type == "degree_embedding":
                x = self.emb(d1.long().view(-1).clamp(0, self.in_features - 1))
            
            # --- 1D PE ---
            elif self.init_type == "positional_encoding":
                x = self.get_pe_for_mass(d1)

            # --- 3D PE (Using d1, d2, d3) ---
            elif self.init_type == "positional_encoding_3d":
                # Compute d2, d3
                d2 = torch.spmm(adj, d1)
                d3 = torch.spmm(adj, d2)
                
                # Get PEs for each "dimension"
                pe_d1 = self.get_pe_for_mass(d1)
                pe_d2 = self.get_pe_for_mass(d2)
                pe_d3 = self.get_pe_for_mass(d3)
                
                # Concatenate them: Size becomes (N, 3*out_features)
                # The self.weight matrix will project this back down to out_features
                x = torch.cat([pe_d1, pe_d2, pe_d3], dim=1)

            elif self.init_type == "degree1" or self.init_type == "degree_pure":
                x = d1
            elif self.init_type == "degree2" or self.init_type == "degree":
                d2 = torch.spmm(adj, d1) # Neighbors sum (A^2*1)
                x = d2
            elif self.init_type == "degree3":
                d2 = torch.spmm(adj, d1)
                d3 = torch.spmm(adj, d2) # Neighbors' neighbor sum (A^3*1)
                x = d3
            elif self.init_type == "degree_mix_sum":
                d2 = torch.spmm(adj, d1)
                d3 = torch.spmm(adj, d2)
                x = torch.cat([d1, d2, d3], dim=1)
            elif self.init_type == "degree_mix_sum_2":
                d2 = torch.spmm(adj, d1)
                x = torch.cat([d1, d2], dim=1)
            elif self.init_type == "degree_mix_mass_3":
                d2 = torch.spmm(adj, d1)
                d3 = torch.spmm(adj, d2)
                x = torch.cat([d1, d1+d2, d1+d2+d3], dim=1)
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

            # Do NOT Log/Norm PE features, they are already bounded [-1, 1] by definition
            if "positional_encoding" not in self.init_type:
                if self.init_type != "degree0" and self.init_type != "degree_embedding":
                    x = torch.log(x + 1)
                    if self.normalize:
                        x = x / (x.max(dim=0)[0] + 1e-6)

            features = [x]
            if self.leverage:
                deg = d1.squeeze()
                row, col = adj.indices()
                vals = adj.values()

                deg_sum = deg[row] + deg[col]
                deg_sum = torch.where(deg_sum > 0, deg_sum, torch.ones_like(deg_sum))

                lev_vals = vals * (1.0 / deg_sum)
                lev_adj = torch.sparse_coo_tensor(adj.indices(), lev_vals, adj.size())
                sum_lev = torch.spmm(lev_adj, ones)

                lev_feat = 2 * sum_lev - 1
                features.append(lev_feat)
            
            if self.lcc:
                N = adj.shape[0]
                dev = adj.device
                idx = adj.indices().cpu()
                val = adj.values().cpu()

                idxA2, valA2 = spspmm(idx, val, idx, val, N, N, N)
                idxA3, valA3 = spspmm(idxA2, valA2, idx, val, N, N, N)

                diag_mask = (idxA3[0] == idxA3[1])
                
                triangles = torch.zeros(N, 1, device=dev)
                
                # Move result diagonal back to device
                tri_idx_on_dev = idxA3[0][diag_mask].to(dev)
                tri_val_on_dev = valA3[diag_mask].to(dev)
                
                triangles[tri_idx_on_dev] = tri_val_on_dev.view(-1, 1)
                
                # Calculate LCC denominator d*(d-1)
                denom = d1 * (d1 - 1)
                denom = torch.where(denom > 0, denom, torch.ones_like(denom))

                lcc_feat = triangles / denom
                lcc_feat = lcc_feat / (lcc_feat.max() + 1e-6)

                features.append(lcc_feat)
            
            x_in = torch.cat(features, dim=1)
            output = torch.mm(x_in, self.weight)
        else:
            # [H=A*W]
            support = self.weight
            
            # SLICING: If the current graph (adj) is smaller than the master weight matrix,
            # we slice the weight matrix to match the graph size.
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