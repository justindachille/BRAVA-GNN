import torch.nn as nn
import torch.nn.functional as F
from layer import GNN_Layer
from layer import GNN_Layer_Init
from layer import MLP
import torch

class GNN_Bet(nn.Module):
    def __init__(self, ninput, nhid, dropout, init_type="AW", num_layers=4):
        super(GNN_Bet, self).__init__()
        
        self.dropout = dropout
        self.num_layers = num_layers
        self.score_layer = MLP(nhid, self.dropout)

        self.gc1 = GNN_Layer_Init(ninput,nhid, init_type=init_type)

        self.layers = nn.ModuleList([GNN_Layer(nhid,nhid) for _ in range(self.num_layers)])

    def forward(self,adj1,adj2):

        x = F.normalize(F.relu(self.gc1(adj1)),p=2,dim=1)
        x2 = F.normalize(F.relu(self.gc1(adj2)),p=2,dim=1)

        score1 = self.score_layer(x,self.dropout)
        score2 = self.score_layer(x2,self.dropout)

        for i, layer in enumerate(self.layers):
            x = F.dropout(F.relu(layer(x, adj1)), self.dropout, training=self.training)
            x2 = F.dropout(F.relu(layer(x2, adj2)), self.dropout, training=self.training)
            
            if i < len(self.layers) - 1:
                x = F.normalize(x, p=2, dim=1)
                x2 = F.normalize(x2, p=2, dim=1)
            
            score1 = score1 + self.score_layer(x, self.dropout)
            score2 = score2 + self.score_layer(x2, self.dropout)


        x = torch.mul(score1,score2)
        return x