import torch.nn as nn
import torch.nn.functional as F
from layer import GNN_Layer
from layer import GNN_Layer_Init
from layer import MLP
import torch 


class GNN_Bet(nn.Module):
    def __init__(self, ninput, nhid, dropout, mode="baseline", repeats=1, init_type="AW", leverage=False, lcc=False, normalize=False, num_layers=4):
        super(GNN_Bet, self).__init__()
        
        self.mode = mode
        self.repeats = repeats
        self.dropout = dropout
        self.num_layers = num_layers
        self.score_layer = MLP(nhid, self.dropout)

        self.gc1 = GNN_Layer_Init(ninput,nhid, init_type=init_type, leverage=leverage, lcc=lcc, normalize=normalize)

        if self.mode == "more_layers":
             # Use repeats to determine number of layers if provided (defaulting to 4 extra layers)
             num_layers = 4 if self.repeats <= 1 else self.repeats
             self.layers = nn.ModuleList([GNN_Layer(nhid,nhid) for _ in range(num_layers)])
        elif self.mode == "baseline":
             self.layers = nn.ModuleList([GNN_Layer(nhid,nhid) for _ in range(self.num_layers)])
        else:
             self.gc2 = GNN_Layer(nhid,nhid)
             self.gc3 = GNN_Layer(nhid,nhid)
             self.gc4 = GNN_Layer(nhid,nhid)
             self.gc5 = GNN_Layer(nhid,nhid)

    def forward(self,adj1,adj2):

        # Initial Layer
        x = F.normalize(F.relu(self.gc1(adj1)),p=2,dim=1)
        x2 = F.normalize(F.relu(self.gc1(adj2)),p=2,dim=1)

        #Score Calculations
        score1 = self.score_layer(x,self.dropout)
        score2 = self.score_layer(x2,self.dropout)

        if self.mode == "baseline":
            for i, layer in enumerate(self.layers):
                x = F.dropout(F.relu(layer(x, adj1)), self.dropout, training=self.training)
                x2 = F.dropout(F.relu(layer(x2, adj2)), self.dropout, training=self.training)
                
                # Normalize all except last
                if i < len(self.layers) - 1:
                    x = F.normalize(x, p=2, dim=1)
                    x2 = F.normalize(x2, p=2, dim=1)
                
                score1 = score1 + self.score_layer(x, self.dropout)
                score2 = score2 + self.score_layer(x2, self.dropout)

        elif self.mode == "repeats":
            layers = [self.gc2, self.gc3, self.gc4, self.gc5]
            for i, layer in enumerate(layers):
                for _ in range(self.repeats):
                    x = F.relu(layer(x, adj1))
                    x2 = F.relu(layer(x2, adj2))

                if i < len(layers) - 1:
                    x = F.normalize(x, p=2, dim=1)
                    x2 = F.normalize(x2, p=2, dim=1)
                
                score1 = score1 + self.score_layer(x, self.dropout)
                score2 = score2 + self.score_layer(x2, self.dropout)

        elif self.mode == "more_layers":
             for i, layer in enumerate(self.layers):
                x = F.relu(layer(x, adj1))
                x2 = F.relu(layer(x2, adj2))
                
                if i < len(self.layers) - 1:
                    x = F.normalize(x, p=2, dim=1)
                    x2 = F.normalize(x2, p=2, dim=1)
                
                score1 = score1 + self.score_layer(x, self.dropout)
                score2 = score2 + self.score_layer(x2, self.dropout)

        x = torch.mul(score1,score2)
        return x