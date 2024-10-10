import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv, global_mean_pool, global_add_pool, GINEConv

# set post layers to be 2



class eGIN(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden_dim, hidden_dim_post, edge_dim, dropout = 0.5):
        super(eGIN, self).__init__()
        """ GCNConv layers """

        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout
        for i in range(num_layers):
            if i==0:
                self.convs.append(GINEConv(
            Sequential(
                Linear(dataset.num_features, hidden_dim),
                BatchNorm1d(hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                BatchNorm1d(hidden_dim),
                ReLU(),
            ),
            train_eps=True,
            edge_dim=edge_dim
            ))
            else:
                self.convs.append(GINConv(
                    Sequential(
                        Linear(hidden_dim, hidden_dim),
                        BatchNorm1d(hidden_dim), 
                        ReLU(),
                        Linear(hidden_dim, hidden_dim),
                        BatchNorm1d(hidden_dim), 
                        ReLU(),
                    ),
                    train_eps=True,
            edge_dim=edge_dim
            ))

        self.lin1 = torch.nn.Linear(hidden_dim, hidden_dim_post)
        self.lin2 = Linear(hidden_dim_post, dataset.num_classes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, training=self.training)

        for i in range(self.num_layers):

           x = F.relu(self.convs[i](x, edge_index))
                      
           # Dropout
           x = F.dropout(x, p=self.dropout, training=self.training)


        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)





class eGAT(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden_dim, hidden_dim_post, edge_dim, dropout = 0.5):
        super(eGAT, self).__init__()
        """ GCNConv layers """

        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout
        for i in range(num_layers):
            if i==0:
                self.convs.append(GATConv(dataset.num_features, hidden_dim,edge_dim = edge_dim))
            else:
                self.convs.append(GATConv(hidden_dim, hidden_dim,edge_dim = edge_dim))

        self.lin1 = torch.nn.Linear(hidden_dim, hidden_dim_post)
        self.lin2 = Linear(hidden_dim_post, dataset.num_classes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, training=self.training)

        for i in range(self.num_layers):

           x = F.relu(self.convs[i](x, edge_index,edge_attr))
                      
           # Dropout
           x = F.dropout(x, p=self.dropout, training=self.training)


        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)
