import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv, global_mean_pool, global_add_pool


# set post layers to be 2

class GIN(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden_dim, hidden_dim_post, dropout = 0.5):
        super(GIN, self).__init__()
        """ GCNConv layers """

        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout
        for i in range(num_layers):
            if i==0:
                self.convs.append(GINConv(
            Sequential(
                Linear(dataset.num_features, hidden_dim),
                BatchNorm1d(hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                BatchNorm1d(hidden_dim),
                ReLU(),
            ),
            train_eps=True))
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
                    train_eps=True))

        self.lin1 = torch.nn.Linear(hidden_dim, hidden_dim_post)
        self.lin2 = Linear(hidden_dim_post, dataset.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(x, training=self.training)

        for i in range(self.num_layers):

           x = F.relu(self.convs[i](x, edge_index))
                      
           # Dropout
           x = F.dropout(x, p=self.dropout, training=self.training)


        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)


class GCN(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden_dim, hidden_dim_post, dropout = 0.5):
        super(GCN, self).__init__()
        """ GCNConv layers """

        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout
        for i in range(num_layers):
            if i==0:
                self.convs.append(GCNConv(dataset.num_features, hidden_dim))
            else:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.lin1 = torch.nn.Linear(hidden_dim, hidden_dim_post)
        self.lin2 = Linear(hidden_dim_post, dataset.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(x, training=self.training)

        for i in range(self.num_layers):

           x = F.relu(self.convs[i](x, edge_index))
                      
           # Dropout
           x = F.dropout(x, p=self.dropout, training=self.training)


        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)


class GAT(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden_dim, hidden_dim_post, dropout = 0.5):
        super(GAT, self).__init__()
        """ GCNConv layers """

        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout
        for i in range(num_layers):
            if i==0:
                self.convs.append(GATConv(dataset.num_features, hidden_dim))
            else:
                self.convs.append(GATConv(hidden_dim, hidden_dim))

        self.lin1 = torch.nn.Linear(hidden_dim, hidden_dim_post)
        self.lin2 = Linear(hidden_dim_post, dataset.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(x, training=self.training)

        for i in range(self.num_layers):

           x = F.relu(self.convs[i](x, edge_index))
                      
           # Dropout
           x = F.dropout(x, p=self.dropout, training=self.training)


        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)

class GraphSAGE(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden_dim, hidden_dim_post, dropout = 0.5):
        super(GraphSAGE, self).__init__()
        """ GCNConv layers """

        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout
        for i in range(num_layers):
            if i==0:
                self.convs.append(SAGEConv(dataset.num_features, hidden_dim))
            else:
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.lin1 = torch.nn.Linear(hidden_dim, hidden_dim_post)
        self.lin2 = Linear(hidden_dim_post, dataset.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(x, training=self.training)

        for i in range(self.num_layers):

           x = F.relu(self.convs[i](x, edge_index))
                      
           # Dropout
           x = F.dropout(x, p=self.dropout, training=self.training)


        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)