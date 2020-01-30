import torch.nn as nn
import torch
import torch.nn.functional as F
from mpnn.layers import MLP, FC, BatchNorm, CGConv
from torch_scatter import scatter_mean
from torch_geometric.nn import NNConv, global_mean_pool, global_sort_pool, GlobalAttention



class MPNN(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch,
                 num_edge_feats, num_edge_hid, num_hid, num_classes,
                 dropout, batch_norm = True):
        super(MPNN, self).__init__()
        nn1 = nn.Sequential(nn.Linear(num_edge_feats, num_edge_hid),
                            nn.ReLU(),
                            nn.Linear(num_edge_hid, in_ch * hid_ch))

        '''
        nn2 = nn.Sequential(nn.Linear(num_edge_feats, num_edge_hid),
                            nn.ReLU(),
                            nn.Linear(num_edge_hid, hid_ch * out_ch))
        '''

        self.nncov1 = NNConv(in_ch, hid_ch, nn1,  aggr='mean')
        self.nncov2 = NNConv(hid_ch, out_ch, FC(num_edge_feats, hid_ch * out_ch), aggr= 'mean')
        self.mlp = MLP(out_ch, num_hid, num_classes)
        self.dropout = dropout
        self.batch_norm = batch_norm


        self.bn1 = BatchNorm(in_channels = hid_ch)
        self.bn2 = BatchNorm(in_channels = out_ch)


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.nncov1(x, edge_index, edge_attr) # (num_node, hid_ch)
        if self.batch_norm:
            x = self.bn1(x)
        x = F.dropout(x, self.dropout, training= self.training)


        x = self.nncov2(x, edge_index, edge_attr) # (num_node, out_ch)
        if self.batch_norm:
            x = self.bn2(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = scatter_mean(x, data.batch, dim=0)    # (batch_size, out_ch)
        #x = global_mean_pool(x, data.batch)

        x = self.mlp(x)      # (batch_size, num_classes)
        return F.log_softmax(x, dim = 1)



class CGCNN(nn.Module):
    def __init__(self, in_ch,
                 num_edge_feats, num_hid, num_classes,
                 dropout, batch_norm = True):
        super(CGCNN, self).__init__()
        self.cgcov1 = CGConv(channels= in_ch, dim = num_edge_feats, aggr = 'mean')
        self.cgcov2 = CGConv(channels= in_ch, dim = num_edge_feats, aggr = 'mean')
        self.mlp = MLP(in_ch, num_hid, num_classes)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.cgcov1(x, edge_index, edge_attr)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.cgcov2(x, edge_index, edge_attr)
        x = F.dropout(x, self.dropout, training=self.training)

        x = scatter_mean(x, data.batch, dim=0)
        x = self.mlp(x)  # (batch_size, num_classes)
        return F.log_softmax(x, dim=1)



class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(61*61*64, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        #x = F.dropout(x, p=0.5, training=self.training)
        #print(x.shape)

        x = x.view(-1,61*61*64 )
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

