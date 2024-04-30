# /*
#  * @Author: wyuhui 
#  * @Date: 2024-03-25 16:13:32 
#  * @Last Modified by:   wyuhui 
#  * @Last Modified time: 2024-03-25 16:13:32 
#  */


import os
import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn   
import torch.nn.functional as F
import networkx as nx
import numpy as np  
import pandas as pd 
from tqdm import tqdm
import random

class Params():
    def __init__(self, point_num, device, project_dir, matrix_dir) -> None:
        self.N = point_num # 仿真采样点数
        self.device = device
        self.project_dir = project_dir
        self.matrix_dir = matrix_dir
        pass

    def set_data_params(self, batch_size, train_val_test_ratio):
        self.batch_size = batch_size
        self.train_val_test_ratio = train_val_test_ratio


    def set_model_params(self, loss_fcn, optimizer, epoches):
        self.loss_fcn = loss_fcn
        self.optimizer = optimizer
        self.epoches = epoches
        return

class EuclideanDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.pairwise_distance = nn.PairwiseDistance()

    def forward(self, x1, x2):
        return torch.mean(self.pairwise_distance(x1, x2)) # 平均距离


class GNN(nn.Module):
    def __init__(self,params: Params, in_size, out_size) -> None:
        super().__init__() 
        # self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        N = params.N # 仿真采样点数,也是地图矩阵的维度
        self.conv1 = dglnn.GraphConv(in_size, 5, weight=True, bias=True) # 升维到N
        # self.conv1 = dglnn.SAGEConv(in_size, N*2, aggregator_type='gcn')
        self.conv2 = dglnn.GraphConv(5, N*N, norm='both', weight=False, bias=False) # 地图矩阵权重层
        self.conv3 = dglnn.GraphConv(N*N, N, norm='both') # 地图矩阵权重层
        # self.conv4 = dglnn.GraphConv(N, N, norm='both', weight=False, bias=False) # 地图矩阵权重层

        # self.conv2 = dglnn.GraphConv(N*2, N, norm='both', weight=True, bias=True)
        self.linear1 = nn.Linear(N, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.linear3 = nn.Linear(1024, 512)
        self.linear4 = nn.Linear(512, 64)
        self.linear5 = nn.Linear(64, out_size)


    def forward(self, g, features, weight=None, edge_weight=None):
        # x = self.conv1(g, features, edge_weight=edge_weight)
        x = self.conv1(g, features)
        x = self.relu(x)
        x = self.conv2(g, x, weight=weight, edge_weight=edge_weight)
        x = self.relu(x)
        x = self.conv3(g, x, edge_weight=edge_weight)
        x = self.relu(x)
        # x = self.conv4(g, x, weight=weight, edge_weight=edge_weight)
        # x = self.relu(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.relu(x)
        x = self.linear5(x)
        return x

def load_graphs(params: Params):
    positions = pd.read_csv(os.path.join(params.project_dir,'positions.csv'), index_col=0).to_numpy()
    mean_x = sum(positions.T[0]) / positions.shape[0]
    mean_y = sum(positions.T[1]) / positions.shape[0]

    connection_matrix = pd.read_csv(os.path.join(params.project_dir, 'connection_matrix.csv'), index_col=0).to_numpy()

    
    all_graphs = []
    for csv_name in tqdm(os.listdir(params.matrix_dir), desc='loading graphs...'):
        matrix = pd.read_csv(os.path.join(params.matrix_dir, csv_name), index_col=0)
        g_nx = nx.from_numpy_array(matrix.to_numpy())
        nodes_id = matrix.columns.astype(int)
        g = dgl.from_networkx(g_nx)
        g.ndata['node_id'] = torch.tensor(nodes_id)
        g.ndata['position'] = torch.tensor(np.array([positions[i] for i in nodes_id]), dtype=torch.float32) # 位置标签
        g.ndata['feat'] = torch.tensor([[mean_x, mean_y] for _ in range(g.number_of_nodes())], dtype=torch.float32) # 初始化节点特征
        # g.ndata['feat'] = torch.tensor(np.array([positions[i] for i in nodes_id]), dtype=torch.float32) # 位置标签
        
        power_features = []
        # 给边特征赋值 power
        for i in range(g.num_edges()):
            src_node_id = nodes_id[int(g.edges()[0][i])]
            dst_node_id = nodes_id[int(g.edges()[1][i])]
            power_features.append(connection_matrix[src_node_id][dst_node_id])
        g.edata['power'] = torch.tensor(power_features, dtype=torch.float32)
        g = g.int().to(params.device)
        all_graphs.append(g)
    return all_graphs


def batch_graphs(params: Params, graphs):
    print('batch dataset...')
    graph_batchs = []
    while len(graphs) > params.batch_size:
        graph_batchs.append(dgl.batch(graphs[:params.batch_size]))
        graphs = graphs[params.batch_size:]

    if len(graphs):
        graph_batchs.append(dgl.batch(graphs))
    print('dataset batched')
    return graph_batchs

def split_dataset(params: Params, dataset):
    [train_ratio, val_ratio, test_ratio] = params.train_val_test_ratio
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)
    test_size = dataset_size - train_size - val_size

    random.shuffle(dataset)
    return {
        'train_data': batch_graphs(params, dataset[:train_size]),
        'val_data': batch_graphs(params, dataset[train_size: train_size + val_size]),
        'test_data': batch_graphs(params, dataset[-test_size: ])
    }

def evaluate(params: Params, dataset, model, map_weight):
    loss_fcn = params.loss_fcn
    
    model.eval()
    with torch.no_grad():
        total_err = 0
        for g in dataset:
            # logits = model(g, g.ndata['feat'], map_weight, g.edata['power'])
            logits = model(g, g.ndata['feat'], map_weight) #!
            err_per_graph = loss_fcn(logits, g.ndata['position'])
            total_err += err_per_graph
        return total_err / len(dataset)


def train(params: Params, dataset, model: GNN, map_weight):
    train_dataset = dataset['train_data']
    val_dataset = dataset['val_data']
    [loss_fcn, optimizer] = [params.loss_fcn, params.optimizer]

    for epoch in range(params.epoches):
        model.train()
        for g in tqdm(train_dataset, 'training...'):
            # logits = model(g, g.ndata['feat'], map_weight, g.edata['power'])
            logits = model(g, g.ndata['feat'], map_weight) #!
            loss = loss_fcn(logits, g.ndata['position'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_loss = evaluate(params, val_dataset, model, map_weight)
        print(
            "Epoch {:05d} | train_loss {:.4f} | val_loss {:.4f}".format(
                epoch+1, loss, val_loss
            )
        )
    return model



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'{device} device found')
    project_dir = './data/project_small_209'

    matrix_dir = os.path.join(project_dir, 'topo-1/6/matrix') #!
    point_num = 209 #!

    map_weight = pd.read_csv(os.path.join(project_dir, 'map_weight.csv'), index_col=0).to_numpy().T
    map_weight = torch.tensor(map_weight, dtype=torch.float32).to(device) # shape = (N*N) * 5
    

    params = Params(point_num, device, project_dir, matrix_dir)

    batch_size = 4
    epoches = 100
    train_val_test_ratio = [0.8, 0.1, 0.1]
    params.set_data_params(batch_size, train_val_test_ratio)
    graphs = load_graphs(params)
    dataset = split_dataset(params, graphs)

    node_feat_dim = graphs[0].ndata['feat'].shape[1] # 输入节点特征维度
    model = GNN(params, in_size=node_feat_dim, out_size=2).to(device)

    loss_fcn = EuclideanDistanceLoss()
    lr = 1e-3

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    params.set_model_params(loss_fcn, optimizer, epoches)

    train(params, dataset, model, map_weight)

    pass

if __name__ == '__main__':
    main()