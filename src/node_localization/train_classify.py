# /*
#  * @Author: wyuhui 
#  * @Date: 2024-03-29 16:09:21 
#  * @Last Modified by:   wyuhui 
#  * @Last Modified time: 2024-03-29 16:09:21 
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
    def __init__(self, point_num, device, project_dir, matrix_dir, num_classes) -> None:
        self.N = point_num # 仿真采样点数
        self.device = device
        self.project_dir = project_dir
        self.matrix_dir = matrix_dir
        self.num_classes = num_classes

        pass

    def set_model_params(self, loss_fcn, optimizer, epoches, batch_size):
        self.loss_fcn = loss_fcn
        self.optimizer = optimizer
        self.epoches = epoches
        self.batch_size = batch_size
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
        self.relu = nn.ReLU()
        N = params.N # 仿真采样点数,也是地图矩阵的维度
        self.conv1 = dglnn.GraphConv(in_size, N*2, weight=True, bias=True) # 升维到N
        self.conv2 = dglnn.GraphConv(N*2, N, norm='both', weight=False, bias=False) # 地图矩阵权重层
        self.linear1 = nn.Linear(N, params.num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features, weight, edge_weight=None):
        x = self.conv1(g, features, edge_weight=edge_weight)
        x = self.dropout(x)
        x = self.conv2(g, x, weight=weight, edge_weight=edge_weight)
        x = self.relu(x)
        x = self.linear1(x)
        x = torch.softmax(x, dim=1)
        return x

def load_graphs(params: Params):
    positions_labeled = pd.read_csv(os.path.join(params.project_dir,'positions_labeled.csv'), index_col=0).to_numpy()
    mean_x = sum(positions_labeled.T[0]) / positions_labeled.shape[0]
    mean_y = sum(positions_labeled.T[1]) / positions_labeled.shape[0]
    
    connection_matrix = pd.read_csv(os.path.join(params.project_dir, 'connection_matrix.csv'), index_col=0).to_numpy()

    all_graphs = []
    for csv_name in tqdm(os.listdir(params.matrix_dir), desc='loading graphs...'):
        matrix = pd.read_csv(os.path.join(params.matrix_dir, csv_name), index_col=0)
        g_nx = nx.from_numpy_array(matrix.to_numpy())
        nodes_id = matrix.columns.astype(int)
        g = dgl.from_networkx(g_nx)
        g.ndata['node_id'] = torch.tensor(nodes_id)
        # g.ndata['position'] = torch.tensor(np.array([positions[i] for i in nodes_id]), dtype=torch.float32) # 位置标签
        labels = torch.tensor([positions_labeled[i][-1] for i in nodes_id]).long()
        g.ndata['label'] = F.one_hot(labels, params.num_classes).float()

        # g.ndata['label'] = torch.tensor(np.array([[positions_labeled[i][-1]] for i in nodes_id]), dtype=torch.float)
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

def split_dataset(dataset, train_val_test_ratio: list[int]):
    [train_ratio, val_ratio, test_ratio] = train_val_test_ratio
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)
    test_size = dataset_size - train_size - val_size

    random.shuffle(dataset)
    return {
        'train_data': dataset[:train_size],
        'val_data': dataset[train_size: train_size + val_size],
        'test_data': dataset[-test_size: ]
    }

def evaluate(params: Params, dataset, model, map_weight):
    model.eval()
    all_acc = []
    with torch.no_grad():
        # total_err = 0
        for g in dataset:
            logits = model(g, g.ndata['feat'], map_weight, g.edata['power'])
            _, key_idx = torch.max(logits, dim=1)
            _, predict_idx = torch.max(g.ndata['label'], dim=1)
            correct = torch.sum(key_idx == predict_idx)
            acc = correct / g.num_nodes()
            all_acc.append(acc)
    return sum(all_acc) / len(all_acc)


def train(params: Params, dataset, model: GNN, map_weight):
    train_dataset = dataset['train_data']
    val_dataset = dataset['val_data']

    [loss_fcn, optimizer] = [params.loss_fcn, params.optimizer]
    model.train()

    for epoch in range(params.epoches):
        for g in tqdm(train_dataset, 'training...'):
            logits = model(g, g.ndata['feat'], map_weight, g.edata['power'])
            loss = loss_fcn(logits, g.ndata['label'])
            loss.backward()
            optimizer.zero_grad()
            optimizer.step()

        val_loss = evaluate(params, val_dataset, model, map_weight)

        print(
            "Epoch {:05d} | train_loss {:.4f} | val_loss {:.4f}".format(
                epoch+1, loss.item(), val_loss
            )
        )
    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'{device} device found')
    project_dir = './data/project_209'
    matrix_dir = os.path.join(project_dir, 'topo-1/3/matrix')
    point_num = 209
    num_classes = 5 # 节点位置类别数量

    connection_matrix = pd.read_csv(os.path.join(project_dir, 'connection_matrix.csv'), index_col=0).to_numpy()
    map_weight = np.concatenate([connection_matrix, connection_matrix.T], axis=0) # 行到列
    map_weight = torch.tensor(map_weight, dtype=torch.float32).to(device)

    params = Params(point_num, device, project_dir, matrix_dir, num_classes)

    graphs = load_graphs(params)
    dataset = split_dataset(graphs, train_val_test_ratio=[0.8, 0.1, 0.1])

    node_feat_dim = graphs[0].ndata['feat'].shape[1] # 输入节点特征维度
    model = GNN(params, in_size=node_feat_dim, out_size=1).to(device)
    loss_fcn = nn.CrossEntropyLoss()
    lr = 1e-2
    batch_size = 5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    epoches = 100
    params.set_model_params(loss_fcn, optimizer, epoches, batch_size)
    train(params, dataset, model, map_weight)

    pass

if __name__ == '__main__':
    main()