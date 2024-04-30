# /*
#  * @Author: wyuhui 
#  * @Date: 2024-03-24 18:28:37 
#  * @Last Modified by:   wyuhui 
#  * @Last Modified time: 2024-03-24 18:28:37 
#  */


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import seaborn
import time


torch.manual_seed(42)
np.random.seed(42)


class Params():
    def __init__(self, device) -> None:
        self.device = device
        return

    def set_project_param(self, project_name, model_name, log_name, size):
        self.project_name = project_name
        self.model_name = model_name
        self.log_save_dir = os.path.join('./model', project_name, model_name, log_name)
        os.makedirs(self.log_save_dir, exist_ok=True)

        self.size = size
        self.N = self.size[0] * self.size[1]
        return
    
    def set_data_param(self, x_data_label, y_data_label): # 数据集中 X 和 Y 的标签名称
        self.x_data_label = x_data_label
        self.y_data_label = y_data_label
        pass
    
    def set_model_param(self, in_size, out_size, batch_size, learning_rate, epochs, scale_factor):
        self.in_size = in_size
        self.out_size = out_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.scale_factor = scale_factor # 调节隐藏层宽度
        pass


def load_data(params: Params):
    data = pd.read_csv(os.path.join('./data', params.project_name, 'data.csv'), index_col=0)
    data = data.sample(frac=1)
    print(data)
    data_x = pd.concat([data[label] for label in params.x_data_label] ,axis=1)
    data_y = pd.concat([data[label] for label in params.y_data_label] ,axis=1)

    scaler_x = StandardScaler()
    scaler_y = MinMaxScaler()
    data_x = scaler_x.fit_transform(data_x)
    data_y = scaler_y.fit_transform(data_y)

    data_scalered = np.hstack((data_x, data_y))

    train_val, test = train_test_split(data_scalered, test_size=0.1)
    train, val = train_test_split(train_val, test_size=1/9)

    torch.save(scaler_x, os.path.join(params.log_save_dir, 'scaler_x.pth'))
    torch.save(scaler_y, os.path.join(params.log_save_dir, 'scaler_y.pth'))

    x_dim = data_x.shape[1]
    y_dim = data_y.shape[1]

    ret = dict() # 返回值
    ret['all_X'] = data_scalered[:, :x_dim] 
    ret['all_Y'] = data_scalered[:, -y_dim:]
    ret['train_X'] = train[:, :x_dim] 
    ret['train_Y'] = train[:, -y_dim:]
    ret['val_X'] = val[:, :x_dim] 
    ret['val_Y'] = val[:, -y_dim:]
    ret['test_X'] = test[:, :x_dim] 
    ret['test_Y'] = test[:, -y_dim:]

    ret_tensor = dict() # 转为cuda上的张量
    for key, value in ret.items():
        ret_tensor[key] = torch.tensor(value, dtype=torch.float32).to(params.device)

    ret_tensor['scaler_X'] = scaler_x
    ret_tensor['scaler_Y'] = scaler_y

    return ret_tensor


class RegressionModel(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super().__init__()
        # self.fn = nn.Identity()
        self.fn = nn.ReLU()
        self.layers = nn.Sequential(
            nn.Linear(in_size, 1 * scale_factor),
            self.fn,
            nn.Linear(1 * scale_factor, 2 * scale_factor),
            self.fn,
            nn.Linear(2 * scale_factor, 4 * scale_factor),
            self.fn,
            nn.Linear(4 * scale_factor, 8 * scale_factor),
            self.fn,
            nn.Linear(8 * scale_factor, 4 * scale_factor),
            self.fn,
            nn.Linear(4 * scale_factor, 2 * scale_factor),
            self.fn,
            nn.Linear(2 * scale_factor, 1 * scale_factor),
            self.fn,
            nn.Linear(1 * scale_factor, out_size)
        )

    def forward(self, x):
        return self.layers(x)
    

    def set_model_params(self, ):
        pass
    
def train(model: RegressionModel, train_loader, criterion, optimizer,  params: Params):
    model.train()
    total_loss = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(params.device), targets.to(params.device) #! 已经在device上了,

        optimizer.zero_grad() # 梯度归零
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward() # 反向传播
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model: RegressionModel, val_loader, criterion, params: Params):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(params.device), targets.to(params.device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def plot_history(train_losses, val_losses, params: Params):
    plt.figure()
    plt.xlabel('Epoches')
    plt.ylabel('MSE Loss')
    plt.plot(train_losses, label='Train loss')
    plt.plot(val_losses, label='Val loss')
    plt.legend()
    plt.savefig(os.path.join(params.log_save_dir, 'regression.png'))

def predict(data, model, mode, params: Params):
    actual_X = data[f'{mode}_X'] # 归一化后的 X 原始值
    actual_Y = data[f'{mode}_Y'] # 归一化后的 Y 原始值
    predict_Y = model(actual_X).detach().cpu().numpy() # Y推理值
    prediction = data['scaler_Y'].inverse_transform(predict_Y) # 反归一化
    actual = data['scaler_Y'].inverse_transform(actual_Y.cpu().numpy())
    return actual, prediction


def plot_error(data, model, mode, params: Params):
    actual, prediction = predict(data, model, mode, params)
    err = prediction - actual
    to_sort = np.hstack((actual, prediction, err))
    to_sort = to_sort[to_sort[:, 0].argsort()] # actual, prediction, err一一对应, 依据原值大小排序, 探究误差与原值大小之间的关系
    actual_sorted = to_sort[:, 0]
    prediction_sorted = to_sort[:, 1]

    # 作图
    length = len(actual)
    indexes = np.arange(length)
    plt.figure()
    plt.plot(indexes, prediction_sorted, label='Prediction')
    plt.plot(indexes, actual_sorted, label='Actual')
    plt.legend()
    plt.title(f'{mode} Database: Index-Actual-Prediction')
    plt.savefig(os.path.join(params.log_save_dir, f'{mode}_indx-actual-prediction.jpg'))
    mean_error = np.mean(np.abs(err))
    print(f'{mode} MAE: {mean_error}')

    return prediction

def darw_heatmap(params: Params, data,  model):
    N = params.N
    shape = params.size
    result_save_path = os.path.join(params.log_save_dir, 'result') # 保存位置

    actual, prediction = predict(data, model, 'all', params)


    for layer in range(N):
        origin_power = np.reshape(actual[layer*N: (layer+1)*N], shape).T
        predict_power = np.reshape(prediction[layer*N: (layer+1)*N], shape).T
        print(f'Tx{layer+1} MAE:{mean_absolute_error(predict_power, origin_power)}')
        err = predict_power - origin_power
        abs_err = np.abs(err)
        plt.figure()

        plt.subplot(2, 2, 1)
        seaborn.heatmap(origin_power, annot=False, cmap="YlOrRd", cbar=True, square='equal')
        plt.xticks([])
        plt.yticks([])
        plt.title(f'Tx{layer+1} origin map')

        plt.subplot(2, 2, 2)
        seaborn.heatmap(predict_power, annot=False, cmap="YlOrRd", cbar=True, square='equal')
        plt.xticks([])
        plt.yticks([])
        plt.title(f'Tx{layer+1} predict map')

        plt.subplot(2, 2, 3)
        seaborn.heatmap(err, annot=False, cmap="YlOrRd", cbar=True, square='equal')
        plt.xticks([])
        plt.yticks([])
        plt.title(f'Tx{layer+1} err map')

        plt.subplot(2, 2, 4)
        seaborn.heatmap(abs_err, annot=False, cmap="YlOrRd", cbar=True, square='equal')
        plt.xticks([])
        plt.yticks([])
        plt.title(f'Tx{layer+1} abs_err map')
        os.makedirs(result_save_path, exist_ok=True)
        plt.savefig(os.path.join(result_save_path, f'Tx{layer+1}.jpg'))
        plt.close()

    pass


def main():
    project_name = 'project_209'
    model_name = 'model_3'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = Params(device)
    size = (19, 11)
    params.set_project_param(project_name, model_name, log_name='log_6', size=size)
    params.set_data_param(x_data_label=['tx_x', 'tx_y', 'rx_x', 'rx_y'], y_data_label=['power'])
    data = load_data(params)

    in_size = data['train_X'].shape[1]
    out_size = data['train_Y'].shape[1]
    print(data['train_Y'].shape)
    depth = 7
    batch_size = 1024
    learning_rate = 1e-5
    epochs = 200
    scale_factor = 256 * 4
    params.set_model_param(in_size, out_size, batch_size, learning_rate, epochs, scale_factor)

    model = RegressionModel(in_size, out_size, scale_factor)
    model.to(device)

    train_dataset = TensorDataset(data['train_X'], data['train_Y'])
    val_dataset = TensorDataset(data['val_X'], data['val_Y'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    times = []
    train_losses, val_losses = [], [] # 记录 loss 以绘制曲线
    for epoch  in range(epochs):
        begin_time = time.time()
        train_loss = train(model, train_loader, criterion, optimizer, params)
        val_loss = evaluate(model, val_loader, criterion, params)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        end_time = time.time()
        times.append(end_time - begin_time)

        print(f'Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    torch.save(model, os.path.join(params.log_save_dir, 'model.pth'))
    print(f'Model saved at {params.log_save_dir}')
    plot_history(train_losses, val_losses, params)
    plot_error(data, model, 'train', params)
    plot_error(data, model, 'val', params)
    print('测试集:------>', end='')
    plot_error(data, model, 'test', params)
    plot_error(data, model, 'all', params)
    # darw_heatmap(params, data, model)
    # val_loss, 耗时, 
    epoch_history = pd.concat((pd.DataFrame(val_losses), pd.DataFrame(times)), axis=1)
    epoch_history.to_csv(os.path.join(params.log_save_dir, f'batchSize={batch_size}_scaler={scale_factor}_depth={depth}.csv'), header=['loss', 'time'])
    print(f'batchSize={batch_size}_scaler={scale_factor}_depth={depth}.csv')

    

if __name__ == '__main__':
    main()

