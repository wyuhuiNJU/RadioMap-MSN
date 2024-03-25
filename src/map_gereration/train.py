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

torch.manual_seed(42)
np.random.seed(42)

class Params():
    def __init__(self, project_name) -> None:
        self.project_name = project_name
        pass

class RegressionModel(nn.Module):
    def __init__(self, input_size, scale_factor):
        super(RegressionModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128 * scale_factor),
            nn.ReLU(),
            nn.Linear(128 * scale_factor, 256 * scale_factor),
            nn.ReLU(),
            nn.Linear(256 * scale_factor, 512 * scale_factor),
            nn.ReLU(),
            nn.Linear(512 * scale_factor, 1024 * scale_factor),
            nn.ReLU(),
            nn.Linear(1024 * scale_factor, 512 * scale_factor),
            nn.ReLU(),
            nn.Linear(512 * scale_factor, 256 * scale_factor),
            nn.ReLU(),
            nn.Linear(256 * scale_factor, 128 * scale_factor),
            nn.ReLU(),
            nn.Linear(128 * scale_factor, 1)
        )

    def forward(self, x):
        return self.layers(x)

def load_data():
    df = pd.read_csv('./model/model2/trainData_dBm.csv').sample(frac=1)
    df_x = df.iloc[:, :5]
    df_y = df.iloc[:, 5:]

    # x y 分别标准化
    scaler_x = StandardScaler()
    scaler_y = MinMaxScaler()
    df_x = scaler_x.fit_transform(df_x)
    df_y = scaler_y.fit_transform(df_y)
    df_scalered = np.hstack((df_x, df_y))

    train_val, test = train_test_split(df_scalered, test_size=0.1)
    train, val = train_test_split(train_val, test_size=1/9)

    torch.save(scaler_x, os.path.join(log_dir, 'scaler_x.pth'))
    torch.save(scaler_y, os.path.join(log_dir, 'scaler_y.pth'))

    data = dict()
    data['all_X'] = torch.tensor(df_scalered[:, :5], dtype=torch.float32).to('cuda')
    data['all_Y'] = torch.tensor(df_scalered[:, 5:], dtype=torch.float32).to('cuda')
    data['train_X'] = torch.tensor(train[:, :5], dtype=torch.float32).to('cuda')
    data['train_Y'] = torch.tensor(train[:, 5:], dtype=torch.float32).to('cuda')
    data['val_X'] = torch.tensor(val[:, :5], dtype=torch.float32).to('cuda')
    data['val_Y'] = torch.tensor(val[:, 5:], dtype=torch.float32).to('cuda')
    data['test'] = torch.tensor(test, dtype=torch.float32).to('cuda')
    data['test_X'] = torch.tensor(test[:, :5], dtype=torch.float32).to('cuda')
    data['test_Y'] = torch.tensor(test[:, 5:], dtype=torch.float32).to('cuda')
    data['scaler_X'] = scaler_x
    data['scaler_Y'] = scaler_y
    return data

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad() #  归零梯度
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward() # 反向传播
        optimizer.step() # 更新模型参数

        total_loss += loss.item()

    return total_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device):
    model.eval() # 将模型切换到评估模式
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()

    return total_loss / len(val_loader)

def plot_history(train_losses, val_losses):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(log_dir, 'regression.png'))

def plot_error(data, model, flag, scaler_y):
    actual_X = data[f'{flag}_X']
    actual_Y = data[f'{flag}_Y']
    predict_Y = model(actual_X).detach().cpu().numpy()
    prediction = scaler_y.inverse_transform(predict_Y)
    actual = scaler_y.inverse_transform(actual_Y.cpu().numpy())
    error = prediction - actual
    data = np.hstack((actual, prediction, error))
    data = data[data[:, 1].argsort()]
    actual_sorted = data[:, 0]
    prediction_sorted = data[:, 1]

    length = len(actual)
    index = np.arange(length)
    plt.figure()
    plt.plot(index, actual_sorted, label='Actual')
    plt.plot(index, prediction_sorted, label='Prediction')
    plt.legend()
    plt.title(f'{flag} Database: Index-Actual-Prediction')
    plt.savefig(os.path.join(log_dir, f'{flag}_indx-actual-prediction.jpg'))
    mean_error = np.mean(np.abs(error))
    print(f'{flag} MAE: {mean_error}')

def main(scale_factor):
    project_name = 'project_170'

    params = Params(project_name)
    data = load_data(params)
    
    input_features = data['train_X'].shape[1]
    model = RegressionModel(input_features, scale_factor)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = TensorDataset(data['train_X'], data['train_Y'])
    val_dataset = TensorDataset(data['val_X'], data['val_Y'])
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    torch.save(model.state_dict(), os.path.join(model_path, 'model.keras'))
    print(f'Model saved at {log_dir}')
    plot_history(train_losses, val_losses)
    plot_error(data, model, flag='train', scaler_y=data['scaler_Y'])
    plot_error(data, model, flag='val', scaler_y=data['scaler_Y'])
    print('*************************************')
    plot_error(data, model, flag='test', scaler_y=data['scaler_Y'])
    print('*************************************')
    plot_error(data, model, flag='all', scaler_y=data['scaler_Y'])

if __name__ == '__main__':
    log_dir = os.path.join('./model/model2/log-torch-1')
    os.makedirs(log_dir, exist_ok=True)
    model_path = os.path.join(log_dir, 'model')
    os.makedirs(model_path, exist_ok=True)
    batch = 128
    lr = 1e-5
    scale_factor = 2
    epochs = 200
    main(scale_factor=2)

