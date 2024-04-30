import torch
import torch.nn as nn
import torch.optim as optim

# 定义逐层训练函数
def train_layer(model, input_data, target, layer_index, num_epochs=100, learning_rate=0.01):
    # 提取当前层的参数
    params = model.layers[layer_index].parameters()
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.SGD(params, lr=learning_rate)
    
    # 训练当前层
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(input_data)
        loss = criterion(outputs, target)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 打印训练信息
        if (epoch+1) % 10 == 0:
            print ('Layer {}, Epoch [{}/{}], Loss: {:.4f}'.format(layer_index+1, epoch+1, num_epochs, loss.item()))

# 示例数据
input_size = 5
hidden_sizes = [10, 20, 10]
output_size = 1
input_data = torch.randn(100, input_size)
target = torch.randn(100, output_size)

# 定义贪婪 MLP 类
class GreedyMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(GreedyMLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # 构建每一层的线性层
        self.layers = nn.ModuleList()
        for i in range(len(hidden_sizes)):
            if i == 0:
                self.layers.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        
        # 输出层
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        
    def forward(self, x):
        # 前向传播
        for layer in self.layers:
            x = layer(x)
            x = torch.relu(x)  # 使用ReLU作为激活函数
        x = self.output_layer(x)
        return x

# 创建贪婪 MLP 实例
model = GreedyMLP(input_size, hidden_sizes, output_size)

# 逐层训练
for i in range(len(hidden_sizes)):
    print(f"Training layer {i+1}")
    train_layer(model, input_data, target, i)
