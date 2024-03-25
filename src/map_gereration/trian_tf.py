from numpy.random import seed
seed(42)
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from keras.optimizers import Adam
import keras
import os
import pickle
from matplotlib import pyplot as plt
from datetime import datetime
import seaborn

begin_time = datetime.now()

scale_factor = 2 # 调节网络规模
lr = 1e-3  # 学习率
epoch = 100
batch = 128
N = 209


project_path = './data/project_170'
save_path = './model/project_170/model_2_tf'
os.makedirs(save_path, exist_ok=True)
ckpt_path = os.path.join(save_path,'ckpt')
log_dir = os.path.join(save_path,f'log-1')
result_dir = os.path.join(log_dir, 'result')
os.makedirs(result_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

model_path = os.path.join(log_dir, 'model')
if not os.path.exists(model_path):
    os.makedirs(model_path)



def load_data(trainDataPath):
    # 输入输出, 分别归一化
    # 打乱 => 分别归一化 => 拆分
    df = pd.read_csv(trainDataPath).sample(frac=1)
    df_x = df.iloc[:,2:6]
    df_y = df.iloc[:, -1:]

    scaler_x = StandardScaler()
    scaler_y = MinMaxScaler()
    df_x = scaler_x.fit_transform(df_x)
    df_y = scaler_y.fit_transform(df_y)
    # df_scalered = pd.concat((df_x, df_y),axis=1)
    df_scalered = np.hstack((df_x, df_y))
    
    train_val, test = train_test_split(df_scalered, test_size=0.1)
    train, val = train_test_split(train_val, test_size=1/9)


    pickle.dump(scaler_x, open(os.path.join(log_dir, 'scaler_x.pkl'), 'wb'))
    pickle.dump(scaler_y, open(os.path.join(log_dir, 'scaler_y.pkl'), 'wb'))

    
    # 数据集
    data = dict()
    data['all_X'] = df_scalered[:, :4]
    data['all_Y'] = df_scalered[:, -1:]
    data['train_X'] = train[:, :4]
    data['train_Y'] = train[:, -1:]
    data['val_X'] = val[:, :4]
    data['val_Y'] = val[:, -1:]
    data['test'] = test
    data['test_X'] = test[:, :4]
    data['test_Y'] = test[:, -1:]
    data['scaler_X'] = scaler_x
    data['scaler_Y'] = scaler_y
    return data


def build_network(input_features=None):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        inputs = Input(shape=(input_features,), name="input")
        x = Dense(128 * scale_factor, activation='relu', name="hidden1")(inputs)
        x = Dense(256 * scale_factor, activation='relu', name="hidden2")(x)
        x = Dense(512 * scale_factor, activation='relu', name="hidden3")(x)
        # x = Dense(1024 * scale_factor, activation='relu', name="hidden4")(x)
        # x = Dense(2048 * scale_factor, activation='relu', name="hidden5")(x)
        x = Dense(1024 * scale_factor, activation='relu', name="hidden0")(x)
        # x = Dense(2048 * scale_factor, activation='relu', name="hidden-5")(x)
        # x = Dense(1024 * scale_factor, activation='relu', name="hidden-4")(x)
        x = Dense(512 * scale_factor, activation='relu', name="hidden-3")(x)
        x = Dense(256 * scale_factor, activation='relu', name="hidden-2")(x)
        x = Dense(128 * scale_factor, activation='relu', name="hidden-1")(x)
        prediction = Dense(1, activation='linear', name="final")(x)
        model = Model(inputs=inputs, outputs=prediction)
        opt = Adam(learning_rate=lr)
        model.compile(optimizer=opt, loss='mean_absolute_error')
    return model

def plot_history(history):
    hist=pd.DataFrame(history.history)
    hist['epoch']=history.epoch
    hist.tail()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.plot(hist['epoch'],hist['loss'],label='train error')
    plt.plot(hist['epoch'],hist['val_loss'],label='val_error')
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(result_dir, 'regression.png'))

def plot_error(data, model, flag):
    acutal_X = data[f'{flag}_X']
    actual_Y = data[f'{flag}_Y']
    predict_Y = model.predict(acutal_X)
    prediction = data['scaler_Y'].inverse_transform(predict_Y)
    actual = data['scaler_Y'].inverse_transform(actual_Y)
    error = prediction - actual
    data = np.hstack((actual, prediction, error))
    data = data[data[:, 1].argsort()]
    actual_sorted = data[:, 0]
    prediction_sorted = data[:, 1]
    error_sorted = data[:, 2]


    length = len(actual)
    index = np.arange(length)
    plt.figure()
    # index-actual-prediction
    plt.plot(index, actual_sorted, label='actual')
    plt.plot(index, prediction_sorted, label='prediction')
    plt.legend()
    plt.title(f'{flag} database: index-actual-prediction')
    plt.savefig(os.path.join(log_dir, f'{flag}_indx-actual-prediction.jpg'))
    mean_error = sum([abs(x) for x in error]) / len(actual)
    print(f'{flag} MAE :{mean_error}')

    return 0

def draw_heatmap(trainDataPath, prediction):
    N = 209
    shape = (19, 11)
    trainData = pd.read_csv(trainDataPath)
    powerData = trainData['power']
    for layer in range(N):
        origin_power = np.reshape(powerData[layer*N: (layer+1)*N], shape).T
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

        plt.savefig(os.path.join(result_dir, f'Tx{layer+1}.jpg'))
        plt.close()


    

def main():
    trainDataPath = './data/project_170/data.csv'
    trainData = pd.read_csv(trainDataPath)
    data = load_data(trainDataPath)
    input_features = data['train_X'].shape[1]
    model = build_network(input_features)
    print('network sturcture')
    print(model.summary)
    print("Training Data Shape: " + str(data["train_X"].shape))

    if os.path.exists(ckpt_path):
        print("--------------------------loading the model-----------------------------")
        model.load_weights(ckpt_path)

    callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path,save_best_only=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(x=data["train_X"], y=data["train_Y"], batch_size=batch, epochs=epoch, validation_data=(data["val_X"], data["val_Y"]), callbacks=[callback, tensorboard_callback])
    model.save(os.path.join(model_path, 'model.keras'))
    scaler_x = data['scaler_X']
    scaler_y = data['scaler_Y']
    prediction = scaler_y.inverse_transform(model.predict(scaler_x.transform(trainData.iloc[:, 2:6])))

    print(f'model saved at {log_dir}')
    plot_history(history)
    plot_error(data, model, flag='train')
    plot_error(data, model, flag='val')
    print('-->', end='')
    plot_error(data, model, flag='test')
    plot_error(data, model, flag='all')
    draw_heatmap(trainDataPath, prediction)

    end_time = datetime.now()
    print(f'total time use: {end_time - begin_time}')


if __name__ == '__main__':
    main()

