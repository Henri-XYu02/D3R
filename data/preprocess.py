import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def getTimeEmbedding(time):
    df = pd.DataFrame(time, columns=['time'])
    df['time'] = pd.to_datetime(df['time'])

    df['minute'] = df['time'].apply(lambda row: row.minute / 59 - 0.5)
    df['hour'] = df['time'].apply(lambda row: row.hour / 23 - 0.5)
    df['weekday'] = df['time'].apply(lambda row: row.weekday() / 6 - 0.5)
    df['day'] = df['time'].apply(lambda row: row.day / 30 - 0.5)
    df['month'] = df['time'].apply(lambda row: row.month / 365 - 0.5)

    return df[['minute', 'hour', 'weekday', 'day', 'month']].values


def getStable(data, period=6, w=128):
    trend = pd.DataFrame(data).rolling(period, center=True).median().values
    stable = data - trend
    return data[w // 2:-w // 2, :], stable[w // 2:-w // 2, :]


def getData(path='./dataset/SIM/', dataset='SIM', period=6, train_rate=0.8, multi_entity=0):

    init_data = np.float32(np.load(path + dataset + '/' + dataset + '_train.npy', allow_pickle=True))
    init_time = getTimeEmbedding(np.load(path + dataset + '/' + dataset + '_train_time.npy', allow_pickle=True))

    test_data = np.float32(np.load(path + dataset + '/' + dataset + '_test.npy', allow_pickle=True))
    test_time = np.load(path + dataset + '/' + dataset + '_test_time.npy', allow_pickle=True)
    test_time = getTimeEmbedding(test_time)
    test_label = np.load(path + dataset + '/' + dataset + '_test_label.npy', allow_pickle=True)

    # 摆烂了没有处理agent id stable问题
    if multi_entity:
        init_data = np.reshape(init_data, (-1, 3))
        test_data = np.reshape(test_data, (-1, 3))
        init_time = init_time
        test_time = test_time
        test_label = test_label.reshape(-1, 1)

    
    scaler = StandardScaler()
    scaler.fit(init_data)

    init_data, init_stable = getStable(init_data, period=period, w=128)
    period = 128
    init_time = init_time[period // 2:-period // 2, :]
    init_label = np.zeros((len(init_data), 1))
    test_stable = np.zeros_like(test_data)
    

    train_data = init_data[:int(train_rate * len(init_data)), :]
    train_time = init_time[:int(train_rate * len(init_time)), :]
    train_stable = init_stable[:int(train_rate * len(init_stable)), :]
    train_label = init_label[:int(train_rate * len(init_label)), :]

    valid_data = init_data[int(train_rate * len(init_data)):, :]
    valid_time = init_time[int(train_rate * len(init_time)):, :]
    valid_stable = init_stable[int(train_rate * len(init_stable)):, :]
    valid_label = init_label[int(train_rate * len(init_label)):, :]

    data = {
        'train_data': train_data, 'train_time': train_time, 'train_stable': train_stable, 'train_label': train_label,
        'valid_data': valid_data, 'valid_time': valid_time, 'valid_stable': valid_stable, 'valid_label': valid_label,
        'init_data': init_data, 'init_time': init_time, 'init_stable': init_stable, 'init_label': init_label,
        'test_data': test_data, 'test_time': test_time, 'test_stable': test_stable, 'test_label': test_label
    }

    return data
