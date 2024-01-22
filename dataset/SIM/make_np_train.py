import os
import numpy as np
import pandas as pd
import pickle as pk
from sklearn.preprocessing import StandardScaler
import random
random.seed(0)

# 比较懒狗，直接for loop了没用multiprocessing

pth = './'
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_folder = 'data/processed_sim_data'
columns = ['agent_id', 'latitude', 'longitude', 'time', 'stay_minutes']
baseline = pd.DataFrame(np.load(os.path.join(ROOT_DIR, 'baseline/data_stays_v2.npy'), allow_pickle=True), columns=columns)
kitware = pd.DataFrame(np.load(os.path.join(ROOT_DIR, 'kitware/data_stays_v2.npy'), allow_pickle=True), columns=columns)
l3harris = pd.DataFrame(np.load(os.path.join(ROOT_DIR, 'l3harris/data_stays_v2.npy'), allow_pickle=True), columns=columns)
min_agent = min(baseline['agent_id'].max(), kitware['agent_id'].max(), l3harris['agent_id'].max())

# window size is 64 for all datasets
win_size = 64

data_dims = 3
b9k1, k9l1, l9b1 = [], [], []
b9k1_tst, k9l1_tst, l9b1_tst = [], [], []
b9k1_label, k9l1_label, l9b1_label =[],[],[]
b9k1_trn_time, k9l1_trn_time, l9b1_trn_time = [], [], []
b9k1_tst_time, k9l1_tst_time, l9b1_tst_time = [], [], []

timegap = 1.4 * 86400
ids = list(set(baseline['agent_id']).intersection(kitware['agent_id']).intersection(l3harris['agent_id']))
ids_trn = ids[0:int(0.2*len(ids))]
ids_tst = ids[int(0.2*len(ids)):int(0.25*len(ids))]

for i in ids_trn:
    d1,d2,d3 = baseline.loc[baseline['agent_id']==i, ['agent_id', 'latitude', 'longitude', 'stay_minutes']], kitware.loc[kitware['agent_id']==i, ['agent_id', 'latitude', 'longitude', 'stay_minutes']], l3harris.loc[l3harris['agent_id']==i, ['agent_id', 'latitude', 'longitude', 'stay_minutes']]
    t1, t2, t3 = baseline.loc[baseline['agent_id']==i, ['time']], kitware.loc[kitware['agent_id']==i, ['time']], l3harris.loc[l3harris['agent_id']==i, ['time']]
    for j in range(len(d1) // win_size):
        b9k1.append(np.array(d1)[win_size*j: win_size*(j+1), 1:])
        b9k1_trn_time.append(np.array(t1)[win_size*j: win_size*(j+1)])
    for j in range(len(d2) // win_size):
        k9l1.append(np.array(d2)[win_size*j: win_size*(j+1), 1:])
        k9l1_trn_time.append(np.array(t2)[win_size*j: win_size*(j+1)])
    for j in range(len(d3) // win_size):
        l9b1.append(np.array(d3)[win_size*j:win_size*(j+1), 1:])
        l9b1_trn_time.append(np.array(t3)[win_size*j:win_size*(j+1)])


for i in ids_tst:
    t = random.uniform(0, 14-1.4) * 86400
    d1,d2,d3 = baseline.loc[baseline['agent_id']==i], kitware.loc[kitware['agent_id']==i], l3harris.loc[l3harris['agent_id']==i]
    tstamp1_unix, tstamp2_unix, tstamp3_unix = np.array([x.timestamp() for x in d1['time']]), np.array([x.timestamp() for x in d2['time']]), np.array([x.timestamp() for x in d3['time']])
    
    t1,t2,t3 = tstamp1_unix[0] + t, tstamp2_unix[0] + t, tstamp3_unix[0] + t
    i1, i2, i3 = d1[(tstamp1_unix >= t1) & (tstamp1_unix < t1 + timegap)], d2[(tstamp2_unix >= t2) & (tstamp2_unix < t2 + timegap)], d3[(tstamp3_unix >= t3) & (tstamp3_unix < t3 + timegap)]
    
    # mix1 = pd.concat([d1[(tstamp1_unix < t1)], i2, d1[(tstamp1_unix >= t1 + timegap)]], axis=0)
    # mix2 = pd.concat([d2[(tstamp2_unix < t2)], i3, d2[(tstamp2_unix >= t2 + timegap)]], axis=0)
    mix3 = pd.concat([d3[(tstamp3_unix < t3)], i1, d3[(tstamp3_unix >= t3 + timegap)]], axis=0)
    
    # some label for one series: 1 or 0
    # label1 = np.zeros(len(mix1))
    # label1[len(d1[(tstamp1_unix < t1)]):len(d1[(tstamp1_unix < t1)])+len(i2)] = 1
    # label2 = np.zeros(len(mix2))
    # label2[len(d2[(tstamp2_unix < t2)]):len(d2[(tstamp2_unix < t2)])+len(i3)] = 1
    label3 = np.zeros(len(mix3))
    label3[len(d3[(tstamp3_unix < t3)]):len(d3[(tstamp3_unix < t3)])+len(i1)] = 1
    
    # for j in range(len(mix1) // win_size):
    #     b9k1_tst.append(np.array(mix1.loc[:,['agent_id', 'latitude', 'longitude', 'stay_minutes']])[win_size*j:win_size*(j+1), 1:])
    #     b9k1_label.append(label1[win_size*j:win_size*(j+1)])
    #     b9k1_tst_time.append(np.array(mix1['time'])[win_size*j:win_size*(j+1)])
        
    # for j in range(len(mix2) // win_size):
    #     k9l1_tst.append(np.array(mix2.loc[['agent_id', 'latitude', 'longitude', 'stay_minutes']])[win_size*j:win_size*(j+1), 1:])
    #     k9l1_label.append(label2[win_size*j:win_size*(j+1)])
    #     k9l1_tst_time.append(mix2[wi.locn_size*j:win_size*(j+1), ['time']])
        
    for j in range(len(mix3) // win_size):
        content = np.array(mix3.loc[:,['agent_id', 'latitude', 'longitude', 'stay_minutes']])[win_size*j:win_size*(j+1), 1:]
        l9b1_tst.append(content)
        l9b1_label.append(label3[win_size*j:win_size*(j+1)])
        time = np.array(mix3['time'])[win_size * j: win_size * (j+1)]
        l9b1_tst_time.append(time)
        if len(content) != len(time):
            print(mix3)
            exit(0)

l9b1_tst = np.array(l9b1_tst)
l9b1_tst_time = np.array(l9b1_tst_time)
print(l9b1_tst.shape, l9b1_tst_time.shape)

b9k1_label = np.array(b9k1_label)
# k9l1_label = np.array(k9l1_label)
l9b1_label = np.array(l9b1_label)

for i in ['b9k1', 'l9b1']:
    np.save(i+ '/' + i + '_train.npy', np.array(eval(i)), allow_pickle=True)
    np.save(i+ '/' + i + '_train_time.npy', np.array(eval(i+'_trn_time')).flatten(), allow_pickle=True)
    np.save(i+ '/' + i + '_test.npy', np.array(eval(i+'_tst')), allow_pickle=True)
    np.save(i+ '/' + i + '_test_label.npy', np.array(eval(i + '_label')), allow_pickle=True)
    np.save(i+ '/' + i + '_test_time.npy', np.array(eval(i+'_tst_time')).flatten(), allow_pickle=True)


# need same entity for both train and test