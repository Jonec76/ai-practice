#%%
import pandas as pd
import matplotlib.pyplot as plt
import csv
train_file_name = "train.csv"
test_file_name = "test.csv"
output_file_name = "pm10_output.csv"
train_days = 9
features_num = 18
label_pm25_idx = 9
train_features = [ 4, 5, 6, 7, 8, 9, 11, 12, 14]
# train_features = [ 8, 9]

def get_file_data(file_name):
    row_data = []
    with open(file_name,
     encoding="big5") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        title = True
        for row in csv_reader:
            if title:
                title = False
                continue
            row_data.append(row)
    return row_data

row_data = get_file_data(train_file_name)
# %%
def clean_data(row_data):
    train_data = [[] for i in range(features_num)] 
    idx = 0
    for row in row_data:
        for d in row[3:]:
            if d == 'NR':
                d = 0
            train_data[idx].append(float(d))
        idx = idx + 1
        idx = idx % features_num
    return train_data
train_data = clean_data(row_data)
# %%
pm2_5 = []
def get_label(train_data):
    pm2_5 = []
    for data in train_data[label_pm25_idx]:
        pm2_5.append(float(data))
    return pm2_5[train_days:]
pm2_5 = get_label(train_data)
# %%
import numpy as np
import random
from statistics import mean
from statistics import stdev

learning_rate = 0.3

dim = train_days * len(train_features)
w = res = [random.randrange(1, 10, 1) for i in range(dim)]
b = 0.0
grad_w = [0.0]*dim
grad_b = [0.0]*dim
# data_num = len(train_data[train_features]) - train_days - 1
data_num = int(5750*0.9)
valid_data_num = int(5750*0.1)
for iter in range(300):
    for idx in range(data_num):   
        # feature scaling
        X = []
        for features_idx in train_features:
            tmp_x = train_data[features_idx][idx:idx+train_days]
            m = mean(tmp_x)
            std = stdev(tmp_x) + 0.01
            for i in tmp_x:
                X.append((i-m)/std)
        
        # SGD training
        y_hat = 0.0
        for i in range(dim):
            y_hat += (w[i] * X[i])
        y_hat += b

        for i in range(dim):
            grad_w[i] = -2/data_num * (pm2_5[idx] - y_hat) * X[i]
            w[i] = w[i] - learning_rate*grad_w[i]
        grad_b = -2/data_num * (pm2_5[idx] - y_hat)
        b = b - learning_rate*grad_b
    
    error = 0.0
    for i in range(valid_data_num):
        test_x = []
        y_hat = 0
        for features_idx in train_features:
            tmp_x = train_data[features_idx][i+4600:i+train_days+4600]
            tmp_x = [float(x) for x in tmp_x]
            #test_data scaling
            m = mean(tmp_x)
            std = stdev(tmp_x) + 0.01
            for x in tmp_x:
                test_x.append((x-m)/std)
            
        for i in range(dim):
            y_hat += (w[i] * test_x[i])
        y_hat += b
        y_hat = y_hat if y_hat > 0 else 0
        error += ((pm2_5[i+4600]-y_hat)**2)
    error /= valid_data_num
    if iter%5 == 0:
        print("[iter %d] error: %f"%(iter, error))
    
# %%
row_data = []
row_data_ctr = 0
with open(test_file_name, encoding="big5") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        row_data.append(row)
        row_data_ctr += 1
data_num = int(row_data_ctr/features_num)
y_hat_arr = []

for i in range(data_num):
    test_x = []
    y_hat = 0
    for features_idx in train_features:
        tmp_x = row_data[features_idx + features_num*i][2:]
        tmp_x = [float(x) for x in tmp_x]
    
        #test_data scaling
        m = mean(tmp_x)
        std = stdev(tmp_x) + 0.01
        for x in tmp_x:
            test_x.append((x-m)/std)
        
    for i in range(dim):
        y_hat += (w[i] * test_x[i])
    y_hat += b
    y_hat = y_hat if y_hat > 0 else 0
    y_hat_arr.append(y_hat)
# %%
def output_pred(y_hat_arr):
    f= open(output_file_name,"w+")
    f.write("id,value\n")
    for i in range(data_num):
        f.write("id_%d,%d\n"%(i, y_hat_arr[i]))

output_pred(y_hat_arr)

# %%

# print(train_data[0][0:20])
def plot_cmp(num):
    for i in range(18):
        fig = plt.figure()
        ax = plt.axes()
        x = np.arange(1,num)
        x = np.linspace(1, num, num-1)
        l = row_data[i][1]
        plt.xlabel("days")
        plt.title(i)
        plt.plot(x, train_data[i][0:num-1],'r-',label=l)
        plt.plot(x, train_data[9][0:num-1], label='pm2.5')
        plt.legend()
# plot_cmp(1500)
# %%
