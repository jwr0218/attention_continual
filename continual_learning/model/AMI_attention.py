import torch
import torch.nn as nn
import torch.nn.init as init

import torch.nn.functional as F
import numpy as np
import math 
class LinearModel(nn.Module):
    def __init__(self, input_size, name='linear'):
        super(LinearModel, self).__init__()
        self.name = name
        self.hidden = nn.Linear(in_features=input_size, out_features=1000)
        self.n_heads = 4
        self.head_dim = 1000 // self.n_heads
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to('cuda')
        self.batchnorm = nn.BatchNorm1d(
            1000,  # Number of feature channels in the input
            eps=0.001,  # Corresponds to `epsilon` in Keras
            momentum=0.01,  # Corresponds to `momentum` in Keras
            affine=True,  # Whether to learn scale (gamma) and bias (beta)
            track_running_stats=True  # Corresponds to `synchronized=False` in Keras
        )
        


        
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.hidden(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        return x
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.linear_1 = LinearModel(72)
        self.linear_2 = LinearModel(1000)
        self.dropout_3 = nn.Dropout(0.3)
        self.linear_3 = LinearModel(1000)
        self.dropout_4 = nn.Dropout(0.3)
        self.linear_4 = LinearModel(1000)
        self.output_ = nn.Linear(1000, 24)
        # init.kaiming_normal_(self.output_.weight )
        

        self.key = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(24, 24),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.linear_1(x)

        x = self.dropout_3(self.linear_2(x))
        x = self.dropout_4(self.linear_3(x))
        x = self.linear_4(x)
        x = self.output_(x)

        key = self.key(x)
        return x , key 


class AttentionModel(nn.Module):
    def __init__(self):
        super(AttentionModel, self).__init__()
        self.encoder1 = Encoder()
        self.encoder2 = Encoder()
        self.encoder3 = Encoder()
        self.encoder4 = Encoder()
        self.encoder5 = Encoder()
        self.Kl = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(24, 1),
            nn.Sigmoid()
        )
        
        self.Ql = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(24, 5),
            nn.Sigmoid()
        )


        self.flatten = nn.Flatten()

        #여기서 he normal
        self.hidden1 = nn.Linear(120, 512)
        self.hidden2 = nn.Linear(512, 256)
        self.output_ = nn.Linear(256, 24)

    def forward(self, features):
        input1, input2, input3, input4 , input5 = features[:,:,0], features[:,:,1], features[:,:,2], features[:,:,3], features[:,:,4]
        out1 , key1= self.encoder1(input1)

        out2 , key2= self.encoder2(input2)
        out3 , key3= self.encoder3(input3)
        out4 , key4= self.encoder4(input4)
        out5 , key5= self.encoder5(input5)
        out1 = out1.unsqueeze(1)
        out2 = out2.unsqueeze(1)
        out3 = out3.unsqueeze(1)
        out4 = out4.unsqueeze(1)
        out5 = out5.unsqueeze(1)
        
        key1 = key1.unsqueeze(1)
        # print(key1.shape)
        # exit()
        key2 = key2.unsqueeze(1)
        key3 = key3.unsqueeze(1)
        key4 = key4.unsqueeze(1)
        key5 = key5.unsqueeze(1)
        
        self_value = torch.cat(
            [
                out1, out2, out3 , out4, out5
            ],
             1
        )

        self_key = torch.cat(
            [
                key1, key2, key3 , key4, key5
            ],
             1
        )
        #print(self_key.shape)
        self_query = self.Ql(self_key)
        self_key = self.Kl(self_key)
        t = torch.matmul(self_query.transpose(1, 2),self_key) / torch.sqrt(torch.tensor(5))

        att = torch.softmax((torch.matmul(self_query.transpose(1, 2),self_key) / torch.sqrt(torch.tensor(5))), 1)

        concat_att_elec = torch.mul(self_value, att)
        flatten = self.flatten(concat_att_elec)
        hidden1 = F.relu(self.hidden1(flatten))
        hidden2 = F.relu(self.hidden2(hidden1))
        output = self.output_(hidden2)
        
        
        #Representation : 
        # t = hidden2
        
        return output , flatten

class EarlyStopping:
    def __init__(self, pre_dir ,seed,patience=50, verbose=False, delta=0 ):
        self.pre_dir = pre_dir
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.seed = seed

    
    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
        return self.early_stop

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.pre_dir+f'/{self.seed}_torch_checkpoint.pt')
        self.val_loss_min = val_loss
        



import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import os
import sys
import torch
from sklearn.preprocessing import MinMaxScaler
    
class ContinualAMIdataset(object):
    def __init__(self, feature_lst , target_lst , batch_size, train=True):
        self.datasets = datasets_per_task
        # Task 별로 Loader 만들기 
        self.loaders = [
                torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True, drop_last=train, num_workers=1)
                for x in self.datasets ]

    def __getitem__(self, idx):
        return self.loaders[idx]

    def __len__(self):
        return len(self.loaders)

class AMIdataset(Dataset):
    def __init__(self, df ,ENERGY , scaler = None):
        window = 72
        if scaler == None:
            self.scaler = MinMaxScaler()
        else:
            self.scaler = scaler
        feature , y = preprocessing(df,ENERGY)
        if scaler == None:
            feature = pd.DataFrame(self.scaler.fit_transform(feature))
            #print('fit_합니다.')
        else:
            feature = pd.DataFrame(self.scaler.transform(feature))
            #print('transform합니다.\n examples : ')
            #print(feature.head(5))
        
        self.feature, self.y = timeseries_data(feature,y,0,len(feature),window,24)
        
    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        x = torch.tensor(self.feature[idx],dtype = torch.float)
        y = torch.tensor(self.y[idx],dtype = torch.float)
        corr = torch.tensor(corr_features(x),dtype = torch.float)
        return x , y , corr 
    def get_scaler(self):
        return self.scaler


class ATTdataset(Dataset):
    def __init__(self, df ,ENERGY , scaler = None):
        window = 72
        self.df = df 
        self.ENERGY = ENERGY
        
        if scaler == None:
            self.scaler = MinMaxScaler()
        else:
            self.scaler = scaler
        feature , y = preprocessing(df,ENERGY)
        if scaler == None:
            feature = pd.DataFrame(self.scaler.fit_transform(feature))
            #print('fit_합니다.')
        else:
            feature = pd.DataFrame(self.scaler.transform(feature))
            #print('transform합니다.\n examples : ')
            #print(feature.head(5))
        
        self.feature, self.y = timeseries_data(feature,y,0,len(feature),window,24)
        
    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        x = torch.tensor(self.feature[idx],dtype = torch.float)
        y = torch.tensor(self.y[idx],dtype = torch.float)
        return x ,y,self.df[self.ENERGY][idx], self.df['time'][idx]
    def get_scaler(self):
        return self.scaler



class AMI2dataset(Dataset):
    def __init__(self, feature,y):
        self.feature = feature
        self.y = y

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        x = torch.tensor(self.feature[idx],dtype = torch.float)
        y = torch.tensor(self.y[idx],dtype = torch.float)
        return x ,y


class ATT2dataset(Dataset):
    def __init__(self, feature,y,time):
        self.feature = feature
        self.y = y
        self.time =time 
    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        x = torch.tensor(self.feature[idx],dtype = torch.float)
        y = torch.tensor(self.y[idx],dtype = torch.float)
        return x ,y,  self.time[idx]



class clusterDataset(Dataset):
    def __init__(self, feature ,y ):
        
        
        self.feature, self.y = feature , y 
        
    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        x = torch.tensor(self.feature[idx],dtype = torch.float)
        y = torch.tensor(self.y[idx],dtype = torch.float)
        return x , y 
    

def preprocessing(data , ENERGY) :
    if ENERGY == 'elec' :
        feature = data.copy()[['time','elec','water','gas','hotwater','hot']]
    elif ENERGY == 'water' :
        feature = data.copy()[['time','water','elec','gas','hotwater','hot']]
    elif ENERGY == 'gas' :
        feature = data.copy()[['time','gas','elec','water','hotwater','hot']]
    elif ENERGY == 'hotwater' :
        feature = data.copy()[['time','hotwater','elec','water','gas','hot']]
    feature.time = pd.to_datetime(feature.time)
    feature.set_index('time',inplace=True)
    y = data[[ENERGY]]
    return feature, y


def timeseries_data(dataset, target, start_index, end_index, window_size, target_size) :
    data = []
    labels = []

    y_start_index = start_index + window_size
    y_end_index = end_index - target_size

    for i in range(y_start_index, y_end_index) :
        data.append(dataset.iloc[i-window_size:i,:].values)
        labels.append(target.iloc[i:i+target_size,:].values)
    data = np.array(data)
    labels = np.array(labels)
    labels = labels.reshape(-1,target_size)
    return data, labels


def corr_features(features):
    corr_list = np.zeros((4))

    #print(features.shape)
    #exit()
    [[corr_main, corr_sub1, corr_sub2, corr_sub3, corr_sub4]]= pd.DataFrame(np.stack(list(features))).corr().iloc[0:1,:].values
    if np.isnan(corr_sub4) == True :
        corr_sub4 = 0
    corr_list[0], corr_list[1], corr_list[2], corr_list[3] = corr_sub1, corr_sub2, corr_sub3, corr_sub4
    
    return corr_list

