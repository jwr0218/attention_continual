import numpy as np
import pandas as pd
import pickle
import sys
import warnings

sys.path.append('/workspace/continual_meta/ML_instance')
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from model.ETRI_utils import FocalLoss
from model.ETRI_attention import LifeLogNet , HumanDataset, EarlyStopping
from model.utils import getF1Score

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os 
from sklearn.metrics import mean_absolute_error , mean_absolute_percentage_error , mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# from multimodal_cluster_util_total import Start_meta_solution , Train  #,Train_MAML
from cluster_continual.multimodal_cluster_util_continual import Multimodal_utils

pre_dir = '/workspace/data/Etri/date_continual/pickle_10000/'

data_csv = os.listdir(pre_dir)
data_csv.sort()

    

arg1 = 'total'

device = 'cuda'
model  = LifeLogNet(class_num = 7).to(device)

optimizer= optim.Adam(model.parameters())
loss_fn = FocalLoss().to(device)
# loss_fn = nn.CrossEntropyLoss().to(device)
# loss_fn = nn.MSELoss().to(device)

#Model Ready 
#data ready 
#pre_dir = '/workspace/data/Etri/processed_continual/101_day/'

result_df = pd.DataFrame()
test_loader_lst = []
memory_size = 10000
cluster_n = 4
multi_modal_utils = Multimodal_utils(cluster_n = cluster_n,memory_size = memory_size,modal_number = 9)
import random
num = random.randint(0,100)

for task_number , data_dir in enumerate(data_csv):

    user_name = data_dir.split('.')[0]
    
    df = pd.read_pickle(pre_dir+data_dir)
    #df = df[:100]
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle = True, random_state=32)
    
    # Set batch sizef
    n_epochs = 200
    batch_size = 512
    train_dataset = HumanDataset(train_df)
    test_dataset = HumanDataset(test_df)
    
    # Create DataLoaders
    #train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_loader_lst.append(test_loader)

    best_f1 = 0

    train_f1, train_acc, train_auc = [], [], []
    valid_f1, valid_acc, valid_auc = [], [], []
    model_desired_path = f"/workspace/experiments/ETRI/models/"
    early_stopping = EarlyStopping(pre_dir= model_desired_path,seed = str(num), patience=50 )    
    print('='*40 + '\n')
    print(data_dir,'-->\ttraining start','Train data size : ',len(train_dataset))
    early_stopping.val_loss_min = np.Inf
    # model,loss_fn,optimizer,new_dataset,memory_size,modal_number)
    model = multi_modal_utils.Start_meta_solution(model,loss_fn,optimizer,train_dataset,early_stopping)
    
    #feature_lst , label_lst , attention_lst = extract_attention(tmp_model,loss_fn,train_loader,replay_epoch)
    #======================== Evaluation Continual ========================
    
    
    model.load_state_dict(torch.load(f'{model_desired_path}/{num}_torch_checkpoint.pt'))
    model.eval()
    model.load_state_dict(torch.load(f'{model_desired_path}{num}_torch_checkpoint.pt'))
    model.eval()
    
    numpy_path = f'/workspace/experiments/ETRI/'
    if not os.path.exists(numpy_path):
        os.makedirs(numpy_path)


    # att_lst = []
    # for batch_id, (features ,emotionTension ) in enumerate(replay_train_loader):
        
    #     features, label = features.to(device), emotionTension.to(device)
    #     y_pred, att = model(features)
    #     att_lst.extend(att)
        

    # att_lst =np.array([att.detach().cpu().numpy() for att in att_lst])
    # print(att_lst.shape)
    # np_path = numpy_path + f'numpy/{task_number}/'
    # if not os.path.exists(np_path):
    #     os.makedirs(np_path)
    # np.save(np_path+f'{num}_meta_{memory_size}.npy', att_lst)
    
    
    tmp_df = pd.DataFrame()

    for idx, test_loader in enumerate(test_loader_lst):
        valid_output = []
        valid_label = []
        
        for batch_id, (feature ,emotionTension ) in enumerate(test_loader):
        
    
            feature = feature.to(device)
            label = emotionTension.to(device)
            #out = model( e4Acc, e4Bvp, e4Eda, e4Hr, e4Temp, mAcc, mGps, mGyr, mMag)
            out, att = model(feature)
            temp_out = out.detach().cpu().numpy()
            temp_label = label.detach().cpu().numpy()
            valid_output += list(temp_out)
            valid_label += list(temp_label)
            
        # mse = mean_squared_error(valid_label, valid_output)
        # mae = mean_absolute_error(valid_label, valid_output)
        # mape = mean_absolute_percentage_error(valid_label, valid_output)
        
        f1 = getF1Score(np.array(valid_label), np.argmax(valid_output, axis=1))
        # print(f'F1 score: {f1}')
        acc = accuracy_score(np.array(valid_label), np.argmax(valid_output, axis=1))
        # print(f'ACC score: {acc}')

        # df_user = pd.DataFrame({'Task':[idx],f'{task_number}_MAE':mae,f'{task_number}_MSE':mse,f'{task_number}_MAPE' : mape})
        df_user = pd.DataFrame({'Task':[idx],f'{task_number}_F1_score':f1,f'{task_number}_accuracy':acc})
        tmp_df = pd.concat([tmp_df,df_user],axis = 0 )
    tmp_df = tmp_df.set_index('Task')
    result_df = pd.concat([result_df,tmp_df],axis = 1)
    print(result_df)
    path = f'/workspace/experiments/ETRI/ER/'
    if not os.path.exists(path):
        os.makedirs(path)
    path_1 = path+arg1
    if not os.path.exists(path_1):
        os.makedirs(path_1)
    result_df.to_csv(f'{path_1}/{num}_meta_{memory_size}.csv')

