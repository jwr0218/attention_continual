import re
from model.AMI_attention import AttentionModel , EarlyStopping, AMIdataset , preprocessing , timeseries_data ,ATTdataset, AMI2dataset  ,ATT2dataset
#from multimodal.attention_model_self_att import AttentionModel , EarlyStopping
from torch.utils.data import Dataset, DataLoader , TensorDataset
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os 
from sklearn.metrics import mean_absolute_error , mean_absolute_percentage_error , mean_squared_error
from sklearn.model_selection import train_test_split
import copy
import random 
import pickle 
import sys 

# Utils 
#from AMI_multimodal_util import Start_meta_solution , Train  
from cluster_continual.multimodal_cluster_util_continual import Multimodal_utils

# Utils 


energy_type = 'elec'

device = 'cuda'


model  = AttentionModel()
model = model.to(device)
optimizer_pred = optim.Adam(model.parameters())
loss_fn = nn.MSELoss().to(device)

#Model Ready 
#data ready 

if len(sys.argv) > 1:
    arg1 = sys.argv[1]

    if arg1 == 'summer':
        pre_dir = '/workspace/data/AMI/summer_np/'
    elif arg1 =='winter':
        pre_dir = '/workspace/data/AMI/winter_np/'
    elif arg1 =='all':
        pre_dir = '/workspace/data/AMI/all_np/'
    else:
        print('맞는 데이터가 없습니다. 다시 시작해주세요.')
        exit()
    print(arg1 ,' : ', pre_dir)
    



# 파일명에서 날짜 부분을 추출하고 정렬하는 함수
def sort_files_by_date(file_list):
    def extract_date(filename):
        # 파일명에서 날짜 부분을 추출
        matches = re.findall(r'\d{4}_\d{1,2}', filename)
        # 연도와 월로 분리
        return [tuple(map(int, date.split('_'))) for date in matches]

    # 추출한 날짜를 기준으로 정렬
    file_list.sort(key=lambda x: extract_date(x))

# 파일 정렬
data_csv = os.listdir(pre_dir)
sort_files_by_date(data_csv)


d = []

for n in data_csv:
    d.append(n.split('-')[0])

csv_name = '-'.join(d)




n_epochs = 200
batch_size = 128
#{'Now' : [task_number] ,'Task':[idx],'tmpLoss':[tmp_loss],'cluster':[n_cluster],'MAPE' : [error_mape], 'MAE' : [error_mae], 'RMSE' : [error_rmse]}
#result_df = pd.DataFrame({'Now' : [] ,'Task':[],'tmpLoss':[],'cluster':[],'cluster Memory Size':[],'MAPE': [], 'MAE': [], 'RMSE': []})
result_df = pd.DataFrame({})
memory_size = 2000 # Memory Size 
loss_df = pd.DataFrame()

test_loader_lst = []

multi_modal_utils = Multimodal_utils(cluster_n = 4,memory_size = memory_size,modal_number = 120)




num = random.randint(0,100)
for task_number , data_dir in enumerate(data_csv):
    model.train()
    season = data_dir.split('.')[0]
    file_name = pre_dir + data_dir
    with open(file_name, 'rb') as file:
        loaded_df = pickle.load(file)
    
    X = loaded_df['feature']
    y = loaded_df['y']
    time = loaded_df['time']
    
    idxes = np.arange(len(X)) 
    # np.random.shuffle(idxes)  # 데이터를 셔플 , 시계열이라 셔플할 필요 없음. 
    

    split_ratio = 0.8
    split_index = int(len(idxes) * split_ratio)
    
    train_feature = X[idxes[:split_index]]
    train_y = y[idxes[:split_index]]
    train_time = time[idxes[:split_index]]
    
    test_feature = X[idxes[split_index:]]
    test_y = y[idxes[split_index:]]
    test_time = time[idxes[split_index:]]
    
    # Train Dataset 
    train_dataset = AMI2dataset(train_feature,train_y)
    #meta_dataset = ATT2dataset(train_feature,train_y,train_time)
    
    #train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
    #meta_loader = DataLoader(meta_dataset, batch_size=batch_size, shuffle=True)
    # Train Dataset 
    
    #Test Dataset
    test_dataset = AMI2dataset(test_feature,test_y)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_loader_lst.append(test_loader)
    
    optimizer_pred = optim.Adam(model.parameters())
    early_stopping = EarlyStopping(pre_dir= f"/workspace/experiments/AMI/models/",seed = str(num), patience=50 )    
    print('='*40 + '\n')
    print(data_dir,'-->\ttraining start','Train data size : ',len(train_dataset))
    early_stopping.val_loss_min = np.Inf
    # model,loss_fn,optimizer,new_dataset,memory_size,modal_number)
    model = multi_modal_utils.Start_meta_solution(model,loss_fn,optimizer_pred,train_dataset,early_stopping)
    
    #feature_lst , label_lst , attention_lst = extract_attention(tmp_model,loss_fn,train_loader,replay_epoch)
    #======================== Evaluation Continual ========================
    
    
    model.load_state_dict(torch.load(f'/workspace/experiments/AMI/models/{num}_torch_checkpoint.pt'))
    model.eval()


    with torch.no_grad():
        r_1 = pd.DataFrame()
        for idx, test_loader in enumerate(test_loader_lst):
            predict_df = []
            real_df = []
            loss_test = [ ] 
            for batch, (features, label) in enumerate(test_loader):
                features= features.to(device)
                label = label.to(device)
                y_pred , t = model(features)
                loss_=loss_fn(label, y_pred)
                predict_df.append(y_pred.detach())
                real_df.append(label.detach())
                loss_test.append(loss_.item()/features.shape[0])
            real_new = [a.cpu().numpy() for b in real_df for a in b]
            predict_new = [a.cpu().numpy().round(1) for b in predict_df for a in b] 
            error_mape = mean_absolute_percentage_error(real_new, predict_new)
            error_mae = mean_absolute_error(real_new, predict_new)
            error_rmse = mean_squared_error(real_new, predict_new)**(0.5)
            df_tmp = pd.DataFrame({'Task':[data_csv[idx].split('.')[0]],f'{task_number}_MAE':error_mae,f'{task_number}_RSME':error_rmse,f'{task_number}_MAPE':error_mape})
            r_1 = pd.concat([r_1,df_tmp],axis=0)
        r_1 = r_1.set_index('Task')

        result_df = pd.concat([result_df,r_1],axis = 1 )
        print(result_df)


    
    desired_path = f"/workspace/experiments/{arg1}/"

    # 해당 경로가 없다면 디렉토리 생성
    if not os.path.exists(desired_path):
        os.makedirs(desired_path)
    
    file_name = f'{num}_result_{memory_size}.csv'
    full_path = os.path.join(desired_path, file_name)

    result_df.to_csv(full_path)
    # result_df.to_csv(desired_path + f'{csv_name}.csv')

