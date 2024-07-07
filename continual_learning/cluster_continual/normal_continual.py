from torch.utils.data import Dataset, DataLoader , TensorDataset
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os 
from sklearn.metrics import mean_absolute_error , mean_absolute_percentage_error , mean_squared_error
from sklearn.model_selection import train_test_split
import random 
#Cluster ===============================
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from torch.utils.data import ConcatDataset , Subset
import copy 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from cosineKmeans import CosineKMeans , EuclideanKMeans




from .grad_utils import get_gradient,update_gradient,project_gradient_qp


device = 'cuda'

class Multimodal_utils():
    
    def __init__(self,cluster_n = 4,memory_size = 10000,modal_number = 9,batch_size = 512 , n_epochs = 200 )  -> None:
        self.clustered_data = []
        self.replay_feature = []
        self.replay_label = []
        self.vali_feature = []
        self.vali_label = []
        
        self.n_cluster = cluster_n
        self.memory_size = memory_size
        self.modal_number = modal_number
        
        self.task_num = 1
        
        self.batch_size = batch_size 
        # self.n_epochs = 10
        self.n_epochs = n_epochs
        
        
    
    
    
    def Train_continual_ER(self,model, optimizer, loss_fn, new_loader,replay_loader, vali_loader, early_stopping):
        
        for epoch in range(self.n_epochs):
            model.train()
            loss_preds = []

            if replay_loader is not None :
                total_loaders = []
                total_loaders.append(replay_loader)
                total_loaders.append(new_loader)
                for batch in zip(*total_loaders):
                    # print([len(b[0]) for b in batch])
                    features = torch.cat([b[0] for b in batch], dim=0)  # 모든 features 합치기
                    labels = torch.cat([b[1] for b in batch], dim=0)  # 모든 labels 합치기
                    loss_total = 0
                    
                    if features.shape[0] <= 1 :
                        # print('사이즈가 작아 break 합니다. ')
                        continue

                    features, labels = features.to(device,dtype=torch.float32), labels.to(device,dtype=torch.float32)
                    # print(features)
                    y_pred, att = model(features)
                    optimizer.zero_grad()
                    
                    loss_pred = loss_fn(y_pred, labels)
                    loss_preds.append(loss_pred.item())
                    loss_pred.backward()
                    
                    optimizer.step()
            else:
                for features, labels  in new_loader:


                    features, labels = features.to(device), labels.to(device)
                    y_pred, att = model(features)
                    optimizer.zero_grad()
                    
                    loss_pred = loss_fn(y_pred, labels)
                    loss_preds.append(loss_pred.item())
                    loss_pred.backward()
                    optimizer.step()

            

            # 에포크의 평균 손실 계산
            avg_loss = sum(loss_preds) / len(loss_preds)

            # 검증 부분
            # if vali_loader is not None:    
            model.eval()
            val_loss = 0
            sha= 0 
            with torch.no_grad():
                for features, label in vali_loader:
                    features, label = features.to(device), label.to(device)
                    y_pred, _ = model(features)
                    val_loss += loss_fn(y_pred, label).item()
                    
                    sha += features.shape[0]
            val_loss /= sha

            #val_loss /= len(vali_loader)

            # 조기 종료 체크
            if early_stopping(val_loss, model):
                print(f'Early stopping at epoch {epoch}')
                break
            if epoch % 50 == 0 :
                print(val_loss)
                pass
        return model , avg_loss

    def Train_continual_AGEM(self,model, optimizer, loss_fn, new_loader,replay_loader, vali_loader, early_stopping):
        
        for epoch in range(self.n_epochs):
            model.train()
            loss_preds = []

            if replay_loader is not None :
                model.zero_grad()
                for re_batch in replay_loader:
                    past_features = re_batch[0]  # 모든 features 합치기
                    past_labels = re_batch[1]  # 모든 labels 합치기
                
                    past_features, past_labels = past_features.to(device,dtype=torch.float32), past_labels.to(device,dtype=torch.float32)
                    
                    if len(past_features)<=1 :
                        continue
                    y_pred, att = model(past_features)
                    loss_pred = loss_fn(y_pred, past_labels)
                    loss_pred.backward()
                
                past_gradient = get_gradient(model)
                    
                for new_batch in new_loader:

                    
                    new_features = new_batch[0]  # 모든 features 합치기
                    new_labels = new_batch[1]  # 모든 labels 합치기
                    loss_total = 0

                    if len(new_features)<=1 :
                        continue
                    # New
                    new_features, new_labels = new_features.to(device,dtype=torch.float32), new_labels.to(device,dtype=torch.float32)
                    
                    model.zero_grad()
                    y_pred, _ = model(new_features)
                    loss = loss_fn(y_pred, new_labels)
                    loss.backward()
                    cur_gradient = get_gradient(model)
                    dotp = torch.dot(cur_gradient, past_gradient)
                    if dotp < 0:
                        ref_mag = torch.dot(past_gradient, past_gradient)
                        if ref_mag > 0:
                            new_grad = cur_gradient - ((dotp / ref_mag) * past_gradient)
                            update_gradient(model, new_grad)
                        else:
                            # Handle zero division or invalid ref_mag scenario
                            update_gradient(model, cur_gradient)
                    optimizer.step()
                    
            else:
                for features, labels  in new_loader:


                    features, labels = features.to(device), labels.to(device)
                    y_pred, att = model(features)
                    optimizer.zero_grad()
                    
                    loss_pred = loss_fn(y_pred, labels)
                    loss_preds.append(loss_pred.item())
                    loss_pred.backward()
                    optimizer.step()

            

            # 에포크의 평균 손실 계산


            # 검증 부분
            # if vali_loader is not None:    
            model.eval()
            val_loss = 0
            sha= 0 
            with torch.no_grad():
                for features, label in vali_loader:
                    features, label = features.to(device), label.to(device)
                    y_pred, _ = model(features)
                    val_loss += loss_fn(y_pred, label).item()
                    
                    sha += features.shape[0]
            val_loss /= sha

            #val_loss /= len(vali_loader)

            # 조기 종료 체크
            if early_stopping(val_loss, model):
                print(f'Early stopping at epoch {epoch}')
                break
            if epoch % 50 == 0 :
                print(val_loss)
                pass
        return model , 0


    def meta_Train(self,model,optimizer,loss_fn,dataloader):
        feature_lst = []
        label_lst = []
        meta_lst = []
        att_lst = []
        loss_lst = []
        model.eval()
        for batch, (features, label) in enumerate(dataloader):
            model.zero_grad()
            optimizer.zero_grad()
            
            feature_lst.extend(features)
            label_lst.extend(label)
            

            features, label = features.to(device), label.to(device) 


            y_pred ,att = model(features)
            # print(att.shape)
            # exit()
            att_lst.extend(att)
            
            #print(list(tmp_model.parameters())[0].grad[0])
            loss_pred = loss_fn(y_pred , label)


            loss_pred.backward()

            loss_lst.append(loss_pred.item())
            # meta_lst.append(grad)

        feature_lst = np.array([feature.detach().cpu().numpy() for feature in feature_lst])
        label_lst = np.array([label.detach().cpu().numpy() for label in label_lst])

        att_lst =np.array([att.detach().cpu().numpy() for att in att_lst])
        att_lst = att_lst.reshape(-1,self.modal_number)
        
        return feature_lst,label_lst, att_lst


    def cluster_based_replay_loader(self,new_dataset,replay):

        new =  DataLoader(new_dataset, batch_size=self.batch_size, shuffle=False)
            
    
        
        #================= Validation ===================
        K = self.memory_size // (self.task_num + 1)
        
        
        vali_sizes = [int(K*0.1) for i in range(len(self.vali_feature))]
            
        for idx, (features, labels) in enumerate(zip(self.vali_feature,self.vali_label)):
            
            self.vali_feature[idx] = features[:vali_sizes[idx]]
            self.vali_label[idx] = labels[:vali_sizes[idx]]
        
        vali_size = int(K*0.1)
        
        indices = list(range(len(new_dataset)))
        vali_indices = indices[-vali_size:] 
        val_dataset = Subset(new_dataset, vali_indices)
        vali_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        
        new_vali_features, new_vali_labels = [], []
        for features, labels in vali_loader:
            new_vali_features.extend(features.numpy())
            new_vali_labels.extend(labels.numpy())
        
        self.vali_feature.append(new_vali_features)
        self.vali_label.append(new_vali_labels)
        
        
        tmp_feature = []
        tmp_label = []
        for features, labels in zip(self.vali_feature,self.vali_label):

            tmp_feature.extend(features)
            tmp_label.extend(labels) 
        vali_dataset = TensorDataset(torch.tensor(tmp_feature), torch.tensor(tmp_label))

        vali_dataloader = DataLoader(vali_dataset, batch_size=self.batch_size, shuffle=True)
        
        
        
        return new, replay ,vali_dataloader





    def allocate_memory(self, memory_size, replay, new, n_cluster):
        K = memory_size // (self.task_num)
        replay_size = K * (self.task_num-1)
        print('replay_size : ',replay_size)
        self.task_num += 1
        replay_loader = replay
        new_loader = new
        self.replay_feature = []
        self.replay_label = []
        # Replay 처리 부분
        if replay_loader is not None:
            replay_feature = []
            replay_label = []

            for feature, label in replay_loader:
                replay_feature.extend(feature)  # .numpy()가 필요한 경우만 사용
                replay_label.extend(label)      # .numpy()가 필요한 경우만 사용

            replay_feature = np.array([feature.numpy() for feature in replay_feature])
            replay_label = np.array([label.numpy() for label in replay_label])
            # exit()
            indice = list(range(len(replay_feature)))
            random.shuffle(indice)
            # exit()
            self.replay_feature.extend(replay_feature[indice[:replay_size]]) # numpy 배열을 리스트로 변환
            self.replay_label.extend(replay_label[indice[:replay_size]])     # numpy 배열을 리스트로 변환
        # New 처리 부분
        new_feature = []
        new_label = []

        for feature, label in new_loader:
            new_feature.extend(feature)  # .numpy()가 필요한 경우만 사용
            new_label.extend(label)      # .numpy()가 필요한 경우만 사용
        new_feature = np.array([feature.numpy() for feature in new_feature])
        new_label = np.array([label.numpy() for label in new_label])
        print('NEW_Feature shape',new_feature.shape)
        
        indice = list(range(len(new_feature)))
        random.shuffle(indice)

        self.replay_feature.extend(new_feature[indice[:K]])  # numpy 배열을 리스트로 변환
        self.replay_label.extend(new_label[indice[:K]])      # numpy 배열을 리스트로 변환
        # print('NEW : ',len(self.replay_feature))

        
        return




    clustered_data = []
    replay_feature = []
    replay_label = []
    vali_feature = []
    vali_label = []

    def make_replay_loader_before(self):
        # 데이터 선택 및 셔플링
        # print(len(self.replay_feature))
        # print(len(self.replay_label))
        
        replay_train_dataset = TensorDataset(torch.tensor(np.array(self.replay_feature)), torch.tensor(np.array(self.replay_label)))


        tmp_feature = []
        tmp_label = []
        
        for features, labels in zip(self.vali_feature,self.vali_label):

            tmp_feature.extend(features)
            tmp_label.extend(labels) 
        replay_vali_dataset = TensorDataset(torch.tensor(tmp_feature), torch.tensor(tmp_label))
        
        return replay_train_dataset, replay_vali_dataset




    def Start_meta_solution(self,model,loss_fn,optimizer,new_dataset,early_stopping,typ):
        
        
        
        if len(self.replay_feature)>0:
            
            replay_train_dataset, replay_vali_dataset = self.make_replay_loader_before()
            replay  = DataLoader(replay_train_dataset, batch_size=self.batch_size, shuffle=True)
            
        else:
            replay = None
       
        # Train, Replay, New 
        
        new_train_loader , replay_train_loader ,replay_vali_loader= self.cluster_based_replay_loader(new_dataset,replay)
        
        # Train, Replay, New
        
        
        
        if typ == 'AGEM':
            
            model, loss = self.Train_continual_AGEM(model,optimizer,loss_fn,new_train_loader,replay_train_loader,replay_vali_loader,early_stopping)
        elif typ == 'ER':
            
            model, loss = self.Train_continual_ER(model,optimizer,loss_fn,new_train_loader,replay_train_loader,replay_vali_loader,early_stopping)
        else:
            print("Wrong Type")
            exit()
        # Model Update 
        
        # Model Update 
        
        
        
        
        
        self.allocate_memory(self.memory_size,replay_train_loader, new_train_loader,self.n_cluster)
            
        # replay_train_loader ,replay_vali_loader= self.cluster_based_replay_loader()
        
        print('Train : 현재 Memory Data : ',len(self.replay_feature))
        print('Vali : 현재 Memory Data : ',sum([len(feature) for feature in self.vali_feature]))
        
        
        return model






#=================#=================#=================#=================#=================#=================#=================

