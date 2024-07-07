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
from torch.utils.data import ConcatDataset
import copy 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from cosineKmeans import CosineKMeans , EuclideanKMeans


device = 'cuda'

class Multimodal_utils():
    
    def __init__(self,cluster_n = 4,memory_size = 10000,modal_number = 9,batch_size = 512 , n_epochs = 200) -> None:
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
        self.n_epochs = n_epochs
        

    def Train(self,model, optimizer, loss_fn, train_loaders, vali_loader, early_stopping):
        
        model.train()
        loss_preds = []

        for batch in zip(*train_loaders):
            # print([len(b[0]) for b in batch])
            features = torch.cat([b[0] for b in batch], dim=0)  # 모든 features 합치기
            labels = torch.cat([b[1] for b in batch], dim=0)  # 모든 labels 합치

            features, labels = features.to(device), labels.to(device)
            y_pred, att = model(features)
            optimizer.zero_grad()
            # print(y_pred.shape)
            # print(labels.shape)
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

          
        return model , val_loss
    
    
    
    def Train2(self,model, optimizer, loss_fn, train_loaders, vali_loader, epochs, early_stopping):
        
        for epoch in range(epochs):
            model.train()
            loss_preds = []

            for batch in zip(*train_loaders):
                # print([len(b[0]) for b in batch])
                features = torch.cat([b[0] for b in batch], dim=0)  # 모든 features 합치기
                labels = torch.cat([b[1] for b in batch], dim=0)  # 모든 labels 합치기
                loss_total = 0
                for b in batch:
                    f = b[0].to(device)
                    l = b[1].to(device)
                    
                    y_pred, att = model(f)
                    optimizer.zero_grad()
                    
                    loss_pred = loss_fn(y_pred, l)
                    loss_total += loss_pred
                
                # print(loss_total)
                loss_preds.append(loss_pred.item())
                loss_total.backward()
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
            if epoch % 10 == 0 :
                print(val_loss)
                pass
        return model , avg_loss




    def meta_Train(self,model,loss_fn,dataloader):
        feature_lst = []
        label_lst = []
        meta_lst = []
        att_lst = []
        loss_lst = []
        model.eval()
        for batch, (features, label) in enumerate(dataloader):
            model.zero_grad()
            
            
            feature_lst.extend(features)
            label_lst.extend(label)
            

            features, label = features.to(device), label.to(device) 


            y_pred ,att = model(features)
            # print(att.shape)
            # exit()
            att_lst.extend(att)
            
            #print(list(tmp_model.parameters())[0].grad[0])
            # loss_pred = loss_fn(y_pred , label)


            # loss_pred.backward()

            # loss_lst.append(loss_pred.item())
            # meta_lst.append(grad)

        feature_lst = np.array([feature.detach().cpu().numpy() for feature in feature_lst])
        label_lst = np.array([label.detach().cpu().numpy() for label in label_lst])

        att_lst =np.array([att.detach().cpu().numpy() for att in att_lst])
        att_lst = att_lst.reshape(-1,self.modal_number)
        
        return feature_lst,label_lst, att_lst


    def replay_loader(self):
        tmp_feature = []
        tmp_label = []
        for features, labels in zip(self.replay_feature,self.replay_label):
            # print(features.shape)
            tmp_feature.append(features)
            tmp_label.append(labels) 
        # Assuming tmp_feature and tmp_label are lists of numpy arrays or lists
        tmp_feature_tensor = torch.stack([f.clone().detach() for f in tmp_feature])
        tmp_label_tensor = torch.stack([l.clone().detach() for l in tmp_label])
        # Create the TensorDataset
        replay_dataset = TensorDataset(tmp_feature_tensor, tmp_label_tensor)

        replay_dataloader = DataLoader(replay_dataset, batch_size=self.batch_size, shuffle=True)
        replay_loaders = []
        replay_loaders.append(replay_dataloader)
        
        return replay_loaders



    def get_validation(self,new):
        K = self.memory_size // (self.task_num )
        new_feature, new_label = new
        
        
        vali_sizes = [int(K*0.1) for i in range(len(self.vali_feature))]
            
        for idx, (features, labels) in enumerate(zip(self.vali_feature,self.vali_label)):
            
            self.vali_feature[idx] = features[:vali_sizes[idx]]
            self.vali_label[idx] = labels[:vali_sizes[idx]]
        
        vali_size = int(K*0.1)
        
        indices = list(range(len(new_label)))
        selected_features = new_feature[indices[-vali_size:]]
        selected_labels = new_label[indices[-vali_size:]]
        self.vali_feature.append(selected_features)
        self.vali_label.append(selected_labels)
        
        
        tmp_feature = []
        tmp_label = []
        for features, labels in zip(self.vali_feature,self.vali_label):

            tmp_feature.extend(features)
            tmp_label.extend(labels) 
        
        # tmp_feature_tensor = torch.stack([torch.tensor(f) for f in tmp_feature])
        # tmp_label_tensor = torch.stack([torch.tensor(l) for l in tmp_label])
        
        tmp_feature_tensor = torch.stack([f.clone().detach() for f in tmp_feature])
        tmp_label_tensor = torch.stack([l.clone().detach() for l in tmp_label])
        
        vali_dataset = replay_dataset = TensorDataset(tmp_feature_tensor, tmp_label_tensor)
        
        vali_dataloader = DataLoader(vali_dataset, batch_size=self.batch_size, shuffle=True)
        return vali_dataloader
    



    def cluster_based_replay_loader(self,replay,new):

        
        replay_clusters, replay_memory_feature, replay_memory_label = replay
        new_clusters, new_memory_feature, new_memory_label = new
        
        
        
        replay_loaders = []
        new_loaders = []
        # replay_feature_tmp = replay_feature[0]
        # replay_label_tmp = replay_label[0]
        
        
        
        
        if replay_clusters is not None:
            replay_clustered_indices = []
            for i in range(self.n_cluster):
                indices = np.where(replay_clusters == i)[0]
                np.random.shuffle(indices) # 섞여도 됨. 
                replay_clustered_indices.append(indices)
        
            replay_total_size = sum([len(f) for f in self.replay_label])
        
        
            for idx in range(len(replay_clustered_indices)):
                cluster_indices = replay_clustered_indices[idx]
                features = replay_memory_feature[cluster_indices]
                labels = replay_memory_label[cluster_indices]
                # print('====='*20)
                # print(len(features))
                # print(len(labels))
                # print(int(batch_size*(len(features)/total_size)))
                replay_dataset = TensorDataset(torch.tensor(features), torch.tensor(labels))
                cluster_batch_size = self.batch_size*(len(features)/replay_total_size)
                if int(cluster_batch_size) == 0  :
                    if len(features) == 0 :
                        cluster_batch_size  = 3
                        continue
                    cluster_batch_size  = len(features)
                replay_dataloader = DataLoader(replay_dataset, batch_size=int(cluster_batch_size), shuffle=True)
                replay_loaders.append(replay_dataloader)

        
            # print('Feature size : ',[len(f) for f in self.replay_feature])
            new_clustered_indices = []
            for i in range(self.n_cluster):
                indices = np.where(new_clusters == i)[0]
                np.random.shuffle(indices) # 섞여도 됨 .
                new_clustered_indices.append(indices)
            new_total_size = sum([len(f) for f in new_clustered_indices])
            
            
            for idx in range(len(new_clustered_indices)):
                cluster_indices = new_clustered_indices[idx]
                features = new_memory_feature[cluster_indices]
                labels = new_memory_label[cluster_indices]
                # print('====='*20)
                # print(len(features))
                # print(len(labels))
                # print(int(batch_size*(len(features)/total_size)))
                new_dataset = TensorDataset(torch.tensor(features), torch.tensor(labels))
                cluster_batch_size = self.batch_size*(len(features)/new_total_size)
                if int(cluster_batch_size) == 0  :
                    if len(features) == 0 :
                        continue
                    cluster_batch_size  = len(features)
                new_dataloader = DataLoader(new_dataset, batch_size=int(cluster_batch_size), shuffle=True)
                new_loaders.append(new_dataloader)

        else:
            # print('Feature size : ',[len(f) for f in self.replay_feature])
            new_clustered_indices = []
            tmp_feature = []
            tmp_label = []
            
            for i in range(self.n_cluster):
                indices = np.where(new_clusters == i)[0]
                np.random.shuffle(indices) # 섞여도 됨 .
                new_clustered_indices.append(indices)
            new_total_size = sum([len(f) for f in new_clustered_indices])
            
            
            for idx in range(len(new_clustered_indices)):
                cluster_indices = new_clustered_indices[idx]
                features = new_memory_feature[cluster_indices]
                labels = new_memory_label[cluster_indices]
                # print('====='*20)
                # print(len(features))
                # print(len(labels))
                # print(int(batch_size*(len(features)/total_size)))
                tmp_feature.extend(features)
                tmp_label.extend(labels)
            new_dataset = TensorDataset(torch.tensor(tmp_feature), torch.tensor(tmp_label))
            new_dataloader = DataLoader(new_dataset, batch_size=self.batch_size, shuffle=True)
            new_loaders.append(new_dataloader)
        
        
        #================= Validation ===================

        
        
        
        return new_loaders, replay_loaders 




    def allocate_memory(self,new_dataset):

        K = self.memory_size // (self.task_num + 1)
        
        
        new_feature , new_label = new_dataset
        
        
        if len(self.replay_feature) > 0 :
            replay_indices = list(range(len(self.replay_feature)))
            np.random.shuffle(replay_indices) # 섞여도 됨. 
            replay_resize = self.memory_size-K
        # Replay Dataset:
        # New Dataset
        
        new_clustered_indices = []
        new_indices = list(range(len(new_feature)))
        np.random.shuffle(new_indices) # 섞여도 됨 .

        new_resize = K 
        
        tmp_feature = []
        tmp_label = []

        # Replay 데이터셋 처리
        if len(self.replay_feature) > 0 :
            replay_selected_indices = replay_indices[:replay_resize]
            new_selected_indices = new_indices[:new_resize]
            # 선택된 샘플 추가
            t_feature = []
            t_label = []
            # print(len(self.replay_label))
            # print(len(replay_selected_indices))
            for index in replay_selected_indices:
                t_feature.append(self.replay_feature[index])
                t_label.append(self.replay_label[index])
            
            t_feature.extend(new_feature[new_selected_indices])
            
            
            t_label.extend(new_label[new_selected_indices])
            
            tmp_feature.extend(t_feature)
            tmp_label.extend(t_label)
            
            # print('replay clusted indices size ',replay_cluster_memory)
        else:
            new_selected_indices = new_indices[:new_resize]
            tmp_feature.extend(new_feature[new_selected_indices])
            tmp_label.extend(new_label[new_selected_indices])

        self.replay_feature = tmp_feature
        self.replay_label = tmp_label

        
        
        print('replay feature length : ',len(self.replay_feature))
        
        
        
        
        vali_sizes = [int(K*0.1) for i in range(len(self.vali_feature))]
            
        for idx, (features, labels) in enumerate(zip(self.vali_feature,self.vali_label)):
            
            self.vali_feature[idx] = features[:vali_sizes[idx]]
            self.vali_label[idx] = labels[:vali_sizes[idx]]
            
        vali_size = int(K*0.1)
        
        selected_features = new_feature[new_indices[-vali_size:]]
        selected_labels = new_label[new_indices[-vali_size:]]
        self.vali_feature.append(selected_features)
        self.vali_label.append(selected_labels)
        self.task_num+=1
        return 




    clustered_data = []
    replay_feature = []
    replay_label = []
    vali_feature = []
    vali_label = []

    def make_replay_loader_before(self):
        # 데이터 선택 및 셔플링

        tmp_feature = []
        tmp_label = []
        
        for features, labels in zip(self.replay_feature,self.replay_label):

            tmp_feature.extend(features)
            tmp_label.extend(labels) 
        replay_train_dataset = TensorDataset(torch.tensor(tmp_feature), torch.tensor(tmp_label))


        tmp_feature = []
        tmp_label = []
        
        for features, labels in zip(self.vali_feature,self.vali_label):

            tmp_feature.extend(features)
            tmp_label.extend(labels) 
        replay_vali_dataset = TensorDataset(torch.tensor(tmp_feature), torch.tensor(tmp_label))
        
        return replay_train_dataset, replay_vali_dataset



    def Start_meta_solution(self,model,loss_fn,optimizer,new_dataset,early_stopping,typ):
        
        
        # print('new\t',new_train_loader)
        # print('replay\t',replay_train_loader)
        # Train, Replay, New
        
        new_loader = DataLoader(new_dataset,batch_size = self.batch_size)
        new_feature, new_label = [] , []
        
        for feature, label in new_loader:
            new_feature.extend(feature)
            new_label.extend(label)
        
        new_dataset = (feature, label)
        
        
            
        self.allocate_memory(new_dataset)
        
        
        replay_loaders = self.replay_loader()
        
        replay_vali_loader = self.get_validation(new_dataset)
        # print(replay_vali_loader)
        for ep in range(1,self.n_epochs):
            
            
            if typ == 'NORMAL':
                # print('NORMAL type! ')
                model, loss = self.Train(model, optimizer, loss_fn, replay_loaders, replay_vali_loader, early_stopping)
                # print(loss)
            else:
                print('????? wrong type! ')
                exit()

            if ep % 25 == 0 :
                # print('new\t',new_train_loader)
                # print('replay\t',replay_loaders)
                print(loss)
                pass
            
            if early_stopping(loss, model):
                print(f'Early stopping at epoch {ep}')
                break
        # Model Update 
        
        # Model Update 
        
       
        print('Train : 현재 Memory Data : ',len(self.replay_feature))
        print('Vali : 현재 Memory Data : ',sum([len(feature) for feature in self.vali_feature]))
        
        
        return model






#=================#=================#=================#=================#=================#=================#=================

