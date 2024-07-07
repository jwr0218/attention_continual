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

from .grad_utils import get_gradient,update_gradient,project_gradient_qp


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
    
    def Train_continual_ER(self,model, optimizer, loss_fn, new_loaders,replay_loaders, vali_loader, early_stopping):
        
        model.train()
        loss_preds = []

        if len(replay_loaders)> 0 :
            total_loaders = []
            total_loaders.extend(replay_loaders)
            total_loaders.extend(new_loaders)
            for batch in zip(*total_loaders):
                # print([len(b[0]) for b in batch])
                features = torch.cat([b[0] for b in batch], dim=0)  # 모든 features 합치기
                labels = torch.cat([b[1] for b in batch], dim=0)  # 모든 labels 합치기
                loss_total = 0
                
                if features.shape[0] <= 1 :
                    # print('사이즈가 작아 break 합니다. ')
                    continue

                features, labels = features.to(device), labels.to(device)
                y_pred, att = model(features)
                optimizer.zero_grad()
                
                loss_pred = loss_fn(y_pred, labels)
                loss_preds.append(loss_pred.item())
                loss_pred.backward()
                
                optimizer.step()
        else:
            for batch in zip(*new_loaders):
                # print([len(b[0]) for b in batch])
                features = torch.cat([b[0] for b in batch], dim=0)  # 모든 features 합치기
                labels = torch.cat([b[1] for b in batch], dim=0)  # 모든 labels 합치기
                loss_total = 0
                if features.shape[0] <= 1 :
                    # print('사이즈가 작아 break 합니다. ')
                    continue

                features, labels = features.to(device), labels.to(device)
                y_pred, att = model(features)
                optimizer.zero_grad()
                
                loss_pred = loss_fn(y_pred, labels)
                loss_preds.append(loss_pred.item())
                loss_pred.backward()
                optimizer.step()

            
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
        

        return model , val_loss 


    def Train_continual_AGEM(self,model, optimizer, loss_fn, new_loaders,replay_loaders, vali_loader, early_stopping):
        
        
        model.train()
        
        
        
        loss_preds = []
        if len(replay_loaders)> 0 :
            model.zero_grad()
            
            loss_preds = []
            grad_lst = []
            # print(train_loaders)
            replay_loaders_iter = [iter(loader) for loader in replay_loaders]
            times = [len(loader) for loader in replay_loaders]
            times = max(times)
            t = 0 
            
            while True:
                batch = []
                t +=1
                if t > times:
                    break
                
                for idx, loader_iter in enumerate(replay_loaders_iter):
                    try:
                        batch.append(next(loader_iter))
                    except StopIteration:
                        # 데이터 로더가 끝난 경우 다시 시작
                        loader_iter = iter(replay_loaders[idx])
                        batch.append(next(loader_iter))
                # 각 데이터 로더에서 데이터 추출

                replay_features = torch.cat([b[0] for b in batch], dim=0)
                replay_labels = torch.cat([b[1] for b in batch], dim=0)
                replay_features, replay_labels = replay_features.to(device), replay_labels.to(device)
                
                y_pred, _ = model(replay_features)
                # loss = loss_fn(y_pred, labels)                
                loss = loss_fn(y_pred, replay_labels)
                loss.backward()
                
            past_gradient = get_gradient(model)    
            model.zero_grad()
            
            all_loaders = []
            
            # all_loaders.extend(replay_loaders)
            print(new_loaders)
            all_loaders.extend(new_loaders)
            
            
            
            loaders_iter = [iter(loader) for loader in all_loaders]

            # print(loaders_iter)
            
            
            times = [len(loader) for loader in loaders_iter]
            times = max(times)
            t = 0 
            # print(times,t)
            while True:
                batch = []
                t+=1
                if t > times:
                    break
                
                for idx, loader_iter in enumerate(loaders_iter):
                    try:
                        batch.append(next(loader_iter))
                    except StopIteration:
                        # 데이터 로더가 끝난 경우 다시 시작
                        loader_iter = iter(all_loaders[idx])
                        batch.append(next(loader_iter))
                # 각 데이터 로더에서 데이터 추출
                
                # 모든 new batches에서 데이터를 추출하고 합칩니다.
                features = torch.cat([b[0] for b in batch], dim=0)
                labels = torch.cat([b[1] for b in batch], dim=0)

                # GPU로 데이터 이동
                
                features, labels = features.to(device), labels.to(device)

                # Replay batch를 사용하여 모델 학습 및 past gradient 계산
                

                # New batch를 사용하여 모델 학습
                # model.zero_grad()
                y_pred, _ = model(features)
                loss = loss_fn(y_pred, labels)
                loss.backward()
                cur_gradient = get_gradient(model)

                # Gradient Projection
                dotp = torch.dot(cur_gradient, past_gradient)
                if dotp < 0:
                    ref_mag = torch.dot(past_gradient, past_gradient)
                    if ref_mag > 0:
                        new_grad = cur_gradient - ((dotp / ref_mag) * past_gradient)
                        update_gradient(model, new_grad)
                    else:
                        # Handle zero division or invalid ref_mag scenario
                        update_gradient(model, cur_gradient)

                # Optimizer step
                optimizer.step()
                



        else:
            
            loaders_iter = [iter(loader) for loader in new_loaders]
            times = [len(loader) for loader in loaders_iter]
            times = max(times)
            t = 0   
            # print(times,t)
            while True:
                batch = []
                t+=1
                if t > times:
                    break
                
                for idx, loader_iter in enumerate(loaders_iter):
                    try:
                        batch.append(next(loader_iter))
                    except StopIteration:
                        # 데이터 로더가 끝난 경우 다시 시작
                        loader_iter = iter(new_loaders[idx])
                        batch.append(next(loader_iter))
            
                features = torch.cat([b[0] for b in batch], dim=0)  # 모든 features 합치기
                labels = torch.cat([b[1] for b in batch], dim=0)  # 모든 labels 합치기
                loss_total = 0
                if features.shape[0] <= 1 :
                    # print('사이즈가 작아 break 합니다. ')
                    continue

                features, labels = features.to(device), labels.to(device)
                y_pred, att = model(features)
                optimizer.zero_grad()
                
                loss_pred = loss_fn(y_pred, labels)
                loss_preds.append(loss_pred.item())
                loss_pred.backward()
                optimizer.step()

        

        # # 에포크의 평균 손실 계산
        # avg_loss = sum(loss_preds) / len(loss_preds)

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
        # if early_stopping(val_loss, model):
        #     print(f'Early stopping at epoch {epoch}')
        #     break
        return model , val_loss

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
            loss_pred = loss_fn(y_pred , label)


            loss_pred.backward()

            loss_lst.append(loss_pred.item())
            # meta_lst.append(grad)

        feature_lst = np.array([feature.detach().cpu().numpy() for feature in feature_lst])
        label_lst = np.array([label.detach().cpu().numpy() for label in label_lst])

        att_lst =np.array([att.detach().cpu().numpy() for att in att_lst])
        att_lst = att_lst.reshape(-1,self.modal_number)
        
        return feature_lst,label_lst, att_lst


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
                    cluster_batch_size  = len(features)
                    
                if len(features) == 0 :
                    continue
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
                # tmp_feature.extend(features)
                # tmp_label.extend(labels)
                cluster_batch_size = int(self.batch_size*(len(features)/new_total_size))
                if int(cluster_batch_size) == 0  :
                    cluster_batch_size  = len(features)
                    
                if len(features) == 0 :
                    continue
                
                new_dataset = TensorDataset(torch.tensor(features), torch.tensor(labels))
                new_dataloader = DataLoader(new_dataset, batch_size=int(self.batch_size*(len(features)/new_total_size)), shuffle=True)
                new_loaders.append(new_dataloader)
        
        
        #================= Validation ===================

        
        
        
        return new_loaders, replay_loaders 


    def get_validation(self,new):
        K = self.memory_size // (self.task_num )
        new_clusters, new_memory_feature, new_memory_label = new
        
        
        vali_sizes = [int(K*0.1) for i in range(len(self.vali_feature))]
            
        for idx, (features, labels) in enumerate(zip(self.vali_feature,self.vali_label)):
            
            self.vali_feature[idx] = features[:vali_sizes[idx]]
            self.vali_label[idx] = labels[:vali_sizes[idx]]
        
        vali_size = int(K*0.1)
        
        indices = list(range(len(new_memory_feature)))
        selected_features = new_memory_feature[indices[-vali_size:]]
        selected_labels = new_memory_label[indices[-vali_size:]]
        self.vali_feature.append(selected_features)
        self.vali_label.append(selected_labels)
        
        
        tmp_feature = []
        tmp_label = []
        for features, labels in zip(self.vali_feature,self.vali_label):

            tmp_feature.extend(features)
            tmp_label.extend(labels) 
        vali_dataset = TensorDataset(torch.tensor(tmp_feature), torch.tensor(tmp_label))

        vali_dataloader = DataLoader(vali_dataset, batch_size=self.batch_size, shuffle=True)
        return vali_dataloader


    def allocate_memory(self,memory_size,replay,new,n_cluster):

        K = memory_size // (self.task_num + 1)
        self.task_num+=1
        
        replay_clusters, replay_memory_feature, replay_memory_label = replay
        new_clusters, new_memory_feature, new_memory_label = new
        
        if replay_clusters is not None:
            replay_clustered_indices = []
            for i in range(n_cluster):
                indices = np.where(replay_clusters == i)[0]
                np.random.shuffle(indices) # 섞여도 됨. 
                replay_clustered_indices.append(indices)
            
            
            total_size = sum([len(cluster) for cluster in replay_clustered_indices])

            replay_cluster_memory = [ int((memory_size-K)*(len(cluster)/total_size)) for cluster in replay_clustered_indices]

            surplus = (memory_size - K ) - sum([cluster_size for cluster_size in replay_cluster_memory])
            for i in range(surplus):
                replay_cluster_memory[i]+=1
        
        # Replay Dataset:
        # New Dataset
        
        new_clustered_indices = []
        # print('new_clusters shape',new_clusters.shape)
        for i in range(n_cluster):
            indices = np.where(new_clusters == i)[0]
            np.random.shuffle(indices) # 섞여도 됨 .
            new_clustered_indices.append(indices)
        
        total_size = sum([len(cluster) for cluster in new_clustered_indices])
        new_cluster_memory = [ int( K * (len(cluster)/total_size)) for cluster in new_clustered_indices]
        surplus = K - sum([cluster_size for cluster_size in new_cluster_memory])
        for i in range(surplus):
            new_cluster_memory[i]+=1


        
        tmp_feature = []
        tmp_label = []

        # Replay 데이터셋 처리
        if replay_clusters is not None:
            for cluster_idx, (replay_memory_size , new_memory_size) in enumerate(zip(replay_cluster_memory,new_cluster_memory)):
                # 클러스터별 샘플 선택
                replay_selected_indices = replay_clustered_indices[cluster_idx][:replay_memory_size]
                new_selected_indices = new_clustered_indices[cluster_idx][:new_memory_size]
                # 선택된 샘플 추가
                t_feature = []
                t_feature.extend(replay_memory_feature[replay_selected_indices])
                t_feature.extend(new_memory_feature[new_selected_indices])
                t_label = []
                t_label.extend(replay_memory_label[replay_selected_indices])
                t_label.extend(new_memory_label[new_selected_indices])
                tmp_feature.append(t_feature)
                tmp_label.append(t_label)
                
            print('replay clusted indices size ',replay_cluster_memory)
        else:
            
        # New 데이터셋 처리
            for cluster_idx, memory_size in enumerate(new_cluster_memory):
                # 클러스터별 샘플 선택
                selected_indices = new_clustered_indices[cluster_idx][:memory_size]
                # 선택된 샘플 추가

                tmp_feature.append(new_memory_feature[selected_indices])
                tmp_label.append(new_memory_label[selected_indices])

        
        print('new clusted indices size ',new_cluster_memory)

        
        if len(self.replay_feature)<1:

            self.replay_feature.extend(tmp_feature)
            self.replay_label.extend(tmp_label)
            # print([len(f) for f in tmp_feature])
            # print([len(f) for f in replay_feature])
            
        else:
            

            for idx in range(len(self.replay_feature)):
                        
                self.replay_feature[idx] = tmp_feature[idx]
                self.replay_label[idx] = tmp_label[idx]
            # print([len(f) for f in tmp_feature])
            # print([len(f) for f in replay_feature])
        
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

    def assign_cluster(self,model,loss_fn,new_dataset):
        if len(self.replay_feature)>0:
            replay_train_dataset, replay_vali_dataset = self.make_replay_loader_before()
            meta_loader_replay  = DataLoader(replay_train_dataset, batch_size=512, shuffle=True)
            meta_loader_new =  DataLoader(new_dataset, batch_size=512, shuffle=False)
            
            replay_feature_lst,replay_label_lst,replay_att_lst = self.meta_Train(model,loss_fn,meta_loader_replay)
            new_feature_lst,new_label_lst,new_att_lst = self.meta_Train(model,loss_fn,meta_loader_new)
            
            clustering_model = EuclideanKMeans(n_clusters = self.n_cluster, max_iter = 1000)
            replay_clusters = clustering_model.fit_predict(replay_att_lst)
            new_clusters = clustering_model.predict(new_att_lst)
        else:
            meta_loader_new =  DataLoader(new_dataset, batch_size=512, shuffle=False)
            new_feature_lst,new_label_lst,  new_att_lst   = self.meta_Train(model,loss_fn,meta_loader_new)
            
            clustering_model = EuclideanKMeans(n_clusters = self.n_cluster, max_iter = 1000)

            new_clusters = clustering_model.fit_predict(new_att_lst)

            replay_clusters,replay_feature_lst,replay_label_lst = None,None,None
        
        replay, new = (replay_clusters,replay_feature_lst,replay_label_lst),(new_clusters, new_feature_lst, new_label_lst)
        return replay,new


    def replay_loader_during_train(self , replay,new):
        # K = self.memory_size // (self.task_num + 1)
        
        
        replay_clusters, replay_memory_feature, replay_memory_label = replay
        new_clusters, new_memory_feature, new_memory_label = new
        
        if replay_clusters is not None:
            replay_clustered_indices = []
            for i in range(self.n_cluster):
                indices = np.where(replay_clusters == i)[0]
                np.random.shuffle(indices) # 섞여도 됨. 
                replay_clustered_indices.append(indices)
            
            
            total_size = sum([len(cluster) for cluster in replay_clustered_indices])

            replay_cluster_memory = [ int((self.memory_size)*(len(cluster)/total_size)) for cluster in replay_clustered_indices]

            surplus = (self.memory_size) - sum([cluster_size for cluster_size in replay_cluster_memory])
            for i in range(surplus):
                replay_cluster_memory[i]+=1
        
        # Replay Dataset:
        # New Dataset
        
        new_clustered_indices = []
        # print('new_clusters shape',new_clusters.shape)
        for i in range(self.n_cluster):
            indices = np.where(new_clusters == i)[0]
            np.random.shuffle(indices) # 섞여도 됨 .
            new_clustered_indices.append(indices)
        
        total_size = sum([len(cluster) for cluster in new_clustered_indices])
        new_cluster_memory = [ int(len(cluster)/total_size) for cluster in new_clustered_indices]
        

        
        replay_tmp_feature = []
        replay_tmp_label = []
        
        new_tmp_feature = []
        new_tmp_label = []
        
        
        
        replay_loaders = [] , new_loaders = []
        
        # Replay 데이터셋 처리
        if replay_clusters is not None:
            for cluster_idx, (replay_memory_size , new_memory_size) in enumerate(zip(replay_cluster_memory,new_cluster_memory)):
                # 클러스터별 샘플 선택
                replay_selected_indices = replay_clustered_indices[cluster_idx][:replay_memory_size]
                new_selected_indices = new_clustered_indices[cluster_idx][:new_memory_size]
                # 선택된 샘플 추가
                t_feature = []
                t_feature.extend(replay_memory_feature[replay_selected_indices])
                # t_feature.extend(new_memory_feature[new_selected_indices])
                t_label = []
                t_label.extend(replay_memory_label[replay_selected_indices])
                # t_label.extend(new_memory_label[new_selected_indices])
                
                replay_tmp_feature.append(t_feature)
                replay_tmp_label.append(t_label)
                
                
                
                t_feature = []
                # t_feature.extend(replay_memory_feature[replay_selected_indices])
                t_feature.extend(new_memory_feature[new_selected_indices])
                t_label = []
                # t_label.extend(replay_memory_label[replay_selected_indices])
                t_label.extend(new_memory_label[new_selected_indices])
                
                new_tmp_feature.append(t_feature)
                new_tmp_label.append(t_label)
                
                
                
            # print('replay clusted indices size ',replay_cluster_memory)
        else:
            
        # New 데이터셋 처리
            for cluster_idx, memory_size in enumerate(new_cluster_memory):
                # 클러스터별 샘플 선택
                selected_indices = new_clustered_indices[cluster_idx][:memory_size]
                # 선택된 샘플 추가

                new_tmp_feature.append(new_memory_feature[selected_indices])
                new_tmp_feature.append(new_memory_label[selected_indices])
        
        
        replay_total_size = sum([len(f) for f in replay_tmp_feature])
        new_total_size = sum([len(f) for f in new_tmp_feature])
        
        for replay_features , replay_labels , new_features , new_labels in zip(replay_tmp_feature,replay_tmp_label , new_tmp_feature , new_tmp_label):
            
            replay_dataset = TensorDataset(torch.tensor(replay_features), torch.tensor(replay_labels))
            cluster_batch_size = self.batch_size*(len(replay_labels)/replay_total_size)
            if int(cluster_batch_size) == 0  :
                if len(replay_labels) == 0 :
                    cluster_batch_size  = 3
                    continue
                cluster_batch_size  = len(replay_labels)
            replay_dataloader = DataLoader(replay_dataset, batch_size=int(cluster_batch_size), shuffle=True)
            replay_loaders.append(replay_dataloader)

            
            new_dataset = TensorDataset(torch.tensor(new_features), torch.tensor(new_labels))
            cluster_batch_size = self.batch_size*(len(new_labels)/new_total_size)
            if int(cluster_batch_size) == 0  :
                if len(new_labels) == 0 :
                    cluster_batch_size  = 3
                    continue
                cluster_batch_size  = len(new_labels)
            new_dataloader = DataLoader(new_dataset, batch_size=int(cluster_batch_size), shuffle=True)
            new_loaders.append(new_dataloader)
            
            
        return replay_loaders , new_loaders


    def Start_meta_solution(self,model,loss_fn,optimizer,new_dataset,early_stopping,typ):
        
        
        # print('new\t',new_train_loader)
        # print('replay\t',replay_train_loader)
        # Train, Replay, New
        
        replay,new = self.assign_cluster(model,loss_fn,new_dataset)
        # replay_train_loaders , new_train_loaders = self.cluster_based_replay_loader(replay,new)
        
        replay_vali_loader = self.get_validation(new)
        # print(replay_vali_loader)
        for ep in range(1,self.n_epochs):
            
                
            replay,new = self.assign_cluster(model,loss_fn,new_dataset)
            new_train_loaders , replay_train_loaders = self.cluster_based_replay_loader(replay,new)
        


            if typ =='ER':
                model, loss = self.Train_continual_ER(model,optimizer,loss_fn,new_train_loaders,replay_train_loaders,replay_vali_loader,early_stopping)
            elif typ == 'AGEM':
                model, loss = self.Train_continual_AGEM(model,optimizer,loss_fn,new_train_loaders,replay_train_loaders,replay_vali_loader,early_stopping)
            else:
                print('????? wrong type! ')
                exit()
            
            if early_stopping(loss, model):
                print(f'Early stopping at epoch {ep}')
                break
        # Model Update 
        
        # Model Update 
        
        
        
        
        
        self.allocate_memory(self.memory_size,replay, new,self.n_cluster)
            
        # replay_train_loader ,replay_vali_loader= self.cluster_based_replay_loader()
        
        print('Train : 현재 Memory Data : ',sum([len(feature) for feature in self.replay_feature]))
        print('Vali : 현재 Memory Data : ',sum([len(feature) for feature in self.vali_feature]))
        
        
        return model






#=================#=================#=================#=================#=================#=================#=================

