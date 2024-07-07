import .modules as modules
import numpy as np
import torch

class HumanDataset(torch.utils.data.Dataset):
    def __init__(self, DatasetDf):
        self.e4Acc = DatasetDf['e4Acc']
        #self.e4Bvp = DatasetDf['e4Bvp__value'].to_list()
        #self.e4Eda = DatasetDf['e4Eda__eda'].to_list()
        #self.e4Hr = DatasetDf['e4Hr__hr'].to_list()
        #self.e4Temp = DatasetDf['e4Temp__temp'].to_list()
        self.e4Bvp = DatasetDf['e4Bvp__x']
        self.e4Eda = DatasetDf['e4Eda__x']
        self.e4Hr = DatasetDf['e4Hr__x']
        self.e4Temp = DatasetDf['e4Temp__x']
        
        self.mAcc = DatasetDf['mAcc']
        self.mGps = DatasetDf['mGps']
        self.mGyr = DatasetDf['mGyr']
        self.mMag = DatasetDf['mMag']

        # self.emotionPositive = DatasetDf['positive_label']
        self.emotionTension = DatasetDf['tension_label']
        # self.action = DatasetDf['action_label']

    def __getitem__(self, i):

        tensor_list = [            
            torch.tensor(self.e4Acc.iloc[i]),
            torch.tensor(self.e4Bvp.iloc[i]).unsqueeze(0),  # 스칼라 값에 차원을 추가
            torch.tensor(self.e4Eda.iloc[i]).unsqueeze(0),
            torch.tensor(self.e4Hr.iloc[i]).unsqueeze(0),
            torch.tensor(self.e4Temp.iloc[i]).unsqueeze(0),
            torch.tensor(self.mAcc.iloc[i]),
            torch.tensor(self.mGps.iloc[i]),
            torch.tensor(self.mGyr.iloc[i]),
            torch.tensor(self.mMag.iloc[i]),
            # torch.tensor(self.emotionPositive.iloc[i]),
            
            # torch.tensor(self.action.iloc[i]),
        ]
        combined_tensor = torch.cat(tensor_list, dim=0)  # `dim=0`은 첫 번째 차원을 따라 텐서를 연결합니다.
        

        return combined_tensor , torch.tensor(self.emotionTension.iloc[i])

    def __len__(self):
        return (len(self.e4Acc))
    
# 각 모달에서 feature을 추출하는 모듈
class ILNet(torch.nn.Module):
    def __init__(self, in_channel_num,class_num):
        super(ILNet, self).__init__()
        self.check = True

        self.in_channel_num = in_channel_num
        if in_channel_num == 0:
            self.check=False
            self.Encoder = torch.nn.Sequential(
                torch.nn.Linear(1, 32),
                torch.nn.BatchNorm1d(32),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(32, 32),
                torch.nn.BatchNorm1d(32),            )
        else:
            self.Encoder = modules.CausalCNNEncoder(
                in_channels = in_channel_num, 
                channels = 8, 
                depth = 2, 
                reduced_size = 16, 
                out_channels = 32,
                kernel_size = 3
            )
        
        self.Vl = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.LogSoftmax(1)
        )
        
        
        
        self.weight_init(self)

    def weight_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)
    def forward(self, x):
        #print('Tensor : ',x.shape)
        x = x.reshape(-1, self.in_channel_num, 1)
        x = self.Encoder(x)
        v = self.Vl(x)

        return x , v
    
# 각 모달에서 추출된 feature을 조합해서 어떠한 cluster에 속해있는지 예측하는 모델
class LifeLogNet(torch.nn.Module):
    def __init__(self, class_num):
        super(LifeLogNet, self).__init__()
        # e4Acc Instance classifier
        self.e4AccILNet = ILNet(in_channel_num = 3,class_num=class_num)
        # e4Bvp Instance classifier
        self.e4BvpILNet = ILNet(in_channel_num = 1,class_num=class_num)
        # e4Eda Instance classifier
        self.e4EdaILNet = ILNet(in_channel_num = 1,class_num=class_num)
        # e4Hr Instance classifier
        self.e4HrILNet = ILNet(in_channel_num = 1,class_num=class_num)
        # e4Temp Instance classifier
        self.e4TempILNet = ILNet(in_channel_num = 1,class_num=class_num)
        # mAcc Instance classifier
        self.mAccILNet = ILNet(in_channel_num = 3,class_num=class_num)
        # mGps Instance classifier
        self.mGpsILNet = ILNet(in_channel_num = 2,class_num=class_num)
        # mGyr Instance classifier
        self.mGyrILNet = ILNet(in_channel_num = 3,class_num=class_num)
        # mMag Instance classifier
        self.mMagILNet = ILNet(in_channel_num = 3,class_num=class_num)
        
        
        self.Kl = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid()
        )
        
        self.Ql = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 9),
            torch.nn.Sigmoid()
        )
        
        
        self.classifier = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.Linear(288, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, class_num),
            torch.nn.LogSoftmax(1)
        )
        
        self.regressor = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.Linear(288, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 1)
        )
        
        
        self.weight_init(self)

    def weight_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)

    def forward(self, features):
        # e4Acc Instance classification
        e4Acc, e4Bvp, e4Eda, e4Hr, e4Temp, mAcc, mGps, mGyr, mMag = features[:,0:3] , features[:,3] , features[:,4] , features[:,5] , features[:,6] , features[:,7:10], features[:,10:12], features[:,12:15] , features[:,15:18]
        e4Acc ,e4Acc_k = self.e4AccILNet(e4Acc)
        e4Acc = e4Acc.reshape(e4Acc.shape[0], 1 , -1)
        e4Acc_k = e4Acc_k.reshape(e4Acc.shape[0], 1, -1)
        
        # e4Bvp Instance classification
        e4Bvp ,e4Bvp_k= self.e4BvpILNet(e4Bvp)
        e4Bvp = e4Bvp.reshape(e4Bvp.shape[0], 1, -1)
        e4Bvp_k = e4Bvp_k.reshape(e4Bvp.shape[0], 1, -1)
        
        # e4Eda Instance classification
        e4Eda ,e4Eda_k= self.e4EdaILNet(e4Eda)
        e4Eda = e4Eda.reshape(e4Eda.shape[0], 1, -1)
        e4Eda_k = e4Eda_k.reshape(e4Eda.shape[0], 1, -1)
        # e4Hr Instance classification
        e4Hr ,e4Hr_k= self.e4HrILNet(e4Hr)
        e4Hr = e4Hr.reshape(e4Hr.shape[0], 1, -1)
        e4Hr_k = e4Hr_k.reshape(e4Hr.shape[0], 1, -1)
        # e4Temp Instance classification
        e4Temp ,e4Temp_k= self.e4TempILNet(e4Temp)
        e4Temp = e4Temp.reshape(e4Temp.shape[0], 1, -1)
        e4Temp_k = e4Temp_k.reshape(e4Temp.shape[0], 1, -1)
        # mAcc Instance classification
        mAcc ,mAcc_k= self.mAccILNet(mAcc)
        mAcc = mAcc.reshape(mAcc.shape[0], 1, -1)
        mAcc_k = mAcc_k.reshape(mAcc.shape[0], 1, -1)
        # mGps Instance classification
        mGps ,mGps_k= self.mGpsILNet(mGps)
        mGps = mGps.reshape(mGps.shape[0], 1, -1)
        mGps_k = mGps_k.reshape(mGps.shape[0], 1, -1)
        # mGyr Instance classification
        mGyr ,mGyr_k = self.mGyrILNet(mGyr)
        mGyr = mGyr.reshape(mGyr.shape[0], 1, -1)
        mGyr_k = mGyr_k.reshape(mGyr.shape[0], 1, -1)
        # mMag Instance classification
        mMag ,mMag_k= self.mMagILNet(mMag)
        mMag = mMag.reshape(mMag.shape[0], 1, -1)
        mMag_k = mMag_k.reshape(mMag.shape[0], 1, -1)
        
        
        
        
        self_value = torch.cat(
            [
                e4Acc, e4Bvp, e4Eda, e4Hr, e4Temp, mAcc, mGps, mGyr, mMag
            ],
            axis = 1
        )
        self_key = torch.cat(
            [
                e4Acc_k, e4Bvp_k, e4Eda_k, e4Hr_k, e4Temp_k, mAcc_k, mGps_k, mGyr_k, mMag_k
            ],
            axis = 1
        )
        
        self_query = self.Ql(self_key)
        self_key = self.Kl(self_key)
        QKT = self_query.transpose(1, 2)@self_key / torch.tensor(9)
        att = torch.softmax((QKT), 1)

        b = torch.mul(att, self_value)
        b = b.reshape(-1,288)
        
        
        
        return self.classifier(b) , QKT ##QKT#att 
    

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
        
        