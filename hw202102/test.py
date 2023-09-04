import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import gc
import torch
import torch.nn as nn
import os
import torch.optim as optim


def importdata(path1,path2,path3):
    train_feature=np.load(path1)
    train_labal=np.load(path2)
    test_data=np.load(path3)
    return train_feature,train_labal,test_data


class MyDataSet(Dataset):
    def __init__(self,feature,labal=None):
        super(MyDataSet, self).__init__()
        self.feature=torch.from_numpy(feature).float()#转换成tensor

        if labal is None:
            self.labal=None
        else:
            labal = labal.astype(int)
            self.labal=torch.LongTensor(labal)

    def __getitem__(self, item):
        if self.labal is None:
            return self.feature[item]
        else:
            return self.feature[item],self.labal[item]

    def __len__(self):
        return self.feature.shape[0]



class Mynetworker(nn.Module):
    def __init__(self,feature_dim=429):
        super(Mynetworker, self).__init__()
        self.fc1=nn.Linear(feature_dim,512)
        self.sigmode=nn.Sigmoid()
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(128, 64)
        self.fc7 = nn.Linear(64,39)
        self.sofmax=nn.Softmax()#使用交叉熵损失时不需要特别使用softmax函数
        self.relu=nn.ReLU()

    def forward(self,x):
        x=self.fc1(x)
        x=self.sigmode(x)
        x = self.fc2(x)
        x = self.sigmode(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.sigmode(x)
        x = self.fc5(x)
        x = self.sigmode(x)
        x = self.fc6(x)
        x = self.sigmode(x)
        x = self.fc7(x)
        return x


    def calloss(self,output,labal):
        lossfun=nn.CrossEntropyLoss()
        loss=lossfun(output,labal)
        return loss



def getDataLoader(train_feature, train_labal, test_data,train_par,batchsize):

    index_train=int(len(train_feature)*train_par)
    train_feature_spl=train_feature[:index_train]
    train_labal_spl=train_labal[:index_train]
    val_featurr_spl=train_feature[index_train:]
    val_labal_spl=train_labal[index_train:]
    MyDataSet_train=MyDataSet(train_feature_spl,train_labal_spl)
    MyDataSet_val=MyDataSet(val_featurr_spl,val_labal_spl)
    MyDataSet_test=MyDataSet(test_data)


    Train_Dataloader=DataLoader(dataset=MyDataSet_train,shuffle=False,batch_size=batchsize)
    val_Dataloader=DataLoader(dataset=MyDataSet_val,shuffle=False,batch_size=batchsize)
    test_Dataloader=DataLoader(dataset=MyDataSet_test,shuffle=False,batch_size=batchsize)

    return Train_Dataloader,val_Dataloader,test_Dataloader,MyDataSet_train,MyDataSet_val,MyDataSet_test

def getdevice():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class MyNetWorker(nn.Module):
    def __init__(self):
        super(MyNetWorker,self).__init__()
        self.layer1=nn.Linear(429,858)
        self.activeS=nn.Sigmoid()
        self.layer2=nn.Linear(858,858*2)
        self.activeR = nn.ReLU()
        self.layer3 = nn.Linear(858*2, 1024)
        #self.active = nn.Sigmoid()
        self.layer4 = nn.Linear(1024, 512)
        #self.active = nn.Sigmoid()
        self.layer5 = nn.Linear(512, 128)
        #self.active = nn.Sigmoid()
        self.out = nn.Linear(128, 39)
        #self.active = nn.Sigmoid()
        self.softmax=nn.Softmax()

    def forward(self,x):
        x=self.layer1(x)
        x=self.activeS(x)

        x=self.layer2(x)
        x=self.activeR(x)

        x=self.layer3(x)
        x = self.activeS(x)

        x=self.layer4(x)
        x = self.activeS(x)

        x=self.layer5(x)
        x = self.activeS(x)

        x=self.out(x)

        return x

    def calloss(self,output,labal):
        lossfun=nn.CrossEntropyLoss()
        loss=lossfun(output,labal)
        return loss


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    same_seeds(123)
    path1="F:\datassssss\\train_11.npy"
    path2="F:\datassssss\\train_label_11.npy"
    path3="F:\datassssss\\test_11.npy"
    train_feature, train_labal, test_data=importdata(path1,path2,path3)


    train_par=0.9
    batchsize=16

    Train_Dataloader,val_Dataloader,test_Dataloader,MyDataSet_train,MyDataSet_val,MyDataSet_test=getDataLoader(train_feature, train_labal, test_data,train_par,batchsize)
    epochs=18
    model=MyNetWorker()
    device=getdevice()
    model.to(device=device)

    learning_rate = 0.0001
    momentum=0.9
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam 优化器
    save_path='./state_dict.ckpt'
    minloss=-1
    bestacc=0
    loss_sum=0
    model.train()
    for i in range(epochs):
        loss_sum=0
        train_acc=0
        train_loss=0
        val_acc=0
        model.train()
        for index,data in enumerate(Train_Dataloader):
            feature, labal=data
            optimizer.zero_grad()
            feature=feature.to(device)
            labal=labal.to(device)
            out=model(feature)

            _, train_pred = torch.max(out, 1)

            loss=model.calloss(out,labal)
            loss.backward()
            optimizer.step()

            train_acc += (train_pred.cpu() == labal.cpu()).sum().item()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            for index, datat in enumerate(val_Dataloader):
                val_feature,val_labal=datat
                optimizer.zero_grad()
                val_feature=val_feature.to(device)
                val_labal=val_labal.to(device)
                val_out = model(val_feature)
                loss = model.calloss(val_out, val_labal)
                loss_sum+=loss.item()



                _, val_pred = torch.max(val_out, 1)

                val_acc += (val_pred.cpu() == val_labal.cpu()).sum().item()

            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                i + 1, epochs, train_acc / len(train_feature), train_loss / len(train_feature),
                val_acc / float(len(train_feature)*(1-train_par)), loss_sum / float(len(train_feature)*(1-train_par))
            ))

            if val_acc>bestacc:
                bestacc=val_acc
                torch.save(model.state_dict(), save_path)




if __name__ == '__main__':
    main()
