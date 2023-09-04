import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import gc
import torch
import torch.nn as nn
import os


def importdata(path):
    print('{} is import!'.format('data'))
    train = np.load(path + 'train_11.npy')
    train_label = np.load(path + 'train_label_11.npy')
    test = np.load(path + 'test_11.npy')

    print('Size of training data: {}'.format(train.shape))
    print('Size of testing data: {}'.format(test.shape))
    return train,train_label,test



def grttrainandval(train,train_label):
    VAL_RATIO = 0.2

    percent = int(train.shape[0] * (1 - VAL_RATIO))
    train_x, train_y, val_x, val_y = train[:percent], train_label[:percent], train[percent:], train_label[percent:]
    print('Size of training set: {}'.format(train_x.shape))
    print('Size of validation set: {}'.format(val_x.shape))
    return train_x,train_y,val_x,val_y


def getDataloader(BATCH_SIZE,train_x,train_y,val_x,val_y):

    train_set = TIMITDataset(train_x, train_y)
    val_set = TIMITDataset(val_x, val_y)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)  # only shuffle the training data
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    return train_set,val_set,train_loader,val_loader

def deldataset(train, train_label, train_x, train_y, val_x, val_y):
    del train, train_label, train_x, train_y, val_x, val_y
    gc.collect()


class TIMITDataset(Dataset):#能够很好的处理训练集，验证集，和测试集
    def __init__(self, X, y=None):
        self.data = torch.from_numpy(X).float()
        if y is not None:
            y = y.astype(int)
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)



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


def get_device():
  return 'cuda' if torch.cuda.is_available() else 'cpu'

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True





def main():
    print('test')
    traindata,trainlabel,testdata=importdata('F:\datassssss\\')
    train_x,train_y,val_x,val_y=grttrainandval(traindata,trainlabel)
    train_set,val_set,train_loader,val_loader=getDataloader(64,train_x,train_y,val_x,val_y)





    # fix random seed for reproducibility
    same_seeds(123)

    # get device
    device = get_device()
    print(f'DEVICE: {device}')

    # training parameters
    num_epoch = 20  # number of training epoch
    learning_rate = 0.0001  # learning rate

    # the path where checkpoint saved
    model_path = './model.ckpt'

    # if not os.path.exists(model_path):
    #     os.makedirs(model_path)
    #     print("文件夹已创建")
    # else:
    #     print("文件夹已存在")

    # create model, define a loss function, and optimizer
    model = MyNetWorker().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    bestacc=0

    for epoch in range(num_epoch):
        train_acc=0
        train_loss=0
        val_acc=0
        val_loss=0

        model.train()
        for i,data in enumerate(train_loader):
            inputs,labels=data
            inputs, labels = inputs.to(device),labels.to(device)
            optimizer.zero_grad()
            output=model(inputs)
            batchloss=criterion(output,labels)
            _,train_pred=torch.max(output,1)
            batchloss.backward()
            optimizer.step()
            train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
            train_loss += batchloss.item()


        if len(val_set)>0:
            model.eval()
            with torch.no_grad():
                for i,data in enumerate(val_loader):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    output=model(inputs)
                    batchloss = criterion(output, labels)
                    _,val_pred=torch.max(output,1)
                    val_acc += (val_pred.cpu() == labels.cpu()).sum().item()  # get the index of the class with the highest probability
                    val_loss += batchloss.item()

                print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                    epoch + 1, num_epoch, train_acc / len(train_set), train_loss / len(train_loader),
                    val_acc / len(val_set), val_loss / len(val_loader)
                ))

                if val_acc>bestacc:
                    bestacc=val_acc
                    torch.save(model.state_dict(),model_path)
                    print('saving model with acc {:.3f}'.format(bestacc / len(val_set)))
        else:
            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc / len(train_set), train_loss / len(train_loader)
            ))

    if len(val_set) == 0:
        torch.save(model.state_dict(), model_path)
        print('saving model at last epoch')



    ckpt = torch.load('./model.ckpt', map_location='cpu')  # Load your best model
    model.load_state_dict(ckpt)



if __name__ == '__main__':
    main()