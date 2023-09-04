import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import numpy
import torch.optim as optim
import csv
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
def getdevice():
    return 'cuda' if torch.cuda.is_available() else 'cpu'



# class COVID19Dataset(Dataset):
#     def __init__(self,path,mode='train'):



def read_csv(path):
    data_list = []
    with open(path, newline='', encoding='utf-8') as csvfile:
        # 2. 使用 csv.reader 函数读取文件对象中的数据
        csv_reader = csv.reader(csvfile)
        # 3. 遍历行并处理数据
        for index,row in enumerate(csv_reader):
            # 假设你的 CSV 文件有三列，你可以通过索引访问每一列的数据

            if index==0:continue
            row = [float(item) for item in row]
            #tensor_row = torch.tensor(float_row, dtype=torch.float32)
            data_list.append(np.array(row))
        numpy_matrix = np.array(data_list)
    tensor_data = torch.FloatTensor(numpy_matrix)
    return tensor_data



class covid_dataset(Dataset):
    def __init__(self,matrix):
        self.matrix=matrix

    def __len__(self):
        return self.matrix.shape[0]

    def __getitem__(self, item):
        return self.matrix[item]

class NeuralNet(nn.Module):
    ''' A simple fully-connected deep neural network '''
    def __init__(self, input_dim=92):
        super(NeuralNet, self).__init__()

        # Define your neural network here
        # TODO: How to modify this model to achieve better performance?
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

        # Mean squared error loss
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        ''' Given input of size (batch_size x input_dim), compute output of the network '''

        return self.net(x)

    def cal_loss(self, pred, target):
        ''' Calculate loss '''
        # TODO: you may implement L2 regularization here
        loss=self.criterion(pred, target)
        # l2_lambda = 0.01  # 正则化系数，可以根据需要进行调整
        # device = get_device()
        # l2_regularization = torch.tensor(0., device=device)  # 初始化正则化项，确保与模型参数在同一设备上
        #
        # for param in self.parameters():
        #     l2_regularization += torch.norm(param.to(device), p=2)  # 计算参数的L2范数，并移动到相同设备上
        #
        # loss += l2_lambda * l2_regularization  # 将正则化项加到损失中
        return loss

class linnernetworker(nn.Module):
    def __init__(self,input_size=92,output_size=1):
        super(linnernetworker,self).__init__()
        self.fc1=nn.Linear(92,128)
        self.sigmode = nn.Sigmoid()
        self.fc2=nn.Linear(128,64)
        self.fc3 = nn.Linear(64, 32)
        self.relu=nn.ReLU()
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 8)
        self.fc6 = nn.Linear(8, 1)
    def forward(self,x):
        x=self.fc1(x)
        x=self.sigmode(x)
        x=self.fc2(x)
        x=self.sigmode(x)
        x=self.fc3(x)
        x=self.relu(x)
        x=self.fc4(x)
        x=self.relu(x)
        x=self.fc5(x)
        x=self.relu(x)
        x=self.fc6(x)
        return x

def main():
    device=getdevice()
    path_train='./covid.train.csv'
    path_test='./covid.test.csv'
    trainnumpy=read_csv(path_train)
    batch_size=270
    shuffle=False
    num_workers=1
    epochs=3000
    testtnumpy=read_csv(path_test)
    train_covid_dataset=covid_dataset(trainnumpy)
    train_data_loader = DataLoader(train_covid_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    model=NeuralNet()
    loss_function =nn.MSELoss(reduction='mean')
    learning_rate = 0.001
    momentum=0.9
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=momentum)  # Adam 优化器
    model=model.to(device)

    loss_small=0
    save_path='./model_parameters.pth'
    for epoch in range(epochs):
        loss_sum=0
        model.train()
        for index,batchdata in enumerate(train_data_loader):
            optimizer.zero_grad()
            print("sdadsa")


            feature=batchdata[:,1:93]
            labal=batchdata[:,-1]
            feature[:, 40:] = \
                (feature[:, 40:] - feature[:, 40:].mean(dim=0, keepdim=True)) \
                / feature[:, 40:].std(dim=0, keepdim=True)


            feature=feature.to(device)
            labal=labal.to(device)

            modeloutput=model(feature).squeeze(1)
            mse_loss = model.cal_loss(modeloutput, labal)
            #modeloutput = model(feature)

            print(labal)
            print(modeloutput)


            mse_loss.backward()
            optimizer.step()
#            print(loss.item())
            loss_sum+=mse_loss.item()
        if epoch==0:
            loss_small=loss_sum/train_covid_dataset.__len__();
        else:
            loss_small=min(loss_small,loss_sum/train_covid_dataset.__len__())
            torch.save(model.state_dict(), save_path)
        print("epoch {} average loss is {}".format(epoch,loss_sum/train_covid_dataset.__len__()))



def main1():
    numpy_array = np.array([[1, 2, 3], [4, 5, 6]])
    device=getdevice()
    print(device)
    # 使用 torch.tensor() 函数将 NumPy 数组转换为 PyTorch 的 Tensor
    tensor_data = torch.tensor(numpy_array)
    print(tensor_data.device)
    tensor_data.to(device)
    print(tensor_data.device)

if __name__ == '__main__':
    main()

