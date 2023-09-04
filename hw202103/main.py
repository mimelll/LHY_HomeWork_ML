import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torchvision.datasets import DatasetFolder
from torchvision.datasets import ImageFolder

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from tqdm.auto import tqdm
import os


train_tfm=transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
    transforms.ToTensor()
])


test_tfm=transforms.Compose([
    transforms.Resize([256,256]),
    transforms.ToTensor()

])


class IMGClassifier(nn.Module):
    def __init__(self):
        super(IMGClassifier,self).__init__()
        self.fclayer=nn.Sequential(
            nn.Conv2d(3,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0)

        )

        self.linlayer = nn.Sequential(
            nn.Linear(256*16*16,256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 11)
        )


    def forward(self,x):
        x=self.fclayer(x)
        x=x.flatten(1)
        x = self.linlayer(x)
        return x
# class IMGDataset(Dataset):
#     def __init__(self, path):
#
#         self.data = torch.from_numpy(X).float()
#         if y is not None:
#             y = y.astype(int)
#             self.label = torch.LongTensor(y)
#         else:
#             self.label = None
#
#     def __getitem__(self, idx):
#         if self.label is not None:
#             return self.data[idx], self.label[idx]
#         else:
#             return self.data[idx]
#
#     def __len__(self):
#         return len(self.data)


def pre_traindat(path):


    # 获取文件夹中的所有文件名
    file_names = os.listdir(path)

    # 打印所有文件名
    for file_name in file_names:
        image_path=path+'/'+file_name
        image = Image.open(image_path)
        filename_parts=file_name.split('_')

        if filename_parts[0]=='0':
            os.makedirs(path + '/labaled/' +filename_parts[0],exist_ok=True)
            output_path=path + '/labaled/' +filename_parts[0]+'/'+file_name
            image.save(output_path)

        elif filename_parts[0]=='1':
            os.makedirs(path + '/labaled/' + filename_parts[0],exist_ok=True)
            output_path = path + '/labaled/' + filename_parts[0] + '/' + file_name
            image.save(output_path)

        elif filename_parts[0] == '2':
            os.makedirs(path + '/labaled/' + filename_parts[0],exist_ok=True)
            output_path = path + '/labaled/' + filename_parts[0] + '/' + file_name
            image.save(output_path)
        elif filename_parts[0] == '3':
            os.makedirs(path + '/labaled/' + filename_parts[0],exist_ok=True)
            output_path = path + '/labaled/' + filename_parts[0] + '/' + file_name
            image.save(output_path)
        elif filename_parts[0] == '4':
            os.makedirs(path + '/labaled/' + filename_parts[0],exist_ok=True)
            output_path = path + '/labaled/' + filename_parts[0] + '/' + file_name
            image.save(output_path)
        elif filename_parts[0] == '5':
            os.makedirs(path + '/labaled/' + filename_parts[0],exist_ok=True)
            output_path = path + '/labaled/' + filename_parts[0] + '/' + file_name
            image.save(output_path)
        elif filename_parts[0] == '6':
            os.makedirs(path + '/labaled/' + filename_parts[0],exist_ok=True)
            output_path = path + '/labaled/' + filename_parts[0] + '/' + file_name
            image.save(output_path)
        elif filename_parts[0] == '7':
            os.makedirs(path + '/labaled/' + filename_parts[0],exist_ok=True)
            output_path = path + '/labaled/' + filename_parts[0] + '/' + file_name
            image.save(output_path)
        elif filename_parts[0] == '8':
            os.makedirs(path + '/labaled/' + filename_parts[0],exist_ok=True)
            output_path = path + '/labaled/' + filename_parts[0] + '/' + file_name
            image.save(output_path)
        elif filename_parts[0] == '8':
            os.makedirs(path + '/labaled/' + filename_parts[0],exist_ok=True)
            output_path = path + '/labaled/' + filename_parts[0] + '/' + file_name
            image.save(output_path)
        elif filename_parts[0] == '9':
            os.makedirs(path + '/labaled/' + filename_parts[0],exist_ok=True)
            output_path = path + '/labaled/' + filename_parts[0] + '/' + file_name
            image.save(output_path)
        elif filename_parts[0] == '10':
            os.makedirs(path + '/labaled/' + filename_parts[0],exist_ok=True)
            output_path = path + '/labaled/' + filename_parts[0] + '/' + file_name
            image.save(output_path)




def main():
    print('sdasd')
    batch_size=8
    #
    #pre_traindat('F:\datassssss\ml2022spring-hw3b\\food11\\test')
    #pre_traindat('F:\datassssss\ml2022spring-hw3b\\food11\\validation')
    #folder_path='F:\datassssss\ml2022spring-hw3b\\food11\\test'
    #custom_dataset = ImageFolder(root=folder_path, transform=test_tfm)

    train_set=DatasetFolder('F:\datassssss\ml2022spring-hw3b\\food11\\training\\labaled'
                            ,loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)

    valid_set = DatasetFolder("F:\datassssss\ml2022spring-hw3b\\food11\\validation\\labaled", loader=lambda x: Image.open(x), extensions="jpg",
                              transform=test_tfm)
    #test类下的文件命名没有按类别，所以使用DatasetFolder函数会报错，应该自定义DataSet类读取
    #test_set = DatasetFolder(custom_dataset, loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    #test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    device='cuda' if torch.cuda.is_available() ==True else 'cpu'
    IMGClass=IMGClassifier()

    model=IMGClass.to(device)
    model.device = device

    cri=nn.CrossEntropyLoss()
    learnrating=0.0000003
    optim=torch.optim.Adam(model.parameters(),lr=learnrating,weight_decay=1e-5)

    # The number of training epochs.
    n_epochs = 30

    # Whether to do semi-supervised learning.
    do_semi = False
    model.train()
    model_path = './30epochmodel.ckpt'
    for epoch in range (n_epochs):
        train_loss = []
        train_accs = []
        model.train()
        for index,data in tqdm(enumerate(train_loader)):
            optim.zero_grad()
            imgs, labels = data
            imgs=imgs.to(device)
            labels=labels.to(device)

            logits = model(imgs)
            loss=cri(logits,labels)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optim.step()

            acc = (logits.argmax(dim=-1) == labels).float().mean()

            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)
            if index%100==0:
                print(f"Train epoch/all {epoch}/{n_epochs} acc is {sum(train_accs) / len(train_accs):.5f}")
                print(f"Train epoch/all {epoch}/{n_epochs} loss is {sum(train_loss) / len(train_loss):.5f}")

        print(f"Train epoch/all {epoch}/{n_epochs} acc is {sum(train_accs)/len(train_accs):.5f}")
        print(f"Train epoch/all {epoch}/{n_epochs} loss is {sum(train_loss) / len(train_loss):.5f}")

        model.eval()
        valid_loss = []
        valid_accs = []
        for index, data in tqdm(enumerate(valid_loader)):
            optim.zero_grad()
            imgs, labels = data
            imgs=imgs.to(device)
            labels=labels.to(device)
            logits = model(imgs)
            loss = cri(logits, labels)
            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs.append(acc)

        print(f"Val epoch/all {epoch}/{n_epochs} acc is {sum(valid_accs)/len(valid_accs):.5f}")
        print(f"Val epoch/all {epoch}/{n_epochs} loss is {sum(valid_loss) / len(valid_loss):.5f}")

    torch.save(model.state_dict(), model_path)

    print('saving model at last epoch')
if __name__ == '__main__':
    main()