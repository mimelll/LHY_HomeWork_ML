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




class DataSetByclass(Dataset):
    def __init__(self,rootdir,class_index,transform):
        self.rootdir=rootdir
        self.class_index=class_index
        self.transform=transform
        self.alldir=os.path.join(rootdir,str(class_index))
        self.file_list = os.listdir(self.alldir)



    def __getitem__(self, item):
        img_name=os.path.join(self.alldir,self.file_list[item])
        image = Image.open(img_name)
        image = self.transform(image)
        return image,self.class_index


    def __len__(self):
        return len(self.file_list)


test_tfm=transforms.Compose([
    transforms.Resize([256,256]),
    transforms.ToTensor()

])

def main():
    print("sda")
    model_30epoch=IMGClassifier()
    model_5epoch=IMGClassifier()
    model_30epoch.load_state_dict(torch.load('./30epochmodel.ckpt'))
    model_5epoch.load_state_dict(torch.load('./model.ckpt'))
    model_30epoch=model_30epoch.to(device='cuda')
    model_5epoch=model_5epoch.to(device='cuda')
    batch_size=8
    DataSetByclass1=DataSetByclass('F:\datassssss\ml2022spring-hw3b\\food11\\validation\\labaled',1,test_tfm)
    DataSetByclass2=DataSetByclass('F:\datassssss\ml2022spring-hw3b\\food11\\validation\\labaled',2,test_tfm)
    DataLoader1=DataLoader(DataSetByclass1,batch_size=batch_size,shuffle=False)
    DataLoader2 = DataLoader(DataSetByclass2, batch_size=batch_size, shuffle=False)

    for index,data in enumerate (DataLoader2):
        img,labal=data
        img=img.to(device='cuda')
        labal=labal.to(device='cuda')

        res1=model_30epoch(img)
        res1=res1.argmax(dim=-1)
        res2=model_5epoch(img)
        res2=res2.argmax(dim=-1)
        print(res1)
        print(res2)


if __name__ == '__main__':
    main()