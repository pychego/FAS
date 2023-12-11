from random import randint
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import glob
import numpy as np
from PIL import Image

figsize = 64

# 数据文件夹格式为
# root
#   -文件夹1：手机型号_场景_人_文件
#       -1.png
#       -2.png
#   -文件夹2：手机型号_场景_人_文件
#       -1.png
#       -2.png
# ...


def GetLabelFromFn(fn):
    fnList = fn.split('\\')
    # 所在文件夹名
    folder_name = fnList[-2]
    # 文件名
    img_fn = fnList[-1]
    # 文件夹命名为：手机型号_场景_人_文件
    phone, session, human_id, access = [int(num)-1 for num in folder_name.split('_')]
    # 文件夹内图片命名为：id.png （id可能代表视频的第id帧）
    photo_id = int(img_fn.split('.')[0])
    #print(phone, session, human_id, access, photo_id)
    return phone, session, human_id, access, photo_id

class OuluNpu(Dataset):
    def __init__(self, root):
        self.root = root
        self.DictOfFolder = {}

        # 'root\\*\\*.png' 列表内容为所有文件夹下、所有文件
        fnList = glob.glob(os.path.join(self.root,'*','*.png'))

        # 生成字典，格式为{‘文件夹名1’ ： [列表1], ...} 其中文件夹1为根目录下的文件夹，列表1内容为文件夹1下所有png的路径
        for fn in fnList:
            folder_name = fn.split('\\')[-2]
            fnSubList = self.DictOfFolder.get(folder_name)
            self.DictOfFolder[folder_name] = [] if not fnSubList else fnSubList
            self.DictOfFolder[folder_name].append(fn)
        for key in self.DictOfFolder:
            self.DictOfFolder[key] = sorted(self.DictOfFolder[key])
        # 所有文件夹名
        self.folderNameList = [key for key in self.DictOfFolder]
        # 长度为文件夹个数 乘 文件夹内文件数
        self.len = len(self.DictOfFolder) * 10
        self.transforms = transforms.Compose([
            transforms.Resize(figsize),                         # 短边调整为64，长宽比不变
            transforms.ToTensor(),                              # 转换为tensor
            transforms.Normalize(mean = (.485, .456, .406),
                                 std  = (.229, .224, .225))
        ])
       
    def __getitem__(self, idx):
        # 获取文件夹(类别)名
        folder_name = self.folderNameList[idx % len(self.DictOfFolder)]
        # 获取该文件夹下的所有文件路径列表
        fnList = self.DictOfFolder[folder_name]
        # 随机获取一个文件下标
        index = randint(0, len(fnList) - 1)
        # 打开该文件
        img = Image.open(fnList[index])
        # 对图片进行处理
        img = self.transforms(img)
        # photo实际在训练中没用到
        phone, session, human_id, access, photo_id = GetLabelFromFn(fnList[index])
        return img, access
    
    def __len__(self):
        return self.len


