import logging
import os
import cv2
import random
import numpy as np
from shutil import copyfile, move
from PIL import Image
import torch
import torch.utils.data as data
from torchvision import transforms
from datasets import custom_transforms as tr
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def read_own_data(root_path, split='train'):
    images = []
    masks = []

    image_root = os.path.join(root_path, split + '/images')
    gt_root = os.path.join(root_path, split + '/labels')

    for image_name in os.listdir(image_root):
        image_path = os.path.join(image_root, image_name)
        label_path = os.path.join(gt_root, image_name)

        images.append(image_path)
        masks.append(label_path)

    return images, masks


def own_data_loader(img_path, mask_path):
    img = Image.open(img_path).convert('RGB')
    # print('image path = ' + str(img_path))
    # print('\n'+'image path = ' + str(mask_path))
    mask = Image.open(mask_path)
    mask = np.array(mask)
    mask[mask > 0] = 1     #这里我把255转到了1
    img = np.array(img)
    return img, mask


class ImageFolder(data.Dataset):

    def __init__(self, args, split='train'):
        self.args = args
        self.root = self.args.data_path
        self.split = split
        self.images, self.labels = read_own_data(self.root, self.split)

    def transform_tr(self, sample):
        composed_transforms = A.Compose([
            A.Resize(height=self.args.img_size, width=self.args.img_size),  # 统一图片大小
            A.HorizontalFlip(p=0.5),  # 随机水平翻转
            A.VerticalFlip(p=0.5),  # 随机垂直翻转
            A.RandomRotate90(p=0.5),  # 随机90度旋转
            A.ElasticTransform(p=0.5),  # 弹性变换
            A.RandomBrightnessContrast(p=0.5),  # 随机亮度对比度
            A.GaussNoise(p=0.5),  # 高斯噪声
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # 归一化
            ToTensorV2()  # 转张量
        ])
        return composed_transforms(image=sample['image'], mask=sample['label'])

    def transform_val(self, sample):
        composed_transforms = A.Compose([
            A.Resize(height=self.args.img_size, width=self.args.img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # 归一化
            ToTensorV2()
            ])
        return composed_transforms(image=sample['image'], mask=sample['label'])

    def __getitem__(self, index):
        img, mask = own_data_loader(self.images[index], self.labels[index])
        if self.split == "train":
            sample = {'image': img, 'label': mask}
            return self.transform_tr(sample)
        elif self.split == 'test':
            img_name = os.path.split(self.images[index])[1]
            sample = {'image': img, 'label': mask}
            sample_ = self.transform_val(sample)
            sample_['case_name'] = img_name[0:-4]
            return sample_
        elif self.split == 'test2':
            img_name = os.path.split(self.images[index])[1]
            sample = {'image': img, 'label': mask}
            sample_ = self.transform_val(sample)
            sample_['case_name'] = img_name[0:-4]
            return sample_
        # return sample

    def __len__(self):
        assert len(self.images) == len(self.labels), 'The number of images must be equal to labels'
        return len(self.images)