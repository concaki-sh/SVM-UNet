import math

import torch
import random
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageFilter
from matplotlib import pyplot as plt
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}


class RandomRotate(object):
    def __init__(self, degree=45):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.uniform(-1 * self.degree, 1 * self.degree)
        rotate_degree = random.choice([0, 90, 180, 270]) if random.random() < 0.5 else rotate_degree
        img = img.rotate(rotate_degree, Image.BILINEAR, expand=True)
        mask = mask.rotate(rotate_degree, Image.NEAREST, expand=True)
        # img = np.array(img)
        # mask = np.array(mask)
        # fig, axs = plt.subplots(1, 2)
        # axs[0].imshow(img)
        # axs[0].set_title("image")
        # axs[1].imshow(mask)
        # axs[1].set_title("mask")
        # plt.show()

        return {'image': img,
                'label': mask}


class RandomMasking(object):
    def __init__(self, probability=0.5, ratio=0.02):
        self.probability = probability
        self.ratio = ratio

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        # Convert PIL images to numpy arrays
        img_np = np.array(img)
        mask_np = np.array(mask)
        pro = random.choice([0.25, 0.5, 0.75])
        # Define augmentation sequence
        seq = iaa.Sequential([
            iaa.Sometimes(self.probability,
                          iaa.CoarseDropout(p=pro, size_percent=self.ratio))
        ])
        augmented = seq(image=img_np)
        augmented_img = Image.fromarray(augmented)
        black_pixels = np.all(augmented == [0, 0, 0], axis=-1)
        mask_np[black_pixels] = 0
        mask_np = Image.fromarray(mask_np)
        return {'image': augmented_img, 'label': mask_np}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'label': mask}


class RandomContrast(object):
    def __init__(self, a_range=(0.5, 1.5)):
        self.seq = iaa.Sequential(
            [iaa.contrast.LinearContrast(alpha=a_range, per_channel=True)], random_order=True
        )

    def __call__(self, sample):
        img = np.array(sample['image'])
        mask = np.array(sample['label'])
        contrast = self.seq(images=[img, mask])
        img = Image.fromarray(contrast[0])
        mask = Image.fromarray(mask)
        return {'image': img, 'label': mask}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = np.array(sample['image'])
        mask = np.array(sample['label'])

        img = cv2.resize(img, (self.crop_size, self.crop_size))
        mask = cv2.resize(mask, (self.crop_size, self.crop_size))

        img = Image.fromarray(img)
        mask = Image.fromarray(mask)
        # fig, axs = plt.subplots(1, 2)
        #
        # axs[0].imshow(img)
        # axs[0].set_title("image")
        # axs[1].imshow(img)
        # axs[1].set_title("image")
        # plt.show()

        # random scale (short edge)
        # short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        # w, h = img.size
        # if h > w:
        #     ow = short_size
        #     oh = int(1.0 * h * ow / w)
        # else:
        #     oh = short_size
        #     ow = int(1.0 * w * oh / h)
        # img = img.resize((ow, oh), Image.BILINEAR)
        # mask = mask.resize((ow, oh), Image.NEAREST)
        # # pad crop
        # if short_size < self.crop_size:
        #     padh = self.crop_size - oh if oh < self.crop_size else 0
        #     padw = self.crop_size - ow if ow < self.crop_size else 0
        #     img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        #     mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # # random crop crop_size
        # w, h = img.size
        # x1 = random.randint(0, w - self.crop_size)
        # y1 = random.randint(0, h - self.crop_size)
        # img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        # mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}


class RandomCr(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        k = random.random()
        if k < 0.5:
            width, height = img.size
            left = random.randint(0, width - self.crop_size)
            top = random.randint(0, height - self.crop_size)
            right = left + self.crop_size
            bottom = top + self.crop_size
            cropped_img = img.crop((left, top, right, bottom))
            cropped_mask = mask.crop((left, top, right, bottom))
            padded_image = Image.new('RGB', (width, height), (0, 0, 0))
            padded_image.paste(cropped_img, (left, top))
            padded_mask = Image.new('L', (width, height), 0)
            padded_mask.paste(cropped_mask, (left, top))
            return {'image': padded_image,
                    'label': padded_mask}
        else:
            return {'image': img,
                    'label': mask}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        # img = sample['image']
        # mask = sample['label']
        img = np.array(sample['image'])
        mask = np.array(sample['label'])

        img = cv2.resize(img, (self.crop_size, self.crop_size))
        mask = cv2.resize(mask, (self.crop_size, self.crop_size))
        img = Image.fromarray(img)
        mask = Image.fromarray(mask)
        # w, h = img.size
        # if w > h:
        #     oh = self.crop_size
        #     ow = int(1.0 * w * oh / h)
        # else:
        #     ow = self.crop_size
        #     oh = int(1.0 * h * ow / w)
        # img = img.resize((ow, oh), Image.BILINEAR)
        # mask = mask.resize((ow, oh), Image.NEAREST)
        # # center crop
        # w, h = img.size
        # x1 = int(round((w - self.crop_size) / 2.))
        # y1 = int(round((h - self.crop_size) / 2.))
        # img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        # mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': mask}
