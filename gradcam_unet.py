import os
import warnings
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
import timm
from datasets.dataset import NPY_datasets
from datasets.own_data import ImageFolder
from tensorboardX import SummaryWriter
from models.vmunet.vmunet import VMUNet

from engine import *
import os
import sys

from utils import *
from configs.config_setting import setting_config as config
from utils_ import *
import warnings
warnings.filterwarnings("ignore")
import argparse
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit
# 忽略警告
warnings.filterwarnings('ignore')

# 定义路径
image_dir = "/home/ivan/VM-UNet-main/data/Gland/test/images"  # 单输入图像文件夹
output_dir = "/home/ivan/VM-UNet-main/hot1p_LS_vss"  # 保存热力图的文件夹

# 创建输出文件夹
os.makedirs(output_dir, exist_ok=True)


def load_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=512, help='input patch size of network input')
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--num_classes', type=int, default=2, help='output channel of network')
    parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')
    args = parser.parse_args()
    model_cfg = config.model_config
    model = VMUNet(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        depths=model_cfg['depths'],
        depths_decoder=model_cfg['depths_decoder'],
        drop_path_rate=model_cfg['drop_path_rate'],
        load_ckpt_path=None,
    ).cuda()
    save_pth = '/home/ivan/VM-UNet-main/results/best-epoch275-dice0.9395.pth'
    check = torch.load(save_pth)
    model.load_state_dict(check, strict=False)
    model.eval()
    return model, args


def reshape_transform(tensor):
    result = tensor.permute(0, 3, 1, 2)
    return result


def process_images(model, args):
    # 遍历单输入文件夹
    for filename in os.listdir(image_dir):
        image_path = os.path.join(image_dir, filename)

        orig_image = Image.open(image_path).resize((512, 512))
        rgb_img = np.array(orig_image.convert('RGB'))
        rgb_img = np.float32(rgb_img) / 255.0

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        )
        input_tensor = transform(orig_image).unsqueeze(0).float().cuda()

        # 模型推理
        with torch.no_grad():
            output = model(input_tensor)
        sem_class_to_idx = {'__background__': 0, 'gland': 1}
        target_category = sem_class_to_idx['gland']
        pred_mask = torch.argmax(output, dim=1).squeeze(0).detach().cpu().numpy()
        target_mask = np.float32(pred_mask == target_category)

        class SemanticSegmentationTarget:
            def __init__(self, category, mask):
                self.category = category
                self.mask = torch.from_numpy(mask).cuda()

            def __call__(self, model_output):
                return (model_output[self.category, :, :] * self.mask).sum()

        # target_layers = [model.vmunet.layers_up[3].blocks[0].self_attention.conv2d]
        target_layers = [model.vmunet.layers[3].blocks[0].self_attention.conv2d]
        targets = [SemanticSegmentationTarget(target_category, target_mask)]

        # 生成热力图
        with GradCAM(model=model, target_layers=target_layers) as cam:#, reshape_transform=reshape_transform
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True,
                                          colormap=cv2.COLORMAP_JET)  # 此处我为双输入因此rgb_img为我的pet
            # cam_image = show_cam_on_image(rgb_img, grayscale_cam)  # 此处我为双输入因此rgb_img为我的pet
            # cam_image = cv2.applyColorMap(cam_image, cv2.COLORMAP_JET)
            # cam_image = cv2.applyColorMap(255 - cam_image, cv2.COLORMAP_JET)

        # 保存热力图
        output_path = os.path.join(output_dir, f"heatmap_{filename}")
        cam_result = Image.fromarray(cam_image)
        cam_result.save(output_path)
        print(f"保存热力图: {output_path}")



if __name__ == "__main__":
    model, args = load_model()
    process_images(model, args)