import cv2
import matplotlib.pyplot as plt
import scipy.ndimage.morphology
from medpy import metric
from scipy.ndimage import zoom
import SimpleITK as sitk
import numpy as np
import torch
from torch.nn.functional import softmax
from torch import nn
from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.ndimage import convolve
from scipy import ndimage
import torch.nn.functional as F
import math
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import measurements
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from torch.autograd import Variable
# GlaS metrics, translated from the official Matlab code:
# code source: https://github.com/DIAGNijmegen/neural-odes-segmentation/blob/master/metrics.py

# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BoundaryLoss(nn.Module):
    """Boundary loss.

    This function is modified from
    `PIDNet <https://github.com/XuJiacong/PIDNet/blob/main/utils/criterion.py#L122>`_.  # noqa
    Licensed under the MIT License.


    Args:
        loss_weight (float): Weight of the loss. Defaults to 1.0.
        loss_name (str): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_boundary'.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 loss_name: str = 'loss_boundary'):
        super().__init__()
        self.loss_weight = loss_weight
        self.loss_name_ = loss_name

    def forward(self, bd_pre: Tensor, bd_gt: Tensor) -> Tensor:
        """Forward function.
        Args:
            bd_pre (Tensor): Predictions of the boundary head.
            bd_gt (Tensor): Ground truth of the boundary.

        Returns:
            Tensor: Loss tensor.
        """
        log_p = bd_pre.permute(0, 2, 3, 1).contiguous().view(1, -1)
        target_t = bd_gt.view(1, -1).float()

        pos_index = (target_t == 1)
        neg_index = (target_t == 0)

        weight = torch.zeros_like(log_p)
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num * 1.0 / sum_num
        weight[neg_index] = pos_num * 1.0 / sum_num

        loss = F.binary_cross_entropy_with_logits(
            log_p, target_t, weight, reduction='mean')

        return self.loss_weight * loss

def caculate_class_weight(taget):
    class_counts = torch.bincount(taget.flatten())
    total_samples = len(taget.flatten())
    class_weights = (total_samples - class_counts) / class_counts.float()
    class_weights[class_counts == 0] = 0
    return class_weights


def ObjectHausdorff(S=None, G=None):
    S = np.array(S).astype(np.uint8)
    G = np.array(G).astype(np.uint8)

    totalAreaS = (S > 0).sum()
    totalAreaG = (G > 0).sum()
    listLabelS = np.unique(S)
    listLabelS = np.delete(listLabelS, np.where(listLabelS == 0))
    listLabelG = np.unique(G)
    listLabelG = np.delete(listLabelG, np.where(listLabelG == 0))

    temp1 = 0
    for iLabelS in range(len(listLabelS)):
        Si = (S == listLabelS[iLabelS])
        intersectlist = G[Si]
        if intersectlist.any():
            indexGi = stats.mode(intersectlist).mode
            Gi = (G == indexGi)
        else:
            tempDist = np.zeros((len(listLabelG), 1))
            for iLabelG in range(len(listLabelG)):
                Gi = (G == listLabelG[iLabelG])
                tempDist[iLabelG] = Hausdorff(Gi, Si)
            minIdx = np.argmin(tempDist)
            Gi = (G == listLabelG[minIdx])
        omegai = Si.sum() / totalAreaS
        temp1 = temp1 + omegai * Hausdorff(Gi, Si)

    temp2 = 0
    for iLabelG in range(len(listLabelG)):
        tildeGi = (G == listLabelG[iLabelG])
        intersectlist = S[tildeGi]
        if intersectlist.any():
            indextildeSi = stats.mode(intersectlist).mode
            tildeSi = (S == indextildeSi)
        else:
            tempDist = np.zeros((len(listLabelS), 1))
            for iLabelS in range(len(listLabelS)):
                tildeSi = (S == listLabelS[iLabelS])
                tempDist[iLabelS] = Hausdorff(tildeGi, tildeSi)
            minIdx = np.argmin(tempDist)
            tildeSi = (S == listLabelS[minIdx])
        tildeOmegai = tildeGi.sum() / totalAreaG
        temp2 = temp2 + tildeOmegai * Hausdorff(tildeGi, tildeSi)

    objHausdorff = (temp1 + temp2) / 2
    return objHausdorff


def Hausdorff(S=None, G=None, *args, **kwargs):
    S = np.array(S).astype(np.uint8)
    G = np.array(G).astype(np.uint8)

    listS = np.unique(S)
    listS = np.delete(listS, np.where(listS == 0))
    listG = np.unique(G)
    listG = np.delete(listG, np.where(listG == 0))

    numS = len(listS)
    numG = len(listG)
    if numS == 0 and numG == 0:
        hausdorffDistance = 0
        return hausdorffDistance
    else:
        if numS == 0 or numG == 0:
            hausdorffDistance = np.Inf
            return hausdorffDistance

    y = np.where(S > 0)
    x = np.where(G > 0)

    x = np.vstack((x[0], x[1])).transpose()
    y = np.vstack((y[0], y[1])).transpose()

    nbrs = NearestNeighbors(n_neighbors=1).fit(x)
    distances, indices = nbrs.kneighbors(y)
    dist1 = np.max(distances)

    nbrs = NearestNeighbors(n_neighbors=1).fit(y)
    distances, indices = nbrs.kneighbors(x)
    dist2 = np.max(distances)

    hausdorffDistance = np.max((dist1, dist2))
    return hausdorffDistance

def F1score(S=None, G=None):

    S = np.array(S).astype(np.uint8)
    G = np.array(G).astype(np.uint8)

    unique_values_S = np.unique(S)
    unique_values_S = np.delete(unique_values_S, np.where(unique_values_S == 0))

    unique_values_G = np.unique(G)
    unique_values_G = np.delete(unique_values_G, np.where(unique_values_G == 0))

    precision_list = []
    recall_list = []

    for value_S in unique_values_S:
        for value_G in unique_values_G:
            if value_S != 0:
                SegObj = (S == value_S)
                GTObj = (G == value_G)

                overlap = np.logical_and(SegObj, GTObj)
                areaOverlap = overlap.sum()
                areaGTObj = GTObj.sum()

                if areaGTObj > 0:
                    precision = areaOverlap / areaGTObj
                    recall = areaOverlap / GTObj.sum()

                    precision_list.append(precision)
                    recall_list.append(recall)

    precision = np.sum(precision_list) / len(precision_list) if precision_list else 0
    recall = np.sum(recall_list) / len(recall_list) if recall_list else 0

    # Handle the case when precision + recall is zero
    if precision + recall == 0:
        return 0

    f1_score = (2 * precision * recall) / (precision + recall)

    return f1_score

def ObjectDice(S, G):
    S = np.array(S).astype(np.uint8)
    G = np.array(G).astype(np.uint8)

    totalAreaG = (G > 0).sum()
    listLabelS = np.unique(S)
    listLabelS = np.delete(listLabelS, np.where(listLabelS == 0))
    numS = len(listLabelS)
    listLabelG = np.unique(G)
    listLabelG = np.delete(listLabelG, np.where(listLabelG == 0))
    numG = len(listLabelG)

    if numS == 0 and numG == 0:
        return 1
    elif numS == 0 or numG == 0:
        return 0

    temp1 = 0
    totalAreaS = (S > 0).sum()
    for iLabelS in range(len(listLabelS)):
        Si = (S == listLabelS[iLabelS])
        intersectlist = G[Si]
        if intersectlist.any():
            indexG1 = stats.mode(intersectlist).mode
            Gi = (G == indexG1)
        else:
            Gi = np.zeros(G.shape)

        omegai = Si.sum() / totalAreaS
        temp1 += omegai * Dice(Gi, Si)

    temp2 = 0
    totalAreaG = (G > 0).sum()
    for iLabelG in range(len(listLabelG)):
        tildeGi = (G == listLabelG[iLabelG])
        intersectlist = S[tildeGi]
        if intersectlist.any():
            indextildeSi = stats.mode(intersectlist).mode
            tildeSi = (S == indextildeSi)  # np logical and?
        else:
            tildeSi = np.zeros(S.shape)

        tildeOmegai = tildeGi.sum() / totalAreaG
        temp2 += tildeOmegai * Dice(tildeGi, tildeSi)

    return (temp1 + temp2) / 2


def Dice(A, B):
    intersection = np.logical_and(A, B)
    return 2. * intersection.sum() / (A.sum() + B.sum())


def contour(x):
    min_pool_x = torch.nn.functional.max_pool2d(x*-1, (3, 3), 1, 1) * -1
    max_min_pool_x = torch.nn.functional.max_pool2d(min_pool_x, (3, 3), 1, 1)
    contours = torch.nn.functional.relu(max_min_pool_x - min_pool_x)
    return contours

def softmax_helper(x):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/nd_softmax.py
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)


class BDLoss(nn.Module):
    def __init__(self):
        """
        compute boudary loss
        only compute the loss of foreground
        ref: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L74
        """
        super(BDLoss, self).__init__()
        # self.do_bg = do_bg

    def forward(self, net_output, target):
        """
        net_output: (batch_size, class, x,y,z)
        target: ground truth, shape: (batch_size, 1, x,y,z)
        bound: precomputed distance map, shape (batch_size, class, x,y,z)
        """
        bound = contour(target)
        net_output = softmax_helper(net_output)
        pc = net_output.type(torch.float32)
        dc = bound.type(torch.float32)

        multipled = torch.einsum("bcxy,bcxy->bcxy", pc, dc)
        bd_loss = multipled.mean()

        return bd_loss

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = ObjectDice(pred, gt)
        hd95 = ObjectHausdorff(pred, gt)
        f1 = F1score(pred, gt)
        return dice, hd95, f1
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0, 0
    else:
        return 0, 0, 0


def test_single_volume(image, label, net, classes, patch_size=[512, 512], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        x, y = image.shape[1], image.shape[2]
        if x != patch_size[0] or y != patch_size[1]:
            print("patch size error!")
        input = torch.from_numpy(image).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            outputs = net(input)
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
        prediction = out
        # fig = plt.figure()
        # ax1 = fig.add_subplot(1, 2, 1)
        # ax1.imshow(out)
        # ax2 = fig.add_subplot(1, 2, 2)
        # ax2.imshow(label)
        # plt.show()
    else:
        input = torch.from_numpy(image).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
            # fig = plt.figure()
            # ax1 = fig.add_subplot(1, 2, 1)
            # ax1.imshow(prediction[1])
            # ax2 = fig.add_subplot(1, 2, 2)
            # ax2.imshow(label[1])
            # plt.show()
    metric_list = []
    metric_list.append(calculate_metric_percase(prediction, label))
    if test_save_path is not None:
        cv2.imwrite(test_save_path + '/' + case + '.bmp', prediction * 255)
    return metric_list

def dilated_img(target):
    # 定义结构元素
    kernel_size = (3, 3)
    dilated_imgs = []
    kernel = torch.ones((kernel_size[0], kernel_size[1]), dtype=torch.float32,device=target.device)
    for i in range(target.size(0)):
        dilate_image = F.conv2d(target[i,...].unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=kernel_size[0]-1//2,stride=1)
        dilated_imgs.append(dilate_image)
    return torch.stack(dilated_imgs).squeeze(1).squeeze(1)


def EDT_Sigmoid(image):
    # 将输入图像转换为浮点数类型
    R = 50.0
    ydt = dilated_img(image)
    ydt_np = ydt.cpu().numpy()
    ycdt = scipy.ndimage.morphology.distance_transform_edt(ydt_np)
    ycdt = torch.from_numpy(ycdt).to(image.device)
    ycdt = ycdt / R
    ycdt = torch.sigmoid(ycdt)
    return ycdt.mean()

def HMD_Sigmoid(image):
    R = 5.0
    ydt = dilated_img(image)
    ydt_np = ydt.cpu().numpy()
    ydt_labels, num_labels = measurements.label(ydt_np)
    transformed_image = np.zeros_like(ydt_np, dtype=np.float32)
    for label in range(1, num_labels + 1):
        indices = np.where(label == ydt_labels)
        min_row = np.min(indices[0])
        max_row = np.max(indices[0])
        min_col = np.min(indices[1])
        max_col = np.max(indices[1])
        width = max_col - min_col
        height = max_row - min_row
        distance = width + height
        transformed_image[indices] = distance

    ycdt = torch.from_numpy(transformed_image).to(image.device)
    ycdt = ycdt / R
    ycdt = torch.sigmoid(ycdt)

    return ycdt.mean()
def distance_map(target, k = 1):
    ydt = dilated_img(target)
    ydt = ydt.mean()
    #ycdt = EDT_Sigmoid(target)
    ycdt = HMD_Sigmoid(target)
    d = ycdt - k * (1 - ydt)
    return d




def flatten(input, target, ignore_index):
    num_class = input.size(1)
    input = input.permute(0, 2, 3, 1).contiguous()

    input_flatten = input.view(-1, num_class)
    target_flatten = target.view(-1)

    mask = (target_flatten != ignore_index)
    input_flatten = input_flatten[mask]
    target_flatten = target_flatten[mask]

    return input_flatten, target_flatten


class JaccardLoss(nn.Module):
    def __init__(self, ignore_index=255, smooth=1.0):
        super(JaccardLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, input, target):
        input, target = flatten(input, target, self.ignore_index)
        input = softmax(input, dim=1)
        num_classes = input.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (target == c).float()
            input_c = input[:, c]

            intersection = (input_c * target_c).sum()
            total = (input_c + target_c).sum()
            union = total - intersection
            IoU = (intersection + self.smooth) / (union + self.smooth)

            losses.append(1 - IoU)

        losses = torch.stack(losses)
        loss = losses.mean()
        return loss


class ComboLoss(nn.Module):
    def __init__(self, ignore_index=255, smooth=1.0):
        super(ComboLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, input, target):
        input, target = flatten(input, target, self.ignore_index)
        input = softmax(input, dim=1)
        num_classes = input.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (target == c).float()
            input_c = input[:, c]

            intersection = (input_c * target_c).sum()
            dice = (2. * intersection + self.smooth) / (input.sum() + target.sum() + self.smooth)

            losses.append(1 - dice)

        losses = torch.stack(losses)
        loss = losses.mean()
        return loss


class TverskyLoss(nn.Module):
    def __init__(self, ignore_index=255, smooth=1.0, alpha=0.5, beta=0.5):
        super(TverskyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta

    def forward(self, input, target):
        input, target = flatten(input, target, self.ignore_index)
        input = softmax(input, dim=1)
        num_classes = input.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (target == c).float()
            input_c = input[:, c]

            t_p = (input_c * target_c).sum()
            f_p = ((1 - target_c) * input_c).sum()
            f_n = (target_c * (1 - input_c)).sum()
            tversky = (t_p + self.smooth) / (t_p + self.alpha * f_p + self.beta * f_n + self.smooth)

            losses.append(1 - tversky)

        losses = torch.stack(losses)
        loss = losses.mean()
        return loss


class FocalTverskyLoss(nn.Module):
    def __init__(self, ignore_index=255, smooth=1.0, alpha=0.5, beta=0.5, gamma=1):
        super(FocalTverskyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, input, target):
        input, target = flatten(input, target, self.ignore_index)
        input = softmax(input, dim=1)
        num_classes = input.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (target == c).float()
            input_c = input[:, c]

            t_p = (input_c * target_c).sum()
            f_p = ((1 - target_c) * input_c).sum()
            f_n = (target_c * (1 - input_c)).sum()
            tversky = (t_p + self.smooth) / (t_p + self.alpha * f_p + self.beta * f_n + self.smooth)
            focal_tversky = (1 - tversky) ** self.gamma
            losses.append(focal_tversky)

        losses = torch.stack(losses)
        loss = losses.mean()
        return loss


class LovaszSoftmaxLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(LovaszSoftmaxLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input, target):
        input, target = flatten(input, target, self.ignore_index)
        input = softmax(input, dim=1)
        num_classes = input.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (target == c).float()
            input_c = input[:, c]
            loss_c = (Variable(target_c) - input_c).abs()
            loss_c_sorted, loss_index = torch.sort(loss_c, 0, descending=True)
            target_c_sorted = target_c[loss_index]
            losses.append(torch.dot(loss_c_sorted, torch.autograd.Variable(lovasz_grad(target_c_sorted))))

        losses = torch.stack(losses)
        loss = losses.mean()
        return loss


def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class FocalLoss(nn.CrossEntropyLoss):
    def __init__(self, ignore_index=255, gamma=0.1):
        super().__init__(reduction='none')
        self.ignore_index = ignore_index
        self.gamma = gamma

    def forward(self, input, target):
        input, target = flatten(input, target, self.ignore_index)
        input_prob = torch.gather(softmax(input, dim=1), 1, target.unsqueeze(1))
        cross_entropy = super().forward(input, target)
        losses = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        loss = losses.mean()
        return loss