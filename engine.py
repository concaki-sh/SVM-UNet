import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
# from utils import save_imgs
from utils import DiceLoss
import torch.nn as nn
from utils_ import *


def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    writer):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 
 
    loss_list = []
    dice_loss = DiceLoss(n_classes=2)
    ce_loss = nn.CrossEntropyLoss()
    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        images = data['image']
        targets = data['mask']
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

        out = model(images)
        loss_dice = dice_loss(out, targets, softmax=False)
        loss_ce = ce_loss(out, targets[:].long())
        loss = loss_dice + loss_ce

        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            #logger.info(log_info)
    scheduler.step() 
    return step


def val_one_epoch(test_loader,
                    model,
                    criterion, 
                    epoch, 
                    logger,
                    config):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    dice_list = []
    hd95_list = []
    f1_list = []
    dice_loss = DiceLoss(n_classes=2)
    ce_loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data in tqdm(test_loader):
            img = data['image']
            msk = data['mask']
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            loss_dice = dice_loss(out, msk, softmax=False)
            loss_ce = ce_loss(out, msk[:].long())
            loss = loss_dice + loss_ce

            loss_list.append(loss.item())
            gts.append(msk.squeeze(0).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 

    if epoch % config.val_interval == 0:
        preds = np.array(preds)
        gts = np.array(gts)
        for i in range(len(preds)):
            pred = preds[i]
            gt = gts[i]
            dice = ObjectDice(pred, gt)
            hd95 = ObjectHausdorff(pred, gt)
            f1 = F1score(pred, gt)
            dice_list.append(dice)
            hd95_list.append(hd95)
            f1_list.append(f1)
        log_info = (f'test of best model, loss: {np.mean(loss_list):.4f}, ObjectDice: {np.mean(dice_list)}'
                        f', ObjectHausdorff:{np.mean(hd95)},'
                        f' f1: {np.mean(f1_list)}')
        print(log_info)
        logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        #logger.info(log_info)
    
    return np.mean(loss_list)


def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    dice_list = []
    hd95_list = []
    f1_list = []
    dice_loss = DiceLoss(n_classes=2)
    ce_loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img = data['image']
            msk = data['mask']
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            loss_dice = dice_loss(out, msk, softmax=False)
            loss_ce = ce_loss(out, msk[:].long())
            loss = loss_dice + loss_ce

            loss_list.append(loss.item())
            gts.append(msk.squeeze(0).cpu().detach().numpy())
            out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 
            # if i % config.save_interval == 0:
            #     save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)

        preds = np.array(preds)
        gts = np.array(gts)
        for i in range(len(preds)):
            pred = preds[i]
            gt = gts[i]
            dice = ObjectDice(pred, gt)
            hd95 = ObjectHausdorff(pred, gt)
            f1 = F1score(pred, gt)
            dice_list.append(dice)
            hd95_list.append(hd95)
            f1_list.append(f1)
        log_info = (f'test of best model, loss: {np.mean(loss_list):.4f}, ObjectDice: {np.mean(dice_list)}'
                    f', ObjectHausdorff:{np.mean(hd95)},'
                    f' f1: {np.mean(f1_list)}')
        print(log_info)
        #logger.info(log_info)

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            #logger.info(log_info)

    return np.mean(loss_list)