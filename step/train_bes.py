import torch
from torch.backends import cudnn
from PIL import Image
import numpy as np

cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F

import importlib

import voc12.dataloader
from misc import pyutils, torchutils
from net.resnet50_bes import Boundary

def run(args):
    model = Boundary()
    train_dataset = voc12.dataloader.VOC12SegmentationDataset(img_name_list_path=args.train_aug_list,
                                                              label_dir=args.boundary_label_dir,
                                                              voc12_root=args.voc12_root,
                                                              hor_flip=True,
                                                              crop_size=args.bes_crop_size,
                                                              crop_method="random",
                                                              rescale=(0.5, 1.5),
                                                              label_scale=1 / 4
                                                              )
    train_data_loader = DataLoader(train_dataset, batch_size=args.bes_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    max_step = (len(train_dataset) // args.bes_batch_size) * args.bes_num_epoches

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups, 'lr': args.bes_learning_rate, 'weight_decay': args.bes_weight_decay},
    ], lr=args.bes_learning_rate, weight_decay=args.bes_weight_decay, max_step=max_step)

    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()

    for ep in range(args.bes_num_epoches):

        print('Epoch %d/%d' % (ep + 1, args.bes_num_epoches))

        for step, pack in enumerate(train_data_loader):

            img = pack['img']
            label = pack['label'].cuda(non_blocking=True)
            sal = pack['sal'].cuda(non_blocking=True)
            sal = F.interpolate(sal.unsqueeze(0), (label.shape[1],label.shape[2]), mode='bilinear', align_corners=False)[0]
            sal_region = sal > 0
            sal_falseregion = sal == 0
            sal_region = sal_region.type(torch.float32)
            sal_falseregion = sal_falseregion.type(torch.float32)
            predict = model(img).squeeze().clamp(min=1e-4, max=1 - 1e-4)
            if torch.max(predict)>1:
                print(torch.max(predict))
            mask_bg_ground = (label == args.boundary_labels['BG']).type(torch.float32)
            mask_fg_ground = (label == args.boundary_labels['FG']).type(torch.float32)


            mask_boundary_uncertain = ((label == args.boundary_labels['BOUNDARY_FG_FG']) | (label == args.boundary_labels['BOUNDARY_FG_BG'])).type(torch.float32)
            mask_boundary_certain = (label == args.boundary_labels['certain_FG_BG']).type(torch.float32)

            #mask_boundary = ((label == args.boundary_labels['BOUNDARY_FG_FG']) | (label == args.boundary_labels['BOUNDARY_FG_BG']) | (label == args.boundary_labels['certain_FG_BG'])).type(torch.float32)
            # boundary FGBG
            # mask_boundary_FGvsBG = (label == args.boundary_labels['certain_FG_BG']).type(torch.float32)
            # mask_boundary_FGvsFG = (label == args.boundary_labels['BOUNDARY_FG_FG']).type(torch.float32)
            loss_bg_ground = (-torch.log(1 - predict) * mask_bg_ground).sum() / (mask_bg_ground.sum() + 1)
            loss_fg_ground = (-torch.log(1 - predict) * mask_fg_ground).sum() / (mask_fg_ground.sum() + 1)
            loss_ground = (loss_fg_ground + loss_bg_ground)#/2
            # #######################
            # # loss_boundary_sal_region = (-torch.log(predict) * torch.pow(predict.detach(), 0.5) * mask_boundary * sal_region).sum() / ((mask_boundary * sal_region).sum() + 1)
            # # loss_boundary_sal_falseregion = (-torch.log(predict) * torch.pow(predict.detach(), 0.5) * mask_boundary * sal_falseregion).sum() / ((mask_boundary * sal_falseregion).sum() + 1)
            # loss_boundary_FGvsBG = (-torch.log(predict) * torch.pow(predict.detach(), 0.5) * mask_boundary_FGvsBG).sum() / ((mask_boundary_FGvsBG).sum() + 1)
            # loss_boundary_FGvsFG = (-torch.log(predict) * torch.pow(predict.detach(), 0.5) * mask_boundary_FGvsFG).sum() / ((mask_boundary_FGvsFG).sum() + 1)

            # boundary all
            mask_boundary = ((label == args.boundary_labels['BOUNDARY_FG_FG']) | (label == args.boundary_labels['certain_FG_BG']) | (label == args.boundary_labels['BOUNDARY_FG_BG'])).type(torch.float32)
            loss_boundary_sal = (-torch.log(predict) * torch.pow(predict.detach(), 0.5) * mask_boundary * sal_region).sum() / ((mask_boundary * sal_region).sum() + 1)
            loss_boundary_falsesal =(-torch.log(predict) * torch.pow(predict.detach(), 0.5) * mask_boundary * sal_falseregion).sum() / ((mask_boundary * sal_falseregion).sum() + 1)
            loss_boundary = loss_boundary_sal + loss_boundary_falsesal
            #loss_boundary = loss_boundary_sal * (1 - args.newL_weight) + loss_boundary_falsesal * args.newL_weight
            # loss_boundary_FGvsFG = loss_boundary_FGvsFG
            # loss_boundary_sal_falseregion = loss_boundary_sal_falseregion * 0.2
            # loss_boundary = loss_boundary_sal_region + loss_boundary_sal_falseregion
            #loss_boundary = loss_boundary_FGvsBG + loss_boundary_FGvsFG
            # loss_boundary = (-torch.log(predict) * torch.pow(predict.detach(), 0.5) * mask_boundary * sal).sum() / ((mask_boundary * sal).sum() + 1)
            # loss_boundary_sal_region = loss_boundary
            # loss_boundary_sal_falseregion = loss_boundary
            loss = (loss_ground + loss_boundary)
            avg_meter.add({'loss1': loss_ground.item(), 'loss2': loss_boundary.item(),'loss_boundary_sal':loss_boundary_sal.item(),'loss_boundary_falsesal':loss_boundary_falsesal.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step - 1) % 100 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss1:%.4f' % (avg_meter.pop('loss1')),
                      'loss2:%.4f' % (avg_meter.pop('loss2')),
                      'loss_boundary_sal:%.4f' % (avg_meter.pop('loss_boundary_sal')),
                      'loss_boundary_falsesal:%.4f' % (avg_meter.pop('loss_boundary_falsesal')),
                      'imps:%.1f' % ((step + 1) * args.bes_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)
        else:
            timer.reset_stage()

    torch.save(model.module.state_dict(), args.bes_weights_name)
    torch.cuda.empty_cache()
