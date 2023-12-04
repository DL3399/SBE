
import os
import numpy as np
import imageio

from torch import multiprocessing, cuda
from torch.utils.data import DataLoader

import voc12.dataloader
from misc import torchutils, imutils
import torch
from torch.nn import functional as F

def _work(process_id, infer_dataset, args):

    databin = infer_dataset[process_id]
    n_gpus = torch.cuda.device_count()
    infer_data_loader = DataLoader(databin, shuffle=False, num_workers=0, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):


        print(args.salbound_thres)
        for iter, pack in enumerate(infer_data_loader):
            img_name = voc12.dataloader.decode_int_filename(pack['name'][0])
            img = pack['img'][0].numpy()
            sal = pack['sal']

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5 * iter + 1) // (len(databin) // 20)), end='')

            # if os.path.exists(os.path.join(args.boundary_label_dir, img_name + '.png')):
            #    continue


            cam_dict = np.load(os.path.join(args.cam_out_dir, img_name + '.npy'), allow_pickle=True).item()

            cams = cam_dict['high_res']
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')

            # 1. find confident fg & bg
            fg_conf_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.conf_fg_thres)
            fg_conf_cam = np.argmax(fg_conf_cam, axis=0)
            pred = imutils.crf_inference_label(img, fg_conf_cam, n_labels=keys.shape[0])
            fg_conf = keys[pred]

            bg_conf_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.conf_bg_thres)
            bg_conf_cam = np.argmax(bg_conf_cam, axis=0)
            pred = imutils.crf_inference_label(img, bg_conf_cam, n_labels=keys.shape[0])
            bg_conf = keys[pred]

            # 2. combine confident fg & bg
            conf = fg_conf.copy()
            conf[fg_conf == 0] = 255
            conf[bg_conf + fg_conf == 0] = 0

            imageio.imwrite(os.path.join(args.cam_vis_dir, img_name + '.png'),
                            conf.astype(np.uint8))

            # 3. convert to boundary label
            window_size = args.window_size
            seg_label = torch.from_numpy(conf).clamp(max=21).cuda()

            valid_bg_prop = (seg_label == 0).type(torch.float32)
            valid_bg_prop = F.avg_pool2d(valid_bg_prop.unsqueeze(0), window_size, padding=window_size // 2,
                                           stride=1, count_include_pad=False).squeeze()
            valid_fg_prop = torch.stack([(seg_label == i).type(torch.float32) for i in range(1, 21)])
            valid_fg_prop = F.avg_pool2d(valid_fg_prop.unsqueeze(0), window_size, padding=window_size // 2,
                                           stride=1, count_include_pad=False).squeeze()
            valid_fg_prop = valid_fg_prop.permute(1, 2, 0)
            valid_fg_prop = torch.sort(valid_fg_prop, dim=2, descending=True)[0]

            boundary_label = torch.ones_like(seg_label, dtype=torch.uint8) * args.boundary_labels['IGNORE']
            
            sal = sal.squeeze(0)
            sal = sal / torch.max(sal)
            background = sal == 0
            sal = sal.unsqueeze(0)
            boundary_label[background] = args.boundary_labels['BG']
            
            background = (valid_bg_prop > args.theta_scale * 2) * (
                    (valid_bg_prop - valid_fg_prop[..., 0]) >= args.theta_diff * 2)
            boundary_label[background] = args.boundary_labels['BG']

            foreground = (valid_fg_prop[..., 0] > args.theta_scale * 2) * (
                    (valid_fg_prop[..., 0] - valid_bg_prop) >= args.theta_diff * 2)
            boundary_label[foreground] = args.boundary_labels['FG']

            boundary_bf = (torch.min(valid_fg_prop[..., 0], valid_bg_prop) > args.theta_scale) * (
                    abs((valid_fg_prop[..., 0] - valid_bg_prop)) < args.theta_diff)
            boundary_label[boundary_bf] = args.boundary_labels['BOUNDARY_FG_BG']

            boundary_ff = (valid_fg_prop[..., 1] > args.theta_scale) * (
                        valid_fg_prop[..., 0] - valid_fg_prop[..., 1] < args.theta_diff)
            boundary_label[boundary_ff] = args.boundary_labels['BOUNDARY_FG_FG']


###
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float64)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float64)
            sobel_x = sobel_x.cuda(non_blocking=True)
            sobel_y = sobel_y.cuda(non_blocking=True)
            sal =sal.cuda(non_blocking=True)
            # 将numpy数组归一化到[0, 1]
            

            # 将numpy数组转换为torch张量
            # sal = torch.from_numpy(sal).float()

            # 将图像张量转换为4D张量(batch_size=1, channels=1, height, width)
            sal = sal.unsqueeze(0)

            # 使用Sobel算子检测边缘
            edge_x = F.conv2d(sal, sobel_x.unsqueeze(0).unsqueeze(0))
            edge_y = F.conv2d(sal, sobel_y.unsqueeze(0).unsqueeze(0))
            edge = torch.sqrt(torch.pow(edge_x, 2) + torch.pow(edge_y, 2))
            edge = F.interpolate(edge, (sal.shape[2],sal.shape[3]), mode='bilinear', align_corners=False)
            edge = edge.squeeze(1).squeeze(0)
            # 将边缘张量转换为numpy数组
            # edge_array = edge.detach().cpu().numpy()

            # 将像素值范围从[0, 1]转换为[0, 255]
            #edge_array = (edge_array * 255).astype(np.uint8)
####
            # edge = edge/torch.max(edge)
            # saledge = edge.cpu().numpy()
            #
            # imageio.imwrite(os.path.join('/mnt/vLinuxData/shark_data/dwz/DL/expriment/BES-main_newL/result/saledge1.5', img_name + '_saledge.png'),
            #                 saledge)

            boundary_label[edge>args.salbound_thres] = args.boundary_labels['certain_FG_BG']



            boundary_label = boundary_label.cpu().numpy()
            imageio.imwrite(os.path.join(args.boundary_label_dir, img_name + '.png'),
                            boundary_label)

            #print(iter)


def run(args):
    n_gpus = torch.cuda.device_count()
    dataset = voc12.dataloader.VOC12ImageDataset(args.train_aug_list, voc12_root=args.voc12_root, img_normal=None, to_torch=False)
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(dataset, args), join=True)
    print(']')
