
import numpy as np
import os
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
import imageio
from PIL import Image

def run(args):
    dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)
    labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]

    preds = []
    preds_crf = []
    for id in dataset.ids:
        cls_predicts_crf = Image.open(os.path.join(args.sem_seg_out_dir, id + '_crf.png'))
        cls_predicts = Image.open(os.path.join(args.sem_seg_out_dir, id + '.png'))
        preds_crf.append(np.array(cls_predicts_crf))
        preds.append(np.array(cls_predicts))
    confusion_crf = calc_semantic_segmentation_confusion(preds_crf, labels)[:21, :21]
    confusion = calc_semantic_segmentation_confusion(preds, labels)[:21, :21]

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    fp = 1. - gtj / denominator
    fn = 1. - resj / denominator
    iou = gtjresj / denominator

    gtj_crf = confusion_crf.sum(axis=1)
    resj_crf = confusion_crf.sum(axis=0)
    gtjresj_crf = np.diag(confusion_crf)
    denominator_crf = gtj_crf + resj_crf - gtjresj_crf
    fp_crf = 1. - gtj_crf / denominator_crf
    fn_crf = 1. - resj_crf / denominator_crf
    iou_crf = gtjresj_crf / denominator_crf

    print(fp_crf[0], fn_crf[0])
    print(np.mean(fp_crf[1:]), np.mean(fn_crf[1:]))
    print({'iou_crf': iou_crf, 'miou_crf': np.nanmean(iou_crf)})

    print(fp[0], fn[0])
    print(np.mean(fp[1:]), np.mean(fn[1:]))
    print({'iou': iou, 'miou': np.nanmean(iou)})
