from __future__ import print_function

import os
import sys
import argparse
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from PIL import Image as PILImage
import numpy as np

from models.model_zoo import get_segmentation_model
from utils.score import SegmentationMetric
from utils.logger import setup_logger
from utils.distributed import synchronize, get_rank, make_data_sampler, make_batch_data_sampler
from dataset.cityscapes import CSTestSet
from PIL import Image
import torchvision.transforms as T



def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Test With Pytorch')
    # model and dataset
    parser.add_argument('--model', type=str, default='deeplabv3',
                        help='model name')  
    parser.add_argument('--method', type=str, default='kd',
                        help='method name')  
    parser.add_argument('--backbone', type=str, default='resnet18',
                        help='backbone name')
    parser.add_argument('--dataset', type=str, default='citys',
                        help='dataset name')
    parser.add_argument('--data', type=str, default='./dataset/cityscapes/',  
                        help='dataset directory')
    parser.add_argument('--data-list', type=str, default='./dataset/list/cityscapes/test.lst',  
                        help='dataset directory')
    parser.add_argument('--workers', '-j', type=int, default=8,
                        metavar='N', help='dataloader threads')
    
    # training hyper params
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    # cuda setting
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)
    # checkpoint and log
    parser.add_argument('--pretrained', type=str, default='psp_resnet18_citys_best_model.pth',
                        help='pretrained seg model')
    parser.add_argument('--save-dir', default='../runs/logs/',
                        help='Directory for saving predictions')
    parser.add_argument('--save-pred', action='store_true', default=False,
                    help='save predictions')

    # validation 
    parser.add_argument('--flip-eval', action='store_true', default=False,
                        help='flip_evaluation')
    parser.add_argument('--scales', default=[1.], type=float, nargs='+', help='multiple scales')
    args = parser.parse_args()

    if args.backbone.startswith('resnet'):
        args.aux = True
    elif args.backbone.startswith('mobile'):
        args.aux = False
    else:
        raise ValueError('no such network')
    return args


class Evaluator(object):
    def __init__(self, args, num_gpus):
        self.args = args
        self.num_gpus = num_gpus
        self.device = torch.device(args.device)

        ignore_label = -1
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                        3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                        7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                        14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                        18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                        28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

        # dataset and dataloader
        self.val_dataset = CSTestSet(args.data, args.data_list)

        val_sampler = make_data_sampler(self.val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=1)
        self.val_loader = data.DataLoader(dataset=self.val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=args.workers,
                                          pin_memory=True)

        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
        self.model = get_segmentation_model(model=args.model, 
                                            backbone=args.backbone,
                                            aux=args.aux, 
                                            pretrained=args.pretrained, 
                                            pretrained_base='None',
                                            local_rank=args.local_rank,
                                            norm_layer=BatchNorm2d).to(self.device)
        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                device_ids=[args.local_rank], output_device=args.local_rank)
        self.model.to(self.device)

        self.metric = SegmentationMetric(self.val_dataset.num_class)

    def id2trainId(self, label, id_to_trainid, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy


    def reduce_tensor(self, tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        return rt

    def predict_whole(self, net, image, tile_size):
        interp = nn.Upsample(size=tile_size, mode='bilinear', align_corners=True)
        prediction = net(image.cuda())
        if isinstance(prediction, tuple) or isinstance(prediction, list):
            prediction = prediction[0]
        prediction = interp(prediction)
        return prediction
    
    def get_color_pallete(self, npimg, dataset='pascal_voc'):
        """Visualize image.

        Parameters
        ----------
        npimg : numpy.ndarray
            Single channel image with shape `H, W, 1`.
        dataset : str, default: 'pascal_voc'
            The dataset that model pretrained on. ('pascal_voc', 'ade20k')

        Returns
        -------
        out_img : PIL.Image
            Image with color pallete

        """
        # recovery boundary
        if dataset in ('pascal_voc', 'pascal_aug'):
            npimg[npimg == -1] = 255
        # put colormap
        if dataset == 'ade20k':
            npimg = npimg + 1
            out_img = Image.fromarray(npimg.astype('uint8'))
            out_img.putpalette(adepallete)
            return out_img
        elif dataset == 'citys':
            # cityspallete = [
            #     128, 64, 128,
            #     244, 35, 232,
            #     70, 70, 70,
            #     102, 102, 156,
            #     190, 153, 153,
            #     153, 153, 153,
            #     250, 170, 30,
            #     220, 220, 0,
            #     107, 142, 35,
            #     152, 251, 152,
            #     0, 130, 180,
            #     220, 20, 60,
            #     255, 0, 0,
            #     0, 0, 142,
            #     0, 0, 70,
            #     0, 60, 100,
            #     0, 80, 100,
            #     0, 0, 230,
            #     119, 11, 32,
            # ]
            cityspallete = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 111, 74, 0, 81, 0, 81, 128, 64,128, 244, 35,232, 250,170,160, 230,150,140, 70, 70, 70, 102,102,156, 190,153,153, 180,165,180, 150,100,100, 150,120, 90, 153,153,153, 153,153,153, 250,170, 30, 220,220, 0, 107,142, 35, 152,251,152, 70,130,180, 220, 20, 60, 255, 0, 0, 0, 0,142, 0, 0, 70, 0, 60,100, 0, 0, 90, 0, 0,110, 0, 80,100, 0, 0,230, 119, 11, 32, 0, 0, 142]
            out_img = Image.fromarray(npimg.astype('uint8'))
            out_img.putpalette(cityspallete)
            return out_img
        elif dataset == 'mhpv1':
            out_img = Image.fromarray(npimg.astype('uint8'))
            out_img.putpalette(mhpv1pallete)
            return out_img
        out_img = Image.fromarray(npimg.astype('uint8'))
        out_img.putpalette(vocpallete)
        return out_img

    def eval(self):
        self.metric.reset()
        self.model.eval()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        logger.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            N_, C_, H_, W_ = image.size()
            tile_size = (H_, W_)
            full_probs = torch.zeros((1, self.val_dataset.num_class, H_, W_)).cuda()

            scales = args.scales
            with torch.no_grad():
                for scale in scales:
                    scale = float(scale)
                    print("Predicting image scaled by %f" % scale)
                    scale_image = F.interpolate(image, scale_factor=scale, mode='bilinear', align_corners=True)
                    scaled_probs = self.predict_whole(model, scale_image, tile_size)

                    if args.flip_eval:
                        print("flip evaluation")
                        flip_scaled_probs = self.predict_whole(model, torch.flip(scale_image, dims=[3]), tile_size)
                        scaled_probs = 0.5 * (scaled_probs + torch.flip(flip_scaled_probs, dims=[3]))
                    full_probs += scaled_probs
                full_probs /= len(scales)  

            if self.args.save_pred:
                pred = torch.argmax(full_probs, 1)
                pred = pred.cpu().data.numpy()
                seg_pred = self.id2trainId(pred, self.id_to_trainid, reverse=True)
                predict = seg_pred.squeeze(0)
                mask = self.get_color_pallete(predict, self.args.dataset)
                # mask = PILImage.fromarray(predict.astype('uint8'))

                fname = os.path.basename(os.path.splitext(" ".join(filename[0]))[0]).split('/')[-1]
                mask.save(os.path.join(args.outdir, fname + '.png'))
                print('Save mask to ' + os.path.join(args.outdir, fname + '.png') + ' Successfully!')

        synchronize()


if __name__ == '__main__':
    args = parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    # TODO: optim code
    outdir = '{}_{}_{}_{}'.format(args.model, args.backbone, args.dataset, args.method)
    args.outdir = os.path.join(args.save_dir, outdir)
    if args.save_pred:
        if (args.distributed and args.local_rank == 0) or args.distributed is False:
            if not os.path.exists(args.outdir):
                os.makedirs(args.outdir)

    logger = setup_logger("semantic_segmentation", args.save_dir, get_rank(),
                          filename='{}_{}_{}_log.txt'.format(args.model, args.backbone, args.dataset), mode='a+')

    evaluator = Evaluator(args, num_gpus)
    evaluator.eval()
    torch.cuda.empty_cache()
