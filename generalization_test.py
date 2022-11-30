from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__, model_loss
from utils import *
from torch.utils.data import DataLoader
from datasets import listfiles as ls
from datasets import eth3dLoader as DA
from datasets import MiddleburyLoader as mid
import gc

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PCW-Net: Pyramid Combination and Warping Cost Volume for Stereo Matching')
parser.add_argument('--model', default='gwcnet-g', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

parser.add_argument('--dataset', required=True, help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--trainlist', required=True, help='training list')
parser.add_argument('--testlist', required=True, help='testing list')

parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--batch_size', type=int, default = 4, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default = 4, help='testing batch size')
parser.add_argument('--epochs', type=int, required=True, help='number of epochs to train')
parser.add_argument('--lrepochs', type=str, required=True, help='the epochs to decay lr: the downscale rate')

parser.add_argument('--logdir', required=True, help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')

# parse arguments, set seeds
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.makedirs(args.logdir, exist_ok=True)

# create summary logger
print("creating new summary file")
logger = SummaryWriter(args.logdir)

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
kittiloadertrain = StereoDataset(args.datapath, args.trainlist, True)
kittiloadertest = StereoDataset(args.datapath, args.testlist, False)
#eth3d
all_left_img, all_right_img, all_left_disp, _ = ls.dataloader('%s/eth3d/'%args.datapath)
eth3dloadertest = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, False)

#middleburyloader
all_left_img, all_right_img, all_left_disp, _ = ls.dataloader('%s/middleburyvalH/'%args.datapath)
# all_left_img, all_right_img, all_left_disp, _ = ls.dataloader('%s/middleburyvalQ/'%args.datapath)
middleburyloadertest = mid.myImageFloder(all_left_img, all_right_img, all_left_disp, False)

TestImgLoaderkitti = DataLoader(kittiloadertest, args.test_batch_size, shuffle=False, num_workers=4, drop_last=False)
TestImgLoadereth3d = DataLoader(eth3dloadertest, 1, shuffle=False, num_workers=4, drop_last=False)
TestImgLoadermiddlebury = DataLoader(middleburyloadertest, 1, shuffle=False, num_workers=4, drop_last=False)


# model, optimizer
model = __models__[args.model](args.maxdisp)
model = nn.DataParallel(model)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

# load parameters
start_epoch = 0
if args.resume:
    # find all checkpoints file and sort according to epoch id
    all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, all_saved_ckpts[-1])
    print("loading the lastest model in logdir: {}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load the checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
print("start at epoch {}".format(start_epoch))


def train():
    bestepoch = 0
    error = 100
    bestepochkitti = 0
    kittierror = 100
    bestepocheth3d = 0
    eth3derror = 100
    bestepochmid = 0
    miderror = 100
    for epoch_idx in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)

    #      # kitti test
        avg_test_scalars_kitti = AverageMeterDict()
        for batch_idx, sample in enumerate(TestImgLoaderkitti):
            global_step = len(TestImgLoaderkitti) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = test_sample(sample, compute_metrics=do_summary)
            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
                save_images(logger, 'test', image_outputs, global_step)
            avg_test_scalars_kitti.update(scalar_outputs)
            del scalar_outputs, image_outputs
            print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs,
                                                                                      batch_idx,
                                                                                      len(TestImgLoaderkitti), loss,
                                                                                      time.time() - start_time))
        avg_test_scalars_kitti = avg_test_scalars_kitti.mean()
        nowerror = avg_test_scalars_kitti["D1"][0]
        if  nowerror < kittierror :
            bestepochkitti = epoch_idx
            kittierror = avg_test_scalars_kitti["D1"][0]
        save_scalars(logger, 'testkitti', avg_test_scalars_kitti, len(TestImgLoaderkitti) * (epoch_idx + 1))
        print("avg_test_scalars", avg_test_scalars_kitti)
        print('MAX epoch %d total test error = %.5f' % (bestepochkitti, kittierror))
        gc.collect()
    #
    #
    #     #mid test
#         avg_test_scalars_mid = AverageMeterDict()
#         for batch_idx, sample in enumerate(TestImgLoadermiddlebury):
#             global_step = len(TestImgLoadermiddlebury) * epoch_idx + batch_idx
#             start_time = time.time()
#             do_summary = global_step % args.summary_freq == 0
#             loss, scalar_outputs, image_outputs = test_sample(sample, compute_metrics=do_summary)
#             if do_summary:
#                 save_scalars(logger, 'test', scalar_outputs, global_step)
#                 save_images(logger, 'test', image_outputs, global_step)
#             avg_test_scalars_mid.update(scalar_outputs)
#             del scalar_outputs, image_outputs
#             print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs,
#                                                                                      batch_idx,
#                                                                                      len(TestImgLoadermiddlebury), loss,
#                                                                                      time.time() - start_time))
#         avg_test_scalars_mid = avg_test_scalars_mid.mean()
#         nowerror = avg_test_scalars_mid["D1"][0]
#         if nowerror < miderror:
#             bestepochmid = epoch_idx
#             miderror = avg_test_scalars_mid["D1"][0]
#         save_scalars(logger, 'testmid', avg_test_scalars_mid, len(TestImgLoaderkitti) * (epoch_idx + 1))
#         print("avg_test_scalars_mid", avg_test_scalars_mid)
#         print('MAX epoch %d total test error = %.5f' % (bestepochmid, miderror))
#         gc.collect()
    #
    #
    #
    #
    #     #eth3d test
        avg_test_scalars_eth3d = AverageMeterDict()
        for batch_idx, sample in enumerate(TestImgLoadereth3d):
            global_step = len(TestImgLoadereth3d) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = test_sample(sample, compute_metrics=do_summary)
            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
                save_images(logger, 'test', image_outputs, global_step)
            avg_test_scalars_eth3d.update(scalar_outputs)
            del scalar_outputs, image_outputs
            print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs,
                                                                                     batch_idx,
                                                                                     len(TestImgLoadereth3d), loss,
                                                                                     time.time() - start_time))
        avg_test_scalars_eth3d = avg_test_scalars_eth3d.mean()
        nowerror = avg_test_scalars_eth3d["D1"][0]
        if nowerror < eth3derror:
            bestepocheth3d = epoch_idx
            eth3derror = avg_test_scalars_eth3d["D1"][0]
        save_scalars(logger, 'testeth3d', avg_test_scalars_eth3d, len(TestImgLoaderkitti) * (epoch_idx + 1))
        print("avg_test_scalars_eth3d", avg_test_scalars_eth3d)
        print('MAX epoch %d total test error = %.5f' % (bestepocheth3d, eth3derror))
        gc.collect()
    #
    print('MAX epoch %d total eth3dtest error = %.5f' % (bestepocheth3d, eth3derror))
    print('MAX epoch %d total midtest error = %.5f' % (bestepochmid, miderror))
    print('MAX epoch %d total kittitest error = %.5f' % (bestepochkitti, kittierror))

@make_nograd_func
def test_sample(sample, dataset = 'kitti', compute_metrics=True):
    model.eval()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    toppad, rightpad = sample['top_pad'], sample['right_pad']
    # print(toppad)
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()
    #model_start_time = time.time()
    disp_ests, pred3 = model(imgL, imgR)
    #print("whole network time= {:3f}".format(time.time() - model_start_time))
    # disp_ests, pred3_s3, pred3_s4  = model(imgL, imgR)
    if dataset == 'mid':
        # print(disp_gt.size())
        #mask = (disp_gt < args.maxdisp * 2) & (disp_gt > 0)
        mask = disp_gt > 0
        disp_ests[0] = disp_ests[0][:, toppad:, :-rightpad]
        # pred3_s3[0] = pred3_s3[0][:, toppad:, :-rightpad]
        # pred3_s4[0] = pred3_s4[0][:, toppad:, :-rightpad]

        disp_ests = F.upsample(disp_ests[0].unsqueeze(1) * 2, [disp_gt.size()[1], disp_gt.size()[2]], mode='bilinear', align_corners=True).squeeze(1)
        # pred3_s3 = F.upsample(pred3_s3[0].unsqueeze(1) * 2, [disp_gt.size()[1], disp_gt.size()[2]], mode='bilinear',
        #                        align_corners=True).squeeze(1)
        # pred3_s4 = F.upsample(pred3_s4[0].unsqueeze(1) * 2, [disp_gt.size()[1], disp_gt.size()[2]], mode='bilinear',
        #                       align_corners=True).squeeze(1)

        disp_ests = [disp_ests]
        # pred3_s3 = [pred3_s3]
        # pred3_s4 = [pred3_s4]
        # inrange_s4 = (disp_gt > 0) & (disp_gt < 256 * 2) & mask

    # mask = disp_gt > 0
    else:
        mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
        inrange_s4 = (disp_gt > 0) & (disp_gt < 256) & mask
    loss = model_loss(disp_ests, disp_gt, mask)

    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}

    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["D1_pre"] = [D1_metric(pred, disp_gt, mask) for pred in pred3]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    if compute_metrics:
        image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests]

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs

if __name__ == '__main__':
    train()
