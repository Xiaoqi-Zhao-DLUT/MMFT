import torch
import torch.nn.functional as F
from torch.autograd import Variable
import pdb, os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from datetime import datetime
from model.depth_free_ablation_study import MMFT
from data_depth_free_multi_task import get_loader
from utils import adjust_lr, AvgMeter
from model.ssim_loss import SSIM
import torch.nn as nn



parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=50, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
parser.add_argument('-beta1_gen', type=float, default=0.5,help='beta of Adam for generator')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight_decay')
parser.add_argument('--feat_channel', type=int, default=64, help='reduced channel of saliency feat')

opt = parser.parse_args()
print('Generator Learning Rate: {}'.format(opt.lr_gen))
# build models
generator = MMFT()
generator.cuda()


generator_params = generator.parameters()
generator_optimizer = torch.optim.Adam(generator_params, opt.lr_gen)

## load data
image_root = '/home/asus/Datasets/binary_segmentation/RGBD_SOD_Datasets/NJUD_NLPR_dutrgbd_depth_scale/RGB/'
gt_root = '/home/asus/Datasets/binary_segmentation/RGBD_SOD_Datasets/NJUD_NLPR_dutrgbd_depth_scale/GT/'
depth_root = '/home/asus/Datasets/binary_segmentation/RGBD_SOD_Datasets/NJUD_NLPR_dutrgbd_depth_scale/resize_depth/'
contour_root = '/home/asus/Datasets/binary_segmentation/RGBD_SOD_Datasets/NJUD_NLPR_dutrgbd_depth_scale/sal_contour/'

train_loader = get_loader(image_root, gt_root, depth_root,contour_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)


CE = torch.nn.BCELoss()
mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)
size_rates = [0.75,1,1.25]  # multi-scale training
criterion = nn.BCEWithLogitsLoss().cuda()
criterion_mae = nn.L1Loss().cuda()
criterion_mse = nn.MSELoss().cuda()
criterion_ssim = SSIM(window_size=11,size_average=True)

def ssimmae(pre,gt):
    maeloss = criterion_mae(pre,gt)
    ssimloss = 1-criterion_ssim(pre,gt)
    loss = ssimloss+maeloss
    return loss

def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))
    pred  = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou  = 1-(inter+1)/(union-inter+1)

    return (wbce+wiou).mean()




## linear annealing to avoid posterior collapse
def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed

for epoch in range(1, opt.epoch+1):
    generator.train()
    loss_record = AvgMeter()
    print('Generator Learning Rate: {}'.format(generator_optimizer.param_groups[0]['lr']))

    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            generator_optimizer.zero_grad()
            images, gts, depths,contours, = pack
            # print(index_batch)
            images = Variable(images)
            gts = Variable(gts)
            depths = Variable(depths)
            contours = Variable(contours)
            images = images.cuda()
            gts = gts.cuda()
            depths = depths.cuda()
            contours = contours.cuda()

            # multi-scale training samples
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear',
                                          align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                contours = F.upsample(contours, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                depths = F.upsample(depths, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            b, c, h, w = gts.size()
            target_1 = F.upsample(gts, size=h // 2, mode='nearest')
            target_2 = F.upsample(gts, size=h // 4, mode='nearest')
            target_3 = F.upsample(gts, size=h // 8, mode='nearest')
            target_4 = F.upsample(gts, size=h // 16, mode='nearest')
            target_5 = F.upsample(gts, size=h // 32, mode='nearest')
            contour_1 = F.upsample(contours, size=h // 2, mode='nearest')
            contour_2 = F.upsample(contours, size=h // 4, mode='nearest')
            contour_3 = F.upsample(contours, size=h // 8, mode='nearest')
            contour_4 = F.upsample(contours, size=h // 16, mode='nearest')
            contour_5 = F.upsample(contours, size=h // 32, mode='nearest')
            # surface_norm_1 = functional.interpolate(surface_norm, size=256 // 16, mode='bilinear')
            depth_1 = F.upsample(depths, size=h // 2, mode='bilinear')
            depth_2 = F.upsample(depths, size=h // 4, mode='bilinear')
            depth_3 = F.upsample(depths, size=h // 8, mode='bilinear')
            depth_4 = F.upsample(depths, size=h // 16, mode='bilinear')
            depth_5 = F.upsample(depths, size=h // 32, mode='bilinear')

            sideout_Depth_5, sideout_Contour_5, sideout_Sal_5, sideout_Depth_4, sideout_Contour_4, sideout_Sal_4, sideout_Depth_3, sideout_Contour_3, sideout_Sal_3, sideout_Depth_2, sideout_Contour_2, sideout_Sal_2, sideout_Depth_1, sideout_Contour_1, sideout_Sal_1, output_depth_final, output_contour_final, output_sal_final = generator.forward(
                images)


            loss1 = ssimmae(F.sigmoid(sideout_Depth_5), depth_5)
            loss2 = ssimmae(F.sigmoid(sideout_Depth_4), depth_4)
            loss3 = ssimmae(F.sigmoid(sideout_Depth_3), depth_3)
            loss4 = ssimmae(F.sigmoid(sideout_Depth_2), depth_2)
            loss5 = ssimmae(F.sigmoid(sideout_Depth_1), depth_1)
            loss6 = ssimmae(F.sigmoid(output_depth_final), depth_1)
            loss7 = ssimmae(sideout_Contour_5, depth_5)
            loss8 = ssimmae(sideout_Contour_4, depth_4)
            loss9 = ssimmae(sideout_Contour_3, depth_3)
            loss10 = ssimmae(sideout_Contour_2, depth_2)
            loss11 = ssimmae(sideout_Contour_1, depth_1)
            loss12 = ssimmae(output_contour_final, depth_1)
            loss13 = structure_loss(sideout_Sal_5, target_5)
            loss14 = structure_loss(sideout_Sal_4, target_4)
            loss15 = structure_loss(sideout_Sal_3, target_3)
            loss16 = structure_loss(sideout_Sal_2, target_2)
            loss17 = structure_loss(sideout_Sal_1, target_1)
            loss18 = structure_loss(output_sal_final, target_1)
            sal_loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14 + loss15 + loss16 + loss17 + loss18
            sal_loss.backward()
            generator_optimizer.step()

            if rate == 1:
                loss_record.update(sal_loss.data, opt.batchsize)


        if i % 10 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], gen Loss: {:.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show()))


    adjust_lr(generator_optimizer, opt.lr_gen, epoch, opt.decay_rate, opt.decay_epoch)
    save_path = 'MMFT_RES50_50epoch_batch4/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if epoch % opt.epoch == 0:
        torch.save(generator.state_dict(), save_path + 'Model' + '_%d' % epoch + '_gen.pth')
