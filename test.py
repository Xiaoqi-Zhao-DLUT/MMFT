import torch
import torch.nn.functional as F
import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from model.depth_free_ablation_study import MMFT
from data import test_dataset
import cv2



parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--latent_dim', type=int, default=3, help='latent dim')
parser.add_argument('--feat_channel', type=int, default=64, help='reduced channel of saliency feat')
opt = parser.parse_args()

dataset_path = '/home/asus/Datasets/binary_segmentation/RGBD_SOD_Datasets/'
depth_path = '/home/asus/Datasets/binary_segmentation/RGBD_SOD_Datasets/'

generator = MMFT()
generator.load_state_dict(torch.load('./MMFT.pth'))


generator.cuda()
generator.eval()

transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.Scale(scales=[0.75, 1, 1.25], interpolation='bilinear', align_corners=False),
    ]
)

#
test_datasets = ['NLPR_test']

for dataset in test_datasets:
    save_path = './NLPR_notta/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_root = dataset_path + dataset + '/RGB_test_300/'
    depth_root = dataset_path + dataset + '/GT/'
    test_loader = test_dataset(image_root, depth_root, opt.testsize)
    for i in range(test_loader.size):
        print(i)
        image, depth, HH, WW, name = test_loader.load_data()
        image = image.cuda()
        depth = depth.cuda()
        generator_pred = generator.forward(image)

        ##TTA
        # mask = []
        # for transformer in transforms:  # custom transforms or e.g. tta.aliases.d4_transform()
        #
        #     rgb_trans = transformer.augment_image(image)
        #     generator_pred = generator.forward(rgb_trans)
        #     deaug_mask = transformer.deaugment_mask(generator_pred)
        #     mask.append(deaug_mask)
        #
        # prediction = torch.mean(torch.stack(mask, dim=0), dim=0)
        # prediction = prediction.sigmoid()

        res = generator_pred
        res = F.upsample(res, size=[WW,HH], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path + name+'.png', res)
