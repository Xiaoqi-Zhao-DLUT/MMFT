import torch.nn.parallel
# from mater import get_FM
import matplotlib.pyplot as plt
plt.set_cmap("jet")
import torch.nn.parallel
import PIL.Image as Image
import numpy as np
import torch
import os



def eva12(salpath, gtpath,ignore_zero=True):
    gtdir = gtpath
    depdir = salpath
    files = os.listdir(gtdir)
    eps = np.finfo(float).eps

    delta1_accuracy = 0
    delta2_accuracy = 0
    delta3_accuracy = 0
    rmse_linear_loss = 0
    rmse_log_loss = 0
    abs_relative_difference_loss = 0
    squared_relative_difference_loss = 0

    for i, name in enumerate(files):
        if not os.path.exists(gtdir + name):
            print(gtdir + name, 'does not exist')
        gt = Image.open(gtdir + name)

        gt = np.array(gt, dtype=np.uint8)
        gt = 255-gt


        gt = (gt - gt.min()) / (gt.max() - gt.min() + eps)


        gt = torch.from_numpy(gt).float()



        pred = Image.open(depdir + name)


        pred = pred.convert('L')
        pred = pred.resize((np.shape(gt)[1], np.shape(gt)[0]))
        pred = np.array(pred, dtype=np.float)
        pred=255-pred



        pred = (pred - pred.min()) / (pred.max() - pred.min() + eps)

        pred = torch.from_numpy(pred).float()
     

        if len(pred.shape) != 2:
           pred= pred[:, :, 0]
        if len(gt.shape) != 2:
           gt= gt[:, :, 0]
        if ignore_zero:
            pred[gt == 0] = 0.0




        delta1_accuracy += threeshold_percentage(pred, gt, 1.25)
        delta2_accuracy += threeshold_percentage(pred, gt, 1.25 * 1.25)
        delta3_accuracy += threeshold_percentage(pred, gt, 1.25 * 1.25 * 1.25)
        rmse_linear_loss += rmse_linear(pred, gt)
        rmse_log_loss += rmse_log(pred, gt)
        abs_relative_difference_loss += abs_relative_difference(pred, gt)
        squared_relative_difference_loss += squared_relative_difference(pred, gt)

    delta1_accuracy /= (i + 1)
    delta2_accuracy /= (i + 1)
    delta3_accuracy /= (i + 1)
    rmse_linear_loss /= (i + 1)
    rmse_log_loss /= (i + 1)
    abs_relative_difference_loss /= (i + 1)
    squared_relative_difference_loss /= (i + 1)

    # logger.scalar_summary("coarse validation loss", coarse_validation_loss, epoch)
    # print('\nValidation set: Average loss(Coarse): {:.4f} \n'.format(coarse_validation_loss))
    print(
        '    {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}'.format(
            delta1_accuracy, delta2_accuracy, delta3_accuracy, rmse_linear_loss,
            rmse_log_loss,
            abs_relative_difference_loss, squared_relative_difference_loss))

def threeshold_percentage(output, target, threeshold_val):
    w=target.shape[0]
    h=target.shape[1]
    output = output.view(1, 1, w, h)
    target = target.view(1, 1, w, h)

    d1 = torch.exp(output) / torch.exp(target)
    d2 = torch.exp(target) / torch.exp(output)

    # d1 = output/target
    # d2 = target/output
    max_d1_d2 = torch.max(d1, d2)
    zero = torch.zeros(output.shape[0], output.shape[1], output.shape[2], output.shape[3])
    one = torch.ones(output.shape[0], output.shape[1], output.shape[2], output.shape[3])
    bit_mat = torch.where(max_d1_d2.cpu() < threeshold_val, one, zero)
    count_mat = torch.sum(bit_mat, (1, 2, 3))
    threeshold_mat = count_mat / (output.shape[2] * output.shape[3])
    return threeshold_mat.mean()


def rmse_linear(output, target):
    w=target.shape[0]
    h=target.shape[1]
    output = output.view(1, 1, w, h)
    target = target.view(1, 1, w, h)
    # output = output.view(1, 1, 256, 256)
    # target = target.view(1, 1, 256, 256)
    actual_output = torch.exp(output)
    actual_target = torch.exp(target)
    # actual_output = output
    # actual_target = target
    diff = actual_output - actual_target
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (1, 2, 3)) / (output.shape[2] * output.shape[3])
    rmse = torch.sqrt(mse)
    return rmse.mean()


def rmse_log(output, target):
    w=target.shape[0]
    h=target.shape[1]
    output = output.view(1, 1, w, h)
    target = target.view(1, 1, w, h)
    # output = output.view(1, 1, 256, 256)
    # target = target.view(1, 1, 256, 256)
    diff = output - target
    # diff = torch.log(output) - torch.log(target)
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (1, 2, 3)) / (output.shape[2] * output.shape[3])
    rmse = torch.sqrt(mse)
    return mse.mean()


def abs_relative_difference(output, target):
    w=target.shape[0]
    h=target.shape[1]
    output = output.view(1, 1, w, h)
    target = target.view(1, 1, w, h)
    # output = output.view(1, 1, 256, 256)
    # target = target.view(1, 1, 256, 256)
    actual_output = torch.exp(output)
    actual_target = torch.exp(target)
    # actual_output = output
    # actual_target = target
    abs_relative_diff = torch.abs(actual_output - actual_target) / actual_target
    abs_relative_diff = torch.sum(abs_relative_diff, (1, 2, 3)) / (output.shape[2] * output.shape[3])
    return abs_relative_diff.mean()


def squared_relative_difference(output, target):
    w=target.shape[0]
    h=target.shape[1]
    output = output.view(1, 1, w, h)
    target = target.view(1, 1, w, h)
    # output = output.view(1, 1, 256, 256)
    # target = target.view(1, 1, 256, 256)
    actual_output = torch.exp(output)
    actual_target = torch.exp(target)
    # actual_output = output
    # actual_target = target
    square_relative_diff = torch.pow(torch.abs(actual_output - actual_target), 2) / actual_target
    square_relative_diff = torch.sum(square_relative_diff, (1, 2, 3)) / (output.shape[2] * output.shape[3])
    return square_relative_diff.mean()


def main():
    print("\n evaluating ....")
    eva12(
        salpath='./SIP_depth_pre' + '/',
        gtpath='./SIP_depth_gt' + '/')


if __name__ == '__main__':
    main()
