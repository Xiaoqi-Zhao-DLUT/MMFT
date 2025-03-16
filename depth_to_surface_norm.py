import cv2
import numpy as np
import os

def surface_normal(input_path, save_path):
    d_im = cv2.imread(input_path, 0)
    d_im = d_im.astype(np.uint8)
    zy, zx = np.gradient(d_im)
    normal = np.dstack((-zx, -zy, np.ones_like(d_im)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n
    normal += 1
    normal /= 2
    normal *= 255
    cv2.imwrite(save_path, normal[:, :, ::-1])

if __name__ == '__main__':
    root_depth = './depth'
    root_surface_normal = './surface_norm'
    img_list = [os.path.splitext(f)[0] for f in os.listdir(root_depth) if f.endswith('.png')]
    for idx, img_name in enumerate(img_list):
        print(os.path.join(root_depth, img_name +'.png'))
        surface_normal(os.path.join(root_depth, img_name +'.png'), os.path.join(root_surface_normal, img_name +'.png'))