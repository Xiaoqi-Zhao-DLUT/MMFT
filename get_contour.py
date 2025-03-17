import os
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2

test = './SOD_mask'
to_test = {'edge':test}
save_path = './SOD_contour'
to_pil = transforms.ToPILImage()

for name, root in to_test.items():
    root1 = os.path.join(root)
    img_list = [os.path.splitext(f)[0] for f in os.listdir(root1) if f.endswith('.png')]
    for idx, img_name in enumerate(img_list):
        print('predicting for %s: %d / %d' % (name, idx + 1, len(img_list)))
        img1 = Image.open(os.path.join(root, img_name + '.png')).convert('L')
        img1 = np.array(img1)
        kernel = np.ones((3,3),np.uint8)
        img2 = cv2.erode(img1,kernel)
        img3 = cv2.dilate(img1,kernel)
        img = np.array(img3-img2)
        img[img >= 128] = 255  
        img[img<255] = 0
        Image.fromarray(img).save(os.path.join(save_path, img_name + '.png'))


