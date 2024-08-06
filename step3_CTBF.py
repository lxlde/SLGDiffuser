import os
import cv2
import numpy as np
from PIL import Image

def gen_foreground(H,W,img_path, mask_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (W, H))
    mask = cv2.imread(mask_path, 0)
    mask = cv2.resize(mask,(W,H))
    # mask[mask<50] = 0
    # mask[mask>=50] = 255
    height, width, channel = img.shape
    b, g, r = cv2.split(img)
    res = np.zeros((4, height, width), dtype=img.dtype)
    res[0][0:height, 0:width] = b
    res[1][0:height, 0:width] = g
    res[2][0:height, 0:width] = r
    res[3][0:height, 0:width] = mask
    res =  cv2.merge(res)
    return res

def get_mask_size(mask):
    xy = np.argwhere(mask > 100)
    left_top = xy[0]
    right_bottom = xy[np.size(xy,0)-1]
    return left_top,right_bottom

def add_text2background(background_path,background_mask_path,text_path,text_mask_path,save_root_normal2):
    mask = cv2.imread(background_mask_path, 0)
    mask = cv2.resize(mask, (512, 512))
    background = cv2.imread(background_path)
    left_top, right_bottom = get_mask_size(mask) 
    H = right_bottom[0] - left_top[0]
    W = right_bottom[1] - left_top[1]
    text = gen_foreground(H, W,text_path, text_mask_path)
    background_gen = np.pad(background, pad_width=((0, 0), (0, 0), (0, 1)), mode='constant',
                            constant_values=((0, 0), (0, 0), (0, 0)))
    b, g, r, d = cv2.split(text)
    background_gen[left_top[0]:right_bottom[0], left_top[1]:right_bottom[1], 0] = b
    background_gen[left_top[0]:right_bottom[0], left_top[1]:right_bottom[1], 1] = g
    background_gen[left_top[0]:right_bottom[0], left_top[1]:right_bottom[1], 2] = r
    background_gen[left_top[0]:right_bottom[0], left_top[1]:right_bottom[1], 3] = d

    result = background
    for i, x in np.ndenumerate(mask):
        if x == 0:
            result[i[0], i[1]] = background[i[0], i[1]]
        else:
            temp = background_gen[i[0], i[1], 0:3] * (background_gen[i[0], i[1], 3] / 255) + background[i[0], i[1]] * (
                        1 - background_gen[i[0], i[1], 3] / 255)
            result[i[0], i[1]] = temp.astype(np.int16)
    cv2.imwrite(save_root_normal2, result)
if __name__ == "__main__":
    path = 'output/'
    for i in os.listdir(path+"background"):
        background_path =path+"background/" + i
        background_mask_path = path+ "background_mask/" + i
        topng = i.replace("jpg","png")
        text_path = path+"text/" + topng
        text_mask_path = path+"text_mask/" + topng
        save_path_normal1 = path+"output/" + i
        save_path_normal2 = path + "output/" + i.split('.')[0]+'_blur.jpg'
        add_text2background(background_path,background_mask_path,text_path,text_mask_path,save_path_normal2)



