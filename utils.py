from model import get
import difflib
import pickle
import numpy as np
import torch

import cv2 as cv
import random
import imgaug as ia
from common import config
from imgaug import augmenters as iaa
from imgaug.augmentables.batches import UnnormalizedBatch
from PIL import Image
ia.seed(2333)


class Dictionary():

    def __init__(self, dict_path):
        f = open(dict_path, 'r')
        lines = f.readlines()
        lines = [l.strip() for l in lines]
        self.word2idx = {}
        self.idx2word = []

        for w in lines:
            self.word2idx[w] = len(self.idx2word)
            self.idx2word.append(w)


    def __getitem__(self, idx):
        return self.idx2word[idx]


def boundary_coor2center_size(box):
    a, b, c, d = box
    x = ((a + c) / 2) / config.width
    y = ((b + d) / 2) / config.height
    w = (c-a) / config.width
    h = (b-d) / config.height
    return [x, y, w, h]


def center_size2boundary_coor(box):
    x, y, w, h = box
    a = int((x - w/2) * config.width)
    b = int((y + h / 2) * config.height)
    c = int((x + w/2) * config.width)
    d = int((y - h/2) * config.height)
    return [a, b, c, d]


def calc_offset(proposal, truth):
    x, y, w, h = truth
    x0, y0, w0, h0 = proposal
    d_x = (x - x0) / w0
    d_y = (y - y0) / h0
    d_w = np.log(w / w0)
    d_h = np.log(h / h0)
    return [d_x, d_y, d_w, d_h]


def apply_offset(proposal, offset):
    x0, y0, w0, h0 = proposal
    d_x, d_y, d_w, d_h = offset
    x = x0 + w0 * d_x
    y = y0 + h0 * d_y
    w = w0 * np.exp(d_w)
    h = h0 * np.exp(d_h)
    return [x, y, w, h]


def get_proposal():
    return [0.5, 0.5, 1.0, 1.0]


def get_augmenter():
    seq = iaa.Sequential([
        iaa.GaussianBlur(sigma=(config.blur_min, config.blur_max)),
        iaa.SaltAndPepper((config.s_p_min, config.s_p_max))
        ])
    return seq


def add_noise(seq, data, batched=True):
    if batched:
        imgs = data.reshape(config.noise_batch, config.minibatch_size // config.noise_batch, config.height, config.width)
        imgs = [[imgs[i][j] for j in range(config.minibatch_size // config.noise_batch)] for i in range(config.noise_batch)]
        img_batches = [UnnormalizedBatch(images=imgs[i]) for i in range(config.noise_batch)]
        img_batches_aug = list(seq.augment_batches(img_batches, background=True))
        imgs_aug = np.stack([np.stack(img_batches_aug[i].images_aug) for i in range(config.noise_batch)]).reshape(config.minibatch_size, 1, config.height, config.width)
    else:
        imgs = data.reshape(-1, config.height, config.width)
        imgs_aug = seq.augment_images(imgs)
        imgs_aug = np.stack([img.reshape(-1, config.height, config.width) for img in imgs_aug])
    return imgs_aug
    


def mergeContour(intervals):
    merged = []
    
    mergedIntervalIndex = []
    for i, higher in enumerate(intervals):
        if not merged:
            merged.append(higher)
            mergedIntervalIndex.append([i])
        else:
            lower = merged[-1]
            # test for intersection between lower and higher:
            # we know via sorting that lower[0] <= higher[0]
            if higher[0] <= lower[1] and higher[1]<=lower[1]:
                upper_bound = max(lower[1], higher[1])
                merged[-1] = (lower[0], upper_bound)
                mergedIntervalIndex[-1].append(i)
                # replace by merged interval
            else:
                merged.append(higher)
                mergedIntervalIndex.append([i])
    
    return mergedIntervalIndex


def addBorder(img, bordersize):
    return cv.copyMakeBorder(img, top=bordersize, bottom=bordersize,
                              left=bordersize, right=bordersize,
                              borderType= cv.BORDER_CONSTANT, 
                              value=[255,255,255] )



def boundingBox(img, imFloodfillInv, name):
    connectivity = 8
    labelnum, _, contours, _ = cv.connectedComponentsWithStats(
        imFloodfillInv, connectivity)
    
    label_range = range(1, labelnum)
    contours = sorted(contours, key = lambda x : x[0])
    bb_img = img.copy()
    
    boxes = []
    for label in range(1,labelnum):
        x,y,w,h,size = contours[label]
        boxes.append([x, y, x+w, y+h])
        
    return boxes


def uyghurTextProcessing(path):
    name = path.split('/')[-1]
    base_image = cv.imread(path, cv.IMREAD_GRAYSCALE)
    gray_image = cv.cvtColor(base_image, cv.IMREAD_GRAYSCALE)
    gray_image = gray_image.astype('uint8')
    Iedge = cv.Canny(gray_image, 100, 200)
    
    #print(image.shape, gray_image.shape)
    #dilation : small size of kernel- can segment between much closer objects and accurately
    #according to experiment and trying several values for kernel:
    #large value of first emlement of kernel such as (15, 3) helps in dilate image
    #vertically. Large value of first element of kernel is useful for
    #creating bounding box for text so that we can include the dots
    #present in text in one Bounding Box. 
    #small values of both elements of kernel such as (3, 3)helps in segmenting and
    #creating bounding box for number 
    kernel = np.ones((15,5), np.uint8)
    img_dilation = cv.dilate(Iedge, kernel, iterations=1)
    
    
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    th, im_th = cv.threshold(img_dilation, 220, 255, 
                              cv.THRESH_BINARY_INV);
    
    # Copy the thresholded image.
    im_floodfill = im_th.copy()
    
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    mask1 = np.zeros((h+10, w+10), np.uint8)
    # Floodfill from point (0, 0)
    cv.floodFill(im_floodfill, mask, (0,0), 255)
    
    # Invert floodfilled image
    im_floodfill_inv = cv.bitwise_not(im_floodfill)
    
    boxes = boundingBox(base_image, im_floodfill_inv, name)
    return base_image, boxes

'''
def sort_box(boxes):
    boxes = sorted(boxes, key=lambda item:item[3])
    
    line = []
    sorted_boxes = []
    for box in boxes:
        if len(line) == 0:
            line.append(box)
        else:
            height_diff1 = line[-1][3] - box[3]
            height_diff2 = line[-1][1] - box[1]
            if height_diff1 <= -20 and np.abs(height_diff2) >= 20:
                sorted_boxes += sorted(line, key=lambda item:item[0])
                line = [box]
            else:
                line.append(box)
    if len(line) > 0:
        sorted_boxes += sorted(line, key=lambda item:item[0])
    assert len(boxes) == len(sorted_boxes)

    return sorted_boxes
'''

def sort_box(boxes):
    boxes = sorted(boxes, key=lambda item:item[3])
    
    line = []
    sorted_boxes = []
    for box in boxes:
        if len(line) == 0:
            line.append(box)
        else:
            height_diff1 = line[-1][3] - box[3]
            height_diff2 = line[-1][1] - box[1]
            if height_diff1 <= -20 and np.abs(height_diff2) >= 20:
                sorted_boxes = sorted(line, key=lambda item:item[0]) + sorted_boxes
                line = [box]
            else:
                line.append(box)
    if len(line) > 0:
        sorted_boxes = sorted(line, key=lambda item:item[0]) + sorted_boxes
    assert len(boxes) == len(sorted_boxes)

    return sorted_boxes


def get_words(path):
    base_image, boxes = uyghurTextProcessing(path)
    boxes = sort_box(boxes)
    pics = []
    for box in boxes:
        a, b, x, y = box
        pics.append(base_image[b:y, a:x])
    pics = [augment(p, config.aug_w, config.aug_h) for p in pics]
    pics = [reshape_pic(p, config.height, config.width) for p in pics]
    
    pic_with_box = np.copy(base_image)
    for box in boxes:
        a, b, x, y = box
        cv.rectangle(pic_with_box, (a, b), (x, y), (128, 1, cv.LINE_AA))
    return base_image, pic_with_box, pics


def augment(pixels, aug_w, aug_h):
    orig_h, orig_w = pixels.shape
    aug_pixels = np.ones((orig_h + 2 * aug_h, orig_w + 2 * aug_w))
    aug_pixels = aug_pixels * 255
    aug_pixels = aug_pixels.astype(np.uint8)
    aug_pixels[aug_h:(aug_h+orig_h), aug_w:(aug_w+orig_w)] = pixels
    return aug_pixels


def reshape_pic(pixels, height, width):
    orig_h, orig_w = pixels.shape
    im = Image.fromarray(pixels, 'L')
    im = im.resize((width, height), Image.ANTIALIAS)
    return np.array(im).astype(np.uint8).reshape(1, height, width)


def load_model(pkl_path):
    net = get()
    net.eval()
    f = open(pkl_path, 'rb')
    ckpt = pickle.load(f)
    f.close()

    param_order = ['conv1:conv1:conv:W', 'conv1:conv1:conv:b',
                   'conv2:conv2:conv:W', 'conv2:conv2:conv:b',
                   'conv3:conv3:conv:W', 'conv3:conv3:conv:b',
                   'conv4:conv4:conv:W', 'conv4:conv4:conv:b',
                   'fc1:fc1:fc:W', 'fc1:fc1:fc:b',
                   'fct:fct:fc:W', 'fct:fct:fc:b', ]
    for i, param in enumerate(net.parameters()):
        if 'fc' not in param_order[i]:
            param.data = torch.from_numpy(ckpt[param_order[i]]).float()
        else:
            param.data = torch.from_numpy(np.transpose(ckpt[param_order[i]])).float()


    return net
