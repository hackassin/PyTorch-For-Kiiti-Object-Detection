# +
# Helper functions
# definition to readfile

import os
import cv2
import numpy as np

def readfile(path):
    file = open(path, 'r')
    return file

# defintion to fetch bounding boxes
def fetch_bboxes(path):
    file = readfile(path)
    lines = file.readlines()
    # print(lines)
    bboxes = np.empty([len(lines),5], dtype=float)
    
    for i,line in enumerate(lines):
        # print(line)
        line = line.split()[0] + ' ' + ' '.join(line.split()[4:8])
        line = line.split()
        bboxes[i,:4] = line[1:]
    return bboxes

def fetch_classes(path):
    file = readfile(path)
    lines = file.readlines()
    # print(lines)
    classes = []
    
    for i,line in enumerate(lines):
        # print(line)
        item = line.split()[0]
        classes.append(item)
    return classes

# defintion to fetch a random number
def gen_rand_num(begin,end):
    num = random.uniform(begin,end)
    return num

# definition to write augmented values to label file
def save_newbbox(newlabel_path,orig_path,bboxes):
    file_r = readfile(orig_path)
    lines = file_r.readlines()
    labels = []
    
    for i,line in enumerate(lines):
        #line[4:8] = '200 300 400 500'
        line = line.split()
        line[4:8] = [str(bbox) for bbox in bboxes[i,:4]]
        line = ' '.join(line)
        line = line + '\n'
        labels.append(line)
    file_w = open(newlabel_path + '.txt', 'w')
    file_w.writelines(labels)
    
# defintion to save augmented image    
def save_newimg(path, img_arr):    
    #image = Image.fromarray(conv2uint8(img_arr))
    #image.save(path + '.jpg')
    cv2.imwrite(path + '.jpg', img_arr)
    
# method to call save_augbbox and save_augimg
def save_newimgdata(image_info):
    save_newimg(image_info['newimg_path'] ,image_info['image_arr'])
    save_newbbox(image_info['newlabel_path'],image_info['origlabel_path'],
                 image_info['img_bboxes'])
    
def conv2uint8(image_arr):
    image_arr = cv2.cvtColor(image_arr.
                             astype("float32"), cv2.COLOR_BGR2RGB)
    return (image_arr * 255).astype(np.uint8)

# Resize images to (960,544)
def resize_img_bbox(img_lb_tupl,dim):
    imgf, labelf = img_lb_tupl
    width, height = dim
    # resize image
    #img = Image.open(imgf)
    #img_arr = np.array(image)
    img_arr = cv2.imread(imgf, cv2.IMREAD_UNCHANGED)
    h0 = img_arr.shape[0]
    w0 = img_arr.shape[1]
    # img = img.resize((width,height))
    img_arr =  cv2.resize(img_arr,(width,height))
    # img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    print("Updated dimensions: ", img_arr.shape)
    # Scale factor in x and y axis
    # of resized image
    scale_x = height/h0
    scale_y = width/w0
    # resize bbox
    bboxes = fetch_bboxes(labelf)
    bboxes[:,[0,2]] = bboxes[:,[0,2]] * scale_x
    bboxes[:,[1,3]] = bboxes[:,[1,3]] * scale_y
    return img_arr,bboxes

def imlabel(impath, labels_path):
    #impath = 'data/kitti/custom_annotated/annotated_images/'
    #labels_path = 'data/kitti/custom_annotated/labels/'
    images = sorted(os.listdir(impath))
    labels = sorted(os.listdir(labels_path))
    imlabel_list = []
    for entity in zip(images, labels):
        entity = list(entity)
        entity[0] = impath + entity[0]
        entity[1] = labels_path + entity[1]
        imlabel_list.append(tuple(entity))
    return imlabel_list
