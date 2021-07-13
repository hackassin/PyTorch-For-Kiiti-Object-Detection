# +
# Helper functions
# definition to readfile

import os
import cv2
import numpy as np
import random

def readfile(path):
    file = open(path, 'r')
    return file

def conv2f(val):
    return float("{:.2f}".format(val))
        
# defintion to fetch bounding boxes
def fetch_bbox_lb(path):
    file = readfile(path)
    lines = file.readlines()
    # print(lines)
    bboxes = np.empty([len(lines),4], dtype=float)
    classes = []
    for i,line in enumerate(lines):
        # print(line)
        line = line.split()[0] + ' ' + ' '.join(line.split()[4:8])
        line = line.split()
        bboxes[i,:4] = line[1:]
        classes.append(line[0])    
    
    return bboxes, np.array(classes)

"""def fetch_classes(path):
    file = readfile(path)
    lines = file.readlines()
    # print(lines)
    classes = []
    
    for i,line in enumerate(lines):
        # print(line)
        item = line.split()[0]
        classes.append(item)
    return classes"""

# defintion to fetch a random number
def gen_rand_num(begin,end):
    num = random.uniform(begin,end)
    return num

# definition to write augmented values to label file
def save_newbbox(newlabel_path,orig_path,bboxes,classes):
    # file_r = readfile(orig_path)
    # lines = file_r.readlines()
    labels = []
     # Create empty line with 5 fields
    for i,bbox in enumerate(bboxes):
        #line[4:8] = '200 300 400 500'
        # line = line.split()
        line = []
        line.append(classes[i])
        line.append(' '.join([str(conv2f(bbox)) for bbox in bboxes[i,:]]))
        line = ' '.join(line)
        line = line + '\n'
        labels.append(line)
    file_w = open(newlabel_path + '.txt', 'w')
    file_w.writelines(labels)
    
# defintion to save augmented image    
def save_newimg(path, img_arr):    
    #image = Image.fromarray(conv2uint8(img_arr))
    #image.save(path + '.jpg')
    cv2.imwrite(path + '.jpg', cv2.cvtColor(img_arr*255, cv2.COLOR_RGB2BGR))
    
# method to call save_augbbox and save_augimg
def save_newimgdata(image_info):
    save_newimg(image_info['newimg_path'] ,image_info['image_arr'])
    save_newbbox(image_info['newlabel_path'],image_info['origlabel_path'],
                 image_info['img_bboxes'], image_info['classes'])
    
def conv2uint8(image_arr):
    """image_arr = cv2.cvtColor(image_arr.
                             astype("float32"), cv2.COLOR_BGR2RGB)"""
    return (image_arr * 255).astype(np.uint8)

# Resize images to dimensions specified
def resize_img_bbox(img_lb_tupl,dim, trunc_boxes=None):
    imgf, labelf = img_lb_tupl
    width, height = dim
    # resize image
    img_arr = cv2.imread(imgf)/255
    # convert to rgb
    img_arr = cv2.cvtColor(img_arr.astype('float32'), cv2.COLOR_BGR2RGB)
    h0 = img_arr.shape[0]
    w0 = img_arr.shape[1]
    # img = img.resize((width,height))
    img_arr =  cv2.resize(img_arr,(width,height))
    # img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    # print("Updated dimensions: ", img_arr.shape)
    # Scale factor in x and y axis
    # of resized image
    scale_x = conv2f(height/h0)
    scale_y = conv2f(width/w0)
    # resize bbox
    if trunc_boxes is not None:
        # print('Truncated bboxes.any: ', trunc_boxes)
        bboxes = trunc_boxes
    else:
        # print('Truncated bboxes: ', trunc_boxes)
        bboxes, _ = fetch_bbox_lb(labelf)  
    bboxes[:,[0,2]] = bboxes[:,[0,2]] * scale_y
    bboxes[:,[1,3]] = bboxes[:,[1,3]] * scale_x
    return img_arr,bboxes

# Returns image and label path tuple
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

# Draw bbox on image
# Ref from Paperspace blog
def draw_bbox(img, cords, labels, color = None):
    """Draw the rectangle on the image
    
    Parameters
    ----------
    
    img : numpy.ndarray
        numpy image 
    
    cords: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
        
    Returns
    -------
    
    numpy.ndarray
        numpy image with bounding boxes drawn on it
        
    """    
    img = img.copy()
    
    cords = cords.reshape(-1,4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if not color:
        color = [255,255,255]
    for i,cord in enumerate(cords):
        
        pt1, pt2 = (cord[0], cord[1]) , (cord[2], cord[3])
                
        pt1 = int(pt1[0]), int(pt1[1])
        pt2 = int(pt2[0]), int(pt2[1])
        label_pt = int(cord[0]+10),int(cord[1]+10)
        img = cv2.rectangle(img, pt1, pt2, color, int(max(img.shape[:2])/200))
        img = cv2.putText(img, labels[i],label_pt, font, 0.5,(255,255,255),1,cv2.LINE_AA)
    return img