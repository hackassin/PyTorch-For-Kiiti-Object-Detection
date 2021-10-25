import os
import sys
import time
import cv2
from PIL import Image,ImageEnhance
import numpy as np
file_dir = os.path.dirname('/workspace/experiment/utils/')
sys.path.append(file_dir)
from helper import fetch_bbox_lb, gen_rand_num, resize_img_bbox
from helper import save_newimgdata, conv2f
from skimage import color
from tqdm.notebook import tqdm_notebook as tqnb



# For resizing image
def resize_dataset(df2, newpath, dim):
    # dim = (960,544)
    resizeimg_info = {}
    newimgdir, newlbdir = newpath
    for item,bar in zip(df2[['image_path', 'label_path']].values,tqnb(range(len(df2)),desc='Resizing Images')):
        resize_imgname = os.path.basename(item[0])[:-4] + '_resized'
        resize_lbboxnm = os.path.basename(item[1])[:-4] + '_resized'
        resizeimg_info['image_arr'], resizeimg_info['img_bboxes'] = resize_img_bbox((item[0],item[1]),dim, 
                                                                                    df2.loc[df2.image_path==item[0],'bboxes'].values[0])
        resizeimg_info['classes'] = df2.loc[df2.image_path==item[0],'classes'].values[0] 
        resizeimg_info['newimg_path'] = newimgdir + resize_imgname
        resizeimg_info['newlabel_path'] = newlbdir + resize_lbboxnm
        resizeimg_info['origlabel_path'] = item[1]
        # print('bboxes before update: ', df2.loc[list(np.where(df2["image_path"] == item[0])[0])[0], 'bboxes'])
        df2.loc[list(np.where(df2["image_path"] == item[0])[0])[0], 'bboxes'] = resizeimg_info['img_bboxes']
        # print('bboxes after update: ', df2.loc[list(np.where(df2["image_path"] == item[0])[0])[0], 'bboxes'])
        save_newimgdata(resizeimg_info)
        time.sleep(0.000015)
    return df2

# For Spatial Augmentation
class Spatial_Aug:
    def __init__(self, image_path, label_path, dim = (512,512), truncated_boxes = None):
        # self.image_arr = cv2.imread(image_path)/255
        # Convert to rgb float32, as opencv only works with float32
        # self.image_arr = cv2.cvtColor(self.image_arr.astype("float32"), cv2.COLOR_BGR2RGB)
        # self.bboxes, _ = fetch_bbox_lb(label_path)
        if truncated_boxes is None:
            self.image_arr, self.bboxes = resize_img_bbox((image_path,label_path),dim)
        else: self.image_arr, self.bboxes = resize_img_bbox((image_path,label_path),dim, truncated_boxes)    
        
    def hflip (self):
        # Fetching image width
        image_width = np.array(self.image_arr.shape[:2])[::-1][0]
        # Horizontally stacking image width for further perusal
        image_width = np.hstack((image_width, image_width))
        # Reversing the image co-ordinates in the 1st dimension
        # i.e Hz_flipping the image
        self.image_arr = self.image_arr[:,::-1,:]
        # Calculating the new distance from origin
        self.bboxes[:,[0,2]] = image_width - self.bboxes[:,[0,2]]
        # Interchanging  x1,x2 co-ordinates
        # Method 1: Calculating the bbox width
        box_width = abs(self.bboxes[:,0] - self.bboxes[:,2])
        self.bboxes[:,0] -= box_width
        self.bboxes[:,2] += box_width
        # Method 2: Swap co-ordinates
        #bbox_x1 = bboxes[:,0]
        #bboxes[:,0] = bboxes[:,2]
        #bboxes[:,1] = bbox_x1
        
        #return self.image_arr,self.bboxes
        return self
    
    def translate (self, TxTy = (8,8)):
        rows,cols = self.image_arr.shape[:2]
        # Making a transition matrix
        # Here the shift is Tx=Ty=8
        Tx,Ty = TxTy
        rand_x = conv2f(gen_rand_num(0,Tx))
        rand_y = conv2f(gen_rand_num(0,Ty))
        M = np.float32([[1,0,rand_x],[0,1,rand_y]])
        self.image_arr = cv2.warpAffine(self.image_arr,M,(cols,rows))
        self.bboxes[:,[0,2]] += rand_x
        self.bboxes[:,[1,3]] += rand_y
        return self
    
# For Color Augmentation
class Color_Aug:
    def __init__(self, image_path, label_path, dim = (512,512), truncated_boxes = None):
        # Fetch values and resize images    
        if truncated_boxes is None:
            self.image_arr, self.bboxes = resize_img_bbox((image_path,label_path),dim)
        else: self.image_arr, self.bboxes = resize_img_bbox((image_path,label_path),dim, truncated_boxes)
    
    # definition to rotate hue of image
    def hue_rotate(self, hue_rot_max=0.5):    
        hsv = color.rgb2hsv(color.gray2rgb(self.image_arr))
        rand_hue_rot = gen_rand_num(0.1,hue_rot_max)
        #print("rand_hue_rot = ",rand_hue_rot)
        hsv[:, :, 0] = rand_hue_rot # adjust hue
        hsv[:, :, 1] = 1  # Turn up the saturation
        #return color.hsv2rgb(hsv)
        self.image_arr = color.hsv2rgb(hsv)
        return self
    
    """# defintion to adjust saturation
    def saturation(self, factor):
        rand_fact = gen_rand_num(0,factor)
        converter = ImageEnhance.Color(self.image)
        return converter.enhance(rand_fact)"""
    
    def saturation (self, factor=1.5):
        #imghsv = color.rgb2hsv(self.image_arr)
        imghsv = cv2.cvtColor(self.image_arr.astype("float32"), cv2.COLOR_BGR2HSV)
        (h, s, v) = cv2.split(imghsv)
        factor = gen_rand_num(0.1,factor)
        # print("Random Factor = ",factor)
        s = s*factor
        s = np.clip(s,0,1)
        imghsv = cv2.merge([h,s,v])
        imgrgb = cv2.cvtColor(imghsv.astype("float32"), cv2.COLOR_HSV2BGR)
        self.image_arr = imgrgb
        return self
    
    """def contrast_scale(self, factor):
        factor = gen_rand_num(0.2,factor)
        imghsv = cv2.cvtColor(self.image_arr.astype("float32"), cv2.COLOR_BGR2HSV)
        imghsv = cv2.convertScaleAbs(self.image_arr, alpha=factor, beta=0)
        imgrgb = cv2.cvtColor(imghsv.astype("float32"), cv2.COLOR_HSV2BGR)
        return imgrgb"""
    
    def brightness (self, factor=1.5):
        imghsv = cv2.cvtColor(self.image_arr.astype("float32"), cv2.COLOR_BGR2HSV)
        (h, s, v) = cv2.split(imghsv)
        factor = gen_rand_num(0.8,factor)
        # print("Random Factor = ",factor)
        v = v*factor
        v = np.clip(v,0,1)
        imghsv = cv2.merge([h,s,v])
        imgrgb = cv2.cvtColor(imghsv.astype("float32"), cv2.COLOR_HSV2BGR)
        self.image_arr = imgrgb
        return self           