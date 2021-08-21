# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import time
import glob
import torch
import torch.nn as nn
import numpy as np, pandas as pd
import torchvision
from torchvision import models
import torch.nn.functional as F
import torch.nn.init as init
from utils import helper as hp
from utils import ssd_helper as ssdhp
import matplotlib.pyplot as plt
import torch.utils.data as data
import os, cv2, itertools
from tqdm.notebook import tqdm_notebook as tqnb
from PIL import Image
from sklearn.model_selection import train_test_split

# ### Prepare Dataset

# +
impath = 'data/kitti/integration/resized_aug/training/images_ssd300/'
labels_path = 'data/kitti/integration/resized_aug/training/labels_ssd300/'
imlabel_list = hp.imlabel(impath, labels_path)
df = pd.DataFrame(columns=['image_path', 'label_path', 'classes', 'bboxes'])
# df = pd.DataFrame()
# df['image_path'] = imlabel_list[:][0]
# df['label_path'] = imlabel_list[:][1]
# df.head()

for item, bar in zip(imlabel_list, tqnb(range(len(imlabel_list)), desc='Preparing dataframe')):
    # print(item)
    # bboxes = helper.fetch_bboxes(item[1]).tolist()
    bboxes, classes = hp.fetch_bbox_lb(item[1], augflag=True)
    df = df.append({'image_path': item[0], 'label_path': item[1],
                    'classes': classes, 'bboxes': bboxes}, ignore_index=True)
print(df.head())

# +
class_dict = {'car': 0,
              'bus': 1, 'motorbike': 2,
              'bicycle': 3, 'cat': 4,
              'person': 5}

def num_label(class_dict, classes):
    classes = [class_dict[x] for x in classes]
    return np.array(classes)

df['classes'] = df.apply(lambda x: num_label(class_dict, x['classes']), axis=1)
print(df.head())

# #### Dataset Class

# + endofcell="--"
"""VOC_LABELS = ('aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person',
        'pottedplant','sheep','sofa','train','tvmonitor',)"""


# DET_LABELS = ['car', 'bus', 'motorbike', 'bicycle', 'cat', 'person']

def det_collate(batch):
    imgs = []
    boxes = []
    classes = []
    for sample in batch:
        imgs.append(sample[0])
        boxes.append(sample[1])
        classes.append(sample[2])
    # return np.array(imgs), np.array(boxes), np.array(classes)
    return torch.transpose(torch.tensor(imgs), 3, 1), np.array(boxes, dtype=object), np.array(classes, dtype=object)

def normalize(im_arr):
    # Normalizes image with imagenet stats."""
    imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]], dtype=np.float32)
    return (im_arr - imagenet_stats[0])/imagenet_stats[1]

class ObjDetDataset(data.Dataset):

    def __init__(self, paths, bboxes, classes):
        self.paths = paths.values
        self.bboxes = bboxes.values
        self.classes = classes.values

    def __getitem__(self, index):
        # img_path = self.data.loc[index,'image_path']
        impath = self.paths[index]
        image = cv2.cvtColor(cv2.imread(impath).astype('float32'),
                             cv2.COLOR_BGR2RGB) / 255
        image = normalize(image)
        height, width = image.shape[0:2]
        boxes = self.bboxes[index]
        boxes[:,0::2] /= width
        boxes[:,1::2] /= height
        labels = self.classes[index]
        return image, boxes, labels

    def __len__(self):
        return len(self.paths)

# Training Dataset Split
X = df.image_path
y = df[['bboxes', 'classes']]
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# ### Building ResNet18

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = self.relu2(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


#  ### Building SSD with Resnet18
# For feature normalization
class L2Norm(nn.Module):

    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = torch.sqrt(x.pow(2).sum(dim=1, keepdim=True)) + self.eps
        x = torch.div(x, norm)
        x = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return x

    # Auxillary Convolution Layers

def extra(net):
    layers = []
    if net == 'ssd_300':
        conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1)
        conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        conv9_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1)
        conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        conv10_1 = nn.Conv2d(256, 128, kernel_size=1, stride=1)
        conv10_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
        conv11_1 = nn.Conv2d(256, 128, kernel_size=1)
        conv11_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1)

        layers = [conv8_1, conv8_2, conv9_1, conv9_2, conv10_1, conv10_2, conv11_1, conv11_2]

    elif net == 'ssd_512':
        conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1)
        conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        conv9_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1)
        conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        conv10_1 = nn.Conv2d(256, 128, kernel_size=1, stride=1)
        conv10_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
        conv11_1 = nn.Conv2d(256, 128, kernel_size=1)
        conv11_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
        conv12_1 = nn.Conv2d(256, 128, kernel_size=1)
        conv12_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
        layers = [conv8_1, conv8_2, conv9_1, conv9_2, conv10_1,
                  conv10_2, conv11_1, conv11_2, conv12_1, conv12_2]

    return layers

# Creating/Modifying location and confidence layers
def feature_extractor(ver, extral, bboxes, num_classes):
    loc_layers = []
    conf_layers = []

    if ver == 'RES18_SSD':
        loc_layers += [nn.Conv2d(128, bboxes[0] * 4, kernel_size=3, padding=1)]
        loc_layers += [nn.Conv2d(256, bboxes[1] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(128, bboxes[0] * num_classes, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, bboxes[1] * num_classes, kernel_size=3, padding=1)]

    for k, v in enumerate(extral[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, bboxes[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, bboxes[k]
                                  * num_classes, kernel_size=3, padding=1)]

    return loc_layers, conf_layers


class RES18_SSD(nn.Module):

    def __init__(self, num_classes, bboxes, pretrained=None, net='ssd_300'):
        super(RES18_SSD, self).__init__()

        self.ver = 'RES18_SSD'
        self.num_classes = num_classes
        self.bboxes = bboxes
        self.extra_list = extra(net)
        self.loc_layers_list, self.conf_layers_list = feature_extractor(self.ver, self.extra_list, self.bboxes,
                                                                        self.num_classes)
        self.L2Norm = L2Norm(128, 6)

        resnet = ResNet18()
        if pretrained:
            net = torch.load('./weights/resnet18-5c106cde.pth')
            print('Hold your horses! ResNet18 Pre-Trained Model Loading ^_^')
            resnet.load_state_dict(net)

        self.res = nn.Sequential(*list(resnet.children())[:-2],
            #*list(resnet.children())[:-2],
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.extras = nn.ModuleList(self.extra_list)
        self.loc = nn.ModuleList(self.loc_layers_list)
        self.conf = nn.ModuleList(self.conf_layers_list)

    #  xavier initialization
    #         layers = [self.extras, self.loc, self.conf]
    #         print(self.vgg)
    #         for i in layers:
    #             for m in i.modules():
    #                 if isinstance(m, nn.Conv2d):
    #                     nn.init.xavier_uniform_(m.weight)
    #                     nn.init.zeros_(m.bias)

    def forward(self, x):
        source = []
        loc = []
        conf = []
        res_source = [5,6]
        for i, v in enumerate(self.res):
            # print("In forward: x.shape", x.shape)
            x = v(x)
            if i in res_source:
                if i == 5:
                    s = self.L2Norm(x)
                else:
                    s = x
                source.append(s)

        for i, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if i % 2 == 1:
                source.append(x)

        for s, l, c in zip(source, self.loc, self.conf):
            loc.append(l(s).permute(0, 2, 3, 1).contiguous())
            conf.append(c(s).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)
        return loc, conf


'''
The corresponding anchors are generated by the size of the feature map. Because the multi-scale feature map is used in the ssd, 
for the convenience of implementation, we usually normalize the generated anchor (cx, cy, w, h) to (0, 1) by the size of the feature map 
where it is located
'''


class MultiBoxEncoder(object):

    def __init__(self, hparam):
        self.variance = hparam.variance
        default_boxes = list()
        # Generate center co-ordinates
        for k in range(len(hparam.grids)):
            for v, u in itertools.product(range(hparam.grids[k]), repeat=2):
                cx = (u + 0.5) * hparam.steps[k]
                cy = (v + 0.5) * hparam.steps[k]

                s = hparam.sizes[k]
                default_boxes.append((cx, cy, s, s))

                s = np.sqrt(hparam.sizes[k] * hparam.sizes[k + 1])
                default_boxes.append((cx, cy, s, s))

                s = hparam.sizes[k]
                for ar in hparam.aspect_ratios[k]:
                    default_boxes.append(
                        (cx, cy, s * np.sqrt(ar), s / np.sqrt(ar)))
                    default_boxes.append(
                        (cx, cy, s / np.sqrt(ar), s * np.sqrt(ar)))

        default_boxes = np.clip(default_boxes, a_min=0, a_max=1)
        self.default_boxes = np.array(default_boxes)

    def encode(self, boxes, labels, threshold=0.5):

        if len(boxes) == 0:
            return (
                np.zeros(self.default_boxes.shape, dtype=np.float32),
                np.zeros(self.default_boxes.shape[:1], dtype=np.int32))

        iou = bbox_iou(point_form(self.default_boxes), boxes)

        # Retrieve index having max iou
        gt_idx = iou.argmax(axis=1)
        iou = iou.max(axis=1)
        # added to avoid float obj attr error for np.log
        boxes = boxes[gt_idx].astype('float32')
        labels = labels[gt_idx]

        loc = np.hstack((
            ((boxes[:, :2] + boxes[:, 2:]) / 2 - self.default_boxes[:, :2]) /
            (self.variance[0] * self.default_boxes[:, 2:]),
            np.log((boxes[:, 2:] - boxes[:, :2]) / self.default_boxes[:, 2:]) /
            self.variance[1]))

        conf = 1 + labels
        conf[iou < threshold] = 0

        return loc.astype(np.float32), conf.astype(np.int32)

    def decode(self, loc):

        boxes = np.hstack((
            self.default_boxes[:, :2] +
            loc[:, :2] * self.variance[0] * self.default_boxes[:, 2:],
            self.default_boxes[:, 2:] * np.exp(loc[:, 2:] * self.variance[1])))
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]

        return boxes

# +
def bbox_iou(box_a, box_b):
    m = box_a.shape[0]
    n = box_b.shape[0]
    tl = np.maximum(box_a[:, None, :2], box_b[None, :, :2]) # top-left
    br = np.minimum(box_a[:, None, 2:], box_b[None, :, 2:]) # bottom-right
    wh = np.maximum(br - tl, 0) # problem was here
    inner = wh[:, :, 0] * wh[:, :, 1]
    a = box_a[:, 2:] - box_a[:, :2]
    b = box_b[:, 2:] - box_b[:, :2]
    a = a[:, 0] * a[:, 1]
    b = b[:, 0] * b[:, 1]
    a = a[:, None]
    b = b[None, :]
    return inner / (a + b - inner)


def nms(boxes, score, threshold=0.4):
    sort_ids = np.argsort(score)
    pick = []
    while len(sort_ids) > 0:
        i = sort_ids[-1]
        pick.append(i)
        if len(sort_ids) == 1:
            break
        sort_ids = sort_ids[:-1]
        box = boxes[i].reshape(1, 4)
        ious = bbox_iou(box, boxes[sort_ids]).reshape(-1)
        sort_ids = np.delete(sort_ids, np.where(ious > threshold)[0])

    return pick


def detect(locations, scores, nms_threshold, gt_threshold):
    scores = scores[:, 1:]
    keep_boxes = []
    keep_confs = []
    keep_labels = []

    for i in range(scores.shape[1]):
        mask = scores[:, i] >= gt_threshold
        label_scores = scores[mask, i]
        label_boxes = locations[mask]
        if len(label_scores) == 0:
            continue

        pick = nms(label_boxes, label_scores, threshold=nms_threshold)
        label_scores = label_scores[pick]
        label_boxes = label_boxes[pick]

        keep_boxes.append(label_boxes.reshape(-1))
        keep_confs.append(label_scores)
        keep_labels.extend([i] * len(label_scores))

    if len(keep_boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    keep_boxes = np.concatenate(keep_boxes, axis=0).reshape(-1, 4)

    keep_confs = np.concatenate(keep_confs, axis=0)
    keep_labels = np.array(keep_labels).reshape(-1)

    return keep_boxes, keep_confs, keep_labels


def point_form(boxes):
    tl = boxes[:, :2] - boxes[:, 2:] / 2
    br = boxes[:, :2] + boxes[:, 2:] / 2

    return np.concatenate([tl, br], axis=1)


def hard_negtives(logits, labels, pos, neg_ratio):
    num_batch, num_anchors, num_classes = logits.shape
    logits = logits.view(-1, num_classes)
    labels = labels.view(-1)

    losses = F.cross_entropy(logits, labels, reduction='none')

    losses = losses.view(num_batch, num_anchors)

    losses[pos] = 0

    loss_idx = losses.argsort(1, descending=True)
    rank = loss_idx.argsort(1)

    num_pos = pos.long().sum(1, keepdim=True)
    num_neg = torch.clamp(neg_ratio * num_pos, max=pos.shape[1] - 1)  # (batch, 1)
    neg = rank < num_neg.expand_as(rank)

    return neg


class MultiBoxLoss(nn.Module):

    def __init__(self, num_classes=6, neg_ratio=3):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.neg_ratio = neg_ratio

    def forward(self, pred_loc, pred_label, gt_loc, gt_label):
        num_batch = pred_loc.shape[0]

        pos_idx = gt_label > 0
        # print('pred_loc.shape: ', pred_loc.shape)
        # print("pos_idx.shape: ", pos_idx.shape, 'pos_idx.unsqueeze(2).shape: ', pos_idx.unsqueeze(2).shape)
        pos_loc_idx = pos_idx.unsqueeze(2).expand_as(pred_loc)
        # Prediction Location Position
        pred_loc_pos = pred_loc[pos_loc_idx].view(-1, 4)
        # Ground Truth Location Position
        gt_loc_pos = gt_loc[pos_loc_idx].view(-1, 4)

        loc_loss = F.smooth_l1_loss(pred_loc_pos, gt_loc_pos, reduction='sum')

        logits = pred_label.detach()
        labels = gt_label.detach()
        neg_idx = hard_negtives(logits, labels, pos_idx, self.neg_ratio)  # neg (batch, n)

        pos_cls_mask = pos_idx.unsqueeze(2).expand_as(pred_label)
        neg_cls_mask = neg_idx.unsqueeze(2).expand_as(pred_label)

        conf_p = pred_label[(pos_cls_mask + neg_cls_mask).gt(0)].view(-1, self.num_classes)
        target = gt_label[(pos_idx + neg_idx).gt(0)]

        cls_loss = F.cross_entropy(conf_p, target, reduction='sum')
        N = pos_idx.long().sum()

        loc_loss /= N
        cls_loss /= N

        return loc_loss, cls_loss


# +
# Hyper-parameter config
class Config:
    # class + 1
    num_classes = 7
    # learning rate
    lr = 0.001
    # ssd paper = 32
    batch_size = 16
    momentum = 0.9
    weight_decay = 0.0005
    # 40k + 10k = 116 epoc
    epochs = 1
    # pre-train VGG root
    # The resnet pre-train model is in lib.res-model...
    save_folder = './weights/'
    # basenet = 'vgg16_reducedfc.pth'
    log_fn = 10
    neg_ratio = 3
    # input-image size
    min_size = 300
    # box out image size
    # for ssd_300
    grids = (38, 19, 10, 5, 3, 1) # feature_maps
    # for ssd_512
    # controls the number of default boxes
    # grids = (64, 32, 16, 8, 4, 2, 1)
    # boxes num
    # for ssd_300
    anchor_num = [4, 6, 6, 6, 4, 4]
    # for ssd_512
    # controls the number of pred boxes
    # anchor_num = [4, 6, 6, 6, 6, 4, 4]
    # 255 * R, G, B
    mean = (104, 117, 123)
    # for ssd_300
    aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))
    # for ssd_512
    # controls the number of default boxes
    """
    aspect_ratios = ((2, ), (2, 3), (2, 3), (2, 3),
                     (1/4, 1/3,  1/2,  1,  2,  3),
                     (1/4, 1/3,  1/2,  1,  2,  3),
                     (1/6, 1/4, 1/3,  1/2,  1,  2,  3))
    """
    steps = [s / 300 for s in (8, 16, 32, 64, 100, 300)]
    sizes = [s / 300 for s in (30, 60, 111, 162, 213, 264, 315)]
    # steps_512 = [s / 512 for s in (8, 16, 32, 64, 128, 256, 512)]
    # sizes_512 = [s / 512 for s in (20, 61, 133, 215, 296, 378, 460, 542)]
    variance = (0.1, 0.2)


hparam = Config()

# +
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def adjust_learning_rate1(optimizer):
    lr = hparam.lr * 0.1
    print('change learning rate, now learning rate is :', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate2(optimizer):
    lr = hparam.lr * 0.01
    print('change learning rate, now learning rate is :', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# ### Preparing Dataset for Dataloader

train_amp = ObjDetDataset(X, y['bboxes'], y['classes'])
# val_amp = ObjDetDataset(X_val, y_val['bboxes'], y_val['classes'])
model = RES18_SSD(hparam.num_classes, hparam.anchor_num, pretrained=True).to(device)
model.train()

mb = MultiBoxEncoder(hparam)

# dataloader = torch.utils.data.DataLoader(dataset, batch_size=hparam.batch_size, collate_fn=det_collate, num_workers=12)
train_loader = torch.utils.data.DataLoader(train_amp, batch_size=hparam.batch_size, collate_fn=det_collate,
                                           num_workers=10)
criterion = MultiBoxLoss(hparam.num_classes, hparam.neg_ratio).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=hparam.lr, momentum=hparam.momentum,
                            weight_decay=hparam.weight_decay)
print("Number of items in train dataloader:", len(train_loader))

# -
def train(model, optimizer, load_model=False, start_epoch=0):

    if load_model==True:
        list_of_files = glob.glob('weights/res18_ssd/*')
        latest_file = max(list_of_files, key=os.path.getctime)
        model, optimizer, start_epoch, loss = ssdhp.load_checkpoint(path=latest_file,
                                                         optimizer=optimizer,
                                                         model=model)
        start_epoch += 1

    for epoch in range(start_epoch, hparam.epochs):
        if epoch == 77:
            adjust_learning_rate1(optimizer)
        elif epoch == 96:
            adjust_learning_rate2(optimizer)
        total_loc_loss = 0
        total_cls_loss = 0
        total_loss = 0
        for i, (img, boxes, labels) in enumerate(train_loader):
            img = img.to(device)
            gt_boxes = []
            gt_labels = []
            for j, box in enumerate(boxes):
                # labels = box[:, 4]
                label = labels[j]
                # box = box[:, :-1]
                match_loc, match_label = mb.encode(box, label)

                gt_boxes.append(match_loc)
                gt_labels.append(match_label)

            gt_boxes = torch.FloatTensor(gt_boxes).to(device)
            gt_labels = torch.LongTensor(gt_labels).to(device)

            p_loc, p_label = model(img)

            loc_loss, cls_loss = criterion(p_loc, p_label, gt_boxes, gt_labels)

            loss = loc_loss + cls_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loc_loss += loc_loss.item()
            total_cls_loss += cls_loss.item()
            total_loss += loss.item()
            if i % hparam.log_fn == 0:
                avg_loc = total_loc_loss / (i + 1)
                avg_cls = total_cls_loss / (i + 1)
                avg_loss = total_loss / (i + 1)
                print(
                    'epoch[{}] | batch_idx[{}] | loc_loss [{:.2f}] | cls_loss [{:.2f}] '
                    '| total_loss [{:.2f}]'.format(epoch,i, avg_loc, avg_cls,  avg_loss))
        if epoch > 0:
            path = os.path.join('weights/res18_ssd/',
                                'namp_loss-{:.2f}.pth'.format(total_loss))
            ssdhp.save_checkpoint(model,optimizer,epoch,total_loss,path)

def train_amp(model, optimizer, load_model=False, start_epoch=0):

    if load_model==True:
        list_of_files = glob.glob('weights/res18_ssd/*')
        latest_file = max(list_of_files, key=os.path.getctime)
        model, optimizer, start_epoch, loss = ssdhp.load_checkpoint(path=latest_file,
                                                         optimizer=optimizer,
                                                         model=model)
        start_epoch += 1
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 0.0005,
        cycle_momentum=False,
        epochs=10,
        steps_per_epoch=int(np.ceil(len(X) / hparam.batch_size)),
    )
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch,hparam.epochs):
        """
        if e == 77:
            adjust_learning_rate1(optimizer)
        elif e == 96:
            adjust_learning_rate2(optimizer)
        """
        total_loc_loss = 0
        total_cls_loss = 0
        total_loss = 0
        for i, (img, boxes, labels) in enumerate(train_loader):
            scheduler.step()
            img = img.to(device)
            gt_boxes = []
            gt_labels = []
            for j, box in enumerate(boxes):
                # labels = box[:, 4]
                label = labels[j]
                # box = box[:, :-1]
                match_loc, match_label = mb.encode(box, label)

                gt_boxes.append(match_loc)
                gt_labels.append(match_label)

            gt_boxes = torch.FloatTensor(gt_boxes).to(device)
            gt_labels = torch.LongTensor(gt_labels).to(device)

            optimizer.zero_grad()
            # Automatic Mixed Precision Training
            with torch.cuda.amp.autocast():
                p_loc, p_label = model(img)
                loc_loss, cls_loss = criterion(p_loc, p_label,
                                               gt_boxes, gt_labels)
                loss = loc_loss + cls_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loc_loss += loc_loss.item()
            total_cls_loss += cls_loss.item()
            total_loss += loss.item()
            if i % hparam.log_fn == 0:
                avg_loc = total_loc_loss / (i + 1)
                avg_cls = total_cls_loss / (i + 1)
                avg_loss = total_loss / (i + 1)
                print(
                    'epoch[{}] | batch_idx[{}] | loc_loss [{:.2f}] | '
                    'cls_loss [{:.2f}] | total_loss [{:.2f}]'.format(epoch, i,avg_loc,
                                                                     avg_cls,avg_loss))
        if epoch >= 0:
            """torch.save(model.state_dict(), os.path.join(hparam.save_folder,
                                                        'loss-{:.2f}.pth'.format(total_loss)))"""
            path = os.path.join('weights/res18_ssd/',
                                'amp_loss-{:.2f}_epoch--{:.2f}.pth'.format(total_loss,epoch))
            ssdhp.save_checkpoint(model,optimizer,epoch,total_loss,path)

start_time = time.time()
train(model,optimizer)
exec_time = time.time() - start_time
exec_time = time.strftime("%H:%M:%S", time.gmtime(exec_time))
print("Training execution time(Without AMP): ", exec_time, " seconds")