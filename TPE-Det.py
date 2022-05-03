import sys
from ensemble_boxes import *
import torch
import numpy as np
import pandas as pd
from glob import glob
from torch.utils.data import Dataset,DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import gc
from matplotlib import pyplot as plt
from effdet import get_efficientdet_config, EfficientDet, DetBenchEval
from effdet.efficientdet import HeadNet
import torch.nn as nn
import os
from datetime import datetime
import time
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import natsort as ns
import re

SEED = 42 #any constant

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)

def euclid_dist(t1, t2):
    return np.sqrt(((t1-t2)**2).sum(axis = 1))

def load_net(checkpoint_path):
    config = get_efficientdet_config('tf_efficientdet_d3')
    net = EfficientDet(config, pretrained_backbone=False)
    
    config.num_classes = 1
    config.image_size=512
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))

    checkpoint = torch.load(checkpoint_path, map_location=device_num)
    net.load_state_dict(checkpoint['model_state_dict'])

    del checkpoint
    gc.collect()

    net = DetBenchEval(net, config)
    net.eval();
    device = torch.device(device_num)
    return net.to(device)

def get_axial_marking(label_path):
    lists_dir = glob(label_path+'*') #label file directorie list
    lists_dir.sort()

    lists_name = [f for f in os.listdir(label_path) if not f.startswith('.')]   #label file list. Neglect hidden files
    lists_name.sort()
    lists_name

    marking = pd.DataFrame(columns=['image_id', 'x', 'y', 'w', 'h'])

    for i in range(len(lists_dir)):
        xlsx = pd.read_excel(lists_dir[i], header = None)    
        temp = pd.DataFrame(columns=['slice', 'x', 'y', 'class'])
        temp2 = pd.DataFrame(columns=['image_id', 'x', 'y', 'w', 'h'])
        for k in range(xlsx.shape[0]):
            temp.loc[k] = list(xlsx.loc[k])
        temp = temp.drop_duplicates(['x','y'], keep = 'first')       #drop out repeated 'x','y' values(= drop out same cmb) -
        temp = temp.sort_values(by = 'slice',ignore_index=True)
        for k in range(temp.shape[0]):
            temp2.loc[k, 'image_id'] = lists_name[i].replace('.xlsx','')+ '_'+ str(temp.loc[k,'slice'])
            temp2.loc[k, 'x'] = temp.loc[k,'x']-44    #Convert coordinates 512X448 -> 360X360
            temp2.loc[k, 'y'] = temp.loc[k,'y']-76
            temp2.loc[k, 'w'] = 20
            temp2.loc[k, 'h'] = 20
        marking = pd.concat([marking, temp2], ignore_index=True)
    return marking

def get_sagittal_marking(label_path):
    lists_dir = glob(label_path+'*') #label file directorie list
    lists_dir.sort()

    lists_name = [f for f in os.listdir(label_path) if not f.startswith('.')]   #label file list. Neglect hidden files
    lists_name.sort() 
    lists_name

    marking = pd.DataFrame(columns=['image_id', 'x', 'y', 'w', 'h'])

    for i in range(len(lists_dir)):
        xlsx = pd.read_excel(lists_dir[i], header = None)    
        temp = pd.DataFrame(columns=['slice', 'x', 'y', 'class'])
        temp2 = pd.DataFrame(columns=['image_id', 'x', 'y', 'w', 'h'])
        for k in range(xlsx.shape[0]):
            temp.loc[k] = list(xlsx.loc[k])
        temp = temp.drop_duplicates(['x','y'], keep = 'first')       #drop out repeated 'x','y' values(= drop out same cmb) -
        temp = temp.sort_values(by = 'slice',ignore_index=True)      #It could delete the CMBs having same (x,y) but different in fact. 
        for k in range(temp.shape[0]):
            temp2.loc[k, 'image_id'] = lists_name[i].replace('.xlsx','')+ '_'+ str(temp.loc[k,'x'])
            temp2.loc[k, 'x'] = round(360-(temp.loc[k,'y']-76))-1    #Convert coordinates 512X448 -> 360X360
            temp2.loc[k, 'y'] = round(360*(72-temp.loc[k,'slice'])/72)+1
            temp2.loc[k, 'w'] = 20
            temp2.loc[k, 'h'] = 30
        temp2 = temp2.sort_values(by = 'image_id',ignore_index=True)    
        marking = pd.concat([marking, temp2], ignore_index=True)
    return marking

def get_coronal_marking(label_path): #sagittal cd -> coronal cd
    lists_dir = glob(label_path+'*') #label file directorie list
    lists_dir.sort()

    lists_name = [f for f in os.listdir(label_path) if not f.startswith('.')]   #label file list. Neglect hidden files
    lists_name.sort()
    lists_name

    marking = pd.DataFrame(columns=['image_id', 'x', 'y', 'w', 'h'])

    for i in range(len(lists_dir)):
        xlsx = pd.read_excel(lists_dir[i], header = None)    
        temp = pd.DataFrame(columns=['slice', 'x', 'y'])
        temp2 = pd.DataFrame(columns=['image_id', 'x', 'y', 'w', 'h'])
        for k in range(xlsx.shape[0]):
            temp.loc[k] = list(xlsx.loc[k])
        temp = temp.drop_duplicates(['x','y'], keep = 'first')       #drop out repeated 'x','y' values(= drop out same cmb) -
        temp = temp.sort_values(by = 'slice',ignore_index=True)      #It could delete the CMBs having same (x,y) but different in fact. 
        for k in range(temp.shape[0]):
            temp2.loc[k, 'image_id'] = lists_name[i].replace('.xlsx','')+ '_'+ str(360+76-temp.loc[k,'x'])
            temp2.loc[k, 'x'] = (temp.loc[k,'slice']-1+1)-44    #Convert coordinates 512X448 -> 360X360
            temp2.loc[k, 'y'] = temp.loc[k,'y']-1
            temp2.loc[k, 'w'] = 20
            temp2.loc[k, 'h'] = 35
        temp2 = temp2.sort_values(by = 'image_id',ignore_index=True)    
        marking = pd.concat([marking, temp2], ignore_index=True)
    return marking


#make the table for 'whole' test set images
def make_whole_marking_axial(label_path,IMAGE_ROOT_PATH, marking_test):
    lists_name = [f for f in os.listdir(label_path) if not f.startswith('.')]   #label file list. Neglect hidden files
    lists_name.sort()
    marking_test_all = pd.DataFrame(columns=['image_id', 'x', 'y', 'w', 'h'])

    for i in range(len(lists_name)):

        patient_name = lists_name[i].replace('.xlsx','')
        im_list = [path.split('/')[-1][:-4] for path in glob(f'{IMAGE_ROOT_PATH}/{patient_name}_*.png')]
        im_list = ns.natsorted(im_list)

        temp2 = pd.DataFrame(columns=['image_id', 'x', 'y', 'w', 'h'])
        temp2['image_id'] = im_list
        temp2['x'] = 1
        temp2['y'] = 1
        temp2['w'] = 1
        temp2['h'] = 1
        marking_test_all = pd.concat([marking_test_all, temp2], ignore_index=True)

    for i in range(len(marking_test)):     # fill the CMBs labels
        index_num = marking_test_all.index[marking_test_all['image_id']==marking_test.loc[i,'image_id']].tolist()
        if marking_test_all.loc[index_num[0],'x'] == 1:     #if it is first CMB on certain slice
            marking_test_all.loc[index_num[0]] = marking_test.loc[i]
        else:   #not first CMB on certain slice
            temp1 = marking_test_all[marking_test_all.index < index_num[0]]
            temp2 = marking_test_all[marking_test_all.index >= index_num[0]]
            marking_test_all = temp1.append(marking_test.loc[i],ignore_index=True).append(temp2, ignore_index=True)
    return marking_test_all

def make_whole_marking_sagittal(label_path, IMAGE_ROOT_PATH, marking_test):
    lists_name = [f for f in os.listdir(label_path) if not f.startswith('.')]   #label file list. Neglect hidden files
    lists_name.sort()
    marking_test_all = pd.DataFrame(columns=['image_id', 'x', 'y', 'w', 'h'])

    for i in range(len(lists_name)):

        patient_name = lists_name[i].replace('.xlsx','')
        im_list = [path.split('/')[-1][:-4] for path in glob(f'{IMAGE_ROOT_PATH}/{patient_name}_*.png')]
        im_list = ns.natsorted(im_list)

        temp2 = pd.DataFrame(columns=['image_id', 'x', 'y', 'w', 'h'])
        temp2['image_id'] = im_list
        temp2['x'] = 1
        temp2['y'] = 1
        temp2['w'] = 1
        temp2['h'] = 1
        marking_test_all = pd.concat([marking_test_all, temp2], ignore_index=True)

    for i in range(len(marking_test)):     # fill the CMBs labels
        index_num = marking_test_all.index[marking_test_all['image_id']==marking_test.loc[i,'image_id']].tolist()
        marking_test_all.loc[index_num[0],'x'] = marking_test.loc[i,'x']
        marking_test_all.loc[index_num[0],'y'] = marking_test.loc[i,'y']
        marking_test_all.loc[index_num[0],'w'] = marking_test.loc[i,'w']
        marking_test_all.loc[index_num[0],'h'] = marking_test.loc[i,'h']
        marking_test_all.loc[index_num[0],'x':'h']
    return marking_test_all

def make_whole_marking_coronal(label_path, IMAGE_ROOT_PATH, marking_test):
    lists_name = [f for f in os.listdir(label_path) if not f.startswith('.')]   #label file list. Neglect hidden files
    lists_name.sort()
    marking_test_all = pd.DataFrame(columns=['image_id', 'x', 'y', 'w', 'h'])

    for i in range(len(lists_name)):

        patient_name = lists_name[i].replace('.xlsx','')
        im_list = [path.split('/')[-1][:-4] for path in glob(f'{IMAGE_ROOT_PATH}/{patient_name}_*.png')]
        im_list = ns.natsorted(im_list)

        temp2 = pd.DataFrame(columns=['image_id', 'x', 'y', 'w', 'h'])
        temp2['image_id'] = im_list
        temp2['x'] = 1
        temp2['y'] = 1
        temp2['w'] = 1
        temp2['h'] = 1
        marking_test_all = pd.concat([marking_test_all, temp2], ignore_index=True)

    for i in range(len(marking_test)):     # fill the CMBs labels
        index_num = marking_test_all.index[marking_test_all['image_id']==marking_test.loc[i,'image_id']].tolist()
        if marking_test_all.loc[index_num[0],'x'] == 1:     #if it is first CMB on certain slice
            marking_test_all.loc[index_num[0]] = marking_test.loc[i]
        else:   #not first CMB on certain slice
            temp1 = marking_test_all[marking_test_all.index < index_num[0]]
            temp2 = marking_test_all[marking_test_all.index >= index_num[0]]
            marking_test_all = temp1.append(marking_test.loc[i],ignore_index=True).append(temp2, ignore_index=True)
    
    return marking_test_all    

def get_valid_transforms_axial():
    return A.Compose(
        [
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )

def get_valid_transforms_sagittal():
    return A.Compose(
        [
            A.HorizontalFlip(p=1),
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )

class DatasetRetriever_cmbs:

    def __init__(self, marking, image_ids, image_root_path, transforms=None, test=False):
        super().__init__()

        self.image_ids = image_ids
        self.marking = marking
        self.transforms = transforms
        self.test = test
        self.image_root_path  = image_root_path

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
    
        image, boxes = self.load_image_and_boxes(index)
        
        # there is only one class
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])


        if self.transforms:
            for i in range(10):
                sample = self.transforms(**{
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                })
                if len(sample['bboxes']) > 0:       
                    image = sample['image']
                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                    target['boxes'][:,[0,1,2,3]] = target['boxes'][:,[1,0,3,2]]  #yxyx: be warning
                    break
        else:
            image = torch.tensor(image)
            target['boxes'] = torch.tensor(boxes)

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def load_image_and_boxes(self, index):
        image_id = self.image_ids[index]
        image = cv2.imread(f'{self.image_root_path}/{image_id}.png', cv2.IMREAD_UNCHANGED)    #get 16bit images
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)    #Convert BGR -> RGB
        image/=65535.0
        records = self.marking[self.marking['image_id'] == image_id]
        boxes = records[['x', 'y', 'w', 'h']].values 
        boxes[:, 0] = boxes[:, 0] - boxes[:, 2]/2         #transforms to left top corner&right bottom corner
        boxes[:, 1] = boxes[:, 1] - boxes[:, 3]/2
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        return image, boxes
    
def collate_fn(batch):
    return tuple(zip(*batch))
    
def euclid_dist(t1, t2):
    return np.sqrt(((t1-t2)**2).sum(axis = 1))

def replaceRight(original, old, new, count_right):
    repeat=0
    text = original
    old_len = len(old)
    
    count_find = original.count(old)
    if count_right > count_find: 
        repeat = count_find
    else :
        repeat = count_right

    while(repeat):
      find_index = text.rfind(old)
      text = text[:find_index] + new + text[find_index+old_len:]

      repeat -= 1
      
    return text

##----------------------------------------------Training---------------------------------------------
train_label_path = '/data/labels/train/'
val_label_path = '/data/labels/validation/'

marking_train = get_axial_marking(train_label_path)
marking_val = get_axial_marking(val_label_path)

train_dataset_aug = DatasetRetriever(
    image_ids=np.array(marking_train['image_id']),  #array with image_ids
    marking=marking_train, 
    transforms=get_train_transforms(),
    test=False,
)

validation_dataset = DatasetRetriever(
    image_ids=np.array(marking_val['image_id']),
    marking=marking_val,
    transforms=get_valid_transforms(),
    test=True,
)

class TrainGlobalConfig:
    num_workers = 20
    batch_size = 1
    n_epochs = 10
    lr = 0.0001

    folder = 'Model_Save(Axial)_D7'

    # -------------------
    verbose = True
    verbose_step = 1
    # -------------------

    # --------------------
    step_scheduler = False # do scheduler.step after optimizer.step
    epoch_scheduler = False
    validation_scheduler = True # do scheduler.step after validation stage loss -> For scheduler 'ReduceLROnPlateau'
    
#     SchedulerClass = torch.optim.lr_scheduler.OneCycleLR
#     scheduler_params = dict(
#         max_lr=0.001,
#         epochs=n_epochs,
#         steps_per_epoch=2*int(len(train_dataset_aug) / batch_size),
#         pct_start=0.31,
#         anneal_strategy='cos', 
#         final_div_factor=10**4
#     )

#     SchedulerClass = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
#     scheduler_params = dict(
#         T_0=5,        # Number of iterations for the first restart.
#         T_mult=2,    
#         eta_min=0.00004,
#         last_epoch=-1, 
#         verbose=False
#     )

#     SchedulerClass = torch.optim.lr_scheduler.ExponentialLR
#     scheduler_params = dict(
#         gamma = 0.7
#     )

    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.1,
        patience=1,
        verbose=False, 
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0, 
        min_lr=0,
        eps=1e-08
    )

device_num = 'cuda:0'

def run_training():

    net = get_net()
    device = torch.device(device_num)
    net.to(device)

    train_loader = torch.utils.data.DataLoader(
        train_dataset_aug,
        batch_size=TrainGlobalConfig.batch_size,     
        sampler=RandomSampler(train_dataset_aug),
        pin_memory=False,
        drop_last=False,   #drop last one for having same batch size
        num_workers=TrainGlobalConfig.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        validation_dataset, 
        batch_size=TrainGlobalConfig.batch_size,
        num_workers=TrainGlobalConfig.num_workers,
        shuffle=False,
        sampler=SequentialSampler(validation_dataset),
        pin_memory=False,
        collate_fn=collate_fn,
    )

    fitter = Fitter(model=net, device=device, config=TrainGlobalConfig)
    best_val_loss, summary_loss_over_itr_train, summary_loss_over_itr_val = fitter.fit(train_loader, val_loader)
    
    return best_val_loss, summary_loss_over_itr_train, summary_loss_over_itr_val

from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet

def get_net():
    config = get_efficientdet_config('tf_efficientdet_d7')
    net = EfficientDet(config, pretrained_backbone=False)
    checkpoint = torch.load('i/efficientdet/efficientdet_d7-f05bf714.pth')
    net.load_state_dict(checkpoint)

    config.num_classes = 1
    config.image_size = 512  #D0

    net.class_net = HeadNet(config, num_outputs=config.num_classes) #Use default batchnorm
    

    
    return DetBenchTrain(net, config)

best_val_loss, summary_loss_over_itr_train, summary_loss_over_itr_val = run_training()

##----------------------------------------------Testing---------------------------------------------

test_label_path = 'data/labels/test/'
IMAGE_ROOT_PATH_AXIAL = 'data/images/axial/'
IMAGE_ROOT_PATH_CORONAL = 'data/images/coronal/'
IMAGE_ROOT_PATH_SAGITTAL = 'data/images/sagittal/'

def make_marking_cd_gt_axial(marking_test_axial):
    marking_cd_gt_axial = pd.DataFrame(columns=['patient_id', 's', 'x', 'y'])
    for i in range(len(marking_test_axial)):
        image_id = marking_test_axial.loc[i]['image_id']
        numbers_axial = re.findall("\d+", image_id)
        slice_num_axial = int(numbers_axial[-1])
        patient_id = replaceRight(image_id, '_'+str(slice_num_axial), '', 1)

        x = 512*marking_test_axial.loc[i]['x']/360
        y = 512*marking_test_axial.loc[i]['y']/360    

        temp = pd.DataFrame(columns=['patient_id', 's', 'x', 'y'])
        temp.loc[0] = [patient_id, slice_num_axial, x, y]
        marking_cd_gt_axial = pd.concat([marking_cd_gt_axial, temp], ignore_index=True)

    return marking_cd_gt_axial

num_cmbs=len(marking_test_all_axial[marking_test_all_axial['y']!=1])

def make_predictions_sagittal(images, score_threshold=0.23): #96% sensitivity(max) -> 0.15
    device = torch.device(device_num)
    images = torch.stack(images).to(device).float()
    predictions = []
    with torch.no_grad():
        det = net_sagittal(images, torch.tensor([1]*images.shape[0]).float().to(device))
        for i in range(images.shape[0]):
            boxes = det[i].detach().cpu().numpy()[:,:4]    
            scores = det[i].detach().cpu().numpy()[:,4]
            indexes = np.where(scores > score_threshold)[0]
            boxes = boxes[indexes]
            boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
            predictions.append({
                'boxes': boxes[indexes],
                'scores': scores[indexes],
            })
    return [predictions]

def run_wbf_sagittal(predictions, image_index, image_size=512, iou_thr=0.2, skip_box_thr=0, weights=None):   #skip_box_thr = confidence score
    boxes = [(prediction[image_index]['boxes']/(image_size-1)).tolist()  for prediction in predictions]
    scores = [prediction[image_index]['scores'].tolist()  for prediction in predictions]
    labels = [np.ones(prediction[image_index]['scores'].shape[0]).tolist() for prediction in predictions]
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes = boxes*(image_size-1)
    return boxes, scores, labels

def make_predictions_coronal(images, score_threshold=0.14): #0.15 -> 96.2% Sensitivity(max) 
    device = torch.device(device_num)
    images = torch.stack(images).to(device).float()
    predictions = []
    with torch.no_grad():
        det = net_coronal(images, torch.tensor([1]*images.shape[0]).float().to(device))
        for i in range(images.shape[0]):
            boxes = det[i].detach().cpu().numpy()[:,:4]    
            scores = det[i].detach().cpu().numpy()[:,4]
            indexes = np.where(scores > score_threshold)[0]
            boxes = boxes[indexes]
            boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
            predictions.append({
                'boxes': boxes[indexes],
                'scores': scores[indexes],
            })
    return [predictions]

def run_wbf_coronal(predictions, image_index, image_size=512, iou_thr=0.4, skip_box_thr=0, weights=None):   #skip_box_thr = confidence score
    boxes = [(prediction[image_index]['boxes']/(image_size-1)).tolist()  for prediction in predictions]
    scores = [prediction[image_index]['scores'].tolist()  for prediction in predictions]
    labels = [np.ones(prediction[image_index]['scores'].shape[0]).tolist() for prediction in predictions]
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes = boxes*(image_size-1)
    return boxes, scores, labels

def make_predictions_axial(images, score_threshold=0.29): #Confidence score...? Default 0.22
    device = torch.device(device_num)
    images = torch.stack(images).to(device).float()
    predictions = []
    with torch.no_grad():
        det = net_axial(images, torch.tensor([1]*images.shape[0]).float().to(device))
        for i in range(images.shape[0]):
            boxes = det[i].detach().cpu().numpy()[:,:4]    
            scores = det[i].detach().cpu().numpy()[:,4]
            indexes = np.where(scores > score_threshold)[0]
            boxes = boxes[indexes]
            boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
            predictions.append({
                'boxes': boxes[indexes],
                'scores': scores[indexes],
            })
    return [predictions]

def run_wbf_axial(predictions, image_index, image_size=512, iou_thr=0.45, skip_box_thr=0, weights=None):
    boxes = [(prediction[image_index]['boxes']/(image_size-1)).tolist()  for prediction in predictions]
    scores = [prediction[image_index]['scores'].tolist()  for prediction in predictions]
    labels = [np.ones(prediction[image_index]['scores'].shape[0]).tolist() for prediction in predictions]
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes = boxes*(image_size-1)
    return boxes, scores, labels

def count_fptp(marking_cd_gt_patient, marking_cd_gt, marking_cd_slice, slice_num, patient_name, x, y):
    #tp_candi -> near_gt: x,y cordi of ground truth from slices adjacent to predicting slice
    #fp_candi -> near_pd: x,y cordi of predicted boxed from slices adjacent to predicting slice
    #near_gt: ground truths (x, y cordi info) in near slices (upper and lower) from current images  
    #near_pd: detections (slice, x, y cordi info) in near slices (only upper) from current images
    global fp 
    global fp_count
    global tp 
    global tp_count
    
    near_gt = marking_cd_gt_patient.loc[abs(slice_num-marking_cd_gt['s'])<5][['x','y']].to_numpy(dtype=float) #GT adjacent to prediction
    near_pd = marking_cd_slice.loc[(slice_num-marking_cd_slice['s']<5)&(marking_cd_slice['patient_id']==patient_name)][['x','y']].to_numpy(dtype=float)

    #count fp 
    if near_gt.shape[0]!=0:  #if predicted slice is adjacent to GT slices
        dists_from_gt_list = euclid_dist([x, y], near_gt)
        if dists_from_gt_list.min() > 20: #if the prediction is far from gt->FP in x, y cordi
            #check whether the fp is consecutive or not
            if near_pd.shape[0]!= 0: #
                if euclid_dist([x, y], near_pd).min() > 10: #not consecutive detection.
                    fp_count +=1
                    fp = np.append(fp, np.array([[patient_name, slice_num, x, y, 0]]), axis=0)
            else:
                fp_count +=1
                fp = np.append(fp, np.array([[patient_name, slice_num, x, y, 1]]), axis=0)
        
        else: #if the prediction is close to gt -> TP
            for q in range(len(near_gt)):
                near_gt_index = marking_cd_gt_patient.loc[(marking_cd_gt['x']==near_gt[q][0]) & (marking_cd_gt['y']==near_gt[q][1])].index[0]
                if dists_from_gt_list[q] < 30 and marking_cd_gt_patient.loc[near_gt_index, 'state'] != 1: # if the gt is close to prediction and not detected.
                    #check whether the tp is consecutive or not
                    if near_pd.shape[0] != 0: #There are dets in the near slices 
                        if euclid_dist([x, y], near_pd).min() > 5: #there is no close x, y det -> not consecutive. 5
                            tp_count += 1
                            tp = np.append(tp, np.array([[patient_name, slice_num, x, y, 0]]), axis=0)
                            marking_cd_gt_patient.loc[near_gt_index, 'state'] = 1
                    else:
                        tp_count += 1
                        tp = np.append(tp, np.array([[patient_name, slice_num, x, y, 1]]), axis=0)
                        marking_cd_gt_patient.loc[near_gt_index, 'state'] = 1
                        
    else:   #Predicted slice is not adjacent to that of GTs               
        if i == 0:   #first slice
            fp_count +=1
            fp = np.append(fp, np.array([[patient_name, slice_num, x, y, 2]]), axis=0)
        else:  
            if near_pd.shape[0] != 0:
                if euclid_dist([x, y], near_pd).min() > 10:
                    fp_count +=1
                    fp = np.append(fp, np.array([[patient_name, slice_num, x, y, 3]]), axis=0)
            else:
                fp_count +=1
                fp = np.append(fp, np.array([[patient_name, slice_num, x, y, 4]]), axis=0)
    return fp, fp_count

def load_net(checkpoint_path):
    config = get_efficientdet_config('tf_efficientdet_d3')
    net = EfficientDet(config, pretrained_backbone=False)
    
    config.num_classes = 1
    config.image_size=512
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))

    checkpoint = torch.load(checkpoint_path, map_location=device_num)
    net.load_state_dict(checkpoint['model_state_dict'])

    del checkpoint
    gc.collect()

    net = DetBenchEval(net, config)
    net.eval();
    device = torch.device(device_num)
    return net.to(device)

net_axial = load_net('../Model/best-checkpoint.bin')
net_sagittal = load_net('../Model/best-checkpoint.bin')
net_coronal = load_net('../Model/best-checkpoint.bin')

test_dataset_all_sagittal = DatasetRetriever_cmbs(
    image_ids=np.array(marking_test_all_sagittal['image_id']),
    marking=marking_test_all_sagittal,
    transforms=get_valid_transforms_sagittal(),
    test=False,
    image_root_path = IMAGE_ROOT_PATH_SAGITTAL,
)

test_data_loader_all_sagittal = DataLoader(
    test_dataset_all_sagittal,
    batch_size=32,     #batchsize = 4
    drop_last=False,
    num_workers=0,
    collate_fn=collate_fn,)
torch.multiprocessing.set_sharing_strategy('file_system')


def make_predictions_sagittal(images, score_threshold=0.23): #96% sensitivity(max) -> 0.15
    device = torch.device(device_num)
    images = torch.stack(images).to(device).float()
    predictions = []
    with torch.no_grad():
        det = net_sagittal(images, torch.tensor([1]*images.shape[0]).float().to(device))
        for i in range(images.shape[0]):
            boxes = det[i].detach().cpu().numpy()[:,:4]    
            scores = det[i].detach().cpu().numpy()[:,4]
            indexes = np.where(scores > score_threshold)[0]
            boxes = boxes[indexes]
            boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
            predictions.append({
                'boxes': boxes[indexes],
                'scores': scores[indexes],
            })
    return [predictions]

def run_wbf_sagittal(predictions, image_index, image_size=512, iou_thr=0.2, skip_box_thr=0, weights=None):   #skip_box_thr = confidence score
    boxes = [(prediction[image_index]['boxes']/(image_size-1)).tolist()  for prediction in predictions]
    scores = [prediction[image_index]['scores'].tolist()  for prediction in predictions]
    labels = [np.ones(prediction[image_index]['scores'].shape[0]).tolist() for prediction in predictions]
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes = boxes*(image_size-1)
    return boxes, scores, labels

#Test in all data
image_count = 0
cd_arr_sagittal = np.empty((0, 3))
boxes_predict_save = np.empty((0, 4))
marking_cd_sagittal = pd.DataFrame(columns=['image_id', 'x', 'y', 'z'])

for j, (images_sagittal, targets_sagittal, image_ids_sagittal) in enumerate(test_data_loader_all_sagittal):
    predictions = make_predictions_sagittal(images_sagittal)
    print(f'Batch{j} prediction done')
    for i in range(len(images_sagittal)):
            
        sample = images_sagittal[i].permute(1,2,0).cpu().numpy()

        boxes_predict_sagittal, scores_sagittal, labels_sagittal = run_wbf_sagittal(predictions, image_index=i)
        boxes_predict_sagittal = boxes_predict_sagittal.astype(np.int32).clip(min=0, max=511)
        
        numbers_sagittal = re.findall("\d+", image_ids_sagittal[i])    #get 3D coordinates
        slice_num_sagittal = int(numbers_sagittal[-1])
        
        boxes_predict_save = np.append(boxes_predict_save, boxes_predict_sagittal, axis=0)
        for box1 in boxes_predict_sagittal:
            x_im = (box1[0]+box1[2])/2   #coordinate in image
            y_im = (box1[1]+box1[3])/2

            x= 512*(slice_num_sagittal-44)/360   #3D coordinate     -> axial is cut 512*412 -> 360*360
            y = x_im
            z = 512*round(72*(512-y_im)/512)/72 #match to 72 axial slices and rescale to 512 
            cd_arr_sagittal = np.append(cd_arr_sagittal, np.array([[x, y, z]]), axis = 0)
            temp = pd.DataFrame(columns=['image_id', 'x', 'y', 'z'])
            temp.loc[0] = [image_ids_sagittal[i], x, y, z]
            marking_cd_sagittal = pd.concat([marking_cd_sagittal, temp], ignore_index=True)

        #ground truth boxes
        image, target, image_id = test_dataset_all_sagittal[image_count]
        boxes_ground_truth = target['boxes'].cpu().numpy().astype(np.int32)

test_dataset_all_coronal = DatasetRetriever_cmbs(
    image_ids=np.array(marking_test_all_coronal['image_id']),
    marking=marking_test_all_coronal,
    transforms=get_valid_transforms_axial(),
    test=False,
    image_root_path = IMAGE_ROOT_PATH_CORONAL,

)

test_data_loader_all_coronal = DataLoader(
    test_dataset_all_coronal,
    batch_size=32,     #batchsize = 4
    drop_last=False,
    num_workers=0,
    collate_fn=collate_fn,)

def make_predictions_coronal(images, score_threshold=0.14): #0.15 -> 96.2% Sensitivity(max) 
    device = torch.device(device_num)
    images = torch.stack(images).to(device).float()
    predictions = []
    with torch.no_grad():
        det = net_coronal(images, torch.tensor([1]*images.shape[0]).float().to(device))
        for i in range(images.shape[0]):
            boxes = det[i].detach().cpu().numpy()[:,:4]    
            scores = det[i].detach().cpu().numpy()[:,4]
            indexes = np.where(scores > score_threshold)[0]
            boxes = boxes[indexes]
            boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
            predictions.append({
                'boxes': boxes[indexes],
                'scores': scores[indexes],
            })
    return [predictions]

def run_wbf_coronal(predictions, image_index, image_size=512, iou_thr=0.4, skip_box_thr=0, weights=None):   #skip_box_thr = confidence score
    boxes = [(prediction[image_index]['boxes']/(image_size-1)).tolist()  for prediction in predictions]
    scores = [prediction[image_index]['scores'].tolist()  for prediction in predictions]
    labels = [np.ones(prediction[image_index]['scores'].shape[0]).tolist() for prediction in predictions]
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes = boxes*(image_size-1)
    return boxes, scores, labels

image_count = 0
cd_arr_coronal = np.empty((0, 3))
boxes_predict_save = np.empty((0, 4))
marking_cd_coronal = pd.DataFrame(columns=['image_id', 'x', 'y', 'z'])

for j, (images, targets, image_ids) in enumerate(test_data_loader_all_coronal):
    predictions = make_predictions_coronal(images)
    print(f'Batch{j} prediction done')
    for i in range(len(images)):
            
        sample = images[i].permute(1,2,0).cpu().numpy()

        boxes_predict, scores, labels = run_wbf_coronal(predictions, image_index=i)
        boxes_predict = boxes_predict.astype(np.int32).clip(min=0, max=511)
        
        numbers = re.findall("\d+", image_ids[i])    #get 3D coordinates
        slice_num = int(numbers[-1])
        
        boxes_predict_save = np.append(boxes_predict_save, boxes_predict, axis=0)
        for box1 in boxes_predict:
            x_im = (box1[0]+box1[2])/2   #coordinate in image
            y_im = (box1[1]+box1[3])/2

            x= x_im   #3D coordinate     -> axial is cut 512*412 -> 360*360
            y = 512*(slice_num-76)/360
            z = 512-y_im
            z = 512*round(72*z/512)/72 #round up to 72 slices in 512
            cd_arr_coronal = np.append(cd_arr_coronal, np.array([[x, y, z]]), axis = 0)
            temp = pd.DataFrame(columns=['image_id', 'x', 'y', 'z'])
            temp.loc[0] = [image_ids[i], x, y, z]
            marking_cd_coronal = pd.concat([marking_cd_coronal, temp], ignore_index=True)

cd_arr_coronal.shape


test_dataset_all_axial = DatasetRetriever_cmbs(
    image_ids=np.array(marking_test_all_axial['image_id']),
    marking=marking_test_all_axial,
    transforms=get_valid_transforms_axial(),
    test=False,
    image_root_path = IMAGE_ROOT_PATH_AXIAL,
)

test_data_loader_all_axial = DataLoader(
    test_dataset_all_axial,
    batch_size=32,     #batchsize = 4
    drop_last=False,
    num_workers=0,
    collate_fn=collate_fn,
)

def make_predictions_axial(images, score_threshold=0.29): #Confidence score...? Default 0.22
    device = torch.device(device_num)
    images = torch.stack(images).to(device).float()
    predictions = []
    with torch.no_grad():
        det = net_axial(images, torch.tensor([1]*images.shape[0]).float().to(device))
        for i in range(images.shape[0]):
            boxes = det[i].detach().cpu().numpy()[:,:4]    
            scores = det[i].detach().cpu().numpy()[:,4]
            indexes = np.where(scores > score_threshold)[0]
            boxes = boxes[indexes]
            boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
            predictions.append({
                'boxes': boxes[indexes],
                'scores': scores[indexes],
            })
    return [predictions]

def run_wbf_axial(predictions, image_index, image_size=512, iou_thr=0.45, skip_box_thr=0, weights=None):
    boxes = [(prediction[image_index]['boxes']/(image_size-1)).tolist()  for prediction in predictions]
    scores = [prediction[image_index]['scores'].tolist()  for prediction in predictions]
    labels = [np.ones(prediction[image_index]['scores'].shape[0]).tolist() for prediction in predictions]
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes = boxes*(image_size-1)
    return boxes, scores, labels

#Combine sagittal info, all patient
mimic_dist = []
image_count = 0
fp_count = 0
tp_count = 0
cd_arr_axial_pd = np.empty((0, 3))
fp = np.empty((0, 5))
tp = np.empty((0, 5))
prev_patient_name = 'empty'

marking_cd_slice_axial = pd.DataFrame(columns=['patient_id', 's', 'x', 'y'])
two_det_sagi = pd.DataFrame(columns=['patient_id', 's', 'x', 'y'])
two_det_coro = pd.DataFrame(columns=['patient_id', 's', 'x', 'y'])

for j, (images_axial, targets_axial, image_ids_axial) in enumerate(test_data_loader_all_axial):
    predictions = make_predictions_axial(images_axial)
    print(f'Batch{j} prediction done')
    for i in range(len(images_axial)):        
        sample = images_axial[i].permute(1,2,0).cpu().numpy()

        boxes_predict_axial, scores_axial, labels_axial = run_wbf_axial(predictions, image_index=i)
        boxes_predict_axial = boxes_predict_axial.astype(np.int32).clip(min=0, max=511)
        numbers_axial = re.findall("\d+", image_ids_axial[i])
        slice_num_axial = int(numbers_axial[-1])

        th_dist_sagi = 7.3 #distance thresh
        th_dist_coro = 5.1  #distance thresh
#         th_dist_coro = 7.2  #distance thresh

        patient_name = replaceRight(image_ids_axial[i], '_'+str(slice_num_axial), '', 1)
        cd_patient_sagittal = marking_cd_sagittal[marking_cd_sagittal['image_id'].str.contains(patient_name)][['x', 'y', 'z']].to_numpy()
        cd_patient_coronal = marking_cd_coronal[marking_cd_coronal['image_id'].str.contains(patient_name)][['x', 'y', 'z']].to_numpy()

        if prev_patient_name != patient_name:
            marking_cd_gt_patient_axial = marking_cd_gt_axial.loc[marking_cd_gt_axial['patient_id'] ==patient_name]
            marking_cd_gt_patient_axial['state'] = np.nan

        boxes_predict_final = np.empty((0, 4))
        if boxes_predict_axial.shape[0]!=0:
            for k, (box1_axial) in enumerate(boxes_predict_axial):
                x = (box1_axial[0]+box1_axial[2])/2
                y = (box1_axial[1]+box1_axial[3])/2
                z = 512*(slice_num_axial)/72
                closest_dist_sagi = euclid_dist([x, y, z], cd_patient_sagittal).min()
                closest_dist_coro = euclid_dist([x, y, z], cd_patient_coronal).min()
                if (closest_dist_sagi < th_dist_sagi)&(closest_dist_coro < th_dist_coro): #same slice, both detected on Sagittal&Co
                    boxes_predict_final = np.append(boxes_predict_final, np.array([boxes_predict_axial[k]]), axis=0)
    #                 mimic_dist.append(closest_dist)

                    count_fptp(marking_cd_gt_patient_axial, marking_cd_gt_axial, marking_cd_slice_axial, slice_num_axial, patient_name, x, y)

                    cd_arr_axial_pd = np.append(cd_arr_axial_pd, np.array([[x, y, z]]), axis=0)
                    temp_cd_slice = pd.DataFrame(columns=['patient_id', 's', 'x', 'y'])
                    temp_cd_slice.loc[0] = [patient_name, slice_num_axial, x, y]
                    marking_cd_slice_axial = pd.concat([marking_cd_slice_axial, temp_cd_slice], ignore_index=True)

                elif (closest_dist_sagi < th_dist_sagi): #detectory         
                    tp_candi_coro = two_det_coro.loc[(slice_num_axial-two_det_coro['s']<2)&(two_det_coro['patient_id']==patient_name)][['x','y']].to_numpy(dtype=float)
                    if tp_candi_coro.shape[0]!=0:  #if prediction adjusts to GT slices
                        if euclid_dist([x, y], tp_candi_coro).min() < 2:
                            boxes_predict_final = np.append(boxes_predict_final, np.array([boxes_predict_axial[k]]), axis=0)
                            
                            count_fptp(marking_cd_gt_patient_axial, marking_cd_gt_axial, marking_cd_slice_axial, slice_num_axial, patient_name, x, y)
    
                            cd_arr_axial_pd = np.append(cd_arr_axial_pd, np.array([[x, y, z]]), axis=0)
                            temp_cd_slice = pd.DataFrame(columns=['patient_id', 's', 'x', 'y'])
                            temp_cd_slice.loc[0] = [patient_name, slice_num_axial, x, y]
                            marking_cd_slice_axial = pd.concat([marking_cd_slice_axial, temp_cd_slice], ignore_index=True)

                    temp_cd_slice = pd.DataFrame(columns=['patient_id', 's', 'x', 'y'])
                    temp_cd_slice.loc[0] = [patient_name, slice_num_axial, x, y]
                    two_det_sagi = pd.concat([two_det_sagi, temp_cd_slice], ignore_index=True)

                elif (closest_dist_coro < th_dist_coro):
                    tp_candi_sagi = two_det_sagi.loc[(slice_num_axial-two_det_sagi['s']<2)&(two_det_sagi['patient_id']==patient_name)][['x','y']].to_numpy(dtype=float)
                    if tp_candi_sagi.shape[0]!=0:  #if prediction adjusts to GT slices
                        if euclid_dist([x, y], tp_candi_sagi).min() < 2:
                            boxes_predict_final = np.append(boxes_predict_final, np.array([boxes_predict_axial[k]]), axis=0)
                            tp_candi = marking_cd_gt_patient_axial.loc[abs(slice_num_axial-marking_cd_gt_axial['s'])<4][['x','y']].to_numpy(dtype=float)
                            #count fp 
                            count_fptp(marking_cd_gt_patient_axial, marking_cd_gt_axial, marking_cd_slice_axial, slice_num_axial, patient_name, x, y)

                            cd_arr_axial_pd = np.append(cd_arr_axial_pd, np.array([[x, y, z]]), axis=0)
                            temp_cd_slice = pd.DataFrame(columns=['patient_id', 's', 'x', 'y'])
                            temp_cd_slice.loc[0] = [patient_name, slice_num_axial, x, y]
                            marking_cd_slice_axial = pd.concat([marking_cd_slice_axial, temp_cd_slice], ignore_index=True)

                    temp_cd_slice = pd.DataFrame(columns=['patient_id', 's', 'x', 'y'])
                    temp_cd_slice.loc[0] = [patient_name, slice_num_axial, x, y]
                    two_det_coro = pd.concat([two_det_coro, temp_cd_slice], ignore_index=True)


            boxes_predict_final = boxes_predict_final.astype(np.int32)
            prev_patient_name = patient_name

        #ground truth boxes
        image, target, image_id = test_dataset_all_axial[image_count]
        boxes_ground_truth = target['boxes'].cpu().numpy().astype(np.int32)
        image_count += 1
        
        if boxes_predict_final.shape[0]!=0 or boxes_ground_truth[0][0]!=0:   #Plot only slices containing final prediction or GT
            fig, ax = plt.subplots(1, 1, figsize=(16, 8))

            for box1 in boxes_predict_final:
                cv2.rectangle(sample, (box1[0], box1[1]), (box1[2], box1[3]), (0, 0, 1), 2)


            for box2 in boxes_ground_truth:
                cv2.rectangle(sample, (box2[1], box2[0]), (box2[3],  box2[2]), (1, 0, 0), 2)

            ax.set_axis_off()
            ax.set_title(f'{i}th image : {image_id}')
            ax.imshow(sample);