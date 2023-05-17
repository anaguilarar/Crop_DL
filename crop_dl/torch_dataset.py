
import torch
from torch.utils.data import Dataset
import os

import tqdm
import numpy as np
#from crop_dl.dataset_utils import InstanceSegmentation
import  torchvision.transforms as T


def coco_total_imgs(cocodataset):
    tmpbool = True
    i = 0
    while tmpbool:
        try:
            cocodataset.loadImgs(i)
            i+=1
            
        except:
            tmpbool = False
    return i-1
    
class InstaSegemenTorchDataset(Dataset):

    def __init__(self, 
                 root_dir, 
                 cococataset, 
                 evaluation= False, 
                 channelsfirst = True,
                 meanstd_imgs = None,
                 valaugoptions = None,
                 random_transform_params = None,
                 multitr_chain = ['flip','clahe','rotation','zoom']):
        
        
        self.root_dir = root_dir
        self.cococataset = cococataset
        self.len_images = coco_total_imgs(cococataset)
        self.scaler = meanstd_imgs
        self.evaluation = evaluation
        self.channelsfirst =channelsfirst
        self.valaugoptions = valaugoptions
        self.random_transform_params = random_transform_params 
        self.multitr_chain =multitr_chain
        if meanstd_imgs is not None:
            self.normalize = T.Normalize(meanstd_imgs[0], meanstd_imgs[1])
        else:
            self.normalize = None

    def __len__(self):
        return self.len_images

    def __getitem__(self, index):
        
        from .dataset_utils import InstanceSegmentation
        instancedata = InstanceSegmentation(self.root_dir ,
                                            self.cococataset,index,
                                            random_parameters = self.random_transform_params,
                                            multitr_chain = self.multitr_chain)
        troption = 'raw'
        if self.evaluation:
            randtr = instancedata.random_multime_transform(augfun='raw')
        else:
            try:
                randtr = instancedata.random_multime_transform()
                troption = list(instancedata.tr_paramaters.keys())[-1]
            except:
                randtr = instancedata.random_multime_transform(augfun='raw')

        ### transform to torch tensor
        boxes = instancedata.bounding_boxes[troption]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        if boxes.shape[0]==0:
            troption = 'raw'
            instancedata = InstanceSegmentation(self.root_dir ,
                                            self.cococataset,index)
            randtr =instancedata.random_multime_transform(augfun=troption)
            boxes = instancedata.bounding_boxes[troption]
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            if boxes.shape[0]==0:
                instancedata = InstanceSegmentation(self.root_dir ,
                                            self.cococataset,
                                            index +1 if index == 0 else index-1)
                randtr =instancedata.random_multime_transform(augfun=troption)
                boxes = instancedata.bounding_boxes[troption]
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                

        numobjs = instancedata.bounding_boxes[troption].shape[0]
        masks = instancedata.target_data[troption]
        img = instancedata.imgs_data[troption]
        # 
        labels = torch.ones((numobjs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        iscrowd  = torch.zeros((numobjs,), dtype=torch.int64)
        image_id = torch.tensor([index])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        channelsfirst = True

        if channelsfirst:
            if img.shape[0] != 3:
                imgtensor = torch.from_numpy(img.swapaxes(2,1).swapaxes(0,1)/255).float()
            elif img.shape[0] == 3:
                imgtensor = torch.from_numpy(img/255).float()
         
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.normalize is not None:
            imgtensor = self.normalize(imgtensor)

        return imgtensor, target