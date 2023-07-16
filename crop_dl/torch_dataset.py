
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import cv2

import tqdm
import numpy as np
#from crop_dl.dataset_utils import InstanceSegmentation
import  torchvision.transforms as T
from .dataset import SplitIds, FolderWithImages
from .dataset_utils import MultiChannelImage,standard_scale

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
                                            self.cococataset,
                                            index,
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
    


class SegmenTorchDataset(Dataset):

    def __init__(self, 
                 root_dir, 
                 cococataset, 
                 evaluation= False, 
                 channelsfirst = True,
                 meanstd_imgs = None,
                 valaugoptions = None,
                 random_transform_params = None,
                 onlythesetransforms = None,
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
        self.onlythesetr = onlythesetransforms
        
        if meanstd_imgs is not None:
            self.normalize = T.Normalize(meanstd_imgs[0], meanstd_imgs[1])
        else:
            self.normalize = None

    def __len__(self):
        return self.len_images

    def __getitem__(self, index):
        
        from .dataset_utils import SegmentationImagesCoCo
        instancedata = SegmentationImagesCoCo(self.root_dir ,
                                            self.cococataset,
                                            index,
                                            random_parameters = self.random_transform_params,
                                            multitr_chain = self.multitr_chain,
                                            onlythese=self.onlythesetr)
                                            
        troption = 'raw'
        if self.evaluation:
            instancedata.random_multime_transform(augfun='raw')
        else:
            try:
                instancedata.random_multime_transform()
                troption = list(instancedata.tr_paramaters.keys())[-1]
            except:
                instancedata.random_multime_transform(augfun='raw')

        # mask
        masks = instancedata.target_data[troption]
        masks = np.max(np.stack(
                    masks), axis = 0)
        target_image = np.expand_dims(masks, axis = 0)
        target_image = torch.from_numpy(target_image).float()
        
        # image
        img = instancedata.imgs_data[troption]
        if img.shape[0] != 3:
            imgtensor = torch.from_numpy(img.swapaxes(2,1).swapaxes(0,1)/255).float()
        elif img.shape[0] == 3:
            imgtensor = torch.from_numpy(img/255).float()
        

        if self.normalize is not None:
            imgtensor = self.normalize(imgtensor)

        return imgtensor, target_image
    


### Classification
def data_standarization(values, meanval = None, stdval = None):
    if meanval is None:
        meanval = np.nanmean(values)
    if stdval is None:
        stdval = np.nanstd(values)
    
    return (values - meanval)/stdval

def data_normalization(data, minval = None, maxval = None):
    if minval is None:
        minval = np.nanmin(data)
    if maxval is None:
        maxval = np.nanmax(data)
    
    return (data - minval) / ((maxval - minval))

def transform_listarrays(values, varchanels = None, scaler = None, scalertype = 'standarization'):
    
    if varchanels is None:
        varchanels = list(range(len(values)))
    if scalertype == 'standarization':
        if scaler is None:
            scaler = {chan:[np.nanmean(values[i]),
                            np.nanstd(values[i])] for i, chan in enumerate(varchanels)}
        fun = data_standarization
    elif scalertype == 'normalization':
        if scaler is None:
            scaler = {chan:[np.nanmin(values[i]),
                            np.nanmax(values[i])] for i, chan in enumerate(varchanels)}
        fun = data_normalization
    
    else:
        raise ValueError('{} is not an available option')
    
    valueschan = {}
    for i, channel in enumerate(varchanels):
        if channel in list(scaler.keys()):
            val1, val2 = scaler[channel]
            scaleddata = fun(values[i], val1, val2)
            valueschan[channel] = scaleddata
    
    return valueschan    

def customdict_transformation(customdict, scaler, scalertype = 'standarization'):
    """scale customdict

    Args:
        customdict (dict): custom dict
        scaler (dict): dictionary that contains the scalar values per channel. 
                       e.g. for example to normalize the red channel you will provide min and max values {'red': [1,255]}  
        scalertype (str, optional): string to mention if 'standarization' or 'normalization' is gonna be applied. Defaults to 'standarization'.

    Returns:
        xrarray: xrarraytransformed
    """
    

    varchanels = list(customdict['variables'].keys())
    values =[customdict['variables'][i] for i in varchanels]
    trvalues = transform_listarrays(values, varchanels = varchanels, scaler = scaler, scalertype =scalertype)
    for chan in list(trvalues.keys()):
        customdict['variables'][chan] = trvalues[chan]
        
    return customdict
        
import pickle
import os

def get_data_from_dict(data, onlythesechannels = None):
            
        dataasarray = []
        channelsnames = list(data['variables'].keys())
        
        if onlythesechannels is not None:
            channelstouse = [i for i in onlythesechannels if i in channelsnames]
        else:
            channelstouse = channelsnames
        for chan in channelstouse:
            dataperchannel = data['variables'][chan] 
            dataasarray.append(dataperchannel)

        return np.array(dataasarray)
    
class C_TBDataSet:
    
    def get_data(self, index, onlythesechannels = None):
        
        file = self.listfiles[index]
        
        customdict = self.mlcclass._read_data(
            path=os.path.dirname(file), 
            fn = os.path.basename(file),
            suffix='pickle')
        
        if self.scalar is not None:
            customdict = customdict_transformation(customdict, self.scalar)
            
        npdata = self.mlcclass.to_array(
            customdict = customdict,
            onlythesechannels = onlythesechannels)
        
        npdata[np.isnan(npdata)] = 0
        targetdata = self.targetvalues[index]
        if self.ids is not None:
            idval = self.ids[index]
        else:
            idval = None
    
        return npdata, targetdata, idval
    
    def __init__(self, df, 
                 targetcolumn,
                 mlcclass,
                 scalar = None,
                 columnpathname = 'path', columnids = None) -> None:
        
        self.ids = None
        self.df_path = df.reset_index()
        self.scalar = scalar
        assert targetcolumn in df.columns
        
        self.targetvalues = df[targetcolumn].values
        self.listfiles = df[columnpathname].values
        assert len(self.listfiles)>1
        if columnids:
            self.ids = df[columnids].values
        
        self.mlcclass = mlcclass


class ClassificationTorchDataset(Dataset):
    
    def __init__(self, 
                 dftarget,
                 targetcolumn,
                 mlcclass,
                 evaluation = False,
                 scalar = None, 
                 onlythesefeatures = None, 
                 depthfirst = True,
                 idcolumnname = None
                 ):
        """_summary_

        Args:
            dftarget (_type_): _description_
            targetcolumn (_type_): _description_
            evaluation (bool, optional): _description_. Defaults to False.
            scalar (_type_, optional): _description_. Defaults to None.
            onlythesefeatures (_type_, optional): _description_. Defaults to None.
            depthfirst (bool, optional): _description_. Defaults to True.
            idcolumnname (_type_, optional): _description_. Defaults to None.
        """
        self.scalar = scalar
        self.evaluation = evaluation
        self.idcolumnname = idcolumnname
        assert type(dftarget) is pd.DataFrame
        self.data = C_TBDataSet(dftarget, targetcolumn,
                                mlcclass,
                                scalar,
            columnids = self.idcolumnname)
        self.depthfirst = depthfirst
        self.nrows = dftarget.shape[0]
        self.onlythesefeatures = onlythesefeatures 
    
        
    def __len__(self):
        return self.nrows

    def __getitem__(self, index):
        
        img, target, idval = self.data.get_data(
            index, self.onlythesefeatures)
        self._idval = idval 
        img = np.array([cv2.resize(img[i], (224,224)) for i in range(img.shape[0])])
        multchan = MultiChannelImage(img)
        if self.evaluation:
            randtr = multchan.random_transform(augfun='raw')
        else:
            try:
                randtr = multchan.random_transform()
            except:
                randtr = multchan.random_transform(augfun='raw')
        
                ### transform to torch tensor
        
        if self.depthfirst:
            imgtensor = torch.from_numpy(randtr).float()
        else:
            imgtensor = torch.from_numpy(randtr.swapaxes(0,1)).float()
        
        targetten = torch.from_numpy(np.expand_dims(np.array(target), 0)).float()
        
        return imgtensor, targetten 