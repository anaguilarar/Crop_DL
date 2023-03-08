
import random
from pycocotools.coco import COCO

from .image_functions import image_rotation,image_zoom,randomly_displace,clahe_img, image_flip,shift_hsv
from .plt_utils import plot_segmenimages
import cv2
import os
import numpy as np

import pickle
import copy


def get_boundingboxfromseg(mask):
    
    pos = np.where(mask)
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    
    return([xmin, ymin, xmax, ymax])


def minmax_scale(data, minval = None, maxval = None, navalue = 0):
    
    if minval is None:
        minval = np.nanmin(data)
    if maxval is None:
        maxval = np.nanmax(data)
    
    if navalue == 0:
        ## mask na
        data[data == navalue] = np.nan
        dasc = (data - minval) / ((maxval - minval)) 
        dasc[np.isnan(dasc)] = navalue 
    else:
        dasc= (data - minval) / ((maxval - minval))
    
    return dasc

def standard_scale(data, meanval = None, stdval = None, navalue = 0):
    if meanval is None:
        meanval = np.nanmean(data)
    if stdval is None:
        stdval = np.nanstd(data)
    if navalue == 0:
        ## mask na
        data[data == navalue] = np.nan
        dasc = (data-meanval)/stdval 
        dasc[np.isnan(dasc)] = navalue 
    else:
        dasc= (data-meanval)/stdval

    return dasc

class MT_Imagery(object):

    @property
    def shape(self):
        dimsnames = self.dictdata['dims'].keys()
        dimsshape = [len(self.features)]
        for i in dimsnames:
            dimsshape.append(len(self.dictdata['dims'][i]))

        return tuple(dimsshape)

    def _checkfeaturesfilter(self,onlythesefeatures):

        if onlythesefeatures is not None:
            truefeat = []
            notfeat = []
            for featname in onlythesefeatures:
                if featname in self.features:
                    truefeat.append(featname)
                else:
                    notfeat.append(featname)
            if len(notfeat) > 0:
                print(f"there is not a feature with names {notfeat}")
            if len(truefeat) >0:
                self.features = truefeat


    def to_npvalues(self, dataformat = 'DCHW',chanelslast = False):

        matrix = np.empty(shape=self.shape, dtype=float)

        for i,featname in enumerate(self.features):
            matrix[i] = copy.deepcopy(self.dictdata['variables'][featname])
        
        if dataformat == 'DCHW':
            matrix = matrix.swapaxes(0,1)
        if chanelslast:
            for i in range(len(matrix.shape)-1):
                matrix = matrix.swapaxes(i,i+1)

        return matrix

    def __init__(self, data = None, fromfile = None, onlythesefeatures = None) -> None:
        
        
        if fromfile is not None:
            if os.path.exists(fromfile):

                with open(fromfile,"rb") as f:
                    self.dictdata = pickle.load(f)

            else:
                raise ValueError(f"{fromfile} doesn't exist")

        else:
            self.dictdata = data

        self.mainkeys = list(self.dictdata.keys())
        self.features = list(self.dictdata['variables'].keys())
        self._checkfeaturesfilter(onlythesefeatures)
        

def perform(fun, *args):
    return fun(*args)

def perform_kwargs(fun, **kwargs):
    return fun(**kwargs)


class FolderWithImages(object):
    @property
    def files_in_folder(self):
        
        return self._look_for_images()
    
    def __init__(self, path,suffix = '.jpg', shuffle = False, seed = None) -> None:
        
        self.path = path
        self.imgs_suffix = suffix
        self.shuffle = shuffle
        self.seed = seed
    
    def _look_for_images(self):
        filesinfolder = [i for i in os.listdir(self.path) if i.endswith(self.imgs_suffix)]
        if len(filesinfolder)==0:
            raise ValueError(f'there are not images in this {self.path} folder')

        if self.seed is not None and self.shuffle:
            random.seed(self.seed)
            random.shuffle(filesinfolder)

        elif self.shuffle:
            random.shuffle(filesinfolder)

        return filesinfolder


def summarise_trasstring(values):
    
    if type (values) ==  list:
        paramsnames = '_'.join([str(j) 
        for j in values])
    else:
        paramsnames = values

    return '{}'.format(
            paramsnames
        )

def _generate_shiftparams(img, max_displacement = None, xshift = None,yshift = None):

    if max_displacement is None:
        max_displacement = random.randint(5,20)/100
    
    if xshift is None:
        xoptions = list(range(-int(img.shape[0]*max_displacement),int(img.shape[0]*maxshift)))
        xshift = random.choice(xoptions)

    if yshift is None:
        yoptions = list(range(-int(img.shape[1]*max_displacement),int(img.shape[1]*maxshift)))
        yshift = random.choice(yoptions)

    return max_displacement, xshift, yshift

class ImageAugmentation(object):

    @property
    def available_transforms(self):
        return list(self._run_default_transforms.keys())
    
    @property
    def _run_default_transforms(self):
        
        return  {
                'rotation': self.rotate_image,
                'zoom': self.expand_image,
                'clahe': self.clahe,
                'shift': self.shift_ndimage,
                'multitr': self.multi_transform,
                'flip': self.flip_image,
                'hsv': self.hsv
            }
        


    @property
    def _augmented_images(self):
        return self._new_images

    @property
    def _random_parameters(self):
        params = None
        default_params = {
                'rotation': random.randint(10,350),
                'zoom': random.randint(-85, -65),
                'clahe': random.randint(0,30),
                'shift': random.randint(5, 20),
                'flip': random.choice([-1,0,1]),
                'hsv': [random.randint(0,30),
                        random.randint(0,30),
                        random.randint(0,30)]
                
            }
        
        if self._init_random_parameters is None:
            params = default_params

        else:
            params = self._init_random_parameters
            assert isinstance(params, dict)
            for i in list(self._run_default_transforms.keys()):
                if i not in list(params.keys()) and i != 'multitr':
                    params[i] = default_params[i]        
            
        return params    
        
    def multi_transform(self, img = None, 
                        chain_transform = ['flip','zoom','shift','rotation'],
                         params = None, update = True):

        if img is None:
            img = self.img_data

        imgtr = copy.deepcopy(img)
        augmentedsuffix = {}
        for i in chain_transform:
            if params is None:
                imgtr = perform_kwargs(self._run_default_transforms[i],
                     img = imgtr,
                     update = False)
            else:
                imgtr = perform(self._run_default_transforms[i],
                     imgtr,
                     params[i], False)

            augmentedsuffix[i] = self._transformparameters[i]
        
        self._transformparameters['multitr'] = augmentedsuffix
        if update:
            self.updated_paramaters(tr_type = 'multitr')
            self._new_images['multitr'] = imgtr

        return imgtr

    def updated_paramaters(self, tr_type):
        self.tr_paramaters.update({tr_type : self._transformparameters[tr_type]})
    
    def rotate_image(self, img = None, angle = None, update = True):

        
        if img is None:
            img = copy.deepcopy(self.img_data)
        if angle is None:
            angle = self._random_parameters['rotation']

        
        imgtr = image_rotation(img,angle = angle)
        self._transformparameters['rotation'] = angle
        if update:
            self.updated_paramaters(tr_type = 'rotation')
            self._new_images['rotation'] = imgtr

        return imgtr

    def hsv(self, img = None, hsvparams =None, update = True):
        if img is None:
            img = copy.deepcopy(self.img_data)
        if hsvparams is None:
            hsvparams = self._random_parameters['hsv']
            
        imgtr,_ = shift_hsv(img,hue_shift=hsvparams[0], sat_shift = hsvparams[1], val_shift = hsvparams[2])
        self._transformparameters['hsv'] = hsvparams
        
        if update:
            self.updated_paramaters(tr_type = 'hsv')
            self._new_images['hsv'] = imgtr

        return imgtr
    
    def flip_image(self, img = None, flipcode = None, update = True):

        if img is None:
            img = copy.deepcopy(self.img_data)
        if flipcode is None:
            flipcode = self._random_parameters['flip']

        
        imgtr = image_flip(img,flipcode = flipcode)
        self._transformparameters['flip'] = flipcode
        if update:
            self.updated_paramaters(tr_type = 'flip')
            self._new_images['flip'] = imgtr

        return imgtr

    def expand_image(self, img = None, ratio = None, update = True):
        if ratio is None:
            ratio = self._random_parameters['zoom']
            
        if img is None:
            img = copy.deepcopy(self.img_data)
            
        imgtr = image_zoom(img, zoom_factor=ratio)
        self._transformparameters['zoom'] = ratio
        if update:
            self.updated_paramaters(tr_type = 'zoom')
            self._new_images['zoom'] = imgtr

        return imgtr
    

    def shift_ndimage(self,img = None, shift = None, update = True,
                      max_displacement = None):

        if max_displacement is None:
            max_displacement = (self._random_parameters['shift'])/100
        if img is None:
            img = copy.deepcopy(self.img_data)

        if shift is not None:
            xshift, yshift= shift   
        else:
            xshift, yshift = None, None

        imgtr, displacement =  randomly_displace(img, 
                                                 maxshift = max_displacement, 
                                                 xshift = xshift, yshift = yshift)

        self._transformparameters['shift'] = displacement
        if update:
            self.updated_paramaters(tr_type = 'shift')
            self._new_images['shift'] = imgtr#.astype(np.uint8)

        return imgtr#.astype(np.uint8)
    
    def clahe(self, img= None, thr_constrast = None, update = True):

        if thr_constrast is None:
            thr_constrast = self._random_parameters['clahe']/10
        
        if img is None:
            img = copy.deepcopy(self.img_data)

        imgtr,_ = clahe_img(img, clip_limit=thr_constrast)

        self._transformparameters['clahe'] = thr_constrast
        if update:
            self.updated_paramaters(tr_type = 'clahe')
            self._new_images['clahe'] = imgtr
        return imgtr

    def random_augmented_image(self,img= None, update = True):
        if img is None:
            img = copy.deepcopy(self.img_data)
        
        imgtr = copy.deepcopy(img)
        augfun = random.choice(list(self._run_default_transforms.keys()))
        
        imgtr = perform_kwargs(self._run_default_transforms[augfun],
                     img = imgtr,
                     update = update)

        return imgtr

    def _transform_as_ids(self, tr_type):

        if type (self.tr_paramaters[tr_type]) ==  dict:
            paramsnames= ''
            for j in list(self.tr_paramaters[tr_type].keys()):
                paramsnames += 'ty_{}_{}'.format(
                    j,
                    summarise_trasstring(self.tr_paramaters[tr_type][j]) 
                )

        else:
            paramsnames = summarise_trasstring(self.tr_paramaters[tr_type])

        return '{}_{}'.format(
                tr_type,
                paramsnames
            )
    
    def augmented_names(self):
        transformtype = list(self.tr_paramaters.keys())
        augmentedsuffix = {}
        for i in transformtype:
            
            augmentedsuffix[i] = self._transform_as_ids(i)

        return augmentedsuffix

    def __init__(self, img, random_parameters = None) -> None:

        self.img_data = None
        if isinstance(img, str):
            self.img_data = cv2.imread(img)
        else:
            self.img_data = copy.deepcopy(img)

        self._transformparameters = {}
        self._new_images = {}
        self.tr_paramaters = {}
        self._init_random_parameters = random_parameters

class ImageData(ImageAugmentation):
    

    @property
    def images_names(self):
        imgnames = {'raw': self.orig_imgname}
        augementednames = self.augmented_names()
        if len(list(augementednames.keys()))> 0:
            for datatype in list(augementednames.keys()):
                currentdata = {datatype: '{}_{}'.format(
                    imgnames['raw'],
                    augementednames[datatype])}
                imgnames.update(currentdata)

        return imgnames

    @property
    def imgs_data(self):
        imgdata = {'raw': self.img_data}
        augementedimgs = self._augmented_images
        if len(list(augementedimgs.keys()))> 0:
            for datatype in list(augementedimgs.keys()):
                currentdata = {datatype: augementedimgs[datatype]}
                imgdata.update(currentdata)

        return imgdata
    
    def resize_image(self, newsize = None):
        resized = cv2.resize(self.img_data['raw'], newsize, interpolation = cv2.INTER_AREA)
        
    def __init__(self, path, img_id = None, **kwargs) -> None:
        
        if img_id is not None:
            self.orig_imgname = img_id
        else:
            self.orig_imgname = "image"

        try:
            img = cv2.imread(path)
            super().__init__(img, **kwargs)
        
        except:
            raise ValueError('check image filename')
    


class MultiChannelImage(ImageAugmentation):

    @property
    def imgs_data(self):
        imgdata = {'raw': self.mlchannel_data}
        augementedimgs = self._augmented_images
        if len(list(augementedimgs.keys()))> 0:
            for datatype in list(augementedimgs.keys()):
                currentdata = {datatype: augementedimgs[datatype]}
                imgdata.update(currentdata)

        return imgdata
    
    @property
    def images_names(self):
        imgnames = {'raw': self.orig_imgname}
        augementednames = self.augmented_names()
        if len(list(augementednames.keys()))> 0:
            for datatype in list(augementednames.keys()):
                currentdata = {datatype: '{}_{}'.format(
                    imgnames['raw'],
                    augementednames[datatype])}
                imgnames.update(currentdata)

        return imgnames
    
    def _tranform_channelimg_function(self, img, tr_name):

        if tr_name == 'multitr':
            params = self.tr_paramaters[tr_name]
            image = self.multi_transform(img=img,
                                chain_transform = list(params.keys()),
                                params= params, update= False)

        else:
            
            image =  perform(self._run_default_transforms[tr_name],
                     img,
                     self.tr_paramaters[tr_name], False)

        return image

    def _transform_multichannel(self, img=None, tranformid = None,  **kwargs):
        
        newimgs = {}
        if img is not None:
            trimgs = img
        else:
            trimgs = self.mlchannel_data

        imgs= [perform_kwargs(self._run_default_transforms[tranformid],
                     img = trimgs[0],
                     **kwargs)]
         
        for i in range(1,trimgs.shape[0]):
            r = self._tranform_channelimg_function(trimgs[i],tranformid)
            imgs.append(r)

        imgs = np.stack(imgs, axis = 0)
        self._new_images[tranformid] = imgs
        
        return imgs

    def shift_multiimages(self, img=None, shift=None, max_displacement=None,update=True):

        self._new_images['shift'] =  self._transform_multichannel(img=img, 
                    tranformid = 'shift', shift = shift, max_displacement=max_displacement,update=update)
        return self._new_images['shift']

    def rotate_multiimages(self, img=None, angle=None, update=True):
        self._new_images['rotation'] = self._transform_multichannel(img=img, 
                    tranformid = 'rotation', angle = angle, update=update)
        
        return self._new_images['rotation']
    
    def flip_multiimages(self, img=None, flipcode=None, update=True):
        self._new_images['flip'] = self._transform_multichannel(img=img, 
                    tranformid = 'flip', flipcode = flipcode, update=update)
        
        return self._new_images['flip']
    
    
    def expand_multiimages(self, img=None, ratio=None, update=True):

        self._new_images['zoom'] = self._transform_multichannel(img=img, 
                    tranformid = 'zoom', ratio = ratio, update=update)
        
        return self._new_images['zoom']


    def multitr_multiimages(self, img=None, 
                        chain_transform=['flip','zoom','shift','rotation'], 
                        params=None, 
                        update=True):
        
        self._new_images['multitr'] = self._transform_multichannel(
                    img=img, 
                    tranformid = 'multitr', 
                    chain_transform = chain_transform,
                    params=params, update=update)
        
        return self._new_images['multitr']


    def __init__(self, img, img_id = None, channels_order = 'first', **kwargs) -> None:

        if img_id is not None:
            self.orig_imgname = img_id
        else:
            self.orig_imgname = "image"

        if channels_order == 'first':
            self._initimg = img[0]
        else:
            self._initimg = img[:,:,0]
        
        self.mlchannel_data = img

        super().__init__(self._initimg, **kwargs)
    

def scale_mtdata(npdata,
                 features,
                       scaler=None, 
                       scale_z = True,
                       name_3dfeature = 'z', 
                       method = 'minmax',
                       shapeorder = 'DCHW'):

    datamod = npdata.copy()
    if shapeorder == 'DCHW':
        datamod = datamod.swapaxes(0,1)


    if scaler is not None:
        
        datascaled = []

        for i, varname in enumerate(features):
            if method == 'minmax':
                scale1, scale2 = scaler
                if scale_z and varname == name_3dfeature:
                        datascaled.append(minmax_scale(
                            datamod[i],
                            minval = scale1[varname], 
                            maxval = scale2[varname]))

                elif varname != name_3dfeature:
                    scale1[varname]
                    datascaled.append(minmax_scale(
                            datamod[i],
                            minval = scale1[varname], 
                            maxval = scale2[varname]))
                else:
                    datascaled.append(datamod[i])
            
            elif method == 'normstd':
                scale1, scale2 = scaler
                if scale_z and varname == name_3dfeature:
                        datascaled.append(standard_scale(
                            datamod[i],
                            meanval = scale1[varname], 
                            stdval = scale2[varname]))

                elif varname != name_3dfeature:
                    scale1[varname]
                    datascaled.append(standard_scale(
                            datamod[i],
                            meanval = scale1[varname], 
                            stdval = scale2[varname]))
                else:
                    datascaled.append(datamod[i])

    if shapeorder == 'DCHW':
        datascaled = np.array(datascaled).swapaxes(0,1)
    else:
        datascaled = np.array(datascaled)

    return datascaled


def perform(fun, *args):
    return fun(*args)

def perform_kwargs(fun, **kwargs):
    return fun(**kwargs)

class SegmentationImages(ImageData):

    
    _maskedpattern = 'mask'
    _masksuffix = '.png'
    _imsgs_suffix = '.jpg'
    _list_not_transform = ['clahe']



    def random_multime_transform(self, augfun = None, verbose = False):
        availableoptions = list(self._run_default_transforms.keys())+['raw']
        
        if augfun is None:
            augfun = random.choice(availableoptions)
        elif type(augfun) is list:
            augfun = random.choice(augfun)
        
        if augfun not in availableoptions:
            print(f"""that augmentation option is not into default parameters {availableoptions},
                     no transform was applied""")
            augfun = 'raw'

        if augfun == 'raw':
            imgtr = self.target_data['raw']
            #imgtarget = self.target_data['raw'][:,:,0]
        else:
            imgtr = perform_kwargs(self._run_default_transforms[augfun])

        if verbose:
            print('{} was applied'.format(augfun))
        #imgtr = self._scale_image(imgtr)
        return imgtr



    def tranform_function(self, tr_name):

        if tr_name == 'multitr':
            params = self.tr_paramaters[tr_name]
            image = self.multi_transform(img=self._mask_imgid,
                                chain_transform = list(params.keys()),params= params, update= False)

        else:
            image =  perform(self._run_default_transforms[tr_name],
                     self._mask_imgid,
                     self.tr_paramaters[tr_name], False)

        return image
    
    def compute_transformfortargets(self, fun_name):

        if fun_name not in self._list_not_transform:
            img = perform(self._run_default_transforms[fun_name],
                     self._mask_imgid,
                     self.tr_paramaters[fun_name], False)
        else:
            img = self._mask_imgid
        
        return img

    def transform_target_image(self):
        trids = list(self.images_names.keys())
        newimgs = {}
        if len(trids)> 1:
            for i in range(1,len(trids)):
                if trids[i] not in self._list_not_transform:
                    newimgs[trids[i]] = self.tranform_function(trids[i])

                else:
                    newimgs[trids[i]] = self._mask_imgid
        
        return newimgs
    
    def __init__(self, input_path, target_path = None, img_id = None, **kwargs) -> None:
        self.input_path = input_path
        super().__init__(path = os.path.join(input_path, img_id), img_id = img_id, **kwargs)


    @property
    def target_data(self):
        imgs_data = {'raw': self._mask_imgid}
        newdata = self.transform_target_image()
        if len(list(newdata.keys()))>0:
            imgs_data.update(newdata)

        return imgs_data

    @property
    def _mask_imgid(self):
        imgid = self.images_names['raw']

        imgid = imgid[0:imgid.index(self._imsgs_suffix)]
        fn = '{}_{}{}'.format(
                imgid,
                self._maskedpattern,
                self._masksuffix)
        
        try:
            img = cv2.imread(os.path.join(self.input_path,fn))

        except:
            raise ValueError(f"check segmentation image file name {fn}")

        return img
    

### instance segmentation

class InstanceSegmentation(ImageData):
    
    _cocosuffix = '.json'
    ### available
    #rotation': self.rotate_image,
    #            'zoom': self.expand_image,
    #            'clahe': self.clahe,
    #            'shift': self.shift_ndimage,
    #            'multitr': self.multi_transform,
    #            'flip':
    _list_transforms = []
    ## the transform functions that affect the color space wwon't be applied to the target segmentation
    _list_not_transform = ['clahe', 'hsv']
    
    @property
    def imgs_len(self):
        
        return self._cocoimgs_len()
    
    @property
    def bounding_boxes(self):
        boxes = {}
        for imgtype in list(self.target_data.keys()):
            boxes[imgtype] = np.array(
                [get_boundingboxfromseg(mask) for mask in self.target_data[imgtype]])
            
        return boxes
            
    @property
    def target_data(self):
        imgs_data = {'raw': self._mask_imgid}
        newdata = self.transform_target_image()
        if len(list(newdata.keys()))>0:
            imgs_data.update(newdata)

        return imgs_data


    @property
    def _mask_imgid(self):
        ### it must be a cocodataset
        assert self.cocodataset
            
        masks = np.array([np.array(self._annotation_cocodata.annToMask(ann) * ann["category_id"]
                        ) for ann in self.anns])

        return masks

    def random_multime_transform(self, augfun = None, verbose = False):
        availableoptions = list(self._run_default_transforms.keys())+['raw']
        
        if augfun is None:
            augfun = random.choice(availableoptions)
        elif type(augfun) is list:
            augfun = random.choice(augfun)
        
        if augfun not in availableoptions:
            print(f"""that augmentation option is not into default parameters {availableoptions},
                     no transform was applied""")
            augfun = 'raw'

        if augfun == 'raw':
            imgtr = self.target_data['raw']
            #imgtarget = self.target_data['raw'][:,:,0]
        else:
            imgtr = perform_kwargs(self._run_default_transforms[augfun])

        if verbose:
            print('{} was applied'.format(augfun))
        #imgtr = self._scale_image(imgtr)
        return imgtr



    def tranform_function(self, tr_name):
        """compute transformation for targets"""
        trmask = np.zeros(self._mask_imgid.shape)
        
        for i,mask in enumerate(self._mask_imgid):
            
            if tr_name == 'multitr':
                params = self.tr_paramaters[tr_name]
                trmask[i] = self.multi_transform(img=mask,
                                    chain_transform = list(params.keys()),
                                    params= params, update= False)

            else:
                trmask[i]  =  perform(self._run_default_transforms[tr_name],
                        mask,
                        self.tr_paramaters[tr_name], False)

        return trmask
    

    def transform_target_image(self):
        trids = list(self.images_names.keys())
        newimgs = {}
        if len(trids)> 1:
            for i in range(1,len(trids)):
                if trids[i] not in self._list_not_transform:
                    newimgs[trids[i]] = self.tranform_function(trids[i])

                else:
                    newimgs[trids[i]] = self._mask_imgid
        
        return newimgs
    
    def plot_output(self,datatype, **kwargs):
        if datatype in list(self.target_data.keys()):
                
            f = plot_segmenimages(self.imgs_data[datatype],np.max(np.stack(
                    self.target_data[datatype]), axis = 0)*255, 
                        boxes=self.bounding_boxes[datatype], 
                        bbtype = 'xminyminxmaxymax',
                        **kwargs)
            
        return f
    
    def _cocoimgs_len(self):
        tmpbool = True
        i = 0
        while tmpbool:
            try:
                self._annotation_cocodata.loadImgs(i)
                i+=1
                
            except:
                tmpbool = False
        return i-1
        
    def __init__(self, input_path, annotation, img_id = None, 
                    cocodataset = True, **kwargs) -> None:
        
        
        self.input_path = input_path
        
        self.cocodataset = cocodataset
        if cocodataset:
            
            ## must be a coco file
            assert isinstance(annotation, COCO)
            
            self._annotation_cocodata = copy.deepcopy(annotation)

            img_data = self._annotation_cocodata.loadImgs(img_id)
            #print(img_data)
            fn = img_data[0]['file_name']
            #print(len(img_data) )
            ### it must have length 1
            assert len(img_data) == 1
            
            self.annotation_ids = self._annotation_cocodata.getAnnIds(
                        imgIds=img_data[0]['id'], 
                        catIds=1, 
                        iscrowd=None
                    )
            self.anns = self._annotation_cocodata.loadAnns(self.annotation_ids)
            self.numobjs = len(self.anns)
            
            super(InstanceSegmentation, self).__init__(
                path = os.path.join(self.input_path, 
                                                 fn), 
                             img_id = fn, **kwargs)
            

    