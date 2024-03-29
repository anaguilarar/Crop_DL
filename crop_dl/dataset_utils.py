
import random

from .image_functions import image_rotation,image_zoom,randomly_displace,clahe_img, image_flip,shift_hsv,diff_guassian_img,illumination_shift
from .plt_utils import plot_segmenimages
import cv2
import os
import numpy as np

import pickle
import copy
from typing import List, Optional, Union

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
    """
    A class for performing image augmentation through various transformations like rotation, zoom, etc.
    Allows randomization of parameters for each transformation and maintains a history of applied transformations.
    """

    
    def __init__(self, img: Union[str, np.ndarray], 
                 random_parameters: Optional[dict] = None, 
                 multitr_chain: Optional[List[str]] = None) -> None:
        """
        Initializes the ImageAugmentation class with an image and optional transformation parameters.

        Parameters:
        ----------
        img : str or ndarray
            The path to an image file or the image data itself.
        random_parameters : dict, optional
            Custom parameters for random transformations.
        multitr_chain : list, optional
            A predefined chain of transformations to apply.
        """
        
        self._transformparameters = {}
        self._new_images = {}
        self.tr_paramaters = {}
        
        self.img_data = None
        self.img_data = cv2.imread(img) if isinstance(img, str) else copy.deepcopy(img)

        self._init_random_parameters = random_parameters
        self._multitr_chain = multitr_chain
        
    
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
                'hsv': self.hsv,
                #'gaussian': self.diff_gaussian_image,
                'illumination':self.change_illumination
            
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
                'gaussian': random.choice([20,30,40, 50]),
                'hsv': [list(range(-30,30,5)),
                        list(range(-20,20,5)), 
                        list(range(-20,20,5))],
                'illumination':random.choice(list(range(-50,50,5)))
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
    
    def _select_random_transforms(self):
        """
        Randomly selects a set of transformations for multi-transform.

        Returns:
        -------
        list
            A list of randomly selected transformation names.
        """
        chain_transform = []
        while len(chain_transform) <= 3:
            trname = random.choice(list(self._run_default_transforms.keys()))
            if trname != 'multitr':
                chain_transform.append(trname)
        return chain_transform

    def updated_paramaters(self, tr_type):
        """
        this function updates the tranformation dictionary information
        
        Parameters:
        ----------
        tr_type : str, optional
            transformation name
        """
        self.tr_paramaters.update({tr_type : self._transformparameters[tr_type]})
    
    
    #['flip','zoom','shift','rotation']
    def multi_transform(self, img: Union[str, np.ndarray] = None, 
                        chain_transform: Optional[List[str]] = None,
                         params: Optional[dict] = None, update: bool = True) -> np.ndarray:

        """
        Applies a chain of multiple transformations to the image.

        Parameters:
        ----------
        img : ndarray, optional
            The image to transform. Uses the class's internal image data if None.
        chain_transform : list, optional
            A list of transformation names to apply.
        params : dict, optional
            Parameters for each transformation in the chain.
        update : bool, optional
            If True, updates the class's internal state with the result.

        Returns:
        -------
        ndarray
            The transformed image.
        """
        
        # Selecting transformations if not provided
        if chain_transform is None:
            chain_transform = self._multitr_chain if self._multitr_chain is not None else self._select_random_transforms()
                 
        if img is None:
            img = self.img_data

        imgtr = copy.deepcopy(img)
        augmentedsuffix = {}
        
        for transform_name in chain_transform:
            if params is None:
                imgtr = perform_kwargs(self._run_default_transforms[transform_name],
                     img = imgtr,
                     update = False)
            else:
                
                imgtr = perform(self._run_default_transforms[transform_name],
                     imgtr,
                     params[transform_name], False)
            #if update:
            augmentedsuffix[transform_name] = self._transformparameters[transform_name]
        
        self._transformparameters['multitr'] = augmentedsuffix
         
        if update:
            
            self.updated_paramaters(tr_type = 'multitr')
            self._new_images['multitr'] = imgtr

        return imgtr


    def apply_transformation(self, transform_func, img=None, transform_param=None, 
                             transform_name=None, update=True):
        """
        Applies a specific image transformation function.

        Parameters:
        ----------
        transform_func : function
            The specific transformation function to apply.
        img : ndarray, optional
            The image to be transformed. If None, uses the class's internal image data.
        transform_param : various types, optional
            The parameter specific to the transformation.
        transform_name : str, optional
            The name of the transformation (for internal tracking).
        update : bool, optional
            If True, updates the class's internal state with the result.

        Returns:
        -------
        ndarray
            The transformed image.
        """

        if img is None:
            img = self.img_data

        if transform_param is None and transform_name:
            transform_param = self._random_parameters.get(transform_name, None)

        # Validate that the function is callable
        if not callable(transform_func):
            raise ValueError("Provided function is not callable.")
        img_transformed = transform_func(img, transform_param)

        if update and transform_name:
            self._transformparameters[transform_name] = transform_param
            self.updated_paramaters(tr_type=transform_name)
            self._new_images[transform_name] = img_transformed

        return img_transformed

    def diff_gaussian_image(self, img: Union[str, np.ndarray] = None, 
                            high_sigma: Optional[float] = None, update: bool = True):
        """
        Applies a differential Gaussian filter to the image.

        Parameters:
        ----------
        img : ndarray, optional
            The image to be processed. If None, uses the class's internal image data.
        high_sigma : float, optional
            The standard deviation for the Gaussian kernel. If None, a random value is chosen based on class parameters.
        update : bool, optional
            If True, updates the class's internal state with the result.

        Returns:
        -------
        ndarray
            The image after applying the differential Gaussian filter.
        """
        
        if img is None:
            img = copy.deepcopy(self.img_data)
            
        if high_sigma is None:
            high_sigma = self._random_parameters['gaussian']

        high_sigma = max(min(high_sigma, 100), 0.1)
        
        imgtr,_ = diff_guassian_img(img,high_sigma = high_sigma)
        self._transformparameters['gaussian'] = high_sigma
        
        if update:
            
            self.updated_paramaters(tr_type = 'gaussian')
            self._new_images['gaussian'] = imgtr

        return imgtr
    
    
    def rotate_image(self, img = None, angle = None, update = True):
        """
        Rotates the given image by a specified angle.

        Parameters:
        ----------
        img : ndarray, optional
            The image to be rotated. If None, uses the class's internal image data.
        angle : int, optional
            The angle in degrees for rotating the image. If None, a random angle is chosen based on class parameters.
        update : bool, optional
            If True, updates the class's internal state with the result.

        Returns:
        -------
        ndarray:
            The rotated image.

        Raises:
        ------
        ValueError:
            If the input image is not in the expected format or dimensions.
        """
        
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
    
    def change_illumination(self, img = None, illuminationparams =None, update = True):
        if img is None:
            img = copy.deepcopy(self.img_data)
        if illuminationparams is None:
            illuminationparams = self._random_parameters['illumination']
            
        imgtr = illumination_shift(img,valuel = illuminationparams)
        
        self._transformparameters['illumination'] = illuminationparams
        if update:
            
            self.updated_paramaters(tr_type = 'illumination')
            self._new_images['illumination'] = imgtr

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

        assert os.path.exists(path) #check image filename
        img = cv2.imread(path)
        super().__init__(img, **kwargs)


class MultiChannelImage(ImageAugmentation):

        
    """
    A subclass of ImageAugmentation designed to handle multi-channel images represented as 3D numpy arrays.

    This class provides functionality to apply various image transformations, specifically tailored for images
    with multiple channels, such as RGB, multispectral, or other types of imaging data.

    Attributes:
    ----------
    orig_imgname : str
        The original name of the image.
    mlchannel_data : numpy.ndarray
        The multi-channel image data.
    tr_paramaters : dict
        A dictionary of parameters for each transformation.
    _new_images : dict
        A dictionary containing transformed image data.

    Methods:
    -------
    Various transformation methods like rotate_multiimages, flip_multiimages, etc., are provided
    to apply respective transformations to multi-channel images.
    """
    
    def __init__(self, img, img_id = None, channels_order = 'first', transforms = None, **kwargs) -> None:
        """
        Initializes the MultiChannelImage class with a multi-channel image and additional parameters.

        Parameters:
        ----------
        img : numpy.ndarray
            The multi-channel image data to be processed.
        img_id : str, optional
            An identifier or name for the image. Default None.
        channels_order : str, optional
            Specifies the ordering of the channels in the input image. Accepted values are 'first' or 'last'.
            'first' indicates channels are in the first dimension, 'last' indicates channels are in the last dimension.
        transforms : list, optional
            A list of transformation names to be available for the image. If None, all default transformations are used.
        **kwargs : dict
            Additional keyword arguments to be passed to the parent ImageAugmentation class.

        Raises:
        ------
        ValueError
            If the input image is not a numpy ndarray or if channels_order is not recognized.
        """
        
        if channels_order not in ['first', 'last']:
            raise ValueError("channels_order must be either 'first' or 'last'.")

        ### set random transforms
        self._availableoptions = list(self._run_multichannel_transforms.keys())+['raw'] if transforms is None else transforms+['raw']

        self.scaler = None
        self.orig_imgname = img_id or "image"
        self._initimg = img[0] if channels_order == 'first' else img[:,:,0]
       
        self.mlchannel_data = img

        super().__init__(self._initimg, **kwargs)
        
    @property
    def _run_multichannel_transforms(self):
        return  {
                'rotation': self.rotate_multiimages,
                'zoom': self.expand_multiimages,
                'illumination': self.diff_illumination_multiimages,
                #'gaussian': self.diff_guassian_multiimages,
                'shift': self.shift_multiimages,
                'multitr': self.multitr_multiimages,
                'flip': self.flip_multiimages
            }
    

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
    
    def _scale_multichannels_data(self,img, method = 'standarization'):
        """
        Scales multi-channel image data using the specified method.

        Parameters:
        ----------
        img : numpy.ndarray
            The multi-channel image data to be scaled.
        method : str, optional
            The scaling method to apply. Supported methods: 'standarization'.

        Returns:
        -------
        numpy.ndarray
            The scaled multi-channel image data.
        """
        
        if self.scaler is not None:
                        
            if method == 'standarization':
                
                datascaled = []
                for z in range(self.mlchannel_data.shape[0]):
                    datascaled.append(standard_scale(
                            img[z].copy(),
                            meanval = self.scaler[z][0], 
                            stdval = self.scaler[z][1]))

        else:
            datascaled = img.copy()
            
        return datascaled
    
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
        """
        Applies a transformation to each channel of the multi-channel image.

        Parameters:
        ----------
        img : numpy.ndarray, optional
            The multi-channel image data to transform. Uses class's image data if None.
        transform_id : str
            The identifier of the transformation to apply.
        **kwargs
            Additional keyword arguments for the transformation.

        Returns:
        -------
        numpy.ndarray
            The transformed multi-channel image data.
        """
        
        if img is not None:
            trimgs = img
        else:
            trimgs = self.mlchannel_data
        if tranformid != 'illumination':
            
            imgs= [perform_kwargs(self._run_default_transforms[tranformid],
                        img = trimgs[0],
                        **kwargs)]
            
            for i in range(1,trimgs.shape[0]):
                r = self._tranform_channelimg_function(trimgs[i],tranformid)
                imgs.append(r)

            imgs = np.stack(imgs, axis = 0)
            
        else:
            imgs= perform_kwargs(self._run_default_transforms[tranformid],
                        img = trimgs,
                        **kwargs)
            
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
    
    def diff_guassian_multiimages(self, img=None, high_sigma=None, update=True):
        self._new_images['gaussian'] = self._transform_multichannel(img=img, 
                    tranformid = 'gaussian', high_sigma = high_sigma, update=update)
        
        return self._new_images['gaussian']
    
    def diff_illumination_multiimages(self, img=None, illuminationparams=None, update=True):
        self._new_images['illumination'] = self._transform_multichannel(img=img, 
                    tranformid = 'illumination', illuminationparams = illuminationparams, update=update)
        
        return self._new_images['illumination']
    
    #def clahe_multiimages(self, img= None, thr_constrast = None, update = True):
    #    self._new_images['clahe'] = self._transform_multichannel(img=img, 
    #                tranformid = 'clahe', thr_constrast = thr_constrast, update=update)
    #    
    #    return self._new_images['clahe']
        
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

    def random_transform(self, augfun = None, verbose = False):
                
        if augfun is None:
            augfun = random.choice(self._availableoptions)
        elif type(augfun) is list:
            augfun = random.choice(augfun)
        
        if augfun not in self._availableoptions:
            print(f"""that augmentation option is not into default parameters {self._availableoptions},
                     no transform was applied""")
            augfun = 'raw'
        
        
        if augfun == 'raw':
            imgtr = self.mlchannel_data.copy()
            imgtr = self._scale_multichannels_data(imgtr)
            
        else:
            imgtr = perform_kwargs(self._run_multichannel_transforms[augfun])
            imgtr = self._scale_multichannels_data(imgtr)
        
        if verbose:
            print('{} was applied'.format(augfun))
        #imgtr = self._scale_image(imgtr)
        return imgtr
    
    

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



class MultiTimeTransform(MultiChannelImage):


    @property
    def _run_random_choice(self):
        return  {
                'rotation': self.rotate_tempimages,
                'zoom': self.expand_tempimages,
                'shift': self.shift_multi_tempimages,
                'illumination': self.illumination_tempimages,
                #'gaussian': self.diff_guassian_tempimages,
                'multitr': self.multtr_tempimages,
                'flip': self.flip_tempimages
            }

    def _scale_image(self, img):
        

        if self.scaler_params is not None:
            ## data D C X Y
            img= scale_mtdata(img,self.features,
                                self.scaler_params['scaler'],
                                scale_z = self.scaler_params['scale_3dimage'], 
                                name_3dfeature = self.scaler_params['name_3dfeature'], 
                                method=self.scaler_params['method'],
                                shapeorder = self._formatorder)

        return img


    def _multi_timetransform(self, tranformn, changeparamsmlt = False, **kwargs):
        imgs= [perform_kwargs(self._run_multichannel_transforms[tranformn],
                     img = self._initdate,
                     **kwargs)]

        for i in range(1,self.npdata.shape[1]):

            if tranformn != 'multitr':
                if changeparamsmlt:
                    params = self._random_parameters[tranformn]
                else:
                    params = self.tr_paramaters[tranformn]
                
                r = perform(self._run_multichannel_transforms[tranformn],
                               self.npdata[:,i,:,:],
                               params,
                               False,
                               )
            else:
                
                r = perform_kwargs(self._run_multichannel_transforms[tranformn],
                               img=self.npdata[:,i,:,:],
                               params = self.tr_paramaters[tranformn],
                               update = False,
                               )
            imgs.append(r)

        imgs = np.stack(imgs,axis=1)

        #imgs = self._scale_image(imgs)
        imgs = self._return_orig_format(imgs)
        self.npdata = copy.deepcopy(self._raw_img)
        
        return imgs

    def shift_multi_tempimages(self, shift=None, max_displacement=None):
        return self._multi_timetransform(tranformn = 'shift',
                                        shift= shift, 
                                        max_displacement= max_displacement)

    def diff_guassian_tempimages(self, high_sigma=None):
        return self._multi_timetransform(tranformn = 'gaussian', high_sigma = high_sigma)
    
    def expand_tempimages(self, ratio=None):
        return self._multi_timetransform(tranformn = 'zoom', ratio = ratio)

    def rotate_tempimages(self, angle=None):
        return self._multi_timetransform(tranformn = 'rotation', angle = angle)
    
    def flip_tempimages(self, flipcode=None):
        return self._multi_timetransform(tranformn = 'flip', flipcode = flipcode)
        
    def illumination_tempimages(self, illuminationparams=None):
        return self._multi_timetransform(tranformn = 'illumination', changeparamsmlt=True,illuminationparams = illuminationparams)

    def multtr_tempimages(self, img=None, chain_transform=['flip','zoom','shift', 'rotation'], params=None):
        return self._multi_timetransform(tranformn = 'multitr', 
                                         chain_transform= chain_transform, 
                                         params = params)

        
    def random_multime_transform(self, augfun = None, verbose = False):
        availableoptions = self._availableoptions
        
        if augfun is None:
            augfun = random.choice(availableoptions)
        elif type(augfun) is list:
            augfun = random.choice(augfun)
        
        if augfun not in availableoptions:
            print(f"""that augmentation option is not into default parameters {availableoptions},
                     no transform was applied""")
            augfun = 'raw'

        if augfun == 'illumination' and self.npdata.shape[0]!=3:
            augfun = 'raw'
            
        if augfun == 'raw':
            imgtr = self.npdata#.swapaxes(0,1)
            imgtr = self._return_orig_format(imgtr)
        else:
            imgtr = perform_kwargs(self._run_random_choice[augfun])

        if verbose:
            print('{} was applied'.format(augfun))
        
        
        #imgtr = self._return_orig_format(imgtr)
        
        return imgtr

    def _return_orig_format(self, imgtr):

        if self._orig_formatorder == "DCHW":
            imgtr = np.einsum('CDHW->DCHW', imgtr)
        
        if self._orig_formatorder == "HWCD":
            imgtr = np.einsum('CDHW->HWCD', imgtr)
        
        return imgtr
        

    
    def __init__(self, 
                 data=None, 
                 onlythesedates = None,
                 img_id=None, 
                 formatorder = 'DCHW',
                 channelslast = False,
                 removenan = True,
                 image_scaler = None,
                 scale_3dimage = False,
                 name_3dfeature = 'z',
                 scale_method = 'minmax',
                 transform_options = None,
                 **kwargs) -> None:
        
        """
        transform multitemporal data

        Args:
            data (_type_, optional): _description_. Defaults to None.
            onlythesedates (_type_, optional): _description_. Defaults to None.
            img_id (_type_, optional): _description_. Defaults to None.
            formatorder (str, optional): _description_. Defaults to 'DCHW'.
            channelslast (bool, optional): _description_. Defaults to False.
            removenan (bool, optional): _description_. Defaults to True.
            image_scaler (_type_, optional): _description_. Defaults to None.
            scale_3dimage (bool, optional): _description_. Defaults to False.
            name_3dfeature (str, optional): _description_. Defaults to 'z'.
            scale_method (str, optional): _description_. Defaults to 'minmax'.
        """

        if image_scaler is not None:
            self.scaler_params = {'scaler':image_scaler,
                                  'method': scale_method,
                                  'scale_3dimage': scale_3dimage,
                                  'name_3dfeature': name_3dfeature}
        else:
            self.scaler_params = None

        self.npdata = copy.deepcopy(data)

        
        if removenan:
            self.npdata[np.isnan(self.npdata)] = 0
        self._orig_formatorder = formatorder
        self._formatorder = "CDHW"
        
        if self._orig_formatorder == "DCHW":
            self.npdata = self.npdata.swapaxes(1,0)
            channelsorder = 'first'
        elif self._orig_formatorder == "CDHW":
            channelsorder = 'first'
        elif channelslast:
            channelsorder = 'last'

        if onlythesedates is not None:
            self.npdata = self.npdata[:,onlythesedates,:,:]

        self._raw_img = copy.deepcopy(self.npdata)

        #if image_scaler is not None:
        #    self.npdata = scale_mtdata(self,[image_scaler[0],image_scaler[1]],
        #                            scale_z = scale_3dimage, name_3dfeature = name_3dfeature, method=scale_method)

        self._initdate = copy.deepcopy(self.npdata[:,0,:,:])

        MultiChannelImage.__init__(self,img = self._initdate, img_id= img_id, 
                                   channels_order = channelsorder, transforms= transform_options,**kwargs)
        



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


class SegmentationImagesCoCo(ImageData):

    
    _cocosuffix = '.json'
    _list_transforms = []
    _list_not_transform = ['clahe', 'hsv']
    
    @property
    def imgs_len(self):
        
        return self._cocoimgs_len()
    
            
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
        
        if len(masks.shape) == 1:
            masks =  np.expand_dims(np.zeros(self.img_data.shape[:2]), axis = 0)    
    
        return masks

    def random_multime_transform(self, augfun = None, verbose = False):
        if self._onlythesetransforms is None:
            availableoptions = list(self._run_default_transforms.keys())+['raw']
        else:
            availableoptions = self._onlythesetransforms+['raw']
        
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
        if len(self._mask_imgid.shape) == 1:
            trmask =  np.expand_dims(np.zeros(self.img_data.shape[:2]), axis = 0)    
        
        else:
            trmask = np.zeros(self._mask_imgid.shape)
        
        for i,mask in enumerate(self._mask_imgid):
            
            if tr_name == 'multitr':
                
                params = self.tr_paramaters[tr_name]
                paramslist = [partr for partr in list(params.keys()) if partr not in self._list_not_transform]
                filteredparams = {}
                for partr in paramslist:
                    filteredparams[partr] = params[partr]
                    
                imgtr = mask.copy()

                for partr in paramslist:
                    #print(partr)
                    imgtr = perform(self._run_default_transforms[partr],
                            imgtr,
                            params[partr], False)
                
                trmask[i] = imgtr

            else:
                trmask[i]  =  perform(self._run_default_transforms[tr_name],
                        mask.copy(),
                        self.tr_paramaters[tr_name], False)

        return trmask
    

    def transform_target_image(self):
        trids = list(self.images_names.keys())
        newimgs = {}
        if len(trids)> 1:
            for i in range(1,len(trids)):
                if trids[i] not in self._list_not_transform:
                    newtr = self.tranform_function(trids[i])
                    newimgs[trids[i]] = newtr

                else:
                    newimgs[trids[i]] = self._mask_imgid
        
        return newimgs
    
    def plot_output(self,datatype, **kwargs):
        if datatype in list(self.target_data.keys()):
                
            f = plot_segmenimages(self.imgs_data[datatype],np.max(np.stack(
                    self.target_data[datatype]), axis = 0)*255,  
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
                    cocodataset = True, onlythese = None,**kwargs) -> None:
        
        from pycocotools.coco import COCO
        
        
        self.input_path = input_path
        
        self.cocodataset = cocodataset
        
        ## must be a coco file
        assert isinstance(annotation, COCO)
        
        self._annotation_cocodata = copy.deepcopy(annotation)
        img_data = self._annotation_cocodata.loadImgs(img_id)
        
        fn = img_data[0]['file_name']
        
        assert len(img_data) == 1
        
        self.annotation_ids = self._annotation_cocodata.getAnnIds(
                    imgIds=img_data[0]['id'], 
                    catIds=1, 
                    iscrowd=None
                )
        self.anns = self._annotation_cocodata.loadAnns(self.annotation_ids)
        self.numobjs = len(self.anns)
        self._onlythesetransforms = onlythese
        
        super(SegmentationImagesCoCo, self).__init__(
            path = os.path.join(self.input_path, 
                                                fn), 
                            img_id = fn, **kwargs)

    

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
            bbs = []
            todelete = []
            for i,mask in enumerate(self.target_data[imgtype]):
                if np.sum(mask)>0:
                    bbs.append(get_boundingboxfromseg(mask))
                else:
                  todelete.append(i)
                    
            boxes[imgtype] = np.array(bbs)
            
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
                paramslist = [partr for partr in list(params.keys()) if partr not in self._list_not_transform]
                filteredparams = {}
                for partr in paramslist:
                    filteredparams[partr] = params[partr]
                    
                imgtr = mask.copy()

                for partr in paramslist:
                    #print(partr)
                    imgtr = perform(self._run_default_transforms[partr],
                            imgtr,
                            params[partr], False)
                
                trmask[i] = imgtr

                #trmask[i] = self.multi_transform(img=mask.copy(),
                #                    chain_transform = paramslist,
                #                    params= filteredparams, update= False)

            else:
                trmask[i]  =  perform(self._run_default_transforms[tr_name],
                        mask.copy(),
                        self.tr_paramaters[tr_name], False)

        return trmask
    

    def transform_target_image(self):
        trids = list(self.images_names.keys())
        newimgs = {}
        if len(trids)> 1:
            for i in range(1,len(trids)):
                if trids[i] not in self._list_not_transform:
                    newtr = self.tranform_function(trids[i])
                    filtered = []
                    for indtr in newtr:
                        if np.sum(indtr)>0:
                            #print(np.sum(indtr>0)/(newtr.shape[1]*newtr.shape[2]))
                            if np.sum(indtr>0)/(newtr.shape[1]*newtr.shape[2])>0.0015:
                                filtered.append(indtr)
                        newimgs[trids[i]] = np.array(filtered)

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
        
        from pycocotools.coco import COCO
        
        
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
            

