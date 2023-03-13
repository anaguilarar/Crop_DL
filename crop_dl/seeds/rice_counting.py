
import os
import torch
import numpy as np
import pycocotools.mask as mask_util
import cv2
import datetime
import math
import tqdm

from .configuration import RiceDetectionConfig
from .utils import decode_masks, euclidean_distance, getmidleheightcoordinates, getmidlewidthcoordinates

from ..plt_utils import plot_segmenimages, random_colors, add_frame_label
from ..dataset import cocodataset_dict_style
from ..image_functions import contours_from_image, pad_images
from ..dataset_utils import get_boundingboxfromseg
import zipfile
import glob


def image_to_tensor(img, size):
    
    if img.shape[0] != size[0] or img.shape[1] != size[1]:
        resimg = cv2.resize(img.copy(), size)
    else:
        resimg = img.copy()
    
    #resimg
    if resimg.shape[0] != 3:
        imgtensor = torch.from_numpy(resimg.swapaxes(2,1).swapaxes(0,1)/255).float()
    elif resimg.shape[0] == 3:
        imgtensor = torch.from_numpy(resimg/255).float()
        
    return imgtensor    

def _apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


class RiceSeedsCounting(object):
    
    def __init__(self, imagepath, model, 
                 inputsize = (512,512), 
                 imagessuffix = ".jpg",
                 device = None) -> None:
        
        
        self.predictions  = None
        self._frames_colors = None
    
        assert os.path.exists(imagepath)
        self.path = imagepath
        self.inputsize = inputsize
        
        self.listfiles = [i for i in os.listdir(imagepath) 
                          if i.endswith(imagessuffix)]
        
        self.model = model
        
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
    
    def evaluate_predictions_by_users(self):
        from matplotlib.pyplot import plot, ion, show, close
        
        print("""
              For the {} image {} bounding boxes were detected
              which method will you prefer to select the correct predictions\n
              1. by threshhold scores
              2. one by one
              
              """.format(self.listfiles[self.idimg], 
                         len(self.predictions[0]['scores'])))
        
        evaluation_method = input()       
        imgtoplot = self._imgastensor.mul(255).permute(1, 2, 0).byte().numpy()

        if evaluation_method == "1":
            minthreshold = input("the mininmum score (0-1): ")
            maxthreshold = input("the maximum score (0-1): ")
            input_1 = float(minthreshold)
            input_2 = float(maxthreshold)
            
            onlythesepos = np.where(np.logical_and(
                self.predictions[0]['scores'].to('cpu').detach().numpy()>input_1,
                self.predictions[0]['scores'].to('cpu').detach().numpy()<input_2))

        if evaluation_method == "2":
            onlythesepos = []
            
            for i in range(len(self.predictions[0]['scores'])):
                mks = self.predictions[0]['masks'].mul(255).byte().cpu().numpy()[i, 0].squeeze()
                bb = self.predictions[0]['boxes'].cpu().detach().numpy()[i]
                
                
                ion()
                f = plot_segmenimages((imgtoplot).astype(np.uint8)[:,:,[2,1,0]],
                                      mks, 
                        boxes=[bb.astype(np.uint16)], 
                        bbtype = 'xminyminxmaxymax')
                response = input("is this prediction [{} of {}] correct (1-yes ; 2-no; 3-exit): ".format(
                    i+1,len(self.predictions[0]['scores'])))
                f.show()
                close()
                if response == "1":
                    onlythesepos.append(i)
                if response == "3":
                    break
            
        print(f"in total {len(onlythesepos)} segmentations are correct")
        msks_preds = self.predictions[0]['masks'].mul(255).byte().cpu().numpy()[onlythesepos, 0].squeeze()
        bbs_preds = self.predictions[0]['boxes'].cpu().detach().numpy()[onlythesepos]
        
        if len(msks_preds.shape)>2:
            if msks_preds.shape[0] == 0:
                msks = np.zeros(self.inputsize)
            else:
                msks = np.max(msks_preds, axis = 0)
        
        ion()
        f = plot_segmenimages((imgtoplot).astype(np.uint8)[:,:,[2,1,0]],msks, 
                        boxes=bbs_preds.astype(np.uint16), 
                        bbtype = 'xminyminxmaxymax')
        response = input("is this prediction correct (1-yes ; 2-no): ")
        f.show()
        close()
        if response == "2":
            response = input("do you want to repeat (1-yes ; 2-no): ")
            if response == "1":
                self.evaluate_predictions_by_users()
        else:
            self.msks_preds = msks_preds
            self.bbs_preds = bbs_preds
        
        return msks_preds, bbs_preds, imgtoplot
    
    def _original_size(self):
        msksc = [0]* len(list(self.msks))
        bbsc = [0]* len(list(self.bbs))
        for i in range(len(self.msks)):
            msksc[i] = cv2.resize(self.msks[i], 
                                  [self.img.shape[1],self.img.shape[0]], 
                                  interpolation = cv2.INTER_AREA)        
            bbsc[i] = get_boundingboxfromseg(msksc[i])
        
        self.msks = np.array(msksc)
        self.bbs = np.array(bbsc)
        
    
    def detect_rice(self, id, threshold = 0.75, segment_threshold = 180, keepsize = False):
        
        imgpath = os.path.join(self.path, self.listfiles[id])
        self.img = cv2.imread(imgpath)
        imgtensor = image_to_tensor(self.img.copy(), self.inputsize)
                
        self.model.eval()
        with torch.no_grad():
            prediction = self.model([imgtensor.to(self.device)])
        
                   
        self.predictions = prediction
        self.idimg = id
        self._imgastensor = imgtensor
        
        pred = self._filter_byscore(threshold)
        
        self.msks = pred[0]
        
        for i in range(len(self.msks)):
            self.msks[i][self.msks[i]<segment_threshold] = 0
            
        self.bbs = pred[1]
        self._frames_colors = random_colors(len(self.bbs))
        
        if keepsize:
            self._original_size()
            self._img = self.img
        else:
            self._img = self._imgastensor.mul(255).permute(1, 2, 0).byte().numpy()
        
    def _filter_byscore(self, threshold):
        
        pred = self.predictions[0] 
        onlythesepos = np.where(
            pred['scores'].to('cpu').detach().numpy()>threshold)
        
        msks = pred['masks'].mul(255).byte().cpu().numpy()[onlythesepos, 0].squeeze()
        bbs = pred['boxes'].cpu().detach().numpy()[onlythesepos]
        
        if msks.shape[0] == 0:
            msks = np.zeros(self.inputsize)
            
        if len(msks.shape)==2:
            msks = np.expand_dims(msks,0)
                
        return msks, bbs
        
    def plot_prediction(self,only_image = True):
        
        plotimage = None
        if self.predictions is not None:
            
            if self.bbs is not None:
                bbs= self.bbs.astype(np.uint16)
                if self._frames_colors is None:
                    self._frames_colors = random_colors(len(self.bbs))
        
            if self.msks.shape[0] == 0:
                msksone = np.zeros(self._img.shape[:2])
            else:
                msksone = np.max(self.msks, axis = 0)
                
            plotimage =  plot_segmenimages((self._img).astype(np.uint8),
                              msksone, 
                        boxes=bbs, 
                        bbtype = 'xminyminxmaxymax',
                        only_image = only_image,
                        default_color = self._frames_colors,
                        sizefactorred = 150)
        
        return plotimage
    

    def predictions_to_cocodataset(self, maskthres = 180):
        
        dataanns = []
        for k in range(len(self.msks_preds)):
            bb = [int(j) for j in self.bbs_preds[k]]
            binarymask = self.msks_preds[k].copy()
            binarymask[binarymask<maskthres] =0
            binarymask[binarymask>=maskthres] = 1
                    #bb = cv2.boundingRect(countours[k])
            dataanns.append(
            {"id":k,
            "image_id":self.idimg,
            "category_id":1,
            "bbox":bb, 
            "area":bb[2]*bb[3],
            "segmentation":decode_masks(binarymask),
            "iscrowd":0
            })

        imgdata = {"id":self.idimg,
                   "license":1,
                   "file_name":self.listfiles[self.idimg],
                   "height":self.msks_preds[0].shape[1],
                   "width":self.msks_preds[0].shape[0],
                   "date_captured":datetime.datetime.now().strftime("%Y-%m-%d")
                   }

        return dataanns, imgdata
    
    ## metrics from seeds
    def plot_individual_seed(self, seed_id,**kwargs):
        
        return self._add_metriclines_to_single_detection(seed_id, **kwargs)

    def calculate_oneseed_metrics(self, seed_id, padding = 20):

        #imageres = self._imgastensor.mul(255).permute(1, 2, 0).byte().numpy()

        maskimage = self._clip_image(self.msks[seed_id], self.bbs[seed_id], padding = padding)
        wrapped_box = self._find_contours(maskimage)
        pheightu, pheigthb, pwidthu, pwidthb = self._get_heights_and_widths(wrapped_box)
        d1 = euclidean_distance(pheightu, pheigthb)
        d2 = euclidean_distance(pwidthu, pwidthb)
        #distper = np.unique([euclidean_distance(wrapped_box[i],wrapped_box[i+1]) for i in range(len(wrapped_box)-1) ])
        ## with this statement there is an assumption that the rice width is always lower than height
        larger = d1 if d1>d2 else d2
        shorter = d1 if d1<d2 else d2
        msksones = maskimage.copy()
        msksones[msksones>0] = 1
        
        area = np.sum(msksones*1.)

        return {
            'fn':[self.listfiles[self.idimg]],
            'seed_id':[seed_id],'height': [larger], 
                'width': [shorter], 'area': [area]}
        
    def one_image_seeds_summary(self):
        import pandas as pd
        summarylist = []
        for i in range(len(self.bbs)):
            summarylist.append(
                pd.DataFrame(self.calculate_oneseed_metrics(i)))

        return pd.concat(summarylist)
    
    def export_all_imagery_predictions(self, outputpath = None, saveaszip = False, **kwargs):
        
        if not os.path.exists(outputpath):
            os.mkdir(outputpath)
        
        images = self.all_image_predictions(**kwargs)
        
        print(f"Saving in: {outputpath}")
        
        for i, img in enumerate(images):
            cv2.imwrite(os.path.join(outputpath,"pred_{}".format(
                self.listfiles[i])), img[:,:,[2,1,0]])
        
        if saveaszip:
            with zipfile.ZipFile(outputpath + '.zip', 'w') as f:
                for file in glob.glob(outputpath + '/*'):
                    f.write(file)
        
        
    
    def all_image_seeds_summary(self, keepsize = True, segment_threshold = 170, threshold = 0.65):
        import pandas as pd
        alldata = []
        for i in tqdm.tqdm(range(len(self.listfiles))):
            self.detect_rice(i, keepsize=keepsize, 
                             segment_threshold = segment_threshold,
                             threshold=threshold)
            
            alldata.append(self.one_image_seeds_summary())

        return pd.concat(alldata)
    
    def all_image_predictions(self, keepsize = True, segment_threshold = 170, threshold = 0.65):
        
        alldata = []
        for i in tqdm.tqdm(range(len(self.listfiles))):
            self.detect_rice(i, keepsize=keepsize, 
                             segment_threshold = segment_threshold, 
                             threshold = threshold)
            
            m = self.plot_prediction(only_image=True)
            alldata.append(m[:,:,[2,1,0]])

        return alldata
        
    def _add_metriclines_to_single_detection(self, 
                                             seed_id, 
                    addlines = True, addlabel = True,
                    padding = 30,
                    mask_image = False,
                    sizefactorred = 250,
                    heightframefactor = .15,
                    widthframefactor = .3,
                    textthickness = 1):
        
        import copy
        if self._frames_colors is None:
            self._frames_colors = random_colors(len(self.bbs))
            
        col = self._frames_colors[seed_id]
        imageres = self._img
        imgclipped = copy.deepcopy(self._clip_image(imageres, self.bbs[seed_id], padding = padding,paddingwithzeros = False))
        maskimage = copy.deepcopy(self._clip_image(self.msks[seed_id], 
                                                               self.bbs[seed_id], padding = padding,paddingwithzeros = False))

        msksones = maskimage.copy()
        msksones[msksones>0] = 1
        
        if mask_image:

            newimg = cv2.bitwise_and(imgclipped[:,:,[2,1,0]],img,mask = msksones)
        else:
            newimg = np.array(imgclipped)
            
        img = _apply_mask(newimg, (msksones).astype(np.uint8), col, alpha=0.2)
        
        linecolor = list((np.array(col)*255).astype(np.uint8))
        m = cv2.drawContours(img,[self._find_contours(maskimage)],0,[int(i) for i in linecolor],1)
        if addlines:
            pheightu, pheigthb, pwidthu, pwidthb = self._get_heights_and_widths(self._find_contours(maskimage))
            m = cv2.line(m, pheightu, pheigthb, (0,0,0), 1)
            m = cv2.line(m, pwidthu, pwidthb, (0,0,0), 1)
        
        if addlabel:
            
            x1,y1,x2,y2 = get_boundingboxfromseg(maskimage)

            m = add_frame_label(m,
                    str(seed_id),
                    [int(x1),int(y1),int(x2),int(y2)],[
                int(i*255) for i in col],
                    sizefactorred = sizefactorred,
                    heightframefactor = heightframefactor,
                    widthframefactor = widthframefactor,
                    textthickness = textthickness)
            
        return m
    
    @staticmethod
    def _get_heights_and_widths(maskcontours):

        p1,p2,p3,p4=maskcontours
        alpharad=math.acos((p2[0] - p1[0])/euclidean_distance(p1,p2))

        pheightu=getmidleheightcoordinates(p2,p3,alpharad)
        pheigthb=getmidleheightcoordinates(p1,p4,alpharad)
        pwidthu=getmidlewidthcoordinates(p4,p3,alpharad)
        pwidthb=getmidlewidthcoordinates(p1,p2,alpharad)

        return pheightu, pheigthb, pwidthu, pwidthb
    
    @staticmethod 
    def _clip_image(image, bounding_box, bbtype = 'xminyminxmaxymax', padding = None, 
                    paddingwithzeros =True):
        
        if bbtype == 'xminyminxmaxymax':
            x1,y1,x2,y2 = bounding_box
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            
        if padding:
            if paddingwithzeros:
                imgclipped = image[
                y1:y2,x1:x2] 
                
                imgclipped = pad_images(imgclipped, padding_factor = padding)
            else:
                height = abs(y1-y2)
                width = abs(x1-x2)
                zoom_factor = padding / 100 if padding > 1 else padding
                new_height, new_width = height + int(height * zoom_factor), width + int(width * zoom_factor)  
                pad_height1, pad_width1 = abs(new_height - height) // 2, abs(new_width - width) //2
                newy1 = 0 if (y1 - pad_height1)<0 else (y1 - pad_height1)
                newx1 = 0 if (x1 - pad_width1)<0 else (x1 - pad_width1)
                imgclipped = image[newy1:newy1+(height+pad_height1*2), 
                                   newx1:newx1+(width+pad_width1*2)] 
        
        
        return imgclipped

    @staticmethod 
    def _find_contours(image):
        maskimage = image.copy()
        #imgmas = (maskimage*255).astype(np.uint8)
        contours = contours_from_image(maskimage)
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        return box

