import os
import numpy as np

import random
import pandas as pd
from sklearn.model_selection import KFold
import json
import copy

def split_dataintotwo(data, idsfirst, idssecond):

    subset1 = data.iloc[idsfirst]
    subset2 = data.iloc[idssecond]

    return subset1, subset2


def split_idsintwo(ndata, ids = None, percentage = None, fixedids = None, seed = 123):

    if ids is None:
        ids = list(range(len(ndata)))

    if percentage is not None:
        if fixedids is None:
            idsremaining = pd.Series(ids).sample(int(ndata*percentage), random_state= seed).tolist()
        else:
            idsremaining = fixedids
        
        main_ids = [i for i in ids if i not in idsremaining]
    
    else:
        idsremaining = None
        main_ids = ids

    return main_ids, idsremaining


class SplitIds(object):

    
    def _ids(self):
        ids = list(range(self.ids_length))
        if self.shuffle:
            ids = pd.Series(ids).sample(n = self.ids_length, random_state= self.seed).tolist()

        return ids


    def _split_test_ids(self, test_perc):
        self.training_ids, self.test_ids = split_idsintwo(self.ids_length, self.ids, test_perc,self.test_ids, self.seed)


    def kfolds(self, kfolds, shuffle = True):
        kf = KFold(n_splits=kfolds, shuffle = shuffle, random_state = self.seed)

        idsperfold = []
        for train, test in kf.split(self.training_ids):
            idsperfold.append([list(np.array(self.training_ids)[train]),
                               list(np.array(self.training_ids)[test])])

        return idsperfold
    
    def __init__(self, ids_length = None, ids = None,val_perc =None, test_perc = None,seed = 123, shuffle = True, testids_fixed = None) -> None:
        
        
        self.shuffle = shuffle
        self.seed = seed
        print(ids_length)
        if ids is None and ids_length is not None:
            self.ids_length = ids_length
            self.ids = self._ids()
        elif ids_length is None and ids is not None:
            self.ids_length = len(ids)
            self.ids = ids
        else:
            raise ValueError ("provide an index list or a data length value")
        
        self.val_perc = val_perc

        if testids_fixed is not None:
            self.test_ids = [i for i in testids_fixed if i in self.ids]
        else:
            self.test_ids = None

        self.training_ids, self.test_ids = split_idsintwo(self.ids_length, self.ids, test_perc,self.test_ids, self.seed)
        if val_perc is not None:
            self.training_ids, self.val_ids = split_idsintwo(len(self.training_ids), self.training_ids, val_perc, seed = self.seed)
        else:
            self.val_ids = None



class FolderWithImages(object):

    @property
    def length(self):
        return len(self._look_for_images())

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
    

class FilesSplit(FolderWithImages, SplitIds):

    def __init__(self, path, suffix='.jpg', shuffle=False, seed=None) -> None:
        FolderWithImages(FilesSplit).__init__(path, suffix, shuffle, seed)
        
        
        

def cocodataset_dict_style(imagesdict, 
                           annotationsdict,
                           exportpath = None, 
                           year = "2023",
                           categoryname = "rice-seed"):
    
    cocodatasetstyle = {"info":{"year":year},
                    "licenses":[
                        {"id":1,"url":"https://creativecommons.org/licenses/by/4.0/",
                         "name":"CC BY 4.0"}]}
    
    categories = [{"id":0,
               "name":categoryname,
               "supercategory":"none"},
              {"id":1,"name":categoryname,
               "supercategory":categoryname}]
    
    
    jsondataset = {"info":cocodatasetstyle,
        "categories":categories,
        "images":imagesdict,
        "annotations":annotationsdict}
    
    if exportpath:
        with open(exportpath, 'w', encoding='utf-8') as f:
            json.dump(jsondataset, f, indent=4)
        
    return jsondataset


### coco update


def update_cocotrainingdataset(cocodatasetpath, newdata_images, newdata_anns):
    
    if os.path.exists(cocodatasetpath):
        with open(cocodatasetpath, 'r') as fn:
            previousdata = json.load(fn)
    else:
        previousdata = None
    
    if previousdata is not None:
        #newdatac = copy.deepcopy(newdata)

        #newdatac = copy.deepcopy(newdata)
        previousdatac = copy.deepcopy(previousdata)

        if isinstance(previousdatac["images"], dict):
            previousdatac["images"] = [previousdatac["images"]]
        
        if isinstance(previousdatac["annotations"], dict):
            previousdatac["annotations"] = [previousdatac["annotations"]]
            
        oldimageslist = copy.deepcopy(previousdatac["images"])
        oldannslist = copy.deepcopy(previousdatac["annotations"])
        
        if isinstance(newdata_images, dict):
            newimageslist = [copy.deepcopy(newdata_images)]
        else:
            newimageslist = copy.deepcopy(newdata_images)

        if isinstance(newdata_anns, dict):
            newannslist = [copy.deepcopy(newdata_anns)]
        else:
            newannslist = copy.deepcopy(newdata_anns)

        
        lastid = oldimageslist[len(oldimageslist)-1]['id']+1
        lastidanns = oldannslist[len(oldannslist)-1]['id']+1
        #print(oldannslist[0]['image_id'])
        for i, newimage in enumerate(newimageslist):
            previousid = newimage['id']
            newimage['id']  = lastid

            for j, newann in enumerate(newannslist):
                if isinstance(newann, list):
                    for k in range(len(newann)):
                        if newann[k]['image_id'] == previousid:
                            newann[k]['image_id'] = lastid
                            newann[k]['id'] = lastidanns
                            lastidanns+=1
                            
                else:

                    if newann['image_id'] == previousid:
                        newann['image_id'] = lastid
                        newann['id'] = lastidanns
                        lastidanns+=1
            lastid+=1
        
        #if len(newimageslist) == 1:
        #    newannslist = [newannslist]
        for newimage in newimageslist:
            previousdatac["images"].append(newimage)
        
        if isinstance(newannslist, list):
            for newann in newannslist:
                previousdatac["annotations"].append(newann)
        else:
            previousdatac["annotations"].append(newannslist)
                
    else:
        
        previousdatac = cocodataset_dict_style(newdata_images, newdata_anns)
        
        #previousdatac = copy.deepcopy(newdata)
        
    return previousdatac

def merge_trainingdataset(prev_cocodatasetpath, new_cocodatasetpath):
    
    if os.path.exists(new_cocodatasetpath):
    
        with open(new_cocodatasetpath, 'r') as fn:
            newdata = json.load(fn)
        
        newdatac = copy.deepcopy(newdata)
        
        newannslist = newdatac["annotations"]
        newimageslist = newdatac["images"]
        
        previousdatac = update_cocotrainingdataset(prev_cocodatasetpath,
                                                   newimageslist, newannslist)
        
        print(len(newimageslist)," new images were added")
        print(len(newannslist)," new annotations were added")
        
    return previousdatac