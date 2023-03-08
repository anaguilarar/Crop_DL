import pycocotools.mask as mask_util
import numpy as np
import math


def euclidean_distance(p1,p2):
    return math.sqrt(
        math.pow(p1[0] - p2[0],2) + math.pow(p1[1] - p2[1],2))


def decode_masks(mask):
    
    if len(mask.shape) == 3:
        msk = mask[:,:,0]
    else:
        msk = mask
    
    binary_mask_fortran = np.asfortranarray(msk)
    rle = {'counts': [], 'size': list(binary_mask_fortran.shape)}
    msk = mask_util.encode(binary_mask_fortran)
    
    rle["counts"] = msk["counts"].decode()
    return rle


def getmidlewidthcoordinates(pinit,pfinal,alpha):

  xhalf=pfinal[0] - math.cos(alpha) * euclidean_distance(pinit,pfinal)/2
  yhalf=pinit[1] - math.sin(alpha) * euclidean_distance(pinit,pfinal)/2
  return int(xhalf),int(yhalf)

def getmidleheightcoordinates(pinit,pfinal,alpha):

  xhalf=math.sin(alpha) * euclidean_distance(pinit,pfinal)/2 + pinit[0]
  yhalf=math.cos(alpha) * euclidean_distance(pinit,pfinal)/2 + pinit[1]
  return int(xhalf),int(yhalf)


