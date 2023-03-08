
import requests
from urllib.parse import urlparse
import os
import zipfile
from io import BytesIO


def filter_files_usingsuffix(filesinside, path, suffix = 'pt'):
    """
    function to pull a zip file from internet
    Parameters:
    --------
    path: str
        path that contains the files
    
    suffix: str
        use a string to filter the files that are inside the extracted folder
    
    Returrn:
    --------
    path to the file
    """

    fileinfolder = [i for i in filesinside if i.endswith(suffix)]
    print(fileinfolder)
    if len(fileinfolder)==1:
        wp = fileinfolder[0]
        wp = os.path.join(path, wp)
    else:
        raise ValueError("there is no files with this extension {}".format(suffix))
       
    return wp


def downloadzip(urlpath, foldername = 'models')-> None: 
    """
    function to pull a zip file from internet
    Parameters:
    --------
    urlpath: str
        url link which contian the file
    
    foldername: str
        the folder name in which the extracted file will be located
    
    Returrn:
    --------
    None
    """
    if foldername is None:
        foldername = ""

    if urlpath.startswith('http'):
        a = urlparse(urlpath)
        
        if not os.path.exists(os.path.join(foldername,os.path.basename(a.path))):
            req = requests.get(urlpath)

            with zipfile.ZipFile(BytesIO(req.content)) as zipobject:
                zipobject.extractall(foldername)
        
        else:
            zipobject = zipfile.ZipFile(os.path.join(foldername,os.path.basename(a.path)))
            if not os.path.exists(os.path.join(foldername,
                zipobject.filelist[0].filename)):
                with zipfile.ZipFile(os.path.basename(a.path)) as zipobject:
                    zipobject.extractall(foldername)

        return zipobject.namelist()

def check_weigth_path(path, suffix = 'h5', weights_path = 'weights'):
    if not path.endswith(suffix):
        filesinside = downloadzip(path, foldername = weights_path)
        path = filter_files_usingsuffix(filesinside, weights_path, suffix=suffix)
    
    return path