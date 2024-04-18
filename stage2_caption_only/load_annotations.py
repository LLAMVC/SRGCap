import json
import pickle
import pandas as pd
from typing import List

def load_MER23_captions(path: str) -> List[str]:

    with open(path, 'r') as infile:
        annotations = json.load(infile)              
    punctuations = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', ' ', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']

    captions = []
    for image_path in annotations:                  
        temp_captions = annotations[image_path]        
        for caption in temp_captions:                 
            caption = caption.strip()                  
            if caption.isupper():                      
                caption = caption.lower()
            caption = caption[0].upper() + caption[1:] 
            if caption[-1] not in punctuations:        
                caption += '.'
            captions.append(caption)                   

    return captions

def load_EMOSEC_captions(path: str) -> List[str]:
    
    with open(path, 'r') as infile:
        annotations = json.load(infile) 
    punctuations = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', ' ', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']

    captions = []
    for image_path in annotations:                  
        temp_captions = annotations[image_path]
        for caption in temp_captions:
            caption = caption.strip()
            if caption.isupper():
                caption = caption.lower()
            caption = caption[0].upper() + caption[1:]
            if caption[-1] not in punctuations:
                caption += '.'
            captions.append(caption)

    return captions

def load_captions(name_of_datasets: str, path_of_datasets: str) -> List[str]:
    """
    Args:
        name_of_datasets: specifying the name of datasets
        path_of_datasets: specifying the path of datasets
    Return:
        [caption1, caption2, ...]
    """
    if name_of_datasets == 'MER23_captions':
        return load_MER23_captions(path_of_datasets)

    if name_of_datasets == 'EMOSEC_captions':
        return load_EMOSEC_captions(path_of_datasets)

    print('The datasets for training fail to load!')


