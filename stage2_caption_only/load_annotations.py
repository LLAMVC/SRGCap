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




def load_MER23_clues(path: str, all_clues: bool = True) -> List[str]:

    with open(path, 'rb') as infile:
        all_objects_attributes_relationships = pickle.load(infile) # dictionary {'relationships': dict, 'attributes': dict, 'objects': dict}
    clues = all_objects_attributes_relationships['objects']     # dictionary {'gqa': set, 'vg': set, 'joint': set}, joint = gqa + vg
    clues = clues['joint']                                   # set
    
    if all_clues:
        clues = [clue.lower().strip() for clue in clues]
    else:
        clues = [clue.lower().strip() for clue in clues if len(clue.split()) == 1]
    clues.sort()  # sort

    return clues

def load_EMOSEC_clues(path: str, all_clues: bool = True) -> List[str]:

    with open(path, 'r') as infile:
        clues = json.load(infile)       # List [category1, category2, ...]
    
    if all_clues:
        clues = [clue.lower().strip() for clue in clues]
    else:
        clues = [clue.lower().strip() for clue in clues if len(clue.split()) == 1]
    clues.sort()  # sort

    return clues



def load_clues_text(name_of_clues: str, path_of_clues: str, all_clues: bool = True) -> List[str]:
    """
    Args:
        name_of_clues: specifying the name of clues text
        path_of_clues: specifying the path of clues text
        all_clues: whether to apply all clues text. True denotes using clues including len(clues.split()) > 1
    Return:
        [clue1, clue2, ...]
    """
    if name_of_clues == 'MER23_clues':
        return load_MER23_clues(path_of_clues, all_clues)
    
    if name_of_clues == 'EMOSEC_clues':
        return load_EMOSEC_clues(path_of_clues, all_clues)

    
    print('The clues text fails to load!')