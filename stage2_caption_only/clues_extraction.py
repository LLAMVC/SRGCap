import os
import nltk
import pickle
from typing import List
from nltk.stem import WordNetLemmatizer
from load_annotations import load_captions

def main(captions: List[str], path: str) -> None:
    # writing list file, i.e., [[[clue1, clue2,...], caption], ...] 

    clues_tags = set(['MD', 'UH', 'RB', 'RBR', 'RBS', 'JJ', 'JJR', 'JJS'])  # 形容词，副词作为 emotion clues
    lemmatizer = WordNetLemmatizer()
    new_captions = []
    for caption in captions:
        detected_clues = []
        pos_tags = nltk.pos_tag(nltk.word_tokenize(caption)) 
        for clues_with_pos in pos_tags:
            if clues_with_pos[1] in clues_tags:
                clue = lemmatizer.lemmatize(clues_with_pos[0].lower().strip())
                detected_clues.append(clue)
        detected_clues = list(set(detected_clues))
        new_captions.append([detected_clues, caption])
    
    with open(path, 'wb') as outfile:
        pickle.dump(new_captions, outfile)
    

if __name__ == '__main__':
    datasets = ['MER23_captions', 'EMOSEC_captions']
    captions_path = [
        './annotations/MER23_captions/train_captions.json',
        './annotations/EMOSEC_captions/train_captions.json'
    ]
    out_path = [
        './annotations/MER23_captions/MER23_with_clues.pickle',
        './annotations/EMOSEC_captions/EMOSEC_with_clues.pickle'
    ]
    
    idx = 0 # only need to change here! 0 -> MER23_captions training data, 1 -> EMOSEC_captions training data
    
    if os.path.exists(out_path[idx]):
        print('Read!')
        with open(out_path[idx], 'rb') as infile:
            captions_with_clues = pickle.load(infile)
        print(f'The length of datasets: {len(captions_with_clues)}')
        captions_with_clues = captions_with_clues[:20]
        for caption_with_clues in captions_with_clues:
            print(caption_with_clues)
        
    else:
        print('Writing... ...')
        captions = load_captions(datasets[idx], captions_path[idx])
        main(captions, out_path[idx])