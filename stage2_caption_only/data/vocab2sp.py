import json
from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
import collections
import unicodedata

with open('EMOSEC.json', 'r') as file:
    data = json.load(file)

with open('EMOSEC.txt', 'w') as f:
    for i in range(len(data)):
        for key, value in data[i].items():
            if key=='caption_en':
                print(key+","+values+"\r")
                f.write(values+"\n")

spm.SentencePieceTrainer.Train(input='../data/EMOSEC.txt', model_prefix='./bert', vocab_size=30522, character_coverage=1, model_type='bpe')

