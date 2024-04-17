import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import json
import random
import soundfile as sf
import os

class SpeechCaptionDataset(Dataset):
    def __init__(self, input_file, wav_scp_file, transforms=None):
        
        self.transcriptions = {}
        self.transforms = transforms
        with open(input_file, 'r') as f:
            self.data = json.load(f)
    
        self.data = self.transforms(self.data)

        self.wav_paths = []
        path=os.path.dirname(os.path.abspath(__file__))
        path=os.path.dirname(path)
        with open(wav_scp_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                utt_id = parts[0]
                wav_path = os.path.join(path,parts[1])
                print(utt_id, self.data.keys())
                if utt_id+'.wav' in self.data.keys():
                    self.wav_paths.append((utt_id, wav_path))

    def __getitem__(self, index):
        # print(self.wav_paths)
        utt_id, wav_path = self.wav_paths[index]
        cap_zn_list, cap_en_list = self.data[utt_id+'.wav']

        return wav_path, cap_zn_list, cap_en_list

    def __len__(self):
        return len(self.data)


def transform_data(original_data):
    # 创建一个新的字典来保存转换后的数据
    transformed_data = {}
    cap_zn = {}
    cap_en = {}
    # 遍历原始数据集中的每个样本
    for sample in original_data:
        # 假设每个样本都有一个唯一的标识符，这里我们使用 'wav' 作为键
        key = sample['wav']
        for i in range(1,6):
            chinese_caption = sample.get('caption_ch'+ str(i), '')  # 获取中文描述，如果没有则为空字符串
            english_caption = sample.get('caption_en'+ str(i), '')  # 获取英文描述，如果没有则为空字符串
            # 将描述添加到新字典中，如果键已存在，则更新其值
            if key in transformed_data:
                cap_zn['caption_ch'].append(chinese_caption)
                cap_en['caption_en'].append(english_caption)
                transformed_data[key] = cap_zn['caption_ch'], cap_en['caption_en']
            else:
                transformed_data[key] = {}
                cap_zn['caption_ch'] = [chinese_caption]
                cap_en['caption_en'] = [english_caption]
    return transformed_data


def collate_fn(batch):
    wav_path, cap_zn_list, cap_en_list = zip(*batch)
    waveforms = []
    cap_zns=[]
    cap_ens=[]
    paths=[]
    for wav, cap_zn, cap_en in zip(wav_path, cap_zn_list, cap_en_list):
        path=wav.split('/')[-1]
        paths.append(path)
        waveform, sample_rate = sf.read(wav)
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(sample_rate, 16000)(torch.tensor(waveform).unsqueeze(0).to(torch.float32)).squeeze(0).numpy()        #print(sample_rate)
        waveforms.append(waveform)
        cap_zns.append(cap_zn)
        cap_ens.append(cap_en)    

    return waveforms, cap_zns, cap_ens, paths


import time
if __name__ == '__main__':
    batch_size = 32
    time1=time.time()
    SC_Dataset = SpeechCaptionDataset("/root/autodl-tmp/stage2_caption_only/data/data.json", "../data/wav_test.scp", transforms=transform_data)
    SC_Dataloader = DataLoader(SC_Dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    data=[]
    wavforms=[]
    trans=[]
    describs=[]
    for batch_idx, (waveforms, cap_zns, cap_ens, _) in enumerate(SC_Dataloader):
        print(batch_idx, waveforms, cap_zns, cap_ens)
    print(time.time()-time1)
        
    
    
