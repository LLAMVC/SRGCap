import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import(
    BertModel,
    BertTokenizer
)
import fairseq
from torch.nn.parallel import DistributedDataParallel as DDP 
# import torch.nn.DataParallel as DP
from torch.utils.tensorboard import Summarywriter
from dataloader.dataloader import AudioMotionDataset, collate_fn
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.utils.tensorboard import Summarywriter
import warnings
warnings.filterwarnings("ignore")
writer = Summarywriter("log/SCCM_Pretrain")


class CaptionRetriever(nn.Module):
    def __init__(self, txt_path, top_k=5):
        super(CaptionRetriever, self).__init__()
        # 加载文本文件中的字幕嵌入
        self.captions = self.load_captions_from_txt(txt_path)
        self.top_k = top_k

    def load_captions_from_txt(txt_path):
        # 从文本文件中加载字幕嵌入
        captions = []
        with open(txt_path, 'r') as f:
            for line in f:
                # 假设每个嵌入向量由空格分隔的数值组成
                caption = np.array(line.strip().split(), dtype=np.float32)
                captions.append(caption)
        return torch.tensor(captions)

    def mean_pooling(model_output, attention_mask): 
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1)/ torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, speech_embedding, caption_embeddings=None):
        
        if not caption_embeddings:
            with torch.no_grad():
                bert_ckpt="./pretrain_bert"
                bertmodel = BertModel.from_pretrained(bert_ckpt)
                berttokenizer = BertTokenizer.from_pretrained(bert_ckpt)
                describtion = [s+"</s>" for s in self.captions]
                bert_input = berttokenizer(describtion, padding=True, truncation=True, return_tensors='pt') 
                txt_feature = bertmodel(**bert_input)
                caption_embeddings = mean_pooling(txt_feature, bert_input['attention_mask']).unsqueeze(1).cuda()
        
        # 计算语音嵌入与所有字幕嵌入之间的余弦相似度
        cosine_similarity = F.cosine_similarity(speech_embedding, caption_embeddings, dim=1)
        # 找到最相似的top-k个字幕嵌入的索引
        _, top_k_indices = torch.topk(cosine_similarity, k=self.top_k, dim=1, largest=True)
        # 从Memory中检索对应的字幕嵌入
        top_k_caption_embeddings = caption_embeddings[top_k_indices]
        return top_k_caption_embeddings



class SemanticCalibrationLearning(nn.Module):
    def __init__(self, caption_embedding_dim=768, center_dim=32, num_centers=5, caption_length=20, hidden_dim=512, temperature=1.0):
        super(SemanticCalibrationLearning, self).__init__()

        self.caption_length = caption_length
        self.hidden_dim = hidden_dim
        self.center_dim = center_dim
        self.num_centers = num_centers
        self.temperature = temperature
        self.center_matrix = nn.Parameter(torch.randn(center_dim, caption_embedding_dim))
        self.aggregation_weights = nn.Parameter(torch.randn(num_centers, center_dim))
        self.projection = nn.Linear(center_dim * num_centers, caption_embedding_dim)

    def forward(self, speech_emotion_embedding, caption_embeddings):

        expanded_embeddings = caption_embeddings.unsqueeze(1).repeat(1, self.caption_length, 1)
        mapped_embeddings = caption_embeddings.matmul(self.center_matrix.T)
        similarities = F.cosine_similarity(speech_emotion_embeddings.unsqueeze(1), mapped_embeddings, dim=2)
        normalized_similarities = torch.exp(similarities / self.temperature)
        
        # aggregate latent semantic center embeddings
        aggregated_embeddings = torch.sum(normalized_similarities * mapped_embeddings, dim=1)
        updated_speech_emotion_embeddings = self.projection(aggregated_embeddings)
        
        return updated_speech_emotion_embeddings




def loss_function(Psem, target_embedding):
    loss_mse = torch.mean((Psem - target_embedding) ** 2)
    L1 = nn.L1Loss()
    loss_11 =L1(Psem, target_embedding)
    return loss_mse + loss_11


def train_SCCM(model_dir, dataloader, model, optimizer, epochs, batch_size):

    model_path =UserDirModule(model_dir)
    fairseq.utils.import_user_module(model_path)
    emo2vec, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_dir])
    emo2vec = emo2vec[0]
    emo2vec.eval()
    emo2vec.cuda()

    iteration=0
    model.train()

    for epoch in range(0, epochs + 1):
        with tqdm(dataloader, total=len(dataloader)//batch_size) as progress_bar:
            for batch_idx, batch in enumerate(progress_bar):
                model.zero_grad()
                audio, _, captions, _ = batch

                # 将语音转换为情感嵌入
                with torch.no_grad():
                    wav_list = []
                    for wav in audio:
                        source = torch.from_numpy(wav).float().cuda()
                        if task.cfg.normalize:
                            source = F.layer_norm(source, source.shape)
                        source = source.view(1, -1)
                        feats = emo2vec.extract_features(source, padding_mask=None)
                        feats = feats['x'].squeeze(0):# torch.size([94, 768])
                        feats = torch.mean(feats, axis=0)#(1, 768)
                        wav_list.append(feats)
                    wav_list = torch.stack(wav_list) # torch.Size([32, 768])
                speech_emotion = wav_list.unsqueeze(1)

                top_k_caption_embeddings = model.CaptionRetriever(speech_emotion)
            
                # 语义校准学习
                Psem = model.SemanticCalibrationLearning(speech_emotion, top_k_caption_embeddings)

                loss = loss_function(Psem, target_embedding)
            
                loss.backward() 
                optimizer.step()
        
                losses = {}
                losses['training_loss']= loss.item()
                for tag, value in losses.items():
                    writer.add_scalar(tag, value, iteration + 1)

                # if batch idx % 2 ==0:
                #   msg =f' Epoch : fepoch}, iteration: {iteration | loss: {loss.item()
                #   progress_bar.set_description(msg)
                #   print(msg)

                print(f'Epoch : {epoch}, iteration : {iteration} | loss: {loss.item()}') 
                iteration += 1

        if epoch % 500 == 0:
            ckpt = model.state_dict()
            torch.save(ckpt, f=f"output/SCCM_Pretrain/SCCM_{epoch}.pt")




def main():

    model_dir = 'emotion2vec-main/upstream' 
    AM_Dataset = AudioMotionDataset("dataset/text.txt", "dataset/wav.scp", "dataset/fid2captions.json") 
    AM_Dataloader = DataLoader(AM_Dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, prefetch_factor=2,persistent_workers=True,num_workers=8)

    # speech_embedding，txt_path
    txt_path = 'path_to_OOD_captions.txt'
    top_k = 5                                # 检索器检索前top-k个captions
    epoch = 5000
    batch_size = 64
    caption_embedding_dim = 768
    num_centers = 5
    caption_length = 20

    model = nn.ModuleList([
        CaptionRetriever(txt_path, top_k=top_k),
        SemanticCalibrationLearning(caption_embedding_dim=caption_embedding_dim, num_centers=num_centers, caption_length=caption_length)
    ])
  
    optimizer = torch.optim.AdamM(filter(lambda p: p.requires_grad, bertmodel.parameters()), lr=0.00001, betas=(0.9, 0.9999), eps=1e-8, weight_decay=1e-6)

    # model = nn.DataParallel(model, device_ids=device ids)
    train_SCCM(model_dir, AM_Dataloader, model, optimizer, epochs=epochs, batch_size=batch_size)


if __name__=='__main__':
    main()