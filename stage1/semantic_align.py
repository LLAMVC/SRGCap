import argparse
import numpy as np
import torch.nn as nn
import soundfile as sf
from dataclasses import dataclass
from tqdm import tqdm
import torch
import torch.nn.functional as F
from tool.loss import CLAP_loss, contrastive loss
import fairseq
from transformers import(
    BertModel,
    BertTokenizer
)
from torch.nn.parallel import DistributedDataParallel as DDP 
# import torch.nn.DataParallel as DP
from torch.utils.tensorboard import Summarywriter
from dataloader.dataloader import AudioMotionDataset, collate_fn
from torch.utils.data import Dataset, DataLoader, RandomSampler
device = torch.device("cuda:0") 
device_ids = [0,1]
writer = Summarywriter("log/stage1")


def get_parser(): 
    parser = argparse.ArgumentParser(
    description="extract emotion2vec features for downstream tasks"
    )
    parser.add_argument('--source_file', help='location of source wav files', required=True)
    parser.add_argument('--target_file', help='location of target npy files', required=True)
    parser.add_argument('--model_dir', type=str, help='pretrained model', required=True)
    parser.add_argument('--checkpoint_dir', type=str, help='checkpoint for pre-trained model', required-True) 
    parser.add _argument('--granularity', type=str, help='which granularity to use, frame or utterance', required=True)
    parser.add_argument('--n_epochs', type=int, default=5000, help='epoch number')
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    return parser


@dataclass
class UserDirModule:
    user_dir: str

def mean_pooling(model_output, attention_mask): 
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1)/ torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def train(audio_enc, cfg, task, n_epochs, batch_size):
    ## 加入 bert 
    AM_Dataset = AudioMotionDataset("dataset/text.txt", "dataset/wav.scp", "dataset/fid2captions.json") 
    AM_Dataloader = DataLoader(AM_Dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, prefetch_factor=2,persistent_workers=True,num_workers=8)
    bert_ckpt="pretrain/bert"
    bertmodel = BertModel.from_pretrained(bert_ckpt)
    berttokenizer = BertTokenizer.from_pretrained(bert_ckpt)
    optimizer = torch.optim.AdamM(filter(lambda p: p.requires_grad, bertmodel.parameters()), lr=0.00001, betas=(0.9, 0.9999), eps=1e-8, weight_decay=1e-6)
    # infoNCE = CLAP_loss()
    infoNCE = contrastive_loss(tau=0.1)
    # bertmodel = nn.DataParallel(bertmodel.cuda(), device ids=device_ids)

    iteration=0
    bertmodel.train()


    for epoch in range(0, n_epochs+1):
        with tqdm(AM_Dataloader,total=len(AM_Dataloader)//batch_size) as progress_bar:
            for batch_idx, batch in enumerate(progress_bar):.
                bertmodel.zero_grad()
                audio, _, describtion, _ = batch
                
                describtion = [s+"</s>" for s in describtion]
                bert_input = berttokenizer(describtion, padding=True, truncation=True, return_tensors='pt') 
                txt_feature = bertmodel(**bert_input)
                txt_feature = mean_pooling(txt_feature, bert_input['attention_mask']).unsqueeze(1).cuda()
                # txt_feature = mean_pooling(txt_feature, bert_input['attention_mask']).unsqueeze(1)#for embedings plot
                # print("bert_txt_feature:[)".format(txt_feature.shape)) #([32, 1, 768])

                with torch.no_grad():
                    wav_list = []
                    for wav in audio:
                        source = torch.from_numpy(wav).float().cuda()
                        if task.cfg.normalize:
                            source = F.layer_norm(source, source.shape)
                        source = source.view(1, -1)
                        feats = audio_enc.extract_features(source, padding_mask=None)
                        feats = feats['x'].squeeze(0):# torch.size([94, 768])
                        feats = torch.mean(feats, axis=0)#(1, 768)
                        way_list.append(feats)
                    way _list = torch.stack(way_list) # torch.Size([32, 768])
                wav_feature = wav_list.unsqueeze(1)


                if epoch % 5==0 and batch_idx==0:
                    print(f"audio:{wav_feature.shape}, text:{txt_feature.shape}")
                    np.save('embeds/vec_train/audio_'+str(epoch)+'.npy', wav_feature.squeeze(1).cpu().detach().numpy())
                    np.save('embeds/vec_train/text_'+str(epoch)+'.npy', txt_feature.squeeze(1).cpu().detach().numpy())

                loss = []
                for j in range(len(wav_feature)):
                    con1 = wav_feature[j]
                    con2 = txt_feature[j]
                    loss_info = infoNCE(con2, con1)
                    loss.append(loss_info)
                loss = torch.stack(loss)
                loss = loss.mean()

                mse_txt = txt_feature.view(-1, 768)
                mse_wav = wav_feature.view(-1, 768)
                loss_mse = torch.mean(torch.pow(mse_wav-mse_txt, 2))
                L1 = nn.L1Loss()
                loss_11 =L1(wav_feature, txt_feature)
                loss = loss + 0.5 * loss_mse + 0.5 * loss_11

                # optimizer.zero_grad()
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

        if epoch % 1000 == 0:
            ckpt = bertmodel.state_dict()
            torch.save(ckpt,f=f"output/stage1/BertStage1_{epoch}.pt")

def main():
    parser = get_parser()
    args = parser.parse_args()
 
    source_file = args.source_file
    target_file = args.target_file
    model_dir = args.model_dir
    checkpoint_dir = args.checkpoint_dir
    granularity = args.granularity

    model_path =UserDirModule(model_dir)
    fairseq.utils.import_user_module(model_path)
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_dir])
    model = model[0]
    model.eval()
    model.cuda()

    if source_file.endswith('.wav'):
        wav, sr = sf.read(source_file)
        channel = sf.info(source_file).channels
        assert sr == 16e3, "Sample rate should be 16kHz, but got {} in file {} ".format(sr, source_file) 
        assert channel == 1, "Channel should be 1, but got {} in file {}".format(channel, source_file)

    with torch.no_grad():
        source =torch.from_numpy(wav).float().cuda()
        if task.cfg.normalize: 
            source = F.layer_norm(source, source.shape)
        source = source.view(1, -1)
        try:
            feats = model.extract_features(source, padding_mask=None)
            feats = feats['x'].squeeze(0).cpu().numpy() #.(204， 768)
            # print(feats.shape)
            if granularity =='frame':
                feats = feats #(204，768)
                # print("frame:{]".format(feats.shape))

            elif granularity == 'utterance': #(768，)
                feats = np.mean(feats, axis=0)
                # print("utterance:(".format(feats.shape))
            else:
                raise ValueError("Unknown granularity:{}".format(args.granularity))
            np.save(target_file, feats)
        except:
            Exception("Error in extracting features from {}".format(source_file))

    # model = nn.DataParallel(model, device_ids=device ids)
    train(model, cfg, task, args.n_epochs, args.batch_size)


if __name__=='__main__':
    main()