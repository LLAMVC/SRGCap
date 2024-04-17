# Caption-only Training  


## 1. Environment setting

```bash
conda create -n llama2 python=3.9 -y
conda activate llama2
pip install -r requirements.txt 
python -m bitsandbytes
```


## 2. Download pretrain model

```bash
python model_download.py --repo_id daryl149/llama-2-7b-chat-hf
```


## 3. Merge tokenizer

```bash
## merge llama and chinese_sp.model
python merge_tokenizers.py \
  --llama_tokenizer_dir ./models/daryl149/llama-2-7b-chat-hf \
  --chinese_sp_model_file ./data/chinese_sp.model \
  --output_sp_dir merged_tokenizer_sp \
  --output_hf_dir merged_tokenizer_hf \


## merge zn_en_llama and bert.model
python merge_tokenizers.py \
  --llama_tokenizer_dir ./merged_tokenizer_hf \
  --chinese_sp_model_file ./data/bert.model \
  --output_sp_dir bert_tokenizer_sp \
  --output_hf_dir bert_tokenizer_hf \
```



## 4. Training

### deepspeed + lora + frozen_bert finetuning LLM-based (Llama-2-7b) Decoder

```bash
./stage2.sh
```

