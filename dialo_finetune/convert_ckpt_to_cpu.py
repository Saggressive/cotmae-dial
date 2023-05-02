import torch
from transformers import AutoModel
# bert=AutoModel.from_pretrained("bert-base-uncased")
model = torch.load("ckpt/ecommerce_ckpt/best_bert-base-chinese.pt")
for key in  model.keys():
    model[key]=model[key].to("cpu")
f="ckpt/ecommerce_ckpt/best_bert-base-chinese_cpu.pt"
torch.save(model,f)
# bert_state_dict = bert.state_dict()
# for key in bert_state_dict.keys():
#     model_key_prefix = "model.bert."
#     model_key = model_key_prefix + key
#     bert_state_dict[key]= model[model_key]
# print("a")