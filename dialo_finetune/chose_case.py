import argparse
from datasets import load_dataset
from data import dialCollator
from transformers import BertModel,BertTokenizer
from torch.utils.data import DataLoader,Dataset
from modeling import dialForPretraining
import sys
from utils import create_optimizer,calculate_candidates_ranking,logits_recall_at_k,logits_mrr
from transformers.optimization import get_scheduler
from transformers.trainer_utils import SchedulerType
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import save_checkpoint
import utils
import math
from tqdm.auto import tqdm
import torch
from torch import nn
import numpy as np
import logging
import os
import time
from transformers import set_seed
from torch.utils.tensorboard import SummaryWriter
import json
# from main import eval_data
sys.path.append("./")

@torch.no_grad()
def choose_case(model,eval_loader,args,mode="test"):
    all_neg = []
    model.eval()
    cos = nn.CosineSimilarity(dim=-1)
    all_scores,all_labels = [],[]
    for batch in tqdm(eval_loader):
        context_cls_hiddens,response_cls_hiddens,labels = model(batch,sent_emb=True,eval_mode=mode)
        scores = torch.mul(context_cls_hiddens,response_cls_hiddens)
        scores = scores.sum(dim=-1)
        # scores = cos(context_cls_hiddens,response_cls_hiddens)
        all_scores.append(scores)
        all_labels.append(labels)
    
    all_scores,all_labels=torch.cat(all_scores,dim=0),torch.cat(all_labels,dim=0)
    total_mrr,total_correct=0,0
    total_examples = 0
    total_prec_at_one,total_map=0,0
    for index in range(all_scores.shape[0]):
        scores,label = all_scores[index],all_labels[index]
        if args.dataset == "ubuntu" or mode=="test":
            rank_by_pred, pos_index, stack_scores = \
                calculate_candidates_ranking(
                    np.array(scores.cpu().tolist()), 
                    np.array(label.cpu().tolist()),
                    10)
        else:
            rank_by_pred, pos_index, stack_scores = \
                calculate_candidates_ranking(
                    np.array(scores.cpu().tolist()), 
                    np.array(label.cpu().tolist()),
                    2)
        # some douban data have not true labels
        if sum(rank_by_pred[0])==0 and args.dataset == "douban":
            continue
        p = pos_index[0][0]
        all_neg.append(str(p)+"\n")
    with open("./choose_label_bert+.txt","w") as f:
        f.writelines(all_neg)
            

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--test_path', default='data/ubuntu/val/data.json', type=str)
    parser.add_argument('--dataset', default='ubuntu', type=str)
    parser.add_argument('--model', default="bert-base-uncased", type=str)
    parser.add_argument('--ckpt_path', default="/mmu_nlp/wuxing/suzhenpeng/cotmae-dial/dialo_finetune/model/ubuntu_epoch5_lr5e-5_b64_correct_data/step150000_en0.15_de0.15_lr3e-4_disable_decoder/best/opt.pth", type=str)
    parser.add_argument('--context_seq_length', default=256,type=int)
    parser.add_argument('--response_seq_length', default=64,type=int)
    parser.add_argument('--batch_size',default=200,type=int)
    parser.add_argument('--k_list',nargs = '+',type=int,default=[1,2,5],help="R@k,recalled top k")
    parser.add_argument('--seed', default=42, type=int)
    return parser.parse_args()

def test(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = dialForPretraining.from_pretrained(args)
    model = utils.load_ckpt_test(model, ckpt_path=args.ckpt_path)
    model = model.to(device)
    tokenizer = BertTokenizer.from_pretrained(args.model)

    test_set = load_dataset(
        'json',
        data_files=args.test_path,
        block_size=2**25
    )['train']

    collator = dialCollator(context_seq_length=args.context_seq_length,response_seq_length=args.response_seq_length,tokenizer=tokenizer)
    test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
            drop_last=True,
            num_workers=10,
            pin_memory=True,
        )
    metrics=choose_case(model,test_loader,args,mode="test")
    print(metrics)

def choose():
    path1="/mmu_nlp/wuxing/suzhenpeng/cotmae-dial/dialo_finetune/choose_label_bert+.txt"
    path2="/mmu_nlp/wuxing/suzhenpeng/cotmae-dial/dialo_finetune/choose_label_cotmae.txt"
    file_path="/mmu_nlp/wuxing/suzhenpeng/cotmae-dial/dialo_finetune/data/ubuntu/val.txt"
    save_path = "/mmu_nlp/wuxing/suzhenpeng/cotmae-dial/dialo_finetune/save_smples.txt"
    with open(path1,"r") as f:
        p1=f.readlines()
    with open(path2,"r") as f:
        p2=f.readlines()
    with open(file_path) as f:
        data = f.readlines()
    print(len(data))
    save_data = []
    for i in range(len(p1)):
        s1=p1[i].strip()
        s2=p2[i].strip()
        if s1=="0" and s2!="0":
            # save_data.extend(data[i*10:(i+1)*10])
            obj = {"context":data[i*10],"bert+":data[i*10+int(s2)]}
            save_data.append(json.dumps(obj)+"\n")
    with open(save_path,"w") as f:
        f.writelines(save_data)
if __name__ == "__main__":
    # args = parser_args()
    # test(args)
    choose()