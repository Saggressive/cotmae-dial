import torch
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
tokenizer_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=False
    )
print("a")