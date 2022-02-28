import os
import random

import numpy as np
import torch

SEED = 42
# Step 1: Set the Random Seed in the program entry 
# earlier is better
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# Step 2: Load BERT hyperparameters
from transformers import AutoConfig, AutoTokenizer

model_name = "bert-base-uncased"
config = AutoConfig.from_pretrained(model_name)   # 'bart-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 3: Define our BERT
from bert_model import BertModel

our_bert = BertModel()

# Step 4: Get the model output without dropout
our_bert.eval()
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
last_hidden_states, _, _ = our_bert(**inputs)
print(last_hidden_states)

# Step 1: Set the Random Seed in the program entry 
# IMPORTANT! SEED must be the same as ours!
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# Step 2: Define a randomly initialized BART model
from transformers import AutoTokenizer, AutoConfig, AutoModel

config = AutoConfig.from_pretrained(model_name)   # 'bart-base'
model = AutoModel.from_config(config)
tokenizer = AutoTokenizer.from_pretrained(model_name)  # 'bart-base'

# Step 3: Get the model output without dropout
model.eval()
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
print(last_hidden_states)