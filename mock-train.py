import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from model import EncoderCNN, DecoderWithAttention
from utils import *

import os
import json
import pickle

# Data parameters
data_folder = 'data'  # folder with data files saved by create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

# Read word map
word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
with open(word_map_file, 'r') as j:
    word_map = json.load(j)

vocab_size = len(word_map)

# Load data
f = open(os.path.join(data_folder, 'sample-input.pkl'), 'rb')
sample_input = pickle.load(f)
img, caps, capslens = sample_input[0], sample_input[1], sample_input[2]

encoder = EncoderCNN(attention=True)
decoder_attention = DecoderWithAttention(vocab_size=vocab_size)

# Move to GPU, if available
decoder_attention = decoder_attention.to(device)
encoder = encoder.to(device)

img = encoder(img)
scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder_attention(
    img, caps, capslens)

# Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
targets = caps_sorted[:, 1:]

# Remove timesteps that we didn't decode at, or are pads
# pack_padded_sequence is an easy trick to do this
scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

# Calculate loss
# Loss function
criterion = nn.CrossEntropyLoss().to(device)
loss = criterion(scores, targets)
print(loss)