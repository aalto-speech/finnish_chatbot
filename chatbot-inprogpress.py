# coding: utf-8

# In[ ]:




# 
# Chatbot Tutorial
# ================
# **Author:** `Matthew Inkawhich <https://github.com/MatthewInkawhich>`_
# 

# Preparations
# ------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math

from encoderDecoder_prep_data import *
from encoderDecoder_voc import Voc
from encoderDecoder_global_variables import *

with open(__file__) as f:
    print(f.read())

################################################
######## ALL VARIABLES HERE ####################
################################################

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

random.seed(SEED)
torch.manual_seed(SEED)

# Corpus & Data variables 
corpus_name = "suomi24"
corpus = os.path.join("../data", corpus_name)
source_txt_file = "10k_suomi24_morfs.txt"
source_csv_file = "formatted_10k_suomi24_v4.csv"


# Define path to new file
inputfile = os.path.join(corpus, source_txt_file)
datafile = os.path.join(corpus, source_csv_file)
delimiter = '\t'
# Unescape the delimiter
delimiter = str(codecs.decode(delimiter, "unicode_escape"))


# Load/Assemble voc and pairs
save_dir = os.path.join("../models", "10k_s2s_suomi24")


small_batch_size = 5

# Configure models
model_name = 'cb_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 3
decoder_n_layers = 3
dropout = 0.3
batch_size = 32

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = None
checkpoint_iter = 4000
#loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                            '{}_checkpoint.tar'.format(checkpoint_iter))

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 0.75
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 4000
print_every = 100
save_every = 4000




###############################################
######### RUNNING THE SCRIPT ##################
###############################################

printLines(os.path.join(corpus, source_txt_file))

# Load & Preprocess Data
# ----------------------
print("\nProcessing corpus...")

# Write new csv file
print("\nWriting newly formatted file...")
createSentencePairsCSV(inputfile, datafile)

# Print a sample of lines
print("\nSample lines from file:")
printLines(datafile)


# Print some pairs to validate
voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)
print("\npairs:")
for pair in pairs[:10]:
    print(pair)

# Load and trim data
# ~~~~~~~~~~~~~~~~~~

# Trim voc and pairs
pairs = trimRareWords(voc, pairs, MIN_COUNT)


# Example for validation
batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

print("input_variable:", input_variable)
print("lengths:", lengths)
print("target_variable:", target_variable)
print("mask:", mask)
print("max_target_len:", max_target_len)


# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']


print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')


# Run Training
# ~~~~~~~~~~~~
# 

# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# If you have cuda, configure cuda to call
for state in encoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

for state in decoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()
    
# Run training iterations
print("Starting Training!")
trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
           embedding, encoder_n_layers, decoder_n_layers, teacher_forcing_ratio, save_dir, n_iteration, batch_size,
           print_every, save_every, clip, corpus_name, loadFilename, device)





# Conclusion
# ----------
