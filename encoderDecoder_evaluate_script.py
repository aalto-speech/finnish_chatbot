#!/usr/bin/env python
# coding: utf-8

# Chatbot Tutorial (Evaluate)
# ================
# **Author:** `Matthew Inkawhich <https://github.com/MatthewInkawhich>`_
# 


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
import sys
import argparse
from shutil import copyfile
import morfessor

from encoderDecoder_voc import Voc
from encoderDecoder_global_variables import *
from encoderDecoder_models import *
from encoderDecoder_evaluation import *

parser = argparse.ArgumentParser(description='Encoder-Decoder script to produce evaluation metrics')
parser.add_argument('hyperparameters_file', type=str, help='Path to the hyperparameteres file')
parser.add_argument('evaluation_file', type=str, help='Path to the evaluation csv/txt file. '
                                                      'txt file already correct tokens. csv will be tokenized')
parser.add_argument('model_tar', type=str, help='Path to the tar file of the model')
parser.add_argument('morfessor_model', type=str, help='The morfessor model')
args = parser.parse_args()

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

random.seed(SEED)
torch.manual_seed(SEED)

morfessorIO = morfessor.MorfessorIO()
morfessorModel = morfessorIO.read_binary_model_file(args.morfessor_model)

# Set checkpoint to load from; set to None if starting from scratch
copyfile(args.hyperparameters_file, "temp_hyperparameters.py")

from temp_hyperparameters import *

voc = Voc(corpus_name)

# Load model if a loadFilename is provided
if args.model_tar:
    # If loading on same machine the model was trained on
    #checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    checkpoint = torch.load(args.model_tar, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']


print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)
if args.model_tar:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

if args.model_tar:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')

# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
evaluation_file_extension = os.path.splitext(args.evaluation_file)[1]
if "csv" in evaluation_file_extension:
    metrics = calculate_evaluation_metrics(args.evaluation_file, voc, encoder, decoder, embedding, 10, 5, "¤",
                                            device, morfessorModel)

elif "txt" in evaluation_file_extension:
    create_N_choose_k_file(args.evaluation_file, "temp_enc2dec_evalution.csv", 10)
    metrics = calculate_evaluation_metrics("temp_enc2dec_evalution.csv", voc, encoder, decoder, embedding, 10, 5, "¤", device)

print('The results are r@k: {0}, N-choose-k: {1}, ppl: {2}, char ppl: {3}, BLEU word: {4}, BLEU morf: {5}, chrF word: {6}, chrF morf: {7}'.format(metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], metrics[5], metrics[6], metrics[7]))
#os.remove("temp_hyperparameters.py")
#os.remove("temp_evaluation.csv")
