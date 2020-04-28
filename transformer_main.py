#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# 
# Chatbot Tutorial
# ================
# **Author:** `Matthew Inkawhich <https://github.com/MatthewInkawhich>`_
# 
# 

# Preparations
# ------------
# 
# 

# In[1]:


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


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


# Load & Preprocess Data
# ----------------------
# 
# 
# 
# 

# In[2]:


corpus_name = "suomi24"
source_txt_file = "suomi24_morfs_2001+2.txt"
output_txt_file = "formatted_morfs_suomi24_2001+2.txt"
corpus = os.path.join("../data", corpus_name)


# In[3]:



printLines(os.path.join(corpus, source_txt_file))


# In[4]:


                


# In[5]:


# Define path to new file
datafile = os.path.join(corpus, output_txt_file)
inputfile = os.path.join(corpus, source_txt_file)

# Load lines and process conversations
print("\nProcessing corpus...")

# Write new csv file
#print("\nWriting newly formatted file...")
#createSentencePairsCSV(inputfile, datafile)


# Print a sample of lines
print("\nSample lines from file:")
printLines(datafile)


# Load and trim data
# ----------
# 
# 

# In[6]:





# In[7]:





# In[8]:


# Load/Assemble voc and pairs
save_dir = os.path.join("../models", "save_suomi24_transformer")
voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)
# Print some pairs to validate
print("\npairs:")
for pair in pairs[:10]:
    print(pair)


# In[9]:





# In[10]:


# Trim voc and pairs
pairs = trimRareWords(voc, pairs, MIN_COUNT)


# Prepare Data for Models
# -----------------------
# 
# 
# 
# 

# In[11]:




# In[12]:


# Example for validation
small_batch_size = 5
batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])

input_variable, lengths, target_variable, mask, max_target_len = batches

print("input_variable:", input_variable)
print("lengths:", lengths)
print("target_variable:", target_variable)
print("mask:", mask)
print("max_target_len:", max_target_len)


# Define Models
# -------------
# 
# 

# In[13]:




# In[15]:



# In[16]:


criterion = nn.NLLLoss(ignore_index=0)


# In[17]:


def train(input_variable, lengths, target_variable, mask, max_target_len, transformer, embedding,
          optimizer, batch_size, clip, max_length=MAX_LENGTH):

    # Zero gradients
    optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Forward pass through encoder
    outputs = transformer(input_variable)
    
    loss = criterion(outputs.view(-1, ntokens), target_variable.view(-1))
    
#    for t in range(max_target_len):
#        # Calculate and accumulate loss
#        mask_loss, nTotal = maskNLLLoss(outputs[t], target_variable[t], mask[t])
#        loss += mask_loss
#        print_losses.append(mask_loss.item() * nTotal)
#        n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(transformer.parameters(), clip)

    # Adjust model weights
    optimizer.step()

    return loss.item()


# In[18]:


def trainIters(model_name, voc, pairs, transformer, optimizer, embedding, nlayers, save_dir, n_iteration, batch_size, print_every, save_every, clip, corpus_name, loadFilename):

    # Load batches for each iteration
    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                      for _ in range(n_iteration)]

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, transformer,
                     embedding, optimizer, batch_size, clip)
        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        # Save checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, corpus_name, '{}_{}'.format(nlayers, nhid))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'tra': transformer.state_dict(),
                'opt': optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


# In[19]:


def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")


# Run Model
# ---------
# 
# 
# 
# 

# In[20]:


# Configure models
model_name = 'transformer_model'
batch_size = 32

ntokens = voc.num_words # the size of vocabulary
emsize = 300 # embedding dimension
nhid = 300 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 6 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 6 # the number of heads in the multiheadattention models
dropout = 0.3 # the dropout value

# Set checkpoint to load from; set to None if starting from scratch
#save_dir = os.path.join("../models", "save_suomi24_2001")
#voc = Voc(corpus_name)
#MAX_LENGTH = 30  # Maximum sentence length to consider

loadFilename = None
#checkpoint_iter = 4000
#loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                            '{}_checkpoint.tar'.format(checkpoint_iter))


# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    #checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    transformer_sd = checkpoint['tra']
    optimizer_sd = checkpoint['opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']


print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(ntokens, emsize)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
if loadFilename:
    transformer.load_state_dict(encoder_sd)
    
# Use appropriate device
transformer = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, embedding, dropout).to(device)
print('Models built and ready to go!')


# In[21]:


# Configure training/optimization
clip = 0.5
learning_rate = 0.01
decoder_learning_ratio = 5.0
n_iteration =1600000 
print_every = 4000
save_every = 50000

# Ensure dropout layers are in train mode
transformer.train()

# Initialize optimizers
print('Building optimizers ...')
optimizer = optim.Adam(transformer.parameters(), lr=learning_rate)
if loadFilename:
    optimizer.load_state_dict(optimizer_sd)

# If you have cuda, configure cuda to call
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()
    
# Run training iterations
print("Starting Training!")
trainIters(model_name, voc, pairs, transformer, optimizer,
           embedding, nlayers, save_dir, n_iteration, batch_size,
           print_every, save_every, clip, corpus_name, loadFilename)


# Run Evaluation
# ~~~~~~~~~~~~~~
# 
# To chat with your model, run the following block.
# 
# 
# 

# In[ ]:


# Set dropout layers to eval mode
transformer.eval()


# Begin chatting (uncomment and run the following line to begin)
#evaluateInput(encoder, decoder, searcher, voc)


# Conclusion
# ----------
# 
# That’s all for this one, folks. Congratulations, you now know the
# fundamentals to building a generative chatbot model! If you’re
# interested, you can try tailoring the chatbot’s behavior by tweaking the
# model and training parameters and customizing the data that you train
# the model on.
# 
# Check out the other tutorials for more cool deep learning applications
# in PyTorch!
# 
# 
# 
