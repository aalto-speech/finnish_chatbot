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


def printLines(file, n=10):
    with open(file, 'r', encoding='utf-8') as datafile:
        for i, line in enumerate(datafile):
            print(line)
            if i > n:
                break

printLines(os.path.join(corpus, source_txt_file))


# In[4]:


# Extracts pairs of sentences from conversations
def createSentencePairsCSV(inputfilename, outputfilename):
    delimiter = '\t'
    # Unescape the delimiter
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))
    
    qa_pair = []
    inputLine = ""
    targetLine = "hyvää uutta vuotta !"
    
    with open(inputfilename, 'r', encoding='utf-8') as txtfile,\
        open(outputfilename, 'w', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=delimiter, lineterminator='\n')
        
        for line in txtfile:
            inputLine = targetLine
            targetLine = line.strip()            
            # Filter wrong samples (if one of the lists is empty)
            if inputLine and targetLine:
                qa_pair = [inputLine, targetLine]
                writer.writerow(qa_pair)
                


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


# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
EOT_token = 3  # End-of-turn token

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", EOT_token: "EOT"}
        self.num_words = 4  # Count SOS, EOS, PAD, EOT

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", EOT_token: "EOT"}
        self.num_words = 4 # Count default tokens

        for word in keep_words:
            self.addWord(word)


# In[7]:


MAX_LENGTH = 100  # Maximum sentence length to consider

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"[^a-zA-Z\x7f-\xff.!?+]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

# Read query/response pairs and return a voc object
def readVocs(datafile, corpus_name):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [" EOT ".join([normalizeString(s) for s in l.split('\t')]) for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs

# Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filterPair(p):
    # Input sequences need to preserve the last word for EOS token
    return len(p.split(' ')) < MAX_LENGTH - 2

# Filter pairs using filterPair condition
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

# Using the functions defined above, return a populated voc object and pairs list
def loadPrepareData(corpus, corpus_name, datafile, save_dir):
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair)
    print("Counted words:", voc.num_words)
    return voc, pairs


# In[8]:


# Load/Assemble voc and pairs
save_dir = os.path.join("../models", "save_suomi24_transformer")
voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)
# Print some pairs to validate
print("\npairs:")
for pair in pairs[:10]:
    print(pair)


# In[9]:


MIN_COUNT = 2    # Minimum word count threshold for trimming

def trimRareWords(voc, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair
        keep_input = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs


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


def inputIndexesFromSentence(voc, sentence):
    li = [voc.word2index[word] for word in sentence.split(' ')]
    return [SOS_token] + li + [EOS_token]

def outputIndexesFromSentence(voc, sentence):
    lo = [voc.word2index[word] for word in sentence.split(' ')]
    return lo + [EOS_token] + [PAD_token]

def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [inputIndexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [outputIndexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x.split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair)
        output_batch.append(pair)
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len


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


class TransformerModel(nn.Module):
    # From pytorch examples
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, embedding, dropout=0.5):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = embedding
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)


# In[14]:


# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    # From pytorch examples
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# In[15]:


def maskNLLLoss(inp, target, mask):
    print("NLLLoss")
    print(inp)
    print(target)
    print(mask)
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    print(loss)
    return loss, nTotal.item()


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
