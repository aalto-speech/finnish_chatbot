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
import operator
import math
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.chrf_score import corpus_chrf

from spacy.lang.fi import Finnish

from encoderDecoder_global_variables import *
from encoderDecoder_prep_data import *
from encoderDecoder_training import maskNLLLoss


# Define Evaluation
# -----------------
# 
# Greedy decoding
# ~~~~~~~~~~~~~~~
# 

class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length, device):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores


# Evaluate my text
# ~~~~~~~~~~~~~~~~
# 

def evaluate(encoder, decoder, searcher, voc, sentence, device, max_length=MAX_LENGTH):
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
    tokens, scores = searcher(input_batch, lengths, max_length, device)
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


def calculate_loss(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding, device, batch_size):
    encoder.eval()
    decoder.eval()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Forward batch of sequences through decoder one time step at a time
    for t in range(max_target_len):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        # Teacher forcing: next input is current target
        decoder_input = target_variable[t].view(1, -1)
        # Calculate and accumulate loss
        mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t], device)
        loss += mask_loss
        print_losses.append(mask_loss.item() * nTotal)
        n_totals += nTotal

    return sum(print_losses) / n_totals


def prepare_sentence(s, voc):
    s_norm = normalizeString(s)
    s_morfs = s_norm.split()
    return_morfs = []
    for morf in s_morfs:
        if morf not in voc.word2index:
            continue
        else:
            return_morfs.append(morf)
    return " ".join(return_morfs)


def morfenize_fi(text, morfessorModel, spacy_fi):
    text = text.replace(" <MS> ", " . ")
    text = text.lower()
    tokens = [tok.text for tok in spacy_fi.tokenizer(text)]
    sentenceAsMorfs = []
    for token in tokens:
        morfs, _ = morfessorModel.viterbi_segment(token)
        if len(morfs) == 1:
            sentenceAsMorfs.append(morfs[0])
        else:
            sentenceAsMorfs.append(morfs[0] + "+")
            for morf in morfs[1:-1]:
                sentenceAsMorfs.append("+" + morf + "+")
            sentenceAsMorfs.append("+" + morfs[-1])
    return " ".join(sentenceAsMorfs)


def morf_list_to_word_list(sentence):
    word_sentence = " ".join(sentence)
    word_sentence = word_sentence.replace("+ +","").replace(" +", "").replace("+ ", "")
    word_sentence = word_sentence.split()
    return word_sentence


def calculate_evaluation_metrics(eval_file_name, voc, encoder, decoder, embedding, N, k, delimiter, device, skip_indices=[], print_indices=[], morfessor=None):

    spacy_fi = Finnish()
    searcher = GreedySearchDecoder(encoder, decoder)

    most_common_word = max(voc.word2count.items(), key=operator.itemgetter(1))[0]

    true_first = 0
    true_top_k = 0
    corpus_hypothesis = []
    corpus_references = []
    true_answer_losses = []
    hypotheses_for_humans = []


    df = pd.read_csv(eval_file_name, sep=delimiter, engine='python')
    for index, row in df.iterrows():
        if index in skip_indices:
            continue

        question = row['TEXT'].strip()  # TODO what if question or answer is zero, make sure it is not in create file?
        if morfessor:
            question = morfenize_fi(question, morfessor, spacy_fi)

        answers = row['CHOICE_SENTENCES'].split('|')
        assert len(answers) >= N, "CSV file does not have enough choices for value of given N"
        answers = answers[:10]
        assert N >= k, "N is not larger than or equal k"

        losses = []
        prepared_question = prepare_sentence(question, voc)
        if len(prepared_question) == 0:
            prepared_question = most_common_word

        first_answer = True
        for answer in answers:
            answer = answer.strip()
            if morfessor:
                answer = morfenize_fi(answer, morfessor, spacy_fi)

            prepared_answer = prepare_sentence(answer, voc)
            if len(prepared_answer) == 0:
                prepared_answer = most_common_word

            # Following gets the length for character normalized perplexity, and saves ref and hyp for BLEU
            if first_answer:

                correct_answer_length_char = max(len(prepared_answer), 1)
                correct_answer_length_tokens = max(len(prepared_answer.split(' ')), 1)

                # Had some problem with indexing so this is done twice for every row
                evaluation_batch = [batch2TrainData(voc, [[prepared_question, prepared_answer]])]
                input_variable, lengths, target_variable, mask, max_target_len = evaluation_batch[0]

                loss = calculate_loss(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder,
                                      embedding, device, 1)
                true_answer_losses.append([loss, correct_answer_length_char, correct_answer_length_tokens])
                first_answer = False


                # Next is for BLEU
                hypothesis = evaluate(encoder, decoder, searcher, voc, prepared_question, device, max_length=MAX_LENGTH)
                try:
                    first_EOS_index = hypothesis.index(voc.index2word[EOS_token])
                except ValueError:
                    first_EOS_index = MAX_LENGTH  # Generated hypothesis has 50 tokens, none is EOS, so is added as 51th.
                hypothesis = hypothesis[:first_EOS_index]
                corpus_hypothesis.append(hypothesis)
                if index in print_indices:
                    hypothesis_string = " ".join(morf_list_to_word_list(hypothesis))
                    hypotheses_for_humans.append([str(index), row['TEXT'].strip(), hypothesis_string])

                answer_in_tokens = answer.split()
                corpus_references.append(answer_in_tokens)

            evaluation_batch = [batch2TrainData(voc, [[prepared_question, prepared_answer]])]
            input_variable, lengths, target_variable, mask, max_target_len = evaluation_batch[0]

            loss = calculate_loss(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder,
                    embedding, device, 1)
            losses.append(loss)
        if np.argmin(np.asarray(losses)) == 0:
            true_first += 1
        if 0 in np.asarray(losses).argsort()[:k]:
            true_top_k += 1

    fraction_of_correct_firsts = true_first / len(true_answer_losses)
    franction_of_N_choose_k = true_top_k / len(true_answer_losses)

    np_true_answer_losses = np.asarray(true_answer_losses)
    #perplexity = np.exp(np.mean(np_true_answer_losses[:,0]))
    cross_entropy = np.mean(np_true_answer_losses[:,0])

    token_to_character_modifier = np_true_answer_losses[:,2] / np_true_answer_losses[:,1]
    #char_perplexity = np.exp(np.mean(np_true_answer_losses[:,0] * token_to_character_modifier))
    char_cross_entropy = np.mean(np_true_answer_losses[:,0] * token_to_character_modifier)

    bleu_morf = corpus_bleu(corpus_references, corpus_hypothesis)
    chrf_morf = corpus_chrf(corpus_references, corpus_hypothesis)
    
    corpus_references_word = [morf_list_to_word_list(sentence) for sentence in corpus_references]
    corpus_hypothesis_word = [morf_list_to_word_list(sentence) for sentence in corpus_hypothesis]
    print(corpus_hypothesis_word)
    print("FOR HUMANS")
    for answer_for_human in hypotheses_for_humans:
        print(" --- ".join(answer_for_human))

    bleu_word = corpus_bleu(corpus_references_word, corpus_hypothesis_word)
    chrf_word = corpus_chrf(corpus_references_word, corpus_hypothesis_word)

    return fraction_of_correct_firsts, franction_of_N_choose_k, cross_entropy, char_cross_entropy, bleu_word, bleu_morf, chrf_word, chrf_morf


def create_N_choose_k_file(source_txt_file_name, output_csv_file_name, N):
    # TODO should I only pick long enough sentences?
    with open(source_txt_file_name, 'r', encoding='utf-8') as source_file,\
            open(output_csv_file_name, 'w', encoding='utf-8') as output_file:
        eval_lines = source_file.readlines()
        lines_count = len(eval_lines)
        assert lines_count >= N + 2, "Not enough lines, eg. options for fake"
        output_file.write("TEXT¤CHOICE_SENTENCES\n")

        for i in range(lines_count - 1):
            bad_indices = []
            bad_indices.append(i)
            bad_indices.append(i + 1)
            answers = []
            question = eval_lines[i].strip()
            true_answer = eval_lines[i + 1].strip()
            if not (len(question) > 10 and len(true_answer) > 10):
                continue
            answers.append(true_answer)

            for _ in range(N - 1):
                fake_answer = random.choice([x for x in range(lines_count) if x not in bad_indices])
                answers.append(eval_lines[fake_answer].strip())
                bad_indices.append(fake_answer)

            line_to_write = '¤'.join([question, '|'.join(answers)]) + '\n'
            output_file.write(line_to_write)
