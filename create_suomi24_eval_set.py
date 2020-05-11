from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import argparse

from encoderDecoder_global_variables import SEED

parser = argparse.ArgumentParser(description='Encoder-Decoder create eval set from file')
parser.add_argument('source_file_name', type=str,
                    help='name of the large eval file')
parser.add_argument('target_file_name', type=str,
                    help='name of the smaller shuffled eval file')
parser.add_argument('how_many', type=int,
                    help='how many lines to new dataset')

args = parser.parse_args()

random.seed(SEED)

# Extracts pairs of sentences from conversations
def createSentencePairsList(inputfilename):
    qa_pairs = []
    inputLine = ""
    targetLine = "hyvää uutta vuotta !"
    with open(inputfilename, 'r', encoding='utf-8') as txtfile:
        for line in txtfile:
            inputLine = targetLine
            targetLine = line.strip()            
            # Filter wrong samples (if one of the lists is empty)
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs


def create_N_choose_k_file(qa_pairs, output_csv_file_name, N):
    # TODO should I only pick long enough sentences?
    with open(output_csv_file_name, 'w', encoding='utf-8') as output_file:
        lines_count = len(qa_pairs)
        assert lines_count >= N + 2, "Not enough lines, eg. options for fake"
        output_file.write("TEXT¤CHOICE_SENTENCES\n")

        for i in range(lines_count - 1):
            bad_indices = []
            bad_indices.append(i)
            bad_indices.append(i + 1)
            answers = []
            question = eval_lines[i].strip()
            true_answer = eval_lines[i + 1].strip()
            answers.append(true_answer)

            for _ in range(N - 1):
                fake_answer = random.choice([x for x in range(lines_count) if x not in bad_indices])
                answers.append(eval_lines[fake_answer].strip())
                bad_indices.append(fake_answer)

            line_to_write = '¤'.join([question, '|'.join(answers)]) + '\n'
            output_file.write(line_to_write)


qa_pairs = createSentencePairsList(args.source_file_name)
random.shuffle(qa_pairs)

writeSentencePairsTXT(qa_pairs[:args.how_many], args.target_file_name)
