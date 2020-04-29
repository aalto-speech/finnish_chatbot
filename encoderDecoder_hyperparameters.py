import os


# Corpus & Data variables 
corpus_name = "opensubs"
corpus = os.path.join("../data", corpus_name)
source_txt_file = "stigs_opensubs.trg"
source_csv_file = "stigs_opensubs_v2.csv"

# Load/Assemble voc and pairs
save_dir = os.path.join("../models", "enc-dec_opensubs", args.job_name)

# Configure models
model_name = 'encoderDecoder'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 800
encoder_n_layers = 4
decoder_n_layers = 4
dropout = 0.4
batch_size = 32

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 0.5
learning_rate = 0.00001
decoder_learning_ratio = 5.0
n_iteration = 1600000
print_every = 100
save_every = 400000
