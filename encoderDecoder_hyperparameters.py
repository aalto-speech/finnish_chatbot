import os


# Corpus & Data variables 
corpus_name = "opensubs"
corpus = os.path.join("../data", corpus_name)
source_txt_file = "1M_stigs_opensubs.trg"
source_csv_file = "1M_stigs_opensubs.csv"

parent_folder_name = "enc-dec_opensubs"

# Configure models
model_name = 'encoderDecoder'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 800
encoder_n_layers = 3
decoder_n_layers = 3
dropout = 0.4
batch_size = 8

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 0.9
learning_rate = 0.00001
decoder_learning_ratio = 5.0
n_iteration = 4800000
print_every = 100
save_every = 800000
