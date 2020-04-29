import os


# Corpus & Data variables 
corpus_name = "opensubs"
corpus = os.path.join("../data", corpus_name)
source_txt_file = "10k_suomi24_morfs.txt"
source_csv_file = "delethis_v1.csv"

parent_folder_name = "enc-dec_delethis"

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
n_iteration = 1600
print_every = 1
save_every = 400
