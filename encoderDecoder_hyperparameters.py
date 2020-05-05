import os


# Corpus & Data variables 
corpus_name = "suomi24"
corpus = os.path.join("../data", corpus_name)
source_txt_file = "1M_processed_suomi24_2001.txt"
source_csv_file = "1M_processed_suomi24_2001.csv"

parent_folder_name = "enc-dec_suomi24"

# Configure models
model_name = 'encoderDecoder'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 3
decoder_n_layers = 3
dropout = 0.3
batch_size = 16

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 0.9
learning_rate = 0.00001
decoder_learning_ratio = 5.0
n_iteration = 3200000
print_every = 100
save_every = 400000
