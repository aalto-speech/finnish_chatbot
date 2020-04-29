import os


# Corpus & Data variables
corpus_name = "suomi24"
corpus = os.path.join("../data", corpus_name)
source_txt_file = "10k_suomi24_morfs.txt"
output_csv_file = "form_10k_v7.csv"

parent_folder_name = "transformer_delethis"

# Configure models
model_name = 'transformer_model'
batch_size = 32

emsize = 200 # embedding dimension
nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value

# Configure training/optimization
clip = 0.5
learning_rate = 0.001
n_iteration =32000 
print_every = 4
save_every = 4000
