# device used for training:
import torch
from pathlib import Path

# DATA PARARMETERS
INPUT_DATA_COLS = ["CO(GT)"]
# add features to end to make itself predict the output col - lstm not limit to 1 feature
#INPUT_DATA_COLS = ["CO(GT)", "PT08.S1(CO)", "NMHC(GT)"]
INPUT_DATA_FEATURES = len(INPUT_DATA_COLS)
OUTPUT_DATA_COLS = ["CO(GT)"]
OUTPUT_DATA_FEATURES = len(OUTPUT_DATA_COLS)
PERCENT_TEST_TRAIN = 0.8  # test-train split percentage
LOOKBACK = 5  # number of units used to make prediction
PREDICT = 1  # number of units that will be predicted
BATCH_SIZE = 8  # number of examples run through in parallel
SHUFFLE_TRAIN_TEST_DATA = False  # whether you want to shuffle data during training or not
RANDOM_SPLIT_SEED = 1234 # seed for shuffle test train split (if true)

# TRAINING PARAMETERS
LEARNING_RATE = 0.01  # the learning rate
N_EPOCHS = 7  # number of epochs

# PATH TO ACCESS TRAIN AND VALIDATION DATA
X_TRAIN_PATH ='./data/processed_train_prediction.npy'
X_TEST_PATH = './data/processed_train_lookback.npy'
Y_TRAIN_PATH ='./data/processed_validation_prediction.npy'
Y_TEST_PATH = './data/processed_validation_lookback.npy'
MODEL_PATH = "data/model.pkl"



# MODEL PARAMETERS
LSTM_INPUT_SIZE = INPUT_DATA_FEATURES
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LSTM_INPUT_SIZE = INPUT_DATA_FEATURES
LSTM_OUT_SIZE = OUTPUT_DATA_FEATURES
LSTM_HIDDEN_COUNT = 4  # number of lstm cells
LSTM_DROPOUT = 0.  # none lol
LSTM_LAYER_COUNT = 2  # number of layers in lstm cell
LSTM_INPUT_SEQ_LEN = LOOKBACK  # length of input sequence to LSTM
LSTM_VERBOSE = False

SEQ2SEQ_VERBOSE = False

# available model names
LSTM_MODEL_NAME = "lstm"
SEQ2SEQ_MODEL_NAME = "seq2seq"
"""
seq2seq: input torch.Size([5, 8, 1])
seq2seq: outputs torch.Size([1, 8, 1])
seq2seq: hn torch.Size([2, 8, 4])
enc: inp shape torch.Size([5, 8, 1])
enc: hn shape torch.Size([2, 8, 4])
seq2seq: enc_out torch.Size([5, 8, 4])
seq2seq: enc_hidden torch.Size([2, 8, 4])
seq2seq: dec_inp torch.Size([8, 1])
seq2seq: dec_hidden torch.Size([2, 8, 4])
dec: inp shape torch.Size([8, 1])
dec: lstm out shape torch.Size([1, 8, 4])
dec: hn shape torch.Size([8, 4])
dec: output shape torch.Size([8, 1])
dec: final output shape torch.Size([8, 1])
dec: final hn shape torch.Size([2, 8, 4])
 seq2seq: final_dec_out torch.Size([8, 1])
 seq2seq: dec_out torch.Size([8, 1])
 seq2seq: dec_hidden hn torch.Size([8, 4])
seq2seq: outputs torch.Size([1, 8, 1])
"""