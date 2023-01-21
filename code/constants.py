# device used for training:
import torch
from pathlib import Path

# DATA PARARMETERS
INPUT_DATA_COLS = ["CO(GT)"]
# add features to end to make itself predict the output col - lstm not limit to 1 feature
# input_data_cols = ["CO(GT)", "PT08.S1(CO)", "NMHC(GT)"]
INPUT_DATA_FEATURES = len(INPUT_DATA_COLS)
OUTPUT_DATA_COLS = ["CO(GT)"]
OUTPUT_DATA_FEATURES = len(OUTPUT_DATA_COLS)
PERCENT_TEST_TRAIN = 0.8  # test-train split percentage
LOOKBACK = 5  # number of units used to make prediction
PREDICT = 1  # number of units that will be predicted
BATCH_SIZE = 8  # number of examples run through in parallel
SHUFFLE_TRAIN_TEST_DATA = True  # whether you want to shuffle data during training or not

# TRAINING PARAMETERS
LEARNING_RATE = 0.0001  # the learning rate
N_EPOCHS = 500  # number of epochs
BATCH_SIZE = 32

# PATH TO ACCESS TRAIN AND VALIDATION DATA
#VALID_DATA_PATH = "/path/to/valid/data/file"
#TRAIN_DATA_PATH = "/path/to/valid/data/file"
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

SEQ2SEQ_VERBOSE = True

# available model names
LSTM_MODEL_NAME = "lstm"
SEQ2SEQ_MODEL_NAME = "seq2seq"

# selected model to be useed:
CURRENT_MODEL = "seq2seq"
