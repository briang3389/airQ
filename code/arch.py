# imports
import torch.nn as nn
from constants import *


class encoder_lstm(nn.Module):
    def __init__(self, inp_size, hid_size, dropout, layer_count):
        super(encoder_lstm, self).__init__()
        self._input_size = inp_size  # number of input features
        self._hidden_size = hid_size  # number of features in hidden state
        self._layer_count = layer_count  # number of stacked LSTM's
        self._dropout = dropout

        self._lstm = nn.LSTM(input_size=self._input_size,
                             hidden_size=self._hidden_size,
                             num_layers=self._layer_count,
                             dropout=self._dropout)

        self._drop = nn.Dropout(dropout)

    def forward(self, x):
        # input shape: (seq_len, count units in batch, input size)
        if SEQ2SEQ_VERBOSE:
            print("enc: inp shape", x.shape)
        out, hidden = self._lstm(x)
        if SEQ2SEQ_VERBOSE:
            # print("enc: out shape",out.shape)
            print("enc: hn shape", hidden[0].shape)

        return out, hidden

    def init_hidden(self, batch_size):
        (h_0, c_0) = (torch.randn(self._layer_count, batch_size, self._hidden_size).requires_grad_().to(DEVICE),
                      torch.randn(self._layer_count, batch_size, self._hidden_size).requires_grad_().to(DEVICE))
        return (h_0, c_0)


class decoder_lstm(nn.Module):
    def __init__(self, inp_size, out_size, hid_size, layer_count, drop):
        super(decoder_lstm, self).__init__()
        self._inp_feature_size = inp_size  # number of input features
        self._out_feature_size = out_size  # number of input features
        self._hid_size = hid_size  # size of hidden size
        self._layer_count = layer_count  # number of hidden lstm cells
        self._dropout = drop

        self._lstm = nn.LSTM(input_size=self._inp_feature_size,
                             hidden_size=self._hid_size,
                             num_layers=self._layer_count,
                             dropout=self._dropout)

        self._l_in = nn.Linear(self._hid_size, self._inp_feature_size)
        self._l_out = nn.Linear(self._hid_size, self._out_feature_size)

    def forward(self, x, enc_hidden_states):
        # x: 2d - the last input time step
        # enc_hidden_states: the last hidden decoder time step
        if SEQ2SEQ_VERBOSE:
            print("dec: inp shape", x.shape)
        # lstm_out, (hn, cn) = self._lstm(x,enc_hidden_states)
        lstm_out, (hn, cn) = self._lstm(x.unsqueeze(0), enc_hidden_states)
        if SEQ2SEQ_VERBOSE:
            print("dec: lstm out shape", lstm_out.shape)
            print("dec: hn shape", hn[0].shape)

        output = self._l_in(lstm_out.squeeze(0))
        final_output = self._l_out(lstm_out.squeeze(0))

        if SEQ2SEQ_VERBOSE:
            print("dec: output shape", output.shape)
            print("dec: final output shape", final_output.shape)
            print("dec: final hn shape", hn.shape)
        return final_output, output, hn


def print_model(model):
    sum = 0
    print(model)
    for param in model.parameters():
        if param.requires_grad:
            sum += param.numel()
    print("total trainable parameters:", sum)


class seq2seq(nn.Module):
    def __init__(self, inp_size=INPUT_DATA_FEATURES, out_size=OUTPUT_DATA_FEATURES, hid_size=LSTM_HIDDEN_COUNT,
                 layer_count=LSTM_LAYER_COUNT, dropout=LSTM_DROPOUT):
        super(seq2seq, self).__init__()
        self._inp_size = inp_size
        self._hid_size = hid_size
        self._layer_count = layer_count
        self._dropout = dropout
        self._out_size = out_size

        self._enc = encoder_lstm(inp_size=self._inp_size, hid_size=self._hid_size, dropout=self._dropout,
                                 layer_count=self._layer_count)
        self._dec = decoder_lstm(inp_size=self._inp_size, out_size=self._out_size, hid_size=self._hid_size,
                                 drop=self._dropout, layer_count=self._layer_count)

        print_model(self)
        self.model_name = "seq2seq"

    def forward(self, inp):
        outputs = torch.zeros(PREDICT, inp.size(1), self._out_size)
        enc_hidden = self._enc.init_hidden(inp.size(1))

        if SEQ2SEQ_VERBOSE:
            print("============")
            print("seq2seq: input", inp.shape)
            print("seq2seq: outputs", outputs.shape)
            print("seq2seq: hn", enc_hidden[0].shape)

        enc_out, enc_hidden = self._enc(inp)

        if SEQ2SEQ_VERBOSE:
            print("seq2seq: enc_out", enc_out.shape)
            print("seq2seq: enc_hidden", enc_hidden[0].shape)

        dec_inp = inp[-1, :, :]  # get the last element
        dec_hidden = enc_hidden  # enc hidden is the last hidden state of the elements

        if SEQ2SEQ_VERBOSE:
            print("seq2seq: dec_inp", dec_inp.shape)
            print("seq2seq: dec_hidden", dec_hidden[0].shape)

        for p in range(PREDICT):
            final_dec_out, dec_out, dec_hidden = self._dec(dec_inp, dec_hidden)
            if SEQ2SEQ_VERBOSE:
                print(" seq2seq: final_dec_out", final_dec_out.shape)
                print(" seq2seq: dec_out", dec_out.shape)
                print(" seq2seq: dec_hidden hn", dec_hidden[0].shape)
            outputs[p] = final_dec_out
            dec_inp = dec_out
        if SEQ2SEQ_VERBOSE:
            print("seq2seq: outputs", outputs.shape)
        # print(outputs)
        return outputs

    def predict_seq(self, x):
        x = x.unsqueeze(1)  # add in dim of 1
        enc_out, (hn, cn) = self._enc(x)

        outputs = torch.zeros(PREDICT, OUTPUT_DATA_FEATURES)
        dec_inp = x[-1, :, :]  # get the last element
        dec_hidden = (hn, cn)  # enc hidden is the last hidden state of the elements

        for p in range(PREDICT):
            final_dec_out, dec_out, dec_hidden = self._dec(dec_inp, dec_hidden)
            if SEQ2SEQ_VERBOSE:
                print(" seq2seq: final_dec_out", final_dec_out.shape)
                print(" seq2seq: dec_out", dec_out.shape)
                print(" seq2seq: dec_hidden hn", dec_hidden[0].shape)
            outputs[p] = final_dec_out
            dec_inp = dec_out
        if SEQ2SEQ_VERBOSE:
            print("seq2seq: outputs", outputs.shape)
        return outputs


# LSTM MODEL
class lstm(nn.Module):
    def __init__(self, inp_size=LSTM_INPUT_SIZE, out_size=LSTM_OUT_SIZE, hidden_size=LSTM_HIDDEN_COUNT,
                 layer_count=LSTM_LAYER_COUNT, seq_len=LSTM_INPUT_SEQ_LEN, dropout=0):
        super(lstm, self).__init__()
        self.model_name = "lstm"
        self.inp_size = inp_size
        self.out_size = out_size
        self.hid_size = hidden_size
        self.layer_count = layer_count
        self.seq_len = seq_len
        self.drop = dropout

        self.lstm = nn.LSTM(input_size=self.inp_size, hidden_size=self.hid_size, num_layers=self.layer_count,
                            batch_first=True, dropout=self.drop)

        self.fc = nn.Linear(hidden_size, out_size)

        print_model(self)

    def forward(self, inp):

        (hn, cn) = self.init_hidden(inp.size(0))
        if LSTM_VERBOSE:
            print("inp shape:", inp.shape)
            print("hn shape:", hn.shape)
            print("cn shape:", cn.shape)
        # try with (hn,cn)
        lstm_out, hidden = self.lstm(inp, (hn, cn))
        out = self.fc(hn[0]).flatten()
        if LSTM_VERBOSE:
            print("hn[0] shape:", hn[0].shape)
            print("out lstm shape:", out.shape)
        return out

    def init_hidden(self, b_size):
        h_0 = torch.randn(self.layer_count, b_size, self.hid_size).to(DEVICE)
        c_0 = torch.randn(self.layer_count, b_size, self.hid_size).to(DEVICE)
        return (h_0, c_0)
