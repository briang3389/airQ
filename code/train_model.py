import torch.optim as optimizer
from arch import *
from constants import *
from data_prep import *

train_loss = []  # track training loss
valid_loss = []  # track validation loss
learning_rate = LEARNING_RATE  # the learning rate
n_epochs = N_EPOCHS  # number of epochs

# model = lstm().to(device)
model = seq2seq().to(DEVICE)
model_name = model.model_name

print(model_name)
loss_func = nn.MSELoss()
optim = optimizer.Adam(model.parameters(), lr=learning_rate)

num_epochs_run = 0


def train_epoch(dl, epoch):
    print_once = True
    model.train(True)

    epoch_train_loss = 0.
    # loop over training batches
    times_run = 0

    for i, (x, y) in enumerate(dl):
        optim.zero_grad()  # zero gradients
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        if model_name == "seq2seq":
            x = x.swapaxes(0, 1)
            y = y.swapaxes(0, 1)
        model_out = model.forward(x)

        # squeeze the tensors to account for 1 dim sizes
        model_out = model_out.squeeze()
        y = y.squeeze()

        loss = loss_func(model_out, y)
        epoch_train_loss += loss.item() * x.size(0)

        times_run += x.size(0)

        # compute the loss
        loss.backward()
        # step the optimizer
        optim.step()

    return epoch_train_loss / times_run


def test_epoch(dl, epoch):
    model.train(False)
    epoch_test_loss = 0.
    times_run = 0
    # loop over testing batches
    for i, (x, y) in enumerate(dl):
        model_out = model(x)
        # squeeze tensors to account for 1 dim sizes
        model_out = model_out.squeeze()
        y = y.squeeze()

        loss = loss_func(model_out, y)
        epoch_test_loss += loss.item() * x.size(0)
        times_run += x.size(0)

    return epoch_test_loss / times_run


for e in range(n_epochs):
    train_dl, valid_dl = get_data_loaders()
    avg_train_loss = train_epoch(train_dl, e)
    avg_valid_epoch = train_epoch(valid_dl, e)
    num_epochs_run += 1
    train_loss.append(avg_train_loss)
    valid_loss.append(avg_valid_epoch)
    print(f"epoch {e}: avg train loss: {avg_train_loss} avg val loss: {avg_valid_epoch}")
