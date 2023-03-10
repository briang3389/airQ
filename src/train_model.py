import torch.optim as optimizer
from arch import *
from constants import *
from data_prep import *
import mlflow

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

# with open(TRAIN_METRIC_JSON,"w") as f:
#   f.write("training metrics")

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
        #print(x.shape)
        if model_name == SEQ2SEQ_MODEL_NAME:
            x = x.swapaxes(0, 1)
            y = y.swapaxes(0, 1)
            #print("swaping axis!")
        #print(x.shape)
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


train_dl, valid_dl = get_data_loaders()
best_val_loss = 1000000

f = open("model.csv", "w")
f.write("hello,world")
f.close

f = open("params.yml", "w")
f.write("hello")
f.close
mlflow.set_tracking_uri("/content/airQ/mlruns")
with mlflow.start_run():
  mlflow.log_params(data_processing_dict)
  for e in range(n_epochs):
      

      avg_train_loss = train_epoch(train_dl, e)
      avg_valid_loss = train_epoch(valid_dl, e)

      mlflow.log_metric("avg train loss",avg_train_loss, step=e)
      mlflow.log_metric("avg validation loss",avg_valid_loss, step=e)
      num_epochs_run += 1
      train_loss.append(avg_train_loss)
      valid_loss.append(avg_valid_loss)
      print(f"epoch {e}: avg train loss: {avg_train_loss} avg val loss: {avg_valid_loss}")

      if avg_valid_loss < best_val_loss:
          best_val_loss = avg_valid_loss
          torch.save(model, MODEL_PATH)

