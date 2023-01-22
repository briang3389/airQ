# Air IQ

## Authors:

Alexiy Buynitsky & Brian Gan

## About:

Air-quality prediction and analysis framework using LSTMs and seq2seqs (transformer-based) models integrated in `DagsHub` using `MLflow` `git` and `DVC`.

## Project Layout:

**Data:**

- Raw xlsx files for data_preprocessing
- Prepared .npy arrays using for training and validation sets for training script

**mlruns**

- information about experiments, including model parameters and training and validation losses

**src**

- Contains following model files:
    - `arch.ph`: seq2seq and lstm model implementation 
    - `constants.py`: Control hyperparameters for data feature extraction and training (ie epochs, input features)
    - `data_prep.py`: Extracts testing and training data, placing it in Dataloaders and Datasets
    - `featurization.py`: Generates test and train .npy arrays, normalizes data, interpolates data
    - `train_model.py`: main training script, contions training loop, optimizer, scheduler
- Contains following notebooks:
    - `main.ipynb`: Google Collab demo of all model files (no dvc/MLflow/git)
    - `dagshub_config.ipynb`: Config Notesbook to run model through DVC/Mlflow/git!
    - `dagshub_run.ipynb`: Demo File for running and training model using DagsHub!

## Configuration

Open `src/main.ipynb` to see a complete run in Collab!

Open `src/dagshub_run.ipynb` and run in Collab!

Make sure you have a Dagshub account and copy your MLFlow link (where you would copy github URL)


