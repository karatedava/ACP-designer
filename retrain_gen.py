## TODO - retraining of generative model

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.config import OUTPUT_PATH, DEVICE
from src.generators.architectures.architectures import GenRNN
from src.utils.utils_ml import Dataset, Vocabulary, collate_fn_no_label

def train_fn(model,optimizer,trainloader,valloader,n_epoch,vocabulary,bs,outdir:Path,device='cpu', patience=2):

    train_losses = []
    val_losses = []
    val_each = 2

    p = patience
    best_val_loss = float('inf')

    for e in range(1,n_epoch + 1):
        train_batch_loss = []
        for i_batch, sample in enumerate(trainloader):
            sequences = sample[0].to(device)
            lenghts = sample[1].to(device)

            entropy = model.entropy(sequences,lenghts)
            loss = entropy.mean()
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            train_batch_loss.append(loss.item())
        
        train_mean_loss = np.mean(train_batch_loss)
        train_losses.append(train_mean_loss)

        val_loss = []
        with torch.no_grad():
            for i_batch, sample in enumerate(valloader):
                sequences = sample[0].to(device)
                lenghts = sample[1].to(device)

                entropy = model.entropy(sequences,lenghts)
                val_loss += entropy
            
            val_loss_mean = torch.stack(val_loss).mean().item()
            val_losses.append(val_loss_mean)

        if e % val_each == 0:

            print('example gen sequences:\n')
            sampled_seq = model.sample(bs, vocabulary, max_len=20)
            for s in sampled_seq[:3]:
                            print("\t\t{}".format(vocabulary.tensor_to_seq(s, debug=True)))
            print(f'epoch: {e}\t train_loss: {train_losses[-1]}\t val_loss: {val_losses[-1]}\t')

        if val_loss_mean < best_val_loss:
            best_val_loss = val_loss_mean
            p = patience
        else:
            p -= 1
        if p < 0:
            print(f"Early stopping at epoch {e}")
            break

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=range(len(train_losses)), y=train_losses, label="Training loss")
    sns.lineplot(x=range(len(val_losses)), y=val_losses, label="ACP Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Losses")
    plt.legend()
    plt.savefig(outdir / 'train_loss.png')

    torch.save(model, outdir / 'rtr_gen.pt')

def main(args):
    
    training_folder = args.training_folder
    output_folder = args.output_folder
    device = args.device
    bs = args.batch_size

    output_folder.mkdir(parents=True, exist_ok=True)

    ### DATA PREPROCESSING ###

    # load train, test, val datasets
    print('loading training data')
    train = pd.read_csv(training_folder / 'train_set.csv',index_col=False)
    test = pd.read_csv(training_folder / 'test_set.csv',index_col=False)
    val = pd.read_csv(training_folder / 'validation_set.csv',index_col=False)

    all_seqs = pd.concat([train,test,val])['sequence'].to_numpy()
    vocabulary = Vocabulary.get_vocabulary_from_sequences(all_seqs)

    train_dataset = Dataset(train,vocabulary,with_label=False)
    test_dataset = Dataset(test,vocabulary,with_label=False)
    val_dataset = Dataset(val,vocabulary,with_label=False)

    train_loader = DataLoader(train_dataset,batch_size=bs,shuffle=False,collate_fn=collate_fn_no_label,drop_last=False)
    test_loader = DataLoader(test_dataset,batch_size=bs,shuffle=False,collate_fn=collate_fn_no_label,drop_last=False)
    val_loader = DataLoader(val_dataset,batch_size=bs,shuffle=False,collate_fn=collate_fn_no_label,drop_last=False)

    model = GenRNN(
        input_size=len(vocabulary.vocabulary),
        embedding_size=100,
        hidden_size=400,
        num_layers=1
    ).to(device)

    ### training ###

    optimizer = torch.optim.Adam(model.parameters(),lr=0.0005,weight_decay=1e-3)

    train_fn(
        model=model,
        optimizer=optimizer,
        trainloader=train_loader,
        valloader=val_loader,
        n_epoch=20,
        device=device,
        bs=bs,
        vocabulary=vocabulary,
        outdir=output_folder
    )

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--training_folder',
        type=Path,
        help="path to folder with train, validation and test set"
    )

    parser.add_argument(
        '--output_folder',
        type=Path,
        default= OUTPUT_PATH / 'GEN_RETRAINING',
        help="where to save newly trained model (.pkl format) and all metadata"
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'mps'],
        default=DEVICE,
        help="device for computation"
    )

    parser.add_argument(
        '--batch_size',
        default=12,
        help="batch size for training"
    )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
    main(args)