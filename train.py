import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import mlflow
import mlflow.pytorch
import pandas as pd
from model import EpigeneModel
from dataloader import EpigeneDataset, split_dataset, load_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args, model, loss_fn, optimizer, train_loader):
    for epoch in range(args.epochs):
        train_loop(train_loader, model, loss_fn, optimizer, device, epoch)


def inst_model(args, device):
    model = EpigeneModel().to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(args.beta_one, args.beta_two), eps=args.epsilon, weight_decay=args.weight_decay)
    #lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=3e-4

    return model, loss_fn, optimizer


def train_loop(train_loader, model, loss_fn, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch, (inpt, label) in enumerate(train_loader):
        inpt = inpt.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        output = model(inpt)
        loss = loss_fn(output, label)

        # backpropagation
        loss.backward()
        optimizer.step()

        # calculate metric for hyperparameter tunning
        total_loss += loss.item()
        n_batches += 1

    avg_mse = total_loss/n_batches
    mlflow.log_metric("MSE", avg_mse, step = epoch)


def parse_args():
    parser = argparse.ArgumentParser()

    # arguments
    parser.add_argument("--training_data", dest='training_data', type=str)
    parser.add_argument("--learning_rate", dest="learning_rate", type=float)
    parser.add_argument("--beta_one", dest="beta_one", type=float)
    parser.add_argument("--beta_two", dest="beta_two", type=float)
    parser.add_argument("--epsilon", dest="epsilon", type=float)
    parser.add_argument("--weight_decay", dest="weight_decay", type=float)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=128)
    parser.add_argument("--epochs", dest="epochs", type=int)
    parser.add_argument("--testset_n", dest="testset_n", type=int)

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    # parse args
    args = parse_args()

    # track the training run
    mlflow.pytorch.autolog()

    # pass data
    dataset = EpigeneDataset(args.training_data)
    trainset, testset = split_dataset(dataset, args.testset_n)

    train_loader = load_data(trainset, args.batch_size)

    # instantiate model
    model, loss_fn, optimizer = inst_model(args, device)

    # run main function
    main(args, model, loss_fn, optimizer, train_loader)

    mlflow.pytorch.log_model(model, artifact_path="model")