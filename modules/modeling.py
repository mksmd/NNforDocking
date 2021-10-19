import torch
from torch.nn import functional as F
from math import pow, sqrt


def distance(v1, v2):
    dist = 0.0
    for a, b in zip(v1, v2):
        dist += pow(a - b, 2)
    return sqrt(dist)


def loss_function(recon_x, x):
    return F.mse_loss(recon_x, x, reduction='sum')


def train(epoch, model, optimizer, device, log_interval, train_loader):
    width = len(str(len(train_loader.dataset)))
    model.train()
    train_loss = 0
    for batch_idx, (_, data_in, data_out) in enumerate(train_loader):
        data_in[0] = data_in[0].to(device)
        data_in[1] = data_in[1].to(device)
        data_in[2] = data_in[2].to(device)
        data_in[3] = data_in[3].to(device)
        optimizer.zero_grad()
        recon_batch = model(data_in)
        loss = loss_function(recon_batch, data_out)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx > 0 and batch_idx % log_interval == 0:
            print('Train Epoch: {} [{:>{w}}/{} {:>3.0f}%]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data_out), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data_out), w = width))
    train_loss /= len(train_loader.dataset)
    print('====> Epoch: {} Average loss on train set: {:.6f}'.format(epoch, train_loss))
    return train_loss


def test(epoch, model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (_, data_in, data_out) in enumerate(test_loader):
            data_in[0] = data_in[0].to(device)
            data_in[1] = data_in[1].to(device)
            data_in[2] = data_in[2].to(device)
            data_in[3] = data_in[3].to(device)
            recon_batch = model(data_in)
            loss = loss_function(recon_batch, data_out)
            test_loss += loss.item()
    test_loss /= len(test_loader.dataset)
    print('====> Epoch: {} Average loss on test set:  {:.6f}\n'.format(epoch, test_loss))
    return test_loss
