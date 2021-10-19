# trains the model for the peptide pose predictor

import argparse
import torch
import torch.utils.data
from torch import optim
import torch.optim.lr_scheduler
import os
import shutil
# load my own stuff
from modules.data import *
from modules.regressor import *
from modules.modeling import *


parser = argparse.ArgumentParser(description='Protein-peptide docking. Optimizie peptide coords.')
parser.add_argument('--batch-size', type=int, default=250,
                    help='input batch size for training (default: 250)')
parser.add_argument('--epochs', type=int, default=0,
                    help='number of epochs to train (default: 0)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=5,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model-every', type=int, default=2,
                    help='how many epochs to wait before saving trained model (default: 2)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate (default: 1e-3)')
parser.add_argument('--offset', action='store_true',
                    help='introduce offset (zero-padding before the data) for input data')

args = parser.parse_args()

torch.manual_seed(args.seed)

device = torch.device("cpu")

kwargs = {}

max_number_atoms = [300, 4000]

if not args.offset:
    # meaningful data first, then zero-padding
    offset_dir = 'offset_n'
else:
    # zero-padding first, then meaningful data
    offset_dir = 'offset_y'

if not os.path.isdir(offset_dir):
    os.makedirs(offset_dir)

path_to_model = offset_dir + '/model_ppd.pt'
path_to_train_log = offset_dir + '/train.log'

path_to_optimal_train_model = offset_dir + '/model_ppd_optimal_train.pt'
path_to_optimal_train_log = offset_dir + '/optimal_train.log'

path_to_optimal_test_model = offset_dir + '/model_ppd_optimal_test.pt'
path_to_optimal_test_log = offset_dir + '/optimal_test.log'

path_to_dictionary = 'data/Dictionary'

dict_forward, dict_backward = make_dicts(path_to_dictionary)
dict_size = len(dict_forward)

prefix = 'data/inputs/'
basic_filenames = ['sample_list',
                   'peptide_adj',
                   'peptide_coords',
                   'peptide_onehot',
                   'receptor_coords',
                   'receptor_onehot']

print('Loading train dataset:')
filehandlers = []
for filename in basic_filenames:
    filehandlers = append_fh(filehandlers, prefix + 'train_' + filename, 'r')
train_dataset = MyDataset(fhs=filehandlers, atom_dict_size=dict_size, max_n_atoms=max_number_atoms, augment=True, offset=args.offset)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
for fh in filehandlers:
    fh.close()
del filehandlers

print('Loading test dataset:')
filehandlers = []
for filename in basic_filenames:
    filehandlers = append_fh(filehandlers, prefix + 'test_' + filename, 'r')
test_dataset = MyDataset(fhs=filehandlers, atom_dict_size=dict_size, max_n_atoms=max_number_atoms, augment=False, offset=args.offset)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
for fh in filehandlers:
    fh.close()
del filehandlers

if os.path.isfile(path_to_model):
    model = torch.load(path_to_model)
else:
    model = regressor().to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch, model, optimizer, device, args.log_interval, train_loader)
    test_loss = test(epoch, model, device, test_loader)

    model_saved = False
    fh = open(path_to_train_log, 'a')
    fh.write(str(train_loss) + '\t' + str(test_loss) + '\n')
    fh.close()
    if epoch % args.save_model_every == 0 or epoch == args.epochs:
        torch.save(model, path_to_model)
        model_saved_path = path_to_model
        model_saved = True
        print('model saved to [ ' + path_to_model + ' ]\n')

    # assessing optimal lossess
    if epoch == 1:
        if os.path.isfile(path_to_optimal_train_log):
            fh = open(path_to_optimal_train_log, 'r')
            train_loss_optimal = float(fh.readline())
            fh.close()
        else:
            train_loss_optimal = train_loss

        if os.path.isfile(path_to_optimal_test_log):
            fh = open(path_to_optimal_test_log, 'r')
            _ = fh.readline()
            test_loss_optimal = float(fh.readline())
            fh.close()
        else:
            test_loss_optimal = test_loss

    # saving model with optimal train loss
    if train_loss < train_loss_optimal:
        if model_saved:
            shutil.copyfile(model_saved_path, path_to_optimal_train_model)
        else:
            torch.save(model, path_to_optimal_train_model)
            model_saved_path = path_to_optimal_train_model
            model_saved = True
        fh = open(path_to_optimal_train_log, 'w')
        fh.write(str(train_loss) + '\n' + str(test_loss) + '\n')
        fh.close()
        train_loss_optimal = train_loss
        print('model with optimal train loss saved to [ ' + path_to_optimal_train_model + ' ]\n')

    # saving model with optimal test loss
    if test_loss < test_loss_optimal:
        if model_saved:
            shutil.copyfile(model_saved_path, path_to_optimal_test_model)
        else:
            torch.save(model, path_to_optimal_test_model)
            model_saved_path = path_to_optimal_test_model
            model_saved = True
        fh = open(path_to_optimal_test_log, 'w')
        fh.write(str(train_loss) + '\n' + str(test_loss) + '\n')
        fh.close()
        test_loss_optimal = test_loss
        print('model with optimal test loss saved to [ ' + path_to_optimal_test_model + ' ]\n')

    scheduler.step()
