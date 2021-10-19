# performs conversion of predicted coordinates for further visualization with current model
import argparse
import torch
import torch.utils.data
import os
# load my own stuff
from modules.data import *


parser = argparse.ArgumentParser(description='Protein-peptide docking. Reconstruct coordinates for peptides from visualization set.')
parser.add_argument('--path-to-model', type=str, default='model_ppd.pt',
                    help='input path to model (default: model_ppd.pt)')
parser.add_argument('--offset', action='store_true',
                    help='introduce offset (zero-padding before the data) for input data')

args = parser.parse_args()

kwargs = {}

max_number_atoms = [300, 4000]

if not args.offset:
    # meaningful data first, then zero-padding
    offset_dir = 'offset_n'
else:
    # zero-padding first, then meaningful data
    offset_dir = 'offset_y'

path_to_model = offset_dir + '/' + args.path_to_model
path_to_dictionary = 'data/Dictionary'
path_to_outputs = 'data/outputs'

if not os.path.isdir(path_to_outputs):
    os.makedirs(path_to_outputs)

if os.path.isfile(path_to_model):
    model = torch.load(path_to_model)

    dict_forward, dict_backward = make_dicts(path_to_dictionary)
    dict_size = len(dict_forward)

    prefix = 'data/inputs/'
    basic_filenames = ['sample_list',
                       'peptide_adj',
                       'peptide_coords',
                       'peptide_onehot',
                       'receptor_coords',
                       'receptor_onehot']

    print('Loading visualization dataset:')
    filehandlers = []
    for filename in basic_filenames:
        filehandlers = append_fh(filehandlers, prefix + 'visualization_' + filename, 'r')
    visualization_dataset = MyDataset(fhs=filehandlers, atom_dict_size=dict_size, max_n_atoms=max_number_atoms, augment=False, offset=args.offset)
    visualization_loader = torch.utils.data.DataLoader(visualization_dataset, batch_size=1, shuffle=False, **kwargs)
    for fh in filehandlers:
        fh.close()
    del filehandlers

    print('Predicting peptide docking with the model  ' + path_to_model + ' :')
    model.eval()
    count = 0
    for (data_list, data_in, _) in visualization_loader:
        with torch.no_grad():
            data_out = model(data_in)
        for (sample_name, d_in, d_out) in zip(data_list, data_in[1], data_out):
            pdb_fh = open(path_to_outputs + '/' + offset_dir + '_' + sample_name + '.pdb', 'w')
            for i, (line_in, line_out) in enumerate(zip(d_in, d_out)):
                atom_name = dict_backward[str(torch.nonzero(line_in, as_tuple=False)[0].item())]
                if atom_name != '0':
                    x = line_out[0].item()
                    y = line_out[1].item()
                    z = line_out[2].item()
                    pdb_fh.write('ATOM  {:>5d}  {:<4s}  {:>19.3f}{:>8.3f}{:>8.3f}\n'.format(i+1, atom_name, x, y, z))
            pdb_fh.close()
        count += 1
        print('  [' + str(count) + ']', end = '\r')
    print('  [' + str(count) + ']')
    print('  Done!')
else:
    print('Error: Model not found, check the path ', path_to_model)
