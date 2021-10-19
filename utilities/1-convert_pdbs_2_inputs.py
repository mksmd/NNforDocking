import numpy as np
import math
import os
import random
import re


def load_dict(filename, atom_dictionary):
    dict_fh = open(filename, 'r')
    for line in dict_fh:
        atom, index = line.strip().split('\t')
        atom_dictionary[atom] = index
    dict_fh.close()
    return atom_dictionary


def count_atoms(filename):
    counter = 0
    fh = open(filename, 'r')
    for line in fh:
        record_type = line[0:6].strip()
        if record_type == 'ATOM':
            counter += 1
    fh.close()
    return counter


def distance(v1, v2):
    dist = 0.0
    for a, b in zip(v1, v2):
        dist += math.pow(a - b, 2)
    return math.sqrt(dist)


def s2f(s):
    f = round(float(s), 2)
    return f


def parse_pdb(filename, atom_dictionary, clear):
    atoms_names = []
    residue_ids = []
    atoms_coords = []
    fh = open(filename, 'r')
    for line in fh:
        record_type = line[0:6].strip()
        if record_type == 'ATOM':
            atom_name = line[12:16].strip()
            residue_id = re.sub('\s+', '-', line[17:27].strip())
            if atom_name not in atom_dictionary:
                clear = False
            if clear:
                atoms_names.append(atom_dictionary[atom_name])
                residue_ids.append(residue_id)
                atoms_coords.append([s2f(line[30:38]), s2f(line[38:46]), s2f(line[46:54])])
    fh.close()
    return atoms_names, residue_ids, atoms_coords, clear


def proc_dir(prefix, max_p_atoms, max_r_atoms, atom_dictionary, filehandlers):
    clear = True

    p_names = []
    p_coords = []
    p_filename = prefix + '/peptide1.pdb'
    number_p_atoms = count_atoms(p_filename)
    if number_p_atoms > max_p_atoms:
        clear = False
    else:
        p_names, p_residues, p_coords, clear = parse_pdb(p_filename, atom_dictionary, clear)

        r_names = []
        r_coords = []
        r_filename = prefix + '/receptor1.pdb'
        number_r_atoms = count_atoms(r_filename)
        if number_r_atoms > max_r_atoms:
            clear = False
        else:
            r_names, r_residues, r_coords, clear = parse_pdb(r_filename, atom_dictionary, clear)

    if clear:
        _, _, _, p = prefix.split('/')

        sample_list_fh = filehandlers[0]
        p_adj_fh =       filehandlers[1]
        p_coords_fh =    filehandlers[2]
        p_onehot_fh =    filehandlers[3]
        r_coords_fh =    filehandlers[4]
        r_onehot_fh =    filehandlers[5]

        sample_list_fh.write(p + ',' + str(number_p_atoms) + ',' + str(number_r_atoms) + '\n')
        p_adj_fh.write(p + '\n')
        p_coords_fh.write(p + '\n')
        p_onehot_fh.write(p + '\n')
        r_coords_fh.write(p + '\n')
        r_onehot_fh.write(p + '\n')

        adj_matr = np.zeros((number_p_atoms, number_p_atoms))
        for i in range(number_p_atoms - 1):
            for j in range(i + 1, number_p_atoms):
                d = round(distance(p_coords[i], p_coords[j]), 2)
                if d <= 2.1 or p_residues[i] == p_residues[j]:
                    adj_matr[i][j] = d
                    adj_matr[j][i] = d
        for i in range(number_p_atoms):
            adj = ''
            for j in range(number_p_atoms):
                if adj_matr[i][j] != 0.0:
                    adj += str(j) + ',' + str(adj_matr[i][j]) + ';'
            p_adj_fh.write(adj.strip(';') + '\n')

        oh = ''
        for name, coord in zip(p_names, p_coords):
            p_coords_fh.write(str(coord[0]) + ',' + str(coord[1]) + ',' + str(coord[2]) + '\n')
            oh += name + ','
        p_onehot_fh.write(oh.strip(',') + '\n')

        oh = ''
        for name, coord in zip(r_names, r_coords):
            r_coords_fh.write(str(coord[0]) + ',' + str(coord[1]) + ',' + str(coord[2]) + '\n')
            oh += name + ','
        r_onehot_fh.write(oh.strip(',') + '\n')


def append_fh(fhs, filename, mode):
    fh = open(filename, mode)
    fhs.append(fh)
    return fhs


# main starts here

max_p_atoms = 300
max_r_atoms = 4000

atom_dictionary = {}
atom_dictionary = load_dict('../data/Dictionary', atom_dictionary)

prefix = '../data/inputs/'
basic_filenames = ['sample_list',
                   'peptide_adj',
                   'peptide_coords',
                   'peptide_onehot',
                   'receptor_coords',
                   'receptor_onehot']

train_filehandlers = []
for filename in basic_filenames:
    train_filehandlers = append_fh(train_filehandlers, prefix + 'train_' + filename, 'w')

test_filehandlers = []
for filename in basic_filenames:
    test_filehandlers = append_fh(test_filehandlers, prefix + 'test_' + filename, 'w')

count = 0
for root, dirs, _ in os.walk('../data/pepbdb/'):
    for dir in dirs:
        count += 1
        prefix = root + dir
        if random.random() > 0.15:
            proc_dir(prefix, max_p_atoms, max_r_atoms, atom_dictionary, train_filehandlers)
        else:
            proc_dir(prefix, max_p_atoms, max_r_atoms, atom_dictionary, test_filehandlers)
        if count % 100 == 0:
            print(count)

for fh in train_filehandlers:
    fh.close()
for fh in test_filehandlers:
    fh.close()
