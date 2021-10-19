import torch
import torch.utils.data
import re
import random
from math import cos, sin, radians

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, fhs, atom_dict_size, max_n_atoms, augment, offset):

        sample_list_fh = fhs[0]
        p_adj_fh =       fhs[1]
        p_coords_fh =    fhs[2]
        p_onehot_fh =    fhs[3]
        r_coords_fh =    fhs[4]
        r_onehot_fh =    fhs[5]

        sample_list = []
        number_of_p_atoms = []
        number_of_r_atoms = []
        for line in sample_list_fh:
            sample, p_atoms, r_atoms = line.strip().split(',')
            sample_list.append(sample)
            number_of_p_atoms.append(int(p_atoms))
            number_of_r_atoms.append(int(r_atoms))

        length = len(sample_list)

        sample_list_augm = []
        max_p_atoms = max_n_atoms[0]
        max_r_atoms = max_n_atoms[1]

        samples_in = []
        samples_out = []
        count = 0

        for sample, p_atoms, r_atoms in zip(sample_list, number_of_p_atoms, number_of_r_atoms):
            if not offset:
                # meaningful data first, then zero-padding
                p_offset = 0
                r_offset = 0
            else:
                # zero-padding first, then meaningful data
                p_offset = max_p_atoms - p_atoms
                r_offset = max_r_atoms - r_atoms

            p_adj = torch.zeros(max_p_atoms, max_p_atoms, dtype=torch.float32)
            p_coords = torch.zeros(max_p_atoms, 3, dtype=torch.float32)
            p_onehot = torch.zeros(max_p_atoms, atom_dict_size, dtype=torch.float32)
            r_coords = torch.zeros(max_r_atoms, 3, dtype=torch.float32)
            r_onehot = torch.zeros(max_r_atoms, atom_dict_size, dtype=torch.float32)

            p_adj =    load_file_to_tensor0(p_adj_fh, p_adj, p_atoms, p_offset)
            p_coords = load_file_to_tensor1(p_coords_fh, p_coords, p_atoms, p_offset)
            p_onehot = load_file_to_tensor2(p_onehot_fh, p_onehot, p_offset)
            r_coords = load_file_to_tensor1(r_coords_fh, r_coords, r_atoms, r_offset)
            r_onehot = load_file_to_tensor2(r_onehot_fh, r_onehot, r_offset)

            for count0 in range(p_offset):
                p_onehot[count0][0] = 1
            for count0 in range(p_offset + p_atoms, max_p_atoms):
                p_onehot[count0][0] = 1

            for count0 in range(r_offset):
                r_onehot[count0][0] = 1
            for count0 in range(r_offset + r_atoms, max_r_atoms):
                r_onehot[count0][0] = 1

            # append original data
            samples_in.append([p_adj, p_onehot, r_coords, r_onehot])
            samples_out.append(p_coords)
            sample_list_augm.append(sample)
            count += 1

            if augment:

                # augmentation by mirroring coordinates (r = -r)
                samples_in.append([p_adj, p_onehot, -r_coords, r_onehot])
                samples_out.append(-p_coords)
                sample_list_augm.append(sample + '_0')

                # augmentation by a random coordinate shift
                shift_vector = get_shift_vector()
                samples_in.append([p_adj, p_onehot, r_coords + shift_vector, r_onehot])
                samples_out.append(p_coords + shift_vector)
                sample_list_augm.append(sample + '_1')

                # augmentation by a rotation over X axis
                angle = random.random() * 90.0
                cos_a = cos(radians(angle))
                sin_a = sin(radians(angle))
                r_coords_rot = rotate_coords_x(r_coords, cos_a, sin_a)
                p_coords_rot = rotate_coords_x(p_coords, cos_a, sin_a)
                samples_in.append([p_adj, p_onehot, r_coords_rot, r_onehot])
                samples_out.append(p_coords_rot)
                sample_list_augm.append(sample + '_2')

                # augmentation by a rotation over Y axis
                angle = random.random() * 90.0
                cos_a = cos(radians(angle))
                sin_a = sin(radians(angle))
                r_coords_rot = rotate_coords_y(r_coords, cos_a, sin_a)
                p_coords_rot = rotate_coords_y(p_coords, cos_a, sin_a)
                samples_in.append([p_adj, p_onehot, r_coords_rot, r_onehot])
                samples_out.append(p_coords_rot)
                sample_list_augm.append(sample + '_3')

                # augmentation by a rotation over Z axis
                angle = random.random() * 90.0
                cos_a = cos(radians(angle))
                sin_a = sin(radians(angle))
                r_coords_rot = rotate_coords_z(r_coords, cos_a, sin_a)
                p_coords_rot = rotate_coords_z(p_coords, cos_a, sin_a)
                samples_in.append([p_adj, p_onehot, r_coords_rot, r_onehot])
                samples_out.append(p_coords_rot)
                sample_list_augm.append(sample + '_4')

            print('  [' + str(count) + ' out of ' + str(length) + ']', end = '\r')

            del p_adj, p_coords, p_onehot, r_coords, r_onehot

        print('  [original samples  - ' + str(count) + ']')
        print('  [augmented samples - ' + str(len(sample_list_augm) - count) + ']')

        self.sample_list = sample_list_augm
        self.samples_in = samples_in
        self.samples_out = samples_out
        self.length = len(sample_list_augm)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.sample_list[index], self.samples_in[index], self.samples_out[index]


def load_file_to_tensor0(fh, input_tensor, n, offset):
    sample_name = fh.readline()
    count0 = 0
    for _ in range(n):
        pairs = fh.readline().strip().split(';')
        for pair in pairs:
            if pair != '':
                count1, item = pair.split(',')
                input_tensor[offset + count0][offset + int(count1)] = float(item)
        count0 += 1
    return input_tensor


def load_file_to_tensor1(fh, input_tensor, n, offset):
    sample_name = fh.readline()
    count0 = 0
    for _ in range(n):
        items = fh.readline().strip().split(',')
        count1 = 0
        for item in items:
            input_tensor[offset + count0][count1] = float(item)
            count1 += 1
        count0 += 1
    return input_tensor


def load_file_to_tensor2(fh, input_tensor, offset):
    sample_name = fh.readline()
    list = fh.readline().split(',')
    count0 = 0
    for count1 in list:
        input_tensor[offset + count0][int(count1)] = 1
        count0 += 1
    return input_tensor


def make_dicts(path_to_dictionary):
    dict_f = {}
    dict_b = {}
    file0 = open(path_to_dictionary, 'r')
    for line in file0:
        pair = re.split('\s+', line.strip())
        dict_f[pair[0]] = pair[1]
        dict_b[pair[1]] = pair[0]
    file0.close()
    return dict_f, dict_b


def append_fh(fhs, filename, mode):
    fh = open(filename, mode)
    fhs.append(fh)
    return fhs


def get_random():
    import random
    return random.random() * 10.0 - 5.0


def get_shift_vector():
    shift_vector = torch.tensor([get_random(), get_random(), get_random()])
    return shift_vector


def rotate_coords_x(coords, cos_a, sin_a):
    coords_rotated = torch.zeros_like(coords)
    for i, triplet in enumerate(coords):
        coords_rotated[i][0] = triplet[0]
        coords_rotated[i][1] = triplet[1] * cos_a - triplet[2] * sin_a
        coords_rotated[i][2] = triplet[1] * sin_a + triplet[2] * cos_a
    return coords_rotated


def rotate_coords_y(coords, cos_a, sin_a):
    coords_rotated = torch.zeros_like(coords)
    for i, triplet in enumerate(coords):
        coords_rotated[i][0] = triplet[0] * cos_a - triplet[2] * sin_a
        coords_rotated[i][1] = triplet[1]
        coords_rotated[i][2] = triplet[0] * sin_a + triplet[2] * cos_a
    return coords_rotated


def rotate_coords_z(coords, cos_a, sin_a):
    coords_rotated = torch.zeros_like(coords)
    for i, triplet in enumerate(coords):
        coords_rotated[i][0] = triplet[0] * cos_a - triplet[1] * sin_a
        coords_rotated[i][1] = triplet[0] * sin_a + triplet[1] * cos_a
        coords_rotated[i][2] = triplet[2]
    return coords_rotated
