import math
import os


def sigmoid(x):
  return 1 / (1 + math.exp(-20.0 * (x - 0.5)))


def get_coords(string):
    x = string[0:8].strip()
    y = string[8:16].strip()
    z = string[16:].strip()
    return float(x), float(y), float(z)


def merge_coords(coords1, coords2, f):
    x1, y1, z1 = get_coords(coords1)
    x2, y2, z2 = get_coords(coords2)
    x = (1.0 - f) * x1 + f * x2
    y = (1.0 - f) * y1 + f * y2
    z = (1.0 - f) * z1 + f * z2
    return x, y, z

pdb_prefix = 'data/outputs'
sample_list_fh = open('data/inputs/visualization_sample_list', 'r')

print('Merging  offest_n_*  and  offset_y_*  predictions:')
count = 0
for line in sample_list_fh:
    sample, p_atoms, r_atoms = line.strip().split(',')
    pdb0 = 'data/pepbdb/' + sample + '/peptide1.pdb'
    pdb1 = 'offset_n_' + sample + '.pdb'
    pdb2 = 'offset_y_' + sample + '.pdb'
    pdb3 = sample + '.pdb'

    if os.path.isfile(pdb0):
        fh0 = open(pdb0, 'r')
        in_pepbdb = True
    else:
        in_pepbdb = False

    fh1 = open(pdb_prefix + '/' + pdb1, 'r')
    fh2 = open(pdb_prefix + '/' + pdb2, 'r')
    fh3 = open(pdb_prefix + '/' + pdb3, 'w')

    for i_atom, (line1, line2) in enumerate(zip(fh1, fh2)):
        if in_pepbdb:
            residue = fh0.readline()[17:26]
        else:
            residue = '         '

        f = sigmoid((i_atom + 1) / int(p_atoms))
        atom = line1[0:17]
        coords1 = line1.strip()[30:]
        coords2 = line2.strip()[30:]
        x,y,z = merge_coords(coords1, coords2, f)
        fh3.write('{:<17s}{:<13s}{:>8.3f}{:>8.3f}{:>8.3f}\n'.format(atom, residue, x, y, z))

    fh1.close()
    fh2.close()
    fh3.close()
    count += 1
    print('  [' + str(count) + ']', end = '\r')
print('  [' + str(count) + ']')

sample_list_fh.close()
print('  Done!')
