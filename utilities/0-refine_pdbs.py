import os
import re


def s2f(s):
    f = round(float(s), 2)
    return f


def get_pdb_center(fh0):
    xcenter = 0.0
    ycenter = 0.0
    zcenter = 0.0
    ncenter = 0
    f0 = open(fh0, 'r')
    for line in f0:
        line = line.strip()
        record_type = line[0:6].strip()
        if record_type == 'ATOM' or record_type == 'HETATM':
            x = s2f(line[30:38])
            y = s2f(line[38:46])
            z = s2f(line[46:54])
            s3 = line[77:79]
            s3 = re.sub('\d', '', s3)
            if s3.strip() != 'H':
                xcenter += x
                ycenter += y
                zcenter += z
                ncenter += 1
    f0.close()
    if ncenter > 0:
        xcenter /= ncenter
        ycenter /= ncenter
        zcenter /= ncenter
    return (xcenter, ycenter, zcenter)


def edit_pdb(fh0, fh1, center):
    f0 = open(fh0, 'r')
    f1 = open(fh1, 'w')
    for line in f0:
        line = line.strip()
        record_type = line[0:6].strip()
        if record_type == 'ATOM' or record_type == 'HETATM':
            s0 = line[0:13]
            s0 = re.sub('HETATM', 'ATOM  ', s0)
            s1 = line[13:17]
            s2 = line[17:30]
            x = s2f(line[30:38]) - center[0]
            y = s2f(line[38:46]) - center[1]
            z = s2f(line[46:54]) - center[2]
            s4 = line[54:77]
            s3 = line[77:79]
            s3 = re.sub('\d', '', s3)
            for i in range(4 - len(s3)):
                s3 = s3 + ' '
            line = s0 + s3 + s2 + s3
            if s3.strip() != 'H':
                f1.write('{:<13s}{:<4s}{:<13s}{:>8.3f}{:>8.3f}{:>8.3f}{:<23s}{:<4s}\n'.format(s0, s3, s2, x, y, z, s4, s3))
    f0.close()
    f1.close()


count = 0
for root, dirs, _ in os.walk('../data/pepbdb/'):
    for dir in dirs:
        count += 1
        prefix = root + dir
        center = get_pdb_center(prefix + '/receptor.pdb')
        edit_pdb(prefix + '/peptide.pdb', prefix + '/peptide1.pdb', center)
        edit_pdb(prefix + '/receptor.pdb', prefix + '/receptor1.pdb', center)
        if count % 100 == 0:
            print(count)
