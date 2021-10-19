def copy_lines(fh0, fh1, number_of_lines):
    for j in range(number_of_lines):
        line = fh0.readline()
        fh1.write(line)


number_of_train_samples = 20
number_of_test_samples = 20

data_prefix='../data/inputs/'

fh0_0 = open(data_prefix + 'train_sample_list', 'r')
fh0_1 = open(data_prefix + 'test_sample_list', 'r')
fh0_2 = open(data_prefix + 'visualization_sample_list', 'w')

fh1_0 = open(data_prefix + 'train_peptide_adj', 'r')
fh1_1 = open(data_prefix + 'test_peptide_adj', 'r')
fh1_2 = open(data_prefix + 'visualization_peptide_adj', 'w')

fh2_0 = open(data_prefix + 'train_peptide_coords', 'r')
fh2_1 = open(data_prefix + 'test_peptide_coords', 'r')
fh2_2 = open(data_prefix + 'visualization_peptide_coords', 'w')

fh3_0 = open(data_prefix + 'train_peptide_onehot', 'r')
fh3_1 = open(data_prefix + 'test_peptide_onehot', 'r')
fh3_2 = open(data_prefix + 'visualization_peptide_onehot', 'w')

fh4_0 = open(data_prefix + 'train_receptor_coords', 'r')
fh4_1 = open(data_prefix + 'test_receptor_coords', 'r')
fh4_2 = open(data_prefix + 'visualization_receptor_coords', 'w')

fh5_0 = open(data_prefix + 'train_receptor_onehot', 'r')
fh5_1 = open(data_prefix + 'test_receptor_onehot', 'r')
fh5_2 = open(data_prefix + 'visualization_receptor_onehot', 'w')

for i in range(number_of_train_samples):
    line = fh0_0.readline().strip()
    fh0_2.write(line + '\n')
    sample_name, n_p, n_r  = line.split(',')

    copy_lines(fh1_0, fh1_2, int(n_p) + 1)
    copy_lines(fh2_0, fh2_2, int(n_p) + 1)
    copy_lines(fh3_0, fh3_2, 2)
    copy_lines(fh4_0, fh4_2, int(n_r) + 1)
    copy_lines(fh5_0, fh5_2, 2)

for i in range(number_of_test_samples):
    line = fh0_1.readline().strip()
    fh0_2.write(line + '\n')
    sample_name, n_p, n_r  = line.split(',')

    copy_lines(fh1_1, fh1_2, int(n_p) + 1)
    copy_lines(fh2_1, fh2_2, int(n_p) + 1)
    copy_lines(fh3_1, fh3_2, 2)
    copy_lines(fh4_1, fh4_2, int(n_r) + 1)
    copy_lines(fh5_1, fh5_2, 2)

fh0_0.close()
fh0_1.close()
fh0_2.close()

fh1_0.close()
fh1_1.close()
fh1_2.close()

fh2_0.close()
fh2_1.close()
fh2_2.close()

fh3_0.close()
fh3_1.close()
fh3_2.close()

fh4_0.close()
fh4_1.close()
fh4_2.close()

fh5_0.close()
fh5_1.close()
fh5_2.close()
