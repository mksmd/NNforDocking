import os


def print_menu(path_to_list):
    menu_list = {}
    i = 0
    print('\n' + str(i) + ' - show this menu')
    i += 1
    print('\n-    reconstruct    -')
    print('---------------------')
    print(str(i) + ' - run sampler.py with current model')
    i += 1
    print(str(i) + ' - run sampler.py with optimal train model')
    i += 1
    print(str(i) + ' - run sampler.py with optimal test model')
    print('---------------------')
    print('\n-     visualize     -')
    print('---------------------')
    sample_list_fh = open(path_to_list, 'r')
    for line in sample_list_fh:
        sample, p_atoms, r_atoms = line.strip().split(',')
        if os.path.isfile('data/outputs/' + sample + '.pdb'):
            i += 1
            print(str(i) + ' - ' + sample)
            menu_list[str(i)] = sample
    if len(menu_list) == 0:
        print('No PDBs to visualize. Run sampler.py first')
    sample_list_fh.close()
    print('---------------------')
    i += 1
    print('\n' + str(i) + ' - quit\n\n')
    return menu_list

def loop_choice(path_to_list):
    menu_list = print_menu(path_to_list)
    loop = True
    while loop:
        choice = input('input: ')
        if not choice.isdigit():
            menu_list = print_menu(path_to_list)
        elif choice == '0':
            menu_list = print_menu(path_to_list)
        elif choice == '1':
            message = os.system('python ./sampler.py')
            message = os.system('python ./sampler.py --offset')
            message = os.system('python ./merge_pdbs.py')
        elif choice == '2':
            message = os.system('python ./sampler.py --path-to-model=model_ppd_optimal_train.pt')
            message = os.system('python ./sampler.py --path-to-model=model_ppd_optimal_train.pt --offset')
            message = os.system('python ./merge_pdbs.py')
        elif choice == '3':
            message = os.system('python ./sampler.py --path-to-model=model_ppd_optimal_test.pt')
            message = os.system('python ./sampler.py --path-to-model=model_ppd_optimal_test.pt --offset')
            message = os.system('python ./merge_pdbs.py')
        elif 3 < int(choice) < len(menu_list) + 4:
            sample = menu_list[choice]
            cmd = 'pymol -x -Q '
            cmd += 'data/pepbdb/' + sample + '/receptor1.pdb '
            cmd += 'data/pepbdb/' + sample + '/peptide1.pdb '
            cmd += 'data/outputs/' + sample + '.pdb '
            cmd += '-d \"select r, receptor1; select p, peptide1; select s, ' + sample + ';'
            cmd += '     show sticks, r; show sticks, p; show spheres, s;'
            cmd += '     color gray50, r; color green, p; color magenta, s;'
            cmd += '     deselect; zoom" >/dev/null 2>&1 &'
            message = os.system(cmd)
        elif choice == str(len(menu_list) + 4):
            loop = False
        else:
            menu_list = print_menu(path_to_list)

loop_choice('data/inputs/visualization_sample_list')
