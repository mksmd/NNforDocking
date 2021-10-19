# Neural network-based predictor for ligand-receptor docking

## Overview

This repository includes a set of scripts implementing an end-to-end pipeline to pre-process data and train a deep machine learning model for the prediction of binding region for a ligand-receptor complex. The PyTorch framework is used to implement the machine learning formalism. In particular, the model utilizes a mix of convolutional and fully connected layers.

 All the training experiments were performed on the data from the [PepBDB](http://huanglab.phys.hust.edu.cn/pepbdb/) dataset, thus, the pipeline assumes that the inputs are organized and formatted in a PepBDB fashion.
Each of the inputs consists of a PDB file with receptor and a PDB file with ligand to be docked. Based on the inputs, the neural network will output the most probable docking location of a ligand on a receptor's surface.
There are two limitations implied with respect to the size of receptors (up to 4000 heavy atoms) and ligands (up to 300 heavy atoms) and one limitation for the atom types (should fit the set of 'C, N, O, P, S, F, Cl, Br, I').
 When converting to inputs, the complexes with lower number of atoms are zero-padded.

## Installation

Prior to model training, all required dependencies can be automatically installed via package manager [pip](https://pip.pypa.io/en/stable/).

```bash
pip install -r requirements.txt
```

## Data

The training data can be downloaded from the [PepBDB](http://huanglab.phys.hust.edu.cn/pepbdb/) site and extracted to `data/pepbdb/`.
All the necessary pre-processing can be done within the `utilities/` folder.

```bash
cd utilities/
python 0-refine_pdbs.py
python 1-convert_pdbs_2_inputs.py
```

The PDB-files from`data/pepbdb/` are examined for correctness and  limitations (sizes and atom types), split into training and test subsets, and converted to inputs (stored in `data/inputs/`).

Additionally, a subset of complexes can be defined for the visualization purposes:

```bash
python 2-define_visualization_set.py
```

## Training

The training is invoked by:

```bash
sh 0-training.sh
```

This shell script calls `trainer.py` twice to perform trainings with left and right zero-padding.
During this procedure, the models and logs for each of the zero-padding versions are stored in `offset_n/` and `offset_y/` folders. Each of the folders will include the final trained model and models performing best on train and test subsets.

## Sampling and visualizing

The trained models can be used for making predictions and visualization of the predicted coordinates of ligands (the set of visualized complexes is defined above via invoking `2-define_visualization_set.py`). Thus, run:

```bash
python visualizer.py
```

and follow the instructions within. In partuclar, it will call `sampler.py` to build predictions with both-zero-paddings and then  `merge_pdbs.py` to merge PDBs into final ones. The PDBs can be viewed with PyMOL.
