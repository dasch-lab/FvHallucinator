#!/usr/bin/env python3

import io
import os
import re
import sys
import glob
import argparse
import shutil
import tarfile
from pyrosetta import *
import pyrosetta.rosetta.protocols.simple_moves as psm
import numpy
import torch
import pandas

import matplotlib.pyplot as plt

init_string = "-mute all -check_cdr_chainbreaks false -detect_disulf true"
pyrosetta.init(init_string)

_aa_1_3_dict = {
    'A': 'ALA',
    'C': 'CYS',
    'D': 'ASP',
    'E': 'GLU',
    'F': 'PHE',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'K': 'LYS',
    'L': 'LEU',
    'M': 'MET',
    'N': 'ASN',
    'P': 'PRO',
    'Q': 'GLN',
    'R': 'ARG',
    'S': 'SER',
    'T': 'THR',
    'V': 'VAL',
    'W': 'TRP',
    'Y': 'TYR',
    '-': 'GAP'
}

def mutate_pose(base_pose,
                ros_positions,
                seq,
                outfile='mutated.pdb',
                dump=True):
    
    def MutateResidue(pose_mod, target_indices, new_residues):
      assert (len(target_indices) == len(new_residues))
      for tind, nr in zip(target_indices, new_residues):
          mutate_mover = psm.MutateResidue(tind, nr)
          mutate_mover.apply(pose_mod)

      return pose_mod

    pose = base_pose.clone()
    seqlist = [_aa_1_3_dict[t] for t in seq]
    new_pose = MutateResidue(pose, ros_positions, seqlist)

    if dump:
        new_pose.dump_pdb(outfile)

    return new_pose

def heavy_bb_rmsd_from_atom_map(pose_1, pose_2, residue_list):
    assert pose_1.size() == pose_2.size()
    atom_id_map = pyrosetta.rosetta.std.map_core_id_AtomID_core_id_AtomID()
    pyrosetta.rosetta.core.scoring.setup_matching_protein_backbone_heavy_atoms(pose_1, pose_2, atom_id_map)
    residues_mask_vector = pyrosetta.rosetta.utility.vector1_bool()
    residues_mask_vector.extend([True if i in residue_list else False for i in range(pose_1.size())])
    rmsd_vector = pyrosetta.rosetta.core.scoring.per_res_rms_at_corresponding_atoms_no_super(pose_1, pose_2, atom_id_map, residues_mask_vector)
    return rmsd_vector

if __name__ == "__main__":

  # Initialize the argument parser
  argparser = argparse.ArgumentParser()
  argparser.add_argument('-t', '--template', help='template', dest='template', default=None, type=str, required=True)
  argparser.add_argument('-i', '--input', help='input folder', dest='input', type=str, required=True)

  # Parse arguments
  args = argparser.parse_args()

  
  # Load template
  native_pose = pyrosetta.pose_from_pdb(args.template)

  # Loop through all files in the folder
  metrics_list = []
  residue_list = []
  for filename in os.listdir(args.input):
    # Check if the file has a .pdb extension
    if not filename.endswith(".pdb"):
      continue

    pose = pyrosetta.pose_from_pdb(os.path.join(args.input, filename))

    

    # Extract sequence
    sequence = pose.sequence()
    target_sequence = native_pose.sequence()

    # Calculate which residues vary
    res_positions = [ i for i in range(len(sequence)) if target_sequence[i] != sequence[i] ]
    ros_positions = [t + 1 for t in res_positions]
    residue_list = list(set(residue_list).union(set(ros_positions)))


  for filename in os.listdir(args.input):
    # Check if the file has a .pdb extension
    if not filename.endswith(".pdb"):
      continue

    pose = pyrosetta.pose_from_pdb(os.path.join(args.input, filename))

    # Superpose structures
    pyrosetta.rosetta.core.scoring.calpha_superimpose_pose(pose, native_pose)

    print(filename)
    # Calculate the per residue RMSD between the two structures using residue_rmsd_super()
    residue_rmsd = {}
    for i in residue_list:
        res1 = native_pose.residue(i).xyz("CA")
        res2 = pose.residue(i).xyz("CA")

        # Convert the lists to NumPy arrays
        array1 = numpy.array(res1)
        array2 = numpy.array(res2)

        # Calculate the Euclidean distance between the arrays
        distance = numpy.linalg.norm(array1 - array2)
        residue_rmsd[i] = distance

    metrics_list.append(residue_rmsd)

    # # pyrosetta.rosetta.core.scoring.calpha_superimpose_pose(pose, native_pose)
    # sequence = pose.sequence()
    # target_sequence = native_pose.sequence()
    
    # ros_positions = [t + 1 for t in res_positions]
    # seq = [sequence[pos] for pos in res_positions]
    # mutated_pose = mutate_pose(native_pose, ros_positions, seq, dump=False)
    # metrics = heavy_bb_rmsd_from_atom_map(pose, mutated_pose, res_positions)
    
    # metrics = {key: value for key, value in metrics.items()}
    # metrics_list.append(metrics)
    # print(metrics)
    # print(type(metrics))

  # Convert to pandas
  df = pandas.DataFrame(metrics_list)

  # Create a plot representation
  fig, ax = plt.subplots(figsize=(10, 10))


  # setting the axis' labels
  ax.set_ylabel('RMSD',fontsize=12)
  ax.set_xlabel('Position',fontsize=12)
  plt.xticks(range(len(residue_list)), list(map(str, residue_list)))

  # Append rows
  for index, row in df.iterrows():
     row_data = row.tolist()
     plt.plot(row_data, label=f'Row {index}')
     

  # Store results
  rmsd_path = os.path.join(args.input, 'rmds.tsv')
  df.to_csv(rmsd_path, sep='\t', index=False)

  # Store plot
  graph_path = os.path.join(args.input, 'rmsd.png')
  plt.tight_layout()
  plt.savefig(graph_path, transparent=True)
  plt.close()