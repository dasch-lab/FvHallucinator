#!/usr/bin/env python3

import io
import os
import re
import sys
import glob
import argparse
import subprocess
import shutil
import tarfile
from urllib.request import urlopen
from pyrosetta import *
import numpy
import torch
import pandas

from hallucinate import get_target_geometries
from generate_fvs_from_sequences import build_structure
from src.deepab.models.AbResNet import load_model
from src.deepab.models.ModelEnsemble import ModelEnsemble

from src.hallucination.SequenceHallucinator import SequenceHallucinator
from src.hallucination.loss.setup_losses import setup_loss_components,\
    setup_loss_weights,\
    get_reference_losses,\
    debug_wt_losses
from src.util.pdb import get_pdb_chain_seq, \
    protein_pairwise_geometry_matrix
from src.util.masking import mask_from_indices_list
from src.hallucination.utils.util import get_model_file,\
    comma_separated_chain_indices_to_dict,\
    get_indices_from_different_methods,\
    convert_chain_aa_to_index_aa_map

# from hallucinate import 
# from src.deepab.build_fv.build_cen_fa \
#     import build_initial_fv, get_cst_file, refine_fv
# from src.deepab.build_fv.utils import migrate_seq_numbering, get_constraint_set_mover
# from src.deepab.build_fv.score_functions import get_sf_fa
# from src.hallucination.utils.util\
#     import get_indices_from_different_methods,\
#     comma_separated_chain_indices_to_dict, get_model_file
# from src.util.pdb import get_pdb_numbering_from_residue_indices, renumber_pdb,\
#     get_pdb_chain_seq

# from subprocess import call
# from subprocess import Popen, PIPE

# init_string = "-mute all -check_cdr_chainbreaks false -detect_disulf true"
# pyrosetta.init(init_string)
# torch.no_grad()

# import src.pdb as pdb
# import src.fragment as fragment
# import src.fasta as fasta

# Get script path
file_path  = os.path.split(__file__)[0]

def _parseFasta(path):
  with open(path, 'r') as fp:
    name, seq = None, []
    for line in fp:
      line = line.rstrip()
      if line.startswith(">"):

        # Append previous
        if name: 
            yield (name, ''.join(seq))

        name, seq = line[1:], []
      else:
        seq.append(line)

    # Append last in list
    if name: 
      yield (name, ''.join(seq))

def _parseInput(path):
  '''
  Parse an input fasta file with paired/unpaired protein sequences
  '''

  sequenceMap = {}
  for (header, sequence) in _parseFasta(path):

    name, chain = header.split(':')
    # Split sequence in VH and VL
    # sequence = sequence.split('/')
    sequenceMap.setdefault(name, {
      'name': name,
      'chain': {}
    })
    sequenceMap[name]['chain'][chain] = sequence
    # for i in range(len(sequence)):
    #   chain_type = 'VH' if i == 0 else 'VL'
    #   sequenceMap[name]['chain'][chain_type] = sequence[i]

  return sequenceMap

def __hallucinate(model, target_pdb, output_path, cdr_list='h1,h2,h3,l1,l2,l3', count=1, max_iters=50, prefix=None):


    # Set default parameters
    # cdr_list=''
    framework=False
    include_indices={}
    exclude_indices={}
    hl_interface=False
    # max_iters=100
    suffix=''
    # seed=0
    n_every=100
    restricted_positions_aa_freq={}
    restricted_dict_keep_aas={}
    disallowed_aas='C'
    # use_manual_seed=True
    autostop=True
    seed_with_WT=False
    apply_lr_scheduler=True
    lr_dict={'learning_rate': 0.05, 'patience': 20, 'cooldown': 10}
    pssm=None
    local_loss_only=True

    # Load hallucination weight
    hallucinate_args = argparse.Namespace()
    hallucinate_args.geometric_loss_list = '1,1,1,1,1,1'
    hallucinate_args.restrict_total_charge = False
    hallucinate_args.restrict_max_aa_freq = False
    hallucinate_args.seq_loss_weight = 0.0
    hallucinate_args.geometric_loss_weight= 1.0
    hallucinate_args.restricted_positions_kl_loss_weight = 10.0
    hallucinate_args.avg_seq_reg_loss_weight = 0.0
    loss_weights_for_run = setup_loss_weights(hallucinate_args)


    target_geometries = get_target_geometries(target_pdb)

    wt_heavy_seq, wt_light_seq = get_pdb_chain_seq(target_pdb,'H'), get_pdb_chain_seq(target_pdb, 'L')

    # Collect indices of positions to hallucinate
    indices_hal = get_indices_from_different_methods(
      target_pdb, cdr_list=cdr_list,
      framework=framework,
      hl_interface=hl_interface,
      include_indices=include_indices,
      exclude_indices=exclude_indices
    )

    wt_seq = wt_heavy_seq + wt_light_seq

    out_dir_losses = os.path.join(output_path, 'losses')
    os.makedirs(out_dir_losses, exist_ok=True)

    outnpy = os.path.join(out_dir_losses, "lossgeomfull_wt.npy")
    list_wt_mask = debug_wt_losses(wt_seq, wt_heavy_seq, model, target_geometries, device, outnpy)
    seq_design_mask = mask_from_indices_list(indices_hal, len(wt_seq))
    os.makedirs(output_path, exist_ok=True)

    # mask_2d = seq_design_mask.unsqueeze(1).expand(-1, 10)
    # plt.imshow(mask_2d, aspect='equal')
    # plt.colorbar()
    # plt.savefig('{}/design_mask.png'.format(outdir))
    # plt.close()
    non_design_mask = None
    seq_for_hal = ''.join(['*' if i in indices_hal
                           else t for i, t in enumerate(wt_seq)])
    # seeding with WT sequence
    if seed_with_WT:
        sequence_seed = wt_seq
    else:
        sequence_seed = None
    print("Sequence input for design: ", seq_for_hal)

    restricted_positions_aa_freq_indexed = {}
    if restricted_positions_aa_freq != {}:
        restricted_positions_aa_freq_indexed = \
            convert_chain_aa_to_index_aa_map(restricted_positions_aa_freq,
                                             target_pdb,
                                             len(wt_heavy_seq))
        print('Positions with restricted AA freqs at Indices: ', restricted_positions_aa_freq_indexed)

    restricted_dict_keep_aas_indexed = {}
    if restricted_dict_keep_aas != {}:
        restricted_dict_keep_aas_indexed = \
            convert_chain_aa_to_index_aa_map(restricted_dict_keep_aas,
                                             target_pdb,
                                             len(wt_heavy_seq))
        print('Positions with restricted AA at Indices: ', restricted_dict_keep_aas_indexed)

    out_dir_losses = os.path.join(output_path, 'losses')
    os.makedirs(out_dir_losses, exist_ok=True)
    loss_components, loss_components_dict = \
        setup_loss_components(wt_seq, model,
                              len(wt_heavy_seq),
                              target_geometries,
                              loss_weights_for_run,
                              seq_design_mask,
                              device=device,
                              restricted_dict_aa_freqs=restricted_positions_aa_freq_indexed,
                              restricted_dict_keep_aas=restricted_dict_keep_aas_indexed,
                              non_design_mask=non_design_mask,
                              pssm=pssm,
                              wt_losses_mask=list_wt_mask,
                              outdir=out_dir_losses,
                              local_loss_only=local_loss_only
                              )
    print('Components in loss ', loss_components_dict)
    wt_geom_loss = None
    if 'geom' in loss_components_dict:
        wt_geom_loss, _ = get_reference_losses(wt_seq, wt_heavy_seq, model,
                                               loss_components, device,
                                               loss_components_dict)

    seq_design_mask = mask_from_indices_list(indices_hal, len(seq_for_hal))
    

    traj_loss_dict = {}
    for key in loss_components_dict:
      traj_loss_dict[key] = []

    out_dir_trajs = os.path.join(output_path, 'trajectories')
    out_dir_int = os.path.join(output_path, 'intermediate')
    os.makedirs(out_dir_int, exist_ok=True)
    os.makedirs(out_dir_trajs, exist_ok=True)

    result_list = []
    prefix = 'sequence' if prefix is None else prefix
    for suffix in range(count):
      print('Producing hallucination {}'.format(suffix))

      sequence_hallucinator = SequenceHallucinator(
        wt_seq,
        len(wt_heavy_seq) - 1,
        model,
        loss_components,
        design_mask=seq_design_mask,
        device=device,
        sequence_seed=sequence_seed,
        apply_lr_scheduler=apply_lr_scheduler,
        lr_config=lr_dict,
        disallowed_aas_at_initialization=disallowed_aas).to(device)

      best_sequence = None
      for itr in range(max_iters):

        list_losses = sequence_hallucinator.update_sequence(disallow_letters=disallowed_aas)

        # Store loss
        total_loss = list_losses[0].sum().item()
        sequence = sequence_hallucinator.get_sequence()
        # print('Loss: {}; {}'.format(total_loss,sequence ))
        best_sequence = {
          'iteration': itr,
          'loss': total_loss,
          'VH': sequence[0],
          'VL': sequence[1]
        } if best_sequence is None or best_sequence['loss'] > total_loss else best_sequence

        if itr == 0:
            sequence_hallucinator.write_sequence_history_file(
                os.path.join(out_dir_int, "sequences_{}_init.fasta".format(suffix)))

        for key in traj_loss_dict:
            if key != 'reg_seq':
                traj_loss_dict[key].append(list_losses[loss_components_dict[key]].numpy())
            else:
                heavy_ll = list_losses[loss_components_dict[key]].numpy()
                light_ll = list_losses[loss_components_dict[key] + 1].numpy()
                traj_loss_dict[key].append((heavy_ll, light_ll))

        # learning rate based autostop criterion
        if autostop:
            print('  Iteration {}; learning rate: {}; start learning rate: {}'.format(itr+1, sequence_hallucinator.lr, sequence_hallucinator.start_lr))
            if sequence_hallucinator.start_lr / float(
                sequence_hallucinator.lr
            ) >= 100.0:
                print("Stopping at {} because learning rate has reached {} at iter {}".format(itr, sequence_hallucinator.lr, itr))
                break

        if (itr + 1) % n_every == 0:
          sequence_hallucinator.write_sequence_history_file(os.path.join(out_dir_int,"sequences_{}_{}.fasta".format(suffix, itr)))

            # outfile = os.path.join(out_dir_losses,
            #                        "loss_{{}}_{}_{}.png".format(suffix, itr))
            # plot_all_losses(traj_loss_dict, outfile,
            #                 max_iters, wt_geom_loss)

      # Best loss
      print('Best loss: {}; iter: {}; VH: {}; VL: {}'.format(best_sequence['loss'], best_sequence['iteration'], best_sequence['VH'], best_sequence['VL']))
      result_list.append({
        'name': prefix,
        'index': itr,
        'VH': best_sequence['VH'],
        'VL': best_sequence['VL'],
        'loss': best_sequence['loss']
      })

      # Write trajectory sequences
      sequence_hallucinator.write_sequence_history_file(os.path.join(out_dir_trajs, "sequences_{}_final.fasta".format(suffix)))

    # Save losses
    # outfile_loss_mat = os.path.join(out_dir_losses,
    #                                 "lossdict_{}_final.npy".format(suffix))
    # numpy.save(outfile_loss_mat, traj_loss_dict)

    # # Plot losses
    # outfile = os.path.join(out_dir_losses,
    #                        "loss_{{}}_{}_final.png".format(suffix))
    # plot_all_losses(traj_loss_dict, outfile,
    #                 max_iters, wt_geom_loss)

    # Done, store results
    df = pandas.DataFrame(result_list)
    df.to_csv(os.path.join(output_path, 'result.tsv'), sep='\t', index=False)

    # Store fasta data
    fasta_path = os.path.join(output_path, 'result.fasta')
    with open(fasta_path, 'w') as handle:
      for index, row in df.iterrows():
         handle.write('>{}_{}:H\n{}\n'.format(row['name'],index, row['VH']))
         handle.write('>{}_{}:L\n{}\n'.format(row['name'],index, row['VL']))
    

if __name__ == "__main__":

  # Initialize the argument parser
  argparser = argparse.ArgumentParser()
  argparser.add_argument('-i', '--input', help='input fasta', dest='input', default=None, type=str, required=True)
  argparser.add_argument('-o', '--output', help='output folder', dest='output', type=str, required=True)
  argparser.add_argument('-d', '--decoy', help='decoy number', dest='decoy', type=int, required=False, default=10)
  argparser.add_argument('-g', '--gpu', action="store_true", dest="gpu", help="Use gpu")

  # Parse arguments
  args = argparser.parse_args()

  # Check output path
  if os.path.isfile(args.output):
    raise Exception('Output must be a directory')

  # Generate the output path
  if not os.path.exists(args.output):
    os.mkdir(args.output)

  # Check if the model is in the cache file
  # model_path = fetchModel(os.path.join(file_path, 'FvHallucinator', 'trained_models', 'ensemble_abresnet'), model='https://data.graylab.jhu.edu/ensemble_abresnet_v1.tar.gz')
  # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  # # device = torch.device('cpu')
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  # torch.cuda.set_device(device)
  model_files = get_model_file()
  model = ModelEnsemble(load_model=load_model,
                        model_files=model_files,
                        device=device,
                        eval_mode=True)
  
  # Generate the sequence map from the input file
  sequenceMap = _parseInput(args.input)

  # Process sequence map
  statistics = {}
  jobList = []
  for name in sequenceMap:
    print('Processing Ab {0} [{1}]'.format(name, ';'.join(list(sequenceMap[name]['chain'].keys()))))

    # Skip unpaired Abs
    if len(sequenceMap[name]['chain']) != 2:
      continue

    # Generate the sequence file
    data_directory = os.path.join(args.output, name)
    if not os.path.exists(data_directory):
      os.mkdir(data_directory)

    # Create the input file
    input_path = os.path.join(data_directory, 'input.fa')
    with open(input_path, 'w') as handle:

      # Store chain data
      for chain in sequenceMap[name]['chain']:
        handle.write('>input:{0}\n{1}\n'.format(
          ('H' if (chain == 'VH' or chain == 'H') else 'L'),
          sequenceMap[name]['chain'][chain]
        ))

    # Append to job list
    jobList.append({
      'name': name,
      'jobid': name,
      # 'patient': sequenceMap[name]['patient'],
      'path': data_directory
    })

  # Process each job
  for i in range(len(jobList)):
    print('Processing job #{0}: {1}'.format(i, jobList[i]['path']))

    # Skip already calculated structures
    output_path = os.path.join(jobList[i]['path'], 'pred.deepab.pdb')
    if os.path.exists(output_path):
      print('  skipping prediction')
      continue

    # Build 3D structure if not already present
    pdb_path = os.path.join(jobList[i]['path'], 'out.deepAb.pdb')
    if not os.path.exists(pdb_path):
      build_structure(
        model, 
        os.path.join(jobList[i]['path'], 'input.fa'),
        jobList[i]['path'],
        None,
        20
        )
      
  # Generate hallucination
  for i in range(len(jobList)):
    print('Hallucinate job #{0}: {1}'.format(i, jobList[i]['path']))
      
    # Hallucinate sequence
    hallucinate_path = os.path.join(jobList[i]['path'], 'hallucination')
    if not os.path.exists(hallucinate_path):
      os.mkdir(hallucinate_path)

    fasta_path = os.path.join(hallucinate_path, 'result.fasta')
    if not os.path.exists(fasta_path):
      __hallucinate(model, pdb_path, hallucinate_path, count=10, max_iters=20, prefix=jobList[i]['name'])

  # Concatenate results
  df = None
  result_path = os.path.join(args.output, 'hallucinate.tsv')
  for i in range(len(jobList)):
    hallucinate_path = os.path.join(jobList[i]['path'], 'hallucination', 'result.tsv')
    if not os.path.exists(hallucinate_path):
      continue

    # read result
    df_hallucinate = pandas.read_csv(hallucinate_path, sep='\t')
    if df is None:
      df = df_hallucinate
      continue

    # Concatenate data
    df = pandas.concat([df, df_hallucinate])

  # Store result
  df.to_csv(result_path, sep='\t', index=False)



    