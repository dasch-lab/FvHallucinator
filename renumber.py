#!/usr/bin/env python3

import io
import os
import argparse
import requests
import time

def renumber_pdb(old_pdb, renum_pdb):
    success = False
    time.sleep(5)
    for i in range(10):
        try:
            with open(old_pdb, 'rb') as f:
                response = requests.post(
                    'http://www.bioinf.org.uk/abs/abnum/abnumpdb.cgi',
                    params={
                        "plain": "1",
                        "output": "-HL",
                        "scheme": "-c"
                    },
                    files={"pdb": f})

            success = response.status_code == 200 and not ("<html>"
                                                           in response.text)

            if success:
                break
            else:
                time.sleep((i + 1) * 5)
        except requests.exceptions.ConnectionError:
            time.sleep(60)

    # if success:
    if success:
        new_pdb_data = response.text
        with open(renum_pdb, "w") as f:
            f.write(new_pdb_data)
    else:
        print(
            "Failed to renumber PDB. This is likely due to a connection error or a timeout with the AbNum server."
        )

if __name__ == "__main__":

  # Initialize the argument parser
  argparser = argparse.ArgumentParser()
  argparser.add_argument('-i', '--input', help='input pdb', dest='input', default=None, type=str, required=True)
  argparser.add_argument('-o', '--output', help='output pdb', dest='output', type=str, required=True)
  
  # Parse arguments
  args = argparser.parse_args()

  # Renumber pdb
  renumber_pdb(args.input, args.output)