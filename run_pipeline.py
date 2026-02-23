"""
CLI - main running script (entry point)
"""

import argparse
from pathlib import Path

from src.config import GENGRU_PATH, CTT_PATH, DIST_DATA_PATH, OUTPUT_PATH, FOLDER_SIGNATURE, DROP_COLS, DEVICE
from src.generators.architectures.architectures import GenRNN
from src.acp_maker import ACPmaker

def main(args):

    # loading arguments
    run_id = args.id
    n_batches = args.nbatch
    to_mutate = args.mutate
    device = args.device

    # running pipeline
    acp_maker = ACPmaker(GENGRU_PATH, CTT_PATH, DIST_DATA_PATH, device)

    output_folder = OUTPUT_PATH / FOLDER_SIGNATURE.replace('XX',str(run_id))
    acp_maker.run_pipeline(n_batches, output_folder, DROP_COLS ,to_mutate)

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--id',
        type=int,
        default=69,
        help="identification number of the run"
    )

    parser.add_argument(
        '--nbatch',
        type=int,
        default=100,
        help="how many batches of peptides to generate"
    )

    parser.add_argument(
        '--mutate',
        type=str,
        default=None,
        help="aa sequence to mutate"
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'mps'],
        default=DEVICE,
        help="device for computation"
    )

    ### custom model selection ###

    parser.add_argument(
        '--model_ctt',
        type=Path,
        default=CTT_PATH,
        help="path to classification model"
    )

    parser.add_argument(
        '--model_gen',
        type=Path,
        default=GENGRU_PATH,
        help="path to generative model"
    )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
    main(args)