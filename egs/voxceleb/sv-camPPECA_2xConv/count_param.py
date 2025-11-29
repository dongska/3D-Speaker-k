from speakerlab.utils.builder import build
import argparse
import os
import sys
import argparse
import torch
import torchaudio
from kaldiio import WriteHelper
from speakerlab.utils.utils import get_logger
from speakerlab.utils.config import build_config
from speakerlab.utils.fileio import load_wav_scp
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Total params (M): {total / 1e6:.3f} M")
    return total

parser = argparse.ArgumentParser(description='Extract embeddings for evaluation.')
parser.add_argument('--exp_dir', default='exp/camPPECA_2xConv', type=str, help='Exp dir')


args = parser.parse_args(sys.argv[1:])
config_file = os.path.join(args.exp_dir, 'config.yaml')
config = build_config(config_file)
embedding_model = build('embedding_model', config)

count_parameters(embedding_model)
