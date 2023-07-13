import argparse
import torch
import torch
import torch.nn as nn



parser = argparse.ArgumentParser()
parser.add_argument('--modelname', type=str, required=True)
args = parser.parse_args()

model_name = args.modelname
print('model_name:',model_name)
net = eval(model_name)()
