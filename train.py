from run_train import run_training
from model import *
import csv
import os
import argparse
import torchvision.models

parser = argparse.ArgumentParser()
parser.add_argument('--modelname', type=str, required=True)
parser.add_argument('--pretrained', type=str, required=False, default=False)
args = parser.parse_args()

# csv file name
csv_filename = "top_accuracy.csv"


# load model
model_name = args.modelname
# print('model_name:',model_name)
print('args :',args)
# print('model_name:',model_name)
if 'torchvision' in model_name:
    net = eval(model_name)(pretrained=(args.pretrained=='True'))
    print((args.pretrained=='True'))
else:
    net = eval(model_name)()




accuracy = run_training(net)

# Save top accuracy to the CSV file
if os.path.exists(csv_filename):
    mode = 'a'  # Append to the existing file
else:
    mode = 'w'  # Create a new file

with open(csv_filename, mode, newline='') as csvfile:
    writer = csv.writer(csvfile)
    if csvfile.tell() == 0:  # Check if the file is empty
        writer.writerow(["Model","pretrained", "Top Accuracy"])  # Write header row if it's a new file
    writer.writerow([model_name, args.pretrained, accuracy])  # Write model name and top accuracy
    

