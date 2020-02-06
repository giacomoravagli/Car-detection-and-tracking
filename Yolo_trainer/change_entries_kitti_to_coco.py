import os, sys, shutil, glob
import numpy as np

label_path = "data/artifacts/labels2/"
output_path = "data/artifacts/labels1/"


for f in os.listdir(label_path):
    input_file = open(label_path + f, 'r')
    output_file = open(output_path + f, 'w')

    for i, line in enumerate(input_file):
        if not line.startswith('6'):
            output_file.write(line)
        else:
            first, other = line.split(None, 1)
            line = '%d %s' % (5, other)
            output_file.write(line)