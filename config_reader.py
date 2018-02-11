import csv
import numpy as np

FILE = 'march.csv'

configs = []
with open(FILE,'rb') as csvf:
    csvr = csv.reader(csvf)
    for row in csvr:
        if row[0] == 'Case Index':
            continue
        char_count = 0
        for char in row:
            if char == '':
                break
            char_count += 1
        configs.append(np.array(row[1:char_count]).reshape(-1,3).astype(int))

np.save('configs.npy',np.array(configs))
