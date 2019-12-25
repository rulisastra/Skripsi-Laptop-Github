import os
import csv
import time
import pandas as pd
from datetime import datetime
import numpy as np

dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset/features/")

dataset = [0]
dataset_header = ['timestamp']

for root, dirs, files in os.walk(dir_path):
	for file in files:
		file_name = file.split(".")[0]
		# print(file_name)
		dataset_header.append(file_name)
		file_path = os.path.join(dir_path, file)
		read_file = pd.read_csv(file_path, header=None).as_matrix()
		temp_file = []
		temp_timestamp = []
		for i in range(len(read_file)):
			prev_row = read_file[i-1] if i-1 > 0 else None
			row = read_file[i]
			next_row = read_file[i+1] if i+1 < len(read_file) else None
			# Convert to UNIX timestamp
			dt_obj = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S").timetuple()
			timestamp = int(time.mktime(dt_obj))
			# Check if timestamp between August 2010 until October 2017 in GMT+7
			if timestamp > 1280509200 and timestamp < 1509469200:
				if str(float(row[1])).lower() == 'nan':
					print('null', row[0])
					if next_row and prev_row:
						row[1] = (next_row + prev_row)/2
					elif prev_row:
						row[1] = prev_row
					else:
						row[1] = next_row
				temp_file.append(row[1])
				temp_timestamp.append(row[0])
		dataset[0] = temp_timestamp
		dataset.append(temp_file)

dataset = np.array(dataset).transpose()

myFile = open('dataset.csv', 'w', newline='')

with myFile:
    writer = csv.writer(myFile)
    writer.writerow(dataset_header)
    writer.writerows(dataset)

dataset_norm = dataset[:, 1:].astype(np.float)
dataset_norm = (dataset_norm - dataset_norm.mean()) / dataset_norm.std()
dataset_norm = dataset_norm.tolist()
for idx, val in enumerate(dataset_norm):
	val.insert(0, dataset[:, 0][idx])

myFileNorm = open('dataset-normalize.csv', 'w', newline='')

with myFileNorm:
	writer = csv.writer(myFileNorm)
	writer.writerow(dataset_header)
	writer.writerows(dataset_norm)