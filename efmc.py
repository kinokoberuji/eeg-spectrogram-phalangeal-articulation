from converter import convert_txt_to_csv

import EEGrunt
import os

data_dirs = ["left-index-flexion",
			 "left-middle-flexion",
			 "left-ring-flexion"]

txt_data_dir = "data/txt"

csv_data_dir = "data/csv"

for data_dir in data_dirs:
	convert_txt_to_csv(os.path.join(txt_data_dir, data_dir),
					   os.path.join(csv_data_dir, data_dir))

