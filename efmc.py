from converter import convert_txt_to_csv
from copy import deepcopy
from eeg_processor import EEGProcessor
from math import ceil
from numpy import log10
from os import listdir
from os.path import join, isfile

import numpy as np
import h5py

class EEGFingerMotorControlModel(object):
	def __init__(self, training_save_fn, freq_points, time_points):
		self.training_save_fn = training_save_fn
		self.freq_points = freq_points
		self.time_points = time_points
		self.efmc = None

	def process_training_data(self):
		""" Preprocesses training data
		"""
		data_dirs = sorted(["left-index-flexion",
							"left-middle-flexion",
							"left-ring-flexion"])
		txt_data_dir = "data/txt"
		csv_data_dir = "data/csv"

		# convert text data into CSV data
		for data_dir in data_dirs:
			convert_txt_to_csv(join(txt_data_dir, data_dir),
							   join(csv_data_dir, data_dir))

		x_train = {channel_idx:[] for channel_idx in range(8)}
		y_train = []
		training_sample_count = 0

		# iterate through all class directories
		for class_idx, data_dir in enumerate(data_dirs):
			class_dir = join(csv_data_dir, data_dir)
			class_files = [class_file for class_file in listdir(class_dir) if (isfile(join(class_dir, class_file))) and (".csv" in class_file)]

			# iterate through all class files
			for class_file in class_files:
				session_title = " ".join(class_file.split("-"))
				eeg_processor =  EEGProcessor(class_dir, class_file, "openbci", session_title)
				eeg_processor.plot = 'show'
				eeg_processor.load_data()

				# iterate through all channels 
				for channel_idx, channel in enumerate(eeg_processor.channels):
					print " ".join(["Processing channel ",
									str(channel_idx + 1)])

					# load and clean channel data
					eeg_processor.load_channel(channel)
					eeg_processor.remove_dc_offset()
					eeg_processor.notch_mains_interference()
					eeg_processor.trim_data(10, 10)

					# calculate spectrogram
					eeg_processor.get_spectrum_data()
					eeg_processor.data = eeg_processor.bandpass(1, 50)
					spec = 10*log10(eeg_processor.spec_PSDperBin)

					# accumulate sample data
					sample_count = spec.shape[1] / self.time_points
					for sample_idx in xrange(sample_count):
						sample = spec[0:self.freq_points,
						              sample_idx*self.time_points:
									  sample_idx*self.time_points+self.time_points]
						sample = np.array([sample]*3)
						x_train[channel_idx].append(sample)
		
				# accumulate label data
				training_sample_count += sample_count
				label = [0]*len(data_dirs)
				label[class_idx] = 1
				label = np.array(label)
				y_train.extend([label]*sample_count)

		# format sample and label data
		x_train = [np.array([cha_1, cha_2, cha_3, cha_4, cha_5, cha_6, cha_7, cha_8])
				   for cha_1, cha_2, cha_3, cha_4, cha_5, cha_6, cha_7, cha_8  
				   in zip(x_train[0], x_train[1], x_train[2], x_train[3],
						  x_train[4], x_train[5], x_train[6], x_train[7])]
		x_train = np.array(x_train)
		y_train = np.array(y_train)
						
		# save sample and label data into HDF5 file
		with h5py.File(self.training_save_fn, "w") as training_save_file:
			training_save_file.attrs["training_classes"] = np.string_(",".join(data_dirs))
			training_save_file.attrs["sample_count"] = training_sample_count
			x_training_dataset = training_save_file.create_dataset("X",
																   shape=(training_sample_count, 8, 3, self.freq_points, self.time_points),
																   dtype="f")
			x_training_dataset[:] = x_train
			y_training_dataset = training_save_file.create_dataset("Y",
																   shape=(training_sample_count, len(data_dirs)),
																   dtype="i")
			y_training_dataset[:] = y_train

if __name__ == "__main__":
	efmc = EEGFingerMotorControlModel(training_save_fn = "training_data.h5",
									  freq_points = 250,
									  time_points = 50)
	efmc.process_training_data()
