from converter import convert_txt_to_csv
from copy import deepcopy
from eeg_processor import EEGProcessor
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.callbacks import Callback
from keras.models import Model, model_from_json
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Nadam
from keras.preprocessing.image import random_rotation, random_shift, random_shear, random_zoom
from keras.utils.io_utils import HDF5Matrix
from pprint import pprint
from math import ceil
from numpy import log10
from os import listdir
from os.path import join, isfile
from json import dumps, loads

import numpy as np
import h5py

class EEGFingerMotorControlModel(object):
	def __init__(self, training_save_fn, samples_generated_per_sample, augmentation,
				 augmentation_magnitude, freq_points, time_points):
		K.set_image_dim_ordering("th")
		self.training_save_fn = training_save_fn
		self.samples_generated_per_sample = samples_generated_per_sample
		self.augmentation = augmentation
		self.augmentation_magnitude = augmentation_magnitude
		self.freq_points = freq_points
		self.time_points = time_points
		self.efmc = None

	def train_efmc_model(self):
		""" Train the EEG finger motor control model
		"""
		print "\nTraining EFMC"
		validation_ratio = 0.3
		batch_size = 32
		with h5py.File(self.training_save_fn, "r") as training_save_file:
			sample_count = int(training_save_file.attrs["sample_count"])
			sample_idxs = range(0, sample_count)
			sample_idxs = np.random.permutation(sample_idxs)
			training_sample_idxs = sample_idxs[0:int((1-validation_ratio)*sample_count)]
			validation_sample_idxs = sample_idxs[int((1-validation_ratio)*sample_count):]
			training_sequence_generator = self.generate_training_sequences(batch_size=batch_size,
																		   training_save_file=training_save_file,
																		   training_sample_idxs=training_sample_idxs)
			validation_sequence_generator = self.generate_validation_sequences(batch_size=batch_size,
																			   training_save_file=training_save_file,
																			   validation_sample_idxs=validation_sample_idxs)
			progress_display = ProgressDisplay()
			metrics_history = self.efmc.fit_generator(generator=training_sequence_generator,
								                      validation_data=validation_sequence_generator,
								                      samples_per_epoch=len(training_sample_idxs),
								                      nb_val_samples=len(validation_sample_idxs),
								                      nb_epoch=10,
								                      max_q_size=1,
								                      verbose=2,
								                      callbacks=[progress_display],
								                      class_weight=None,
								                      nb_worker=1)
		return metrics_history

	def generate_training_sequences(self, batch_size, training_save_file, training_sample_idxs):
		""" Generates training sequences from HDF5 file on demand
		"""
		while True:
			# generate sequences for training
			training_sample_count = len(training_sample_idxs)
			batches = int(training_sample_count/batch_size)
			remainder_samples = training_sample_count%batch_size
			if remainder_samples:
				batches = batches + 1
			# generate batches of samples
			for idx in xrange(0, batches):
				if idx == batches - 1:
					batch_idxs = training_sample_idxs[idx*batch_size:]
				else:
					batch_idxs = training_sample_idxs[idx*batch_size:idx*batch_size+batch_size]
				batch_idxs = sorted(batch_idxs)

				X = training_save_file["X"][batch_idxs]
				Y = training_save_file["Y"][batch_idxs]

				yield (np.array(X), np.array(Y))

	def generate_validation_sequences(self, batch_size, training_save_file, validation_sample_idxs):
		""" Generates validation sequences from HDF5 file on demand
		"""
		while True:
			# generate sequences for validation
			validation_sample_count = len(validation_sample_idxs)
			batches = int(validation_sample_count/batch_size)
			remainder_samples = validation_sample_count%batch_size
			if remainder_samples:
				batches = batches + 1
			# generate batches of samples
			for idx in xrange(0, batches):
				if idx == batches - 1:
					batch_idxs = validation_sample_idxs[idx*batch_size:]
				else:
					batch_idxs = validation_sample_idxs[idx*batch_size:idx*batch_size+batch_size]
				batch_idxs = sorted(batch_idxs)

				X = training_save_file["X"][batch_idxs]
				Y = training_save_file["Y"][batch_idxs]

				yield (np.array(X), np.array(Y))

	def print_efmc_summary(self):
		""" Prints a summary representation of the OSR model
		"""
		print "\n*** MODEL SUMMARY ***"
		self.efmc.summary()

	def generate_efmc_model(self):
		""" Builds the EEG finger motor control model
		"""
		print "\nGenerating EEG finger motor control model..."
		with h5py.File(self.training_save_fn, "r") as training_save_file:
			class_count = len(training_save_file.attrs["training_classes"].split(","))

		# input layer
		spectrograms = Input(shape=(8,
								    3,
								    self.freq_points,
								    self.time_points))

		# CNN layers
		cnn_base = VGG16(input_shape=(3,
									  self.freq_points,
									  self.time_points),
						 weights="imagenet",
						 include_top=False)
		cnn_out = GlobalAveragePooling2D()(cnn_base.output)
		cnn = Model(input=cnn_base.input, output=cnn_out)
		cnn.trainable = False
		encoded_spectrograms = TimeDistributed(cnn)(spectrograms)

		# RNN layers
		encoded_spectrograms = LSTM(256)(encoded_spectrograms)

		# MLP layers
		hidden_layer = Dense(output_dim=1024, activation="relu")(encoded_spectrograms)
		outputs = Dense(output_dim=class_count, activation="softmax")(hidden_layer)

		# compile model
		efmc = Model([spectrograms], outputs)
		optimizer = Nadam(lr=0.00015,
						  beta_1=0.9,
						  beta_2=0.999,
						  epsilon=1e-08,
						  schedule_decay=0.004)
		efmc.compile(loss="categorical_crossentropy",
					 optimizer=optimizer,
					 metrics=["categorical_accuracy"])
		self.efmc = efmc

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
			class_files = [class_file
			               for class_file in listdir(class_dir)
			               if (isfile(join(class_dir, class_file))) and (".csv" in class_file)]

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

						format_spec = lambda spectrogram: np.array([spectrogram]*3)
						x_train[channel_idx].append(format_spec(sample))

						# increase sample data via image augmentation
						if self.augmentation:
							for _ in xrange(0, self.samples_generated_per_sample-1):
								shifted_sample = random_shift(np.array([sample]),
									                          wrg = self.augmentation_magnitude, 
									                          hrg = self.augmentation_magnitude)
								x_train[channel_idx].append(format_spec(shifted_sample[0]))
						# increase sample data via image replication
						else:
							for _ in xrange(0, self.samples_generated_per_sample-1):
								x_train[channel_idx].append(format_spec(sample))

				# accumulate label data
				training_sample_count += sample_count*self.samples_generated_per_sample
				label = [0]*len(data_dirs)
				label[class_idx] = 1
				label = np.array(label)
				y_train.extend([label]*(sample_count*self.samples_generated_per_sample))

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

		print "\nGenerated {0} training samples for training classes {1}".format(int(training_sample_count),
			                                                                     ", ".join(data_dirs))

class ProgressDisplay(Callback):
	""" Progress display callback
	"""
	def on_batch_end(self, epoch, logs={}):
		print "    Batch {0:<4d} => Accuracy: {1:>8.4f} | Loss: {2:>8.4f} | Size: {3:>4d}".format(int(logs["batch"])+1,
																					              float(logs["categorical_accuracy"]),
																					              float(logs["loss"]),
																					              int(logs["size"]))

# auxilliary functions
def get_training_configuration(training_config_fn):
	""" Acquires training configuration from a file
	"""
	with open(training_config_fn, "r") as training_config_file:
    	training_config = loads(training_config_file)
    return training_config

if __name__ == "__main__":
	efmc = EEGFingerMotorControlModel(training_save_fn = "training_data.h5",
									  samples_generated_per_sample = 10,
									  augmentation = True,
									  augmentation_magnitude = 0.05,
									  freq_points = 250,
									  time_points = 50)
	efmc.process_training_data()
	efmc.generate_efmc_model()
	efmc.print_efmc_summary()
	metrics_history = efmc.train_efmc_model()

