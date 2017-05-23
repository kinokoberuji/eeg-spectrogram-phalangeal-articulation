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
from json import dump, load

import h5py
import numpy as np
import os
import sys

class ESPAModel(object):
	""" Core class for instantiating and interfacing with ESPA models
	"""
	def __init__(self,
		         data_save_fn, validation_ratio, testing_ratio, samples_generated_per_sample,
		         augmentation, augmentation_magnitude, freq_points, time_points,
		         espa_save_fn, espa_weights_save_fn):
		K.set_image_dim_ordering("th")
		self.data_save_fn = data_save_fn
		self.validation_ratio = validation_ratio
		self.testing_ratio = testing_ratio
		self.samples_generated_per_sample = samples_generated_per_sample
		self.augmentation = augmentation
		self.augmentation_magnitude = augmentation_magnitude
		self.freq_points = freq_points
		self.time_points = time_points
		self.espa_save_fn = espa_save_fn
		self.espa_weights_save_fn = espa_weights_save_fn
		self.espa = None

	def train_espa_model(self):
		""" Trains the ESPA model
		"""
		print "\nTraining ESPA Model"
		batch_size = 32
		with h5py.File(self.data_save_fn, "r") as data_save_file:
			# indices
			training_sample_idxs = np.random.permutation(range(int(data_save_file.attrs["training_sample_count"])))
			validation_sample_idxs = np.random.permutation(range(int(data_save_file.attrs["validation_sample_count"])))

			# generators
			training_sequence_generator = self.generate_data(data_save_file=data_save_file,
				                                             data_set="train",
															 sample_idxs=training_sample_idxs,
															 batch_size=batch_size)
			validation_sequence_generator = self.generate_data(data_save_file=data_save_file,
				                                               data_set="val",
															   sample_idxs=validation_sample_idxs,
															   batch_size=batch_size)

			# fit model
			progress_display = ProgressDisplay()
			metrics_history = self.espa.fit_generator(generator=training_sequence_generator,
								                      validation_data=validation_sequence_generator,
								                      samples_per_epoch=len(training_sample_idxs),
								                      nb_val_samples=len(validation_sample_idxs),
								                      nb_epoch=10,
								                      verbose=2,
								                      callbacks=[progress_display],
								                      class_weight=None,
								                      nb_worker=1)
		return metrics_history

	def test_espa_model(self):
		""" Tests the ESPA model
		"""
		print "\nTesting ESPA Model"
		batch_size = 32
		with h5py.File(self.data_save_fn, "r") as data_save_file:
			# indices
			testing_sample_idxs = np.random.permutation(range(int(data_save_file.attrs["testing_sample_count"])))

			# generators
			testing_sequence_generator = self.generate_data(data_save_file=data_save_file,
				                                            data_set="test",
															sample_idxs=testing_sample_idxs,
															batch_size=batch_size)

			# calculate steps
			sample_count = len(testing_sample_idxs)
			batches = int(sample_count/batch_size)
			remainder_samples = sample_count%batch_size
			if remainder_samples:
				batches = batches + 1

			# test model
			metrics = self.espa.evaluate_generator(testing_sequence_generator,
				                                   batches)

			# map metric names to metric values
			metrics = {metric_name: metric_value for metric_name, metric_value in zip(self.espa.metrics_names, metrics)}

			print "Accuracy: {0:>8.4f} | Loss: {1:>8.4f}".format(float(metrics["categorical_accuracy"]),
																 float(metrics["loss"]))

		return metrics

	def generate_data(self, data_save_file, data_set, sample_idxs, batch_size):
		""" Generates data from HDF5 file on demand
		"""
		while True:
			# determine batches
			sample_count = len(sample_idxs)
			batches = int(sample_count/batch_size)
			remainder_samples = sample_count%batch_size
			if remainder_samples:
				batches = batches + 1

			# generate batches
			for idx in xrange(batches):
				# incomplete batches
				if idx == batches - 1:
					batch_idxs = sample_idxs[idx*batch_size:]

				# complete batches
				else:
					batch_idxs = sample_idxs[idx*batch_size:idx*batch_size+batch_size]

				batch_idxs = sorted(batch_idxs)

				X = data_save_file["_".join(["x", data_set])][batch_idxs]
				Y = data_save_file["_".join(["y", data_set])][batch_idxs]

				yield (np.array(X), np.array(Y))

	def print_espa_summary(self):
		""" Prints a summary representation of the ESPA model
		"""
		print "\n*** MODEL SUMMARY ***"
		self.espa.summary()

	def generate_espa_model(self):
		""" Compiles the ESPA model
		"""
		print "\nGenerating ESPA model..."
		with h5py.File(self.data_save_fn, "r") as data_save_file:
			class_count = len(data_save_file.attrs["classes"].split(","))

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
		espa = Model([spectrograms], outputs)
		optimizer = Nadam(lr=0.0002,
						  beta_1=0.9,
						  beta_2=0.999,
						  epsilon=1e-08,
						  schedule_decay=0.004)
		espa.compile(loss="categorical_crossentropy",
					 optimizer=optimizer,
					 metrics=["categorical_accuracy"])
		self.espa = espa

	def save_espa_model(self):
		""" Saves the ESPA model and weights
		"""
		# delete save files, if they already exist
		try:
			print "\nESPA save file \"{0}\" already exists! Overwriting previous saved file.".format(self.espa_save_fn)
			os.remove(self.espa_save_fn)
		except OSError:
			pass
		try:
			print "ESPA weights save file \"{0}\" already exists! Overwriting previous saved file.\n".format(self.espa_weights_save_fn)
			os.remove(self.espa_weights_save_fn)
		except OSError:
			pass

		# save ESPA model
		print "\nSaving ESPA model to \"{0}\"".format(self.espa_save_fn)
		with open(self.espa_save_fn, "w") as espa_save_file:
			espa_model_json = self.espa.to_json()
			espa_save_file.write(espa_model_json)

		# save ESPA model weights
		print "Saving ESPA model weights to \"{0}\"".format(self.espa_weights_save_fn)
		self.espa.save_weights(self.espa_weights_save_fn)

		print "Saved ESPA model and weights to disk\n"

	def load_espa_model(self):
		""" Loads the ESPA model and weights
		"""
		print "\nLoading ESPA model from \"{0}\"".format(self.espa_save_fn)
		with open(self.espa_save_fn, "r") as espa_save_file:
			espa_model_json = espa_save_file.read()
			self.espa = model_from_json(espa_model_json)

		print "Loading ESPA model weights from \"{0}\"".format(self.espa_weights_save_fn)
		with open(self.espa_weights_save_fn, "r") as espa_weights_save_file:
			self.espa.load_weights(self.espa_weights_save_fn)
			
		print "Loaded ESPA model and weights from disk\n"

	def process_data(self):
		""" Preprocesses sample data
		"""
		print "\nProcessing data..."

		data_dirs = sorted(["left-index-flexion",
							"left-middle-flexion",
							"left-ring-flexion"])
		txt_data_dir = "data/txt"
		csv_data_dir = "data/csv"

		# convert text data into CSV data
		for data_dir in data_dirs:
			convert_txt_to_csv(join(txt_data_dir, data_dir),
							   join(csv_data_dir, data_dir))

		X = {channel_idx:[] for channel_idx in range(8)}
		Y = []
		data_sample_count = 0

		# iterate through all class directories
		for class_idx, data_dir in enumerate(data_dirs):
			class_dir = join(csv_data_dir, data_dir)
			class_files = [class_file
			               for class_file in listdir(class_dir)
			               if (isfile(join(class_dir, class_file))) and (".csv" in class_file)]


			sys.stdout = open(os.devnull, "w") # silence EEG data processing standard outputs

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
						X[channel_idx].append(format_spec(sample))

				# accumulate label data
				data_sample_count += sample_count
				label = [0]*len(data_dirs)
				label[class_idx] = 1
				label = np.array(label)
				Y.extend([label]*(sample_count))

			sys.stdout = sys.__stdout__ # stop silencing standard outputs

		# format sample and label data
		X = [np.array([cha_1, cha_2, cha_3, cha_4, cha_5, cha_6, cha_7, cha_8])
			 for cha_1, cha_2, cha_3, cha_4, cha_5, cha_6, cha_7, cha_8  
			     in zip(X[0], X[1], X[2], X[3],
				     	X[4], X[5], X[6], X[7])]
		X = np.array(X)
		Y = np.array(Y)
						
		# save sample and label data into HDF5 file
		with h5py.File(self.data_save_fn, "w") as data_save_file:
			# partition data into training, validation, and testing sets
			sample_idxs = np.random.permutation(range(data_sample_count))
			training_sample_idxs = sample_idxs[0:int((1.0-self.validation_ratio-self.testing_ratio)*data_sample_count)]
			validation_sample_idxs = sample_idxs[int((1.0-self.validation_ratio-self.testing_ratio)*data_sample_count):int((1.0-self.testing_ratio)*data_sample_count)]
			testing_sample_idxs = sample_idxs[int((1.0-self.testing_ratio)*data_sample_count):]

			x_train, y_train = self.replicate_augment_data(X[training_sample_idxs], Y[training_sample_idxs])
			x_val, y_val = self.replicate_augment_data(X[validation_sample_idxs], Y[validation_sample_idxs])
			x_test, y_test = self.replicate_augment_data(X[testing_sample_idxs], Y[testing_sample_idxs])

			data_save_file.attrs["classes"] = np.string_(",".join(data_dirs))
			data_save_file.attrs["training_sample_count"] = len(x_train)
			data_save_file.attrs["validation_sample_count"] = len(x_val)
			data_save_file.attrs["testing_sample_count"] = len(x_test)

			# training set
			x_train_ds = data_save_file.create_dataset("x_train",
				                                       shape=(len(x_train), 8, 3, self.freq_points, self.time_points),
				                                       dtype="f")
			y_train_ds = data_save_file.create_dataset("y_train",
				                                       shape=(len(y_train), len(data_dirs)),
				                                       dtype="i")
			x_train_ds[:] = x_train
			y_train_ds[:] = y_train

			# validation set
			x_val_ds = data_save_file.create_dataset("x_val",
				                                     shape=(len(x_val), 8, 3, self.freq_points, self.time_points),
				                                     dtype="f")
			y_val_ds = data_save_file.create_dataset("y_val",
				                                     shape=(len(y_val), len(data_dirs)),
				                                     dtype="i")
			x_val_ds[:] = x_val
			y_val_ds[:] = y_val

			# testing set
			x_test_ds = data_save_file.create_dataset("x_test",
				                                      shape=(len(x_test), 8, 3, self.freq_points, self.time_points),
				                                      dtype="f")
			y_test_ds = data_save_file.create_dataset("y_test",
				                                      shape=(len(y_test), len(data_dirs)),
				                                      dtype="i")
			x_test_ds[:] = x_test
			y_test_ds[:] = y_test

	def replicate_augment_data(self, X_h, Y_h):
		""" Replicates/augments sample data
		"""
		# samples after replication/augmentation
		X = []
		Y = []

		# spectrogram formatting function
		format_spec = lambda spectrogram: np.array([spectrogram]*3)

		# replicate/augment samples
		for x_h, y_h in zip(X_h, Y_h):
			# increase sample data via image augmentation
			if self.augmentation:
				for _ in xrange(self.samples_generated_per_sample):
					x = [] # new sample

					for spectrogram in x_h:
						spectrogram = spectrogram[0] # first colour channel only
						shifted_sample = random_shift(np.array([spectrogram]),
							                          wrg = self.augmentation_magnitude, 
							                          hrg = self.augmentation_magnitude)
						x.append(format_spec(shifted_sample[0]))

					X.append(np.array(x))
					Y.append(y_h)

			# increase sample data via image replication
			else:
				for _ in xrange(self.samples_generated_per_sample):
					X.append(x_h)
					Y.append(y_h)

		# replicated/augmented samples
		return np.array(X), np.array(Y)

class ProgressDisplay(Callback):
	""" Progress display callback
	"""
	def on_batch_end(self, epoch, logs={}):
		""" Displays metric values at the end of each batch
		"""
		print "    Batch {0:<4d} => Accuracy: {1:>8.4f} | Loss: {2:>8.4f} | Size: {3:>4d}".format(int(logs["batch"])+1,
																					              float(logs["categorical_accuracy"]),
																					              float(logs["loss"]),
																					              int(logs["size"]))

# auxilliary functions
def get_training_configuration(training_config_fn):
	""" Acquires training configuration data from a training configuration file
	"""
	with open(training_config_fn, "r") as training_config_file:
		training_config = load(training_config_file)
	return training_config

def execute_training_runs(training_config):
	""" Executes training runs from a specified training configuration
	"""
	results = {}
	# iterate through training runs
	for training_run in training_config:
		run_name = training_run["run_name"]
		trials = training_run["trials"]
		data_save_fn = training_run["data_save_fn"]
		validation_ratio = training_run["validation_ratio"]
		testing_ratio = training_run["testing_ratio"]
		samples_generated_per_sample = training_run["samples_generated_per_sample"]
		augmentation = training_run["augmentation"]
		augmentation_magnitude = training_run["augmentation_magnitude"]
		freq_points = training_run["freq_points"]
		time_points = training_run["time_points"]
		espa_save_fn = training_run["espa_save_fn"]
		espa_weights_save_fn = training_run["espa_weights_save_fn"]

		results[run_name] = {}
		print "\n".join(["="*80,
			             "EXECUTING TRAINING RUN: {0}".format(run_name),
			             "="*80])

		# iterate through trials
		for trial in xrange(trials):
			print "\n".join(["-"*80,
				             "TRIAL: {0}".format(trial),
			                 "-"*80])

			espa = ESPAModel(data_save_fn = data_save_fn,
							 validation_ratio = validation_ratio,
							 testing_ratio = testing_ratio,
							 samples_generated_per_sample = samples_generated_per_sample,
							 augmentation = augmentation,
							 augmentation_magnitude = augmentation_magnitude,
							 freq_points = freq_points,
							 time_points = time_points,
							 espa_save_fn = espa_save_fn,
						     espa_weights_save_fn = espa_weights_save_fn)
			espa.process_data()
			espa.generate_espa_model()
			espa.print_espa_summary()
			training_metrics_history = espa.train_espa_model()
			testing_metrics = espa.test_espa_model()
			espa.save_espa_model()

			# organize results
			results[run_name][trial] = {}
			results[run_name][trial]["train"] = training_metrics_history.history
			results[run_name][trial]["test"] = testing_metrics

	return results

def save_results(results, results_save_fn):
	""" Save compiled results data to a JSON file
	"""
	with open(results_save_fn, "w") as results_save_file:
		dump(results, results_save_file)

if __name__ == "__main__":
	training_config = get_training_configuration("training_config.json")
	results = execute_training_runs(training_config)
	save_results(results, "results/results.json")