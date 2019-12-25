import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

input_dimension = 14
output_dimension = 2
data_length = 1300
batch_size = 128
k_fold = KFold(n_splits=5, shuffle=True)
state_size = 20
learning_rate = 0.01
epoch = 200
opt_optimizer = 'adam'
result_filename = 'rnn_result_' + str(epoch)

class Model:
	def __init__(self, data, target, initial_state, timesteps):
		x = data
		y = target
		init_state = initial_state

		cell = tf.contrib.rnn.BasicRNNCell(state_size, activation=tf.nn.relu)
		cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=0.5)
		val, state = tf.nn.dynamic_rnn(cell, x, initial_state=init_state, dtype=tf.float32)

		val = tf.transpose(val, [1, 0, 2])
		rnn_last = tf.gather(val, timesteps - 1)

		weight = tf.Variable(tf.truncated_normal([state_size, output_dimension]))
		bias = tf.Variable(tf.zeros(output_dimension))

		logits = tf.matmul(rnn_last, weight) + bias
		prediction = tf.nn.softmax(logits)

		losses = tf.losses.softmax_cross_entropy(onehot_labels=y ,logits=logits)

		self._cost =  tf.reduce_mean(losses)
		
		if(opt_optimizer == "adam"):
			self._optimize = tf.train.AdamOptimizer(learning_rate).minimize(self._cost)
		elif(opt_optimizer == "adadelta"):
			self._optimize = tf.train.AdadeltaOptimizer(learning_rate).minimize(self._cost)
		elif(opt_optimizer == "rmsprop"):
			self._optimize = tf.train.RMSPropOptimizer(learning_rate).minimize(self._cost)
		else:
			self._optimize = tf.train.GradientDescentOptimizer(learning_rate).minimize(self._cost)

		self._recall = tf.metrics.recall(tf.argmax(y, 1), tf.argmax(prediction, 1))
		self._precision = tf.metrics.precision(tf.argmax(y, 1), tf.argmax(prediction, 1))
		self._accuracy = tf.metrics.accuracy(tf.argmax(y, 1), tf.argmax(prediction, 1))

	@property
	def cost(self):
		return self._cost

	@property
	def optimize(self):
		return self._optimize

	@property
	def recall(self):
		return self._recall

	@property
	def precision(self):
		return self._precision

	@property
	def accuracy(self):
		return self._accuracy

def visualize_loss(total_losses, timesteps):
	plt.plot([z for z in range(2, 61, 2)], total_losses, label="Average Loss - Timesteps " + str(timesteps))
	print("Average Train Losses for with learning rate " + str(learning_rate) + " -> " + str(np.mean(total_losses)))
	plt.xticks([i for i in range(2, 61, 2)])
	plt.xlabel("Time Window (days)")

def visualize_val_loss(total_val_losses, timesteps):
	plt.plot([z for z in range(2, 61, 2)], total_val_losses, label="Average Valid Loss - Timesteps " + str(timesteps))
	print("Average Validation Losses for with learning rate " + str(learning_rate) + " -> " + str(np.mean(total_val_losses)))
	plt.xticks([i for i in range(2, 61, 2)])
	plt.xlabel("Time Window (days)")

def prepare_train_data(days_predict, timesteps):
	df_norm = pd.read_csv("dataset-normalize.csv").drop('timestamp', axis=1).as_matrix()[1:]
	df_output = pd.read_csv("labels/" + str(days_predict) + "days.csv").as_matrix()
	train_input = []
	train_output = []
	for i in range(0, data_length):
		train_input.append(df_norm[i:i+timesteps])
		train_output.append(df_output[i+timesteps])

	return np.asarray(train_input), np.asarray(train_output)

def train_network(train_input, train_output, days_predict, timesteps, save=True):
	total_acc = []
	total_train_losses = []
	total_val_losses = []
	total_val_precision = []
	total_val_recall = []

	train_losses_by_epoch = []
	val_losses_by_epoch = []

	tf.reset_default_graph()
	x = tf.placeholder(tf.float32, [None, None, input_dimension], name='input_placeholder')
	y = tf.placeholder(tf.float32, [None, output_dimension], name='labels_placeholder')
	init_state = tf.placeholder(tf.float32, [None, state_size], name='state_placeholder')
	model = Model(x, y, init_state, timesteps)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())

		for train_indices, val_indices in k_fold.split(train_input, train_output):
			train_input_kfold = train_input[train_indices]
			train_output_kfold = train_output[train_indices]

			val_input_kfold = train_input[val_indices]
			val_output_kfold = train_output[val_indices]

			no_of_batches = int(len(train_input_kfold)/batch_size)
			
			train_loss = []
			train_losses = []
			val_accuracy = []
			val_losses = []
			val_precision = []
			val_recall = []

			for i in range(epoch):
				epoch_step = i + 1
				ptr = 0
				for j in range(no_of_batches):
					inp, out = train_input_kfold[ptr:ptr+batch_size], train_output_kfold[ptr:ptr+batch_size]
					ptr+=batch_size
					rnn_init_weight = np.eye(batch_size, state_size)
					training_step, training_cost = sess.run([model.optimize, model.cost], 
						feed_dict={
							x: inp,
							y: out,
							init_state: rnn_init_weight
						})
					train_loss.append(training_cost)
				
				rnn_init_weight_validation = np.eye(len(val_input_kfold), state_size)
				accuracy_, precision_, recall_, cost_ = sess.run([model.accuracy, model.precision, model.recall, model.cost], 
					feed_dict={
						x: val_input_kfold, 
						y: val_output_kfold,
						init_state: rnn_init_weight_validation
					})

				train_losses.append(np.mean(train_loss))
				val_losses.append(cost_)
				val_accuracy.append(accuracy_[0]*100)
				val_precision.append(precision_[0])
				val_recall.append(recall_[0])
				train_loss = []

			val_accuracy = np.asarray(val_accuracy)

			total_acc.append(val_accuracy[-1])
			total_train_losses.append(train_losses[-1])
			total_val_losses.append(val_losses[-1])
			total_val_precision.append(val_precision[-1])
			total_val_recall.append(val_recall[-1])
		
		if(save):
			checkpoint_dir = "./checkpoint/rnn/timesteps-" + str(timesteps)
			if not os.path.exists(checkpoint_dir):
				os.makedirs(checkpoint_dir)
			checkpoint_dest = checkpoint_dir + '/' + 'days-' + str(days_predict) + '/model'
			tf.train.Saver().save(sess, checkpoint_dest)

		f_acc = np.mean(total_acc)
		f_train_losses = np.mean(total_train_losses)
		f_val_losses = np.mean(total_val_losses)
		f_precision = np.mean(total_val_precision)
		f_recall = np.mean(total_val_recall)
		f_fscore = 2*((f_precision*f_recall)/(f_precision+f_recall))

		# with open(result_filename, 'a') as f:
		# 	print("------------ Day", days_predict, "• Timesteps", timesteps, "------------", file=f)
		# 	print("Average Acc {:.2f}".format(f_acc), file=f)
		# 	print("Average Train Loss", f_train_losses, file=f)
		# 	print("Average Validation Loss", f_val_losses, file=f)
		# 	print("Precision", f_precision, file=f)
		# 	print("Recall", f_recall, file=f)
		# 	print("F-Measure", f_fscore, file=f)
		# 	print("", file=f)

		print("------------ Day", days_predict, "• Timesteps", timesteps, "------------")
		print("Average Acc {:.2f}".format(f_acc))
		print("Average Train Loss", f_train_losses)
		print("Average Validation Loss", f_val_losses)
		print("Precision", f_precision)
		print("Recall", f_recall)
		print("F-Measure", f_fscore)
		print()

		return f_acc, f_precision, f_recall, f_fscore, f_val_losses, f_train_losses

def main():
	top_acc = 0
	top_acc_details = ""

	for i in range(1, 4):
		timesteps = 2 * i + 1

		total_acc = []
		total_losses = []
		total_val_losses = []

		for j in range(2, 61, 2):
			train_input, train_output = prepare_train_data(j, timesteps)
			acc_, precision_, recall_, fscore_, val_losses_, train_losses_ = train_network(train_input=train_input, train_output=train_output, days_predict=j, timesteps=timesteps, save=False)
			
			total_acc.append(acc_)
			total_losses.append(train_losses_)
			total_val_losses.append(val_losses_)

			if(acc_ > top_acc):
				top_acc = acc_
				top_acc_details = "Days " + str(j) + " Timesteps " + str(timesteps)

		plt.plot([z for z in range(2, 61, 2)], total_acc, label="Accuracy - Timesteps " + str(timesteps))

	# with open(result_filename, 'a') as f:
	# 	print(top_acc, top_acc_details, file=f)

	print(top_acc, top_acc_details)
	plt.yticks([i for i in range(40, 81, 10)])
	plt.ylabel("Accuracy")
	plt.xticks([i for i in range(2, 61, 2)])
	plt.xlabel("Time Window (days)")
	plt.legend()
	plt.show()

main()