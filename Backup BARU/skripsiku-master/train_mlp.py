import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from math import sqrt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

feature_dimension = 14
output_dimension = 2
data_length = 1300
batch_size = 128
k_fold = KFold(n_splits=5, shuffle=True)
learning_rate = 0.01
epoch = 200
opt_optimizer = 'adam'
result_filename = 'mlp_result_' + str(epoch)

class Model:
	def __init__(self, data, target, input_dimension):
		hidden_node_1 = int(sqrt((output_dimension+2)*input_dimension) + 2*sqrt(input_dimension/(output_dimension+2)))
		hidden_node_2 = int(output_dimension*sqrt(input_dimension/(output_dimension+2)))

		x = data
		y = target
		hidden_1 = tf.layers.dense(inputs=x, units=hidden_node_1, activation=tf.nn.relu)
		dropout_1 = tf.layers.dropout(hidden_1, rate=0.5)
		hidden_2 = tf.layers.dense(inputs=dropout_1, units=hidden_node_2, activation=tf.nn.relu)
		dropout_2 = tf.layers.dropout(hidden_2, rate=0.5)
		logits = tf.layers.dense(inputs=dropout_2, units=output_dimension)

		prediction = tf.nn.softmax(logits)

		losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)

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

	input_dimension = feature_dimension * timesteps
	x = tf.placeholder(tf.float32, [None, input_dimension], name='input_placeholder')
	y = tf.placeholder(tf.float32, [None, output_dimension], name='labels_placeholder')
	model = Model(x, y, input_dimension)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())

		# TRAIN USING K-FOLD CROSS VALIDATION
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
					inp, out = train_input_kfold[ptr:ptr+batch_size], train_output_kfold[ptr:ptr		+batch_size]
					inp = inp.reshape(batch_size, input_dimension)
					ptr+=batch_size
					_, training_cost = sess.run([model.optimize, model.cost], 
						feed_dict={
							x: inp,
							y: out
						})
					train_loss.append(training_cost)
				
				val_input_kfold = val_input_kfold.reshape(len(val_indices), input_dimension)
				accuracy_, precision_, recall_, cost_ = sess.run([model.accuracy, model.precision, model.recall, model.cost], 
					feed_dict={
						x: val_input_kfold, 
						y: val_output_kfold
					})

				train_losses.append(np.mean(train_loss))
				train_loss = []
				val_accuracy.append(accuracy_[0]*100)
				val_losses.append(cost_)
				val_precision.append(precision_[0])
				val_recall.append(recall_[0])
		
			val_accuracy = np.asarray(val_accuracy)

			total_acc.append(val_accuracy[-1])
			total_train_losses.append(train_losses[-1])
			total_val_losses.append(val_losses[-1])
			total_val_precision.append(val_precision[-1])
			total_val_recall.append(val_recall[-1])
		
		if(save):
			checkpoint_dir = "./checkpoint/mlp/timesteps-" + str(timesteps)
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
	plt.yticks([i for i in range(40, 91, 10)])
	plt.ylabel("Accuracy")
	plt.xticks([i for i in range(2, 61, 2)])
	plt.xlabel("Time Window (days)")
	plt.legend()
	plt.show()

main()