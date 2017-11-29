# all tensorflow api is accessible through this
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import helper

def main(job_id, learning_rate, dropout):

	print("Starting up job: " + str(job_id))

	# load data
	mnist = helper.getMnist()

	def bias_variable(shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)
	
	# 1. Define Variables and Placeholders
	X = tf.placeholder(tf.float32, [None, 784])
	XX = tf.reshape(X, [-1, 28, 28, 1])
	Y_ = tf.placeholder(tf.float32) 	

	# sizes of the different layers
	layer1_w = 5
	layer1_h = 5
	layer1_d = 1
	layer1_out = 4

	layer2_w = 5
	layer2_h = 5
	layer2_out = 8

	layer3_w = 4
	layer3_h = 4
	layer3_out = 12

	layer4_w = layer3_w * layer3_h * layer2_out
	layer4_h = 1
	layer4_out = 200

	layer5_size = 10

	# placeholder for probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at   À training time
	pkeep = tf.placeholder(tf.float32)

	#define weight variable for a convolutional layer
	W1 = tf.Variable(tf.truncated_normal([layer1_w, layer1_h, layer1_d, layer1_out], stddev=0.1))
	B1 = bias_variable([layer1_out])
	Y1 = tf.nn.max_pool(
		tf.nn.dropout(
			tf.nn.relu(tf.nn.conv2d(XX, W1, strides=[1, 1, 1, 1], padding='SAME') + B1), 
			pkeep),
		ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')
		

	W2 = tf.Variable(tf.truncated_normal([layer2_w, layer2_h, layer1_out, layer2_out], stddev=0.1))
	B2 = bias_variable([layer2_out])
	Y2 = tf.nn.max_pool(
			tf.nn.dropout(
				tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, 2, 2, 1], padding='SAME') + B2),
				pkeep),
			ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	W3 = tf.Variable(tf.truncated_normal([layer3_w, layer3_h, layer2_out, layer3_out], stddev=0.1))
	B3 = bias_variable([layer3_out])
	Y3 = tf.nn.max_pool(
			tf.nn.dropout(
				tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, 2, 2, 1], padding='SAME') + B3),
				pkeep),
			ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	#	Densly connected layer #
	# Reshape output of layer 3 to a flat array of size layer4_w
	YY3 = tf.reshape(Y3, [-1, Y3.shape[1] * Y3.shape[2] * Y3.shape[3]])  
	W4 = tf.Variable(tf.truncated_normal([int(Y3.shape[1] * Y3.shape[2] * Y3.shape[3]), layer4_out], stddev=0.1))
	B4 = bias_variable([layer4_out])
	Y4 = tf.nn.relu(tf.matmul(YY3, W4) + B4)

	## Redout layer ##
	W5 = tf.Variable(tf.truncated_normal([layer4_out, layer5_size], stddev=0.1))
	B5 = tf.Variable(tf.zeros([layer5_size]))

	Ylogits = tf.matmul(Y4, W5) + B5
	Y = tf.nn.softmax(Ylogits) ## <- always softmax

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Ylogits))

	# 4. Define the accuracy
	correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(Y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# 5. Define an optimizer
	#global_step = tf.Variable(0, trainable=False)
	#starter_learning_rate = 0.005
	#learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 10000, 0.90, staircase=True)

	#train_step_gd = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
	train_step_gd = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
	train_step_ao = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

	####
	# Task 1, Couldn't even get propper results 
	# Task 2, Adam optimizer. 100 iterations. Fixed learning rate of 0,005: 
	# Loss => 0.249494
	# Accuracy => 0.9266
	#
	def training_step(_pkeep, _iterations, batch_size):
		# initialize
		sess = tf.InteractiveSession()
		tf.global_variables_initializer().run()

		# 6. Train and test the model, store the accuracy and loss per iteration
		y_acc = []
		y_loss = []
		x_axis = []
		train_step = train_step_ao
		#train_step = train_step_ao
		acc = 0
		loss = 0

		for i in range(_iterations):
			
			#Train
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			sess.run(train_step, feed_dict={X: batch_xs, Y_: batch_ys, pkeep: _pkeep})

			# print every 100 iterations
			if(i % 100 == 0):
				#Test
				acc = sess.run(accuracy, feed_dict={X: mnist.test.images, Y_: mnist.test.labels, pkeep: 1})
				loss = sess.run(cross_entropy, feed_dict={X: mnist.test.images, Y_: mnist.test.labels, pkeep: 1})
				
				#save data for the graphs
				y_acc.append(acc)
				y_loss.append(loss)
				x_axis.append(i)

				if(i % 1000 == 0):
					print("[ITERATTIONS] => " + str(i) + "/" + str(_iterations))

		return (acc, loss, y_acc, y_loss, x_axis)

	return training_step(dropout, 50000, 100)

'''
Job number	Learning rate	Dropout
1			0.001			0.45
2			0.001			0.9
3			0.0005			0.45
4			0.0005			0.7
5			0.0001			0.45
6			0.0001			0.7
'''
acc1, loss1, y_acc1, y_loss1, x_axis1 = main(1, 0.001, 0.45)
print("Loss1 => " +  str(loss1))
print("Accuracy1 => " +  str(acc1))
acc2, loss2, y_acc2, y_loss2, x_axis2 = main(2, 0.001, 0.9)
print("Loss2 => " +  str(loss2))
print("Accuracy2 => " +  str(acc2))
acc3, loss3, y_acc3, y_loss3, x_axis3 = main(3, 0.0005, 0.45)
print("Loss3 => " +  str(loss3))
print("Accuracy3 => " +  str(acc3))
acc4, loss4, y_acc4, y_loss4, x_axis4 = main(4, 0.0005, 0.7)
print("Loss4 => " +  str(loss4))
print("Accuracy4 => " +  str(acc4))
acc5, loss5, y_acc5, y_loss5, x_axis5 = main(5, 0.0001, 0.45)
print("Loss5 => " +  str(loss5))
print("Accuracy5 => " +  str(acc5))
acc6, loss6, y_acc6, y_loss6, x_axis6 = main(6, 0.0001, 0.7)
print("Loss6 => " +  str(loss6))
print("Accuracy6 => " +  str(acc6))

helper.plotLossAndAccuracy(x_axis1, y_loss1, x_axis1, y_acc1, False, "JobNR1")
helper.plotLossAndAccuracy(x_axis2, y_loss2, x_axis2, y_acc2, False, "JobNR2")
helper.plotLossAndAccuracy(x_axis3, y_loss3, x_axis3, y_acc3, False, "JobNR3")
helper.plotLossAndAccuracy(x_axis4, y_loss4, x_axis4, y_acc4, False, "JobNR4")
helper.plotLossAndAccuracy(x_axis5, y_loss5, x_axis5, y_acc5, False, "JobNR5")
helper.plotLossAndAccuracy(x_axis6, y_loss6, x_axis6, y_acc6, True, "JobNR6")
