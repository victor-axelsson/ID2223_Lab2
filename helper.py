# all tensorflow api is accessible through this
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

def getMnist():
	return input_data.read_data_sets('data/fashion', source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/', one_hot=True)

def train_epochs(iterations, batch_size, mnist, train_step, accuracy, cross_entropy):

	# initialize
	init = tf.initialize_all_variables() 
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()
	
	y_acc = []
	y_loss = []
	x_axis = []

	acc = 0
	loss = 0
	for epoch in range(iterations):

		for i in range(int(mnist.train.num_examples / batch_size)):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			sess.run(train_step, feed_dict={XX: batch_xs, Y_: batch_ys})
		
		acc = sess.run(accuracy, feed_dict={XX: mnist.test.images, Y_: mnist.test.labels})
		loss = sess.run(cross_entropy, feed_dict={XX: mnist.test.images, Y_: mnist.test.labels})
		y_acc.append(acc)
		y_loss.append(loss)
		x_axis.append(epoch )
		print("[EPOCHS] => " + str(epoch) + "/" + str(iterations))

	return (acc, loss, y_acc, y_loss, x_axis)


def train_iterations(iterations, batch_size, mnist, train_step, accuracy, cross_entropy):

	# initialize
	init = tf.initialize_all_variables() 
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()
	
	acc = 0
	loss = 0
	y_acc = []
	y_loss = []
	x_axis = []

	for i in range(iterations):

		batch_xs, batch_ys = mnist.train.next_batch(batch_size)
		sess.run(train_step, feed_dict={XX: batch_xs, Y_: batch_ys})
		
		if(i % 100 == 0):
			acc = sess.run(accuracy, feed_dict={XX: mnist.test.images, Y_: mnist.test.labels})
			loss = sess.run(cross_entropy, feed_dict={XX: mnist.test.images, Y_: mnist.test.labels})
			y_acc.append(acc)
			y_loss.append(loss)
			x_axis.append(i)
			print("[ITERATIONS] => " + str(i) + "/" + str(iterations))

	return (acc, loss, y_acc, y_loss, x_axis)

def plotLossAndAccuracy(x_axis_loss, y_axis_loss, x_axis_acc, y_axis_acc):
	plt.xlabel("Iterations")
	plt.ylabel("Accuracy/Cross Entropy")
	plt.title("Accuracy over iterations")

	plt.plot(x_axis_loss, y_axis_loss, label='Loss')
	plt.plot(x_axis_acc, y_axis_acc, label='Accuracy')
	plt.legend()
	plt.show()
