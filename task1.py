# all tensorflow api is accessible through this
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import helper

# load data
#mnist = input_data.read_data_sets('input/fashion', one_hot=True)
mnist = helper.getMnist()

# 1. Define Variables and Placeholders
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32, shape=[None, 10]) 					# correct answers(labels)
W = tf.Variable(tf.zeros([784, 10])) 				# weights W[784, 10] 784=28*28
b = tf.Variable(tf.zeros([10]))						# biases b[10]
XX = tf.reshape(X, [-1, 784])						# flatten the images into a single line of pixels    

# 2. Define the model
Y = tf.nn.softmax(tf.matmul(XX, W) + b)

# 3. Define the loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y_ * tf.log(Y), reduction_indices=[1]))

# 4. Define the accuracy
correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 5. Define an optimizer
train_step_gd = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
train_step_ao = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 6. Train and test the model, store the accuracy and loss per iteration
iterations = 100
y_acc = []
y_loss = []
x_axis = []
#train_step = train_step_gd
train_step = train_step_ao


def train_epochs(iterations, batch_size):

	# initialize
	init = tf.initialize_all_variables() 
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()
	
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

	return (acc, loss)

def train_iterations(iterations, batch_size):

	# initialize
	init = tf.initialize_all_variables() 
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()
	
	acc = 0
	loss = 0
	for i in range(iterations):

		batch_xs, batch_ys = mnist.train.next_batch(batch_size)
		sess.run(train_step, feed_dict={XX: batch_xs, Y_: batch_ys})
		
		if(i % 100 == 0):
			acc = sess.run(accuracy, feed_dict={XX: mnist.test.images, Y_: mnist.test.labels})
			loss = sess.run(cross_entropy, feed_dict={XX: mnist.test.images, Y_: mnist.test.labels})
			y_acc.append(acc)
			y_loss.append(loss)
			x_axis.append(i )
			print("[ITERATIONS] => " + str(i) + "/" + str(iterations))

	return (acc, loss)

acc, loss = train_iterations(20000, 100)

print("Loss => " +  str(loss))
print("Accuracy => " +  str(acc))

helper.plotLossAndAccuracy(x_axis, y_loss, x_axis, y_acc)

# 7a. If you are using Python/Docker, plot and visualise the accuracy and loss
# 7b. If you are using hops.site, write to Tensorboard logs, and visualize using Tensorboard
