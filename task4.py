# all tensorflow api is accessible through this
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# load data
mnist = input_data.read_data_sets('input/fashion', one_hot=True)

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

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

iterations = 200

#learning rate placeholder
lr = tf.placeholder(tf.float32)

# placeholder for probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at   À training time
pkeep = tf.placeholder(tf.float32)

#define weight variable for a convolutional layer
W1 = tf.Variable(tf.truncated_normal([layer1_w, layer1_h, layer1_d, layer1_out], stddev=0.1))
B1 = bias_variable([layer1_out])
Y1 = tf.nn.max_pool(
	tf.nn.relu(tf.nn.conv2d(XX, W1, strides=[1, 1, 1, 1], padding='SAME') + B1), 
	ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')


W2 = tf.Variable(tf.truncated_normal([layer2_w, layer2_h, layer1_out, layer2_out], stddev=0.1))
B2 = bias_variable([layer2_out])
Y2 = tf.nn.max_pool(
	tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, 2, 2, 1], padding='SAME') + B2),
	ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W3 = tf.Variable(tf.truncated_normal([layer3_w, layer3_h, layer2_out, layer3_out], stddev=0.1))
B3 = bias_variable([layer3_out])
Y3 = tf.nn.max_pool(
	tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, 2, 2, 1], padding='SAME') + B3),
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
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.6
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, iterations, 0.90, staircase=True)

train_step_gd = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
train_step_ao = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)


def training_step(_pkeep, _iterations):

	# initialize
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	# 6. Train and test the model, store the accuracy and loss per iteration
	y_acc = []
	y_loss = []
	#train_step = train_step_gd
	train_step = train_step_gd
	acc = 0
	loss = 0

	for epoch in range(_iterations):
		
		#Train
		batch_xs, batch_ys = mnist.train.next_batch(100)
		sess.run(train_step, feed_dict={X: batch_xs, Y_: batch_ys, pkeep: _pkeep, global_step: epoch})

		#Test
		acc = sess.run(accuracy, feed_dict={X: mnist.test.images, Y_: mnist.test.labels, pkeep: 1})
		loss = sess.run(cross_entropy, feed_dict={X: mnist.test.images, Y_: mnist.test.labels, pkeep: 1})
		
		#save data for the graphs
		y_acc.append(acc)
		y_loss.append(loss)

		# print every 100 iterations
		if epoch % 1 == 0:
			print("[ITERATTIONS] => " + str(epoch) + "/" + str(_iterations))

	print("Loss => " +  str(loss))
	print("Accuracy => " +  str(acc))

	plt.xlabel("Iterations")
	plt.ylabel("Accuracy/Cross Entropy")
	plt.title("Accuracy over iterations")

	plt.plot(range(0, iterations), y_acc, label='Accuracy')
	plt.plot(range(0, iterations), y_loss, label='Loss')
	plt.legend()
	plt.show()

training_step(0.75, iterations)


