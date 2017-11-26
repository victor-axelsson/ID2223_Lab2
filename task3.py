# all tensorflow api is accessible through this
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# load data
mnist = input_data.read_data_sets('input/fashion', one_hot=True)

# 1. Define Variables and Placeholders
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32) 	

# sizes of the different layers
layer1_size = 200
layer2_size = 100
layer3_size = 60
layer4_size = 30
layer5_size = 10

#learning rate placeholder
lr = tf.placeholder(tf.float32)

# placeholder for probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at   À training time
pkeep = tf.placeholder(tf.float32)

#Layer one
W1 = tf.Variable(tf.truncated_normal([784, layer1_size], stddev=0.1))
B1 = tf.Variable(tf.zeros([layer1_size]))

#Layer two
W2 = tf.Variable(tf.truncated_normal([layer1_size, layer2_size], stddev=0.1))
B2 = tf.Variable(tf.zeros([layer2_size]))

#Layer three
W3 = tf.Variable(tf.truncated_normal([layer2_size, layer3_size], stddev=0.1))
B3 = tf.Variable(tf.zeros([layer3_size]))

#Layer four
W4 = tf.Variable(tf.truncated_normal([layer3_size, layer4_size], stddev=0.1))
B4 = tf.Variable(tf.zeros([layer4_size]))

#Layer five
W5 = tf.Variable(tf.truncated_normal([layer4_size, layer5_size], stddev=0.1))
B5 = tf.Variable(tf.zeros([layer5_size]))

#Define the model
XX = tf.reshape(X, [-1, 784])   
Y1 = tf.nn.relu(tf.matmul(XX, W1) + B1)
Y1d = tf.nn.dropout(Y1, pkeep)

Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
Y2d = tf.nn.dropout(Y2, pkeep)

Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
Y3d = tf.nn.dropout(Y3, pkeep)

Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
Y4d = tf.nn.dropout(Y4, pkeep)

Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits) ## <- always softmax

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Ylogits))

# 4. Define the accuracy
correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 5. Define an optimizer
# For starter_learning_rate = 0.7 and 0.96
# Loss => 0.104842
# Accuracy => 0.9791
#
# starter_learning_rate = 0.6, 0.90
# Loss => 0.0996504
# Accuracy => 0.9795
#
# starter_learning_rate = 0.1, 0.90
# Loss => 0.0930191
# Accuracy => 0.9798
#
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.1
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 10000, 0.90, staircase=True)

train_step_gd = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
train_step_ao = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)


def training_step(_pkeep, _lr, iterations):

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

	for epoch in range(iterations):
		
		#Train
		batch_xs, batch_ys = mnist.train.next_batch(100)
		sess.run(train_step, feed_dict={XX: batch_xs, Y_: batch_ys, pkeep: _pkeep, global_step: epoch})

		#Test
		acc = sess.run(accuracy, feed_dict={XX: mnist.test.images, Y_: mnist.test.labels, pkeep: 1})
		loss = sess.run(cross_entropy, feed_dict={XX: mnist.test.images, Y_: mnist.test.labels, pkeep: 1})
		
		#save data for the graphs
		y_acc.append(acc)
		y_loss.append(loss)

		# print every 100 iterations
		if epoch % 100 == 0:
			print("[ITERATTIONS] => " + str(epoch) + "/" + str(iterations))

	print("Loss => " +  str(loss))
	print("Accuracy => " +  str(acc))

	plt.xlabel("Iterations")
	plt.ylabel("Accuracy/Cross Entropy")
	plt.title("Accuracy over iterations")

	plt.plot(range(0, iterations), y_acc, label='Accuracy')
	plt.plot(range(0, iterations), y_loss, label='Loss')
	plt.legend()
	plt.show()

training_step(0.75, 0.02, 10000)


