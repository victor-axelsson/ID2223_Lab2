# all tensorflow api is accessible through this
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import helper

# load data
mnist = helper.getMnist()

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

Y2 = tf.nn.relu(tf.matmul(Y1d, W2) + B2)
Y2d = tf.nn.dropout(Y2, pkeep)

Y3 = tf.nn.relu(tf.matmul(Y2d, W3) + B3)
Y3d = tf.nn.dropout(Y3, pkeep)

Y4 = tf.nn.relu(tf.matmul(Y3d, W4) + B4)
Y4d = tf.nn.dropout(Y4, pkeep)

Ylogits = tf.matmul(Y4d, W5) + B5
Y = tf.nn.softmax(Ylogits) ## <- always softmax

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Ylogits))

# 4. Define the accuracy
correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 5. Define an optimizer
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.005
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 10000, 0.90, staircase=True)

train_step_gd = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
train_step_ao = tf.train.AdamOptimizer(starter_learning_rate).minimize(cross_entropy)
train_step = train_step_ao

def train_iterations(iterations, batch_size, _pkeep):

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

		#Train
		batch_xs, batch_ys = mnist.train.next_batch(batch_size)
		sess.run(train_step, feed_dict={XX: batch_xs, Y_: batch_ys, pkeep: _pkeep, global_step: i})
		
		if(i % 100 == 0):
			#Test
			acc = sess.run(accuracy, feed_dict={XX: mnist.test.images, Y_: mnist.test.labels, pkeep: 1})
			loss = sess.run(cross_entropy, feed_dict={XX: mnist.test.images, Y_: mnist.test.labels, pkeep: 1})
			
			# For the plot
			y_acc.append(acc)
			y_loss.append(loss)
			x_axis.append(i )
			print("[ITERATIONS] => " + str(i) + "/" + str(iterations))

	return (acc, loss, y_acc, y_loss, x_axis)

acc, loss, y_acc, y_loss, x_axis = train_iterations(50000, 100, 0.9)

print("Loss => " +  str(loss))
print("Accuracy => " +  str(acc))

helper.plotLossAndAccuracy(x_axis, y_loss, x_axis, y_acc)


