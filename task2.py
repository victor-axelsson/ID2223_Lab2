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

# Weights initialised with small random values between -0.2 and +0.2
layer1_size = 200
layer2_size = 100
layer3_size = 60
layer4_size = 30
layer5_size = 10

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
## Sigmoid and softmax
## ReLu and softmax
#
# With sigmoid:
# Loss => 0.123101
# Accuracy => 0.9717
#
# With relu:
# Loss => 0.155508
# Accuracy => 0.978
#
XX = tf.reshape(X, [-1, 784])   
Y1 = tf.nn.relu(tf.matmul(XX, W1) + B1)
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits) ## <- always softmax


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Ylogits))

# 4. Define the accuracy
correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 5. Define an optimizer
train_step_gd = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
train_step_ao = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)

# initialize
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# 6. Train and test the model, store the accuracy and loss per iteration
iterations = 10000
y_acc = []
y_loss = []
#train_step = train_step_gd
train_step = train_step_ao
acc = 0
loss = 0

for x in range(iterations):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={XX: batch_xs, Y_: batch_ys})
	acc = sess.run(accuracy, feed_dict={XX: mnist.test.images, Y_: mnist.test.labels})
	loss = sess.run(cross_entropy, feed_dict={XX: mnist.test.images, Y_: mnist.test.labels})
	y_acc.append(acc)
	y_loss.append(loss)

	# print every 100 iterations
	if x % 100 == 0:
		print("[ITERATTIONS] => " + str(x) + "/" + str(iterations))


print("Loss => " +  str(loss))
print("Accuracy => " +  str(acc))

plt.xlabel("Iterations")
plt.ylabel("Accuracy/Cross Entropy")
plt.title("Accuracy over iterations")

plt.plot(range(0, iterations), y_acc, label='Accuracy')
plt.plot(range(0, iterations), y_loss, label='Loss')
plt.legend()
plt.show()

# 7a. If you are using Python/Docker, plot and visualise the accuracy and loss
# 7b. If you are using hops.site, write to Tensorboard logs, and visualize using Tensorboard
