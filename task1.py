# all tensorflow api is accessible through this
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# load data
mnist = input_data.read_data_sets('input/fashion', one_hot=True)

# 1. Define Variables and Placeholders
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32) 					# correct answers(labels)
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
train_step_ao = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)

# initialize
init = tf.initialize_all_variables() 
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# 6. Train and test the model, store the accuracy and loss per iteration
iterations = 1000
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

	#sess.run(init)

# 7a. If you are using Python/Docker, plot and visualise the accuracy and loss
# 7b. If you are using hops.site, write to Tensorboard logs, and visualize using Tensorboard
