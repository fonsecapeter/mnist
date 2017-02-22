# https://www.tensorflow.org/get_started/mnist/beginners
# virtualenv -p /usr/local/bin/python3.6 venv
# goal is to read a picture of a hand-drawn number and
# guess what it is
from tensorflow.examples.tutorials.mnist import input_data

# mnist is all the data
#   mnist.train: 55_000 points
#   mnist.test: 10_000 points
#   mnist.validation: 5_000 points
#
# mnist.train.images are the drawings (28x28px) or an array flattened
# into a vector of 28x28 = 784 (-dimensional vector space) (each point
# being 0-1 representation of greyscale)
#   => so mnist.train.images is a tensor (n-dimensional array) of shape
#      [55_000, 784]
#
# mnist.train.labes are the gold-standards. Each is a on-hot vector
# description of the actual number drawn in the corresponding image
# (by array idx): ie numbers are 0-9, represented as an array.
#   => so 3 would be represented as [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
#      or a tensor of shape [55_000, 10]
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# softmax will assign prob of being one of many things
# ie im 80% sure this is an 8, 4% sure its a 9, ...
# 2 steps: 1. add up evidecne of input being in certain classes
#          2. convert evidence to probabilites
#   => outputs a list of values between 0 and 1 that add up to 1
#
# here we have a weighted sum of pixel intensities, (-) if high intensity
# is evidence against image being that class, (+) if in favor
#
# add extra evidence called bias: y = softmax(Wx + b)
#
# tensorflow does its maths outside of python, but, instead of just
# exiting, doing math, & re-entering py for every computatoin,
# we define one operation independent from py by describing a graph
# of interacting operations to run
import tensorflow as tf
x = tf.placeholder(tf.float32, [None, 784])
# placeholder: we'll input it when we ask tf to comput
# None means any dimensional length

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# Variable is a modifiable tensor that lives in the tf graph
# generally the model parameters
# here both are tensors full of zeroes, since we will learn W and b,
# their initial values don't super matter

# implement the model
y = tf.nn.softmax(tf.matmul(x, W) + b)
# y = softmax(Wx + b)

# -------------------------------------------------------------------------------
# now we train our model

# then we add a new placeholder to input the correct answers before
# we can check the cross-entropy (measure of how bad the model is)
y_ = tf.placeholder(tf.float32, [None, 10])

# implement cross entropy function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# ask tf to minimize this loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# launch the model in an interactive session
sess = tf.InteractiveSession()

# interactive session must explicitly init vars
tf.initialize_all_variables().run()
# train 1_000 times
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# -------------------------------------------------------------------------------
# lets evaluate good our model is
# tf.argmax(y, 1) is the label our model thinks is most likely for each input
# tf.argmax(y_, 1) is the correct label.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# gives us a list of booleans, find fraction of correct by casting floats and
# taking the mean
# ex [True, False, True, True] -> [1, 0, 1, 1] -> 0.75

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# display the accuracy
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
# => 0.9205 aka 92.5%
