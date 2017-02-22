import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# tf parameters
learning_rate = 0.01
training_iteration = 30
batch_size = 100
display_step = 2
dir_path = os.path.dirname(os.path.realpath(__file__))
write_path = os.path.join(dir_path, 'summary')

# tf graph input
x = tf.placeholder('float', [None, 784])  # mnist data image of shape 28*28px=784
y = tf.placeholder('float', [None, 10])  # 0-9 digits recognition => 10 classes (one-hot)

# initial model weights, will be overwritten in training
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

with tf.name_scope('Wx_b') as scope:
    # construct linear model
    model = tf.nn.softmax(tf.matmul(x, W) + b)  # softmax approach

# add summary ops to collect data
w_h = tf.summary.histogram('weights', W)
b_h = tf.summary.histogram('biases', b)

with tf.name_scope('cost_function'):
    # minimize error using cross-entropy
    cost_function = -tf.reduce_sum(y * tf.log(model))
    # create summary to monitor the cost function
    cf_s = tf.summary.scalar('cost_function', cost_function)

with tf.name_scope('train') as scope:
    # gradient descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

init = tf.initialize_all_variables()

# merge all summaries into single operator
merged_summary_op = tf.summary.merge([w_h, b_h, cf_s])

# launch graph
with tf.Session() as sess:
    sess.run(init)

    summary_writer = tf.train.SummaryWriter(write_path, graph_def=sess.graph_def)
    
    # train cycle
    for iteration in range(training_iteration):
        avg_cost = 0.
        total_batch = int(mnist.train.num_example / batch_size)
        # loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # fit training using batch data
            sess.run(optimizer, feed_dict={x: batch, y: batch_ys})
            summary_writer.add_summary(summary_str, iteration * total_batch + i)
        # display logs for each iteration step
        if iteration % display_step == 0:
            print('Iteration: %04d cost = %d' % (iteration + 1, avg_cost))

    print('---------------------------')
    print('tuning completed')

    # test the model
    predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    # calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(predictions, 'float'))
    print('accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

