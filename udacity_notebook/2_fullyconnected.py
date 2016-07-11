
# coding: utf-8

# Deep Learning
# =============
#
# Assignment 2
# ------------
#
# Previously in `1_notmnist.ipynb`, we created a pickle with formatted datasets for training, development and testing on the [notMNIST dataset](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html).
#
# The goal of this assignment is to progressively train deeper and more accurate models using TensorFlow.

# In[ ]:

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range


# First reload the data we generated in `1_notmnist.ipynb`.

# In[ ]:

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)


# Reformat into a shape that's more adapted to the models we're going to train:
# - data as a flat matrix,
# - labels as float 1-hot encodings.

# In[ ]:

image_size = 28
num_labels = 10

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


# We're first going to train a multinomial logistic regression using simple gradient descent.
#
# TensorFlow works like this:
# * First you describe the computation that you want to see performed: what the inputs, the variables, and the operations look like. These get created as nodes over a computation graph. This description is all contained within the block below:
#
#       with graph.as_default():
#           ...
#
# * Then you can run the operations on this graph as many times as you want by calling `session.run()`, providing it outputs to fetch from the graph that get returned. This runtime operation is all contained in the block below:
#
#       with tf.Session(graph=graph) as session:
#           ...
#
# Let's load all the data into TensorFlow and build the computation graph corresponding to our training:

# In[ ]:

# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.
train_subset = 10000

graph = tf.Graph()
with graph.as_default():

    # Input data.
    # Load the training, validation and test data into constants that are
    # attached to the graph.
    tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
    tf_train_labels = tf.constant(train_labels[:train_subset])
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    # These are the parameters that we are going to be training. The weight
    # matrix will be initialized using random valued following a (truncated)
    # normal distribution. The biases get initialized to zero.
    weights = tf.Variable(
      tf.truncated_normal([image_size * image_size, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    # We multiply the inputs with the weight matrix, and add biases. We compute
    # the softmax and cross-entropy (it's one operation in TensorFlow, because
    # it's very common, and it can be optimized). We take the average of this
    # cross-entropy across all training examples: that's our loss.
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # Optimizer.
    # We are going to find the minimum of this loss using gradient descent.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    # These are not part of training, but merely here so that we can report
    # accuracy figures as we train.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(
      tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)


# Let's run this computation and iterate:

# In[ ]:

num_steps = 801

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

if False:
    with tf.Session(graph=graph) as session:
        # This is a one-time operation which ensures the parameters get initialized as
        # we described in the graph: random weights for the matrix, zeros for the
        # biases.
        tf.initialize_all_variables().run()
        print('Initialized')
        for step in range(num_steps):
            # Run the computations. We tell .run() that we want to run the optimizer,
            # and get the loss value and the training predictions returned as numpy
            # arrays.
            _, l, predictions = session.run([optimizer, loss, train_prediction])
            if (step % 100 == 0):
                print('Loss at step %d: %f' % (step, l))
                print('Training accuracy: %.1f%%' % accuracy(
                  predictions, train_labels[:train_subset, :]))
                # Calling .eval() on valid_prediction is basically like calling run(), but
                # just to get that one numpy array. Note that it recomputes all its graph
                # dependencies.
                print('Validation accuracy: %.1f%%' % accuracy(
                  valid_prediction.eval(), valid_labels))
        print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))


# Let's now switch to stochastic gradient descent training instead, which is much faster.
#
# The graph will be similar, except that instead of holding all the training data into a constant node, we create a `Placeholder` node which will be fed actual data at every call of `session.run()`.

# In[ ]:

batch_size = 128

graph = tf.Graph()
with graph.as_default():

    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    weights = tf.Variable(
      tf.truncated_normal([image_size * image_size, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(
      tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)


# Let's run it:

# In[ ]:

num_steps = 3001

if False:
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print("Initialized")
        for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
            _, l, predictions = session.run(
              [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 500 == 0):
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % accuracy(
                  valid_prediction.eval(), valid_labels))
        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


# ---
# Problem
# -------
#
# Turn the logistic regression example with SGD into a 1-hidden layer neural network with rectified linear units [nn.relu()](https://www.tensorflow.org/versions/r0.7/api_docs/python/nn.html#relu) and 1024 hidden nodes. This model should improve your validation / test accuracy.
#
# ---

# brute-force 800 step
# Test accuracy: 82.7%
#
# stochastic 3k step
# Test accuracy: 86.1%

graph = tf.Graph()
with graph.as_default():
    num_feature = image_size * image_size
    num_hidden_feature = 1024

    X = tf.placeholder(tf.float32, [None, num_feature], name="input")
    Y = tf.placeholder(tf.float32, [None, num_labels], name="output")
    W1 = tf.Variable(tf.truncated_normal([num_feature, num_hidden_feature], stddev=0.1), name="weight1")
    b1 = tf.Variable(tf.zeros([num_hidden_feature]), name="bias1")

    hidden1 = tf.nn.relu(tf.matmul(X, W1) + b1)
#     hidden1 = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
    W2 = tf.Variable(tf.truncated_normal([num_hidden_feature, num_labels], stddev=0.1), name="weight2")
    b2 = tf.Variable(tf.zeros([num_labels]), name="bias2")

    Y_pred = tf.nn.softmax(tf.matmul(hidden1, W2) + b2)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(Y_pred), reduction_indices= 1))

#     gradient_descent_learning_rate = 0.1
#     optimizer = tf.train.GradientDescentOptimizer(gradient_descent_learning_rate).minimize(cross_entropy)
    adam_learning_rate = 0.001;
    optimizer = tf.train.AdamOptimizer(learning_rate=adam_learning_rate).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


batch_size = 100
training_epochs = 2

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("Initialized")
    for epoch in range(training_epochs):
        n_train = train_labels.shape[0]
        n_batch = n_train / batch_size
        random_index = np.random.permutation(n_train)
        for step in range(n_batch):
            start = step * batch_size
            end = (step + 1) * batch_size
            if end > n_train:
                end = n_train

            # Generate a minibatch.
            batch_data = train_dataset[random_index[start:end], :]
            batch_labels = train_labels[random_index[start:end], :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {X : batch_data, Y : batch_labels}
            _, l, a = session.run(
              [optimizer, cross_entropy, accuracy], feed_dict=feed_dict)
            if (step % 500 == 0):
                print("Minibatch loss at step %d: %f" % (step + (epoch * n_batch), l))
                print("Minibatch accuracy: %.1f%%" % (a * 100))
    print("Complete training")
    cross_entropy_value, accuracy_value = session.run([cross_entropy, accuracy], feed_dict={X: train_dataset, Y: train_labels})
    print("Train [CrossEntropy / Training Accuracy] {:05.4f} / {:05.4f}".format(cross_entropy_value, accuracy_value))
    cross_entropy_valid = session.run(cross_entropy, feed_dict={X: valid_dataset, Y: valid_labels})
    accuracy_valid = accuracy.eval({X: valid_dataset, Y: valid_labels})
    print("Validation [CrossEntropy / Training Accuracy] {:05.4f} / {:05.4f}".format(cross_entropy_valid, accuracy_valid))
    cross_entropy_test = session.run(cross_entropy, feed_dict={X: test_dataset, Y: test_labels})
    accuracy_test = accuracy.eval({X: test_dataset, Y: test_labels})
    print("Testing [CrossEntropy / Training Accuracy] {:05.4f} / {:05.4f}".format(cross_entropy_test, accuracy_test))


# relu with AdamOptimizer(0.001)
# training_epochs = 2
# Train [CrossEntropy / Training Accuracy] 0.3432 / 0.9020
# Validation [CrossEntropy / Training Accuracy] 0.4154 / 0.8859
# Testing [CrossEntropy / Training Accuracy] 0.2241 / 0.9412

# sigmoid with GradientDescentOptimizer(0.1)
# Train [CrossEntropy / Training Accuracy] 0.5468 / 0.8438
# Validation [CrossEntropy / Training Accuracy] 0.5441 / 0.8468
# Testing [CrossEntropy / Training Accuracy] 0.3382 / 0.9045

# sigmoid with AdamOptimizer(0.001)
# training_epochs = 1
# Train [CrossEntropy / Training Accuracy] 0.4534 / 0.8695
# Validation [CrossEntropy / Training Accuracy] 0.4593 / 0.8711
# Testing [CrossEntropy / Training Accuracy] 0.2732 / 0.9220
# training_epochs = 2
# Train [CrossEntropy / Training Accuracy] 0.3817 / 0.8875
# Validation [CrossEntropy / Training Accuracy] 0.4145 / 0.8804
# Testing [CrossEntropy / Training Accuracy] 0.2353 / 0.9327
# training_epochs = 3
# Train [CrossEntropy / Training Accuracy] 0.3296 / 0.9022
# Validation [CrossEntropy / Training Accuracy] 0.3866 / 0.8883
# Testing [CrossEntropy / Training Accuracy] 0.2102 / 0.9404
