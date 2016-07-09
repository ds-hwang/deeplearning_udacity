
# coding: utf-8

# Deep Learning
# =============
#
# Assignment 3
# ------------
#
# Previously in `2_fullyconnected.ipynb`, you trained a logistic regression and a neural network model.
#
# The goal of this assignment is to explore regularization techniques.

# In[ ]:

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
import math
import time
from datetime import timedelta
from six.moves import cPickle as pickle


# First reload the data we generated in _notmist.ipynb_.

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
    # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


# In[ ]:

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


# ---
# Problem 1
# ---------
#
# Introduce and tune L2 regularization for both logistic and neural network models.
# Remember that L2 amounts to adding a penalty on the norm of the weights to the loss.
# In TensorFlow, you can compute the L2 loss for a tensor `t` using `nn.l2_loss(t)`.
# The right amount of regularization should improve your validation / test accuracy.
#
# ---

def run_p1(batch_size, training_epochs, adam_learning_rate, beta_regularization_value):
    graph = tf.Graph()
    with graph.as_default():
        num_feature = image_size * image_size
        num_hidden_feature = 1024

        X = tf.placeholder(tf.float32, [None, num_feature], name="input")
        Y = tf.placeholder(tf.float32, [None, num_labels], name="output")
        W1 = tf.Variable(tf.truncated_normal([num_feature, num_hidden_feature], stddev=0.1), name="weight1")
        b1 = tf.Variable(tf.zeros([num_hidden_feature]), name="bias1")

        hidden1 = tf.nn.relu(tf.matmul(X, W1) + b1)
        W2 = tf.Variable(tf.truncated_normal([num_hidden_feature, num_labels], stddev=0.1), name="weight2")
        b2 = tf.Variable(tf.zeros([num_labels]), name="bias2")

        Y_pred = tf.nn.softmax(tf.matmul(hidden1, W2) + b2)
        beta = tf.constant(beta_regularization_value, tf.float32, name="beta")
        l2_loss = beta * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2))

        cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(Y_pred), reduction_indices= 1)) + l2_loss

        optimizer = tf.train.AdamOptimizer(learning_rate=adam_learning_rate).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print("Initialized")
        for epoch in range(int(math.ceil(training_epochs))):
            n_train = train_labels.shape[0]
            n_batch = n_train / batch_size
            # training_epochs can be 0.001 to test overfit
            fraction = training_epochs - epoch
            if (fraction) < 1:
                n_batch = n_batch * fraction
            random_index = np.random.permutation(n_train)
            for step in range(int(math.ceil(n_batch))):
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

                total_step = step + (epoch * n_batch)
                if (step % 500 == 0):
                    print("Minibatch loss at step %d: %f" % (total_step, l))
                    print("Minibatch accuracy: %.1f%%" % (a * 100))
                if (step + 1) == n_batch:
                    batch_data = train_dataset[random_index[0:end], :]
                    batch_labels = train_labels[random_index[0:end], :]
                    cross_entropy_value, accuracy_value = session.run([cross_entropy, accuracy],
                                                                       feed_dict={X: batch_data, Y: batch_labels})
                    print("epoch complete: step %d num_training %d" %
                           (total_step, (total_step + 1) * batch_size))
                    print("epoch complete: Train [CrossEntropy / Training Accuracy] {:05.4f} / {:05.4f}".
                          format(cross_entropy_value, accuracy_value))
        print("Complete training")
        cross_entropy_value, accuracy_value = session.run([cross_entropy, accuracy], feed_dict={X: train_dataset, Y: train_labels})
        print("Train [CrossEntropy / Training Accuracy] {:05.4f} / {:05.4f}".format(cross_entropy_value, accuracy_value))
        cross_entropy_valid, accuracy_valid = session.run([cross_entropy, accuracy], feed_dict={X: valid_dataset, Y: valid_labels})
        print("Validation [CrossEntropy / Training Accuracy] {:05.4f} / {:05.4f}".format(cross_entropy_valid, accuracy_valid))
        cross_entropy_test = cross_entropy.eval({X: test_dataset, Y: test_labels})
        accuracy_test = accuracy.eval({X: test_dataset, Y: test_labels})
        print("Testing [CrossEntropy / Training Accuracy] {:05.4f} / {:05.4f}".format(cross_entropy_test, accuracy_test))

batch_size = 100
training_epochs = 2
adam_learning_rate = 0.001
beta_regularization_value = 0.001
# run_p1(batch_size, training_epochs, adam_learning_rate, beta_regularization_value)

# Train [CrossEntropy / Training Accuracy] 0.5298 / 0.8748
# Validation [CrossEntropy / Training Accuracy] 0.5676 / 0.8641
# Testing [CrossEntropy / Training Accuracy] 0.3540 / 0.9290

# beta_regularization_value 0.1 : underfit
#  Train [CrossEntropy / Training Accuracy] 1.3734 / 0.7949
#  Validation [CrossEntropy / Training Accuracy] 1.3693 / 0.7928
#  Testing [CrossEntropy / Training Accuracy] 1.1861 / 0.8658
# beta_regularization_value 0.001 : looks ok
#  Train [CrossEntropy / Training Accuracy] 0.5298 / 0.8748
#  Validation [CrossEntropy / Training Accuracy] 0.5676 / 0.8641
#  Testing [CrossEntropy / Training Accuracy] 0.3540 / 0.9290


# ---
# Problem 2
# ---------
# Let's demonstrate an extreme case of overfitting. Restrict your training data to just a few batches. What happens?
#
# ---

batch_size = 100
training_epochs = 0.001
adam_learning_rate = 0.001
beta_regularization_value = 0.001
# run_p1(batch_size, training_epochs, adam_learning_rate, beta_regularization_value)

# Underfitting results
training_epochs = 0.001
# epoch complete: step 1 num_training 100
# epoch complete: Train [CrossEntropy / Training Accuracy] 4.1210 / 0.6900
# Complete training
# Train [CrossEntropy / Training Accuracy] 5.0205 / 0.4450
# Validation [CrossEntropy / Training Accuracy] 5.0145 / 0.4469
# Testing [CrossEntropy / Training Accuracy] 4.8151 / 0.4863
training_epochs = 0.005
# epoch complete: step 9 num_training 900
# epoch complete: Train [CrossEntropy / Training Accuracy] 3.8830 / 0.8150
# Complete training
# Train [CrossEntropy / Training Accuracy] 4.2011 / 0.7367
# Validation [CrossEntropy / Training Accuracy] 4.1680 / 0.7406
# Testing [CrossEntropy / Training Accuracy] 3.8511 / 0.8037
training_epochs = 0.01
# epoch complete: step 19 num_training 1900
# epoch complete: Train [CrossEntropy / Training Accuracy] 3.8317 / 0.8265
# Complete training
# Train [CrossEntropy / Training Accuracy] 4.0659 / 0.7834
# Validation [CrossEntropy / Training Accuracy] 4.0634 / 0.7835
# Testing [CrossEntropy / Training Accuracy] 3.6864 / 0.8557
training_epochs = 0.1
# epoch complete: step 199 num_training 19900
# epoch complete: Train [CrossEntropy / Training Accuracy] 2.9298 / 0.8629
# Complete training
# Train [CrossEntropy / Training Accuracy] 3.0846 / 0.8298
# Validation [CrossEntropy / Training Accuracy] 3.1098 / 0.8233
# Testing [CrossEntropy / Training Accuracy] 2.8359 / 0.8921
training_epochs = 1
# epoch complete: step 1999 num_training 199900
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.6265 / 0.8767
# Complete training
# Train [CrossEntropy / Training Accuracy] 0.6265 / 0.8767
# Validation [CrossEntropy / Training Accuracy] 0.6569 / 0.8685
# Testing [CrossEntropy / Training Accuracy] 0.4553 / 0.9306
training_epochs = 2
# epoch complete: step 3999 num_training 399900
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.5137 / 0.8827
# Complete training
# Train [CrossEntropy / Training Accuracy] 0.5137 / 0.8827
# Validation [CrossEntropy / Training Accuracy] 0.5486 / 0.8755
# Testing [CrossEntropy / Training Accuracy] 0.3367 / 0.9356


# ---
# Problem 3
# ---------
# Introduce Dropout on the hidden layer of the neural network.
# Remember: Dropout should only be introduced during training, not evaluation,
# otherwise your evaluation results would be stochastic as well.
# TensorFlow provides `nn.dropout()` for that, but you have to make sure it's only inserted during training.
#
# What happens to our extreme overfitting case?
#
# ---

def run_p3(batch_size, dropout_probability, training_epochs, adam_learning_rate, beta_regularization_value):
    def create_variables(num_feature, num_labels):
        n_layer0 = num_feature
        n_layer1 = 1024
        n_layer2 =  num_labels
        W = {
            'W1': tf.get_variable("W1", shape=[n_layer0, n_layer1], initializer=xavier_init(n_layer0, n_layer1)),
            'W2': tf.get_variable("W2", shape=[n_layer1, n_layer2], initializer=xavier_init(n_layer1, n_layer2)),
#             'W1': tf.Variable(tf.truncated_normal([n_layer0, n_layer1], stddev=0.1), name="weight1"),
#             'W2': tf.Variable(tf.truncated_normal([n_layer1, n_layer2], stddev=0.1), name="weight1"),
        }
        b = {
            'b1': tf.Variable(tf.random_normal([n_layer1])),
            'b2': tf.Variable(tf.random_normal([n_layer2])),
        }
        return W, b

    def create_nn(X, W, b, dropout):
        layer1 = tf.nn.dropout(tf.nn.relu(tf.matmul(X, W['W1']) + b['b1']), dropout, name="hidden1")
        layer2 = tf.nn.softmax(tf.matmul(layer1, W['W2']) + b['b2'])
        return layer2

    # X. Glorot and Y. Bengio, "Understanding the difficulty of training deep feedforward neural networks", 2010.
    def xavier_init(n_inputs, n_outputs, uniform=True):
        if uniform:
            init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
            return tf.random_uniform_initializer(-init_range, init_range)
        else:
            stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)

    graph = tf.Graph()
    with graph.as_default():
        num_feature = image_size * image_size

        W, b = create_variables(num_feature, num_labels)
        X = tf.placeholder(tf.float32, [None, num_feature], name="input")
        Y = tf.placeholder(tf.float32, [None, num_labels], name="output")
        dropout_prob = tf.placeholder(tf.float32, name="dropout")
        Y_pred = create_nn(X, W, b, dropout_prob)

        beta = tf.constant(beta_regularization_value, tf.float32, name="beta")
        l2_loss = beta * (tf.nn.l2_loss(W['W1']) + tf.nn.l2_loss(W['W2']))
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(Y_pred), reduction_indices= 1)) + l2_loss

        optimizer = tf.train.AdamOptimizer(learning_rate=adam_learning_rate).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print("Initialized")
        for epoch in range(int(math.ceil(training_epochs))):
            n_train = train_labels.shape[0]
            n_batch = n_train / batch_size
            # training_epochs can be 0.001 to test overfit
            fraction = training_epochs - epoch
            if (fraction) < 1:
                n_batch = n_batch * fraction
            random_index = np.random.permutation(n_train)
            total_step = int(math.ceil(n_batch))
            for step in range(total_step):
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
                feed_dict = {X : batch_data, Y : batch_labels, dropout_prob: dropout_probability}
                _, l, a = session.run(
                  [optimizer, cross_entropy, accuracy], feed_dict=feed_dict)

                accumulated_step = step + (epoch * n_batch)
                if (step % 500 == 0):
                    print("Minibatch loss at step %d: %f" % (accumulated_step, l))
                    print("Minibatch accuracy: %.1f%%" % (a * 100))
                if (step + 1) == total_step:
                    print("epoch complete: step %d num_training %d" %
                           (accumulated_step, (accumulated_step + 1) * batch_size))
                    max_n_training = 50000
                    start = 0
                    if end > max_n_training:
                        start = end - max_n_training
                    batch_data = train_dataset[random_index[start:end], :]
                    batch_labels = train_labels[random_index[start:end], :]
                    cross_entropy_value, accuracy_value = session.run([cross_entropy, accuracy],
                                                                       feed_dict={X: batch_data, Y: batch_labels, dropout_prob: 1})
                    print("epoch complete: Train [CrossEntropy / Training Accuracy] {:05.4f} / {:05.4f}".
                          format(cross_entropy_value, accuracy_value))
        print("Complete training")
        cross_entropy_valid, accuracy_valid = session.run([cross_entropy, accuracy],
                                                           feed_dict={X: valid_dataset, Y: valid_labels, dropout_prob: 1})
        print("Validation [CrossEntropy / Training Accuracy] {:05.4f} / {:05.4f}".format(cross_entropy_valid, accuracy_valid))
        cross_entropy_test = cross_entropy.eval({X: test_dataset, Y: test_labels, dropout_prob: 1})
        accuracy_test = accuracy.eval({X: test_dataset, Y: test_labels, dropout_prob: 1})
        print("Testing [CrossEntropy / Training Accuracy] {:05.4f} / {:05.4f}".format(cross_entropy_test, accuracy_test))

batch_size = 100
dropout_probability = 1
training_epochs = 0.001
adam_learning_rate = 0.001
beta_regularization_value = 0.001
# run_p3(batch_size, dropout_probability, training_epochs, adam_learning_rate, beta_regularization_value)

# Results
dropout_probability = 0.7
training_epochs = 1
# epoch complete: step 1999 num_training 200000
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.5632 / 0.8684
# Complete training
# Validation [CrossEntropy / Training Accuracy] 0.5873 / 0.8641
# Testing [CrossEntropy / Training Accuracy] 0.3698 / 0.9266
dropout_probability = 0.5
training_epochs = 1
# epoch complete: step 1999 num_training 200000
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.5989 / 0.8599
# Complete training
# Validation [CrossEntropy / Training Accuracy] 0.6189 / 0.8545
# Testing [CrossEntropy / Training Accuracy] 0.4010 / 0.9207

dropout_probability = 0.5
training_epochs = 0.001
# epoch complete: step 1 num_training 200
# epoch complete: Train [CrossEntropy / Training Accuracy] 2.2176 / 0.4800
# Complete training
# Validation [CrossEntropy / Training Accuracy] 2.3713 / 0.4006
# Testing [CrossEntropy / Training Accuracy] 2.2627 / 0.4360
dropout_probability = 1
training_epochs = 0.001
# epoch complete: step 1 num_training 200
# epoch complete: Train [CrossEntropy / Training Accuracy] 2.0113 / 0.6400
# Complete training
# Validation [CrossEntropy / Training Accuracy] 2.2921 / 0.4955
# Testing [CrossEntropy / Training Accuracy] 2.1788 / 0.5451

# ---
# Problem 4
# ---------
#
# Try to get the best performance you can using a multi-layer model! The best reported test accuracy using a deep network is [97.1%](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html?showComment=1391023266211#c8758720086795711595).
#
# One avenue you can explore is to add multiple layers.
#
# Another one is to use learning rate decay:
#
#     global_step = tf.Variable(0)  # count the number of steps taken.
#     learning_rate = tf.train.exponential_decay(0.5, global_step, ...)
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
#
#  ---
#

def run_p4(params):
    # Alec Karfonta  configuration in
    # http://yaroslavvb.blogspot.fi/2011/09/notmnist-dataset.html?showComment=1458776573519#c1260767559237460571
    # Test accuracy: 97.4%
    # Validation accuracy: 91.9%
    # Minibatch accuracy: 97.9%
    #
    # First I applied a Phash to every image and removed any with direct collisions. Then I split the large folder into
    # ~320k training
    # ~80k validation
    # ~17k testing
    #
    # Here are the parameters:
    # Mini-batch size: 1024
    # Hidden layer 1 size: 4096
    # Hidden layer 2 size: 2048
    # Hidden layer 3 size: 1024
    # Initial learning rate: 0.1
    # Dropout probability: 0.5
    # 150k iterations
    def create_variables(num_feature, num_labels):
        n_layer0 = num_feature
        n_layer1 = 1024
        n_layer2 = 512
        n_layer3 = 256
        n_layer4 =  num_labels
        W = {
            'W1': tf.get_variable("W1", shape=[n_layer0, n_layer1], initializer=xavier_init(n_layer0, n_layer1)),
            'W2': tf.get_variable("W2", shape=[n_layer1, n_layer2], initializer=xavier_init(n_layer1, n_layer2)),
            'W3': tf.get_variable("W3", shape=[n_layer2, n_layer3], initializer=xavier_init(n_layer1, n_layer2)),
            'W4': tf.get_variable("W4", shape=[n_layer3, n_layer4], initializer=xavier_init(n_layer1, n_layer2)),
        }
        b = {
            'b1': tf.Variable(tf.random_normal([n_layer1])),
            'b2': tf.Variable(tf.random_normal([n_layer2])),
            'b3': tf.Variable(tf.random_normal([n_layer3])),
            'b4': tf.Variable(tf.random_normal([n_layer4])),
        }
        return W, b

    def create_nn(X, W, b, dropout):
        layer1 = tf.nn.dropout(tf.nn.relu(tf.matmul(X, W['W1']) + b['b1']), dropout, name="hidden1")
        layer2 = tf.nn.dropout(tf.nn.relu(tf.matmul(layer1, W['W2']) + b['b2']), dropout, name="hidden2")
        layer3 = tf.nn.dropout(tf.nn.relu(tf.matmul(layer2, W['W3']) + b['b3']), dropout, name="hidden3")
        layer4 = tf.nn.softmax(tf.matmul(layer3, W['W4']) + b['b4'], name="prediction")
        return layer4

    # X. Glorot and Y. Bengio, "Understanding the difficulty of training deep feedforward neural networks", 2010.
    def xavier_init(n_inputs, n_outputs, uniform=True):
        if uniform:
            init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
            return tf.random_uniform_initializer(-init_range, init_range)
        else:
            stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)

    graph = tf.Graph()
    with graph.as_default():
        num_feature = image_size * image_size

        W, b = create_variables(num_feature, num_labels)
        X = tf.placeholder(tf.float32, [None, num_feature], name="input")
        Y = tf.placeholder(tf.float32, [None, num_labels], name="output")
        dropout_prob = tf.placeholder(tf.float32, name="dropout")
        Y_pred = create_nn(X, W, b, dropout_prob)

        beta = tf.constant(params['beta_regularization_value'], tf.float32, name="beta")
        l2_loss = beta * (tf.nn.l2_loss(W['W1']) + tf.nn.l2_loss(W['W2']))
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(Y_pred), reduction_indices= 1)) + l2_loss

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(params['starter_learning_rate'], global_step,
                                             100000, 0.96, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
#         optimizer = tf.train.AdamOptimizer(learning_rate=starter_learning_rate).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print("Initialized")
        batch_size = params['batch_size']
        training_epochs = params['training_epochs']
        pre_cross_entropy_valid = float('inf')
        start_time = time.time()
        for epoch in range(int(math.ceil(training_epochs))):
            n_train = train_labels.shape[0]
            n_batch = n_train / batch_size
            # training_epochs can be 0.001 to test overfit
            fraction = training_epochs - epoch
            if (fraction) < 1:
                n_batch = n_batch * fraction
            random_index = np.random.permutation(n_train)
            total_step = int(math.ceil(n_batch))
            for step in range(total_step):
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
                feed_dict = {X : batch_data, Y : batch_labels, dropout_prob: params['dropout_probability']}
                _, l, a = session.run(
                  [optimizer, cross_entropy, accuracy], feed_dict=feed_dict)

                threshold_print = 50000 / batch_size
                accumulated_step = step + (epoch * n_batch)
                accumulated_n_training = (accumulated_step + 1) * batch_size
                if (step % threshold_print == 0):
                    print("Minibatch loss at num_training %d: %f" % (accumulated_n_training, l))
                    print("Minibatch accuracy: %.1f%%" % (a * 100))

            print("epoch complete: step %d num_training %d time:%s" %
                   (accumulated_step, accumulated_n_training, timedelta(seconds=(time.time() - start_time))))
            # remove it in fast machine
            max_n_training = 50000
            start = 0
            if end > max_n_training:
                start = end - max_n_training
            batch_data = train_dataset[random_index[start:end], :]
            batch_labels = train_labels[random_index[start:end], :]
            cross_entropy_value, accuracy_value = session.run([cross_entropy, accuracy],
                                                               feed_dict={X: batch_data, Y: batch_labels, dropout_prob: 1})
            print("epoch complete: Train [CrossEntropy / Training Accuracy] {:05.4f} / {:05.4f}".
                  format(cross_entropy_value, accuracy_value))

            cross_entropy_valid, accuracy_valid = session.run([cross_entropy, accuracy],
                                               feed_dict={X: valid_dataset, Y: valid_labels, dropout_prob: 1})
            print("Validation [CrossEntropy / Training Accuracy] {:05.4f} / {:05.4f}".format(cross_entropy_valid, accuracy_valid))
            if cross_entropy_valid > pre_cross_entropy_valid:
                print("BREAK: Validation CrossEntropy [now / previous] {:05.4f} / {:05.4f}".format(cross_entropy_valid, pre_cross_entropy_valid))
                break
            pre_cross_entropy_valid = cross_entropy_valid
            if (time.time() - start_time) > params['max_test_time']:
                print("BREAK: TIME OVER")
                break

        print("Complete training. Total time:%s" % timedelta(seconds=(time.time() - start_time)))
        cross_entropy_test = cross_entropy.eval({X: test_dataset, Y: test_labels, dropout_prob: 1})
        accuracy_test = accuracy.eval({X: test_dataset, Y: test_labels, dropout_prob: 1})
        print("Testing [CrossEntropy / Training Accuracy] {:05.4f} / {:05.4f}".format(cross_entropy_test, accuracy_test))


learning_params = {
    'batch_size': 300,
    'dropout_probability': 0.7,
    'training_epochs': 1000,
    'max_test_time': 60*60*5,
    'starter_learning_rate': 0.1,
    'beta_regularization_value': 0.001
}
run_p4(learning_params)

n_layer1 = 1024
n_layer2 = 1024
n_layer3 = 1024
dropout_probability = 0.7
training_epochs = 1
# epoch complete: step 1999 num_training 200000
# epoch complete: Train [CrossEntropy / Training Accuracy] 1.1376 / 0.8537
# Complete training
# Validation [CrossEntropy / Training Accuracy] 1.1530 / 0.8495
# Testing [CrossEntropy / Training Accuracy] 0.9392 / 0.9165
n_layer1 = 1024
n_layer2 = 512
n_layer3 = 256
dropout_probability = 0.7
training_epochs = 1
# epoch complete: step 1999 num_training 200000
# epoch complete: Train [CrossEntropy / Training Accuracy] 1.0318 / 0.8530
# Complete training
# Validation [CrossEntropy / Training Accuracy] 1.0487 / 0.8478
# Testing [CrossEntropy / Training Accuracy] 0.8313 / 0.9159
n_layer1 = 1024
n_layer2 = 512
n_layer3 = 256
dropout_probability = 1
training_epochs = 1
# epoch complete: step 1999 num_training 200000
# epoch complete: Train [CrossEntropy / Training Accuracy] 1.0168 / 0.8561
# Complete training
# Validation [CrossEntropy / Training Accuracy] 1.0343 / 0.8520
# Testing [CrossEntropy / Training Accuracy] 0.8188 / 0.9199
n_layer1 = 1024
n_layer2 = 512
n_layer3 = 256
dropout_probability = 1
training_epochs = 2
# epoch complete: step 3999 num_training 400000
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.7887 / 0.8773
# Complete training
# Validation [CrossEntropy / Training Accuracy] 0.8158 / 0.8722
# Testing [CrossEntropy / Training Accuracy] 0.6166 / 0.9335





n_layer1 = 1024
n_layer2 = 512
n_layer3 = 256
dropout_probability = 0.7
training_epochs = 10
# epoch complete: step 665 num_training 199800
# epoch complete: Train [CrossEntropy / Training Accuracy] 1.2380 / 0.8394
# epoch complete: step 1331 num_training 399600
# epoch complete: Train [CrossEntropy / Training Accuracy] 1.1063 / 0.8506
# epoch complete: step 1997 num_training 599400
# epoch complete: Train [CrossEntropy / Training Accuracy] 1.0114 / 0.8575
# epoch complete: step 2663 num_training 799200
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.9255 / 0.8647
# epoch complete: step 3329 num_training 999000
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.8576 / 0.8671
# epoch complete: step 3995 num_training 1198800
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.7958 / 0.8728
# epoch complete: step 4661 num_training 1398600
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.7398 / 0.8775
# epoch complete: step 5327 num_training 1598400
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.7024 / 0.8778
# epoch complete: step 5993 num_training 1798200
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.6586 / 0.8804
# epoch complete: step 6659 num_training 1998000
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.6208 / 0.8833
# Complete training
# Validation [CrossEntropy / Training Accuracy] 0.6425 / 0.8777
# Testing [CrossEntropy / Training Accuracy] 0.4521 / 0.9383
n_layer1 = 1024
n_layer2 = 512
n_layer3 = 256
dropout_probability = 0.7
training_epochs = 30
# epoch complete: step 5993 num_training 1798200
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.6493 / 0.8844
# epoch complete: step 6659 num_training 1998000
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.6191 / 0.8850
# Complete training
# Validation [CrossEntropy / Training Accuracy] 0.6445 / 0.8762
# Testing [CrossEntropy / Training Accuracy] 0.4514 / 0.9387

learning_params = {
    'batch_size': 300,
    'dropout_probability': 0.7,
    'training_epochs': 1000,
    'max_test_time': 60*60*5,
    'starter_learning_rate': 0.1,
    'beta_regularization_value': 0.001
}
# epoch complete: step 131201 num_training 39360600 time:4:55:36.291662
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.2408 / 0.9577
# Validation [CrossEntropy / Training Accuracy] 0.3970 / 0.9116
# epoch complete: step 131867 num_training 39560400 time:4:57:05.501906
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.2351 / 0.9603
# Validation [CrossEntropy / Training Accuracy] 0.3906 / 0.9128
# epoch complete: step 132533 num_training 39760200 time:4:58:35.461150
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.2298 / 0.9615
# Validation [CrossEntropy / Training Accuracy] 0.3928 / 0.9139
# epoch complete: step 133199 num_training 39960000 time:5:00:04.257040
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.2387 / 0.9590
# Validation [CrossEntropy / Training Accuracy] 0.3955 / 0.9148
# BREAK: TIME OVER
# Complete training. Total time:5:00:13.045505
# Testing [CrossEntropy / Training Accuracy] 0.2168 / 0.9653
