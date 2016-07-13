
# coding: utf-8

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import math
import time
from datetime import timedelta
from tensorflow.examples.tutorials.mnist import input_data

def load_data():
    mnist = input_data.read_data_sets('data', one_hot=True)
    train_dataset   = mnist.train.images
    train_labels = mnist.train.labels
    valid_dataset   = mnist.validation.images
    valid_labels = mnist.validation.labels
    test_dataset    = mnist.test.images
    test_labels  = mnist.test.labels
    return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels

train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = load_data();

image_size = int(math.sqrt(train_dataset.shape[1]))
num_labels = train_labels.shape[1]
num_channels = 1 # grayscale

# Reformat into a shape that's more adapted to the models we're going to train:
# - data as a flat matrix,
# - labels as float 1-hot encodings.
def reformat(dataset, labels):
    dataset = dataset.reshape(
      (-1, image_size, image_size, num_channels)).astype(np.float32)
    return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# Convince yourself that the data is still good after shuffling!
def show_dataset(dataset, labels):
    sample_size = 10
    for i in range(num_labels * sample_size):
        plt.subplot(10, sample_size, i + 1)
        plt.title(chr(ord('0') + np.argmax(labels[i])))
        plt.axis('off')
        plt.imshow(dataset[i, :, :, 0], cmap=plt.get_cmap('gray'))
    plt.show()
# show_dataset(train_dataset, train_labels)
# show_dataset(test_dataset, test_labels)
# show_dataset(valid_dataset, valid_labels)


# Hints from http://culurciello.github.io/tech/2016/06/04/nets.html
# middle layer uses max pool and final layer uses average pool
# multiple 3x3 convolution instead of 5x5 or 7x7
# 1x1 convolution to reduce computation. 256×64 × 1×1 , 64x64 x 3x3, 64×256 × 1×1 instead of 256x256 x 3x3. (usually 1/4 of the input)
# use mini-batch size around 128 or 256.
# use the linear learning rate decay policy.
# cleanliness of the data is more important then the size.
def run_conv(params):
    def create_variables(num_labels):
        patch_size = params['patch_size']
        depth_1 = params['depth_1']
        depth_2 = params['depth_2']
        hidden_input_image_size = image_size//4
        hidden_n_input = hidden_input_image_size * hidden_input_image_size * depth_2
        hidden_n_output_1 = params['hidden_n_output_1']
        hidden_n_output_2 = params['hidden_n_output_2']
        W = {
            'L1_ConvF': tf.Variable(tf.truncated_normal(shape=[patch_size, patch_size, num_channels, depth_1], stddev = 0.1, name="L1_ConvF")),
            'L2_ConvF': tf.Variable(tf.truncated_normal(shape=[patch_size, patch_size, depth_1, depth_2], stddev = 0.1, name="L2_ConvF")),
            'L3_W': tf.get_variable("L3_W", shape=[hidden_n_input, hidden_n_output_1], initializer=xavier_init(hidden_n_input, hidden_n_output_1)),
            'L4_W': tf.get_variable("L4_W", shape=[hidden_n_output_1, hidden_n_output_2], initializer=xavier_init(hidden_n_output_1, hidden_n_output_2)),
            'L5_W': tf.get_variable("L5_W", shape=[hidden_n_output_2, num_labels], initializer=xavier_init(hidden_n_output_2, num_labels)),
        }
        b = {
            'L1_b': tf.Variable(tf.zeros([depth_1])),
            'L2_b': tf.Variable(tf.zeros([depth_2])),
            'L3_b': tf.Variable(tf.random_normal([hidden_n_output_1])),
            'L4_b': tf.Variable(tf.random_normal([hidden_n_output_2])),
            'L5_b': tf.Variable(tf.random_normal([num_labels])),
        }
        return W, b

    def create_nn(X, W, b, dropout):
        l1_conv = tf.nn.relu(tf.nn.dropout(tf.nn.bias_add(
                             tf.nn.conv2d(X, W['L1_ConvF'], [1, 1, 1, 1], padding='SAME'),
                             b['L1_b']), dropout, name="l1_conv"))
        l1_pool = tf.nn.max_pool(l1_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        l2_conv = tf.nn.relu(tf.nn.dropout(tf.nn.bias_add(
                             tf.nn.conv2d(l1_pool, W['L2_ConvF'], [1, 1, 1, 1], padding='SAME'),
                             b['L2_b']), dropout, name="l2_conv"))
        l2_pool = tf.nn.avg_pool(l2_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        l2_output = tf.reshape(l2_pool, [-1, W['L3_W'].get_shape().as_list()[0]])

        l3_hidden = tf.nn.dropout(tf.nn.relu(tf.matmul(l2_output, W['L3_W']) + b['L3_b']), dropout, name="l3_hidden")
        l4_hidden = tf.nn.dropout(tf.nn.relu(tf.matmul(l3_hidden, W['L4_W']) + b['L4_b']), dropout, name="l4_hidden")
        logits = tf.nn.softmax(tf.matmul(l4_hidden, W['L5_W']) + b['L5_b'], name="logits")
        return logits, l1_conv, l2_conv

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
        W, b = create_variables(num_labels)
        X = tf.placeholder(tf.float32, [None, image_size, image_size, num_channels], name="input")
        Y = tf.placeholder(tf.float32, [None, num_labels], name="output")
        dropout_prob = tf.placeholder(tf.float32, name="dropout")
        Y_pred, l1_conv, l2_conv = create_nn(X, W, b, dropout_prob)

        beta = tf.constant(params['beta_regularization_value'], tf.float32, name="beta")
        # TODO: need to regularize 'L1_ConvF'?
        l2_loss = beta * (tf.nn.l2_loss(W['L4_W']) + tf.nn.l2_loss(W['L5_W']))
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
        start_time = time.time()
        epochs = []
        train_cross_entropys = []
        valid_cross_entropys = []
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
                batch_data = train_dataset[random_index[start:end], :, :, :]
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
                    print("Minibatch loss at num_training %d: %f accuracy: %.1f%%" % (accumulated_n_training, l, a * 100))

            print("epoch complete: step %d num_training %d time:%s" %
                   (accumulated_step, accumulated_n_training, timedelta(seconds=(time.time() - start_time))))
            # remove it in fast machine
            max_n_training = 50000
            start = 0
            if end > max_n_training:
                start = end - max_n_training
            batch_data = train_dataset[random_index[start:end], :, :, :]
            batch_labels = train_labels[random_index[start:end], :]
            cross_entropy_value, accuracy_value = session.run([cross_entropy, accuracy],
                                                               feed_dict={X: batch_data, Y: batch_labels, dropout_prob: 1})
            print("epoch complete: Train [CrossEntropy / Training Accuracy] {:05.4f} / {:05.4f}".
                  format(cross_entropy_value, accuracy_value))

            cross_entropy_valid, accuracy_valid = session.run([cross_entropy, accuracy],
                                               feed_dict={X: valid_dataset, Y: valid_labels, dropout_prob: 1})
            print("Validation [CrossEntropy / Training Accuracy] {:05.4f} / {:05.4f}".format(cross_entropy_valid, accuracy_valid))
            epochs.append(epoch)
            train_cross_entropys.append(cross_entropy_value)
            valid_cross_entropys.append(cross_entropy_valid)
            if (time.time() - start_time) > params['max_test_time']:
                print("BREAK: TIME OVER")
                break
        print("Complete training. Total time:%s" % timedelta(seconds=(time.time() - start_time)))
        cross_entropy_test = cross_entropy.eval({X: test_dataset, Y: test_labels, dropout_prob: 1})
        accuracy_test = accuracy.eval({X: test_dataset, Y: test_labels, dropout_prob: 1})
        print("Testing [CrossEntropy / Training Accuracy] {:05.4f} / {:05.4f}".format(cross_entropy_test, accuracy_test))

        # cross entropy learning curve
        cross_entropys = np.vstack([train_cross_entropys, valid_cross_entropys])
        plt.plot(epochs, cross_entropys.T, linewidth=2)
        plt.show()

        # Visual representation of hidden layers
        def show_layer(layer):
            width = 10
            for i in range(layer.shape[3]):
                plt.subplot(width, width, i + 1)
                plt.axis('off')
                plt.imshow(layer[0, :, :, i], cmap=plt.get_cmap('gray'))
            plt.show()

        batch_data = train_dataset[random_index[0:1], :, :, :]
        l1_conv_layer, l2_conv_layer = session.run([l1_conv, l2_conv],
                                                   feed_dict={X: batch_data, dropout_prob: 1})
        show_layer(l1_conv_layer)
        show_layer(l2_conv_layer)

# LeNet5
# http://yaroslavvb.blogspot.fi/2011/09/notmnist-dataset.html?showComment=1458776573519#c1260767559237460571
learning_params = {
    'patch_size': 3,
    'depth_1': 6,
    'depth_2': 16,
    'hidden_n_output_1': 120,
    'hidden_n_output_2': 84,
    'batch_size': 128,
    'dropout_probability': 0.7,
    'training_epochs': 20,
    'max_test_time': 60 * 10,
    'starter_learning_rate': 0.1,
    'beta_regularization_value': 0.001
}
run_conv(learning_params)


learning_params = {
    'patch_size': 3,
    'depth_1': 6,
    'depth_2': 16,
    'hidden_n_output_1': 120,
    'hidden_n_output_2': 84,
    'batch_size': 128,
    'dropout_probability': 0.7,
    'training_epochs': 20,
    'max_test_time': 60 * 10,
    'starter_learning_rate': 0.1,
    'beta_regularization_value': 0.001
}
# epoch complete: step 428 num_training 54912 time:0:00:18.917990
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.3381 / 0.9187
# Validation [CrossEntropy / Training Accuracy] 0.3222 / 0.9282
# epoch complete: step 857 num_training 109824 time:0:00:54.837309
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.2337 / 0.9484
# Validation [CrossEntropy / Training Accuracy] 0.2191 / 0.9566
# epoch complete: step 1286 num_training 164736 time:0:01:31.977322
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.1871 / 0.9601
# Validation [CrossEntropy / Training Accuracy] 0.1776 / 0.9648
# epoch complete: step 1715 num_training 219648 time:0:02:05.933158
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.1639 / 0.9666
# Validation [CrossEntropy / Training Accuracy] 0.1541 / 0.9714
# epoch complete: step 2144 num_training 274560 time:0:02:40.228236
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.1526 / 0.9670
# Validation [CrossEntropy / Training Accuracy] 0.1476 / 0.9710
# epoch complete: step 2573 num_training 329472 time:0:03:14.293120
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.1320 / 0.9733
# Validation [CrossEntropy / Training Accuracy] 0.1285 / 0.9734
# epoch complete: step 3002 num_training 384384 time:0:03:48.845990
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.1356 / 0.9709
# Validation [CrossEntropy / Training Accuracy] 0.1323 / 0.9736
# BREAK: Validation CrossEntropy [now / previous] 0.1323 / 0.1285
# Complete training. Total time:0:04:03.069134
# Testing [CrossEntropy / Training Accuracy] 0.1340 / 0.9702

