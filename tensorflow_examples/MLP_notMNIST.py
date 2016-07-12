
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
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import math
import time
from datetime import timedelta
from six.moves import cPickle as pickle
import notMNIST_downloader

notMNIST_downloader.download();

# First reload the data we generated in _notmist.ipynb_.
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


# Convince yourself that the data is still good after shuffling!
def show_dataset(dataset, labels):
    num_labels = 10
    sample_size = 10
    for i in range(num_labels * sample_size):
        plt.subplot(10, sample_size, i + 1)
        plt.title(chr(ord('A') + labels[i]))
        plt.axis('off')
        plt.imshow(dataset[i, :, :], cmap=plt.get_cmap('gray'))
    plt.show()
# show_dataset(train_dataset, train_labels)
# show_dataset(test_dataset, test_labels)
# show_dataset(valid_dataset, valid_labels)


# Reformat into a shape that's more adapted to the models we're going to train:
# - data as a flat matrix,
# - labels as float 1-hot encodings.

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


def run_mlp(params):
    def create_variables(num_feature, num_labels):
        n_layer0 = num_feature
        n_layer1 = params['n_layer1']
        n_layer2 = params['n_layer2']
        n_layer3 = params['n_layer3']
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
                    print("Minibatch loss at num_training %d: %f accuracy: %.1f%%" % (accumulated_n_training, l, a * 100))

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
            epochs.append(epoch)
            train_cross_entropys.append(cross_entropy_value)
            valid_cross_entropys.append(cross_entropy_valid)
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

        # cross entropy learning curve
        cross_entropys = np.vstack([train_cross_entropys, valid_cross_entropys])
        plt.plot(epochs, cross_entropys.T, linewidth=2)
        plt.show()

        # Visual representation of hidden layers
        def show_W(dataset):
            width = 10
            for i in range(width * width):
                plt.subplot(width, width, i + 1)
                plt.axis('off')
                plt.imshow(W1[i, :, :], cmap=plt.cm.gray)
            plt.show()
        image_size_1 = int(math.sqrt(params['n_layer1']))
        W1 = W['W1'].eval().reshape((-1, image_size_1, image_size_1))
        show_W(W1)
        image_size_2 = int(math.sqrt(params['n_layer2']))
        W2 = W['W2'].eval().reshape((-1, image_size_2, image_size_2))
        show_W(W2)
        image_size_3 = int(math.sqrt(params['n_layer3']))
        W3 = W['W3'].eval().reshape((-1, image_size_3, image_size_3))
        show_W(W3)

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
learning_params = {
    'n_layer1': 4096,
    'n_layer2': 2025,
    'n_layer3': 1024,
    'batch_size': 1024,
    'dropout_probability': 0.5,
    'training_epochs': 150000,
    'max_test_time': 60 * 60 * 9,
    'starter_learning_rate': 0.1,
    'beta_regularization_value': 0.001
}
run_mlp(learning_params)

learning_params = {
    'n_layer1': 1024,
    'n_layer2': 512,
    'n_layer3': 256,
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



learning_params = {
    'n_layer1': 4096,
    'n_layer2': 2025,
    'n_layer3': 1024,
    'batch_size': 1024,
    'dropout_probability': 0.5,
    'training_epochs': 150000,
    'max_test_time': 60 * 60 * 9,
    'starter_learning_rate': 0.1,
    'beta_regularization_value': 0.001
}
# saturated around 65 epoch
# epoch complete: step 12089 num_training 12380160 time:5:29:17.438472
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.6011 / 0.8861
# Validation [CrossEntropy / Training Accuracy] 0.6111 / 0.8849
# epoch complete: step 12284 num_training 12579840 time:5:34:00.698868
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.5944 / 0.8856
# Validation [CrossEntropy / Training Accuracy] 0.6024 / 0.8842
# epoch complete: step 12479 num_training 12779520 time:5:38:48.527591
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.5954 / 0.8856
# Validation [CrossEntropy / Training Accuracy] 0.5955 / 0.8858
# epoch complete: step 12674 num_training 12979200 time:5:43:32.467080
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.5871 / 0.8855
# Validation [CrossEntropy / Training Accuracy] 0.5885 / 0.8876
# epoch complete: step 12869 num_training 13178880 time:5:48:17.051833
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.5898 / 0.8826
# Validation [CrossEntropy / Training Accuracy] 0.5946 / 0.8833
# BREAK: Validation CrossEntropy [now / previous] 0.5946 / 0.5885
# Complete training. Total time:5:48:44.815504
# Testing [CrossEntropy / Training Accuracy] 0.4216 / 0.9353