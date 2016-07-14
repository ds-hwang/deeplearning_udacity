
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
num_channels = 1 # grayscale

def reformat(dataset, labels):
    dataset = dataset.reshape(
      (-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


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
        depth_3 = params['depth_3']
        hidden_input_image_size = image_size//4
        # ceil(7 / 2) = 4
        hidden_input_image_size = math.ceil(hidden_input_image_size / 2.0)
        hidden_n_input = hidden_input_image_size * hidden_input_image_size * depth_3
        hidden_n_output_1 = params['hidden_n_output_1']
        hidden_n_output_2 = params['hidden_n_output_2']
        W = {
            'L1_ConvF': tf.Variable(tf.truncated_normal(shape=[patch_size, patch_size, num_channels, depth_1], stddev = 0.1, name="L1_ConvF")),
            'L2_ConvF': tf.Variable(tf.truncated_normal(shape=[patch_size, patch_size, depth_1, depth_2], stddev = 0.1, name="L2_ConvF")),
            'L3_ConvF': tf.Variable(tf.truncated_normal(shape=[patch_size, patch_size, depth_2, depth_3], stddev = 0.1, name="L3_ConvF")),
            'L4_W': tf.get_variable("L4_W", shape=[hidden_n_input, hidden_n_output_1], initializer=xavier_init(hidden_n_input, hidden_n_output_1)),
            'L5_W': tf.get_variable("L5_W", shape=[hidden_n_output_1, hidden_n_output_2], initializer=xavier_init(hidden_n_output_1, hidden_n_output_2)),
            'L6_W': tf.get_variable("L6_W", shape=[hidden_n_output_2, num_labels], initializer=xavier_init(hidden_n_output_2, num_labels)),
        }
        b = {
            'L1_b': tf.Variable(tf.zeros([depth_1])),
            'L2_b': tf.Variable(tf.zeros([depth_2])),
            'L3_b': tf.Variable(tf.zeros([depth_3])),
            'L4_b': tf.Variable(tf.random_normal([hidden_n_output_1])),
            'L5_b': tf.Variable(tf.random_normal([hidden_n_output_2])),
            'L6_b': tf.Variable(tf.random_normal([num_labels])),
        }
        return W, b

    def create_nn(X, W, b, dropout):
        l1_conv = tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(
                                tf.nn.conv2d(X, W['L1_ConvF'], [1, 1, 1, 1], padding='SAME'),
                                b['L1_b'])), dropout, name="l1_conv")
        l1_pool = tf.nn.max_pool(l1_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        l2_conv = tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(
                                tf.nn.conv2d(l1_pool, W['L2_ConvF'], [1, 1, 1, 1], padding='SAME'),
                                b['L2_b'])), dropout, name="l2_conv")
        l2_pool = tf.nn.max_pool(l2_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        l3_conv = tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(
                                tf.nn.conv2d(l2_pool, W['L3_ConvF'], [1, 1, 1, 1], padding='SAME'),
                                b['L3_b'])), dropout, name="l3_conv")
        l3_pool = tf.nn.avg_pool(l3_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        l3_output = tf.reshape(l3_pool, [-1, W['L4_W'].get_shape().as_list()[0]])

        l4_hidden = tf.nn.dropout(tf.nn.relu(tf.matmul(l3_output, W['L4_W']) + b['L4_b']), dropout, name="l4_hidden")
        l5_hidden = tf.nn.dropout(tf.nn.relu(tf.matmul(l4_hidden, W['L5_W']) + b['L5_b']), dropout, name="l5_hidden")
        logits = tf.nn.softmax(tf.matmul(l5_hidden, W['L6_W']) + b['L6_b'], name="logits")
        out = {
               'logits': logits,
               'l1_conv': l1_conv,
               'l2_conv': l2_conv,
               'l3_conv': l3_conv,
               'l4_hidden': l4_hidden,
               'l5_hidden': l5_hidden,
            }
        return out

    # TODO: make it work.
#     epoch complete: Train [CrossEntropy / Training Accuracy] 2.4094 / 0.0994
#     Validation [CrossEntropy / Training Accuracy] 2.4095 / 0.1000
    def create_variables_1to1(num_labels):
        patch_size = params['patch_size']
        depth_1 = params['depth_1']
        depth_2 = params['depth_2']
        depth_3 = params['depth_3']
        hidden_input_image_size = image_size//4
        # ceil(7 / 2) = 4
        hidden_input_image_size = math.ceil(hidden_input_image_size / 2.0)
        hidden_n_input = hidden_input_image_size * hidden_input_image_size * depth_3
        hidden_n_output_1 = params['hidden_n_output_1']
        hidden_n_output_2 = params['hidden_n_output_2']
        W = {
            'L1_ConvF': tf.Variable(tf.truncated_normal(shape=[patch_size, patch_size, num_channels, depth_1], stddev = 0.1, name="L1_ConvF")),
            'L1_Conv1to1F': tf.Variable(tf.truncated_normal(shape=[1, 1, depth_1, depth_1//4], stddev = 0.1, name="L1_Conv1to1F")),
            'L2_ConvF': tf.Variable(tf.truncated_normal(shape=[patch_size, patch_size, depth_1//4, depth_2//4], stddev = 0.1, name="L2_ConvF")),
            'L2_Conv1to1F1': tf.Variable(tf.truncated_normal(shape=[1, 1, depth_2//4, depth_2], stddev = 0.1, name="L2_Conv1to1F1")),
            'L2_Conv1to1F2': tf.Variable(tf.truncated_normal(shape=[1, 1, depth_2, depth_2//4], stddev = 0.1, name="L2_Conv1to1F2")),
            'L3_ConvF': tf.Variable(tf.truncated_normal(shape=[patch_size, patch_size, depth_2//4, depth_3//4], stddev = 0.1, name="L3_ConvF")),
            'L3_Conv1to1F': tf.Variable(tf.truncated_normal(shape=[1, 1, depth_3//4, depth_3], stddev = 0.1, name="L3_Conv1to1F")),
            'L4_W': tf.get_variable("L4_W", shape=[hidden_n_input, hidden_n_output_1], initializer=xavier_init(hidden_n_input, hidden_n_output_1)),
            'L5_W': tf.get_variable("L5_W", shape=[hidden_n_output_1, hidden_n_output_2], initializer=xavier_init(hidden_n_output_1, hidden_n_output_2)),
            'L6_W': tf.get_variable("L6_W", shape=[hidden_n_output_2, num_labels], initializer=xavier_init(hidden_n_output_2, num_labels)),
        }
        b = {
            'L1_b': tf.Variable(tf.zeros([depth_1])),
            'L2_b': tf.Variable(tf.zeros([depth_2//4])),
            'L3_b': tf.Variable(tf.zeros([depth_3//4])),
            'L4_b': tf.Variable(tf.random_normal([hidden_n_output_1])),
            'L5_b': tf.Variable(tf.random_normal([hidden_n_output_2])),
            'L6_b': tf.Variable(tf.random_normal([num_labels])),
        }
        return W, b

    def create_nn_1to1(X, W, b, dropout):
        l1_conv = tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(
                                tf.nn.conv2d(X, W['L1_ConvF'], [1, 1, 1, 1], padding='SAME'),
                                b['L1_b'])), dropout, name="l1_conv")
        l1_conv1to1 = tf.nn.relu(tf.nn.conv2d(l1_conv, W['L1_Conv1to1F'], [1, 1, 1, 1], padding='SAME'))
        l1_pool = tf.nn.max_pool(l1_conv1to1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        l2_conv = tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(
                                tf.nn.conv2d(l1_pool, W['L2_ConvF'], [1, 1, 1, 1], padding='SAME'),
                                b['L2_b'])), dropout, name="l2_conv")
        # TODO: double 1to1 is right?
        l2_conv1to1_1 = tf.nn.relu(tf.nn.conv2d(l2_conv, W['L2_Conv1to1F1'], [1, 1, 1, 1], padding='SAME'))
        l2_conv1to1_2 = tf.nn.relu(tf.nn.conv2d(l2_conv1to1_1, W['L2_Conv1to1F2'], [1, 1, 1, 1], padding='SAME'))
        l2_pool = tf.nn.max_pool(l2_conv1to1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        l3_conv = tf.nn.relu(tf.nn.dropout(tf.nn.bias_add(
                     tf.nn.conv2d(l2_pool, W['L3_ConvF'], [1, 1, 1, 1], padding='SAME'),
                     b['L3_b']), dropout, name="l3_conv"))
        l3_conv1to1 = tf.nn.relu(tf.nn.conv2d(l3_conv, W['L3_Conv1to1F'], [1, 1, 1, 1], padding='SAME'))
        l3_pool = tf.nn.avg_pool(l3_conv1to1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        l3_output = tf.reshape(l3_pool, [-1, W['L4_W'].get_shape().as_list()[0]])

        l4_hidden = tf.nn.dropout(tf.nn.relu(tf.matmul(l3_output, W['L4_W']) + b['L4_b']), dropout, name="l4_hidden")
        l5_hidden = tf.nn.dropout(tf.nn.relu(tf.matmul(l4_hidden, W['L5_W']) + b['L5_b']), dropout, name="l5_hidden")
        logits = tf.nn.softmax(tf.matmul(l5_hidden, W['L6_W']) + b['L6_b'], name="logits")
        out = {
               'logits': logits,
               'l1_conv': l1_conv,
               'l2_conv': l2_conv,
               'l3_conv': l3_conv,
               'l4_hidden': l4_hidden,
               'l5_hidden': l5_hidden,
            }
        return out

    # X. Glorot and Y. Bengio, "Understanding the difficulty of training deep feedforward neural networks", 2010.
    # http://www.deeplearning.net/tutorial/mlp.html#weight-initialization
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
        nn_internal = create_nn(X, W, b, dropout_prob)
        Y_pred = nn_internal['logits']

        beta = tf.constant(params['beta_regularization_value'], tf.float32, name="beta")
        # TODO: need to regularize 'L1_ConvF'?
        l2_loss = beta * (tf.nn.l2_loss(W['L4_W']) + tf.nn.l2_loss(W['L5_W']))
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(Y_pred), reduction_indices= 1)) + l2_loss

        global_step = tf.Variable(0, trainable=False)
        gd_learning_rate = tf.train.exponential_decay(params['starter_learning_rate'], global_step, 100000, 0.96, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(gd_learning_rate).minimize(cross_entropy)
        # Adadelta are "safer" because they don't depend so strongly on setting of learning rates
        # http://cs.stanford.edu/people/karpathy/convnetjs/demo/trainers.html
#         optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.001).minimize(cross_entropy)
#         optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

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

            print("epoch complete: epoch:%d step:%d num_training:%d time:%s" %
                   (epoch, accumulated_step, accumulated_n_training, timedelta(seconds=(time.time() - start_time))))
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
        def show_layer(layers):
            width = 10
            index = 1
            for i, layer in enumerate(layers, start=1):
                for j in range(layer.shape[3]):
                    if (index > width * width):
                        break
                    plt.subplot(width, width, index)
                    if (j == 0):
                        plt.title("L" + str(i))
                    index += 1
                    plt.axis('off')
                    plt.imshow(layer[0, :, :, j], cmap=plt.get_cmap('gray'))
            plt.show()

        batch_data = train_dataset[random_index[0:1], :, :, :]
        l1_conv_layer, l2_conv_layer, l3_conv_layer = session.run([nn_internal['l1_conv'], nn_internal['l2_conv'], nn_internal['l3_conv']],
                                                   feed_dict={X: batch_data, dropout_prob: 1})
        layers = [l1_conv_layer, l2_conv_layer, l3_conv_layer]
        show_layer(layers)
        batch_data = train_dataset[random_index[1:2], :, :, :]
        l1_conv_layer, l2_conv_layer, l3_conv_layer = session.run([nn_internal['l1_conv'], nn_internal['l2_conv'], nn_internal['l3_conv']],
                                                   feed_dict={X: batch_data, dropout_prob: 1})
        layers = [l1_conv_layer, l2_conv_layer, l3_conv_layer]
        show_layer(layers)

# Gabriel configuration in
# http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html?showComment=1462342783838#c3340894427566315850
learning_params = {
    'patch_size': 3,
    'depth_1': 8,
    'depth_2': 16,
    'depth_3': 32,
    'hidden_n_output_1': 256,
    'hidden_n_output_2': 128,
    'batch_size': 128,
    'dropout_probability': 0.7,
    'training_epochs': 300,
    'max_test_time': 60 * 30,
    'starter_learning_rate': 0.1,
    'beta_regularization_value': 0.001
}
run_conv(learning_params)


learning_params = {
    'patch_size': 3,
    'depth_1': 8,
    'depth_2': 16,
    'depth_3': 32,
    'batch_size': 128,
    'dropout_probability': 0.7,
    'training_epochs': 3,
    'max_test_time': 60 * 60 * 9,
    'starter_learning_rate': 0.1,
    'beta_regularization_value': 0.001
}
# epoch complete: step 1561 num_training 199936 time:0:01:40.440831
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.6621 / 0.8588
# Validation [CrossEntropy / Training Accuracy] 0.6533 / 0.8650
# epoch complete: step 3123 num_training 399872 time:0:03:33.332888
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.5380 / 0.8885
# Validation [CrossEntropy / Training Accuracy] 0.5383 / 0.8904
# epoch complete: step 4685 num_training 599808 time:0:05:24.742359
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.4673 / 0.8967
# Validation [CrossEntropy / Training Accuracy] 0.4617 / 0.9004
# Complete training. Total time:0:05:46.011102
# Testing [CrossEntropy / Training Accuracy] 0.3068 / 0.9508

# 171k parameters
learning_params = {
    'patch_size': 3,
    'depth_1': 8,
    'depth_2': 16,
    'depth_3': 32,
    'hidden_n_output_1': 256,
    'hidden_n_output_2': 128,
    'batch_size': 128,
    'dropout_probability': 0.7,
    'training_epochs': 10000,
    'max_test_time': 60 * 60 * 9,
    'starter_learning_rate': 0.1,
    'beta_regularization_value': 0.001
}
# epoch complete: step 267101 num_training 34189056 time:8:48:56.779424
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.2758 / 0.9344
# Validation [CrossEntropy / Training Accuracy] 0.2961 / 0.9263
# epoch complete: step 268663 num_training 34388992 time:8:51:04.722086
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.2702 / 0.9340
# Validation [CrossEntropy / Training Accuracy] 0.2911 / 0.9280
# epoch complete: step 270225 num_training 34588928 time:8:53:12.534637
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.2735 / 0.9335
# Validation [CrossEntropy / Training Accuracy] 0.2967 / 0.9250
# epoch complete: step 271787 num_training 34788864 time:8:55:17.339205
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.2740 / 0.9330
# Validation [CrossEntropy / Training Accuracy] 0.2921 / 0.9249
# epoch complete: step 273349 num_training 34988800 time:8:57:26.478109
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.2683 / 0.9348
# Validation [CrossEntropy / Training Accuracy] 0.2926 / 0.9286
# epoch complete: step 274911 num_training 35188736 time:8:59:38.210218
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.2729 / 0.9347
# Validation [CrossEntropy / Training Accuracy] 0.2935 / 0.9288
# epoch complete: step 276473 num_training 35388672 time:9:01:38.853269
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.2745 / 0.9331
# Validation [CrossEntropy / Training Accuracy] 0.2966 / 0.9262
# BREAK: TIME OVER
# Complete training. Total time:9:01:57.943160
# Testing [CrossEntropy / Training Accuracy] 0.1643 / 0.9698
