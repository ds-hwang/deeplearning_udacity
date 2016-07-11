
# coding: utf-8

# Deep Learning
# =============
#
# Assignment 1
# ------------
#
# The objective of this assignment is to learn about simple data curation practices, and familiarize you with some of the data we'll be reusing later.
#
# This notebook uses the [notMNIST](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html) dataset to be used with python experiments. This dataset is designed to look like the classic [MNIST](http://yann.lecun.com/exdb/mnist/) dataset, while looking a little more like real data: it's a harder task, and the data is a lot less 'clean' than MNIST.

# In[ ]:

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

# Config the matlotlib backend as plotting inline in IPython
# get_ipython().magic(u'matplotlib inline')


# First, we'll download the dataset to our local machine. The data consists of characters rendered in a variety of fonts on a 28x28 image. The labels are limited to 'A' through 'J' (10 classes). The training set has about 500k and the testset 19000 labelled examples. Given these sizes, it should be possible to train models quickly on any machine.

# In[ ]:

url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None

def download_progress_hook(count, blockSize, totalSize):
    """A hook to report the progress of a download. This is mostly intended for users with
    slow internet connections. Reports every 1% change in download progress.
    """
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

        last_percent_reported = percent

def maybe_download(filename, expected_bytes, force=False):
    """Download a file if not present, and make sure it's the right size."""
    if force or not os.path.exists(filename):
        print('Attempting to download:', filename)
        filename, _ = urlretrieve(url + filename, filename, reporthook=download_progress_hook)
        print('\nDownload Complete!')
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        raise Exception(
          'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename

train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)


# Extract the dataset from the compressed .tar.gz file.
# This should give you a set of directories, labelled A through J.

# In[ ]:

num_classes = 10
np.random.seed(133)

def maybe_extract(filename, force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall()
        tar.close()
    data_folders = [
      os.path.join(root, d) for d in sorted(os.listdir(root))
      if os.path.isdir(os.path.join(root, d))]
    if len(data_folders) != num_classes:
        raise Exception(
          'Expected %d folders, one per class. Found %d instead.' % (
            num_classes, len(data_folders)))
    print(data_folders)
    return data_folders

train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)


# ---
# Problem 1
# ---------
#
# Let's take a peek at some of the data to make sure it looks sensible. Each exemplar should be an image of a character A through J rendered in a different font. Display a sample of the images that we just downloaded. Hint: you can use the package IPython.display.
#
# ---

def show_examples(data_folders):
    sample_size = 10
    images = []
    for folder in data_folders:
        image_files = os.listdir(folder)
        for image_index, image in enumerate(image_files):
            image_file = os.path.join(folder, image)
            if image_index == sample_size:
                break
            images.append(image_file)

    for image_index, image in enumerate(images, start=1):
        plt.subplot(10, sample_size, image_index)
        img = plt.imread(image)
        plt.axis('off')
        plt.imshow(img, cmap=plt.cm.winter)
    plt.show()
#show_examples(train_folders)

# Now let's load the data in a more manageable format. Since, depending on your computer setup you might not be able to fit it all in memory, we'll load each class into a separate dataset, store them on disk and curate them independently. Later we'll merge them into a single dataset of manageable size.
#
# We'll convert the entire dataset into a 3D array (image index, x, y) of floating point values, normalized to have approximately zero mean and standard deviation ~0.5 to make training easier down the road.
#
# A few images might not be readable, we'll just skip them.

# In[ ]:

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                           dtype=np.float32)
    print(folder)
    for image_index, image in enumerate(image_files):
        image_file = os.path.join(folder, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) -
                          pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[image_index, :, :] = image_data
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    num_images = image_index + 1
    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset

def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names

train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)


# ---
# Problem 2
# ---------
#
# Let's verify that the data still looks good. Displaying a sample of the labels and images from the ndarray. Hint: you can use matplotlib.pyplot.
#
# ---

def show_pickle(pickle_files):
    sample_size = 10
    num_classes = len(pickle_files)
    images = np.ndarray((sample_size * num_classes, image_size, image_size),
                         dtype=np.float32)
    start_v = 0
    end_v = sample_size
    for _, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                images[start_v:end_v, :, :] = letter_set[0:sample_size, :, :]
                start_v += sample_size
                end_v += sample_size
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    for i in range(images.shape[0]):
        plt.subplot(10, sample_size, i + 1)
        plt.axis('off')
        plt.imshow(images[i, :, :], cmap=plt.cm.gray)
    plt.show()
# show_pickle(train_datasets)

# ---
# Problem 3
# ---------
# Another check: we expect the data to be balanced across classes. Verify that.
#
# ---

def check_balance(pickle_files):
    tolerance = 10
    num_classes = len(pickle_files)
    num_data = []
    for _, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                num_data.append(letter_set.shape[0])
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    mean = np.mean(num_data)
    for label, num in enumerate(num_data):
        if (abs(mean - num) > tolerance):
            print('class %s: num %s while mean %s' % (label, num, mean))
    plt.show()
check_balance(train_datasets)
check_balance(test_datasets)


# Merge and prune the training data as needed. Depending on your computer setup, you might not be able to fit it all in memory, and you can tune `train_size` as needed. The labels will be stored into a separate array of integers 0 through 9.
#
# Also create a validation dataset for hyperparameter tuning.

# In[ ]:

def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class+tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels


pickle_file = 'notMNIST.pickle'

train_size = 200000
valid_size = 10000
test_size = 10000


# Next, we'll randomize the data. It's important to have the labels well shuffled for the training and test distributions to match.

# In[ ]:

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

train_dataset = None
train_labels = None
test_dataset = None
test_labels = None
valid_dataset = None
valid_labels = None
if not os.path.exists(pickle_file):
    valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
      train_datasets, train_size, valid_size)
    _, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

    train_dataset, train_labels = randomize(train_dataset, train_labels)
    test_dataset, test_labels = randomize(test_dataset, test_labels)
    valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
else:
    print('Load compressed pickle to reuse')
    try:
        with open(pickle_file, 'rb') as f:
            save = pickle.load(f)
            train_dataset = save['train_dataset']
            train_labels = save['train_labels']
            valid_dataset = save['valid_dataset']
            valid_labels = save['valid_labels']
            test_dataset = save['test_dataset']
            test_labels = save['test_labels']
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

# ---
# Problem 4
# ---------
# Convince yourself that the data is still good after shuffling!
#
# ---

def show_dataset(dataset, labels):
    sample_size = 10
    for i in range(num_classes * sample_size):
        plt.subplot(10, sample_size, i + 1)
        plt.title(chr(ord('A') + labels[i]))
        plt.axis('off')
        plt.imshow(dataset[i, :, :], cmap=plt.cm.gray)
    plt.show()
# show_dataset(train_dataset, train_labels)
# show_dataset(test_dataset, test_labels)
# show_dataset(valid_dataset, valid_labels)


# Finally, let's save the data for later reuse:

# In[ ]:

try:
    if os.path.exists(pickle_file):
        print('%s already present - Skipping saving compressed pickle.'
              % pickle_file)
    else:
        f = open(pickle_file, 'wb')
        save = {
          'train_dataset': train_dataset,
          'train_labels': train_labels,
          'valid_dataset': valid_dataset,
          'valid_labels': valid_labels,
          'test_dataset': test_dataset,
          'test_labels': test_labels,
          }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()

        statinfo = os.stat(pickle_file)
        print('Compressed pickle size:', statinfo.st_size)
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise


# In[ ]:



# ---
# Problem 5
# ---------
#
# By construction, this dataset might contain a lot of overlapping samples, including training data that's also contained in the validation and test set! Overlap between training and test can skew the results if you expect to use your model in an environment where there is never an overlap, but are actually ok if you expect to see training samples recur when you use it.
# Measure how much overlap there is between training, validation and test samples.
#
# Optional questions:
# - What about near duplicates between datasets? (images that are almost identical)
# - Create a sanitized validation and test set, and compare your accuracy on those in subsequent assignments.
# ---




# ---
# Problem 6
# ---------
#
# Let's get an idea of what an off-the-shelf classifier can give you on this data. It's always good to check that there is something to learn, and that it's a problem that is not so trivial that a canned solution solves it.
#
# Train a simple model on this data using 50, 100, 1000 and 5000 training samples. Hint: you can use the LogisticRegression model from sklearn.linear_model.
#
# Optional question: train an off-the-shelf model on all the data!
#
# ---
train_dataset_flat = train_dataset.reshape(train_dataset.shape[0], -1)
valid_dataset_flat = valid_dataset.reshape(valid_dataset.shape[0], -1)
test_dataset_flat = test_dataset.reshape(test_dataset.shape[0], -1)
num_feature = train_dataset_flat.shape[1]

# 1. sklearn LinearRegression
# Refer to scipy_2015_sklearn_tutorial/notebooks/02.1%20Supervised%20Learning%20-%20Classification.ipynb
# add 200000 if you want
if False:
    for i in [50, 100, 1000, 5000]:
        train_dataset_flat_new = train_dataset_flat[:i, :]
        train_labels_new = train_labels[:i]

        regressor = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        regressor.fit(train_dataset_flat_new, train_labels_new)
        train_pred_labels = regressor.predict(train_dataset_flat_new)
        valid_pred_labels = regressor.predict(valid_dataset_flat)
        test_pred_labels = regressor.predict(test_dataset_flat)
        print("num of training set:", i)
        print("train score: %s mean: %s" %
              (regressor.score(train_dataset_flat_new, train_labels_new),
               np.mean(train_pred_labels == train_labels_new)))
        print("valid score: %s mean: %s" %
              (regressor.score(valid_dataset_flat, valid_labels),
               np.mean(valid_pred_labels == valid_labels)))
        print("test score: %s mean: %s" %
              (regressor.score(test_dataset_flat, test_labels),
               np.mean(test_pred_labels == test_labels)))
## Results
# num of training set: 50
# train score: 1.0 mean: 1.0
# valid score: 0.5998 mean: 0.5998
# test score: 0.6516 mean: 0.6516
# num of training set: 100
# train score: 1.0 mean: 1.0
# valid score: 0.7206 mean: 0.7206
# test score: 0.788 mean: 0.788
# num of training set: 1000
# train score: 0.999 mean: 0.999
# valid score: 0.7754 mean: 0.7754
# test score: 0.845 mean: 0.845
# num of training set: 5000
# train score: 0.9728 mean: 0.9728
# valid score: 0.7703 mean: 0.7703
# test score: 0.8446 mean: 0.8446
# num of training set: 200000
# train score: 0.83876 mean: 0.83876
# valid score: 0.8333 mean: 0.8333
# test score: 0.9012 mean: 0.9012


# 2. TensorFlow Logistic regression
import tensorflow as tf

train_labels_n = np.transpose(np.array([(train_labels == i).astype(int) for i in range(10)]))
valid_labels_n = np.transpose(np.array([(valid_labels == i).astype(int) for i in range(10)]))
test_labels_n = np.transpose(np.array([(test_labels == i).astype(int) for i in range(10)]))

print('TensorFlow Logistic regression')
print('Training:', train_dataset_flat.shape, train_labels_n.shape)
print('Validation:', valid_dataset_flat.shape, valid_labels_n.shape)
print('Testing:', test_dataset_flat.shape, test_labels_n.shape)

X = tf.placeholder(tf.float32, [None, num_feature], name="input")
Y = tf.placeholder(tf.float32, [None, num_classes], name="output")
W = tf.Variable(tf.zeros([num_feature, num_classes]), name="weight")
b = tf.Variable(tf.zeros([num_classes]), name="bias")

Y_pred = tf.nn.softmax(tf.matmul(X, W) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(Y_pred), reduction_indices= 1))

learning_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
training_epochs = 5 #50
display_epoch = 1 #5
batch_size = 100   # For each time, we will use 100 samples to update parameters
n_train = train_dataset_flat.shape[0]

correct_prediction = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

for i in [100, 1000, 5000, n_train]:
    print("num of training set:", i)
    train_dataset_flat_new = train_dataset_flat[:i, :]
    train_labels_new = train_labels_n[:i, :]
    new_n_train = train_dataset_flat_new.shape[0]
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(training_epochs):
            num_batch  = int(new_n_train / batch_size)
            ran_idx =  np.random.permutation(new_n_train)
            for i in range(num_batch):
                X_batch = train_dataset_flat_new[ran_idx[i * batch_size: (i + 1) * batch_size], :]
                Y_batch = train_labels_new[ran_idx[i * batch_size: (i + 1) * batch_size], :]
                sess.run(optimizer, feed_dict = {X: X_batch, Y: Y_batch})

            if (epoch+1) % display_epoch == 0:
                cross_entropy_value = sess.run(cross_entropy, feed_dict={X: train_dataset_flat_new, Y: train_labels_new})
                accuracy_value = accuracy.eval({X: train_dataset_flat_new, Y: train_labels_new})
                print("(epoch {})".format(epoch+1))
                print("[CrossEntropy / Training Accuracy] {:05.4f} / {:05.4f}".format(cross_entropy_value, accuracy_value))
                print(" ")

        print("Complete training")
        cross_entropy_value = sess.run(cross_entropy, feed_dict={X: train_dataset_flat_new, Y: train_labels_new})
        accuracy_value = accuracy.eval({X: train_dataset_flat_new, Y: train_labels_new})
        print("Train [CrossEntropy / Training Accuracy] {:05.4f} / {:05.4f}".format(cross_entropy_value, accuracy_value))
        cross_entropy_valid = sess.run(cross_entropy, feed_dict={X: valid_dataset_flat, Y: valid_labels_n})
        accuracy_valid = accuracy.eval({X: valid_dataset_flat, Y: valid_labels_n})
        print("Validation [CrossEntropy / Training Accuracy] {:05.4f} / {:05.4f}".format(cross_entropy_valid, accuracy_valid))
        cross_entropy_test = sess.run(cross_entropy, feed_dict={X: test_dataset_flat, Y: test_labels_n})
        accuracy_test = accuracy.eval({X: test_dataset_flat, Y: test_labels_n})
        print("Testing [CrossEntropy / Training Accuracy] {:05.4f} / {:05.4f}".format(cross_entropy_test, accuracy_test))

# Results
# Complete training
# Train [CrossEntropy / Training Accuracy] 0.6219 / 0.8354
# Validation [CrossEntropy / Training Accuracy] 0.6332 / 0.8320
# Testing [CrossEntropy / Training Accuracy] 0.3952 / 0.8970
