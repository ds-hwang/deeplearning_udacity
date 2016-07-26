# coding: utf-8

# The goal of this assignment is to train a Word2Vec skip-gram model over [Text8](http://mattmahoney.net/dc/textdata) data.

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from sklearn.manifold import TSNE

def get_word2vec(epochs, override):
    pickle_file = 'word2vec.pickle'
    if os.path.exists(pickle_file) and not override:
        print('%s already present' % pickle_file)
        with open(pickle_file, 'rb') as f:
            save = pickle.load(f)
            data = save['data']
            count = save['count']
            dictionary = save['dictionary']
            embeddings = save['embeddings']
            normalized_embeddings = save['normalized_embeddings']
            weights = save['weights']
            biases = save['biases']
            del save  # hint to help gc free up memory
        return data, count, dictionary, embeddings, normalized_embeddings, weights, biases

    # Download the data from the source website if necessary.
    url = 'http://mattmahoney.net/dc/'

    def maybe_download(filename, expected_bytes):
        """Download a file if not present, and make sure it's the right size."""
        if not os.path.exists(filename):
            filename, _ = urlretrieve(url + filename, filename)
        statinfo = os.stat(filename)
        if statinfo.st_size == expected_bytes:
            print('Found and verified %s' % filename)
        else:
            print(statinfo.st_size)
            raise Exception(
              'Failed to verify ' + filename + '. Can you get to it with a browser?')
        return filename

    filename = maybe_download('text8.zip', 31344016)

    # Read the data into a string.
    def read_data(filename):
        """Extract the first file enclosed in a zip file as a list of words"""
        with zipfile.ZipFile(filename) as f:
            data = tf.compat.as_str(f.read(f.namelist()[0])).split()
        return data

    words = read_data(filename)
    # 17005207
    print('Data size %d' % len(words))

    # Build the dictionary and replace rare words with UNK token.
    vocabulary_size = 50000

    def build_dataset(words):
        count = [['UNK', -1]]
        count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = list()
        unk_count = 0
        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count = unk_count + 1
            data.append(index)
        count[0][1] = unk_count
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return data, count, dictionary, reverse_dictionary

    data, count, dictionary, reverse_dictionary = build_dataset(words)
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10])
    del words  # Hint to reduce memory.

    def generate_batch(index, batch_size, num_skips, skip_window):
        data_index = index
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1 # [ skip_window target skip_window ]
        windows_buffer = collections.deque(maxlen=span)
        for _ in range(span):
            windows_buffer.append(data[data_index])
            data_index = (data_index + 1) % len(data)
        for i in range(batch_size // num_skips):
            target = skip_window  # target label at the center of the windows_buffer
            targets_to_avoid = [ skip_window ]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * num_skips + j] = windows_buffer[skip_window]
                labels[i * num_skips + j, 0] = windows_buffer[target]
            windows_buffer.append(data[data_index])
            data_index = (data_index + 1) % len(data)
        return batch, labels, data_index

    print('data:', [reverse_dictionary[di] for di in data[:8]])

    for batch_size, num_skips, skip_window in [(8, 2, 1), (8, 4, 2), (12, 6, 3), (16, 8, 4)]:
        batch, labels, _ = generate_batch(0, batch_size=batch_size, num_skips=num_skips, skip_window=skip_window)
        print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
        print('    batch:', [reverse_dictionary[bi] for bi in batch])
        print('    labels:', [reverse_dictionary[li] for li in labels.reshape(batch_size)])

    # ('Most common words (+UNK)', [['UNK', 418391], ('the', 1061396), ('of', 593677), ('and', 416629), ('one', 411764)])
    # ('Sample data', [5239, 3084, 12, 6, 195, 2, 3137, 46, 59, 156])
    # ('data:', ['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse', 'first'])
    #
    # with num_skips = 2 and skip_window = 1:
    # ('    batch:', ['originated', 'originated', 'as', 'as', 'a', 'a', 'term', 'term'])
    # ('    labels:', ['anarchism', 'as', 'originated', 'a', 'as', 'term', 'a', 'of'])
    #
    # with num_skips = 4 and skip_window = 2:
    # ('    batch:', ['as', 'as', 'as', 'as', 'a', 'a', 'a', 'a'])
    # ('    labels:', ['originated', 'term', 'anarchism', 'a', 'originated', 'of', 'as', 'term'])

    # Train a skip-gram model.

    # In[ ]:


    def run_skip_gram():
        batch_size = 126
        num_sampled = 64 # Number of negative examples to sample.

        # according to http://cs224d.stanford.edu/lectures/CS224d-Lecture3.pdf
        # 17005207 (i.e 17M) -> 1B
        embedding_size = 300 # Dimension of the embedding vector.
        skip_window = 3 # How many words to consider left and right.
        num_skips = 6 # How many times to reuse an input to generate a label.

        # We pick a random validation set to sample nearest neighbors. here we limit the
        # validation samples to the words that have a low numeric ID, which by
        # construction are also the most frequent.
        valid_size = 16 # Random set of words to evaluate similarity on.
        valid_window = 100 # Only pick dev samples in the head of the distribution.
        valid_examples = np.array(random.sample(range(valid_window), valid_size))

        graph = tf.Graph()
        with graph.as_default(), tf.device('/cpu:0'):

            # Input data.
            train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
            # random 16 samples
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

            # Variables.
            # vocabulary_size = 50000, embedding_size = 128
            # embeddings == U (i.e. input vector) in cs224d
            embeddings = tf.Variable(
              tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            # weights == V (i.e. output vector) in cs224d
            softmax_weights = tf.Variable(
              tf.truncated_normal([vocabulary_size, embedding_size],
                                   stddev=1.0 / math.sqrt(embedding_size)))
            softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

            # Model.
            # Look up embeddings for inputs.
            embed = tf.nn.embedding_lookup(embeddings, train_dataset)
            # Compute the softmax loss, using a sample of the negative labels each time.
            loss = tf.reduce_mean(
              tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, embed,
                                         train_labels, num_sampled, vocabulary_size))

            # Optimizer.
            optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

            # Compute the similarity between minibatch examples and all embeddings.
            # We use the cosine distance:
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(
              normalized_embeddings, valid_dataset)
            # cosine distance
            similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

        with tf.Session(graph=graph) as session:
            tf.initialize_all_variables().run()
            print('Initialized')
            average_loss = 0
            n_train = len(data) * num_skips
            n_batch = n_train / batch_size
            data_index = 0
            for epoch in range(int(math.ceil(epochs))):
                # epochs can be 0.001 to test overfit
                fraction = epochs - epoch
                if (fraction) < 1:
                    n_batch = n_batch * fraction
                total_step = int(math.ceil(n_batch))
                for step in range(total_step):
                    batch_data, batch_labels, data_index = generate_batch(
                      data_index, batch_size, num_skips, skip_window)
                    feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
                    _, l = session.run([optimizer, loss], feed_dict=feed_dict)
                    average_loss += l

                    threshold_print = 50000 / batch_size
                    if step % threshold_print == 0:
                        if step > 0:
                            average_loss = average_loss / 2000
                        # The average loss is an estimate of the loss over the last 2000 batches.
                        print('Average loss at step/progress %d, %f: %f' %
                               (step, (1.0 * step / total_step), average_loss))
                        average_loss = 0
                    threshold_print = 2000000 / batch_size
                    # note that this is expensive (~20% slowdown if computed every 500 steps)
                    if step % 10000 == 0:
                        sim = similarity.eval()
                        for i in range(valid_size):
                            valid_word = reverse_dictionary[valid_examples[i]]
                            top_k = 8 # number of nearest neighbors
                            nearest = (-sim[i, :]).argsort()[1:top_k+1] # - for descending
                            log = 'Nearest to %s:' % valid_word
                            for k in range(top_k):
                                close_word = reverse_dictionary[nearest[k]]
                                log = '%s %s,' % (log, close_word)
                            print(log)

            return embeddings.eval(), normalized_embeddings.eval(), softmax_weights.eval(), softmax_biases.eval()

    embeddings, normalized_embeddings, weights, biases = run_skip_gram()

    with open(pickle_file, 'wb') as f:
        save = {
          'data': data,
          'count': count,
          'dictionary': reverse_dictionary,
          'embeddings': embeddings,
          'normalized_embeddings': normalized_embeddings,
          'weights': weights,
          'biases': biases,
          }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)

        statinfo = os.stat(pickle_file)
        print('Compressed pickle size:', statinfo.st_size)

    return data, count, reverse_dictionary, embeddings, normalized_embeddings, weights, biases

if __name__ == "__main__":
    data, count, dictionary, embeddings, normalized_embeddings, weights, biases = get_word2vec(2, False)
    # Average loss at step/progress 797148, 0.984412: 0.709467
    # Average loss at step/progress 797544, 0.984901: 0.710918
    # Average loss at step/progress 797940, 0.985390: 0.700519
    # Average loss at step/progress 798336, 0.985879: 0.681821
    # Average loss at step/progress 798732, 0.986368: 0.707560
    # Average loss at step/progress 799128, 0.986857: 0.686896
    # Average loss at step/progress 799524, 0.987346: 0.636758
    # Average loss at step/progress 799920, 0.987835: 0.707642
    # Nearest to be: have, been, is, are, easily, serve, were, that,
    # Nearest to with: by, when, in, s, between, UNK, while, two,
    # Nearest to of: the, and, in, to, for, as, or, one,
    # Nearest to used: referred, employed, preferred, applied, spoken, using, use, added,
    # Nearest to have: has, had, be, are, require, having, contain, hold,
    # Nearest to six: seven, eight, five, four, three, nine, two, zero,
    # Nearest to however: but, although, though, when, that, because, this, if,
    # Nearest to from: in, through, while, including, across, on, UNK, after,
    # Nearest to some: many, all, several, most, various, these, they, other,
    # Nearest to while: although, when, though, where, but, still, whilst, however,
    # Nearest to it: he, she, this, there, only, that, they, but,
    # Nearest to four: three, five, seven, six, eight, two, zero, nine,
    # Nearest to zero: five, four, eight, seven, six, nine, three, two,
    # Nearest to he: she, it, they, his, him, who, her, himself,
    # Nearest to often: generally, sometimes, frequently, typically, usually, commonly, traditionally, normally,
    # Nearest to are: were, is, all, have, other, be, include, these,

    num_points = 400
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    two_d_embeddings = tsne.fit_transform(normalized_embeddings[1:num_points+1, :])

    def plot(embeddings, labels):
        assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
        pylab.figure(figsize=(15,15))  # in inches
        for i, label in enumerate(labels):
            x, y = embeddings[i,:]
            pylab.scatter(x, y)
            pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                           ha='right', va='bottom')
        pylab.show()

    words = [dictionary[i] for i in range(1, num_points+1)]
    plot(two_d_embeddings, words)
