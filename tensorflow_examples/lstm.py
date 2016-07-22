# coding: utf-8

# Deep Learning
# =============
#
# Assignment 6
# ------------
#
# After training a skip-gram model in `5_word2vec.ipynb`,
# the goal of this notebook is to train a LSTM character model
# over [Text8](http://mattmahoney.net/dc/textdata) data.

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import math
import numpy as np
import random
import os
import tensorflow as tf
import time
from datetime import timedelta
import word2vec

def lstm(params):
    data, count, dictionary, embeddings, normalized_embeddings, weights, biases = word2vec.get_word2vec(2, False)
    words_size = embeddings.shape[0]
    embedding_size = embeddings.shape[1]
    print('embedding size:%s data:%s' % (embedding_size, [dictionary[word] for word in data[:100]]))

    # Create a small validation set.
    valid_size = 1000
    valid_text = data[:valid_size]
    train_text = data[valid_size:]
    train_size = len(train_text)

    p_num_unrollings = params['num_unrollings']
    p_batch_size = params['batch_size']

    class BatchGenerator(object):
        def __init__(self, text, batch_size, num_unrollings):
            assert batch_size >= 1
            assert num_unrollings >= 1
            self._text = text
            self._text_size = len(text)
            self._batch_size = batch_size
            self._num_unrollings = num_unrollings
            segment = self._text_size // batch_size
            self._cursor_boundary = [offset * segment for offset in range(batch_size)]
            self._cursor = self._cursor_boundary[:]
            self._last_batch = self._next_batch()

        def _next_batch(self):
            """Generate a single batch from the current cursor position in the data."""
            batch = np.zeros(shape=(self._batch_size, embedding_size), dtype=np.float)
            for b in range(self._batch_size):
                batch[b] = embeddings[self._text[self._cursor[b]]]
                self._cursor[b] = (self._cursor[b] + 1)
            if self._cursor[self._batch_size - 1] == self._text_size:
                self._cursor = self._cursor_boundary[:]
            return batch

        def next(self):
            """Generate the next array of batches from the data. The array consists of
            the last batch of the previous array, followed by p_num_unrollings new ones.
            """
            batches = [self._last_batch]
            for _ in range(self._num_unrollings):
                batches.append(self._next_batch())
            self._last_batch = batches[-1]
            return batches

    def batches2string(batches):
        """Convert a sequence of batches back into their (most likely) string
        representation."""
        s = [''] * batches[0].shape[0]
        for b in batches:
            words = [dictionary[w] for w in np.argmax(np.matmul(b, normalized_embeddings.T), 1)]
            s = [' '.join(x) for x in zip(s, words)]
        return s

    train_batches = BatchGenerator(train_text, p_batch_size, p_num_unrollings)
    valid_batches = BatchGenerator(valid_text, 1, 1)

    print(batches2string(train_batches.next()))
    print(batches2string(train_batches.next()))
    print(batches2string(train_batches.next()))
    print(batches2string(valid_batches.next()))
    print(batches2string(valid_batches.next()))
    print(batches2string(valid_batches.next()))

    def logprob(predictions, labels):
        """Log-probability of the true labels in a predicted batch."""
        predictions[predictions < 1e-10] = 1e-10
        return np.sum(-np.log([predictions[i, label] for i, label in enumerate(labels)])) / labels.shape[0]

    graph = tf.Graph()
    with graph.as_default():
        p_num_nodes = params['num_nodes']
        p_max_k = params['max_k']
        def create_trainable_variables():
            '''
            Parameters:
                num_nodes*0:num_nodes*1 : Input gate
                num_nodes*1:num_nodes*2 : Forget gate
                num_nodes*2:num_nodes*3 : Output gate
                num_nodes*3:num_nodes*4 : New memory cell
            '''
            W = {
                'L1_W': tf.Variable(tf.truncated_normal([embedding_size, p_num_nodes * 4], mean=0, stddev=0.1, name="L1_W")),
                'L1_U': tf.Variable(tf.truncated_normal([p_num_nodes, p_num_nodes * 4], mean=0, stddev=0.1, name="L1_U")),
                'L1_b': tf.Variable(tf.zeros([1, p_num_nodes * 4]), name="L1_b"),
                'L2_W': tf.Variable(tf.truncated_normal([p_num_nodes, p_num_nodes * 4], mean=0, stddev=0.1, name="L2_W")),
                'L2_U': tf.Variable(tf.truncated_normal([p_num_nodes, p_num_nodes * 4], mean=0, stddev=0.1, name="L2_U")),
                'L2_b': tf.Variable(tf.zeros([1, p_num_nodes * 4]), name="L2_b"),
                'L3_W': tf.Variable(tf.truncated_normal([p_num_nodes, embedding_size], mean=0, stddev=0.1, name="L2_W")),
                'L3_b': tf.Variable(tf.zeros([embedding_size]), name="L3_b"),
            }

            return W

        def create_variables(batch_size, num_unrollings):
            # Input data.
            train_data = list()
            for _ in range(num_unrollings + 1):
                train_data.append(
                  tf.placeholder(tf.float32, shape=[batch_size, embedding_size]))

            inputs = {
                'inputs': train_data[:num_unrollings],
                'labels': train_data[1:],  # labels are inputs shifted by one time step.
                'data': train_data,
                'dropout': tf.placeholder(tf.float32, name="dropout"),
            }

            # Variables saving state across unrollings.
            last_state = {
                'h1': tf.Variable(tf.zeros([batch_size, p_num_nodes]), trainable=False, name="h1"),
                'c1': tf.Variable(tf.zeros([batch_size, p_num_nodes]), trainable=False, name="c1"),
                'h2': tf.Variable(tf.zeros([batch_size, p_num_nodes]), trainable=False, name="h2"),
                'c2': tf.Variable(tf.zeros([batch_size, p_num_nodes]), trainable=False, name="c2"),
            }

            return inputs, last_state

        # Definition of the cell computation.
        def lstm_cell(x, h, c, W, U, b):
            """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
            Note that in this formulation, we omit the various connections between the
            previous c (i.e. state) and the gates."""
            raw_data = tf.matmul(x, W) + tf.matmul(h, U) + b
            gates = tf.sigmoid(raw_data[:, :p_num_nodes*3])
            input_gate = gates[:, :p_num_nodes] # p_batch_size x p_num_nodes
            forget_gate = gates[:, p_num_nodes:p_num_nodes*2] # p_batch_size x p_num_nodes
            output_gate = gates[:, p_num_nodes*2:p_num_nodes*3] # p_batch_size x p_num_nodes
            new_memory_cell = raw_data[:, p_num_nodes*3:] # p_batch_size x p_num_nodes
            c_next = forget_gate * c + input_gate * tf.tanh(new_memory_cell) # p_batch_size x p_num_nodes
            h_next = output_gate * tf.tanh(c_next)
            return h_next, c_next

        def create_model(W, inputs, last_state, norm_embeddings):
            h2s = list()
            h1 = last_state['h1']
            c1 = last_state['c1']
            h2 = last_state['h2']
            c2 = last_state['c2']
            # construct 2 layer LSTM
            for x in inputs['inputs']:
                h1, c1 = lstm_cell(x, h1, c1, W['L1_W'], W['L1_U'], W['L1_b'])
                x2 = tf.nn.dropout(h1, inputs['dropout'], name="dropout")
                h2, c2 = lstm_cell(x2, h2, c2, W['L2_W'], W['L2_U'], W['L2_b'])
                h2s.append(h2)

            def gather_2d(params, indices):
                # we want output[i] = params[i,indices[i]]
                params_shape = params.get_shape().as_list()
                indices_flattened = tf.cast(tf.range(0, params_shape[0]) * params_shape[1], tf.int64) + indices
                return tf.gather(tf.reshape(params, [-1]),  # flatten input
                              indices_flattened)  # use flattened indices

            # State saving across unrollings.
            with tf.control_dependencies([last_state['h1'].assign(h1),
                                          last_state['c1'].assign(c1),
                                          last_state['h2'].assign(h2),
                                          last_state['c2'].assign(c2)]):
                # Classifier.
                logits = tf.nn.xw_plus_b(tf.concat(0, h2s), W['L3_W'], W['L3_b'])
                # 640 x 50000
                logits_onehot = tf.matmul(logits, norm_embeddings)
                # stabilize numerical computation (i.e. big_num / big_num)
                logits_onehot = logits_onehot - tf.expand_dims(tf.reduce_max(logits_onehot, reduction_indices=[1]), 1)
                logits_exp = tf.exp(logits_onehot)
                # 640
                logits_denominator = tf.reduce_sum(logits_exp, 1)
                # 640
                Y = tf.argmax(tf.matmul(tf.concat(0, inputs['labels']), norm_embeddings), 1)
                # 640
                Y_pred = gather_2d(logits_exp, Y)
                # manual softmax to compute only Y element.
                Y_pred = tf.div(Y_pred, logits_denominator)
                loss = tf.reduce_mean(-tf.log(Y_pred))

            model = {
               'loss': loss,
               'Y_pred': Y_pred,
               'logits_exp': logits_exp,
            }
            return model

        # Convert vec to word
        norm_embeddings = tf.constant(normalized_embeddings.T)

        W = create_trainable_variables()
        inputs, last_state = create_variables(p_batch_size, p_num_unrollings)

        # Unrolled LSTM loop.
        model = create_model(W, inputs, last_state, norm_embeddings)

        # Optimizer.
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
          10.0, global_step, 5000, 0.1, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        gradients, v = zip(*optimizer.compute_gradients(model['loss']))
        gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
        optimizer = optimizer.apply_gradients(
          zip(gradients, v), global_step=global_step)

        # Sampling and validation eval: batch 1, no unrolling.
        sample_batch_size = 1
        sample_num_unrollings = 1
        sample_inputs, sample_last_state = create_variables(sample_batch_size, sample_num_unrollings)
        sample_model = create_model(W, sample_inputs, sample_last_state, norm_embeddings)
        reset_sample_state = tf.group(
          sample_last_state['h1'].assign(tf.zeros([sample_batch_size, p_num_nodes])),
          sample_last_state['c1'].assign(tf.zeros([sample_batch_size, p_num_nodes])),
          sample_last_state['h2'].assign(tf.zeros([sample_batch_size, p_num_nodes])),
          sample_last_state['c2'].assign(tf.zeros([sample_batch_size, p_num_nodes])))
        sample_next = tf.nn.top_k(sample_model['logits_exp'], p_max_k)[1]

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    p_epochs = params['epochs']
    p_summary_frequency = params['summary_frequency']
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('Initialized')
        if os.path.exists(params['savefile']) and params['resume']:
            # Restore variables from disk.
            saver.restore(session, params['savefile'])
            print("Model restored.")

        start_time = time.time()
        n_batch = len(data) // p_batch_size
        for epoch in range(int(math.ceil(p_epochs))):
            # p_epochs can be 0.001 to test overfit
            fraction = p_epochs - epoch
            if (fraction) < 1:
                n_batch = n_batch * fraction
            total_step = int(math.ceil(n_batch))
            mean_loss = 0
            print("Epoch %s start / total p_epochs %s, total steps %s" % (epoch, p_epochs, total_step))
            for step in range(total_step):
                batches = train_batches.next()
                inputs_dict = dict()
                for i in range(p_num_unrollings + 1):
                    inputs_dict[inputs['data'][i]] = batches[i]
                inputs_dict[inputs['dropout']] = params['dropout']
                _, loss_e, learning_rate_e = session.run(
                  [optimizer, model['loss'], learning_rate], feed_dict=inputs_dict)
                mean_loss += loss_e
                if step % p_summary_frequency == 0:
                    # Save the variables to disk.
                    save_path = saver.save(session, params['savefile'])

                    mean_loss = mean_loss / p_summary_frequency
                    # The mean loss is an estimate of the loss over the last few batches.
                    # PP = exp(CE) = exp(-log(prediction)) = 1/prediction. max PP = 1 / (1/50000) = 50000
                    print(
                      'Average loss at step(%d):%f perplexity:%.2f learning rate:%.2f time:%s saved:%s' %
                      (step, mean_loss, np.exp(mean_loss), learning_rate_e,
                        timedelta(seconds=(time.time() - start_time)), save_path))
                    mean_loss = 0

                    def sample(candiate_indices):
                        # check https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py#L62
                        k = int(abs(random.normalvariate(0, p_max_k/2))) % p_max_k
                        index = candiate_indices[k]
                        # Skip UNK
                        while index == 0:
                            k = int(abs(random.normalvariate(0, p_max_k/2))) % p_max_k
                            index = candiate_indices[k]
                        return index

                    if step % (p_summary_frequency * 10) == 0:
                        # Generate some samples.
                        print('=' * 80)
                        for _ in range(5):
                            word = int(random.uniform(0, 1) * words_size) % words_size
                            feed = np.array([embeddings[word]])
                            sentence = dictionary[word]
                            reset_sample_state.run()
                            for _ in range(79):
                                prediction = sample_next.eval({sample_inputs['inputs'][0]: feed, sample_inputs['dropout']: 1,})
                                index = sample(prediction[0, :])
                                feed = np.array([embeddings[index]])
                                sentence += ' ' + dictionary[index]
                            print(sentence)
                        print('=' * 80)

                        # Measure validation set perplexity.
                        valid_mean_loss = 0
                        reset_sample_state.run()
                        for _ in range(valid_size):
                            validation_batches = valid_batches.next()
                            sample_feeds = {
                                sample_inputs['inputs'][0]: validation_batches[0],
                                sample_inputs['labels'][0]: validation_batches[1],
                                sample_inputs['dropout']: 1,
                            }
                            valid_loss = session.run([sample_model['loss']],
                                             feed_dict=sample_feeds)
                            valid_mean_loss += valid_loss[0]
                        print('Validation set perplexity: %.2f.' %
                              (np.exp(valid_mean_loss / valid_size)))

if __name__ == "__main__":
    # check karpathy configuration. https://github.com/karpathy/char-rnn/blob/master/train.lua#L38
    params = {
      'batch_size': 256,
      'epochs': 2,
      'num_unrollings': 10,
      # LSTM hidden layer
      'num_nodes': 128,
      # refer to this [article](http://arxiv.org/abs/1409.2329)
      'dropout': 0.5,
      # Generator needs some randomness to prevent from repeating same phase.
      'max_k': 5,
      'summary_frequency': 10,
      'savefile': 'lstm.ckpt',
      'resume': True,
    }
    lstm(params)

'''
Validation set perplexity: 644.17
Average loss at step 3000: 6.093622 perplexity: 443.02 learning rate: 10.00
================================================================================
rtp in a small sequence or to two four five minutes and to have no a single variant to move a new variant the and UNK to have become to an individual and other members or other members that is an important of the status of UNK and or any other species the first major variant the status of a number that the number or are in one two to a type is considered another to the actual issue of
democrat the current member of their current president for a house on house to international rights which the house is an important member is the national committee member federal national international organizations organizations or international international law to follow them from UNK to participate the committee to be the first international organisation in europe one two eight seven international international union in japan modern modern organisation international international law the states community on the national organisation rights has the union
hanja a variant or the first of an UNK from the first century also called an early one four seven five zero two nine six three the german model as a member in his career of a variety the modern modern concept that the of UNK UNK a single class which had a concept for instance of their own ideas and a number the version of these features of the development of evolutionary the concept evolution of social social cultural
ox s a series that UNK was named the and its father s a woman while to his father and mother in one seven four four with the latter as his name in UNK was never developed the beginning of the early two two year after and later and his wife UNK later and her her wife with the death and his father s own father to be reported on this article in europe the last world the most popular
singin and and other UNK in its terms with all of their the most of these cases the first variant of these groups in one eight two one which had become a UNK for example in modern literature as UNK the german french name german society for example is a UNK in particular culture for these are not UNK and a particular ethnic society as also a particular aspect that a concept has a concept not to change to a
'''
