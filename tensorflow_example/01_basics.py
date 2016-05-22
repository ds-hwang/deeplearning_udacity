
# coding: utf-8

# In[ ]:

"""Summary of tensorflow basics.

Parag K. Mital, Jan 2016."""


# In[13]:

# %% Import tensorflow and pyplot
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# In[ ]:

# %% tf.Graph represents a collection of tf.Operations
# You can create operations by writing out equations.
# By default, there is a graph: tf.get_default_graph()
# and any new operations are added to this graph.
# The result of a tf.Operation is a tf.Tensor, which holds
# the values.


# In[14]:

# %% First a tf.Tensor
n_values = 32
x = tf.linspace(-3.0, 3.0, n_values)


# In[17]:

# %% Construct a tf.Session to execute the graph.
sess = tf.Session()
result = sess.run(x)
print(result)


# In[20]:

# %% Alternatively pass a session to the eval fn:
x.eval(session=sess)
# x.eval() does not work, as it requires a session!
# x.eval()


# In[30]:

# %% We can setup an interactive session if we don't
# want to keep passing the session around:
sess.close()
sess = tf.InteractiveSession()


# In[31]:

# %% Now this will work!
x.eval()


# In[32]:

# %% Now a tf.Operation
# We'll use our values from [-3, 3] to create a Gaussian Distribution
sigma = 1.0
mean = 0.0
z = (tf.exp(tf.neg(tf.pow(x - mean, 2.0) /
                   (2.0 * tf.pow(sigma, 2.0)))) *
     (1.0 / (sigma * tf.sqrt(2.0 * 3.1415))))


# In[33]:

# %% By default, new operations are added to the default Graph
assert z.graph is tf.get_default_graph()
print z.graph

# In[ ]:

#plt.close('all')
# %% Execute the graph and plot the result
plt.plot(z.eval())
plt.show()


# In[ ]:

# %% We can find out the shape of a tensor like so:
print(z.get_shape())


# In[35]:

# %% Or in a more friendly format
print(z.get_shape().as_list())


# In[36]:

# %% Sometimes we may not know the shape of a tensor
# until it is computed in the graph.  In that case
# we should use the tf.shape fn, which will return a
# Tensor which can be eval'ed, rather than a discrete
# value of tf.Dimension
print(tf.shape(z).eval())


# In[ ]:

# %% We can combine tensors like so:
print(tf.pack([tf.shape(z), tf.shape(z), [3], [4]]).eval())


# In[ ]:

# %% Let's multiply the two to get a 2d gaussian
z_2d = tf.matmul(tf.reshape(z, [n_values, 1]), tf.reshape(z, [1, n_values]))


# In[ ]:

# %% Execute the graph and store the value that `out` represents in `result`.
plt.imshow(z_2d.eval())


# In[ ]:

# %% For fun let's create a gabor patch:
x = tf.reshape(tf.sin(tf.linspace(-3.0, 3.0, n_values)), [n_values, 1])
y = tf.reshape(tf.ones_like(x), [1, n_values])
z = tf.mul(tf.matmul(x, y), z_2d)
plt.imshow(z.eval())


# In[ ]:

# %% We can also list all the operations of a graph:
ops = tf.get_default_graph().get_operations()
print([op.name for op in ops])


# In[ ]:

# %% Lets try creating a generic function for computing the same thing:
def gabor(n_values=32, sigma=1.0, mean=0.0):
    x = tf.linspace(-3.0, 3.0, n_values)
    z = (tf.exp(tf.neg(tf.pow(x - mean, 2.0) /
                       (2.0 * tf.pow(sigma, 2.0)))) *
         (1.0 / (sigma * tf.sqrt(2.0 * 3.1415))))
    gauss_kernel = tf.matmul(
        tf.reshape(z, [n_values, 1]), tf.reshape(z, [1, n_values]))
    x = tf.reshape(tf.sin(tf.linspace(-3.0, 3.0, n_values)), [n_values, 1])
    y = tf.reshape(tf.ones_like(x), [1, n_values])
    gabor_kernel = tf.mul(tf.matmul(x, y), gauss_kernel)
    return gabor_kernel


# In[ ]:

# %% Confirm this does something:
plt.imshow(gabor().eval())


# In[ ]:

# %% And another function which can convolve
def convolve(img, W):
    # The W matrix is only 2D
    # But conv2d will need a tensor which is 4d:
    # height x width x n_input x n_output
    if len(W.get_shape()) == 2:
        dims = W.get_shape().as_list() + [1, 1]
        W = tf.reshape(W, dims)

    if len(img.get_shape()) == 2:
        # num x height x width x channels
        dims = [1] + img.get_shape().as_list() + [1]
        img = tf.reshape(img, dims)
    elif len(img.get_shape()) == 3:
        dims = [1] + img.get_shape().as_list()
        img = tf.reshape(img, dims)
        # if the image is 3 channels, then our convolution
        # kernel needs to be repeated for each input channel
        W = tf.concat(2, [W, W, W])

    # Stride is how many values to skip for the dimensions of
    # num, height, width, channels
    convolved = tf.nn.conv2d(img, W,
                             strides=[1, 1, 1, 1], padding='SAME')
    return convolved


# In[ ]:

# %% Load up an image:
from skimage import data
img = data.astronaut()
plt.imshow(img)
plt.show()
print(img.shape)


# In[ ]:

# %% Now create a placeholder for our graph which can store any input:
x = tf.placeholder(tf.float32, shape=img.shape)


# In[ ]:

# %% And a graph which can convolve our image with a gabor
out = convolve(x, gabor())


# In[ ]:

# %% Now send the image into the graph and compute the result
result = tf.squeeze(out).eval(feed_dict={x: img})
plt.imshow(result)
plt.show()
