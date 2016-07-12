# Deep Learning Udacity
* Self-study of [Udacity Deep Learning](https://www.udacity.com/course/deep-learning--ud730)
 
# Folder structure
* `udacity_notebook`: Udacity homework, which is copied from [https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/udacity](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/udacity)
* `tensorflow_examples`: Extract essential from the homework.

# Tensorflow Examples
## `MLP_notMNIST`
* With the configuration
```
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
```

* The Cross entropy is saturated at ~50 epoch

![Alt text](tensorflow_examples/images/MLP_notMNIST.png "https://docs.google.com/spreadsheets/d/1IQtKzPI4cpE_JwI0uLcyAojrMR-0cf_uFlxhOzYUs7M/edit screenshot")

* Results
```
# epoch complete: step 133199 num_training 39960000 time:5:00:04.257040
# epoch complete: Train [CrossEntropy / Training Accuracy] 0.2387 / 0.9590
# Validation [CrossEntropy / Training Accuracy] 0.3955 / 0.9148
# BREAK: TIME OVER
# Complete training. Total time:5:00:13.045505
# Testing [CrossEntropy / Training Accuracy] 0.2168 / 0.9653
```
