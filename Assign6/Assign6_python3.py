__author__ = 'deepika'

import numpy as np
import pickle
import tensorflow as tf
from sklearn.metrics import confusion_matrix

cifar10_folder = './cifar10/'
train_datasets = ['data_batch_1']
                  #'data_batch_2', 'data_batch_3', 'data_batch_4' ]
test_dataset = ['test_batch']
validation_dataset = ['data_batch_5']
c10_image_height = 32
c10_image_width = 32
c10_image_depth = 1
c10_num_labels = 10
c10_image_size = 32


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

def one_hot_encode(np_array):
    return (np.arange(10) == np_array[:,None]).astype(np.float32)

def reformat_data(dataset, labels, image_width, image_height, image_depth):
    grayscale = 0.21*dataset[:,0:1024] + 0.72*dataset[:,1024:2048] + 0.07*dataset[:,2048:3072]

    np_dataset_ = np.array([np.array(image_data).reshape(image_width, image_height, image_depth) for image_data in grayscale])
    np_labels_ = one_hot_encode(np.array(labels, dtype=np.float32))
    np_dataset, np_labels = randomize(np_dataset_, np_labels_)
    return np_dataset, np_labels

def flatten_tf_array(array):
    shape = array.get_shape().as_list()
    return tf.reshape(array, [shape[0], shape[1] * shape[2] * shape[3]])

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


with open(cifar10_folder + validation_dataset[0], 'rb') as f0:
    c10_validation_dict = pickle.load(f0, encoding='bytes')

c10_validation_dataset, c10_validation_labels = c10_validation_dict[b'data'], c10_validation_dict[b'labels']
validation_dataset_cifar10, validation_labels_cifar10 = reformat_data(c10_validation_dataset, c10_validation_labels, c10_image_size, c10_image_size, c10_image_depth)

with open(cifar10_folder + test_dataset[0], 'rb') as f0:
    c10_test_dict = pickle.load(f0, encoding='bytes')

c10_test_dataset, c10_test_labels = c10_test_dict[b'data'], c10_test_dict[b'labels']
test_dataset_cifar10, test_labels_cifar10 = reformat_data(c10_test_dataset, c10_test_labels, c10_image_size, c10_image_size, c10_image_depth)

c10_train_dataset, c10_train_labels = [], []
for train_dataset in train_datasets:
    with open(cifar10_folder + train_dataset, 'rb') as f0:
        c10_train_dict = pickle.load(f0, encoding='bytes')
        c10_train_dataset_, c10_train_labels_ = c10_train_dict[b'data'], c10_train_dict[b'labels']

        c10_train_dataset.append(c10_train_dataset_)
        c10_train_labels += c10_train_labels_

c10_train_dataset = np.concatenate(c10_train_dataset, axis=0)
train_dataset_cifar10, train_labels_cifar10 = reformat_data(c10_train_dataset, c10_train_labels, c10_image_size, c10_image_size, c10_image_depth)
del c10_train_dataset
del c10_train_labels

print("The training set contains the following labels: {}".format(np.unique(c10_train_dict[b'labels'])))
print('Training set shape', train_dataset_cifar10.shape, train_labels_cifar10.shape)
print('Test set shape', test_dataset_cifar10.shape, test_labels_cifar10.shape)
print('Validation shape', validation_dataset_cifar10.shape, validation_labels_cifar10.shape)

########################
# Neural Net
########################

LENET5_BATCH_SIZE = 32
LENET5_PATCH_SIZE = 5
LENET5_PATCH_DEPTH_1 = 6
LENET5_PATCH_DEPTH_2 = 16
LENET5_NUM_HIDDEN_1 = 120
LENET5_NUM_HIDDEN_2 = 84

def variables_lenet5(patch_size = LENET5_PATCH_SIZE, patch_depth1 = LENET5_PATCH_DEPTH_1,
                     patch_depth2 = LENET5_PATCH_DEPTH_2,
                     num_hidden1 = LENET5_NUM_HIDDEN_1, num_hidden2 = LENET5_NUM_HIDDEN_2,
                     image_depth = 1, num_labels = 10):

    w1 = tf.Variable(tf.truncated_normal([patch_size, patch_size, image_depth, patch_depth1], stddev=0.1))
    b1 = tf.Variable(tf.zeros([patch_depth1]))

    w2 = tf.Variable(tf.truncated_normal([patch_size, patch_size, patch_depth1, patch_depth2], stddev=0.1))
    b2 = tf.Variable(tf.constant(1.0, shape=[patch_depth2]))

    w3 = tf.Variable(tf.truncated_normal([5*5*patch_depth2, num_hidden1], stddev=0.1))
    b3 = tf.Variable(tf.constant(1.0, shape = [num_hidden1]))

    w4 = tf.Variable(tf.truncated_normal([num_hidden1, num_hidden2], stddev=0.1))
    b4 = tf.Variable(tf.constant(1.0, shape = [num_hidden2]))

    w5 = tf.Variable(tf.truncated_normal([num_hidden2, num_labels], stddev=0.1))
    b5 = tf.Variable(tf.constant(1.0, shape = [num_labels]))
    variables = {
        'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5,
        'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4, 'b5': b5
    }
    return variables

def model_lenet5(data, variables):
    layer1_conv = tf.nn.conv2d(data, variables['w1'], [1, 1, 1, 1], padding='VALID') + variables['b1']
    layer1_conv = tf.nn.relu(layer1_conv)

    layer1_pool = tf.nn.max_pool(layer1_conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')

    layer2_conv = tf.nn.conv2d(layer1_pool, variables['w2'], [1, 1, 1, 1], padding='VALID') + variables['b2']
    layer2_conv = tf.nn.relu(layer2_conv)

    layer2_pool = tf.nn.max_pool(layer2_conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')

    flat_layer = flatten_tf_array(layer2_pool)

    layer3_fccd = tf.matmul(flat_layer, variables['w3']) + variables['b3']
    layer3_actv = tf.nn.relu(layer3_fccd)

    layer4_fccd = tf.matmul(layer3_actv, variables['w4']) + variables['b4']
    layer4_actv = tf.nn.relu(layer4_fccd)

    logits = tf.matmul(layer4_actv, variables['w5']) + variables['b5']
    return logits

# Run the model
# parameters determining the model size
image_size = 32
num_labels = 10

#the datasets
train_dataset = train_dataset_cifar10
train_labels = train_labels_cifar10
test_dataset = test_dataset_cifar10
test_labels = test_labels_cifar10
validation_dataset = validation_dataset_cifar10
validation_labels = validation_labels_cifar10

#number of iterations and learning rate
num_steps = 100
display_step = 10
learning_rate = 0.001
batch_size = 64

graph = tf.Graph()
with graph.as_default():
    #1) First we put the input data in a tensorflow friendly form.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, c10_image_width, c10_image_height, c10_image_depth))
    tf_train_labels = tf.placeholder(tf.float32, shape = (batch_size, num_labels))
    tf_test_dataset = tf.constant(test_dataset, tf.float32)
    tf_validation_dataset = tf.constant(validation_dataset, tf.float32)

    #2) Then, the weight matrices and bias vectors are initialized
    variables = variables_lenet5(image_depth = c10_image_depth, num_labels = num_labels)

    #3. The model used to calculate the logits (predicted labels)
    model = model_lenet5
    logits = model(tf_train_dataset, variables)

    #4. then we compute the softmax cross entropy between the logits and the (actual) labels
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))

    #5. The optimizer is used to calculate the gradients of the loss function
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    validation_prediction = tf.nn.softmax(model(tf_validation_dataset, variables))
    test_prediction = tf.nn.softmax(model(tf_test_dataset, variables))

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized with learning_rate' + str(learning_rate))
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]

        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        train_accuracy = accuracy(predictions, batch_labels)

        if step % display_step == 0:
            validation_accuracy = accuracy(validation_prediction.eval(), validation_labels)
            message = "step {:04d} : loss is {:06.2f}, accuracy on training set {:02.2f} %, accuracy on vaidation set {:02.2f} %".format(step, l, train_accuracy, validation_accuracy)
            print(message)

    print("Accuracy on test model:" + str(accuracy(test_prediction.eval(), test_labels)))
    result_test_pred = test_prediction.eval()

print("Confusion Matrix")
f = open(cifar10_folder + 'batches.meta', 'rb')
datadict = pickle.load(f, encoding='bytes')
f.close()
test_l = datadict['label_names']

predictions = [np.argmax(pred) for pred in result_test_pred]
true_labels = [np.argmax(lbl) for lbl in test_labels]

cm = confusion_matrix(true_labels, predictions)
for i in range(c10_num_labels):
    class_name = "({}) {}".format(i, test_l[i])
    print(str(cm[i, :]) + class_name)
class_numbers = [" ({0})".format(i) for i in range(len(test_l))]
print("".join(class_numbers))