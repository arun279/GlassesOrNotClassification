# Nitin Nataraj (50246850) and Arun Krishnamurthy (50247445)
from scipy import ndimage,misc
import scipy
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import keras
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tensorflow.python.client import device_lib

import numpy as np
import random as ran
import tensorflow as tf
import pandas as pd

import os
import numpy as np
import pandas as pd

imagesize = 28
epochs = 1000
batch_size = 250
numSamples = 20000

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

#Read the labels and check how many of each type exist
labelPath = '../Anno/list_attr_celeba.txt'
labels = []
with open(labelPath,'r') as f:
    count = 0
    for line in f.readlines():
        if count == 0:
            numLabels = int(line.split('\n')[0])
        if count == 1:
            columns = line.split()
            labelIndex = columns.index('Eyeglasses')
        if count >= 2:
            labels.append(int(line.split()[labelIndex]))
        count += 1

labels = np.array(labels)       

unique, counts = np.unique(labels, return_counts=True)
labelCounts = dict(zip(unique, counts))
numNoGlasses = labelCounts[-1]
numGlasses = labelCounts[1]
print("No. of images with eyeglasses: %d " %(numGlasses) )
print("No. of images with no eyeglasses: %d " %(numNoGlasses) )
#The dataset is highly skewed. We need to downsample the "No Glasses" category,
#and take in the glasses category as it is

noGlassesInds = np.argwhere(labels == -1).flatten()
glassesInds = np.argwhere(labels == 1).flatten()
noGlassesIndsDownsampled = np.reshape(np.random.choice(noGlassesInds,numSamples),[-1,1])

noGlassesLabels = np.reshape(labels[noGlassesIndsDownsampled],[-1,1])
glassesLabels = np.reshape(labels[glassesInds],[-1,1])

numNoGlasses = len(noGlassesIndsDownsampled)
numGlasses = len(glassesInds)

print("Shape of downsampled labels is: ", str(noGlassesIndsDownsampled.shape))
print(noGlassesLabels.shape, glassesLabels.shape)
print("Labels loaded")

path = "../img_align_celeba/img_align_celeba"
#Now the images with these indices need to be acquired
dirContents = list(os.walk(path))
fnames = np.array([fname for fname in dirContents[0][2] if '.jpg' in fname]).flatten()
fnamesNoGlasses = fnames[noGlassesIndsDownsampled]
fnamesGlasses = fnames[glassesInds]
nImages = len(fnames)
noGlassesFnames = list(fnames[noGlassesIndsDownsampled].flatten())
glassesFnames = list(fnames[glassesInds].flatten())

noGlassesImages = np.zeros([numNoGlasses, imagesize,imagesize])
glassesImages = np.zeros([numGlasses, imagesize, imagesize])


for i, fname in enumerate(noGlassesFnames):
    fpath = os.path.join(path,fname)
    #Read Image
    im = plt.imread(fpath)
    
    #Append to array
    im_resize = scipy.misc.imresize(im,[imagesize,imagesize])
    im_resize = rgb2gray(im_resize)
    noGlassesImages[i,:,:] = im_resize

noGlassesImages = np.array(noGlassesImages)

for i, fname in enumerate(glassesFnames):
    fpath = os.path.join(path,fname)
    #Read Image
    im = plt.imread(fpath)
    #Append to array
    im_resize = scipy.misc.imresize(im,[imagesize,imagesize])
    im_resize = rgb2gray(im_resize)
    glassesImages[i,:,:] = im_resize
glassesImages = np.array(glassesImages)

print(noGlassesImages.shape,glassesImages.shape)
print("Images loaded")


noGlassesImages = np.array(noGlassesImages)
glassesImages = np.array(glassesImages)
all_x = np.concatenate((noGlassesImages, glassesImages), axis = 0).astype('float32') / 255.0
all_y = np.concatenate((noGlassesLabels,glassesLabels),axis = 0)

X_train, X_test, y_train, y_test = train_test_split(all_x, all_y, test_size=0.33, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.33, random_state=42)
X_train = np.reshape(X_train,[-1,imagesize*imagesize])
X_test = np.reshape(X_test,[-1,imagesize*imagesize])
X_valid = np.reshape(X_valid,[-1,imagesize*imagesize])
y_train = y_train.flatten()
y_test = y_test.flatten()
y_valid = y_valid.flatten()
print("Data split: " + str(X_train.shape) + ' ' + str(X_test.shape))

y_train_onehot = []

for i,label in enumerate(y_train):
    if label == -1:
        y_train_onehot.append([1,0])
    elif label == 1:
        y_train_onehot.append([0,1])
y_train_onehot = np.array(y_train_onehot)        
y_test_onehot = []

for i,label in enumerate(y_test):
    if label == -1:
        y_test_onehot.append([1,0])
    elif label == 1:
        y_test_onehot.append([0,1])
y_test_onehot = np.array(y_test_onehot)

y_valid_onehot = []

for i,label in enumerate(y_valid):
    if label == -1:
        y_valid_onehot.append([1,0])
    elif label == 1:
        y_valid_onehot.append([0,1])
y_valid_onehot = np.array(y_valid_onehot)

x = tf.placeholder(tf.float32, [None, imagesize*imagesize])
W = tf.Variable(tf.zeros([imagesize*imagesize, 2]))
b = tf.Variable(tf.zeros([2]))


y_ = tf.placeholder(tf.float32, [None, 2])

# Weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

# Convolution and Pooling
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

###### First Convolutional Layer
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x,[-1,imagesize,imagesize,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#Size now is batch_size x 28 x 28 x 128
h_pool1 = max_pool_2x2(h_conv1)
#size now is now batch_size x 14 x 14 x 128

# Second Convolutional Layer
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densley Connected Layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer
W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Train and Evaluate Model
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def return_batch(X, y, lower, upper):
    return X[lower:upper], y[lower:upper]

losses = []
train_accs = []
valid_accs = []
test_accs = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    lower = 0
    upper = batch_size
    
    for i in range(epochs):
        epoch_loss = 0
        
   
        batch_x,batch_y = return_batch(X_train,y_train_onehot,lower,upper)

        lower += batch_size
        upper += batch_size
        if upper >= len(X_train):
            lower = 0
            upper = batch_size
        if i % 100 == 0:

            train_accuracy = accuracy.eval(feed_dict={x: X_train, y_: y_train_onehot, keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
            valid_accuracy_m = accuracy.eval({x:X_valid, y_: y_valid_onehot, keep_prob: 1.0})
            print('step %d, validation accuracy %g' % (i, valid_accuracy_m))
        _,c = sess.run([train_step, cross_entropy], feed_dict={x: batch_x, y_: batch_y, keep_prob:0.5})
        losses.append(c)
        train_accuracy_m = accuracy.eval({x:X_train, y_: y_train_onehot, keep_prob: 1.0})
        test_accuracy_m = accuracy.eval({x:X_test, y_: y_test_onehot, keep_prob: 1.0})
        valid_accuracy_m = accuracy.eval({x:X_valid, y_: y_valid_onehot, keep_prob: 1.0})
        train_accs.append(train_accuracy_m)
        test_accs.append(test_accuracy_m)
        valid_accs.append(valid_accuracy_m)
        
print("Accuracies obtained after %d epochs is: Train: %.2f,Test: %.2f,Validation: %.2f"
    %(epochs, train_accs[-1],test_accs[-1],valid_accs[-1]))