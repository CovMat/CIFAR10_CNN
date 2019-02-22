from read_cifar10 import my_read
import numpy as np
import tensorflow as tf
from tensorflow.layers import conv2d, max_pooling2d, dropout, flatten, dense

# Define some parameters and hyperparameters
nb_classes = 10
my_learning_rate = 0.003
logdir = "./logs/test3"
#
nb_conv_layers = 3
nb_filters = [64, 128, 256]
size_filters = [3, 3, 3]
size_pooling = [2, 2, 2]
stride_pooling=[2, 2, 2]
dropout_rate = [0, 0, 0]
# FC layer 4
nb_units4 = 1024
dropout_rate4 = 0
# output layer 5
nb_units5 = nb_classes

# read data
training_image, training_label, testing_image, testing_label = my_read("data")
# data augmentation
# random flip left right
X0 = tf.image.random_flip_left_right( training_image )
# standardization:
X1 = tf.map_fn( lambda image: tf.image.per_image_standardization( image ), X0, dtype = tf.float32, parallel_iterations = 20 )
sess0 = tf.Session()
X2 = sess0.run( X1 )
training_image = np.asarray(X2)

# Build the CNN
X = tf.placeholder( tf.float32, [None, 32, 32, 3] )   
Y = tf.placeholder( tf.int32, [None] )
is_training = tf.placeholder( tf.bool )
#
c1 = conv2d( X, nb_filters[0], size_filters[0], 1, 'same', activation=tf.nn.relu)
p1 = max_pooling2d( c1, size_pooling[0], stride_pooling[0], 'same' )
d1 = dropout( p1, dropout_rate[0], training = is_training )
#
c2 = conv2d( d1, nb_filters[1], size_filters[1], 1, 'same', activation=tf.nn.relu)
p2 = max_pooling2d( c2, size_pooling[1], stride_pooling[1], 'same' )
d2 = dropout( p2, dropout_rate[2], training = is_training )
#
c3 = conv2d( d2, nb_filters[2], size_filters[2], 1, 'same', activation=tf.nn.relu)
p3 = max_pooling2d( c3, size_pooling[2], stride_pooling[2], 'same' )
d3 = dropout( p3, dropout_rate[2], training = is_training )
#
f4f = flatten( d3 )
f4 = dense( f4f, nb_units4, tf.nn.relu )
df4 = dropout( f4, dropout_rate4, training = is_training )
X_out = dense( df4, nb_units5 )
cost = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits( labels=Y, logits=X_out ) )
optimizer = tf.train.AdamOptimizer( learning_rate = my_learning_rate ).minimize( cost )
# for TensorBoard
cost_summ = tf.summary.scalar( "Cost Function", cost )

# run (mini-batch)
summary = tf.summary.merge_all()
sess = tf.Session()
sess.run( tf.global_variables_initializer() )
writer = tf.summary.FileWriter( logdir )
writer.add_graph( sess.graph )
#
batch_size = 512
batch_num = 50000 // batch_size
#
global_step = 0
for epoch in range(6):
    for i in range(batch_num):
        st = i*batch_size
        ed = (i+1)*batch_size
        if ( i == batch_num-1 ):
            ed = 50000
        s, _ = sess.run( [ summary, optimizer], feed_dict={\
                                        X:training_image[ st:ed, :, :, :],\
                                        Y:training_label[ st:ed ],\
                                        is_training:True} )
        writer.add_summary( s, global_step = global_step )
        global_step += 1

# Accuracy
logit_train = sess.run( X_out, feed_dict={X:training_image, is_training:False} )
max_index = sess.run( tf.argmax(logit_train, axis=1) )
accuracy = sess.run( tf.reduce_mean(tf.cast( max_index == training_label, dtype=tf.float32) ) )
print( "Accuracy of Training set:"+ str(accuracy) )
logit_test = sess.run( X_out, feed_dict={X:testing_image, is_training:False} )
max_index = sess.run( tf.argmax(logit_test, axis=1) )
accuracy = sess.run( tf.reduce_mean(tf.cast( max_index == testing_label, dtype=tf.float32) ) )
print( "Accuracy of Testing set:"+ str(accuracy) )
