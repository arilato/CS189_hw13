import tensorflow as tf
import datetime
import os
import sys
import argparse

slim = tf.contrib.slim


class Solver(object):

    def __init__(self, net, data):

     
        self.net = net
        self.data = data
       
        self.max_iter = 5000
        self.summary_iter = 200
        


      
        self.learning_rate = 0.1
       
        self.saver = tf.train.Saver()
      
        self.summary_op = tf.summary.merge_all()

        self.global_step = tf.get_variable(
            'global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        '''
        Tensorflow is told to use a gradient descent optimizer 
        In the function optimize you will iteratively apply this on batches of data
        '''
        self.train_step = tf.train.MomentumOptimizer(.003, .9)
        self.train = self.train_step.minimize(self.net.class_loss)
        

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    


    def optimize(self):

        self.train_losses = []
        self.test_losses = []

        '''
        Performs the training of the network. 
        Implement SGD using the data manager to compute the batches
        Make sure to record the training and test loss through out the process
        '''
        for i in range(1000):
            image_train, label_train = self.data.get_train_batch()
            image_test, label_test = self.data.get_validation_batch()
            feed_dict_train = {self.net.images: images_train, self.net.labels: labels_train}
            feed_dict_test = {self.net.images: images_test, self.net.labels: labels_test}
            self.sess.run([self.train],feed_dict=feed_dict)
            self.train_losses.append(self.run(self.net.accuracy, feed_dict=feed_dict_train))
            self.test_losses.append(self.run(self.net.accuracy, feed_dict=feed_dict_test))


   
