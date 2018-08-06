from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time

import tensorflow as tf
import numpy as np

def with_self_graph(function):
    def wrapper(self, *args, **kwargs):
        with self.graph.as_default():
            return function(self, *args, **kwargs)
    return wrapper

class CNNRNN(object):
    def __init__(self, graph=None, *args, **kwargs):
        """
        This is a CNN-RNN model for document classification. The model is divided into
        two main components. One is CNN which models sentence representation. The other
        is gated bidirectional RNN which models document representation from sentence
        representation.
      
        Args:
            - max_sent_length: maximum length of sentences in a document
            - batch_size: number of batches into the model. Usually this is
                          inferred from input shape, but since we need to
                          input CNN output into RNN model, this is specified
                          at the class level.
            - num_classes: number of classes for the response variable
            - sequence_length: maximum length of each sentence
            - vocab_size: vocaulbary size
            - embedding_size: size of the word embedding for the word vector
            - filter_sizes: (list) of filter widths for CNN
            - conv_output_len: output length of linear layer of CNN
            - rnn_hidden_units: number of units for the gated RNN

        The model has the below structure:
        
        1. Word vectors using word2vec
        2. Different filters for CNN with different widths.
           For each filter, there are:
            - Linear layer
            - Average pooling
            - tanh transformation
        3. Average of outputs from three filters to find the sentence representation.
        4. Input the CNN sentence representations to the bi-directional gated RNN.
        5. Outputs from forward RNN and backward RNN are concatenated for each sentence,
           and the concatenated vectors are averaged to give the document representation.
        6. Input the document representation to softmax layer.
        """
        
        # Set TensorFlow graph. All TF code will work on this graph.
        self.graph = graph or tf.Graph()
        self.SetParams(*args, **kwargs)

    @with_self_graph
    def SetParams(self, max_sent_length, batch_size, num_classes, sequence_length, vocab_size, embedding_size, \
                  filter_sizes, conv_output_len, num_rnn_units):

        self.max_sent_length = max_sent_length
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.conv_output_len = conv_output_len
        self.num_rnn_units = num_rnn_units

        # Training hyperparameters; these can be changed with feed_dict
        with tf.name_scope("Training_Parameters"):
            self.learning_rate_ = tf.placeholder(tf.float32, [], name="learning_rate")
            self.no_op_ = tf.no_op()

            
    @with_self_graph
    def BuildCoreGraph(self):

        self.input_x_ = tf.placeholder(tf.int32, [None, self.max_sent_length, self.sequence_length], name="input_x")
        self.input_y_ = tf.placeholder(tf.int32, [None, self.num_classes], name="input_y")

        with tf.name_scope("batch_size"):
            self.batch_size_ = tf.shape(self.input_x_)[0]
        with tf.name_scope("max_time"):
            self.max_time_ = tf.shape(self.input_x_)[2]

        # Embedding layer
        with tf.name_scope("embedding"):

            self.embedding_ = tf.placeholder(tf.float32, shape=(self.vocab_size, self.embedding_size))
            self.W_in_ = tf.Variable(self.embedding_, trainable=False, name="W")
            
        # Output logits, which can be used by loss functions or for prediction.
        self.logits_ = None

        self.loss_ = None

        self.rnn_inputs_ = []
        
        # For each batch, we run CNN so that the output from
        # CNN can be used as RNN input.
        for i in range(self.batch_size):
            
            with tf.name_scope("batch-%s" % i):
                
                self.input_x_conv_ = tf.gather_nd(self.input_x_, [[i]])
                self.input_x_conv_ = tf.reshape(self.input_x_conv_, [self.max_sent_length, self.sequence_length])

                self.embedded_chars_ = tf.nn.embedding_lookup(self.W_in_, self.input_x_conv_)
                self.embedded_chars_expanded_ = tf.expand_dims(self.embedded_chars_, -1)
                
                # Loop for each filter size
                self.conv_outputs_ = []
                for i, filter_size in enumerate(self.filter_sizes):

                    with tf.name_scope("conv-%s" % filter_size):

                        # Convolution Layer
                        filter_shape = [filter_size, self.embedding_size, 1, self.conv_output_len]
                        W = tf.Variable(tf.random_uniform(filter_shape, -1.0, 1.0), name="W")
                        b = tf.zeros([self.conv_output_len], dtype=tf.float32, name="b")
                        conv = tf.nn.conv2d(
                            self.embedded_chars_expanded_,
                            W,
                            strides=[1, 1, 1, 1],
                            padding="VALID",
                            name="conv")

                        # Average pooling over the outputs
                        pooled = tf.nn.avg_pool(
                            tf.nn.bias_add(conv, b),
                            ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                            strides=[1, 1, 1, 1],
                            padding='VALID',
                            name="pool")

                        # Apply nonlinearity
                        h = tf.nn.relu(pooled, name="relu")
                        self.conv_outputs_.append(h)

                # Combine all the pooled features
                self.output_total_ = self.conv_output_len * len(self.filter_sizes)
                self.cnn_o_ = tf.concat(self.conv_outputs_, 3)
                self.cnn_o_ = tf.reshape(self.cnn_o_, [-1, self.output_total_])
                self.rnn_inputs_.append(self.cnn_o_)
        
        
        # Bidirectional gated RNN
        with tf.name_scope("rnn"):
            self.input_x_rnn_ = tf.convert_to_tensor(self.rnn_inputs_) 
            self.fw_cell_ = tf.nn.rnn_cell.GRUCell(self.num_rnn_units)
            self.bw_cell_ = tf.nn.rnn_cell.GRUCell(self.num_rnn_units)
            
            self.rnn_o_, _ = tf.nn.bidirectional_dynamic_rnn(self.fw_cell_, self.bw_cell_, self.input_x_rnn_, dtype=tf.float32)
            self.output_fw_ = self.rnn_o_[0]
            self.output_bw_ = self.rnn_o_[1]

            # Concatenate results from forward RNN and backward RNN
            self.rnn_o_ = tf.concat([self.output_fw_, self.output_bw_], 2)
            
            # Calculate average of the concatenated vectors
            self.rnn_o_ = tf.reduce_mean(self.rnn_o_, 1) 

        # Softmax output layer
        with tf.name_scope("Output_Layer"):
            self.W_out_ = tf.Variable(tf.random_normal([2*self.num_rnn_units, self.num_classes]), name="W_out")
            self.b_out_ = tf.Variable(tf.zeros([self.num_classes,], dtype=tf.float32), name="b_out")
            self.logits_ = tf.add(tf.matmul(self.rnn_o_, self.W_out_), self.b_out_, name="logits")
            self.predictions_ = tf.argmax(self.logits_, 1, name="predictions")

        # Loss computation (true loss, for prediction)
        with tf.name_scope("Cost_Function"):
            # Full softmax loss, for scoring
            self.per_example_loss_ = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y_, logits=self.logits_, \
                                                                             name="per_example_loss")
            self.loss_ = tf.reduce_mean(self.per_example_loss_, name="loss")

        # Accuracy calculation
        with tf.name_scope("Accuracy"):
            correct_predictions = tf.equal(self.predictions_, tf.argmax(self.input_y_, 1))
            self.accuracy_ = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            
            
    @with_self_graph
    def BuildTrainGraph(self):
        self.train_step_ = None

        # Define optimizer and training op
        with tf.name_scope("Training"):
            self.no_op_ = tf.train.AdamOptimizer(self.learning_rate_)
            self.train_step_ = self.no_op_.minimize(self.loss_)
