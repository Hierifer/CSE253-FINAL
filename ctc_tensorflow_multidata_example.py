#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import time
import tensorflow as tf
import numpy as np

from six.moves import xrange as range
import matplotlib.pyplot as plt

#import scipy.io.wavfile as wav
#try:
#    from python_speech_features import mfcc
#except ImportError:
#    print("Failed to import python_speech_features.\n Try pip install python_speech_features.")
#    raise ImportError
    
    
from utils import sparse_tuple_from as sparse_tuple_from
from utils import pad_sequences as pad_sequences
pool_time=5
poolings=(1<<pool_time)
############################################################################3
from genIDCard  import *

OUTPUT_SHAPE = (45,330)
obj = gen_id_card()

alphabet_diction={"0":0,"1":1,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,"+":10,"-":11,"*":12,"/":13}
alphabet='0123456789+-*/'


#转化一个序列列表为稀疏矩阵    
def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)
    #print(values)
    indices = np.asarray(indices, dtype=np.int64)
    
    values = [alphabet_diction[index] for index in values]
    #values = np.asarray(values, dtype=dtype)
    
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
    
    return indices, values, shape
# 生成一个训练batch
def get_next_batch(batch_size=128):
    obj = gen_id_card()
    #(batch_size,256,32)
    inputs = np.zeros([batch_size, OUTPUT_SHAPE[1],OUTPUT_SHAPE[0]])
    codes = []

    for i in range(batch_size):
        #生成不定长度的字串
        image, text, vec = obj.gen_image(True)
        #np.transpose 矩阵转置 (32*256,) => (32,256) => (256,32)
        inputs[i,:] = np.transpose(image.reshape((OUTPUT_SHAPE[0],OUTPUT_SHAPE[1])))
        codes.append(list(text))
    targets = [np.asarray(i) for i in codes]
    #print (targets)
    sparse_targets = sparse_tuple_from(targets)
    #(batch_size,) 值都是256
    seq_len = np.ones(inputs.shape[0]) * OUTPUT_SHAPE[1]
    
    return inputs, sparse_targets, seq_len

inputs, sparse_targets,seq_len = get_next_batch(10)

###############################################################################

#def fake_data(num_examples, num_features, num_labels, min_size = 10, max_size=100):
#
#    # Generating different timesteps for each fake data
#    timesteps = np.random.randint(min_size, max_size, (num_examples,))
#
#    # Generating random input
#    inputs = np.asarray([np.random.randn(t, num_features).astype(np.float32) for t in timesteps])
#
#    # Generating random label, the size must be less or equal than timestep in order to achieve the end of the lattice in max timestep
#    labels = np.asarray([np.random.randint(0, num_labels, np.random.randint(1, inputs[i].shape[0], (1,))).astype(np.int64) for i, _ in enumerate(timesteps)])
#
#    return inputs, labels



###############################################################################

def decode_sparse_tensor(sparse_tensor):
    decoded_indexes = list()
    current_i = 0
    current_seq = []
    for offset, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(offset)
    decoded_indexes.append(current_seq)
    result = []
    for index in decoded_indexes:
        result.append(decode_a_seq(index, sparse_tensor))
    return result
    
def decode_a_seq(indexes, spars_tensor):
    decoded = []
    for m in indexes:
        str = alphabet[spars_tensor[1][m]]
        decoded.append(str)
    return decoded



###############################################################################
    
## Constants
#SPACE_TOKEN = '<space>'
#SPACE_INDEX = 0
#FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

#TrainData = Eqdata();
#ValidData = Eqdata(train=False,valid=True);
#TestData = Eqdata(train=False,valid=False);
    



# Some configs
num_features =OUTPUT_SHAPE[0] #13
# Accounting the 0th indice +  space + blank label = 28 characters
num_classes = obj.len + 1 + 1  #ord('z') - ord('a') + 1 + 1 + 1



# Hyper-parameters
num_epochs = 100
num_hidden = 1000
num_layers = 1

batch_size = 128

initial_learning_rate = 1e-3
momentum = 0.9



num_examples = batch_size*5
num_batches_per_epoch = int(num_examples/batch_size)
#
#inputs, labels = fake_data(num_examples, num_features, num_classes - 1)
#
## You can preprocess the input data here
#train_inputs = inputs
#
## You can preprocess the target data here
#train_targets = labels
REPORT_STEP = 5


###############################################################################
# THE MAIN CODE!

graph = tf.Graph()
with graph.as_default():
    # e.g: log filter bank or MFCC features
    # Has size [batch_size, max_stepsize, num_features], but the
    # batch_size and max_stepsize can vary along each step
    inputs = tf.placeholder(tf.float32, [None, None, num_features])

    # Here we use sparse_placeholder that will generate a
    # SparseTensor required by ctc_loss op.
    targets = tf.sparse_placeholder(tf.int32)

    # 1d array of size [batch_size]
    seq_len = tf.placeholder(tf.int32, [None])

    # Defining the cell
    # Can be:
    #   tf.nn.rnn_cell.RNNCell
    #   tf.nn.rnn_cell.GRUCell
    def cell():
        return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True),input_keep_prob=0.8)

    # Stacking rnn cells
    stack = tf.contrib.rnn.MultiRNNCell([cell() for i in range(num_layers)],
                                        state_is_tuple=True)
    outputs=tf.reshape(inputs,[batch_size,OUTPUT_SHAPE[1],num_features,1])
    #
    
    outputs=tf.layers.conv2d(outputs, 16, 3,strides=(1, 1), padding='SAME')
    outputs=tf.layers.conv2d(outputs, 16, 3,strides=(1, 1), padding='SAME')
    outputs=tf.layers.max_pooling2d(outputs,pool_size=(2, 2), strides=2)
    #
    outputs=tf.layers.conv2d(outputs, 32, 3,strides=(1, 1), padding='SAME')
    outputs=tf.layers.conv2d(outputs, 32, 3,strides=(1, 1), padding='SAME')
    outputs=tf.layers.max_pooling2d(outputs,pool_size=(2, 2), strides=2)
    #
    outputs=tf.layers.conv2d(outputs, 64, 3,strides=(1, 1), padding='SAME')
    outputs=tf.layers.conv2d(outputs, 64, 3,strides=(1, 1), padding='SAME')
    outputs=tf.layers.max_pooling2d(outputs,pool_size=(2, 2), strides=2)
    #
    outputs=tf.layers.conv2d(outputs, 128, 3,strides=(1, 1), padding='SAME')
    outputs=tf.layers.conv2d(outputs, 128, 3,strides=(1, 1), padding='SAME')
    outputs=tf.layers.max_pooling2d(outputs,pool_size=(2, 2), strides=2)
    #
    outputs=tf.layers.conv2d(outputs, 256, 3,strides=(1, 1), padding='SAME')
    outputs=tf.layers.conv2d(outputs, 256, 3,strides=(1, 1), padding='SAME')
    outputs=tf.layers.max_pooling2d(outputs,pool_size=(2, 2), strides=2)
    
    #seq_len//=2
    outputs=tf.reshape(outputs,[batch_size,OUTPUT_SHAPE[1]//poolings,-1])
    outputs, _ = tf.nn.dynamic_rnn(stack, outputs, seq_len, dtype=tf.float32)
    
   #shape = tf.shape(outputs)
    #outputs=tf.reshape(inputs,[batch_size,OUTPUT_SHAPE[1],OUTPUT_SHAPE[0],-1])
    #outputs=tf.layers.conv2d(outputs,16,3,strides=(1,1),padding='SAME')
    #outputs=tf.reshape(outputs,[batch_size,OUTPUT_SHAPE[1],-1])
    
    # The second output is the last state and we will no use that
   # outputs, _ = tf.nn.dynamic_rnn(stack, outputs, seq_len, dtype=tf.float32)

    shape = tf.shape(inputs)
    batch_s, max_timesteps = shape[0], shape[1]

    # Reshaping to apply the same weights over the timesteps
    outputs = tf.reshape(outputs, [-1, num_hidden])

    # Truncated normal with mean 0 and stdev=0.1
    # Tip: Try another initialization
    # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
    W = tf.Variable(tf.truncated_normal([num_hidden,
                                         num_classes],
                                        stddev=0.1))
    # Zero initialization
    # Tip: Is tf.zeros_initializer the same?
    b = tf.Variable(tf.constant(0., shape=[num_classes]))

    # Doing the affine projection
    logits = tf.matmul(outputs, W) + b

    # Reshaping back to the original shape
    logits = tf.reshape(logits, [batch_s, -1, num_classes])

    # Time major
    logits = tf.transpose(logits, (1, 0, 2))

    loss = tf.nn.ctc_loss(targets, logits, seq_len)
    cost = tf.reduce_mean(loss)

    optimizer = tf.train.MomentumOptimizer(initial_learning_rate,
                                           0.9).minimize(cost)

    # Option 2: tf.nn.ctc_beam_search_decoder
    # (it's slower but you'll get better results)
    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)

    # Inaccuracy: label error rate
    ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                          targets))    
    


###############################################################################
print();
print(" Training & Validation   start:");
print();

train_accuracy = []
train_loss_set = []
hold_accuracy = []
hold_loss_set = []
highest = 0

with tf.Session(graph=graph) as session:
    
    # Initializate the weights and biases
    tf.global_variables_initializer().run()

    saved_loss=0;
    saved_errrate=0;

    for curr_epoch in range(num_epochs):
        
        train_cost = train_ler = 0
        my_train_error_rate = 0
        start = time.time()


        for batch in range(num_batches_per_epoch):

#            # Getting the index
#            indexes = [i % num_examples for i in range(batch * batch_size, (batch + 1) * batch_size)]
#            
#            batch_train_inputs = train_inputs[indexes]
#            
#            # Padding input to max_time_step of this batch
#            batch_train_inputs, batch_train_seq_len = pad_sequences(batch_train_inputs)
#
#            # Converting to sparse representation so as to to feed SparseTensor input
#            batch_train_targets = sparse_tuple_from(train_targets[indexes])


            #batch_train_inputs, batch_train_targets, batch_train_seq_len = TrainData.get_next_batch();
            batch_train_inputs, batch_train_targets, batch_train_seq_len = get_next_batch()

            feed = {inputs: batch_train_inputs,
                    targets: batch_train_targets,
                    seq_len: batch_train_seq_len//poolings}


            batch_cost, _ = session.run([cost, optimizer], feed)
            train_cost += batch_cost*batch_size
            train_ler += session.run(ler, feed_dict=feed)*batch_size






            d = session.run(decoded[0], feed_dict=feed);

            
            train_targets = decode_sparse_tensor(batch_train_targets);
            
            
            
            total = 0
            error_num = 0
            
            # Decoding
            dense_decoded = tf.sparse_tensor_to_dense(d, default_value=-1).eval(session=session)
        
            for i, seq in enumerate(dense_decoded):
        
                seq = [s-1 for s in seq if s != -1]
                seq = [alphabet[s] for s in seq]
                
                total+=1
                if(not(train_targets[i]==seq)):
                    error_num+=1
                
           
            my_train_error_rate += error_num/total*batch_size
        






#        # Shuffle the data
#        shuffled_indexes = np.random.permutation(num_examples)
#        train_inputs = train_inputs[shuffled_indexes]
#        train_targets = train_targets[shuffled_indexes]


        # Metrics mean
        train_cost /= num_examples
        train_ler /= num_examples
        my_train_error_rate /= num_examples



        log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, time = {:.3f}"
        print(log.format(curr_epoch+1, num_epochs, train_cost, train_ler, time.time() - start)+"  my errorRate: "+str(my_train_error_rate))
        #print(log.format(curr_epoch+1, num_epochs, train_cost, train_ler, time.time() - start))

        saved_loss = train_cost;
        saved_errrate = train_ler;

        #######################################################################
        # Validation
        
#        saver = tf.train.Saver();
#        saver.save(session, './my-model');
        
        
        #if ((curr_epoch%REPORT_STEP)==(REPORT_STEP-1)):
            # Decoding all at once. Note that this isn't the best way
        
        #    # Padding input to max_time_step of this batch
        #    batch_train_inputs, batch_train_seq_len = pad_sequences(train_inputs)
        #
        #    # Converting to sparse representation so as to to feed SparseTensor input
        #    batch_train_targets = sparse_tuple_from(train_targets)
        
        batch_val_inputs, batch_val_targets, batch_val_seq_len = get_next_batch();
       
        
        val_feed = {inputs: batch_val_inputs,
                targets: batch_val_targets,
                seq_len: batch_val_seq_len//poolings
                }
    
    
        val_targets = decode_sparse_tensor(batch_val_targets);
        
        
        val_cost, val_ler, val_d = session.run([cost, ler, decoded[0]], feed_dict=val_feed)
        

        
        train_accuracy.append(1-train_ler)
        train_loss_set.append(train_cost)
        hold_accuracy.append(1-val_ler)
        hold_loss_set.append(val_cost)
       
        if(highest < 1-val_ler):
            highest = 1-val_ler
            
        total = 0
        error_num = 0
        
        # Decoding
        val_dense_decoded = tf.sparse_tensor_to_dense(val_d, default_value=-1).eval(session=session)
    
        for i, seq in enumerate(val_dense_decoded):
    
            seq = [s for s in seq if s != -1]
            seq = [alphabet[s] for s in seq]
            
            total+=1
            if(not(val_targets[i]==seq)):
                error_num+=1
            
            val_ler = error_num/total
       
            #print('Sequence %d' % i)
            #print('\t Original:%s' % val_targets[i])
            #print('\t Decoded:%s' % seq)
        print();
        print();
    
        print("          validation loss:"+str(val_cost)+"   validation error rate:"+str(val_ler));
    
        
        
        
    saver = tf.train.Saver();
    saver.save(session, './my-model');
    print();
    print();
    print();
    print("saved when train loss is "+str(saved_loss)+"   and train error rate is "+str(saved_errrate))
    print();



###############################################################################
print();
print("Test");
print();
            
    
with tf.Session(graph=graph) as test_session:
    tf.global_variables_initializer().run()
     
    saver = tf.train.import_meta_graph('my-model.meta')
    saver.restore(test_session,tf.train.latest_checkpoint('./'))
    
    
    #batch_test_inputs, batch_test_targets, batch_test_seq_len = TrainData.get_next_batch()
    
    
    batch_test_inputs, batch_test_targets, batch_test_seq_len = get_next_batch();
               
    test_feed = {inputs: batch_test_inputs,
            targets: batch_test_targets,
            seq_len: batch_test_seq_len//poolings
            }
    
    test_targets = decode_sparse_tensor(batch_test_targets);
                
    
    test_cost, test_ler, test_d = test_session.run([cost, ler, decoded[0]], feed_dict=test_feed);
    
                    

    test_dense_decoded = tf.sparse_tensor_to_dense(test_d, default_value=-1).eval()
    
        
    total = 0        
    error_num = 0

                
    for i, seq in enumerate(test_dense_decoded):         
        seq = [s for s in seq if s != -1]
        seq = [alphabet[s] for s in seq]
    
        print('Sequence %d' % i)
        print('\t Original:%s' % test_targets[i])
        print('\t Decoded:%s' % seq)

        total+=1
        if(not(test_targets[i]==seq)):
            error_num+=1
            
    test_ler = error_num/total
    
    print();           
    print("test loss:"+str(test_cost)+"     test error rate:"+str(test_ler));   
    print();
    
    
    
    
plt.figure(num=None, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
plt.title('Correct Rate:optimized')
   
l1,=plt.plot(train_accuracy, 'r')
l2,=plt.plot(hold_accuracy, 'b')
plt.legend((l1,l2),('Train Set', 'Hold on Set'),
    loc='upper right',fontsize=13)
plt.xlabel('epoch/updates')
plt.ylabel('Correct Rate')

plt.figure(num=None, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
plt.title('training loss/validation loss VS number of epochs')
   
l1,=plt.plot(train_loss_set, 'r')
l2,=plt.plot(hold_loss_set, 'b')
plt.legend((l1,l2),('Training Set', 'Validation Set'),
              loc='upper right',fontsize=13)
plt.xlabel('number of epochs')
plt.ylabel('loss')