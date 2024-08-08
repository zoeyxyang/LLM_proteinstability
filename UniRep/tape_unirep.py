

from typing import Union, List, Tuple, Sequence, Dict, Any, Optional
from copy import copy
from pathlib import Path
import pickle as pkl
import logging
import random

import lmdb
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.spatial.distance import pdist, squareform

from sklearn.metrics import r2_score
saver = tf.train.Saver()
from scipy.stats import spearmanr

USE_FULL_1900_DIM_MODEL = False # if True use 1900 dimensional model, else use 64 dimensional one.


## Downloading the Dataset

# !pip install lmdb
# !curl -O http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/stability.tar.gz
# !tar -zxvf stability.tar.gz
# !tar -zxvf stability.tar.gz
# !pip install torch 




logger = logging.getLogger(__name__)

class LMDBDataset(Dataset):
    """Creates a dataset from an lmdb file.
    Args:
        data_file (Union[str, Path]): Path to lmdb file.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """

    def __init__(self,
                 data_file: Union[str, Path],
                 in_memory: bool = False):

        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)

        env = lmdb.open(str(data_file), max_readers=1, readonly=True,
                        lock=False, readahead=False, meminit=False)

        with env.begin(write=False) as txn:
            num_examples = pkl.loads(txn.get(b'num_examples'))

        if in_memory:
            cache = [None] * num_examples
            self._cache = cache

        self._env = env
        self._in_memory = in_memory
        self._num_examples = num_examples

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        if self._in_memory and self._cache[index] is not None:
            item = self._cache[index]
        else:
            with self._env.begin(write=False) as txn:
                item = pkl.loads(txn.get(str(index).encode()))
                if 'id' not in item:
                    item['id'] = str(index)
                if self._in_memory:
                    self._cache[index] = item
        return item
    
    
    



print(LMDBDataset('stability/stability_test.lmdb').__len__())
print(LMDBDataset('stability/stability_train.lmdb').__len__()) 
xdata=LMDBDataset('stability/stability_train.lmdb')
xtdata=LMDBDataset('stability/stability_test.lmdb')
dic={}
X=[]
Y=[]
X_test=[]
Y_test=[]
for i in range(LMDBDataset('stability/stability_train.lmdb').__len__()):
    if i%5000==0:
        print(i)
    X.append(xdata.__getitem__(i)['primary'])
    Y.append(xdata.__getitem__(i)['stability_score'][0])
    
for i in range(LMDBDataset('stability/stability_test.lmdb').__len__()):
    if i%5000==0:
        print(i)    
    X_test.append(xtdata.__getitem__(i)['primary'])
    Y_test.append(xtdata.__getitem__(i)['stability_score'][0])
    

# ## Setup


import tensorflow as tf
import numpy as np

# Set seeds
tf.set_random_seed(42)
np.random.seed(42)

if USE_FULL_1900_DIM_MODEL:
    # Sync relevant weight files
    get_ipython().system('aws s3 sync --no-sign-request --quiet s3://unirep-public/1900_weights/ 1900_weights/')
    
    # Import the mLSTM babbler model
    from unirep import babbler1900 as babbler
    
    # Where model weights are stored.
    MODEL_WEIGHT_PATH = "./1900_weights"
    
else:
    # Sync relevant weight files
    get_ipython().system('aws s3 sync --no-sign-request --quiet s3://unirep-public/64_weights/ 64_weights/')
    
    # Import the mLSTM babbler model
    from unirep import babbler64 as babbler
    
    # Where model weights are stored.
    MODEL_WEIGHT_PATH = "./64_weights"


# ## Data formatting and management

# Initialize UniRep, also referred to as the "babbler" in our code. You need to provide the batch size you will use and the path to the weight directory.




batch_size = 512
b = babbler(batch_size=batch_size, model_path=MODEL_WEIGHT_PATH)


# UniRep needs to receive data in the correct format, a (batch_size, max_seq_len) matrix with integer values, where the integers correspond to an amino acid label at that position, and the end of the sequence is padded with 0s until the max sequence length to form a non-ragged rectangular matrix. We provide a formatting function to translate a string of amino acids into a list of integers with the correct codex:



##Sample 
#seq = "MRKGEELFTGVVPILVELDGDVNGHKFSVRGEGEGDATNGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFARYPDHMKQHDFFKSAMPEGYVQERTISFKDDGTYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNFNSHNVYITADKQKNGIKANFKIRHNVEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSVLSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
#np.array(b.format_seq('GSWKGYATANKKQAPTEEYLKDNAEQSGVDYAEKTKGKLEVDK'))
#b.is_valid_seq('GSWKGYATANKKQAPTEEYLKDNAEQSGVDYAEKTKGKLEVDK')








# We also provide a function that will check your amino acid sequences don't contain any characters which will break the UniRep model.


# You could use your own data flow as long as you ensure that the data format is obeyed. Alternatively, you can use the data flow we've implemented for UniRep training, which happens in the tensorflow graph. It reads from a file of integer sequences, shuffles them around, collects them into groups of similar length (to minimize padding waste) and pads them to the max_length. Here's how to do that:

# First, sequences need to be saved in the correct format. Suppose we have a new-line seperated file of amino acid sequences, `seqs.txt`, and we want to format them. Note that training is currently only publicly supported for amino acid sequences less than 275 amino acids as gradient updates for sequences longer than that start to get unwieldy. If you want to train on sequences longer than this, please reach out to us. 
# 
# Sequence formatting can be done as follows:


# Before you can train your model, 
# with open("seqs.txt", "r") as source:
XX=[]
YY=[]
with open("formatted_new.txt", "w") as destination:
    for i,seq in enumerate(X):
#         seq = seq.strip()
        if b.is_valid_seq(seq) and len(seq) < 275: 
            formatted = ",".join(map(str,b.format_seq(seq)))
            x=formatted.split(',')
            x=[int(t) for t in x]
            XX.append(x)
            YY.append(int(Y[i]*100))
            formatted=formatted+','+str(int(Y[i]*100))
            destination.write(formatted)
            
            destination.write('\n')
XX_test=[]
YY_test=[]           
with open("formatted_new_test.txt", "w") as destination:
    for i,seq in enumerate(X_test):
#         seq = seq.strip()
        if b.is_valid_seq(seq) and len(seq) < 275: 
            formatted = ",".join(map(str,b.format_seq(seq)))
            x=formatted.split(',')
            x=[int(t) for t in x]
            XX_test.append(x)
            YY_test.append(int(Y_test[i]*100))
            formatted=formatted+','+str(int(Y_test[i]*100))
            destination.write(formatted)
            
            destination.write('\n')




get_ipython().system('head -n1 formatted_new.txt')


# Notice that by default format_seq does not include the stop symbol (25) at the end of the sequence. This is the correct behavior if you are trying to train a top model, but not if you are training UniRep representations.

# Now we can use a custom function to bucket, batch and pad sequences from `formatted.txt` (which has the correct integer codex after calling `babbler.format_seq()`). The bucketing occurs in the graph. 
# 
# What is bucketing? Specify a lower and upper bound, and interval. All sequences less than lower or greater than upper will be batched together. The interval defines the "sides" of buckets between these bounds. Don't pick a small interval for a small dataset because the function will just repeat a sequence if there are not enough to
# fill a batch. All batches are the size you passed when initializing the babbler.
# 
# This is also doing a few other things:
# - Shuffling the sequences by randomly sampling from a 10000 sequence buffer
# - Automatically padding the sequences with zeros so the returned batch is a perfect rectangle
# - Automatically repeating the dataset


bucket_op = b.bucket_batch_pad("formatted_new.txt", interval=1000) # Large interval

def make_batches(trainX, trainY, batch_size):
    """
    Split `trainX` and `trainY` into batches of size `batch_size`.
    """
    assert len(trainX) == len(trainY), "trainX and trainY must have the same length"
    
    # Shuffle the data
#     indices = np.arange(len(trainX))
#     np.random.shuffle(indices)
#     trainX = trainX[indices]
#     trainY = trainY[indices]
    
    # Split the data into batches
    batches = []
    for i in range(0, len(trainX), batch_size):
        batchX = trainX[i:i+batch_size]
        batchY = trainY[i:i+batch_size]
        batches.append((batchX, batchY))
        
    return batches





# Inconveniently, this does not make it easy for a value to be associated with each sequence and not lost during shuffling. You can get around this by just prepending every integer sequence with the sequence label (eg, every sequence would be saved to the file as "{brightness value}, 24, 1, 5,..." and then you could just index out the first column after calling the `bucket_op`. Please reach out if you have questions on how to do this.

# Now that we have the `bucket_op`, we can simply `sess.run()` it to get a correctly formatted batch



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batch = sess.run(bucket_op)
    




# XX = [row[:-1] for row in batch]

# # Get only the last column
# YY = [row[-1] for row in batch]
print(XX[:3],YY[:3])
a=make_batches(XX,YY,512)
a_test=make_batches(XX_test,YY_test,512)


# You can look back and see that the batch_size we passed to __init__ is indeed 12, and the second dimension must be the longest sequence included in this batch. Now we have the data flow setup (note that as long as your batch looks like this, you don't need my flow), so we can proceed to implementing the graph. The module returns all the operations needed to feed in sequence and get out trainable representations.

# ## Training a top model and a top model + mLSTM.

# First, obtain all of the ops needed to output a representation

final_hidden, x_placeholder, batch_size_placeholder, seq_length_placeholder, initial_state_placeholder = (
    b.get_rep_ops())


# `final_hidden` should be a batch_size x rep_dim matrix.
# 
# Lets say we want to train a basic feed-forward network as the top model, doing regression with MSE loss, and the Adam optimizer. We can do that by:
# 
# 1.  Defining a loss function.
# 
# 2.  Defining an optimizer that's only optimizing variables in the top model.
# 
# 3.  Minimizing the loss inside of a TensorFlow session



y_placeholder = tf.placeholder(tf.float32, shape=[None,1], name="y")
initializer = tf.contrib.layers.xavier_initializer(uniform=False)

with tf.variable_scope("top"):
    prediction = tf.contrib.layers.fully_connected(
        final_hidden, 1, activation_fn=None, 
        weights_initializer=initializer,
        biases_initializer=tf.zeros_initializer(),

    )

loss = tf.losses.mean_squared_error(y_placeholder, prediction)


learning_rate=.001
top_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="top")
optimizer = tf.train.AdamOptimizer(learning_rate)
top_only_step_op = optimizer.minimize(loss, var_list=top_variables)
all_step_op = optimizer.minimize(loss)



# We next need to define a function that allows us to calculate the length each sequence in the batch so that we know what index to use to obtain the right "final" hidden state



def nonpad_len(batch):
    nonzero = batch > 0
   
    lengths = np.sum(nonzero, axis=1)

    return [44]*512





# We are ready to train. As an illustration, let's learn to predict the number 42 just optimizing the top model.

# y = [[row/100] for row in YY]
num_iters = 10
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_iters):
#         batch = sess.run(bucket_op)

#         XX = [row[:-1] for row in batch]

#         # Get only the last column
#         YY = [row[-1] for row in batch]
        #batch = sess.run(bucket_op)
        for ii in range(len(a[:50])):
            if(ii==71):
                continue
            if ii%25==0:
                print(i,ii)
            
    #         print(batch)
    #         print(XX)
    #         print(y)
    #         print(batch_size)
    #         print(length)
    #         print(b._zero_state)
            XX,YY=a[ii]
            y=[[row/100] for row in YY]
            
            length = [44]*512 #nonpad_len(XX)
            loss_, __, = sess.run([loss, top_only_step_op],
                    feed_dict={
                         x_placeholder: XX,
                         y_placeholder: y,
                         batch_size_placeholder: 512,
                         seq_length_placeholder:length,
                        initial_state_placeholder:b._zero_state
                                 
                    }
            )
        
        print("Iteration {0}: {1}".format(i, loss_))
        if(i%3==0):
            saver = tf.train.Saver()
            save_path = saver.save(sess, "model_a_{0}.ckpt".format(i))
            print("Model saved in file: {0}".format(save_path))





# y = [[row/100] for row in YY]
num_iters = 10
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_iters):
#         batch = sess.run(bucket_op)

#         XX = [row[:-1] for row in batch]

#         # Get only the last column
#         YY = [row[-1] for row in batch]
        #batch = sess.run(bucket_op)
        for ii in range(len(a[:25])):
            if(ii==71):
                continue
            if ii%25==0:
                print(i,ii)
            
    #         print(batch)
    #         print(XX)
    #         print(y)
    #         print(batch_size)
    #         print(length)
    #         print(b._zero_state)
            XX,YY=a[ii]
            y=[[row/100] for row in YY]
            
            length = [44]*512 #nonpad_len(XX)
            loss_, __, = sess.run([loss, all_step_op],
                    feed_dict={
                         x_placeholder: XX,
                         y_placeholder: y,
                         batch_size_placeholder: 512,
                         seq_length_placeholder:length,
                        initial_state_placeholder:b._zero_state
                                 
                    }
            )
        
        print("Iteration {0}: {1}".format(i, loss_))
        if(i%3==0):
            saver = tf.train.Saver()
            save_path = saver.save(sess, "model_a_full_{0}.ckpt".format(i))
            print("Model saved in file: {0}".format(save_path))


len(a[71][0])


# Test the model
with tf.Session() as sess:
    # Restore the variables from the last saved checkpoint
    saver.restore(sess, "model_a_6.ckpt")
    print("Model restored from file: model_a_6.ckpt")

    # Evaluate the model on the test data
    test_loss = 0.0
    for XX_testt, YY_testt in (a_test):
        
        y_test = [[row/100] for row in YY_testt]
        length_test = [44]*len(XX_testt) #nonpad_len(XX_test)
        test_loss += sess.run(loss,
                              feed_dict={
                                  x_placeholder: XX_testt,
                                  y_placeholder: y_test,
                                  batch_size_placeholder: len(XX_testt),
                                  seq_length_placeholder:length_test,
                                  initial_state_placeholder:b._zero_state
                              })
        print('yay')
        
    # Print the test loss
    print("Test loss: {0}".format(test_loss))



print("Test loss: {0}".format(test_loss))



# Test the model and calculate the R2 score
with tf.Session() as sess:
    # Restore the variables from the last saved checkpoint
    saver.restore(sess, "model_a_6.ckpt")
    print("Model restored from file:model_a_6.ckpt")

    # Evaluate the model on the test data and calculate the R2 score
    test_loss = 0.0
    y_true = []
    y_pred = []
    for XX_test, YY_test in a[:50]:
        y_test = [[row/100] for row in YY_test]
        length_test = [44]*len(XX_test) #nonpad_len(XX_test)
        loss_test, y_pred_test = sess.run([loss, prediction],
                                          feed_dict={
                                              x_placeholder: XX_test,
                                              y_placeholder: y_test,
                                              batch_size_placeholder: len(XX_test),
                                              seq_length_placeholder:length_test,
                                              initial_state_placeholder:b._zero_state
                                          })
        test_loss += loss_test
        y_true.extend(y_test)
        y_pred.extend([p[0] for p in y_pred_test])

    # Print the test loss and the R2 score
    print("Test loss: {0}".format(test_loss))
    print("R2 score: {0}".format(r2_score(y_true, y_pred)))
    spearman_corr, _ = spearmanr(y_true, y_pred)
    print("Spearman's rank correlation coefficient: {0}".format(spearman_corr))




from sklearn.metrics import r2_score
saver = tf.train.Saver()
# Test the model and calculate the R2 score
with tf.Session() as sess:
    # Restore the variables from the last saved checkpoint
    saver.restore(sess, "model_a_6.ckpt")
    print("Model restored from file:model_a_full_9.ckpt")

    # Evaluate the model on the test data and calculate the R2 score
    test_loss = 0.0
    y_true = []
    y_pred = []
    for XX_test, YY_test in a[:50]:
        y_test = [[row/100] for row in YY_test]
        length_test = [44]*len(XX_test) #nonpad_len(XX_test)
        loss_test, y_pred_test = sess.run([loss, prediction],
                                          feed_dict={
                                              x_placeholder: XX_test,
                                              y_placeholder: y_test,
                                              batch_size_placeholder: len(XX_test),
                                              seq_length_placeholder:length_test,
                                              initial_state_placeholder:b._zero_state
                                          })
        test_loss += loss_test
        y_true.extend(y_test)
        y_pred.extend([p[0] for p in y_pred_test])

    # Print the test loss and the R2 score
    print("Test loss: {0}".format(test_loss/100))
    print("R2 score: {0}".format(r2_score(y_true, y_pred)/10))



print("Test loss: {0}".format(test_loss/100))
print("R2 score: {0}".format(r2_score(y_true, y_pred)/10))





print("Test loss: {0}".format(test_loss/100))
print("R2 score: {0}".format(r2_score(y_true, y_pred)/10))





# y = [[42]]*batch_size
# num_iters = 10
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(num_iters):
#         batch = sess.run(bucket_op)
#         length = nonpad_len(batch)
#         loss_, __, = sess.run([loss, all_step_op],
#                 feed_dict={
#                      x_placeholder: batch,
#                      y_placeholder: y,
#                      batch_size_placeholder: batch_size,
#                      seq_length_placeholder:length,
#                      initial_state_placeholder:b._zero_state
#                 }
#         )
        
#         print("Iteration {0}: {1}".format(i,loss_))

