# -*- coding: utf-8 -*-
"""RNN RnD.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WMbrZpuGxULcWp7wpN1QwJEQpaflortr
"""

import numpy as np
import pandas as pd

import string

chr_to_idx = {j:i for i,j in enumerate(string.ascii_lowercase)}
idx_to_chr = {i:j for i,j in enumerate(string.ascii_lowercase)}

chr_to_idx['\n'] = 26
idx_to_chr[26] = '\n'

vocab_size = len(chr_to_idx)

def sigmoid(x):

  return 1/(1 + np.exp(-x))

def softmax(x):

  e_x = np.exp(x - np.max(x))

  return e_x / e_x.sum(axis = 0)

def get_initial_loss(vocab_size, seq_length):

    return -np.log(1.0/vocab_size)*seq_length

def smooth(loss, cur_loss):

    return loss * 0.999 + cur_loss * 0.001

def print_sample(sample_ix, ix_to_char):
  txt = ''.join(ix_to_char[ix] for ix in sample_ix)
  print (txt)

def initialize_parameters(n_a, n_x, n_y):

  Wax = np.random.randn(n_a, n_x)*0.01
  Waa = np.random.randn(n_a, n_a)*0.01
  Wya = np.random.randn(n_y, n_a)*0.01

  ba = np.zeros((n_a, 1))
  by = np.zeros((n_y, 1))

  parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}

  return parameters

def clip(gradients, maxValue):

    dWaa, dWax, dWya, dba, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['dba'], gradients['dby']

    for gradient in [*gradients.keys()]:
        np.clip(gradients[gradient], -maxValue, maxValue, gradients[gradient])

    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "dba": dba, "dby": dby}

    return gradients

def update_parameters(parameters, gradients, lr):

    parameters['Wax'] += -lr * gradients['dWax']
    parameters['Waa'] += -lr * gradients['dWaa']
    parameters['Wya'] += -lr * gradients['dWya']
    parameters['ba']  += -lr * gradients['dba']
    parameters['by']  += -lr * gradients['dby']

    return parameters

def rnn_cell_forward(xt, a_prev, parameters):

  Wax = parameters['Wax']
  Waa = parameters['Waa']
  Wya = parameters['Wya']

  ba = parameters['ba']
  by = parameters['by']

  a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba)
  yt_pred = softmax(np.dot(Wya, a_next) + by)

  cache = (a_next, a_prev, xt, parameters)

  return a_next, yt_pred, cache

def rnn_forward(x, a_prev, parameters):

  caches = []
  loss = 0

  n_x, m, T_x = x.shape
  n_y, n_a = parameters['Wya'].shape

  a = np.zeros(shape = (n_a, m, T_x))
  y_pred = np.zeros(shape = (n_y, m, T_x))

  for t in range(T_x):
    xt = x[:, :, t]
    a_next, yt_pred, cache = rnn_cell_forward(xt, a_prev, parameters)
    a[:, :, t] = a_next
    y_pred[:, :, t] = yt_pred
    a_prev = a_next
    caches.append(cache)

  caches = (caches, x)

  return a, y_pred, caches

def compute_cost(y_pred, Y):

  m = Y.shape[1]
  cost = (-1/m)*(np.sum(Y * np.log(y_pred)))

  return cost

def rnn_cell_backward(da_next, dZt, cache):

    (a_next, a_prev, xt, parameters) = cache
    m = xt.shape[1]

    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    dz = 1 - np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba) ** 2

    dxt = np.dot(Wax.T, da_next * dz)
    dWax = (1/m) * np.dot(da_next * dz, xt.T)

    da_prev = np.dot(Waa.T, da_next * dz)
    dWaa = (1/m) * np.dot(da_next * dz, a_prev.T)

    dba = (1/m) * np.sum(da_next * dz, axis = 1, keepdims = True)

    dWya = (1/m) * np.dot(dZt, a_next.T)
    dby = (1/m) * np.sum(dZt, axis = 1, keepdims=True)

    gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba, "dWya": dWya, "dby": dby}

    return gradients

def rnn_backward(X, Y, y_pred, caches):

    (caches, x) = caches
    (a1, a0, x1, parameters) = caches[0]

    n_a, m = a0.shape
    n_x, m, T_x = X.shape
    n_y, n_a = parameters['Wya'].shape

    dx = np.zeros((n_x, m, T_x))
    dWax = np.zeros((n_a, n_x))
    dWaa = np.zeros((n_a, n_a))
    dba = np.zeros((n_a, 1))
    dWya = np.zeros((n_y, n_a))
    dby = np.zeros((n_y, 1))

    da0 = np.zeros((n_a, m))
    da_next = np.zeros((n_a, m))

    dZ = y_pred - Y

    for t in reversed(range(T_x)):

      dZt = dZ[:, :, t]
      da_next = np.dot(parameters['Wya'].T, dZt) + da_next

      gradients = rnn_cell_backward(da_next, dZt, caches[t])
      dxt, da_next, dWaxt, dWaat, dbat, dWyat, dbyt = gradients["dxt"], gradients["da_prev"], gradients["dWax"], gradients["dWaa"], gradients["dba"], gradients["dWya"], gradients["dby"]
      dx[:, :, t] = dxt
      dWax += dWaxt
      dWaa += dWaat
      dba += dbat
      dWya += dWyat
      dby += dbyt

    da0 = da_next

    gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa,"dba": dba, "dWya": dWya,"dby": dby}

    return gradients





def split(word):
  return [i for i in word]

def punc_check(x):
  return True not in ([i in [i for i in string.punctuation] for i in x])

def letter_check(x):
  return False in ([i not in [i for i in string.ascii_lowercase] for i in x])

def number_check(x):
  return False not in ([i not in [str(i) for i in [*range(0, 9)]] for i in x])

def token_idx(tokens):
  return [chr_to_idx[i] for i in tokens]

def end_pad(token, max_length):
  return token + [chr_to_idx['\n']] * (max_length - len(token))

def one_hot(A, classes):

  _, m, T = A.shape
  n_C = classes

  A_one_hot = np.zeros((n_C, m, T))

  for t in range(T):
    for i, j in enumerate(np.squeeze(A[:, :, t])):
      A_one_hot[j, i, t] = 1

  return A_one_hot


#orig_data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/names.csv', header = None)
orig_data = pd.read_csv('/Users/Manoj/Downloads/names.csv')
data = orig_data
data.shape

data.columns = ['name', 'dummy']
data.drop('dummy', axis = 1, inplace = True)

data.dropna(inplace = True)

data['name'] = data['name'].apply(lambda x: x.split(' ')[0])

data = data['name'].apply(lambda x: split(x))

data = data[data.apply(lambda x: punc_check(x))]

data = data[data.apply(lambda x: letter_check(x))]

data = data[data.apply(lambda x: number_check(x))]

#data.apply(lambda x: len(x)).plot.hist()

data = data[data.apply(lambda x: len(x) > 3 and len(x) < 8)]

data = data.apply(lambda x: token_idx(x))

max_length = max([len(i) for i in data])
data = data.apply(lambda x: end_pad(x, max_length))

data = data.sample(frac = 1)

data_subset = data[0:1000]
m = len(data_subset)

X_tokens = np.array(data_subset.values.tolist()).reshape(1, m, -1)
Y_tokens = np.array(data_subset.apply(lambda x: x[1:]).apply(lambda x: x + [chr_to_idx['\n']]).values.tolist()).reshape(1, m, -1)

X = one_hot(X_tokens, vocab_size)
Y = one_hot(Y_tokens, vocab_size)

print(X.shape)
print(Y.shape)

def sample(parameters, char_to_ix):

    Waa, Wax, Wya, by, ba = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['ba']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]

    x = np.zeros(shape = (vocab_size, 1))

    a_prev = np.zeros((n_a, 1))

    indices = []

    idx = -1

    counter = 0
    newline_character = char_to_ix['\n']

    while (idx != newline_character and counter != 50):

        a = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, x) + ba)
        z = np.dot(Wya, a) + by
        y = softmax(z)

        idx = np.random.choice(range(vocab_size), p = np.ravel(y))
        indices.append(idx)
        x = np.zeros(shape = (vocab_size, 1))
        x[idx] = 1

        a_prev = a
        counter +=1

    if (counter == 50):
        indices.append(char_to_ix['\n'])

    return indices

def optimize(X, Y, a_prev, parameters, learning_rate = 0.001):

    a, y_pred, caches = rnn_forward(X, a_prev, parameters)

    cost = compute_cost(y_pred, Y)

    gradients = rnn_backward(X, Y, y_pred, caches)

    gradients = clip(gradients, 5)

    parameters = update_parameters(parameters, gradients, learning_rate)

    return cost, gradients, a[:, :, -1]

def model(X, Y, idx_to_chr, chr_to_idx, num_iterations = 35000, n_a = 50, sample_size = 7, vocab_size = 27):

    n_x, n_y = vocab_size, vocab_size

    m = Y.shape[1]

    parameters = initialize_parameters(n_a, n_x, n_y)

    loss = get_initial_loss(vocab_size, sample_size)

    a_prev = np.zeros((n_a, m))

    # Optimization loop
    for j in range(num_iterations):

      curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters)

      loss = smooth(loss, curr_loss)


      # Every 2000 Iteration, generate "n" characters thanks to sample() to check if the model is learning properly
      if j % 5000 == 0:
        print('Iteration: %d, Loss: %f' % (j, loss) + '\n')

        for name in range(sample_size):
          sampled_indices = sample(parameters, chr_to_idx)
          print_sample(sampled_indices, idx_to_chr)

    return parameters

parameters = model(X, Y, idx_to_chr, chr_to_idx, num_iterations = 35000, n_a = 50, sample_size = 7, vocab_size = vocab_size)

# import pandas as pd

# df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/dinos.txt', sep = '\n', header = None)

# df = pd.Series(np.squeeze(df.values))
# df = df.apply(lambda x: split(x.lower()))
# df = df[df.apply(lambda x: len(x) > 7 and len(x) < 14)]
# #df = df[df.apply(lambda x: len(x) == 9)]
# df = df.apply(lambda x: token_idx(x))

# max_length = max([len(i) for i in df])
# df = df.apply(lambda x: end_pad(x, max_length))

# df = df.sample(frac = 1)
# data_subset = df

# m = len(data_subset)

# X_tokens = np.array(data_subset.values.tolist()).reshape(1, m, -1)
# Y_tokens = np.array(data_subset.apply(lambda x: x[1:]).apply(lambda x: x + [chr_to_idx['\n']]).values.tolist()).reshape(1, m, -1)

# X = one_hot(X_tokens, vocab_size)
# Y = one_hot(Y_tokens, vocab_size)

# parameters = model(X, Y, idx_to_chr, chr_to_idx, num_iterations = 35000, n_a = 50, sample_size = 7, vocab_size = vocab_size)
