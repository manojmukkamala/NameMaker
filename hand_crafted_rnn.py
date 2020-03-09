import numpy as np

### Useful Functions

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis = 0)

def get_initial_loss(vocab_size, seq_length):
    return -np.log(1.0 / vocab_size) * seq_length

def smooth(loss, cur_loss):
    return loss * 0.999 + cur_loss * 0.001

def print_sample(sample_ix, ix_to_char):
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print (txt)


### Model Building


#### Parameters Initialization

def initialize_parameters(n_a, n_x, n_y):

    Wax = np.random.randn(n_a, n_x) * 0.01
    Waa = np.random.randn(n_a, n_a) * 0.01
    Wya = np.random.randn(n_y, n_a) * 0.01

    ba = np.zeros((n_a, 1))
    by = np.zeros((n_y, 1))

    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}

    return parameters


#### Gradient Clipping

def clip(gradients, maxValue):

    dWaa, dWax, dWya, dba, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['dba'], gradients['dby']

    for gradient in [*gradients.keys()]:
        np.clip(gradients[gradient], -maxValue, maxValue, gradients[gradient])

    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "dba": dba, "dby": dby}

    return gradients


#### Parameter Updates

def update_parameters(parameters, gradients, learning_rate):

    parameters['Wax'] += -learning_rate * gradients['dWax']
    parameters['Waa'] += -learning_rate * gradients['dWaa']
    parameters['Wya'] += -learning_rate * gradients['dWya']
    parameters['ba']  += -learning_rate * gradients['dba']
    parameters['by']  += -learning_rate * gradients['dby']

    return parameters


#### Forward Propagation

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


#### Cost Function

def compute_cost(y_pred, Y):

    m = Y.shape[1]
    cost = (-1/m)*(np.sum(Y * np.log(y_pred)))

    return cost


#### BackPropagation

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


#### Sampling

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


#### Putting Together

def optimize(X, Y, a_prev, parameters, learning_rate = 0.001):

    a, y_pred, caches = rnn_forward(X, a_prev, parameters)
    
    cost = compute_cost(y_pred, Y)

    gradients = rnn_backward(X, Y, y_pred, caches)

    gradients = clip(gradients, 5)

    parameters = update_parameters(parameters, gradients, learning_rate)
    
    return cost, gradients, a[:, :, -1]


#### Model

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
      if j % 2000 == 0:
        print('Iteration: %d, Loss: %f' % (j, loss) + '\n')

        for name in range(sample_size):
          sampled_indices = sample(parameters, chr_to_idx)
          print_sample(sampled_indices, idx_to_chr)

    return parameters


