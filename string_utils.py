import numpy as np
import string
import nltk

chr_to_idx = {j:i for i,j in enumerate(string.ascii_lowercase)}
idx_to_chr = {i:j for i,j in enumerate(string.ascii_lowercase)}

chr_to_idx['\n'] = 26
idx_to_chr[26] = '\n'

vocab_size = len(chr_to_idx)

def split_word(word): 
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
