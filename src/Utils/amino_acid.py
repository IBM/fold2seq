import numpy as np
from collections import OrderedDict
my_seqlabel = ["!","A","C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", 
    "S", "T", "V", "W", "Y", "*", '0' ] # * is the start symbol ! is the end symbol

my_seqlabel_reverse  = OrderedDict([
    ("A", 1),
    ("C", 2),
    ("D", 3),
    ("E", 4),
    ("F", 5),
    ("G", 6),
    ("H", 7),
    ("I", 8),
    ("K", 9),
    ("L", 10),
    ("M", 11),
    ("N", 12),
    ("P", 13),
    ("Q", 14),
    ("R", 15),
    ("S", 16),
    ("T", 17),
    ("V", 18),
    ("W", 19),
    ("Y", 20),])

def seqlabel(seq):
    vec = np.zeros(len(seq)+1, dtype=np.int32)
    for i in range(len(seq)):
        vec[i] = my_seqlabel_reverse[seq[i]]
    return vec


def Nature_seq(seq):

	for i in seq:
		if i not in amino_acid:
			return False

	return True



import matplotlib.pyplot as plt
def seq_length_plot(seq,name='seq_dis', maxlen=2000):

        length = []
        for i in seq:
                length.append(len(i['seq']))

        plt.hist(length, bins=100, range=[0,maxlen])
        plt.xlabel("seq length")
        plt.ylabel("number")
        plt.savefig(name+".eps", format='eps')
        plt.show()


def transformer_integer_padding(seq, maxlen=502):
    vec = np.zeros((maxlen)+2, dtype=np.int32)+my_seqlabel.index('0')
    vec[0] = my_seqlabel.index('*')
    vec[len(seq)+1] =my_seqlabel.index('!')

    for i in range(0, len(seq)):
        vec[i+1] = my_seqlabel.index(seq[i])

    return vec


def transformer_integer(seq):
    vec = np.zeros(len(seq)+2, dtype=np.int32)
    vec[0]=len(my_seqlabel)-1
    vec[-1] = 0

    for i in range(0, len(seq)):
        vec[i+1] = my_seqlabel_reverse[seq[i]]

    return vec

# interger encoding and padding
def to_intger_padding(seq, maxlen):
    vec = np.zeros((maxlen), dtype=np.int32)
    for i in range(len(seq)):
        #print (np.where(amino_acid==seq[i]), seq[i], seq)
        vec[i] = amino_acid.index(seq[i])+1
    
    return vec


def to_onehot(seq, hparam, start=0):
    onehot = np.zeros((hparam['MAXLEN'], hparam['vocab_size']), dtype=np.int32)
    l = min(MAXLEN, len(seq))
    for i in range(start, start + l):
        onehot[i, AAINDEX.get(seq[i - start], 0)] = 1
    onehot[0:start, 0] = 1
    onehot[start + l:, 0] = 1
    return onehot


def to_int(seq, hparam):

    out =  np.zeros(hparam['MAXLEN'], dtype=np.int32)
    for i in range(len(seq)):
        out[i] = AAINDEX[seq[i]]

    return out