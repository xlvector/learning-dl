# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
sys.path.insert(0, "../../python")
import numpy as np
import mxnet as mx

from lstm import lstm_unroll
from sort_io import BucketSentenceIter, default_build_vocab
from rnn_model import LSTMInferenceModel

def MakeInput(char, vocab, arr):
    idx = vocab[char]
    tmp = np.zeros((1,))
    tmp[0] = idx
    arr[:] = tmp

if __name__ == '__main__':
    batch_size = 32
    #buckets = [10, 20, 30, 40, 50, 60]
    #buckets = [32]
    buckets = []
    num_hidden = 200
    num_embed = 200
    num_lstm_layer = 3

    num_epoch = 3
    learning_rate = 0.1
    momentum = 0.9

    # dummy data is used to test speed without IO
    dummy_data = False

    contexts = [mx.context.gpu(i) for i in range(1)]

    vocab = default_build_vocab("./data/sort.train.txt")
    rvocab = {}
    for k, v in vocab.items():
        rvocab[v] = k
    def sym_gen(seq_len):
        return lstm_unroll(num_lstm_layer, seq_len, len(vocab),
                           num_hidden=num_hidden, num_embed=num_embed,
                           num_label=len(vocab))

    init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h

    if len(buckets) == 1:
        # only 1 bucket, disable bucketing
        symbol = sym_gen(buckets[0])
    else:
        symbol = sym_gen

    _, arg_params, __ = mx.model.load_checkpoint("sort", 3)

    model = LSTMInferenceModel(num_lstm_layer, len(vocab),
                               num_hidden=num_hidden, num_embed=num_embed,
                               num_label=len(vocab), arg_params=arg_params, ctx=contexts, dropout=0.2)

    tks = sys.argv[1:]
    input_ndarray = mx.nd.zeros((1,))
    for k in range(len(tks)):
        MakeInput(tks[k], vocab, input_ndarray)
        prob = model.forward(input_ndarray, False)
        idx = np.argmax(prob, axis=1)[0]
        print tks[k], prob.shape, idx, rvocab[idx]
    for k in range(len(tks)):
        MakeInput(' ', vocab, input_ndarray)
        prob = model.forward(input_ndarray, False)
        idx = np.argmax(prob, axis=1)[0]
        print ' ', prob.shape, idx, rvocab[idx]
