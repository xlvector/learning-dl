# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys, random, os
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
from operator import itemgetter

class Batch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

class DataIter(mx.io.DataIter):
    def __init__(self, names, batch_size):
        super(DataIter, self).__init__()
        self.cache = []
        for name in names:
            self.cache.append(np.load(name))
        print 'load data ok'
        self.batch_size = batch_size
        self.provide_data = [('same', (batch_size, 3, 32, 32)), \
                             ('diff', (batch_size, 3, 32, 32)), \
                             ('one', (batch_size, ))]
        self.provide_label = [('anchor', (batch_size, 3, 32, 32))]
        
    def generate_batch(self, n):
        n1, n2 = random.sample(range(len(self.cache)), 2)
        d1 = self.cache[n1]
        d2 = self.cache[n2]
        ret = []
        while len(ret) < n:
            k1 = random.randint(0, len(d1) - 1)
            k2 = random.randint(0, len(d1) - 1)
            k3 = random.randint(0, len(d2) - 1)
            if k1 == k2:
                continue
            ret.append((d1[k1], d1[k2], d2[k3]))
        return ret

    def __iter__(self):
        print 'begin'
        count = 100000 / self.batch_size
        for i in range(count):
            batch = self.generate_batch(self.batch_size)
            batch_anchor = [x[0] for x in batch]
            batch_same = [x[1] for x in batch]
            batch_diff = [x[2] for x in batch]
            batch_one = np.ones(self.batch_size)
                        
            data_all = [mx.nd.array(batch_same), mx.nd.array(batch_diff), \
                        mx.nd.array(batch_one)]
            label_all = [mx.nd.array(batch_anchor)]
            data_names = ['same', 'diff', 'one']
            label_names = ['anchor']
            
            data_batch = Batch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        pass

def get_conv(data, conv_weight, conv_bias, fc_weight, fc_bias):
    cdata = data
    ks = [5, 3, 3]
    for i in range(3):
        cdata = mx.sym.Convolution(data=cdata, kernel=(ks[i],ks[i]), num_filter=32,
                                   weight = conv_weight[i], bias = conv_bias[i],
                                   name = 'conv' + str(i))
        cdata = mx.sym.Pooling(data=cdata, pool_type="avg", kernel=(2,2), stride=(1, 1))
        cdata = mx.sym.Activation(data=cdata, act_type="relu")

    cdata = mx.sym.Flatten(data = cdata)
    cdata = mx.sym.FullyConnected(data = cdata, num_hidden = 1024,
                                  weight = fc_weight, bias = fc_bias, name='fc')
    cdata = mx.sym.L2Normalization(data = cdata)
    return cdata

def get_sim_net():
    same = mx.sym.Variable('same')
    diff = mx.sym.Variable('diff')
    conv_weight = []
    conv_bias = []
    for i in range(3):
        conv_weight.append(mx.sym.Variable('conv' + str(i) + '_weight'))
        conv_bias.append(mx.sym.Variable('conv' + str(i) + '_bias'))
    fc_weight = mx.sym.Variable('fc_weight')
    fc_bias = mx.sym.Variable('fc_bias')
    fs = get_conv(same, conv_weight, conv_bias, fc_weight, fc_bias)
    fd = get_conv(diff, conv_weight, conv_bias, fc_weight, fc_bias)
    fs = fs - fd
    fs = fs * fs
    return mx.sym.sum(fs, axis = 1)


def get_net(batch_size):
    same = mx.sym.Variable('same')
    diff = mx.sym.Variable('diff')
    anchor = mx.sym.Variable('anchor')
    one = mx.sym.Variable('one')
    one = mx.sym.Reshape(data = one, shape = (-1, 1))
    conv_weight = []
    conv_bias = []
    for i in range(3):
        conv_weight.append(mx.sym.Variable('conv' + str(i) + '_weight'))
        conv_bias.append(mx.sym.Variable('conv' + str(i) + '_bias'))
    fc_weight = mx.sym.Variable('fc_weight')
    fc_bias = mx.sym.Variable('fc_bias')
    fa = get_conv(anchor, conv_weight, conv_bias, fc_weight, fc_bias)
    fs = get_conv(same, conv_weight, conv_bias, fc_weight, fc_bias)
    fd = get_conv(diff, conv_weight, conv_bias, fc_weight, fc_bias)
    
    fs = fa - fs
    fd = fa - fd
    fs = fs * fs
    fd = fd * fd
    fs = mx.sym.sum(fs, axis = 1, keepdims = 1)
    fd = mx.sym.sum(fd, axis = 1, keepdims = 1)
    loss = fd - fs
    loss = one - loss
    loss = mx.sym.Activation(data = loss, act_type = 'relu')
    return mx.sym.MakeLoss(loss)

class Auc(mx.metric.EvalMetric):
    def __init__(self):
        super(Auc, self).__init__('auc')

    def update(self, labels, preds):
        pred = preds[0].asnumpy().reshape(-1)
        self.sum_metric += np.sum(pred)
        self.num_inst += len(pred)

if __name__ == '__main__':
    batch_size = 128
    network = get_net(batch_size)
    devs = [mx.gpu(2)]
    model = mx.model.FeedForward(ctx = devs,
                                 symbol = network,
                                 num_epoch = 100,
                                 learning_rate = 0.01,
                                 wd = 0.00001,
                                 initializer = mx.init.Xavier(factor_type="in", magnitude=2.34),
                                 momentum = 0.0)
    names = []
    root = sys.argv[1]
    for fn in os.listdir(root):
        if fn.endswith('.npy'):
            names.append(root + '/' + fn)
    print len(names)
    data_train = DataIter(names, batch_size)
    
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    
    metric = Auc()
    model.fit(X = data_train,
              eval_metric = metric, 
              kvstore = 'local_allreduce_device',
              batch_end_callback=mx.callback.Speedometer(batch_size, 50),)
    
    model.save(sys.argv[2])
