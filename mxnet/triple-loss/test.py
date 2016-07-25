import sys, random, os
sys.path.insert(0, '../../python')
import mxnet as mx
import numpy as np
import imgnet_triloss, cv2
from operator import itemgetter

_, arg_params, __ = mx.model.load_checkpoint(sys.argv[2], 100)

batch_size = 1
network = imgnet_triloss.get_sim_net()

input_shapes = dict([('same', (batch_size, 3, 32, 32)),\
                     ('diff', (batch_size, 3, 32, 32))])
executor = network.simple_bind(ctx = mx.gpu(), **input_shapes)
for key in executor.arg_dict.keys():
    if key in arg_params:
        print key, arg_params[key].shape, executor.arg_dict[key].shape
        arg_params[key].copyto(executor.arg_dict[key])

root = sys.argv[1]
names = []
for fn in os.listdir(root):
    if fn.endswith('.npy'):
        names.append(root + '/' + fn)
random.shuffle(names)

imgs = []
for i in range(10):
    imgs.append(np.load(names[i]))

def save_img(fname, im):
    a = np.copy(im) * 255.0
    cv2.imwrite(fname, a.transpose(1, 2, 0))

src = imgs[0][random.randint(0, len(imgs[0]) - 1)]
save_img("src.png", src)
dsts = []
for i in range(10):
    for j in range(128):
        k = random.randint(0, len(imgs[i]) - 1)
        dst = imgs[i][k]
        outputs = executor.forward(is_train = True, same = mx.nd.array([src]),
                         diff = mx.nd.array([dst]))
        dis = outputs[0].asnumpy()[0]
        dsts.append((dst, dis, i))
        
i = 0
for img, w, la in sorted(dsts, key = itemgetter(1))[:10]:
    print w, la
    save_img("dst_" + str(i) + ".png", img)
    i += 1

