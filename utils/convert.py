import sys
import os
import caffe.proto.caffe_pb2 as pb2
import numpy as np
import h5py

# mainly taken from:
# https://github.com/chuckcho/c3d-keras/blob/master/convert_caffe_model.py

file_in = sys.argv[1] 
file_out = os.path.abspath(os.path.join(file_in, os.pardir))
file_out = os.path.join(file_out, 'C3D_weights.h5')

print('converting {}'.format(file_in))
print('to {}'.format(file_out))

p = pb2.NetParameter()
p.ParseFromString(
    open(file_in, 'rb').read()
)

# use this function to convert the first fully connected layer
def conv_first_fc(w):
    # kernel: (8192, 4096): (512x1x4x4, 4096) -> (1x4x4x512, 4096)
    wo = np.zeros_like(w)
    for i in range(w.shape[1]):
        wi = np.squeeze(w[:,i])
        wo[:,i] = np.transpose(np.reshape(wi, (512,4,4)), (1, 2, 0)).flatten()
    return wo

with h5py.File(file_out, 'w') as ff: 
    for layer in p.layers:
        name = layer.name

        if 'conv' not in name and 'fc' not in name:
            continue

        print('converting layer:')
        print(name)

        # somehow blobs[0].length is deprecated therefore the height
        # has to be computed separate for the fc layers
        # in the conv layers the kernel size is always [3, 3, 3] 
        num = layer.blobs[0].num
        ch = layer.blobs[0].channels
        width = layer.blobs[0].width

        bias = np.array(layer.blobs[1].diff, dtype=np.float32)
        kernel = np.array(layer.blobs[0].diff, dtype=np.float32)

        if 'conv' in name:
            shape = [num, ch, 3, 3, 3]
            kernel = kernel.reshape(shape)
            kernel = np.transpose(kernel,  (2, 3, 4, 1, 0))
        else:
            height = kernel.shape[0] // width
            shape = [width, height]
            kernel = kernel.reshape(shape).T
            if 'fc6' in name:
                kernel = conv_first_fc(kernel)

        layer_grp = ff.create_group(name)
        layer_grp.create_dataset('bias', data=bias) 
        layer_grp.create_dataset('kernel', data=kernel) 
