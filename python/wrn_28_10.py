from __future__ import print_function
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import caffe

def batchnorm_scale_relu(bottom):
    batch_norm = L.BatchNorm(bottom, use_global_stats=False, in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    relu = L.ReLU(scale, in_place=True)
    return relu
    
def wrn_expansion(bottom, ks, nout, stride_first_conv=1, pad_first_conv=1):
    conv_1_1 = L.Convolution(bottom, kernel_size=ks, stride=stride_first_conv,
                                num_output=nout, pad=pad_first_conv, bias_term=False, weight_filler=dict(type='msra'))
    batch_norm = L.BatchNorm(conv_1_1, use_global_stats=False, in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    relu = L.ReLU(scale, in_place=True)
    conv_2_1 = L.Convolution(relu, kernel_size=ks, stride=1,
                                num_output=nout, pad=1, bias_term=False, weight_filler=dict(type='msra'))
    conv_1_2 = L.Convolution(bottom, kernel_size=1, stride=stride_first_conv,
                                num_output=nout, pad=0, bias_term=False, weight_filler=dict(type='msra'))
    addition = L.Eltwise(conv_2_1, conv_1_2, operation=P.Eltwise.SUM)
    return addition

def wrn_no_expansion(bottom, ks, nout):
    batch_norm = L.BatchNorm(bottom, use_global_stats=False, in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    relu = L.ReLU(scale, in_place=True)
    conv_1_1 = L.Convolution(relu, kernel_size=ks, stride=1,
                                num_output=nout, pad=1, bias_term=False, weight_filler=dict(type='msra'))
    batch_norm2 = L.BatchNorm(conv_1_1, use_global_stats=False, in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    scale2 = L.Scale(batch_norm2, bias_term=True, in_place=True)
    relu2 = L.ReLU(scale2, in_place=True)
    conv_2_1 = L.Convolution(relu2, kernel_size=ks, stride=1,
                                num_output=nout, pad=1, bias_term=False, weight_filler=dict(type='msra'))
    addition = L.Eltwise(conv_2_1, bottom, operation=P.Eltwise.SUM)
    return addition

def wrn(train_lmdb, test_lmdb, batch_size=32, stages=[16, 160, 320, 640], input_size=32, first_output=32, include_acc=False):
    #code can't recognize include phase at the moment thus there is only be a TEST phase data layer, TRAIN layer needs to be added manually
    data, label = L.Data(source=train_lmdb, backend=P.Data.LMDB, batch_size=batch_size, ntop=2, transform_param=dict(crop_size=32, mean_value=[104, 117, 123], mirror=True), include=dict(phase=getattr(caffe_pb2, 'TRAIN')))
    data, label = L.Data(source=test_lmdb, backend=P.Data.LMDB, batch_size=batch_size, ntop=2, transform_param=dict(crop_size=32, mean_value=[104, 117, 123], mirror=True), include=dict(phase=getattr(caffe_pb2, 'TEST')))

    conv1 = L.Convolution(data, kernel_size=3, stride=1, num_output=stages[0], pad=1, weight_filler=dict(type='msra'))

    bn_relu_1 = batchnorm_scale_relu(conv1)
    stage1_expansion = wrn_expansion(bn_relu_1, 3, stages[1], stride_first_conv=1, pad_first_conv=1)
    stage1_resblock_1 = wrn_no_expansion(stage1_expansion, 3, stages[1])
    stage1_resblock_2 = wrn_no_expansion(stage1_resblock_1, 3, stages[1])
    stage1_resblock_3 = wrn_no_expansion(stage1_resblock_2, 3, stages[1])

    stage2_expansion = wrn_expansion(stage1_resblock_3, 3, stages[2], stride_first_conv=2, pad_first_conv=1)
    stage2_resblock_1 = wrn_no_expansion(stage2_expansion, 3, stages[2])
    stage2_resblock_2 = wrn_no_expansion(stage2_resblock_1, 3, stages[2])
    stage2_resblock_3 = wrn_no_expansion(stage2_resblock_2, 3, stages[2])
    
    stage3_expansion = wrn_expansion(stage2_resblock_3, 3, stages[3], stride_first_conv=2, pad_first_conv=1)
    stage3_resblock_1 = wrn_no_expansion(stage3_expansion, 3, stages[3])
    stage3_resblock_2 = wrn_no_expansion(stage3_resblock_1, 3, stages[3])
    stage3_resblock_3 = wrn_no_expansion(stage3_resblock_2, 3, stages[3])

    glb_pool = L.Pooling(stage3_resblock_3, pool=P.Pooling.AVE, global_pooling=True);
    fc = L.InnerProduct(glb_pool, num_output=10)
    loss = L.SoftmaxWithLoss(fc, label)
    acc = L.Accuracy(fc, label, include=dict(phase=getattr(caffe_pb2, 'TEST')))
    return to_proto(loss, acc)

def make_net():
    with open('wrn_28_10_cifar10.prototxt', 'w') as f:
        print(wrn('MY_PATH/cifar10_train_lmdb', 'MY_PATH/cifar10_test_lmdb'), file=f)

if __name__ == '__main__':
    make_net()
    caffe.Net('wrn_28_10_cifar10.prototxt', caffe.TEST)  # test loading the net