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

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def wrn(train_lmdb, test_lmdb, batch_size=16, stages=[32, 64, 128, 160, 320, 640]):
    # there will only be a TEST phase data layer, TRAIN needsto be added manually  
    data, label = L.Data(source=train_lmdb, backend=P.Data.LMDB, batch_size=batch_size, ntop=2, transform_param=dict(crop_size=119, mean_value=[104, 117, 123], mirror=True), include=dict(phase=getattr(caffe_pb2, 'TRAIN')))
    data, label = L.Data(source=test_lmdb, backend=P.Data.LMDB, batch_size=batch_size, ntop=2, transform_param=dict(crop_size=119, mean_value=[104, 117, 123], mirror=True), include=dict(phase=getattr(caffe_pb2, 'TEST')))

    #stem
    conv1 = L.Convolution(data, kernel_size=3, stride=2, num_output=stages[0], pad=0, weight_filler=dict(type='msra')) # 151 x 151 -> 75 x 75, 119 x 119 -> 59 x 59 - 32
    bn_relu_1 = batchnorm_scale_relu(conv1)
    
    conv2 = L.Convolution(bn_relu_1, kernel_size=3, stride=1, num_output=stages[0], pad=1, weight_filler=dict(type='msra')) # 75 x 75 -> 75 x 75, 59 x 59 -> 59 x 59 - 32
    bn_relu_2 = batchnorm_scale_relu(conv2)
    
    conv3 = L.Convolution(bn_relu_2, kernel_size=3, stride=1, num_output=stages[1], pad=1, weight_filler=dict(type='msra')) # 75 x 75 -> 75 x 75, 59 x 59 -> 59 x 59 - 64
    bn_relu_3 = batchnorm_scale_relu(conv3)
    
    max_pool_1 = max_pool(bn_relu_3, ks = 3, stride=2) # 75 x 75 -> 37 x 37, 59 x 59 -> 29 x 29 - 64
    
    conv4 = L.Convolution(max_pool_1, kernel_size=1, stride=1, num_output=stages[2], pad=0, weight_filler=dict(type='msra')) # 37 x 37 -> 37 x 37, 29 x 29 -> 29 x 29 - 128
    bn_relu_4 = batchnorm_scale_relu(conv4)

    conv5 = L.Convolution(bn_relu_4, kernel_size=3, stride=1, num_output=stages[2], pad=1, weight_filler=dict(type='msra')) #37 x 37 -> 37 x 37, 29 x 29 -> 29 x 29 - 128 

    #WRN
    bn_relu_2 = batchnorm_scale_relu(conv5)
    stage1_expansion = wrn_expansion(bn_relu_2, 3, stages[3],stride_first_conv=1, pad_first_conv=1) # 37 x 37 -> 37 x 37 , 29 x 29 -> 29 x 29 - 160
    stage1_resblock_1 = wrn_no_expansion(stage1_expansion, 3, stages[3])
    stage1_resblock_2 = wrn_no_expansion(stage1_resblock_1, 3, stages[3])
    stage1_resblock_3 = wrn_no_expansion(stage1_resblock_2, 3, stages[3])

    stage2_expansion = wrn_expansion(stage1_resblock_3, 3, stages[4], stride_first_conv=2, pad_first_conv=1) # 37 x 37 -> 19 x 19, 29 x 29 -> 15 x 15 - 320
    stage2_resblock_1 = wrn_no_expansion(stage2_expansion, 3, stages[4])
    stage2_resblock_2 = wrn_no_expansion(stage2_resblock_1, 3, stages[4])
    stage2_resblock_3 = wrn_no_expansion(stage2_resblock_2, 3, stages[4])
    
    stage3_expansion = wrn_expansion(stage2_resblock_3, 3, stages[5], stride_first_conv=2, pad_first_conv=1) # 19 x 19 -> 10 x 10, 15 x 15 -> 8 x 8 - 640
    stage3_resblock_1 = wrn_no_expansion(stage3_expansion, 3, stages[5])
    stage3_resblock_2 = wrn_no_expansion(stage3_resblock_1, 3, stages[5])
    stage3_resblock_3 = wrn_no_expansion(stage3_resblock_2, 3, stages[5])

    #glb_pool = L.Pooling(stage5_resblock_2, kernel_size=4, stride=4, pool=P.Pooling.AVE, global_pooling=True);
    glb_pool = L.Pooling(stage3_resblock_3, kernel_size=4, stride=4, pool=P.Pooling.AVE); # 10 x 10 -> 2 x 2, 8 x 8 -> 2 x 2
    fc = L.InnerProduct(glb_pool, num_output=1000)
    loss = L.SoftmaxWithLoss(fc, label)
    acc = L.Accuracy(fc, label, include=dict(phase=getattr(caffe_pb2, 'TEST')))
    return to_proto(loss, acc)

def make_net():
    with open('wrn_ilsvrc_119_v8.prototxt', 'w') as f:
        print(wrn('/PATH/TO/train_imagenet_128_lmdb', '/PATH/TO/val_imagenet_128_lmdb'), file=f)

if __name__ == '__main__':
    make_net()
    caffe.Net('wrn_ilsvrc_11_v8.prototxt', caffe.TEST)  # test loading the net
