# wide_residual_nets_caffe
Training wide residual nets in caffe

Here I try training wide residual nets with 28 convolutional layers and widening factor 10 (wrn_28_10) first published in 
http://arxiv.org/abs/1605.07146 on the CIFAR 10 dataset using the Caffe framework.

CIFAR10 has been taken directly from Caffe, i.e. the lmdbs that are provided with the distro. Different from the paper I do not perform ZCA-whitening
nor global contrast normalization. Also I am not padding images with 4 pixels on each side and then taking random crops. Lastly I am not 
using ReflectionPadding but simple ZeroPadding (but see https://twitter.com/karpathy/status/720622989289644033).

The train_val can be found here: https://gist.github.com/revilokeb/471b9358617822dc10f89ccf6f40b088 and 
the solver here: https://gist.github.com/revilokeb/1029518fc55c8a254b4f24dccba74487

My lowest (top 1) validation error is 7.46% which is pretty bad when compared with the 4.17% in http://arxiv.org/abs/1605.07146,
but given my rather modest data preprocessing it could be ok.
![Alt text](./wrn_cifar10_nesterov.png?raw=true "Current Validation Error / Training Loss")
