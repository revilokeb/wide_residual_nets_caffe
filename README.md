# wide_residual_nets_caffe
Training wide residual nets in caffe

###CIFAR 10###

Here I try training wide residual nets with 28 convolutional layers and widening factor 10 (wrn_28_10) first published in 
http://arxiv.org/abs/1605.07146 on the CIFAR 10 dataset using the Caffe framework.

CIFAR10 has been taken directly from Caffe, i.e. the lmdbs that are provided with the distro. Different from the paper I do not perform ZCA-whitening
nor global contrast normalization. Also I am not padding images with 4 pixels on each side and then taking random crops. Lastly I am not 
using ReflectionPadding but simple ZeroPadding (but see https://twitter.com/karpathy/status/720622989289644033).

The train_val of the first experiment on wrn_28_10 without dropout using solver Nesterov as in the paper can be found here: https://gist.github.com/revilokeb/471b9358617822dc10f89ccf6f40b088 and 
the solver here: https://gist.github.com/revilokeb/1029518fc55c8a254b4f24dccba74487

My lowest (top 1) validation error is 7.46% which is pretty bad when compared with the 4.17% in http://arxiv.org/abs/1605.07146,
but given my rather modest data preprocessing it could be ok. My last snapshot can be found here: https://drive.google.com/open?id=0B1qLpHDbczM2SlByaHk2V3FkQkk
<p align="center">
<img src="https://github.com/revilokeb/wide_residual_nets_caffe/blob/master/images/wrn_cifar10_nesterov.png" | width="350">
</p>

When using solver RMSProp and a learning rate schedule similar to http://arxiv.org/abs/1602.07261 I am learning faster but the final accuracy is much worse at aroung 13.5% validation error.
<p align="center">
<img src="https://github.com/revilokeb/wide_residual_nets_caffe/blob/master/images/wrn_cifar10_rmsprop.png" | width="350">
</p>

When I include dropout between convolutional layers I get a very similar learning and accuracy as without dropout, below when using Nesterov (train_val found here: https://gist.github.com/revilokeb/77cce8c373d4d9687d1ecf6cc7bef166; RMSProp again leads to significantly worse final accuracy): 
<p align="center">
<img src="https://github.com/revilokeb/wide_residual_nets_caffe/blob/master/images/wrn_cifar10_dropout_nesterov.png" | width="350">
</p>

###Imagenet 1k###
Currently I am exploring Wide Residual Nets on the full Imagenet 1k set resized to 128x128, or more precisely I have resized Imagenet 1k to 1. 128x128 and 2. 156x156 and typically crop 119x119 from the first dataset or 151x151 from the second dataset. My current approach for tackling Imagenet 1k with architectures similar to https://github.com/szagoruyko/wide-residual-networks is to combine e.g. WRN-28-10 sitting on the stem of an Inception-Resnet-V1 of http://arxiv.org/abs/1602.07261 (p.6), i.e. the WRN-part replacing the inception part of Inception-Resnet-V1 (stem of the Inception-Resnet-V2 could be an alternative). Reason I am running Imagenet resized to relatively small sizes is entirely due to the fact that computations dont take as long as on larger sizes such as 256 x 256 or 299 x 299 etc and demand less GPU ressources (beforementioned approach could be as easily applied to larger image sizes). 

The most simple architecture I could think of to run a WRN on Imagenet 1k was to use two successive 3x3 convolutional layers to scale down the 151 x 151 image crops and then employ the best performing WRN on top (from CIFAR 10, WRN-28-10): see wrn_ilsvrc_151_v7.prototxt / wrn_ilsvrc_151_v7.py. Unfortunately I needed to interrupt training after 14 epochs, which I might resume when I have some spare time / computing ressources available again for that task. 
<p align="center">
<img src="https://github.com/revilokeb/wide_residual_nets_caffe/blob/master/images/wrn_ilsvrc_151_14epochs_v7.png" | width="350">
</p>

The above very simple achitecture could be set up in a somewhat more elaborate manner by e.g. taking the stem of the Inception-ResNet-V1 for scaling down. An example architecture wrn_ilsvrc_119_v8.prototxt is included in the repository (which I am currently running between some other tasks, I will post results when available). 


