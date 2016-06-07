# wide_residual_nets_caffe
Training wide residual nets in caffe

Here I try training wide residual nets with 28 convolutional layers and widening factor 10 (wrn_28_10) first published in 
http://arxiv.org/abs/1605.07146 on the CIFAR 10 dataset using the Caffe framework.

CIFAR10 has been taken directly from Caffe, i.e. the lmdbs that are provided with the distro. Different from the paper I do not perform ZCA-whitening
nor global contrast normalization. Also I am not padding images with 4 pixels on each side and then taking random crops. Lastly I am not 
using ReflectionPadding but simple ZeroPadding (but see https://twitter.com/karpathy/status/720622989289644033).

The train_val of the first experiment on wrn_28_10 without dropout using solver Nesterov as in the paper can be found here: https://gist.github.com/revilokeb/471b9358617822dc10f89ccf6f40b088 and 
the solver here: https://gist.github.com/revilokeb/1029518fc55c8a254b4f24dccba74487

My lowest (top 1) validation error is 7.46% which is pretty bad when compared with the 4.17% in http://arxiv.org/abs/1605.07146,
but given my rather modest data preprocessing it could be ok. My last snapshot can be found here: https://drive.google.com/open?id=0B1qLpHDbczM2SlByaHk2V3FkQkk
![Alt text](./wrn_cifar10_nesterov.png?raw=true "Current Validation Error / Training Loss")

When using solver RMSProp and a learning rate schedule similar to http://arxiv.org/abs/1602.07261 I am learning faster but the final accuracy is much worse at aroung 13.5% validation error.
![Alt text](./wrn_cifar10_rmsprop.png?raw=true "Current Validation Error / Training Loss")

Currently I am running the experiments with dropout between conv layers.
