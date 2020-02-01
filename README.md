# TSPSelector
initial commit
Python 3.7 version 

MPNN/ -- model and layers of Message Passing Neural Network

main.sh --running shell for automl

run.sh --running shell for python command line

para.yml --parameters configuration for automl

train.py --program entrance, training and validation

*Model Parameters

--Data argumentation

num_rotate (int): # of coordinates rotation in [0, 2*pi]

num_grid (int): (grid * grid) image for put coordinates

scale_factor (int): reduce the image resolution by scale_factor

flip (bool) : whether to flip image

--Model

model_type (str): 'alexnet', 'resnet18', ...

--Training

epoches (int) 

learing_rate (float)

batch_size (int)

weight_decay (float): L2 regularization of Adam

decay_factor (float): learning rate decay factor (exp)

decay_patience (int): # epoches for one lr decay

--Other 

num_workers (int): # of workers for Dataset