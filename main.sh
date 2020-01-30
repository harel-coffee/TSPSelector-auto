#!/bin/bash

python train.py \
  --epoches ##epoches## \
  --batch_size ##batch_size## \
  --learning_rate ##learning_rate## \
  --weight_decay ##weight_decay## \
  --decay_factor ##decay_factor## \
  --model_type ##model_type## \
  --verbose False \
  --instances_path "/home/kfzhao/data/ECJ_instances"