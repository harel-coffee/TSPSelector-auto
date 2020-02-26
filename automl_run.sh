#!/bin/bash
PYTHON=/root/anaconda3/bin/python

$PYTHON train.py \
  --epoches ##epoches## \
  --batch_size ##batch_size## \
  --learning_rate ##learning_rate## \
  --weight_decay ##weight_decay## \
  --decay_factor ##decay_factor## \
  --decay_patience ##decay_patience## \
  --weight_exp ##weight_exp## \
  --model_type ##model_type## \
  --num_grid ##num_grid## \
  --num_rotate ##num_rotate## \
  --scale_factor ##scale_factor## \
  --flip ##flip## \
  --num_workers 16 \
  --num_fold 5 \
  --instances_path "ECJ_instances_coo"