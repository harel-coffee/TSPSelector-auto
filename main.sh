#!/bin/bash
unset http_proxy
unset https_proxy
mkdir -p /opt/ml/env/out
mkdir -p /opt/ml/disk/out
cp /opt/ml/disk/ECJ_instances_coo.zip .
unzip ECJ_instances_coo.zip
ls .
pwd .
echo START
CUDA_VISIBLE_DEVICES=0 sh automl_run.sh
#mv /opt/ml/env/model_dir/ /opt/ml/model/
#rm -r /opt/ml/env/outOAE
echo END