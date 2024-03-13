#!/bin/bash
train=true
export TZ="GMT-8"

# Experiment variables
exp="inD_multi_test"

# Iteration variables
emb_epochs=20
sup_epochs=20
gan_epochs=20

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python main.py \
--device            cuda \
--exp               $exp \
--is_train          $train \
--seed              42 \
--feat_pred_no      1 \
--max_seq_len       100 \
--train_rate        0.5 \
--emb_epochs        $emb_epochs \
--sup_epochs        $sup_epochs \
--gan_epochs        $gan_epochs \
--batch_size        128 \
--hidden_dim        20 \
--num_layers        3 \
--dis_thresh        0.15 \
--optimizer         adam \
--learning_rate     1e-3 \
>> /home/rzhou/Projects/scenariogenerationai/timegan-pytorch/log/inD_multi-test.log 2>&1 &