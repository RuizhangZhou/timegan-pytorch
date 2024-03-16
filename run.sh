#!/bin/bash
train=true
export TZ="GMT-8"

# Experiment variables
exp="inD_multi_Epoch5000_withoutZfilter_min_G_Loss_period1000epoch"

# Iteration variables
emb_epochs=5000
sup_epochs=5000
gan_epochs=5000

CUDA_VISIBLE_DEVICES=1,2,0 nohup python main.py \
--device            cuda \
--exp               $exp \
--is_train          $train \
--seed              42 \
--feat_pred_no      1 \
--max_seq_len       100 \
--train_rate        0.5 \
--scaling_method    minmax \
--emb_epochs        $emb_epochs \
--sup_epochs        $sup_epochs \
--gan_epochs        $gan_epochs \
--batch_size        128 \
--hidden_dim        20 \
--num_layers        3 \
--dis_thresh        0.15 \
--optimizer         adam \
--learning_rate     1e-3 \
>> /home/rzhou/Projects/timegan-pytorch/log/inD_multi_Epoch5000_withoutZfilter_min_G_Loss_period1000epoch.log 2>&1 &