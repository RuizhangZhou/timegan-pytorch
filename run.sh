#!/bin/bash
train=true
export TZ="GMT-8"

# Experiment variables
exp="rounD_09-23_seq250_nfea2_Epoch10000"

# Iteration variables
emb_epochs=10000
sup_epochs=10000
gan_epochs=10000

CUDA_VISIBLE_DEVICES=3,2,1,0 nohup python main.py \
--device            cuda \
--exp               $exp \
--is_train          $train \
--seed              42 \
--feat_pred_no      1 \
--max_seq_len       250 \
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
>> /home/rzhou/Projects/timegan-pytorch/log/rounD_09-23_seq250_nfea2_Epoch10000.log 2>&1 &