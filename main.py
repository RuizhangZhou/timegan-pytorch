# -*- coding: UTF-8 -*-
# Local modules
import argparse
import logging
import os
import pickle
import random
import shutil
import time

# 3rd-Party Modules
import numpy as np
import torch
import joblib
from sklearn.model_selection import train_test_split

# Self-Written Modules
from data.data_preprocess import data_preprocess
from metrics.metric_utils import (
    feature_prediction, one_step_ahead_prediction, reidentify_score
)

from models.timegan import TimeGAN
from models.utils import timegan_trainer, timegan_generator, rescale

def main(args):
    ##############################################
    # Initialize output directories
    ##############################################

    ## Runtime directory
    code_dir = os.path.abspath(".")
    if not os.path.exists(code_dir):
        raise ValueError(f"Code directory not found at {code_dir}.")

    ## Data directory
    data_path = os.path.abspath("./data")
    if not os.path.exists(data_path):
        raise ValueError(f"Data file not found at {data_path}.")
    data_dir = os.path.dirname(data_path)
    data_file_name = os.path.basename(data_path)

    ## Output directories
    args.model_path = os.path.abspath(f"./output/{args.exp}/")
    out_dir = os.path.abspath(args.model_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    # TensorBoard directory
    tensorboard_path = os.path.abspath("./tensorboard")
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path, exist_ok=True)

    print(f"\nCode directory:\t\t\t{code_dir}")
    print(f"Data directory:\t\t\t{data_path}")
    print(f"Output directory:\t\t{out_dir}")
    print(f"TensorBoard directory:\t\t{tensorboard_path}\n")

    ##############################################
    # Initialize random seed and CUDA
    ##############################################

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda" and torch.cuda.is_available():
        print("Using CUDA\n")
        # args.device = torch.device("cuda:2")
        # torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("Using CPU\n")
        args.device = torch.device("cpu")

    #########################
    # Load and preprocess data for model
    #########################

    data_path = "/DATA1/rzhou/ika/single_testcases/rounD/rounD_single_09-23_seq250.csv"
    X, T, params_rescale, args.max_seq_len, args.padding_value = data_preprocess(
        file_name=data_path, max_seq_len=args.max_seq_len,scaling_method=args.scaling_method
    )
    #padding_value: float=-10.0,
    #impute_method: str="mode", 
    #scaling_method: str="minmax",
    #这边args.max_seq_len好像没必要再输出一遍，data_preprocess中args.max_seq_len并没有什么变化

    print(f"Processed data: {X.shape} (Idx x MaxSeqLen x Features)\n")
    print(f"Original data preview:\n{X[:4, :20, :10]}\n")

    args.feature_dim = X.shape[-1]
    args.Z_dim = X.shape[-1]
    #这两个是同一个东西？Z是Z-score？

    # Train-Test Split data and time
    train_data, test_data, train_time, test_time = train_test_split(
        X, T, test_size=args.train_rate, random_state=args.seed
    )
    # train_rate=0.5(defaulted)
    
    #########################
    # Initialize and Run model
    #########################

    # Log start time
    start = time.time()

    model = TimeGAN(args)
    #要改,是否从某个checkpoint继续训练
    # model_path = "/home/rzhou/Projects/timegan-pytorch/output/rounD_multi_09-23_seq250_numfea10_Epoch5000/min_G_loss_model_epoch_5000.pt"
    # model.load_state_dict(torch.load(model_path))
    
    if args.is_train == True:
        timegan_trainer(model, train_data, train_time, args)
    generated_data = timegan_generator(model, train_time, args)
    rescaled_generated_data=rescale(generated_data,args.scaling_method,params_rescale)
    generated_time = train_time

    # Log end time
    end = time.time()

    print(f"Generated data preview:\n{generated_data[:2, -10:, ]}\n")
    print(f"Rescaled generated data preview:\n{rescaled_generated_data[:2, -10:, ]}\n")
    print(f"Model Runtime: {(end - start)/60} mins\n")

    #########################
    # Save train and generated data for visualization
    #########################
    
    # Save splitted data and generated data
    with open(f"{args.model_path}/train_data.pickle", "wb") as fb:
        pickle.dump(train_data, fb)
    with open(f"{args.model_path}/train_time.pickle", "wb") as fb:
        pickle.dump(train_time, fb)
    with open(f"{args.model_path}/test_data.pickle", "wb") as fb:
        pickle.dump(test_data, fb)
    with open(f"{args.model_path}/test_time.pickle", "wb") as fb:
        pickle.dump(test_time, fb)
    with open(f"{args.model_path}/fake_data.pickle", "wb") as fb:
        pickle.dump(generated_data, fb)
    with open(f"{args.model_path}/rescaled_fake_data.pickle", "wb") as fb:
        pickle.dump(rescaled_generated_data, fb)
    with open(f"{args.model_path}/fake_time.pickle", "wb") as fb:
        pickle.dump(generated_time, fb)

    #########################
    # Preprocess data for seeker
    #########################

    # Define enlarge data and its labels
    enlarge_data = np.concatenate((train_data, test_data), axis=0)
    enlarge_time = np.concatenate((train_time, test_time), axis=0)
    enlarge_data_label = np.concatenate((np.ones([train_data.shape[0], 1]), np.zeros([test_data.shape[0], 1])), axis=0)

    # Mix the order
    idx = np.random.permutation(enlarge_data.shape[0])
    enlarge_data = enlarge_data[idx]
    enlarge_data_label = enlarge_data_label[idx]

    #########################
    # Evaluate the performance
    #########################

    # 1. Feature prediction
    feat_idx = np.random.permutation(train_data.shape[2])[:args.feat_pred_no]
    print("Running feature prediction using original data...")
    ori_feat_pred_perf = feature_prediction(
        (train_data, train_time), 
        (test_data, test_time),
        feat_idx
    )
    print("Running feature prediction using generated data...")
    new_feat_pred_perf = feature_prediction(
        (generated_data, generated_time),
        (test_data, test_time),
        feat_idx
    )

    feat_pred = [ori_feat_pred_perf, new_feat_pred_perf]

    print('Feature prediction results:\n' +
          f'(1) Ori: {str(np.round(ori_feat_pred_perf, 4))}\n' +
          f'(2) New: {str(np.round(new_feat_pred_perf, 4))}\n')
    '''
    这段代码的核心功能是使用其他特征来预测某个特定特征。这在时间序列数据分析中是一个常见的任务，尤其是在特征工程和数据预处理阶段。这个过程分别对原始数据集 (train_data, test_data) 和生成数据集 (generated_data, generated_time) 进行了特征预测任务，最后比较了这两种数据在特征预测任务上的性能。
    在这个上下文中，ori_feat_pred_perf 和 new_feat_pred_perf 分别代表：
    ori_feat_pred_perf: 使用原始数据进行特征预测时的性能指标列表。每个元素代表对一个特定特征预测任务的性能，这里使用的是均方根误差（Root Mean Square Error, RMSE）作为性能指标。这个列表中的值越小，表示原始数据用于预测该特征的性能越好。
    new_feat_pred_perf: 使用生成数据进行特征预测时的性能指标列表。同样，列表中的每个元素表示使用生成数据对一个特定特征进行预测的RMSE。这里的值同样是越小越好，表示生成数据用于预测该特征的准确性越高。
    最后的输出部分打印了这两个列表的内容，并通过np.round函数四舍五入到四位小数，以便于比较和阅读。输出的目的是让你可以直观地看到使用原始数据和生成数据对特征进行预测时的性能差异，这对于评估生成模型的质量和实用性非常有帮助。如果生成数据的预测性能接近原始数据，这可能表明生成模型能够捕捉到原始数据的关键统计特性，因此生成的数据在某种程度上是“真实的”或至少是有用的。'''

    # 2. One step ahead prediction
    print("Running one step ahead prediction using original data...")
    ori_step_ahead_pred_perf = one_step_ahead_prediction(
        (train_data, train_time), 
        (test_data, test_time)
    )
    print("Running one step ahead prediction using generated data...")
    new_step_ahead_pred_perf = one_step_ahead_prediction(
        (generated_data, generated_time),
        (test_data, test_time)
    )

    step_ahead_pred = [ori_step_ahead_pred_perf, new_step_ahead_pred_perf]

    print('One step ahead prediction results:\n' +
          f'(1) Ori: {str(np.round(ori_step_ahead_pred_perf, 4))}\n' +
          f'(2) New: {str(np.round(new_step_ahead_pred_perf, 4))}\n')

    print(f"Total Runtime: {(time.time() - start)/60} mins\n")

    return None

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    # Inputs for the main function
    parser = argparse.ArgumentParser()

    # Experiment Arguments
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        default='cuda',
        type=str)
    parser.add_argument(
        '--exp',
        default='test',
        type=str)
    parser.add_argument(
        "--is_train",
        type=str2bool,
        default=True)
    parser.add_argument(
        '--seed',
        default=0,
        type=int)
    parser.add_argument(
        '--feat_pred_no',
        default=2,
        type=int)

    # Data Arguments
    parser.add_argument(
        '--max_seq_len',
        default=250,
        type=int)
    parser.add_argument(
        '--train_rate',
        default=0.5,
        type=float)
    parser.add_argument(
        '--scaling_method',
        choices=['minmax', 'standard'],
        default="minmax",
        type=str)

    # Model Arguments
    parser.add_argument(
        '--emb_epochs',
        default=600,
        type=int)
    parser.add_argument(
        '--sup_epochs',
        default=600,
        type=int)
    parser.add_argument(
        '--gan_epochs',
        default=600,
        type=int)
    parser.add_argument(
        '--batch_size',
        default=128,
        type=int)
    parser.add_argument(
        '--hidden_dim',
        default=20,
        type=int)
    parser.add_argument(
        '--num_layers',
        default=3,
        type=int)
    parser.add_argument(
        '--dis_thresh',
        default=0.15,
        type=float)
    parser.add_argument(
        '--optimizer',
        choices=['adam'],
        default='adam',
        type=str)
    parser.add_argument(
        '--learning_rate',
        default=1e-3,
        type=float)

    args = parser.parse_args()

    # Call main function
    main(args)
