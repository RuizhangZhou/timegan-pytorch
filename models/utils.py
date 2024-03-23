# -*- coding: UTF-8 -*-
# Local modules
import os
import pickle
from typing import Dict, Union

# 3rd party modules
import numpy as np
from tqdm import tqdm, trange

import torch
from torch.utils.tensorboard import SummaryWriter

# Self-written modules
from models.dataset import TimeGANDataset

def embedding_trainer(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    e_opt: torch.optim.Optimizer, 
    r_opt: torch.optim.Optimizer, 
    args: Dict, 
    writer: Union[torch.utils.tensorboard.SummaryWriter, type(None)]=None
) -> None:
    """The training loop for the embedding and recovery functions
    """  
    logger = trange(args.emb_epochs, desc=f"Epoch: 0, Loss: 0")
    for epoch in logger:   
        for X_mb, T_mb in dataloader:
            # Reset gradients
            model.zero_grad()

            # Forward Pass
            # time = [args.max_seq_len for _ in range(len(T_mb))]
            _, E_loss0, E_loss_T0 = model(X=X_mb, T=T_mb, Z=None, obj="autoencoder")
            loss = np.sqrt(E_loss_T0.item())

            # Backward Pass
            E_loss0.backward()

            # Update model parameters
            e_opt.step()
            r_opt.step()

        # Log loss for final batch of each epoch (29 iters)
        logger.set_description(f"Epoch: {epoch}, Loss: {loss:.4f}")
        if writer:
            writer.add_scalar(
                "Embedding/Loss:", 
                loss, 
                epoch
            )
            writer.flush()

def supervisor_trainer(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    s_opt: torch.optim.Optimizer, 
    g_opt: torch.optim.Optimizer, 
    args: Dict, 
    writer: Union[torch.utils.tensorboard.SummaryWriter, type(None)]=None
) -> None:
    """The training loop for the supervisor function
    """
    logger = trange(args.sup_epochs, desc=f"Epoch: 0, Loss: 0")
    for epoch in logger:
        for X_mb, T_mb in dataloader:
            # Reset gradients
            model.zero_grad()

            # Forward Pass
            S_loss = model(X=X_mb, T=T_mb, Z=None, obj="supervisor")

            # Backward Pass
            S_loss.backward()
            loss = np.sqrt(S_loss.item())

            # Update model parameters
            s_opt.step()

        # Log loss for final batch of each epoch (29 iters)
        logger.set_description(f"Epoch: {epoch}, Loss: {loss:.4f}")
        if writer:
            writer.add_scalar(
                "Supervisor/Loss:", 
                loss, 
                epoch
            )
            writer.flush()

def joint_trainer(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    e_opt: torch.optim.Optimizer, 
    r_opt: torch.optim.Optimizer, 
    s_opt: torch.optim.Optimizer, 
    g_opt: torch.optim.Optimizer, 
    d_opt: torch.optim.Optimizer, 
    args: Dict, 
    writer: Union[torch.utils.tensorboard.SummaryWriter, type(None)]=None, 
) -> None:
    """The training loop for training the model altogether
    """
    logger = trange(
        args.sup_epochs, 
        desc=f"Epoch: 0, E_loss: 0, G_loss: 0, D_loss: 0"
    )
    
    # 初始化最小G_loss记录变量
    min_G_loss_periodic = np.inf
    min_G_loss_periodic_model = None
    min_G_loss_periodic_epoch = 0
    periodic_epoch_interval = 1000  # 每1000个epoch保存一次
    min_G_losses_periodic = []  # 记录每1000个epoch周期内的最小G_loss及其epoch
    
    min_G_loss = np.inf
    min_G_loss_model=model
    min_G_loss_epoch=0
    for epoch in logger:
        for X_mb, T_mb in dataloader:
            ## Generator Training
            for _ in range(2):
                # Random Generator
                Z_mb = torch.rand((args.batch_size, args.max_seq_len, args.Z_dim))

                # Forward Pass (Generator)
                model.zero_grad()
                G_loss = model(X=X_mb, T=T_mb, Z=Z_mb, obj="generator")
                G_loss.backward()
                G_loss = np.sqrt(G_loss.item())

                # Update model parameters
                g_opt.step()
                s_opt.step()

                # Forward Pass (Embedding)
                model.zero_grad()
                E_loss, _, E_loss_T0 = model(X=X_mb, T=T_mb, Z=Z_mb, obj="autoencoder")
                E_loss.backward()
                E_loss = np.sqrt(E_loss.item())
                
                # Update model parameters
                e_opt.step()
                r_opt.step()

            # Random Generator
            Z_mb = torch.rand((args.batch_size, args.max_seq_len, args.Z_dim))

            ## Discriminator Training
            model.zero_grad()
            # Forward Pass
            D_loss = model(X=X_mb, T=T_mb, Z=Z_mb, obj="discriminator")

            # Check Discriminator loss
            if D_loss > args.dis_thresh:
                # Backward Pass
                D_loss.backward()

                # Update model parameters
                d_opt.step()
            D_loss = D_loss.item()
            
        if G_loss < min_G_loss_periodic:
            min_G_loss_periodic = G_loss
            min_G_loss_periodic_model = model.state_dict()
            min_G_loss_periodic_epoch = epoch

        # 每1000个epoch结束时执行操作
        if (epoch + 1) % periodic_epoch_interval == 0:
            # 保存周期内最小G_loss的模型状态
            torch.save(min_G_loss_periodic_model, f"{args.model_path}/min_G_loss_model_epoch_{epoch+1}.pt")
            print(f"Epoch {min_G_loss_periodic_epoch + 1}: Periodic minimum G_loss {min_G_loss_periodic:.4f} - Model saved")

            # 记录周期内最小G_loss及其epoch
            min_G_losses_periodic.append((min_G_loss_periodic, min_G_loss_periodic_epoch + 1))

            # 重置周期内最小G_loss的跟踪变量
            min_G_loss_periodic = np.inf
            min_G_loss_periodic_model = None
            min_G_loss_periodic_epoch = 0
            
        if G_loss < min_G_loss:
            min_G_loss = G_loss  # 更新最小G_loss
            # 保存模型
            min_G_loss_model=model
            min_G_loss_epoch=epoch
        
        logger.set_description(
            f"Epoch: {epoch}, E: {E_loss:.4f}, G: {G_loss:.4f}, D: {D_loss:.4f}"
        )
        if writer:
            writer.add_scalar(
                'Joint/Embedding_Loss:', 
                E_loss, 
                epoch
            )
            writer.add_scalar(
                'Joint/Generator_Loss:', 
                G_loss, 
                epoch
            )
            writer.add_scalar(
                'Joint/Discriminator_Loss:', 
                D_loss, 
                epoch
            )
            writer.flush()

    torch.save(min_G_loss_model.state_dict(), f"{args.model_path}/min_G_loss_model.pt")
    print(f"Epoch {min_G_loss_epoch}: minimum G_loss {min_G_loss:.4f} - min_G_loss_model saved")
    
    
def timegan_trainer(model, data, time, args):
    """The training procedure for TimeGAN
    Args:
        - model (torch.nn.module): The model that generates synthetic data
        - data (numpy.ndarray): The data for training the model
        - time (numpy.ndarray): The time for the model to be conditioned on
        - args (dict): The model/training configurations
    Returns:
        no return
    """

    # Initialize TimeGAN dataset and dataloader
    dataset = TimeGANDataset(data, time)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False    
    )

    model.to(args.device)

    # Initialize Optimizers
    e_opt = torch.optim.Adam(model.embedder.parameters(), lr=args.learning_rate)
    r_opt = torch.optim.Adam(model.recovery.parameters(), lr=args.learning_rate)
    s_opt = torch.optim.Adam(model.supervisor.parameters(), lr=args.learning_rate)
    g_opt = torch.optim.Adam(model.generator.parameters(), lr=args.learning_rate)
    d_opt = torch.optim.Adam(model.discriminator.parameters(), lr=args.learning_rate)
    
    # TensorBoard writer
    writer = SummaryWriter(os.path.join(f"tensorboard/{args.exp}"))

    print("\nStart Embedding Network Training")
    embedding_trainer(
        model=model, 
        dataloader=dataloader, 
        e_opt=e_opt, 
        r_opt=r_opt, 
        args=args, 
        writer=writer
    )

    print("\nStart Training with Supervised Loss Only")
    supervisor_trainer(
        model=model,
        dataloader=dataloader,
        s_opt=s_opt,
        g_opt=g_opt,
        args=args,
        writer=writer
    )

    print("\nStart Joint Training")
    joint_trainer(
        model=model,
        dataloader=dataloader,
        e_opt=e_opt,
        r_opt=r_opt,
        s_opt=s_opt,
        g_opt=g_opt,
        d_opt=d_opt,
        args=args,
        writer=writer,
    )

    # Save model, args, and hyperparameters
    torch.save(args, f"{args.model_path}/args.pickle")
    torch.save(model.state_dict(), f"{args.model_path}/model.pt")
    print(f"\nSaved at path: {args.model_path}")

def timegan_generator(model, T, args):
    """The inference procedure for TimeGAN
    Args:
        - model (torch.nn.module): The model that generates synthetic data
        - T (List[int]): The time to be generated on
        - args (dict): The model/training configurations
    Returns:
        - generated_data (np.ndarray): The synthetic data generated by the model
    """
    # Load model for inference
    if not os.path.exists(args.model_path):
        raise ValueError(f"Model directory not found...")

    # Load arguments and model
    with open(f"{args.model_path}/args.pickle", "rb") as fb:
        args = torch.load(fb)
    
    model.load_state_dict(torch.load(f"{args.model_path}/min_G_loss_model.pt"))
    
    print("\nGenerating Data...")
    # Initialize model to evaluation mode and run without gradients
    model.to(args.device)
    model.eval()
    with torch.no_grad():
        # Generate fake data
        Z = torch.rand((len(T), args.max_seq_len, args.Z_dim))
        
        generated_data = model(X=None, T=T, Z=Z, obj="inference")

    return generated_data.numpy()


def rescale(generated_data,scaling_method,params_rescale):
    """The inference procedure for TimeGAN
    Args:
        - model (torch.nn.module): The model model that generates synthetic data
        - T (List[int]): The time to be generated on
        - args (dict): The model/training configurations
    Returns:
        - generated_data (np.ndarray): The synthetic data generated by the model
    """
    cut_params_rescale=[arr[1:] for arr in params_rescale]
    rescaled_generated_data = np.empty_like(generated_data)
    if scaling_method=="minmax":
        data_min_ = cut_params_rescale[0]  # 这里是最小值数组
        data_max_ = cut_params_rescale[1]  # 这里是最大值数组大值数组
        for feature_idx in range(generated_data.shape[-1]):
            # 对每个特征单独处理
            min_val = data_min_[feature_idx]
            max_val = data_max_[feature_idx]
            # MinMaxScaler 的逆变换公式
            rescaled_generated_data[..., feature_idx] = generated_data[..., feature_idx] * (max_val - min_val) + min_val
    elif scaling_method=="standard":
        mean_ = cut_params_rescale[0]  # 这里是平均值数组
        scale_ = cut_params_rescale[1]  # 这里是标准差数组
        for feature_idx in range(generated_data.shape[-1]):
            # 对每个特征单独处理
            mean_val = mean_[feature_idx]
            scale_val = scale_[feature_idx]
            # StandardScaler 的逆变换公式
            rescaled_generated_data[..., feature_idx] = generated_data[..., feature_idx] * scale_val + mean_val
    return rescaled_generated_data

