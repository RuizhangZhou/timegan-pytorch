import pickle
import torch
import os
import sys

sys.path.append('../')

from models.timegan import TimeGAN
from models.utils import rescale
from data.data_preprocess import data_preprocess

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import random
from PIL import Image


path="/home/rzhou/Projects/timegan-pytorch/output/rounD_multi_09-23_seq250_numfea2_Epoch5000"
with open(f"{path}/args.pickle", "rb") as fb:
    args = torch.load(fb)
with open(f"{path}/fake_time.pickle", "rb") as fb:
    fake_time = pickle.load(fb)

fake_time = torch.tensor(fake_time, dtype=torch.float32)

data_path="/DATA1/rzhou/ika/single_testcases/rounD/rounD_single_09-23_min_seq250.csv"

X, T, params_rescale, args.max_seq_len, args.padding_value = data_preprocess(
    file_name=data_path, max_seq_len=args.max_seq_len,scaling_method=args.scaling_method
)

epoch=4000#要改
model = TimeGAN(args)
model_path = "/home/rzhou/Projects/timegan-pytorch/output/rounD_multi_09-23_seq250_numfea2_Epoch5000/min_G_loss_model_epoch_4000.pt"
model.load_state_dict(torch.load(model_path))
model.to(args.device)
model.eval()
with torch.no_grad():
    # Generate fake data
    Z = torch.rand((len(fake_time), args.max_seq_len, args.Z_dim))
    generated_data = model(X=None, T=fake_time, Z=Z, obj="inference").numpy()

fake_data=rescale(generated_data,args.scaling_method,params_rescale)


seq_length=250#要改
num_feature=2#要改

fake_data[fake_data < -200] = 0

num_cases=5 #画几张图
random_indices = np.random.choice(fake_data.shape[0], num_cases, replace=False)
print(f"Random indices: {random_indices}")


num_v=num_feature//2
Dtype="rounD"#要改
index_map=9#要改
bg_image_path = f'/DATA1/rzhou/ika/{Dtype}/data/{index_map:02d}_background.png'
bg_img = Image.open(bg_image_path)
width, height = bg_img.size
figsize = (width / 100, height / 100)
colors = plt.cm.jet(np.linspace(0, 1, num_v))

# 读取CSV文件
df_recordingMeta = pd.read_csv(f"/DATA1/rzhou/ika/{Dtype}/data/{index_map:02d}_recordingMeta.csv")
# 读取最后一列"orthoPxToMeter"的值
# ortho_px_to_meter = 0.01
ortho_px_to_meter = df_recordingMeta["orthoPxToMeter"].iloc[0]


def init():
        for line in lines:
            line.set_data([], [])
        for path in paths:
            path.set_data([], [])
        return lines + paths

def animate(i):
    for j, (line, path) in enumerate(zip(lines, paths)):
        x = fake_data[random_index][i, j*2]
        y = fake_data[random_index][i, j*2+1]
        if x == 0 and y == 0:
            line.set_data([], [])
        else:
            line.set_data([x], [y])
            px, py = path.get_data()
            path.set_data(np.append(px, x), np.append(py, y))
    return lines + paths

def checkDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


for random_index in random_indices:
    fig, ax = plt.subplots(figsize=figsize)
    bg_img = plt.imread(bg_image_path)
    ax.set_xlim(0, width*ortho_px_to_meter*10)
    ax.set_ylim(-height*ortho_px_to_meter*10, 0)
    ax.imshow(bg_img, extent=[0, width*ortho_px_to_meter*10, -height*ortho_px_to_meter*10, 0])

    lines = [ax.plot([], [], marker='o', linestyle='', color=colors[i])[0] for i in range(num_v)]
    paths = [ax.plot([], [], color=colors[i], linewidth=1)[0] for i in range(num_v)]  # For drawing paths

    anim = FuncAnimation(fig, animate, init_func=init, frames=seq_length, interval=40, blit=True)
    plt.legend([f"Point {i+1}" for i in range(num_v)], loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=25, metadata=dict(artist='Me'), bitrate=1800)
    animPath = f'/home/rzhou/Projects/timegan-pytorch/output/rounD_multi_09-23_seq250_numfea2_Epoch5000/animations/{epoch}/{random_index}.mp4'
    checkDir(os.path.dirname(animPath))
    anim.save(animPath, writer=writer)
    
    # Save the trajectory image
    pltPath=f'/home/rzhou/Projects/timegan-pytorch/output/rounD_multi_09-23_seq250_numfea2_Epoch5000/fig/{epoch}/{random_index}_trajectory.png'
    checkDir(os.path.dirname(pltPath))
    plt.savefig(pltPath)
    plt.close(fig)  # 关闭当前绘图窗口，防止过多图形打开