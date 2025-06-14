import torch
import numpy as np
import vedo
import trimesh

from sklearn import datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def plot_t_SNE(X, y):
    tsne = TSNE(n_components=2, random_state=42)

    X_2d = tsne.fit_transform(X)

    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis')
    plt.colorbar()
    plt.show()

def _convert_to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        pass
    else:
        raise TypeError("Input must be a torch.Tensor or np.ndarray")
    return x


def show_pc(pc):
    pc = _convert_to_numpy(pc)
    pts = vedo.Points(pc, r=10, c=(0.2, 0.5, 1.0)).clean()
    vedo.show(pts, title="pc", axes=False)
    vedo.close()


def show_mix_pc(pc1, pc2, colors=[(0.2, 0.5, 1.0), (1, 0, 0)]):
    pc1 = _convert_to_numpy(pc1)
    pc2 = _convert_to_numpy(pc2)
    pts1 = vedo.Points(pc1, r=20, c=colors[0]).clean()
    pts2 = vedo.Points(pc2, r=20, c=colors[1]).clean()
    vedo.show(pts2 + pts1, title="mix pc", axes=False)
    vedo.close()


def show_pc_list(pc_list, colors_list):
    vedo_pc = []
    for pc, color in zip(pc_list, colors_list):
        pc = _convert_to_numpy(pc)
        vedo_pc.append(vedo.Points(pc, r=20, c=color).clean())
    vedo.show(*vedo_pc, title="pc list", axes=False)
    vedo.close()


def show_pc_with_highlight(pc, hp_idx, colors='red'):
    pc = _convert_to_numpy(pc)
    hp_idx = _convert_to_numpy(hp_idx)

    pts = vedo.Points(pc, r=20, c=(0.2, 0.5, 1.0)).clean()

    hl_pc = pc[hp_idx, :]
    hl_pts = vedo.Points(hl_pc, r=25, c=colors).clean()

    vedo.show(hl_pts + pts, title="highlight pc", axes=False)
    vedo.close()


def show_pc_with_score(pc, score):
    pc = _convert_to_numpy(pc)
    score  = _convert_to_numpy(score)
    p = vedo.Points(pc, r=20).cmap('jet', score, on='points').add_scalarbar('length', font_size=15)
    vedo.show(p)
    vedo.close()


def show_mesh(mesh):
    plt = vedo.Plotter()
    plt.add(mesh)
    plt.show()
    plt.close()

def show_mesh_off(file_path):
    mesh = trimesh.load_mesh(file_path)
    show_mesh(mesh)

