import torch
import matplotlib.pyplot as plt
import numpy as np
import trimesh
import pyrender


def viz_render(gt_rgb, gt_mask, pred_pkg, save_path=None):
    pred_rgb = pred_pkg["rgb"].permute(1, 2, 0)
    pred_mask = pred_pkg["alpha"].squeeze(0)
    pred_depth = pred_pkg["dep"].squeeze(0)
    fig = plt.figure(figsize=(20, 5))
    plt.subplot(1, 5, 1)
    plt.imshow(torch.clamp(gt_rgb, 0.0, 1.0).detach().cpu().numpy())
    plt.title("GT"), plt.axis("off")
    plt.subplot(1, 5, 2)
    plt.imshow(torch.clamp(pred_rgb, 0.0, 1.0).detach().cpu().numpy())
    plt.title("Pred view"), plt.axis("off")
    plt.subplot(1, 5, 3)
    error = torch.clamp(abs(pred_rgb - gt_rgb), 0.0, 1.0).detach().cpu().numpy().max(axis=-1)
    cmap = plt.imshow(error)
    plt.title("Render Error (max in rgb)"), plt.axis("off")
    plt.colorbar(cmap, shrink=0.8)

    plt.subplot(1, 5, 4)
    error = torch.clamp(pred_mask - gt_mask, -1.0, 1.0).detach().cpu().numpy()
    cmap = plt.imshow(error)
    plt.title("(Pr - GT) Mask Error"), plt.axis("off")
    plt.colorbar(cmap, shrink=0.8)
    
    plt.subplot(1, 5, 5)
    depth = pred_depth.detach().cpu().numpy()
    cmap = plt.imshow(depth)
    plt.title("Pred Depth"), plt.axis("off")
    plt.colorbar(cmap, shrink=0.8)

    plt.tight_layout()
    fig.canvas.draw()
    fig_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    fig_np = fig_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    if save_path is not None:
        plt.savefig(save_path)
    plt.close(fig)
    return fig_np


def copy2cpu(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    else:
        return None


def render_mesh(image, v, f, K):
    image, v, f, K = copy2cpu(image), copy2cpu(v), copy2cpu(f), copy2cpu(K)
    mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices=v,faces=f))
    scene = pyrender.Scene()
    scene.add(mesh)
    camera = pyrender.IntrinsicsCamera(fx=K[0,0], fy=K[1,1], cx=K[0,2], cy=K[1,2], znear=0.05, zfar=100000.0, name=None)
    camera_pose = np.array([
       [1.0,  0.0, 0.0, 0.0],
       [0.0,  -1.0, 0.0, 0.0],
       [0.0,  0.0, -1.0, 0.0],
       [0.0,  0.0, 0.0, 1.0],
    ])
    scene.add(camera, pose=camera_pose)
    light = pyrender.SpotLight(color=np.asarray([158, 219, 251]) / 255, intensity=128,
                               innerConeAngle=np.pi/16.0,
                               outerConeAngle=np.pi/6.0)
    scene.add(light, pose=camera_pose)
    r = pyrender.OffscreenRenderer(image.shape[1], image.shape[0])
    color, depth = r.render(scene)
    blend = (color == 255)
    blend = blend[:,:,0] & blend[:,:,1] & blend[:,:,2]
    blend = (1-blend.astype(np.float32)) * 0.8
    blend = blend[:,:,None]
    blended_image = color*blend + image*(1-blend)
    return blended_image.astype(np.uint8)
