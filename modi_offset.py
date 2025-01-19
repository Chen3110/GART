import cv2
import pyrender
import torch
import pickle
import os
import json
import numpy as np
import trimesh

from lib_gart.hmr2.models import load_hmr2
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from lib_gart.hmr2.models.smpl_wrapper import SMPL

def copy2cpu(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    else:
        return None

def render_mesh(v, f, K):
    mm = trimesh.Trimesh(vertices=v,faces=f)
    mesh = pyrender.Mesh.from_trimesh(mm)
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
    light = pyrender.SpotLight(color=np.ones(3), intensity=16,
                               innerConeAngle=np.pi/8.0,
                               outerConeAngle=np.pi/3.0)
    scene.add(light, pose=camera_pose)
    r = pyrender.OffscreenRenderer(1920, 1080)
    color, depth = r.render(scene)
    return color

def generator(points=None, pred_vertices=None, opti_vertices=None, gt_vertices=None, faces=None):
    if points is not None:
        batch_size = len(points)
    elif pred_vertices is not None:
        batch_size = len(pred_vertices)
    elif opti_vertices is not None:
        batch_size = len(opti_vertices)
    elif gt_vertices is not None:
        batch_size = len(gt_vertices)
    for i in range(batch_size):
        res = {}
        res.update(dict(
            points = dict(
                pcl = copy2cpu(points[i][:,:3]) if points is not None else None,
                # colors = copy2cpu(points[i][:,3:6]) if points[:,3:6] else [0,0,0.8],
                color = [0,0,0.8],
            ),
            pred_verts = dict(
                mesh = [copy2cpu(pred_vertices)[i], copy2cpu(faces)] if pred_vertices is not None else None,
                color = np.asarray([143, 240, 166]) / 255
            ),
            opti_verts = dict(
                mesh = [copy2cpu(opti_vertices)[i], copy2cpu(faces)] if opti_vertices is not None else None,
                color = np.asarray([158, 219, 251]) / 255
            ),
            label_verts = dict(
                mesh = [copy2cpu(gt_vertices)[i], copy2cpu(faces)] if gt_vertices is not None else None,
                color = np.asarray([235, 189, 191]) / 255,
            ),
        ))
        yield res

smpl_fn = "data/aist/gBR_sBM_c03_d04_mBR0_ch08/anno.pkl"
data = np.load(smpl_fn, allow_pickle=True)
gt_pose = torch.from_numpy(data['smpl_poses']).reshape(-1, 24, 3).cuda()
trans = torch.from_numpy(data['smpl_trans']).cuda()
scale = torch.from_numpy(data['smpl_scaling']).cuda()
hmr_model, _ = load_hmr2('data/hmr/epoch=35-step=1000000.ckpt')
hmr_model = hmr_model.cuda()
smpl = SMPL(model_path='data/smpl-meta', num_body_joints=23, mean_params='data/smpl-meta/smpl_mean_params.npz').cuda()
with open('data/aist/gBR_sBM_c03_d04_mBR0_ch08/setting1.json', 'rt') as f:
    setting = json.load(f)
K = np.array(setting[2]['matrix'])
R = axis_angle_to_matrix(torch.tensor(setting[2]['rotation'])[None])[0].cuda()
t = torch.tensor(setting[2]['translation']).cuda()
beta_list = []

with open("data/aist/gBR_sBM_c03_d04_mBR0_ch08/keypoints_3d.pkl", "rb") as f:
    kp = pickle.load(f)
keypoints3d = torch.tensor(kp['keypoints3d']).float().cuda()

from lib_render.visualization import StreamVisualization
o3d_viz = StreamVisualization()

with torch.no_grad():
    for i, img in enumerate(sorted(os.listdir('data/aist/gBR_sBM_c03_d04_mBR0_ch08/image_crop'))):
        rgb = cv2.imread(os.path.join('data/aist/gBR_sBM_c03_d04_mBR0_ch08/image_crop', img))
        rgb = torch.from_numpy(rgb).cuda().permute(2,0,1)[None]/255.
        res_rgb = torch.nn.functional.interpolate(rgb, size=(256, 256), mode='bilinear')
        hmr_output = hmr_model(res_rgb)
        beta_list.append(hmr_output['pred_smpl_params']['betas'])
        gt_smpl_params = {}
        gt_smpl_params['global_orient'] = axis_angle_to_matrix(gt_pose[i][:1][None])
        gt_smpl_params['body_pose'] = axis_angle_to_matrix(gt_pose[i][1:][None])
        gt_smpl_params['betas'] = hmr_output['pred_smpl_params']['betas']
        gt_smpl_output = smpl(**{k: v.float() for k,v in gt_smpl_params.items()}, pose2rot=False)
        root_joint = gt_smpl_output.joints[0, 0]
        root_rota = R @ axis_angle_to_matrix(gt_pose[i][:1])
        opti_root_pose = matrix_to_axis_angle(root_rota)[0]
        opti_trans = R @ trans[i]/scale + t/scale 
        opti_smpl_params = {}
        opti_smpl_params['global_orient'] = axis_angle_to_matrix(opti_root_pose[None])
        opti_smpl_params['body_pose'] = axis_angle_to_matrix(gt_pose[i][1:][None])
        opti_smpl_params['betas'] = hmr_output['pred_smpl_params']['betas']
        opti_smpl_output = smpl(**{k: v.float() for k,v in opti_smpl_params.items()}, pose2rot=False)
        offset = (gt_smpl_output.joints[0,0] + trans[i]/scale) @ R.T + t/scale - (opti_smpl_output.joints[0,0] + opti_trans)
        gen = generator(
            points=(keypoints3d[i]/scale @ R.T + t/scale)[None],
            # points=(gt_smpl_output.joints + trans[i]/scale) @ R.T + t/scale,
            pred_vertices=hmr_output['pred_vertices'],
            opti_vertices=opti_smpl_output.vertices + opti_trans + offset,
            gt_vertices=(gt_smpl_output.vertices + trans[i]/scale) @ R.T + t/scale,
            faces=smpl.faces
        )
        o3d_viz.show(gen)
        trans[i] = opti_trans + offset
        gt_pose[i][:1] = opti_root_pose
data['smpl_beta'] = torch.cat(beta_list, dim=0).cpu().numpy()
data['smpl_poses'] = gt_pose[:-1].cpu().numpy().reshape(-1, 72)
data['smpl_trans'] = trans[:-1].cpu().numpy()

# with open("data/aist/gBR_sBM_c03_d04_mBR0_ch08/smpl.pkl", "wb") as f:
#     pickle.dump(data, f)