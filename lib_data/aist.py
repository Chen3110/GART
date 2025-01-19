import os
import os.path as osp
import numpy as np
import imageio
import torch
import cv2
import json
from torch.utils.data import Dataset

class AISTDataset(Dataset):
    def __init__(
        self,
        data_root="data/aist",
        video_name="gBR_sBM_c03_d04_mBR0_ch08",
        split="train",
        image_zoom_ratio=0.5,
        bg_color=0.0,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.video_name = video_name
        self.image_zoom_ratio = image_zoom_ratio
        self.bg_color = bg_color

        # Set the root directory
        root = osp.join(data_root, video_name)

        # Load the annotation file (JSON)
        anno_fn = osp.join(root, "setting1.json")
        with open(anno_fn, 'rt') as f:
            annots = json.load(f)

        self.cams = annots

        # Load all image paths
        image_dir = osp.join(root, "image")
        self.ims = sorted([osp.join("image", img) for img in os.listdir(image_dir) if img.endswith(".jpg")])

        # Load all mask paths
        mask_dir = osp.join(root, "mask")
        self.masks = sorted([osp.join("mask", mask) for mask in os.listdir(mask_dir) if mask.endswith(".png")])

        # Load SMPL parameters
        smpl_fn = osp.join(root, "smpl.pkl")
        self.smpl_data = np.load(smpl_fn, allow_pickle=True)

        # # Camera parameters
        # num_cams = len(self.cams)
        # test_view = [i for i in range(num_cams) if i not in training_view]
        # if len(test_view) == 0:
        #     test_view = [0]
        training_view = int(video_name[9:11]) - 1
        if split == "train" or split == "prune":
            self.view = training_view
        # elif split == "test":
        #     self.view = test_view
        # elif split == "val":
        #     self.view = test_view[::4]
        #     # self.view = test_view

        # Prepare lists to store SMPL pose, translation, and shape parameters
        self.smpl_theta_list, self.smpl_trans_list, smpl_beta_list = [], [], []
        for frame_idx in range(len(self.ims)):
            smpl_theta = self.smpl_data["smpl_poses"][frame_idx].reshape((24, 3))
            smpl_trans = self.smpl_data["smpl_trans"][frame_idx]
            smpl_beta = self.smpl_data["smpl_beta"][frame_idx]

            self.smpl_theta_list.append(smpl_theta)
            self.smpl_trans_list.append(smpl_trans)
            smpl_beta_list.append(smpl_beta)

        self.beta = np.array(smpl_beta_list).mean(0)


    def __len__(self):
        # Return the total number of frames in the dataset
        return len(self.ims)

    def __getitem__(self, index):
        # Load the image for the current index
        img_path = osp.join(self.data_root, self.video_name, self.ims[index])
        img = imageio.imread(img_path).astype(np.float32) / 255.0

        # Load the mask for the current index
        mask_path = osp.join(self.data_root, self.video_name, self.masks[index])
        msk = imageio.imread(mask_path)

        # Resize the mask to match the image dimensions
        H, W = img.shape[:2]
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)

        # Get the camera intrinsic parameters for the current view
        K = np.array(self.cams[self.view]["matrix"])
        D = np.array(self.cams[self.view]["distortions"])

        # Undistort the image and the mask using the camera intrinsics
        img = cv2.undistort(img, K, D)
        msk = cv2.undistort(msk, K, D)

        # Scale the image and mask based on the zoom ratio
        H, W = int(img.shape[0] * self.image_zoom_ratio), int(img.shape[1] * self.image_zoom_ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)

        # Adjust the camera intrinsics according to the scaling
        K[:2] = K[:2] * self.image_zoom_ratio

        # Set the background of the image to the given color where the mask is zero
        img[msk == 0] = self.bg_color

        # Prepare the return dictionary with image, mask, camera intrinsics, and SMPL parameters
        ret = {
            "rgb": img.astype(np.float32),
            "mask": msk.astype(np.bool_).astype(np.float32),
            "K": K.copy().astype(np.float32),
            "smpl_beta": self.beta.astype(np.float32),
            "smpl_pose": self.smpl_theta_list[index].astype(np.float32),
            "smpl_trans": self.smpl_trans_list[index].astype(np.float32),
            "idx": index,
        }
        meta_info = {
            "video": self.video_name,
            "cam_ind": self.view,
            "frame_idx": self.ims[index],
        }
        viz_id = f"video{self.video_name}_dataidx{index}"
        meta_info["viz_id"] = viz_id

        return ret, meta_info

if __name__ == "__main__":
    # Usage example
    dataset = AISTDataset(
        data_root="data/aist",
        video_name="gBR_sBM_c03_d04_mBR0_ch08",
        split="train",
    )
    data = dataset[0]
    print(data)
