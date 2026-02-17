import glob
import json
import os

import numpy as np
import PIL.Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils.color import get_rays_np, load_hdr, sample_env_map


class JinxSynthetic(Dataset):
    def __init__(self, img_path, hdr_path):
        self.img_path = img_path
        self.hdr_path = hdr_path

        # Camera information.
        cam_transform = f"{img_path}/transforms.json"
        with open(cam_transform, "r") as fp:
            meta = json.load(fp)
        H = W = 512
        camera_angle_x = float(meta["camera_angle_x"])
        focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
        self.num_views = len(meta["view"])
        self.sample_rays = []
        for view_idx in range(self.num_views):
            pose = np.array(meta["view"][view_idx]["transform_matrix"]).astype(np.float32)
            rays_o, rays_d = get_rays_np(H, W, focal, pose)
            self.sample_rays.append(rays_d)

        # Jewelry information.
        self.num_jewelries = len(os.listdir(f"{self.img_path}/real_data_relighting")) // self.num_views
        self.all_fnames = sorted(glob.glob(f"{self.img_path}/real_data_relighting/*/*.png"))

        # HDR information.
        self.num_hdrs = len(self.all_fnames) // (self.num_jewelries * self.num_views)

    def __len__(self):
        return len(self.all_fnames)

    def __getitem__(self, index):
        fname = self.all_fnames[index]
        # print(f"index: {index}")
        # print(f"fname: {fname}")
        path, filename = os.path.split(fname)
        jewelry_view = os.path.basename(path)
        jewelry_name, view_idx = jewelry_view.rsplit("_", 1)
        hdr_name = os.path.splitext(filename)[0]

        # Load images.
        gt = PIL.Image.open(fname)
        gt = np.array(gt) / 255.0  # [0, 255] => [0, 1]
        image = PIL.Image.open(f"{self.img_path}/preprocessed_data/img/{jewelry_name}_{view_idx}.png")
        image = np.array(image) / 255.0  # [0, 255] => [0, 1]
        mask = PIL.Image.open(f"{self.img_path}/preprocessed_data/mask/{jewelry_name}_{view_idx}.png")
        mask = np.array(mask) / 255.0  # [0, 255] => [0, 1]
        if os.path.exists(f"{self.hdr_path}/{hdr_name}.exr"):
            hdr_name = f"{self.hdr_path}/{hdr_name}.exr"
        elif os.path.exists(f"{self.hdr_path}/{hdr_name}.hdr"):
            hdr_name = f"{self.hdr_path}/{hdr_name}.hdr"
        else:
            raise FileNotFoundError(f"{hdr_name} does not exist under the directory {self.hdr_path}")
        env_hdr = load_hdr(hdr_name)
        env_hdr = torch.tensor(env_hdr, dtype=torch.float32)

        # Preprocess images.
        alpha = gt[..., 3:]
        mask = mask[..., :1]
        rays_d = self.sample_rays[int(view_idx)]
        rays_d = torch.tensor(rays_d, dtype=torch.float32)
        rays_d = F.normalize(rays_d, dim=-1)  # [H, W, 3]
        sample_hdr = sample_env_map(env_hdr, rays_d)
        bg = sample_hdr.clamp(0, 1)
        bg = bg ** (1 / 2.2)  # gamma correction
        bg = bg.numpy()
        gt = gt[..., :3] * alpha + bg * (1 - alpha)
        composite = image[..., :3] * alpha + bg * (1 - alpha)

        return {"composite": composite.transpose(2, 0, 1), "gt": gt.transpose(2, 0, 1), "bg": bg.transpose(2, 0, 1), "mask": mask.transpose(2, 0, 1), "alpha": alpha.transpose(2, 0, 1)}
