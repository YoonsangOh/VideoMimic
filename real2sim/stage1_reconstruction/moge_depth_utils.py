import os
import cv2
import numpy as np
import torch
from tqdm import tqdm


def load_moge_model(pretrained_model="Ruicheng/moge-vitl", device=None):
    try:
        from moge.model import MoGeModel
    except ImportError as exc:
        raise ImportError(
            "MoGe import failed. Ensure `real2sim/moge` exists and MoGe dependencies are installed."
        ) from exc

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MoGeModel.from_pretrained(pretrained_model).to(device).eval()
    return model


def infer_moge_inverse_depths(moge_model, image_paths, out_dir=None, save=False, fovs=None):
    """
    Run MoGe over frames and return inverse-depth maps.
    droid_slam_optimize expects disparity-like priors.
    """
    outputs = []
    model_device = next(moge_model.parameters()).device

    for idx, image_path in enumerate(tqdm(image_paths)):
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image_tensor = torch.tensor(
            image / 255.0, dtype=torch.float32, device=model_device
        ).permute(2, 0, 1)

        with torch.no_grad():
            if fovs is not None and idx < len(fovs) and fovs[idx] is not None:
                fov_x = torch.tensor(float(fovs[idx]), dtype=torch.float32, device=model_device)
                result = moge_model.infer(image_tensor, fov_x=fov_x, force_projection=False)
            else:
                result = moge_model.infer(image_tensor, force_projection=False)

        raw_depth = result["raw_depth"].clone()
        raw_mask = result["raw_mask"].clone()
        raw_mask[~torch.isfinite(raw_depth)] = False

        raw_inv_depth = 1.0 / torch.clamp(raw_depth, 1e-4, 1e4)
        raw_inv_depth[~raw_mask] = 0.0
        inv_depth_np = np.float32(raw_inv_depth.detach().cpu().numpy().squeeze())
        outputs.append(inv_depth_np)

        if save and out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            np.save(
                os.path.join(out_dir, os.path.basename(image_path).split(".")[0] + ".npy"),
                inv_depth_np,
            )

    return outputs

