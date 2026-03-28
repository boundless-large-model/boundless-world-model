"""Data processing operators for robot action/state data.

This module provides operators for loading and normalizing robot
data from parquet files, supporting various action representations.
"""

import json
import math
import os
from typing import Dict, Any, Optional, List, Tuple

import imageio
import imageio.v3 as iio
import numpy as np
import pyarrow.parquet as pq
import torch
import torchvision
from PIL import Image

from diffsynth.core.data.operators import DataProcessingOperator, ToAbsolutePath, FrameSamplerByRateMixin


"""
Class: DataProcessingOperator
-----------------------------

Overloads the right-shift operator (`>>`) utilizing the `__rshift__` magic method.

This implementation facilitates intuitive pipeline composition, allowing multiple 
data processing operators to be chained together sequentially 
(e.g., `operator_A >> operator_B`).
"""
class ApplyOperatorToDict(DataProcessingOperator):
    """
    Applies a given operator to a specific key in a dictionary.

    Args:
        key: The dictionary key whose value will be processed.
        operator: The DataProcessingOperator to apply to the value.
        inplace: If True, modifies the input dictionary directly. 
                 If False, creates a shallow copy. Default is False.
    """
    def __init__(self, key: str, operator: DataProcessingOperator, inplace: bool = False):
        self.key = key
        self.operator = operator
        self.inplace = inplace

    def __call__(self, data: dict):
        if not isinstance(data, dict):
            raise TypeError(f"ApplyToKey expects a dictionary, got {type(data).__name__}.")
        
        if self.key not in data:
            raise KeyError(f"Key '{self.key}' not found in the input dictionary.")

        if self.inplace:
            data[self.key] = self.operator(data[self.key])
            return data
        else:
            updated_data = data.copy()
            updated_data[self.key] = self.operator(data[self.key])
            return updated_data


class ResolvePromptEmbPath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path

    def __call__(self, data: str):
        if os.path.isabs(data):
            return data
        return os.path.join(self.base_path, data)


class LoadVideoChunk(DataProcessingOperator, FrameSamplerByRateMixin):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x, frame_rate=24, fix_frame_rate=False):
        FrameSamplerByRateMixin.__init__(self, num_frames, time_division_factor, time_division_remainder, frame_rate, fix_frame_rate)
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor

    def __call__(self, data: str, start_frame=None, end_frame=None):
        reader = self.get_reader(data)
        raw_frame_rate = reader.get_meta_data()['fps']
        total_raw_frames = reader.count_frames()
        
        start = max(0, start_frame if start_frame is not None else 0)
        end = min(total_raw_frames, end_frame if end_frame is not None else total_raw_frames)
        clip_frames = max(0, end - start)

        # x / clip_frames = self.frame_rate / raw_frame_rate
        available_frames = int(clip_frames * self.frame_rate / raw_frame_rate) if self.fix_frame_rate else clip_frames
        num_frames = self.num_frames
        if available_frames < num_frames:
            num_frames = available_frames
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        
        frames = []
        for frame_id in range(num_frames):
            frame_id = self.map_single_frame_id(frame_id, raw_frame_rate, clip_frames)
            frame = reader.get_data(start + frame_id)
            frame = Image.fromarray(frame)
            frame = self.frame_processor(frame)
            frames.append(frame)
        reader.close()
        return frames
    

class LoadGIFChunk(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor

    def get_num_frames(self, clip_frames):
        num_frames = self.num_frames
        if clip_frames < num_frames:
            num_frames = clip_frames
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str, start_frame=None, end_frame=None):
        images = iio.imread(data, mode="RGB")
        total_raw_frames = len(images)
        
        start = max(0, start_frame if start_frame is not None else 0)
        end = min(total_raw_frames, end_frame if end_frame is not None else total_raw_frames)
        clip_frames = max(0, end - start)

        num_frames = self.get_num_frames(clip_frames)
        frames = []
        for img in images[start : start + num_frames]:
            frame = Image.fromarray(img)
            frame = self.frame_processor(frame)
            frames.append(frame)
        return frames


class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height=None, width=None, max_pixels=None, height_division_factor=1, width_division_factor=1, resize_mode: str = "fit"):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.resize_mode = resize_mode # "fit" / "crop"

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size

        if self.resize_mode == "crop":
            scale = max(target_width / width, target_height / height)
            image = torchvision.transforms.functional.resize(
                image,
                (round(height*scale), round(width*scale)),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            )
            image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
            return image

        elif self.resize_mode == "fit":
            image = torchvision.transforms.functional.resize(
                image, 
                [target_height, target_width], 
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            )
            return image
        
    def get_height_width(self, image):
        if self.resize_mode == "crop" and self.height is not None and self.width is not None:
            return self.height, self.width

        width, height = image.size
        max_area = self.height * self.width if (self.height is not None and self.width is not None) else self.max_pixels
        if max_area is not None and width * height > max_area:
            scale = (width * height / max_area) ** 0.5
            height, width = int(height / scale), int(width / scale)
        height = height // self.height_division_factor * self.height_division_factor
        width = width // self.width_division_factor * self.width_division_factor

        return height, width

    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image
    
    
class ToVideoTensor(DataProcessingOperator):
    """Convert loaded video frames to float tensor in (V, C, T, H, W), range [-1, 1].

    This operator converts a list of PIL Images or list of lists (for multi-view)
    into a normalized video tensor.
    """

    @staticmethod
    def _frame_to_tensor(frame: Image.Image) -> torch.Tensor:
        """Convert a single PIL Image to CHW tensor in range [-1, 1]."""
        if not isinstance(frame, Image.Image):
            raise TypeError(f"Expected PIL.Image, got {type(frame).__name__}")
        
        if frame.mode != "RGB":
            frame = frame.convert("RGB")
            
        array = np.asarray(frame, dtype=np.float32)
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()  # (C, H, W)
        tensor = tensor * (2.0 / 255.0) - 1.0
        return tensor

    def _frames_to_video_tensor(self, frames) -> torch.Tensor:
        """Convert a list of frames to (C, T, H, W) tensor."""
        if not isinstance(frames, (list, tuple)) or len(frames) == 0:
            raise ValueError("Expected non-empty frame list.")
        
        frame_tensors = [self._frame_to_tensor(frame) for frame in frames]
        video = torch.stack(frame_tensors, dim=1)  # (C, T, H, W)
        return video

    def __call__(self, data):
        """Convert data to video tensor.

        Args:
            data: One of:
                - torch.Tensor: Already a tensor, validate shape
                - PIL.Image: Single frame, treat as 1-frame video
                - list of PIL.Image: Single-view video
                - list of list of PIL.Image: Multi-view video

        Returns:
            torch.Tensor of shape (V, C, T, H, W) in range [-1, 1]
        """
        if isinstance(data, torch.Tensor):
            if data.ndim != 5:
                raise ValueError(f"Expected video tensor with shape (V,C,T,H,W), got {tuple(data.shape)}")
            
            return data.to(dtype=torch.float32)

        if isinstance(data, Image.Image):
            data = [data]

        if not isinstance(data, (list, tuple)) or len(data) == 0:
            raise TypeError("Expected loaded video frames as list/tuple.")

        # Check if multi-view (list of lists)
        if isinstance(data[0], (list, tuple)):
            views = [self._frames_to_video_tensor(view) for view in data]
            return torch.stack(views, dim=0)  # (V, C, T, H, W)

        # Single view
        video = self._frames_to_video_tensor(data).unsqueeze(0)  # (1, C, T, H, W)
        return video
    
# TODO: change "OBS_ACTION_NAMES" to "JOINT_AND_EEF_NAMES" 
JOINT_AND_EEF_NAMES = [
    "left_arm_joint_1_rad",
    "left_arm_joint_2_rad",
    "left_arm_joint_3_rad",
    "left_arm_joint_4_rad",
    "left_arm_joint_5_rad",
    "left_arm_joint_6_rad",
    "left_gripper_open",
    "left_eef_pos_x_m",
    "left_eef_pos_y_m",
    "left_eef_pos_z_m",
    "left_eef_rot_euler_x_rad",
    "left_eef_rot_euler_y_rad",
    "left_eef_rot_euler_z_rad",
    "right_arm_joint_1_rad",
    "right_arm_joint_2_rad",
    "right_arm_joint_3_rad",
    "right_arm_joint_4_rad",
    "right_arm_joint_5_rad",
    "right_arm_joint_6_rad",
    "right_gripper_open",
    "right_eef_pos_x_m",
    "right_eef_pos_y_m",
    "right_eef_pos_z_m",
    "right_eef_rot_euler_x_rad",
    "right_eef_rot_euler_y_rad",
    "right_eef_rot_euler_z_rad",
]

JOINT_NAMES = [
    "left_arm_joint_1_rad",
    "left_arm_joint_2_rad",
    "left_arm_joint_3_rad",
    "left_arm_joint_4_rad",
    "left_arm_joint_5_rad",
    "left_arm_joint_6_rad",
    "left_gripper_open",
    "right_arm_joint_1_rad",
    "right_arm_joint_2_rad",
    "right_arm_joint_3_rad",
    "right_arm_joint_4_rad",
    "right_arm_joint_5_rad",
    "right_arm_joint_6_rad",
    "right_gripper_open",
]

# TODO: change "POSE_NAMES" to "EEF_NAMES"
EEF_NAMES = [
    "left_eef_pos_x_m",
    "left_eef_pos_y_m",
    "left_eef_pos_z_m",
    "left_eef_rot_euler_x_rad",
    "left_eef_rot_euler_y_rad",
    "left_eef_rot_euler_z_rad",
    "left_gripper_open",
    "right_eef_pos_x_m",
    "right_eef_pos_y_m",
    "right_eef_pos_z_m",
    "right_eef_rot_euler_x_rad",
    "right_eef_rot_euler_y_rad",
    "right_eef_rot_euler_z_rad",
    "right_gripper_open",
]


class LoadCobotAction(DataProcessingOperator):
    def __init__(
        self,
        base_path="",
        action_type="joint_abs",
        stat=None,
        use_percentile_stats=True,
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
    ):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        """
            joint_abs (原 state_joint：关节绝对位置)
            eef_abs (原 state_pose：末端绝对位姿)
            joint_delta (原 action_joint：关节相对动作/增量)
            eef_delta (原 action_pose：末端相对动作/增量)
        """
        if action_type not in ("joint_abs", "eef_abs", "joint_delta", "eef_delta"):
            raise ValueError(f"Unsupported action type: {action_type}")
        self.base_path = base_path
        self.action_type = action_type
        self.stat = stat or {}
        self.use_percentile_stats = use_percentile_stats
        # TODO: rename "use_state" to "use_absolute"
        self.use_absolute = action_type.endswith("_abs")
        self.use_joint = action_type.startswith("joint_")
        name_to_idx = {name: idx for idx, name in enumerate(JOINT_AND_EEF_NAMES)}
        self.indices = [name_to_idx[name] for name in (JOINT_NAMES if self.use_joint else EEF_NAMES)]
        self._stat_min = None
        self._stat_max = None
        if self.stat and action_type in self.stat:
            entry = self.stat[action_type]
            if self.use_percentile_stats:
                # Filter out abnormal sensor spikes 
                self._stat_min = np.asarray(entry.get("p01", []), dtype=np.float32)
                self._stat_max = np.asarray(entry.get("p99", []), dtype=np.float32)
            else:
                self._stat_min = np.asarray(entry.get("min", []), dtype=np.float32)
                self._stat_max = np.asarray(entry.get("max", []), dtype=np.float32)

    def _resolve_parquet_info(self, data, start_frame, end_frame):
        if isinstance(data, dict):
            parquet_rel = data.get("data")
            if start_frame is None:
                start_frame = data.get("start_frame")
            if end_frame is None:
                end_frame = data.get("end_frame")
        else:
            parquet_rel = data
        
        if not parquet_rel:
            raise KeyError("Missing parquet path in metadata 'data' field.")
        
        if os.path.isabs(parquet_rel):
            parquet_path = parquet_rel
        else:
            parquet_path = os.path.join(self.base_path, parquet_rel)

        start_frame = int(start_frame)
        end_frame = int(end_frame)
        return parquet_path, start_frame, end_frame

    def _get_min_max(self):
        if self._stat_min is not None and self._stat_max is not None:
            return self._stat_min, self._stat_max
        raise KeyError(f"Missing normalization stats for action type: {self.action_type}")

    def _normalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1.0,
        clip_max: float = 1.0,
        eps: float = 1e-8,
    ) -> np.ndarray:
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1.0
        return np.clip(ndata, clip_min, clip_max)

    def _read_slice(self, parquet_path, column, start_frame, num_frames):
        start = int(start_frame)
        end = start + int(num_frames)
        table = pq.read_table(parquet_path, columns=[column])
        data = table.to_pydict()[column]
        if end > len(data):
            raise ValueError(
                f"Not enough rows in {parquet_path} for slice "
                f"start={start_frame}, num_frames={num_frames}"
            )
        return np.asarray(data[start:end], dtype=np.float32)

    def get_num_frames(self, total_frames):
        num_frames = int(self.num_frames)
        if int(total_frames) < num_frames:
            num_frames = int(total_frames)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames

    def __call__(self, data: str, start_frame=None, end_frame=None):
        parquet_path, start_frame, end_frame = self._resolve_parquet_info(
            data, start_frame, end_frame
        )
        num_frames = self.get_num_frames(end_frame - start_frame + 1)
        column = "observation.state" if self.use_absolute else "action"
        arr = self._read_slice(parquet_path, column, start_frame, num_frames)
        if arr.ndim != 2:
            raise ValueError(f"Unexpected action shape {arr.shape} in {parquet_path}")
        if arr.shape[1] == len(JOINT_AND_EEF_NAMES):
            arr = arr[:, self.indices]
        elif self.use_joint and arr.shape[1] == len(JOINT_NAMES):
            pass
        elif (not self.use_joint) and arr.shape[1] == len(EEF_NAMES):
            pass
        else:
            raise ValueError(
                f"Unexpected action width {arr.shape[1]} for action type {self.action_type} in {parquet_path}"
            )
        min_vals, max_vals = self._get_min_max()
        arr = self._normalize_bound(arr, min_vals, max_vals)
        return arr[None, ...]


