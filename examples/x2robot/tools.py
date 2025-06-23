import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
import torchvision
import torch
from contextlib import contextmanager
from typing import Union, List, Dict, Tuple


_ACTION_KEY_FULL_MAPPING = {
    'follow_right_arm_joint_pos': 'follow_right_joint_pos',
    'follow_right_arm_joint_dev': 'follow_right_joint_dev',
    'follow_right_arm_joint_cur': 'follow_right_joint_cur',
    'follow_right_ee_cartesian_pos': 'follow_right_position',
    'follow_right_ee_rotation': 'follow_right_rotation',
    'follow_right_gripper': 'follow_right_gripper',
    'master_right_arm_joint_pos': 'master_right_joint_pos',
    'master_right_arm_joint_dev': 'master_right_joint_dev',
    'master_right_arm_joint_cur': 'master_right_joint_cur',
    'master_right_ee_cartesian_pos': 'master_right_position',
    'master_right_ee_rotation': 'master_right_rotation',
    'master_right_gripper': 'master_right_gripper',
    'follow_left_arm_joint_pos': 'follow_left_joint_pos',
    'follow_left_arm_joint_dev': 'follow_left_joint_dev',
    'follow_left_arm_joint_cur': 'follow_left_joint_cur',
    'follow_left_ee_cartesian_pos': 'follow_left_position',
    'follow_left_ee_rotation': 'follow_left_rotation',
    'follow_left_gripper': 'follow_left_gripper',
    'master_left_arm_joint_pos': 'master_left_joint_pos',
    'master_left_arm_joint_dev': 'master_left_joint_dev',
    'master_left_arm_joint_cur': 'master_left_joint_cur',
    'master_left_ee_cartesian_pos': 'master_left_position',
    'master_left_ee_rotation': 'master_left_rotation',
    'master_left_gripper': 'master_left_gripper',
    'master_left_joint_pos': 'master_left_joint_pos',
    'master_right_joint_pos': 'master_right_joint_pos',
    "base_movement": "base_movement",
    "car_pose": "car_pose",
    'head_actions': 'head_rotation',
    "height":"lifting_mechanism_position",

}

def decode_video_torchvision(file_name, keyframes_only=True, backend = 'pyav'):
    '''
    Decode video using torchvision.io.VideoReader
    '''
    torchvision.set_video_backend(backend)
    reader = torchvision.io.VideoReader(file_name, "video")
    reader.seek(0, keyframes_only=keyframes_only)

    # load all frames until last requested frame
    loaded_frames = []
    loaded_ts = []
    for frame in reader:
        current_ts = frame["pts"]
        loaded_frames.append(frame["data"])
        loaded_ts.append(current_ts)

    reader.container.close()
    reader = None
    loaded_frames = torch.stack(loaded_frames).numpy()

    return loaded_frames

def process_action(file_path, action_key_mapping=_ACTION_KEY_FULL_MAPPING, filter_angle_outliers=True):
    file_name = os.path.basename(file_path)
    action_path = os.path.join(file_path, f"{file_name}.json")
    
    # 直接预加载键映射关系，避免重复调用 .keys()
    key_map = action_key_mapping
    
    # 使用 defaultdict(list) 替代 lambda:[] 提高初始化效率
    trajectories = defaultdict(list)
    
    with open(action_path, 'r') as file:
        actions = json.load(file)
        data = actions['data']
        
        for action in data:
            for key, val in action.items():
                new_key = key_map.get(key)
                if new_key is not None:
                    trajectories[new_key].append(val)  # 先收集原始数据
    
    trajectories = {k: np.array(v, dtype=np.float32) for k, v in trajectories.items()}
    
    if filter_angle_outliers:
        trajectories = smooth_action(trajectories)
    
    return trajectories


def smooth_action(action):
    def _filter(traj, threshold = 3, alpha = 0.05, window=10):
        # Convert to pandas Series but preserve the original dtype
        orig_dtype = traj.dtype
        data = pd.Series(traj)
        derivatives = np.diff(data)

        spike_indices = np.where(abs(derivatives) > threshold)[0]
        if len(spike_indices) > 0:
            ema = data.ewm(alpha=alpha, adjust=True).mean()
            
            # Fix: Ensure the slice indices are within bounds
            start_idx = max(0, spike_indices[0] - window)
            end_idx = min(len(data), spike_indices[-1] + window + 1)
            
            # Get the corresponding segment from the EMA
            modified_seg = ema.iloc[start_idx:end_idx]
            
            # Ensure the lengths match before assignment and explicitly convert to the original dtype
            if len(modified_seg) > 0:
                # Convert values back to the original dtype before assignment
                data.iloc[start_idx:end_idx] = modified_seg.values.astype(orig_dtype)
                
        return data.to_numpy().astype(orig_dtype)  # Ensure we return the same dtype

    for key in ['follow_right_ee_rotation', 'follow_left_ee_rotation']:
        if key in action:  # Check if the key exists in the action dictionary
            try:
                # Process each dimension separately while preserving dtype
                orig_dtype = action[key].dtype
                filtered_traj = np.stack([_filter(action[key][:,i]) for i in range(3)], axis=1)
                if not np.isnan(filtered_traj).any():
                    action[key] = filtered_traj.astype(orig_dtype)  # Ensure consistent dtype
            except (IndexError, ValueError) as e:
                print(f"Warning: Could not smooth {key} due to error: {e}")
    
    return action