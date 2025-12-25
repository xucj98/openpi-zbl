import os
import tqdm 
import glob
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
os.environ["HF_LEROBOT_HOME"] = "/root/.cache/hf_home"  # Set it to yours

REPO_NAME = "test_case_carrot"  # TODO: Name of the output dataset, also used for the Hugging Face Hub: Change it to your own
RAW_DATASET_PATHS = [
    # '/x2robot/zhengwei/10055/20250529-day-fasten_the_belt',
    # '/x2robot/zhengwei/factory10026/20250529-day-fasten_the_belt',
    # '/root/weibolu_64',
    '/root/weibolu_41_1205',
    # '/x2robot/zhengwei/10055/20250530-day-fasten_the_belt',
    # '/x2robot/zhengwei/factory10026/20250606-day-fasten_the_belt',
    # '/x2robot/zhengwei/10055/20250606-day-fasten_the_belt',
]

FILE_CAME_MAPPING = {
    "face_view": "faceImg.mp4",
    "left_wrist_view": "leftImg.mp4",
    "right_wrist_view": "rightImg.mp4"
}

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
    "base_movement": "base_movement",
    "car_pose": "car_pose",
    'head_actions': 'head_rotation'

}
_ACTION_KEY_FULL_MAPPING_INV = {v:k for k,v in _ACTION_KEY_FULL_MAPPING.items()}

import dataclasses
import enum
import logging
import socket
import time
import copy
from pathlib import Path
import matplotlib.pyplot as plt
import time

import struct
import tyro
import json
import cv2
import numpy as np
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

import numpy as np
from scipy.spatial.transform import Rotation as R
from openpi.shared.tools import process_action, decode_video_torchvision
import einops

def interpolates_actions(actions, num_actions=20, target_num_actions = 80, action_dim=7):
    # 假设 actions 是你的动作序列，shape 为 [num_actions, action_dim]
    # 其中，欧拉角为 actions[:, 3:6]
    # return interpolated_actions 现在包含了插值后的动作序列，其中角度使用了球面插值
    # 生成目标动作序列的索引
    original_indices = np.linspace(0, num_actions - 1, num_actions)
    target_indices = np.linspace(0, num_actions - 1, target_num_actions)
    # 初始化插值后的动作序列数组
    interpolated_actions = np.zeros((target_num_actions, action_dim))
    if action_dim == 2: # 头部动作直接线性插值
        for i in range(action_dim):
            interpolated_actions[:, i] = np.interp(target_indices, original_indices, actions[:, i])
        return interpolated_actions

    # 对[x, y, z, gripper]使用线性插值
    for i in range(3):
        interpolated_actions[:, i] = np.interp(target_indices, original_indices, actions[:, i])
    interpolated_actions[:, -1] = np.interp(target_indices, original_indices, actions[:, -1])
    # 将欧拉角转换为四元数
    quaternions = R.from_euler('xyz', actions[:, 3:6]).as_quat()  # shape: [num_actions, 4]
    # 初始化插值后的四元数数组
    interpolated_quats = np.zeros((target_num_actions, 4))
    # 对四元数进行球面插值
    for i in range(4):  # 对四元数的每个分量进行插值
        interpolated_quats[:, i] = np.interp(target_indices, original_indices, quaternions[:, i])
    # 四元数规范化，确保插值后仍为单位四元数
    interpolated_quats = interpolated_quats / np.linalg.norm(interpolated_quats, axis=1, keepdims=True)
    # 将插值后的四元数转换回欧拉角
    interpolated_eulers = R.from_quat(interpolated_quats).as_euler('xyz')  # shape: [target_num_actions, 3]
    # 更新插值后动作序列的角度部分
    interpolated_actions[:, 3:6] = interpolated_eulers
    # print(interpolated_actions.shape)
    return interpolated_actions

class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    X2ROBOT = "x2robot"

@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


import threading



@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM
    default_prompt: str | None = None
    port: int = 8000
    record: bool = False
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)
    log_replay: bool = False
    openloop: bool = True
    openloop_filepath: str | None = '/home/fangxinyuan/projects/dataset/test_dataset'


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi0_aloha",
        dir="s3://openpi-assets/checkpoints/pi0_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="s3://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi0_fast_droid",
        dir="s3://openpi-assets/checkpoints/pi0_fast_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi0_fast_libero",
        dir="s3://openpi-assets/checkpoints/pi0_fast_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)

def recv_all(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def read_img(conn,i,save_path,count=0):
    image_size = struct.unpack('<L', conn.recv(4))[0]
    image = recvall(conn, image_size)
    nparr = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("image-{i}-{count}.jpg", image) #
    return image

def normalize_action(action_input, min_range, max_range):
    return (action_input - min_range) / (max_range - min_range)

def unnormalize_action(action_input, min_range, max_range):
    return action_input * (max_range - min_range) + min_range



def main(args: Args) -> None:

    policy = create_policy(args)

    count = 0

    if not args.openloop:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setblocking(True) #设置通信是阻塞式
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        ip = '192.168.77.58' #
        port = 57770
        sock.bind((ip, port))
        sock.listen(1)
        print(f"Server is listening on {ip}:{port}")

        conn, addr = sock.accept()
        print(f"Connection from {addr}")

    max_time_step = 100000

    # TODO: 需要修改
    min_range = np.array([-0.1, -0.5, -0.5, -3.0, -3.0, -3.0 , -9, -0.1, -0.5, -0.5, -3.0, -3.0, -3.0 , -9], dtype=np.float32)
    max_range = np.array([0.5,  0.5,  0.5, 3.0, 3.0, 3.0, 9,0.5,  0.5,  0.5, 3.0, 3.0, 3.0, 9], dtype=np.float32)

    while True:
        if count>max_time_step:
            break

        if not args.openloop:
            data_size = struct.unpack('<L', conn.recv(4))[0]
            # data = conn.recv(data_size)
            data = recv_all(conn, data_size)
            action_data = json.loads(data.decode('utf8'))

            left_agent_data = action_data['follow1_pos'] # (7)
            right_agent_data = action_data['follow2_pos'] # (7)

            save_path = 'None'
            image1 = read_img(conn,1,save_path,count) # left
            image2 = read_img(conn,2,save_path,count) # front
            image3 = read_img(conn,3,save_path,count) # right
            h,w,c = np.array(image1).shape
            camera_front = np.array(image2).reshape(h,w,c)
            camera_left = np.array(image1).reshape(h,w,c)
            camera_right = np.array(image3).reshape(h,w,c)
            state = np.concatenate([left_agent_data, right_agent_data])


        for raw_dataset_path in tqdm.tqdm(RAW_DATASET_PATHS):
            episode_paths = []
            for dataset_path in sorted(glob.glob(f'{raw_dataset_path}/*')):
                if os.path.isdir(dataset_path):
                    has_mp4_files = len(glob.glob(f'{dataset_path}/*.mp4')) > 0
                    if has_mp4_files:
                        episode_paths.append(dataset_path)

            # iterate over all discovered episodes (not only the last one)
            for episode_path in tqdm.tqdm(episode_paths):
                print(episode_path)
                pose_dicts = process_action(episode_path, action_key_mapping=_ACTION_KEY_FULL_MAPPING_INV)
                pose_dicts["follow_left_gripper"] = pose_dicts["follow_left_gripper"].reshape(-1, 1)
                pose_dicts["follow_right_gripper"] = pose_dicts["follow_right_gripper"].reshape(-1, 1)
                #pose_dicts["master_left_gripper"] = pose_dicts["master_left_gripper"].reshape(-1, 1)
                #pose_dicts["master_right_gripper"] = pose_dicts["master_right_gripper"].reshape(-1, 1)
                pred_poses = np.concatenate([pose_dicts[key] for key in ["follow_left_ee_cartesian_pos", "follow_left_ee_rotation", "follow_left_gripper", "follow_right_ee_cartesian_pos", "follow_right_ee_rotation", "follow_right_gripper"]], axis=1)
                real_poses = np.concatenate([pose_dicts[key] for key in ["follow_left_ee_cartesian_pos", "follow_left_ee_rotation", "follow_left_gripper", "follow_right_ee_cartesian_pos", "follow_right_ee_rotation", "follow_right_gripper"]], axis=1)
                # TODO: ADD trim stationary in the future

                ### process video
                video_frames = {}
                for key in ["face_view", "left_wrist_view", "right_wrist_view"]:
                    video_path = os.path.join(episode_path, FILE_CAME_MAPPING[key])
                    frames = decode_video_torchvision(video_path) # TODO: Upgrade video decoder to faster method
                    frames = einops.rearrange(frames, 't c h w -> t h w c')
                    video_frames[key] = frames # uint8

                ground_truths = []
                model_predicts = []
                
                # Sliding window approach: reuse action chunks
                # Get action_horizon from first inference result (dynamic)
                current_chunk = None
                chunk_step_idx = 0
                action_horizon = None
                
                for i in range(len(pred_poses) - 50):  # Leave buffer for unknown action_horizon
                    # When chunk is exhausted or not yet initialized, run inference
                    if current_chunk is None or chunk_step_idx >= action_horizon:
                        obs = {
                            'images': {
                                'left_wrist_view': video_frames["left_wrist_view"][i],
                                'face_view': video_frames["face_view"][i],
                                'right_wrist_view': video_frames["right_wrist_view"][i]
                            },
                            'prompt': 'pick up the cup',
                            'state': real_poses[i]
                        }
                        
                        action_pred_result = policy.infer(obs)
                        current_chunk = action_pred_result['actions']  # shape: [action_horizon, action_dim]
                        
                        # Dynamically set action_horizon from first inference result
                        if action_horizon is None:
                            action_horizon = current_chunk.shape[0]
                            print(f"Detected action_horizon from model: {action_horizon}")
                        
                        chunk_step_idx = 0
                    
                    # Skip if we don't have enough future steps to compare
                    if i + action_horizon >= len(pred_poses):
                        break
                    
                    # Use current step from chunk
                    predicted_action = current_chunk[chunk_step_idx]
                    ground_truth_action = real_poses[i + action_horizon]  # Align with the end of predicted chunk
                    
                    model_predicts.append(predicted_action)
                    ground_truths.append(ground_truth_action)
                    
                    chunk_step_idx += 1

                ground_truths = np.concatenate(ground_truths, axis=0)
                model_predicts = np.concatenate(model_predicts, axis=0)
                dim = 14
                ground_truths = ground_truths.reshape(-1, dim)
                model_predicts = model_predicts.reshape(-1, dim)
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 4*dim))

                for i in range(dim):
                    plt.subplot(dim, 1, i + 1)

                    # plot every 10th action
                    plt.xticks(np.arange(0, len(ground_truths), step=10))

                    plt.plot(ground_truths[:, i], label='Ground Truth', color='blue')
                    plt.plot(model_predicts[:, i], label='Model Output', color='orange')
                    plt.title(f'Action Dimension {i + 1}')
                    plt.xlabel('Time Step')
                    plt.ylabel('Action Value')
                    plt.legend()

                plt.tight_layout()
                # Ensure output directory exists and save per-episode with timestamp to avoid overwrites
                out_dir = Path('openloop_results')
                out_dir.mkdir(parents=True, exist_ok=True)
                episode_name = Path(episode_path).name if 'episode_path' in locals() else 'episode'
                ts = time.strftime('%Y%m%d_%H%M%S')
                png_name = out_dir / f'openloop_{episode_name}_{ts}.png'
                npz_name = out_dir / f'openloop_{episode_name}_{ts}.npz'

                # Save raw arrays for later quantitative analysis
                # try:
                #     np.savez_compressed(npz_name, ground_truths=ground_truths, model_predicts=model_predicts)
                # except Exception:
                #     # fallback: save individual .npy files if compressed save fails
                #     np.save(out_dir / f'ground_truths_{episode_name}_{ts}.npy', ground_truths)
                #     np.save(out_dir / f'model_predicts_{episode_name}_{ts}.npy', model_predicts)

                plt.savefig(str(png_name))
                print(f"Saved comparison png to {png_name} and arrays to {npz_name}")
                #import pdb; pdb.set_trace()





if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO, force=True)
    test_argv = [
        "policy:checkpoint",
        # "--policy.config=weibolu_64_act40",
        # "--policy.dir=/root/openpi/checkpoints/weibolu_64_act40/weibolu_64_act40/29999",
        "--policy.config=weibolu_41_1205_act40x2",
        "--policy.dir=/root/openpi/checkpoints/weibolu_41_1205_act40x2/weibolu_41_1205_act40x2/49999",

    ]

    main(tyro.cli(Args, args=test_argv))