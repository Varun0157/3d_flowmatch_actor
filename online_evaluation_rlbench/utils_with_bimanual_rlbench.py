import os
import glob
import random

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import BimanualDiscrete, assert_action_shape
from rlbench.action_modes.arm_action_modes import BimanualEndEffectorPoseViaPlanning
from rlbench.backend.exceptions import InvalidActionError
from pyrep.errors import IKError, ConfigurationPathError
from pyrep.const import RenderMode

from modeling.encoder.text import fetch_tokenizers
from online_evaluation_rlbench.get_stored_demos import get_stored_demos


def task_file_to_task_class(task_file):
    import importlib

    name = task_file.replace(".py", "")
    class_name = "".join([w[0].upper() + w[1:] for w in name.split("_")])
    mod = importlib.import_module("rlbench.bimanual_tasks.%s" % name)
    mod = importlib.reload(mod)
    task_class = getattr(mod, class_name)
    return task_class


class Mover:

    def __init__(self, task, max_tries=1):
        self._task = task
        self._last_action = None
        self._max_tries = max_tries

    def __call__(self, action, collision_checking=False):
        # action is an array (2, 8)
        obs = None
        terminate = None
        reward = 0

        # Try to reach the desired pose without changing the gripper state
        target = action.copy()
        if self._last_action is not None:
            action[:, 7] = self._last_action[:, 7].copy() # copy gripper state
        for _ in range(self._max_tries):
            action_collision = np.ones((action.shape[0], action.shape[1]+1))
            action_collision[:, :-1] = action
            if collision_checking:
                action_collision[:, -1] = 0
            # Peract2 takes (right, left) action, but we predict (left, right)
            action_collision = action_collision[::-1]
            action_collision = action_collision.ravel()
            obs, reward, terminate = self._task.step(action_collision)

            # Check if we reached the desired pose (planner may be inaccurate)
            l_pos = obs.left.gripper_pose[:3]
            r_pos = obs.right.gripper_pose[:3]
            l_dist_pos = np.sqrt(np.square(target[0, :3] - l_pos).sum())
            r_dist_pos = np.sqrt(np.square(target[1, :3] - r_pos).sum())
            criteria = (l_dist_pos < 5e-3, r_dist_pos < 5e-3)

            if all(criteria) or reward == 1:
                break

        # Then execute with gripper action (open/close))
        action = target
        if (
            not reward == 1.0
            and self._last_action is not None
            and (  # if any gripper state has changed, re-execute
                action[0, 7] != self._last_action[0, 7]
                or action[1, 7] != self._last_action[1, 7]
            )
        ):
            action_collision = np.ones((action.shape[0], action.shape[1]+1))
            action_collision[:, :-1] = action
            if collision_checking:
                action_collision[:, -1] = 0
            action_collision = action_collision[::-1]
            action_collision = action_collision.ravel()
            obs, reward, terminate = self._task.step(action_collision)

        # Store the last action action for the gripper state
        self._last_action = action.copy()

        return obs, reward, terminate


class Actioner:

    def __init__(self, policy=None, backbone='clip'):
        self._policy = policy.cuda()
        self._policy.eval()
        self._instr = None
        self.tokenizer = fetch_tokenizers(backbone)

    def load_episode(self, descriptions):
        instr = [random.choice(descriptions)]
        self._instr = self.tokenizer(instr).cuda(non_blocking=True)

    def predict(self, rgbs, pcds, gripper, prediction_len=1):
        """
        Args:
            rgbs: (1, ncam, 3, H, W)
            pcds: (1, ncam, 3, H, W)
            gripper: (1, nhist, 2*8)
            prediction_len: int

        Returns:
            trajectory: (1, nhist, nhand=2, 8)
        """
        return self._policy(
            None,
            torch.full([1, prediction_len, 2], False).cuda(non_blocking=True),
            rgbs,
            None,
            pcds,
            self._instr,
            gripper.unflatten(-1, (2, -1)),  # (1, nhist, nhand=2, 8)
            run_inference=True
        )


class RLBenchEnv:

    def __init__(
        self,
        data_path,
        task_str=None,
        image_size=(256, 256),
        apply_rgb=False,
        apply_depth=False,
        apply_pc=False,
        headless=False,
        apply_cameras=("over_shoulder_left", "over_shoulder_right", "wrist_left", "wrist_right", "front"),
        collision_checking=False
    ):

        # setup required inputs
        self.data_path = data_path
        self.apply_cameras = apply_cameras

        # setup RLBench environments
        self.obs_config = self.create_obs_config(
            image_size, apply_rgb, apply_depth, apply_pc, apply_cameras
        )

        self.action_mode = BimanualMoveArmThenGripper(
            arm_action_mode=BimanualEndEffectorPoseViaPlanning(collision_checking=collision_checking),
            gripper_action_mode=HandoverDiscrete() if 'handover' in task_str else BimanualDiscrete()
        )
        self.env = Environment(
            self.action_mode, str(data_path), self.obs_config,
            headless=headless, robot_setup="dual_panda"
        )

    def get_rgb_pcd_gripper_from_obs(self, obs):
        """
        Return rgb, pcd, and gripper from a given observation
        :param obs: an Observation from the env
        :return: rgb, pcd, gripper
        """
        rgb = torch.stack([
            torch.tensor(obs.perception_data["{}_rgb".format(cam)]).float().permute(2, 0, 1) / 255.0
            for cam in self.apply_cameras
        ]).unsqueeze(0)  # 1, N, C, H, W
        pcd = torch.stack([
            torch.tensor(obs.perception_data["{}_point_cloud".format(cam)]).float().permute(2, 0, 1)
            for cam in self.apply_cameras
        ]).unsqueeze(0)  # 1, N, C, H, W
        # action is an array of length 16 = (7+1)*2
        gripper = torch.from_numpy(np.concatenate([
            obs.left.gripper_pose, [obs.left.gripper_open],
            obs.right.gripper_pose, [obs.right.gripper_open]
        ])).float().unsqueeze(0)  # 1, D

        return rgb, pcd, gripper

    def evaluate_task_on_multiple_variations(
        self,
        task_str,
        max_steps,
        actioner,
        max_tries=1,
        prediction_len=1,
        num_history=1
    ):
        self.env.launch()
        task_type = task_file_to_task_class(task_str)
        task = self.env.get_task(task_type)
        task_variations = glob.glob(
            os.path.join(self.data_path, task_str, "variation*")
        )
        task_variations = [
            int(n.split('/')[-1].replace('variation', ''))
            for n in task_variations
        ]

        var_success_rates = {}
        var_num_valid_demos = {}

        for variation in tqdm(task_variations):
            task.set_variation(variation)
            success_rate, valid, num_valid_demos = (
                self._evaluate_task_on_one_variation(
                    task_str=task_str,
                    task=task,
                    max_steps=max_steps,
                    variation=variation,
                    actioner=actioner,
                    max_tries=max_tries,
                    prediction_len=prediction_len,
                    num_history=num_history
                )
            )
            if valid:
                var_success_rates[variation] = success_rate
                var_num_valid_demos[variation] = num_valid_demos

        self.env.shutdown()

        total_valid = sum(var_num_valid_demos.values())
        if total_valid > 0:
            var_success_rates["mean"] = (
                sum(var_success_rates.values()) / total_valid
            )
        else:
            var_success_rates["mean"] = 0.0
            print("Warning: No valid demos completed for this task")

        return var_success_rates

    @torch.no_grad()
    def _evaluate_task_on_one_variation(
        self,
        task_str,  # this is str
        task,  # this instance of TaskEnvironment
        max_steps,
        variation,
        actioner,
        max_tries=1,
        prediction_len=50,
        num_history=1
    ):
        success_rate = 0
        total_reward = 0
        var_demos = get_stored_demos(
            amount=-1,
            dataset_root=self.data_path,
            variation_number=variation,
            task_name=task_str,
            random_selection=False,
            from_episode_number=0
        )

        num_valid_demos = 0
        for demo_id, demo in enumerate(var_demos):

            grippers = torch.Tensor([]).cuda(non_blocking=True)
            try:
                descriptions, obs = task.reset_to_demo(demo)
            except RuntimeError as e:
                print(f"Skipping demo {demo_id} due to reset error: {e}")
                continue
            num_valid_demos += 1
            actioner.load_episode(descriptions)

            move = Mover(task, max_tries=max_tries)
            max_reward = 0.0

            for step_id in range(max_steps):

                # Fetch the current observation and predict one action
                rgb, pcd, gripper = self.get_rgb_pcd_gripper_from_obs(obs)
                rgbs_input = rgb.cuda(non_blocking=True)
                pcds_input = pcd.cuda(non_blocking=True)
                gripper = gripper.cuda(non_blocking=True)
                grippers = torch.cat([grippers, gripper.unsqueeze(1)], 1)

                # Prepare proprioception history
                gripper_input = grippers[:, -num_history:]
                npad = num_history - gripper_input.shape[1]
                gripper_input = F.pad(
                    gripper_input, (0, 0, npad, 0), mode='replicate'
                )

                output = actioner.predict(
                    rgbs_input,
                    pcds_input,
                    gripper_input,
                    prediction_len=prediction_len
                )

                # Update the observation based on the predicted action
                try:
                    # Execute entire predicted trajectory step by step
                    actions = output[-1].cpu().numpy()
                    actions[..., -1] = actions[..., -1].round()

                    # execute
                    for action in actions:
                        obs, reward, _ = move(action, collision_checking=False)

                    max_reward = max(max_reward, reward)

                    if reward == 1:
                        success_rate += 1
                        break

                except (IKError, ConfigurationPathError, InvalidActionError, RuntimeError) as e:
                    print(task_str, demo, step_id, success_rate, e)
                    reward = 0

            total_reward += max_reward

            print(
                task_str,
                "Variation",
                variation,
                "Demo",
                demo_id,
                "Reward",
                f"{reward:.2f}",
                "max_reward",
                f"{max_reward:.2f}",
                f"SR: {success_rate}/{demo_id+1}",
                f"SR: {total_reward:.2f}/{demo_id+1}",
                "# valid demos", demo_id + 1
            )

        # Compensate for failed demos
        valid = num_valid_demos > 0

        return success_rate, valid, num_valid_demos

    def create_obs_config(
        self, image_size, apply_rgb, apply_depth, apply_pc, apply_cameras, **kwargs
    ):
        """
        Set up observation config for RLBench environment.
            :param image_size: Image size.
            :param apply_rgb: Applying RGB as inputs.
            :param apply_depth: Applying Depth as inputs.
            :param apply_pc: Applying Point Cloud as inputs.
            :param apply_cameras: Desired cameras.
            :return: observation config
        """
        # Define a config for an unused camera with all applications as False.
        unused_cams = CameraConfig()
        unused_cams.set_all(False)

        # Define a config for a used camera with the given image size and flags
        used_cams = CameraConfig(
            rgb=apply_rgb,
            point_cloud=apply_pc,
            depth=apply_depth,
            mask=False,
            image_size=image_size,
            render_mode=RenderMode.OPENGL3,  # note OPENGL3 for Peract2!
            **kwargs
        )

        # apply_cameras is a tuple with the names(str) of all the cameras
        camera_names = apply_cameras
        cameras = {}
        for name in camera_names:
            cameras[name] = used_cams

        obs_config = ObservationConfig(
            camera_configs=cameras,
            joint_forces=False,
            joint_positions=False,
            joint_velocities=True,
            task_low_dim_state=False,
            gripper_touch_forces=False,
            gripper_pose=True,
            gripper_open=True,
            gripper_matrix=True,
            gripper_joint_positions=True
        )

        return obs_config


class HandoverDiscrete(BimanualDiscrete):
    """
    A custom gripper action mode for the handover task.
    It forces one gripper to release so that the other grasps.
    """

    def action(self, scene, action):
        assert_action_shape(action, self.action_shape(scene.robot))
        if 0.0 > action[0] > 1.0:
            raise InvalidActionError(
                'Gripper action expected to be within 0 and 1.')

        if 0.0 > action[1] > 1.0:
            raise InvalidActionError(
                'Gripper action expected to be within 0 and 1.')

        right_open_condition = all(
            x > 0.9 for x in scene.robot.right_gripper.get_open_amount())

        left_open_condition = all(
            x > 0.9 for x in scene.robot.left_gripper.get_open_amount())

        right_current_ee = 1.0 if right_open_condition else 0.0
        left_current_ee = 1.0 if left_open_condition else 0.0

        right_action = float(action[0] > 0.5)
        left_action = float(action[1] > 0.5)

        if right_current_ee != right_action or left_current_ee != left_action:
            if not self._detach_before_open:
                self._actuate(scene, action)

        # Move objects between grippers
        if right_current_ee != right_action:
            if right_action == 0.0 and self._attach_grasped_objects:
                left_grasped_objects = scene.robot.left_gripper.get_grasped_objects()
                for g_obj in scene.task.get_graspable_objects():
                    if g_obj in left_grasped_objects:
                        scene.robot.left_gripper.release()
                        scene.robot.right_gripper.grasp(g_obj)
                    else:
                        scene.robot.right_gripper.grasp(g_obj)
            else:
                scene.robot.right_gripper.release()
        if left_current_ee != left_action:
            if left_action == 0.0 and self._attach_grasped_objects:
                right_grasped_objects = scene.robot.right_gripper.get_grasped_objects()
                for g_obj in scene.task.get_graspable_objects():
                    if g_obj in right_grasped_objects:
                        scene.robot.right_gripper.release()
                        scene.robot.left_gripper.grasp(g_obj)
                    else:
                        scene.robot.left_gripper.grasp(g_obj)
            else:
                scene.robot.left_gripper.release()

        if right_current_ee != right_action or left_current_ee != left_action:
            if self._detach_before_open:
                self._actuate(scene, action)
            if right_action == 1.0 or left_action == 1.0:
                # Step a few more times to allow objects to drop
                for _ in range(10):
                    scene.pyrep.step()
                    scene.task.step()
