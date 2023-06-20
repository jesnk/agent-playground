import os

from gym.utils.ezpickle import EzPickle

from gym_robotics.envs.fetch_env import MujocoFetchEnv, MujocoPyFetchEnv
import numpy as np
# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join("fetch", "reach.xml")

# get current file path
print(os.path.realpath(__file__))


class MujocoPyFetchReachEnv(MujocoPyFetchEnv, EzPickle):
    def __init__(self, reward_type: str = "sparse", **kwargs):
        initial_qpos = {
            "robot0:slide0": 0.4049,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
        }
        MujocoPyFetchEnv.__init__(
            self,
            model_path=MODEL_XML_PATH,
            has_object=False,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)
        
    
    def compute_terminated(self, achieved_goal, desired_goal, info):
        # Get distance
        assert achieved_goal.shape == desired_goal.shape
        return np.linalg.norm(achieved_goal-desired_goal, axis=-1) < self.distance_threshold
    
    def compute_truncated(self, achievec_goal, desired_goal, info):
        """The environments will be truncated only if setting a time limit with max_steps which will automatically wrap the environment in a gym TimeLimit wrapper."""
        return False

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal."""
        return (
            np.linalg.norm(achieved_goal - desired_goal, axis=-1)
            < self.distance_threshold
        )


class MujocoFetchReachEnv(MujocoFetchEnv, EzPickle):
    def __init__(self, reward_type: str = "sparse", **kwargs):
        initial_qpos = {
            "robot0:slide0": 0.4049,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
        }
        MujocoFetchEnv.__init__(
            self,
            model_path=MODEL_XML_PATH,
            has_object=False,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)
