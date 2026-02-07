# inhand_env.py (Final Corrected Version)
import os
import numpy as np
import mujoco
import mujoco.viewer as mjv
import gymnasium as gym
from gymnasium import spaces
from enum import Enum, auto
from scipy.spatial.transform import Rotation as R


from simulation import Simulation

MAX_EPISODE_STEPS = 500

class MacroAction(Enum):
    TWIST_CW_SMALL = 0
    TWIST_CW_LARGE = 1
    TWIST_CCW_SMALL = 2
    TWIST_CCW_LARGE = 3
    GRIP_TIGHTER = 4
    GRIP_LOOSER = 5
    REPOSITION = 6

class CanRotateEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, render_mode=None):
        super(CanRotateEnv, self).__init__()
        
        # Initialize simulation and get object IDs
        self.sim = Simulation(
            scene_path=os.path.join(os.path.dirname(__file__), "scene.xml"),
            output_dir="rl_output"
        )
        self.sim.load()
        self.obj_body_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY, "obj1") #
        self.site_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site") #
        self.sim.ids_by_name(["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"], mujoco.mjtObj.mjOBJ_JOINT, 'arm') #
        self.sim.ids_by_name(["1", "0", "2", "3", "5", "4", "6", "7", "9", "8", "10", "11", "12", "13", "14", "15"], mujoco.mjtObj.mjOBJ_JOINT, 'hand') #
        self.sim.actuators_for_joints('arm') #
        self.sim.actuators_for_joints('hand') #
        self.can_geom_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_GEOM, "obj1")
        self.fingertip_geom_ids = {
            mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_GEOM, "fingertip"),
            mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_2"),
            mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_GEOM, "thumb_fingertip"),
        }
        
        self.can_geom_ids = set(
            gid for gid in range(self.sim.model.ngeom)
            if self.sim.model.geom_bodyid[gid] == self.obj_body_id
        )
        
        self.start_yaw = self._get_yaw_rad()               
        self.target_delta_rad = np.deg2rad(90.0)           
        self.target_yaw = self.start_yaw + self.target_delta_rad
        # Initial error to the goal
        curr_yaw = self._get_yaw_rad()
        self.prev_err = abs(self.normalize_angle(self.target_yaw - curr_yaw))

        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(MacroAction))
        obs_size = len(self.sim.hand_joint_ids) + 7
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        self.render_mode = render_mode
        if render_mode == "human":
            self.viewer = mjv.launch_passive(self.sim.model, self.sim.data) 
        else:
            self.viewer = None
        self.step_count = 0

    def _get_obs(self):
        finger_qpos = np.array([self.sim.data.qpos[self.sim.model.jnt_qposadr[j]] for j in self.sim.hand_joint_ids]) #
        obj_jnt_adr = self.sim.model.body_jntadr[self.obj_body_id] #
        obj_qpos_adr = self.sim.model.jnt_qposadr[obj_jnt_adr] #
        object_pose = self.sim.data.qpos[obj_qpos_adr : obj_qpos_adr + 7] #
        return np.concatenate([finger_qpos, object_pose])
    
    def _get_yaw_rad(self):
        w, x, y, z = self.sim.data.xquat[self.obj_body_id]
        quat = [x, y, z, w]
        r = R.from_quat(quat)
        yaw_rad = r.as_euler("xyz", degrees=False)[1]
        return yaw_rad
    
    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _calculate_reward(self):
        # --- Rotation Reward ---
        # Get current yaw and normalize
        curr_yaw = self.normalize_angle(self._get_yaw_rad())
        target_yaw = self.normalize_angle(self.target_yaw)
        # Compute error to target
        curr_err = abs(self.normalize_angle(target_yaw - curr_yaw))
        # Compute progress since last step
        progress = self.prev_err - curr_err
        # Ignore tiny progress to prevent jitter rewards
        epsilon = 1e-3
        if abs(progress) < epsilon:
            progress = 0.0
        # Normalize reward so full 90° rotation gives +1
        rotation_reward = progress / self.target_delta_rad
        rotation_reward = np.clip(rotation_reward, -1.0, 1.0)
        # Update previous error
        self.prev_err = curr_err

        # --- Contact Reward ---
        # cube_contact_count = self.cube_contact_count()
        # # Continuous reward: linearly scale from 0 (no contact) to 1 (all fingers)
        # contact_reward = np.clip(cube_contact_count / 22.0, 0.0, 1.0)

        # --- Drop penalty ---
        can_z_pos = self.sim.data.xpos[self.obj_body_id][2] #
        palm_z_pos = self.sim.data.site_xpos[self.site_id][2] #
        drop_penalty = 0.0
        if can_z_pos < (palm_z_pos - 0.05):
            drop_penalty = -100.0

        # --- Combine all components ---
        total_reward = (rotation_reward * 100.0) + drop_penalty

        return total_reward


    def _is_terminated(self):
        can_z_pos = self.sim.data.xpos[self.obj_body_id][2] #
        palm_z_pos = self.sim.data.site_xpos[self.site_id][2] #
        return can_z_pos < (palm_z_pos - 0.05) or self.step_count > MAX_EPISODE_STEPS

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        
        mujoco.mj_resetData(self.sim.model, self.sim.data)

        target_pos_up = np.array([0.4, 0.0, .5]) #
        target_euler_up = np.array([0, 0, 0]) #
        q_palm_up = self.sim.desired_qpos_from_ik(self.site_id, target_pos_up, target_euler_up) #
        self.sim.set_joint_positions(self.sim.arm_joint_ids, q_palm_up) #
        for i, act_id in enumerate(self.sim.arm_act_ids):
            self.sim.data.ctrl[act_id] = q_palm_up[i] #
        
        mujoco.mj_forward(self.sim.model, self.sim.data)

        palm_surface_pos = self.sim.data.site_xpos[self.site_id].copy() #
        object_start_pos = palm_surface_pos + np.array([0.011, -0.03, 0.075]) #
        obj_jnt_adr = self.sim.model.body_jntadr[self.obj_body_id] #
        obj_qpos_adr = self.sim.model.jnt_qposadr[obj_jnt_adr] #
        self.sim.data.qpos[obj_qpos_adr : obj_qpos_adr + 3] = object_start_pos #
        self.sim.data.qpos[obj_qpos_adr + 3 : obj_qpos_adr + 7] = [1, 0, 0, 0] #

        mujoco.mj_forward(self.sim.model, self.sim.data)

        for _ in range(20):
            mujoco.mj_step(self.sim.model, self.sim.data) #

        q_open_angles = np.array([1.0, 0.3, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.3, 1.0, 1.3, 1.0, 0.8, 1.3, 0.8, 0.5]) #
        self.sim.set_joint_positions(self.sim.hand_joint_ids, q_open_angles) #
        for i, act_id in enumerate(self.sim.hand_act_ids):
            self.sim.data.ctrl[act_id] = q_open_angles[i] #

        mujoco.mj_forward(self.sim.model, self.sim.data)

        self.start_yaw = self._get_yaw_rad()               
        self.target_delta_rad = np.deg2rad(90.0)           
        self.target_yaw = self.start_yaw + self.target_delta_rad
        # Initial error to the goal
        curr_yaw = self._get_yaw_rad()
        self.prev_err = abs(self.normalize_angle(self.target_yaw - curr_yaw))

        if self.render_mode != "headless":
            self.viewer.sync()
        
        return self._get_obs(), {}
    
    def cube_contact_count(self) -> int:
        count = 0
        for i in range(self.sim.data.ncon):
            c = self.sim.data.contact[i]
            if c.geom1 in self.can_geom_ids or c.geom2 in self.can_geom_ids:
                count += 1
        return count
    
    def apply_macro_action(self, macro_action: "MacroAction"):
        # Tunable scalar gains
        SMALL_TWIST = 0.015
        LARGE_TWIST = 0.035
        GRIP_DELTA  = 0.015
        REPOS_DELTA = 0.01

        # Create 16D abstract delta vector
        delta = np.zeros(16)

        if macro_action == MacroAction.TWIST_CW_SMALL:
            delta[0:4]  =  SMALL_TWIST   # index
            delta[4:8]  = -SMALL_TWIST   # middle
            delta[8:12] = -SMALL_TWIST   # ring
            delta[12:16]=  SMALL_TWIST   # thumb

        elif macro_action == MacroAction.TWIST_CW_LARGE:
            delta[0:4]  =  LARGE_TWIST
            delta[4:8]  = -LARGE_TWIST
            delta[8:12] = -LARGE_TWIST
            delta[12:16]=  LARGE_TWIST

        elif macro_action == MacroAction.TWIST_CCW_SMALL:
            delta[0:4]  = -SMALL_TWIST
            delta[4:8]  =  SMALL_TWIST
            delta[8:12] =  SMALL_TWIST
            delta[12:16]= -SMALL_TWIST

        elif macro_action == MacroAction.TWIST_CCW_LARGE:
            delta[0:4]  = -LARGE_TWIST
            delta[4:8]  =  LARGE_TWIST
            delta[8:12] =  LARGE_TWIST
            delta[12:16]= -LARGE_TWIST

        elif macro_action == MacroAction.GRIP_TIGHTER:
            delta[:] = GRIP_DELTA

        elif macro_action == MacroAction.GRIP_LOOSER:
            delta[:] = -GRIP_DELTA

        elif macro_action == MacroAction.REPOSITION:
            delta[8:12] = REPOS_DELTA     # ring MCP-PIP-DIP-fingertip
            delta[12:16] = -REPOS_DELTA   # thumb MCP-PIP-DIP-fingertip

        return delta

    def step(self, action: "MacroAction"):
        # action is a MacroAction enum
        delta = self.apply_macro_action(action)

        current_angles = np.array([
            self.sim.data.qpos[self.sim.model.jnt_qposadr[j]]
            for j in self.sim.hand_joint_ids
        ])

        target_angles = current_angles + delta
        self.sim.move_gripper_to_angles(target_angles, 0.5)

        if self.render_mode != "headless":
            self.viewer.sync()

        self.step_count += 1
        
        observation = self._get_obs()
        reward = self._calculate_reward()
        terminated = self._is_terminated()
        truncated = self.step_count >= MAX_EPISODE_STEPS

        return observation, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch(self.sim.model, self.sim.data)
            
            # Check if the viewer is still active before trying to sync
            try:
                if self.viewer.is_running():
                    self.viewer.sync()
                else:
                    # If the user closed the window, we must handle it
                    self.close() # Properly close the viewer resources
                    self.viewer = mujoco.viewer.launch(self.sim.model, self.sim.data) # And re-launch it
            except Exception:
                # This can happen if the viewer was closed abruptly
                self.viewer = mujoco.viewer.launch(self.sim.model, self.sim.data)
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None