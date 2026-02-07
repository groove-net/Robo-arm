# inhand_test.py
import numpy as np
import time
import mujoco
from inhand_env import CanRotateEnv, MacroAction
from scipy.spatial.transform import Rotation as R
from enum import Enum
from dataclasses import dataclass
import torch
import torch.nn as nn
import numpy as np
import time
import statistics

# Create environment (choose "human" to see it)
env = CanRotateEnv(render_mode="human")

# State discretization sizes (must match your env/state design)
N_ANGLE = 36   # 36 bins of 10 degrees
N_ANGVEL = 5
N_CONTACT = 4
NUM_STATES = N_ANGLE * N_ANGVEL * N_CONTACT

# ------------------------------
# ENUMS FOR DISCRETIZED BINS
# ------------------------------
class AngVelBin(Enum):
    # 5 bins for angular velocity.
    FAST_CCW  = 0  # large negative
    SLOW_CCW  = 1  # small negative
    ZERO      = 2  # near zero
    SLOW_CW   = 3  # small positive
    FAST_CW   = 4  # large positive


class ContactBin(Enum):
    # Simple: cube held or not.
    OPEN = 0
    LOOSE = 1
    NORMAL = 2
    TIGHT = 3

# -----------------------------------------
# Angle discretization (−180°..180° → 0..35)
# -----------------------------------------
def discretize_angle() -> int:
    # MuJoCo gives quat as [w, x, y, z]
    w, x, y, z = env.sim.data.xquat[env.obj_body_id]

    # SciPy expects [x, y, z, w]
    quat_xyzw = [x, y, z, w]

    r = R.from_quat(quat_xyzw)
    yaw_deg = r.as_euler("xyz", degrees=True)[2]

    # Convert [-180, 180] → [0, 360]
    yaw_deg = (yaw_deg + 180) % 360

    return int(yaw_deg // 10)

# -----------------------------------------
# Angular velocity discretization
# -----------------------------------------
def discretize_angvel() -> AngVelBin:
    obj_vel = np.zeros(6)
    mujoco.mj_objectVelocity(env.sim.model, env.sim.data,
                             mujoco.mjtObj.mjOBJ_BODY,
                             env.obj_body_id,
                             obj_vel, 0)

    angvel = obj_vel[5]  # angular velocity around Z
    if angvel < -5.0: return AngVelBin.FAST_CCW
    if angvel < -0.5:  return AngVelBin.SLOW_CCW
    if angvel < 0.5:   return AngVelBin.ZERO
    if angvel < 5.0:  return AngVelBin.SLOW_CW
    return AngVelBin.FAST_CW

# -----------------------------------------
# Contact discretization
# -----------------------------------------
def discretize_contact() -> ContactBin:
    cube_contact_count = env.cube_contact_count()
    if cube_contact_count > 15:
        return ContactBin.TIGHT
    elif cube_contact_count > 9:
        return ContactBin.NORMAL
    elif cube_contact_count > 5:
        return ContactBin.LOOSE
    else:
        return ContactBin.OPEN
    
# ------------------------------
# STATE OBJECT
# ------------------------------
@dataclass(frozen=True)
class State:
    angle: int
    angvel: AngVelBin
    contact: ContactBin

    def to_index(self) -> int:
        """
        Converts (angle, angvel, contact) into a single integer index for Q-table.
        State count = 36 * 5 * 4 = 720
        """
        return (
            self.angle +
            N_ANGLE * self.angvel.value +
            N_ANGLE * N_ANGVEL * self.contact.value
        )
    
# DQN Architecture
class DQNetwork(nn.Module):
    def __init__(self, obs_dim, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        return self.net(x)
    
def test_q_learning(qtable):
    # LOAD Q-TABLE
    q_table = np.load(qtable, allow_pickle=False)
    print("Q-table shape:", q_table.shape)

    env.reset()
    done = False

    while not done:
        discrete_angle = discretize_angle()
        discrete_angvel = discretize_angvel()
        discrete_contact = discretize_contact()
        # print(f"Current state: [{discrete_angle}, {discrete_angvel}, {discrete_contact}]")
        state_idx = State(
            angle = discrete_angle,
            angvel = discrete_angvel,
            contact = discrete_contact
        ).to_index()
        # print(f"Current state idx: {state_idx}")
        action = np.argmax(q_table[state_idx])   # always pick best action
        # print(f"Current action: {action}\n")
        for _ in range(15):
            env.step(MacroAction(action))
        # Pause for observation
        # time.sleep(0.2)
    env.close()

def test_dqn(dqn):
    obs, _ = env.reset()

    obs_dim = len(obs)
    num_actions = env.action_space.n

    model = DQNetwork(obs_dim, num_actions)
    model.load_state_dict(torch.load(dqn))
    model.eval()

    done = False

    start = time.perf_counter()

    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = model(obs_t)
            action = int(torch.argmax(q_values, dim=1).item())

        # macro-actions
        for _ in range(10):
            next_obs, _, terminated, truncated, _ = env.step(MacroAction(action))
            obs = next_obs
            if terminated or truncated:
                done = True
                break

        # distance-to-goal termination
        curr_yaw = env.normalize_angle(env._get_yaw_rad())
        target_yaw = env.normalize_angle(env.target_yaw)
        curr_err = abs(env.normalize_angle(target_yaw - curr_yaw))

        if curr_err < 0.06:
            done = True

    elapsed = time.perf_counter() - start

    return elapsed

def run_benchmark(dqn, iterations=100):
    log_file = "dqn_test_times.log"
    times = []

    with open(log_file, "w") as f:
        for i in range(iterations):
            t = test_dqn(dqn)

            times.append(t)

            print(f"Run {i+1}: {t:.6f} seconds  |  Completed Rotation")
            f.write(f"Run {i+1}: {t:.6f} seconds  |  Completed Rotation\n")

    avg_time = statistics.mean(times)
    print(f"Average time over {iterations} runs: {avg_time:.6f} seconds")
    
    return avg_time

if __name__ == "__main__":
    run_benchmark("dqn_ep5000.pt")
    env.close()