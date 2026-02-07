## Objective

In this project, the objective is to create a Reinforcement Learning agent capable of efficiently rotating a cube by 90 degrees in the robotic hand. Our task is to implement the Q-Learning algorithm to achieve this goal.

## Action

I designed a discrete macro-action space containing seven high-level actions that control the robotic hand. These actions abstract away the low-level torques normally required for dexterous manipulation and instead provide a simplified interface for Q-Learning. The macro actions are:

```python
class MacroAction(Enum):
		TWIST_CW_SMALL = 0
		TWIST_CW_LARGE = 1
		TWIST_CCW_SMALL = 2
		TWIST_CCW_LARGE = 3
		GRIP_TIGHTER = 4
		GRIP_LOOSER = 5
		REPOSITION = 6
```

---

The function below maps each discrete macro action to a 16-dimensional joint-delta vector. Each block of four values corresponds to one finger (index, middle, ring, thumb). The deltas are added to the current joint positions and then passed to the simulation, allowing the agent to control the hand at a high level. The chosen deltas were tuned by hand to ensure that the motions were gentle, repeatable, and physically plausible.

```python
def apply_macro_action(self, macro_action: "MacroAction"):
		# Tunable scalar gains
		SMALL_TWIST = 0.015
		LARGE_TWIST = 0.035
		GRIP_DELTA  = 0.015
		REPOS_DELTA = 0.01
		
		# Create 6D abstract delta vector
		delta = np.zeros(16)
		
		if macro_action == MacroAction.TWIST_CW_SMALL:
				delta[0:4]  =  SMALL_TWIST   # index
				delta[4:8]  = -SMALL_TWIST   # middle
				delta[8:12] = -SMALL_TWIST   # ring
				delta[12:16]=  SMALL_TWIST   # thumb
				
		elif macro_action == MacroAction.TWIST_CW_LARGE:
				delta[0:4]  =  LARGE_TWIST
				delta[4:8]  = -LARGE_TWIST
				delta[8:12] = -LARGE_TWIST
				delta[12:16]=  LARGE_TWIST
		
		elif macro_action == MacroAction.TWIST_CCW_SMALL:
				delta[0:4]  = -SMALL_TWIST
				delta[4:8]  =  SMALL_TWIST
				delta[8:12] =  SMALL_TWIST
				delta[12:16]= -SMALL_TWIST
		
		elif macro_action == MacroAction.TWIST_CCW_LARGE:
				delta[0:4]  = -LARGE_TWIST
				delta[4:8]  =  LARGE_TWIST
				delta[8:12] =  LARGE_TWIST
				delta[12:16]= -LARGE_TWIST
		
		elif macro_action == MacroAction.GRIP_TIGHTER:
				delta[:] = GRIP_DELTA
		
		elif macro_action == MacroAction.GRIP_LOOSER:
				delta[:] = -GRIP_DELTA
		
		elif macro_action == MacroAction.REPOSITION:
				delta[8:12] = REPOS_DELTA     # ring MCP-PIP-DIP-fingertip
				delta[12:16] = -REPOS_DELTA   # thumb MCP-PIP-DIP-fingertip
		
		return delta
```

---

## State

The state representation is kept deliberately low-dimensional to make Q-Learning feasible. It consists of: Angle bin (0–35), Angular velocity bin (5 categories), and Contact bin (4 categories). This yields a total of 720 possible states.

```python
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
				Converts (angle, angle vel, contact) into a single
				integer index for Q-table.
				State count = 36 * 5 * 4 = 720
				"""
				return (
						self.angle +
						N_ANGLE * self.angvel.value +
						N_ANGLE * N_ANGVEL * self.contact.value
				)
```

These functions below convert continuous simulation quantities (angle, rotational velocity, and contact count) into discrete categories. This discretization is necessary because Q-learning operates over a discrete state space. The cube’s orientation is extracted from its quaternion, converted to yaw, and mapped to a 36-bin representation. Angular velocity along the Z-axis is categorized into five bins. Contact count is binned into four levels based on thresholds chosen empirically.

```python
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
```

---

```python
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
```

---

```python
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
```

---

Now we are able to get the current state at any given point in time

```python
# ------------------------------
# GET CURRENT STATE INDEX
# ------------------------------
def get_current_state_index() -> int:
    return State(
        angle = discretize_angle(),
        angvel = discretize_angvel(),
        contact = discretize_contact()
    ).to_index()
```

## Reward

The reward function is designed to promote three behaviors:

1. **Rotating the cube** — the primary objective.
    
    The reward scales with angular velocity around the Z-axis. Faster spin toward the target direction yields higher reward.
    
2. **Maintaining a stable grasp** — essential for consistent manipulation.
    
    Stability is encouraged through a survival reward that decreases as the cube drifts farther from the palm.
    
3. **Maintaining surface contact** — prevents dropping or losing control.
    
    A normalized contact reward ensures the agent learns to keep multiple fingertips in contact while rotating.
    

Rotation reward is weighted more heavily than the others to keep the learning objective focused on rotation rather than simply gripping or repositioning.

```python
def _calculate_reward(self):
    # --- Rotation Reward ---
    obj_vel = np.zeros(6)
    mujoco.mj_objectVelocity(
        self.sim.model, self.sim.data, mujoco.mjtObj.mjOBJ_BODY,
        self.obj_body_id, obj_vel, 0
    )
    angular_velocity_z = obj_vel[5]
    # Normalize angular velocity to roughly [-1, 1] based on max observed ±10 rad/s
    normalized_angvel = np.clip(angular_velocity_z / 10.0, -1.0, 1.0)
    rotation_reward = normalized_angvel

    # --- Survival / Position Reward ---
    can_pos = self.sim.data.xpos[self.obj_body_id]
    palm_pos = self.sim.data.site_xpos[self.site_id]
    distance_from_palm = np.linalg.norm(can_pos - palm_pos)
    # Encourage staying near the palm, scale down to match rotation reward
    MIN_DISTANCE = 0.07  # best-case hold
    MAX_DISTANCE = 0.09  # start of slip
    # linear scaling: 1.0 at MIN_DISTANCE, 0.0 at MAX_DISTANCE
    survival_reward = np.clip((MAX_DISTANCE - distance_from_palm) / (MAX_DISTANCE - MIN_DISTANCE), 0.0, 1.0)

    # --- Contact Reward ---
    cube_contact_count = self.cube_contact_count()
    # Continuous reward: linearly scale from 0 (no contact) to 1 (all fingers)
    contact_reward = np.clip(cube_contact_count / 22.0, 0.0, 1.0)

    # --- Combine all components ---
    total_reward = (rotation_reward * 2.0) + (survival_reward * 1.0) + (contact_reward * 1.0)

    return total_reward
```

---

## The Q-learning algorithm

These hyperparameters were chosen based on common Q-learning defaults and constraints of the problem. A higher learning rate (α = 0.1) allows fast adaptation, and a moderate discount rate (γ = 0.9) encourages long-term returns while still valuing immediate rotational progress. Exploration begins at ε = 0.1 and decays slowly to ensure the agent continues sampling actions that may lead to meaningful discoveries in later training.

```python
# Hyperparameters
ALPHA = 0.1
GAMMA = 0.9
EPSILON_START = 0.1
EPISODES = 10000
MAX_EPISODE_STEPS = 400
```

Main Q-learning training loop:

```python
# -------------------------
# Main Q-learning training loop
# -------------------------
def train():
    if not hasattr(env.action_space, 'n'):
        raise RuntimeError("Expected env.action_space to be Discrete macro-actions.")

    NUM_ACTIONS = env.action_space.n
    Q = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.float64)

    epsilon = EPSILON_START

    start_time = time.time()
    for ep in range(EPISODES):
        env.reset()
        s_idx = get_current_state_index()
        episode_return = 0.0
       
        print(f"Episode {ep+1}/{EPISODES}")
        for _ in range(MAX_EPISODE_STEPS):
            a = epsilon_greedy(Q, s_idx, epsilon)

            # step in env (action must be integer index for Discrete macro actions)
            for _ in range(15):
                _, reward, terminated, truncated, _ = env.step(MacroAction(a))
                episode_return += reward

            # compute next state index
            next_s_idx = get_current_state_index()

            # Q-learning update
            max_next_Q = np.max(Q[next_s_idx])
            Qsa = Q[s_idx, a]
            Q[s_idx, a] = Qsa + ALPHA * (reward + GAMMA * max_next_Q - Qsa)

            s_idx = next_s_idx

            if terminated or truncated:
                break

        # optionally decay epsilon (simple schedule)
        # keep a small floor to ensure some exploration
        if ep % 100 == 0 and ep > 0:
            epsilon = max(0.01, EPSILON_START * (1.0 - (ep / EPISODES)))

        print(f"Episode {ep+1}/{EPISODES}  Return={episode_return:.2f}  epsilon={epsilon:.3f}  elapsed={elapsed:.1f}s")

    # save Q table
    np.save("q_table.npy", Q)
    print("Training finished. Q table saved as q_table.npy")
    env.close()
   
if __name__ == "__main__":
    train()
```

## Training history & progress

Here were the first 20 episodes of the training:

```python
Episode 1/10000  Return=327.50  epsilon=0.100  elapsed=17.8s
Episode 2/10000  Return=320.50  epsilon=0.100  elapsed=35.0s
Episode 3/10000  Return=350.51  epsilon=0.100  elapsed=51.7s
Episode 4/10000  Return=348.99  epsilon=0.100  elapsed=68.6s
Episode 5/10000  Return=319.28  epsilon=0.100  elapsed=84.9s
Episode 6/10000  Return=332.97  epsilon=0.100  elapsed=102.2s
Episode 7/10000  Return=294.94  epsilon=0.100  elapsed=119.7s
Episode 8/10000  Return=334.47  epsilon=0.100  elapsed=136.8s
Episode 9/10000  Return=334.09  epsilon=0.100  elapsed=153.8s
Episode 10/10000  Return=330.38  epsilon=0.100  elapsed=170.6s
Episode 11/10000  Return=343.83  epsilon=0.100  elapsed=188.5s
Episode 12/10000  Return=268.91  epsilon=0.100  elapsed=205.3s
Episode 13/10000  Return=343.41  epsilon=0.100  elapsed=222.2s
Episode 14/10000  Return=261.03  epsilon=0.100  elapsed=238.4s
Episode 15/10000  Return=215.25  epsilon=0.100  elapsed=254.7s
Episode 16/10000  Return=266.23  epsilon=0.100  elapsed=271.6s
Episode 17/10000  Return=353.29  epsilon=0.100  elapsed=288.1s
Episode 18/10000  Return=334.97  epsilon=0.100  elapsed=304.8s
Episode 19/10000  Return=266.01  epsilon=0.100  elapsed=322.9s
Episode 20/10000  Return=335.79  epsilon=0.100  elapsed=340.1s
```

and here are the last 20 episodes of the training:

```python
Episode 981/10000  Return=417.81  epsilon=0.083  elapsed=30898.7s
Episode 982/10000  Return=406.29  epsilon=0.083  elapsed=30947.6s
Episode 983/10000  Return=355.86  epsilon=0.083  elapsed=30965.6s
Episode 984/10000  Return=452.38  epsilon=0.083  elapsed=30984.8s
Episode 985/10000  Return=456.69  epsilon=0.083  elapsed=31003.9s
Episode 986/10000  Return=396.38  epsilon=0.083  elapsed=31023.8s
Episode 987/10000  Return=462.81  epsilon=0.083  elapsed=31042.1s
Episode 988/10000  Return=446.60  epsilon=0.083  elapsed=31061.7s
Episode 989/10000  Return=442.85  epsilon=0.083  elapsed=31080.7s
Episode 990/10000  Return=445.37  epsilon=0.083  elapsed=31099.5s
Episode 991/10000  Return=442.85  epsilon=0.083  elapsed=31118.9s
Episode 992/10000  Return=442.85  epsilon=0.083  elapsed=31138.0s
Episode 993/10000  Return=442.85  epsilon=0.083  elapsed=31157.6s
Episode 994/10000  Return=381.91  epsilon=0.083  elapsed=31174.6s
Episode 995/10000  Return=459.77  epsilon=0.083  elapsed=31192.5s
Episode 996/10000  Return=363.88  epsilon=0.083  elapsed=31211.7s
Episode 997/10000  Return=438.93  epsilon=0.083  elapsed=31231.2s
Episode 998/10000  Return=357.38  epsilon=0.083  elapsed=31250.4s
Episode 999/10000  Return=427.35  epsilon=0.083  elapsed=31268.4s
Episode 1000/10000  Return=381.91  epsilon=0.083  elapsed=31286.0s
```

You can already start to see a general improvement

## Evaluation

During evaluation on rotation tasks of 90°, 180°, 270°, and 360°, the agent currently achieves only partial rotation, often failing to complete even the 90° target 80% of the time. On occasions it achieved 90, it took about 40 seconds. These data points were calculated on over 200 samples. This underperformance can be attributed to several factors.

Firstly, the model selected for evaluation was generated after only 1000 episodes of training due to time constraints. More training time would improve results.

Secondly, the present reward structure may not sufficiently incentivize progressive rotational improvement. The rotation reward is normalized by a maximum angular velocity of 10 rad/s, which may not reflect the actual speed range achievable by the agent’s actuators. If the robot rarely reaches velocities close to this normalization value, most rotational movements will produce relatively small reward magnitudes. Consequently, Q-learning receives weak signals for rotation compared to other actions such as adjusting grip tension.

Additionally, the contact reward, being proportional to the cube’s contact point count, may unintentionally over-penalize or overshadow rotation attempts. If rotation causes temporary drops in contact count, the agent may learn that staying still or loosening the grip produces more consistent rewards early in training. This can lead to a behavior policy that prioritizes grip adjustments rather than generating torque to rotate the object.

Addressing these design factors will better align the agent’s incentives with the task objectives and should lead to significantly improved performance on the 90°, 180°, 270°, and 360° rotation benchmarks. However, due to time constraints these factors were not implemented. Nevertheless, I believe my current state structure and macro actions are sufficient enough to accomplish this task. Although, I do wonder if there is a better way to encode rotation progress in the state. As of now, I’m using the angular velocity in the Z direction but there may be a better way.
