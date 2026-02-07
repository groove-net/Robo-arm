This project details the development of a Reinforcement Learning agent using Q-Learning to rotate a cube in a robotic hand within a MuJoCo simulation environment.

---

# Robotic Hand Cube Rotation (Q-Learning)

This repository contains the implementation of a discrete Q-Learning agent designed to perform dexterous manipulation. The agent controls a robotic hand to rotate a cube by target increments by 90° using a set of high-level macro actions.

## Action Space: Discrete Macro Actions

To simplify the complex torque control of a dexterous hand, the agent uses a **Macro-Action Space**. These 7 actions map to specific 16-dimensional joint-delta vectors across the index, middle, ring, and thumb fingers.

- `TWIST_CW_SMALL / LARGE`: Rotates the object clockwise.
- `TWIST_CCW_SMALL / LARGE`: Rotates the object counter-clockwise.
- `GRIP_TIGHTER / LOOSER`: Adjusts the inward pressure on the cube.
- `REPOSITION`: Adjusts the ring finger and thumb to reset the grasp.

## State Representation

The continuous environment is discretized into a state space of **720 possible states** to make tabular Q-Learning computationally feasible:

- **Angle Bin (36 bins)**: Maps -180° to 180° into 10° increments.
- **Angular Velocity (5 bins)**: Categorizes Z-axis rotation from `FAST_CCW` to `FAST_CW`.
- **Contact Bin (4 bins)**: Categorizes the number of contact points between the hand and cube (`OPEN`, `LOOSE`, `NORMAL`, `TIGHT`).

## Reward Structure

The reward function balances three competing objectives:

1. **Rotation Reward (Weight: 2.0)**: Based on normalized angular velocity around the Z-axis.
2. **Survival Reward (Weight: 1.0)**: Penalizes the agent if the cube drifts away from the center of the palm.
3. **Contact Reward (Weight: 1.0)**: Encourages keeping multiple fingertips in contact with the object.

## Training & Hyperparameters

The agent was trained using the following parameters:

- **Alpha (α):** 0.1 (Learning Rate)
- **Gamma (γ):** 0.9 (Discount Factor)
- **Epsilon (ε):** 0.1 starting value with decay.

---

## Getting Started

1. Ensure `mujoco` and `scipy` are installed.
2. Run `python inhand_train.py` to start the Q-Learning loop.
3. The resulting policy is saved as `q_table.npy`.
4. Then Run `python inhand_test.py` to evaluate and see the policy in action
