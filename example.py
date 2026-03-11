import gymnasium as gym
import fruit_gym

import cv2

# Create environment
env = gym.make("PickMultiStrawbEnv", render_mode="human")

# Reset and visualize
obs, info = env.reset()

# Run for 500 steps with random actions
for step in range(500):
    action = env.action_space.sample()  # Random action

    # print(action) # list of 7 atributes -> all robot dofs

    obs, reward, terminated, truncated, info = env.step(action)

    # print(obs["state"])

    # tcp_pose [x, y, z, qx, qy, qz, qw]
    #   dz      dy      dx      droll   dpitch  dyaw    dgrasp
    # tcp_vel [vx, vy, vz, ωx, ωy, ωz]
    # gripper_pos [number] means how much is the gripper open
    # gripper_vec
    #Vertical approach (your case)
    # gripper_vec = [0., 0., 1., 0.]  # Pointing up
    # # Horizontal approach (side grasp)
    # gripper_vec = [1., 0., 0., 0.]  # Pointing forward
    # # Top-down approach
    # gripper_vec = [0., 0., -1., 0.]  # Pointing down
    # # Diagonal approach
    # gripper_vec = [0.707, 0., 0.707, 0.]  # 45° angle

    # print(obs["images"]["wrist2"])
    # wrist2 upper camera
    # wrist1 camera bellow
    # cv2.imshow("wrist1", obs["images"]["wrist2"])
    # cv2.waitKey(1)
    # print(obs["images"]["wrist2"].shape)

    # print(info)

    print(reward)

    env.render()

    if terminated or truncated:
        obs, info = env.reset()

env.close()
