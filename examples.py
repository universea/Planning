"""
examples.py

Path planning algorithm examples.

Author: universea

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from planning.pp2d.env import Environments

from planning.pp2d.rrt import RRT
from planning.pp2d.rrt_connect import RRTConnect
from planning.pp2d.rrt_star import RRTStar

if __name__ == '__main__':

    envs = Environments()

    x_start = (2, 2)  # Starting node
    x_goal = (300, 400)  # Goal node

    # generate map
    #env = envs.generate_random_environment(1000,1000,0.01)
    env = envs.generate_custom_environment(width=500, height=500, num_obstacles=10, min_radius=20, max_radius=50,\
                                           min_side_length=20, max_side_length=50)

    #planner = RRT(env, x_start, x_goal, 0.5, 0.05, 10000)
    #planner = RRTConnect(env, x_start, x_goal, 0.8, 0.05, 5000)
    planner = RRTStar(env, x_start, x_goal, 10, 0.10, 20, 10000)

    path = planner.planning()



    if path:

        height, width = env.shape
        map_image = np.zeros((height, width, 3), dtype=np.uint8)

        # 将可行驶区域设置为白色
        map_image[env == 0] = (255, 255, 255)

        # 将障碍物区域设置为黑色/暗色
        map_image[env >= 1] = (150, 150, 150)

        # draw path
        for i in range(len(path) - 1):
            x1, y1 = int(path[i][0]), int(path[i][1])
            x2, y2 = int(path[i+1][0]), int(path[i+1][1])
            cv2.line(map_image, (x1, y1), (x2, y2), (255, 0, 0), thickness=1)

        # draw start point and end point
        map_image[int(x_start[1]), int(x_start[0])] = (0, 0, 255)  # Blue
        map_image[int(x_goal[1]), int(x_goal[0])] = (0, 255, 0)    # Green

        cv2.imwrite('path.png',map_image)

        plt.imshow(map_image)
        plt.title("Map with Path")
        plt.show()
    else:
        print("No Path Found!")
