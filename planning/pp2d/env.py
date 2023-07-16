import cv2
import numpy as np


class Environments:
    def __init__(self):
        # Initialize any necessary variables or resources here
        pass

    def generate_random_environment(self, width, height, obstacle_prob):
        random_env = np.random.choice([0, 1], size=(height, width), p=[1 - obstacle_prob, obstacle_prob])
        return random_env

    def generate_custom_environment(self, width, height, num_obstacles, min_radius, max_radius, min_side_length, max_side_length):
        custom_env = np.zeros((height, width))
        for _ in range(num_obstacles):
            if np.random.rand() < 0.5:  # Generate a circle obstacle
                center_x = np.random.randint(0, width)
                center_y = np.random.randint(0, height)
                radius = np.random.randint(min_radius, max_radius)
                cv2.circle(custom_env, (center_x, center_y), radius, 150, -1)
            else:  # Generate a square obstacle
                top_left_x = np.random.randint(0, width)
                top_left_y = np.random.randint(0, height)
                side_length = np.random.randint(min_side_length, max_side_length)
                cv2.rectangle(custom_env, (top_left_x, top_left_y), (top_left_x + side_length, top_left_y + side_length), 150, -1)
        return custom_env
