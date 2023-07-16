"""
rrt.py

This file implements the RRT (Rapidly-Exploring Random Tree) algorithm in 2D space.
It generates a feasible path for a robot to navigate from a start point to a goal point,
avoiding obstacles in the environment.

Original Author: Huiming Zhou

Modified by: universea

"""

import os
import sys
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt


class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None


class RRT:
    def __init__(self, env, s_start, s_goal, step_len, goal_sample_rate, iter_max):
        self.s_start = Node(s_start)
        self.s_goal = Node(s_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.iter_max = iter_max
        self.vertex = [self.s_start]

        self.env = env
        self.x_range = (0, self.env.shape[1])
        self.y_range = (0, self.env.shape[0])


    def is_collision(self, node_start, node_end):
        delta = self.step_len
        x_start, y_start = int(node_start.x), int(node_start.y)
        x_end, y_end = int(node_end.x), int(node_end.y)

        dx = abs(x_end - x_start)
        dy = abs(y_end - y_start)

        if dx == 0 and dy == 0:
            return self.env[y_start, x_start] >= 1

        if dx > dy:
            x, y = x_start, y_start
            x_step = 1 if x_end > x_start else -1
            y_step = dy / dx if y_end > y_start else -dy / dx

            while x != x_end:
                if self.env[int(round(y)), x] >= 1:
                    return True
                x += x_step
                y += y_step
        else:
            x, y = x_start, y_start
            y_step = 1 if y_end > y_start else -1
            x_step = dx / dy if x_end > x_start else -dx / dy

            while y != y_end:
                if self.env[y, int(round(x))] >= 1:
                    return True
                x += x_step
                y += y_step

        return False

    def planning(self):
        for i in range(self.iter_max):
            node_rand = self.generate_random_node(self.goal_sample_rate)
            node_near = self.nearest_neighbor(self.vertex, node_rand)
            node_new = self.new_state(node_near, node_rand)

            if node_new and not self.is_collision(node_near, node_new):
                self.vertex.append(node_new)
                dist, _ = self.get_distance_and_angle(node_new, self.s_goal)

                if dist <= self.step_len and not self.is_collision(node_new, self.s_goal):
                    self.new_state(node_new, self.s_goal)
                    return self.extract_path(node_new)

        return None

    def generate_random_node(self, goal_sample_rate):
        delta = 0.5

        if np.random.random() > goal_sample_rate:
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

        return self.s_goal

    @staticmethod
    def nearest_neighbor(node_list, n):
        return node_list[int(np.argmin([math.hypot(nd.x - n.x, nd.y - n.y)
                                        for nd in node_list]))]

    def new_state(self, node_start, node_end):
        dist, theta = self.get_distance_and_angle(node_start, node_end)

        dist = min(self.step_len, dist)
        node_new = Node((node_start.x + dist * math.cos(theta),
                         node_start.y + dist * math.sin(theta)))
        node_new.parent = node_start

        return node_new

    def extract_path(self, node_end):
        path = [(self.s_goal.x, self.s_goal.y)]
        node_now = node_end

        while node_now.parent is not None:
            node_now = node_now.parent
            path.append((node_now.x, node_now.y))

        return path

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)
