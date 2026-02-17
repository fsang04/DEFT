#!/usr/bin/env python3

import sys
from roahm_experiments.services import KortexSrvs
import rclpy
import numpy as np
from enum import Enum, auto
from typing import Optional
from collections import deque
import logging
import time
from numpy.linalg import norm, solve
from roahm_experiments.services import KortexSrvs
import pinocchio
import rclpy
import zipfile
from rclpy.node import Node
from std_msgs.msg import Empty


def degrees_to_radians(degrees_list):
    # Initialize an empty list to store the converted arrays
    radians_list = []
    
    # Loop through each array in the list
    for degrees in degrees_list:
        # Convert the degrees to radians and append to the list
        radians = np.radians(degrees)
        radians_list.append(radians)
    
    return radians_list

def wait_for_message(node, topic, msg_type):
    """Block until one message is received on the given topic."""
    future = rclpy.task.Future()
    def cb(msg):
        if not future.done():
            future.set_result(msg)
    sub = node.create_subscription(msg_type, topic, cb, 1)
    while rclpy.ok() and not future.done():
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_subscription(sub)
    return future.result()




rclpy.init()
kortex_srv = KortexSrvs()
node = rclpy.create_node("pause_example")
zip_path = '/home/baiyuew/Downloads/Sep16waypoint.zip'
intermediate_waypoint = np.deg2rad(np.array([325.56,16.28,178.43,291.64,15.29,284.82,52.85]))
intermediate_waypoint = intermediate_waypoint.astype(np.float32)
for i in range(10):

    file_name = f'Sep16waypoint/projected_kinova_case{i+1}_waypoints.npy'
    print('at start location ',i+1)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        with zf.open(file_name) as file:
            config_list = np.load(file)

    for j, config in enumerate(config_list):
        
        config = config.astype(np.float32)

        # Send the configuration to the Kortex service
        kortex_srv.goto(config)
        
        if j == 0:
            print('at initial waypoint, waiting for continue signal')
            time.sleep(1.0)
        #     node.get_logger().info("[PAUSE] Waiting for '/continue_signal'...")
        #     wait_for_message(node, "/continue_signal", Empty)
        #     node.get_logger().info("[PAUSE] Received '/continue_signal', continuing...")
        
    kortex_srv.goto(intermediate_waypoint)
    # print('at intermediate waypoint, waiting for continue signal')
    # # time.sleep(1.0)
    # node.get_logger().info("[PAUSE] Waiting for '/continue_signal'...")
    # wait_for_message(node, "/continue_signal", Empty)
    # node.get_logger().info("[PAUSE] Received '/continue_signal', continuing...")
node.destroy_node()
rclpy.shutdown()
# init_config = config_list[0]
# kortex_srv.goto(init_config)
# kortex_srv.goto(goal_config)