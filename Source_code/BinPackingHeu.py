#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 18:05:39 2023

@author: chunlongyu
"""
import ctypes
import platform
import numpy as np
import random

# call C++ programed BestFit heuristic

# Load the shared library
my_cpp_lib = ctypes.CDLL(r'D:\BestFit.dll')
# should be adjusted by user


# Define the function signature for the Print function
my_cpp_lib.BestFitRotate.argtypes = (ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float)
my_cpp_lib.BestFitRotate.restype = ctypes.POINTER(ctypes.c_float)
my_cpp_lib.FirstFitRotate.argtypes = (ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float)
my_cpp_lib.FirstFitRotate.restype = ctypes.POINTER(ctypes.c_float)


def BestFitRotate(bin_sizes, mac_width):
    num_bins = len(bin_sizes) // 2

    # Convert Python list to C array
    c_bin_sizes = (ctypes.c_float * len(bin_sizes))(*bin_sizes)

    # Call the C++ function
    item_coor_ptr = my_cpp_lib.BestFitRotate(c_bin_sizes, num_bins, mac_width)

    # Convert the returned pointer to a Python list
    item_coor_list = [item_coor_ptr[i] for i in range(num_bins*4)]

    return item_coor_list


