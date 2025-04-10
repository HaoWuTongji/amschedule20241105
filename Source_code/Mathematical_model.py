"""
@author: chunlongyu
modified by haowu 2024.10
"""
# ======= The mathematical model for Parallel AM Machines scheduling =======

# 1. Consider m parallel 3D printing machines and a set of parts. In this project, a part is represented by its outbox.

# 2. Each part has several eligible build orientaitons, and each orientation has the corresponding projection on the build platform, height, and supporting structure volumn

# 3. The batch processing time is the sum of the machine setup time, laser scanning time (related to the total part and supports volumn), and recoater moving time (related to the batch height)

# 4. The goal of the algorithm is to find the schedule that minimizes total number of tardy parts

# For more details please refer to the paper "Mathematical Models for Minimizing Total Tardiness on Parallel Additive Manufacturing Machines" by Yu et al., 2022, in press
# ==================================================================

import Visualizer as vs
from SIM_ALNS import *
from NSPM_MILP import *
import re
import random
import json

## Set random seed
random.seed()

path = "TestInstances/Calibration/5-20.txt"  # adjust by user
[types_mac, types_parts, num_mac, num_parts, v1, v2, S1, S2, T1, T2, D1, D2, L, W, HM, NL, y_slice, Dp, Ds, Tr, v, a, Kj, h, l, w, s] = readInstance(path)

d_list = GenerateDueDate(path, 0.6, 0.6, 1)
# d_list = GenerateDueDate(filePath, TF, RDD, RndSeed)
d = [x / 3600 for x in d_list]
# print(d)

## ====== Inputs to the Scheduling Algorithm ======
# types_mac,types_parts
# num_mac: Number of machines, or #mac
# num_parts: Number of parts, or #part
# L:  Length of the machine platform, 1 by #mac vector
# W:  Width of the machine platform,  1 by #mac vector
# HM: Height of the machine platform, 1 by #mac vector
# v:  Volumn of parts,                1 by #part vector

# Kj: Orientation of parts, a python dictionary mapping the (part id, orientation id) to orientation id.
#     The mapping here is NOT used actually, but the dictionary serves as a preferred stucture to STORE the parts' orientations.
#     E.g. (1,1):1 means that 1-th part's 1-th orientation is indexed as 1
#          (0,0):0 means that 0-th part's 0-th orientation is indexed as 0

# h:  Height of parts' build orientations, a python dictionary mapping the (part id, orientation id) tuple to height.
#     E.g., (0,0):5.0 means that the 0-th part's 0-th orientation has a height of 5
#           (3,1):8.0 means that the 8-th part's 1-th  orientation has a heigth of 8

# l:  Length of parts' build orientations, a python dictionary mapping the (part id, orietation id) tuple to length

# w:  Width of parts' build orientations, a python dictionary mapping the (part id, orietation id) tuple to width

# s:  Supporting structure volumn of parts' build orientations, a python dictionary mapping the (part id, orietation id) tuple to the volumn of supports

# t_run: Max running time of the scheduling algorithm (The algorithm stops when it finds the optimal solution, or when it reaches the time limit, at this case, it tries to return a feasible solution)


t_run = 7200  # Maximum running time of the Scheduling algorithm

model = NSPM_ML_TJ_exp(num_mac, num_parts, v1, v2, S1, S2, T1, T2, D1, D2, L, W, HM, NL, y_slice, Dp, Ds, Tr, v, a, Kj, h, l, w, s, t_run, d)
# minimizes total number of tardy parts

# Extract the data from the Scheduling algorithm
Xs, Ys, Os, Ks, Ls, Ws, Hs, Is, Bs, Filename_sol, Filename_rec, Batches, P_Batches, C_Batches, Solutions, Rectangles = vs.getVisData(model, L, W, l, w, h)

## ====== Outputs from the Scheduling Algorithm ======
# 算法输出
# Xs: x-coordinates of the parts in the platform,  #part by 1 vector
# Ys: y-coordinates of the parts in the platform,  1 by #part vector
# Os: binary vector indicating parts rotations
#     e.g., Os[j] = 1 means that a part has been rotated by 90 degree abuut the z axis, i.e., part's original length is now parallel to the platform width. Otherwise Os[j] =0.

# Ks: Part orienation selection, 1 by # part vector.
#     e.g., Ks[j] = 0 means that part j selects its 0-th build orientation

# Ls: Length of the parts' projection on the platform, 1 by # part vector.
#     e.g., Ls[j] = 28.0 means that part j's projection on the platform is 28

# Ws: Width of the parts' projection on the platform, 1 by # part vector.
# Hs: Height of the parts on the platform, 1 by # part vector.

# Is: Machine indexed of the parts, 1 by # part
#     e.g., Is[j] = 0 means that part j is printed on machine 0
#           Is[j] = 1 means that part j is printed on machine 1

# Bs: Batch indexed of the parts, 1 by # part
#     e.g., Bs[j] = 0 means that part j is printed on the 0-th batch of the machine

# Note: So Is[j] and Bs[j] together specify in which batch the part will be printed
#     e.g., Is[j] = 0, Bs[j] = 0 means part j will be printed in the 0-th batch of machine 0


# Filename_sol: Name of the solution file
# Filename_rec: Name of the rectangle file

# Batches: Batch contents, a python dictionray mapping (machine id, batch id) to part list
#          e.g., (1,0):[0,3,4,7,9] means that the 0-th batch of machine 1 includes part 0,3,4,7 and 9

# Batches: Batch completion time, a python dictionray mapping (machine id, batch id) to batch completion time

# Solutions: A table-like data structure recording the parts' coordinates
# Rectangles: A table-like data structure recording the parts' projection information (outbox projections)

runtime = model.Runtime  # Actual running time of the scheduling algorithm
objval = model.Objval  # Objective value (Maximum completion time of all batches)
print("runtime:", runtime)
print("objval:", objval)

## =======================  Visualizatio  ====================================
## Plot the batches

idx = 0
Fig_bin = []
for b in Batches:
    mac = b[0]
    Sol_in = Solutions[idx]
    Rec_in = Rectangles[idx]
    fig_bin = vs.plotBin3D(Sol_in, Rec_in, L[mac], W[mac], HM[mac])
    Fig_bin.append(fig_bin)
    idx = idx + 1

## Plot the Schedule
T = [b for b in Batches]
ST = dict()
ET = dict()
P = [b[0] for b in Batches]
for b in Batches:
    ST[b] = C_Batches[b] - P_Batches[b]
    ET[b] = C_Batches[b]

vs.GantPlot(T, ST, ET, P)

## ==================== ======================================================


