
"""
@author: chunlongyu
"""


import gurobipy
from gurobipy import *
import numpy as np
import pdb  # set breakpoint
import random


def getKeyList(dict): 
    return list(dict.keys())


def GetBatchNum(num_mac,num_parts, V,U,S,L,W,HM,v,Kj,h,l,w,s):
    Bi = [ num_parts for i in range(num_mac)]
    K = dict()   # dict of list of alternative build orientations
    for j in range(num_parts):
        K[j] = [ k2 for k1,k2 in Kj.keys() if k1 == j]
    
    for i in range(num_mac):
        n_eli_parts = 0
        for j in range(num_parts):
            for k in K[j]:
                if ( (l[(j,k)] <= L[i]) and (w[(j,k)] <= W[i] ) and (h[(j,k)] <=HM[i]) ) \
                    or ( (w[(j,k)] <= L[i]) and (l[(j,k)] <= W[i] ) and (h[(j,k)] <=HM[i]) ):
                    n_eli_parts  = n_eli_parts + 1
                    break;
        Bi[i] = n_eli_parts
    return Bi


def NSPM_ML_TJ_exp(num_mac, num_parts, v1, v2, S1, S2, T1, T2, D1, D2, L, W, HM, NL, y_slice, Dp, Ds, Tr, v, a, Kj, h, l, w, s, t_run, d):
    # NSPM model with constrainted big M
    # NSPM model with expected value of the batch processing time

    # Sets
    I = [i for i in range(num_mac)]
    J = [i for i in range(num_parts)]
    B = J
    K = dict()  # dict of list of alternative build orientations
    for j in range(len(J)):
        K[j] = [k2 for k1, k2 in Kj.keys() if k1 == j]

    # Parameters
    # Big M value
    M1 = len(J)
    M58 = max(list(l.values()))  # 58
    M67 = max(list(w.values()))  # 67
    M9 = max(L) + max(list(l.values()))  # 9
    M10 = max(L) + max(list(w.values()))  # 10
    M11 = max(W) + max(list(w.values()))  # 11
    M12 = max(W) + max(list(l.values()))  # 12
    M14 = max(list(h.values()))  # 14

    maxs = max(list(s.values()))
    maxh = max(list(h.values()))
    maxa = max(a)
    maxv = max(v)

    # build time M
    m17 = []
    for i in I:
        bt_lv = sum([((2 * a[j] / v1[i]) + (v[j] / (Dp[i] * v2[i])) + (maxs / (Ds[i] * v2[i])))
                     / (NL[i] * y_slice[i]) for j in J]) + (Tr[i] * maxh / y_slice[i])
        m17.append(bt_lv / 3600)

    M17 = max(m17)  # 17
    print(M17)

    m18 = []
    for i in I:
        ct_max = len(J) * ((((2 * maxa / v1[i]) + (maxv / (Dp[i] * v2[i])) + (maxs / (Ds[i] * v2[i])))
                     / (NL[i] * y_slice[i]) + (Tr[i] * maxh / y_slice[i])) + (S1[i] + S2[i]) / 2 + (0.0838 * (T2[i]**2 - T1[i]**2) + 2.364 * (T2[i] - T1[i])) + (0.5048 * (T1[i]**2 - T2[i]**2) - 192.96 * (T1[i] - T2[i])) + (D1[i] + D2[i]) / 2)
        m18.append(ct_max / 3600)

    M18 = max(m18)  # 18
    print(M18)

    m19 = []
    for j in J:
        PTj = []
        Hj = []
        Sj = []
        for k in K[j]:
            hj = h[j, k]
            sj = s[j, k]
            Hj.append(hj)
            Sj.append(sj)
        hmax = max(Hj)
        smax = max(Sj)
        for i in I:
            pt = ((((2 * a[j] / v1[i]) + (v[j] / (Dp[i] * v2[i])) + (smax / (Ds[i] * v2[i]))) / (NL[i] * y_slice[i]) + (Tr[i] * hmax / y_slice[i])) + (S1[i] + S2[i]) / 2 + (0.0838 * (T2[i]**2 - T1[i]**2) + 2.364 * (T2[i] - T1[i])) + (0.5048 * (T1[i]**2 - T2[i]**2) - 192.96 * (T1[i] - T2[i])) + (D1[i] + D2[i]) / 2)
            PTj.append(pt / 3600)
        pt_max = max(PTj)
        m19.append(pt_max)
    M20 = sum(m19)
    M19 = sum(m19) - min(d)
    print(M19, M20)

    M4 = max(list(s.values()))  # 42

    # ---- Decision variables ----
    model = Model('NSPM')
    # X_jib, 1 if part j is assigned to the b-th batch of machine i, 0 otherwise
    X = model.addVars(J, I, B, vtype=GRB.BINARY, name="X")
    Y = model.addVars(Kj, vtype=GRB.BINARY, name="Y")
    E = model.addVars(J, I, B, vtype=GRB.CONTINUOUS, name="E")

    x = model.addVars(J, vtype=GRB.CONTINUOUS, name="x")
    y = model.addVars(J, vtype=GRB.CONTINUOUS, name="y")
    o = model.addVars(J, vtype=GRB.BINARY, name="o")

    Z = model.addVars(I, B, vtype=GRB.BINARY, name="Z")
    PL = model.addVars(J, J, vtype=GRB.BINARY, name="PL")
    PB = model.addVars(J, J, vtype=GRB.BINARY, name="PB")

    # PT_ib
    P = model.addVars(I, B, lb=0, vtype=GRB.CONTINUOUS, name="P")
    # BT_ib
    P_lv = model.addVars(I, B, lb=0, vtype=GRB.CONTINUOUS, name="P_lv")
    # C_ib
    C = model.addVars(I, B, lb=0, vtype=GRB.CONTINUOUS, name="C")
    # c_j
    c = model.addVars(J, lb=0, vtype=GRB.CONTINUOUS, name="c")
    # T_j
    Delay = model.addVars(J, vtype=GRB.BINARY, name="Delay")

    # 机器i上批次b的表面积、体积、支撑体积、高度
    area = model.addVars(I, B, lb=0, vtype=GRB.CONTINUOUS, name="area")
    volume = model.addVars(I, B, lb=0, vtype=GRB.CONTINUOUS, name="volume")
    support = model.addVars(I, B, lb=0, vtype=GRB.CONTINUOUS, name="support")
    H = model.addVars(I, B, lb=0, vtype=GRB.CONTINUOUS, name="H")  # height of the b-th batch on machine i

    # ---- End of Decision variables ----

    # ---- Constraints ----

    # Constrait 1, a batch cannot be assigend to if not formed
    model.addConstrs(
        (quicksum(X[j, i, b] for j in J) <= M1 * Z[i, b] for i in I for b in B), "C1")

    # Constraint 2, a part is assigned to only one batch
    model.addConstrs(
        (quicksum(X[j, i, b] for i in I for b in B) == 1 for j in J), "C2")

    # Constraint 3, a part selects only one build orientation
    model.addConstrs(
        (quicksum(Y[j, k] for k in K[j]) == 1 for j in J), "C3")

    # Constraint 4-1, calculate the processing time of a batch
    model.addConstrs(
        (P[i, b] == ((S1[i] + S2[i]) / 7200 + (0.0838 * (T2[i]**2 - T1[i]**2) + 2.364 * (T2[i] - T1[i])) / 3600 + (0.5048 * (T1[i]**2 - T2[i]**2) - 192.96 * (T1[i] - T2[i])) / 3600 + (D1[i] + D2[i]) / 7200) * Z[i, b] + P_lv[i, b] for i in I for b in B), "C4.1")

    model.addConstrs(
        (P_lv[i, b] == (((2 * area[i, b] / v1[i]) + (volume[i, b] / (Dp[i] * v2[i])) + (support[i, b] / (Ds[i] * v2[i])))
         / (NL[i] * y_slice[i]) + Tr[i] * H[i, b] / y_slice[i] - M17 * (1 - Z[i, b])) / 3600
         for i in I for b in B), "C4.2")

    model.addConstrs(
        (E[j, i, b] >= quicksum(Y[j, k] * s[j, k] for k in K[j]) - M4 * (1 - X[j, i, b]) for j in J for i in I for b in
         B), "C4.3")

    # Constraint 5, guarantee that the part is within the build boundaries
    model.addConstrs(
        (
        x[j] + quicksum(Y[j, k] * l[j, k] for k in K[j]) <= L[i] + M58 * (1 - quicksum(X[j, i, b] for b in B)) + M58 * (
                    1 - o[j]) for j in J for i in I), "C5")

    # Constraint 6, guarantee that the part is within the build boundaries
    model.addConstrs(
        (
        x[j] + quicksum(Y[j, k] * w[j, k] for k in K[j]) <= L[i] + M67 * (1 - quicksum(X[j, i, b] for b in B)) + M67 * (
        o[j]) for j in J for i in I), "C6")

    # Constraint 7, guarantee that the part is within the build boundaries
    model.addConstrs(
        (
        y[j] + quicksum(Y[j, k] * w[j, k] for k in K[j]) <= W[i] + M67 * (1 - quicksum(X[j, i, b] for b in B)) + M67 * (
                    1 - o[j]) for j in J for i in I), "C7")

    # Constraint 8, guarantee that the part is within the build boundaries
    model.addConstrs(
        (
        y[j] + quicksum(Y[j, k] * l[j, k] for k in K[j]) <= W[i] + M58 * (1 - quicksum(X[j, i, b] for b in B)) + M58 * (
        o[j]) for j in J for i in I), "C8")

    # Constraint 9, guarantee that the part j is placed to the left of jj if PL[j,jj] =1
    model.addConstrs(
        (x[j] + quicksum(Y[j, k] * l[j, k] for k in K[j]) <= x[jj] + M9 * (1 - PL[j, jj]) + M9 * (1 - o[j]) for j in J
         for jj in J if j != jj), "C9")  # I am here

    # Constraint 10, guarantee that the part j is placed to the left of jj if PL[j,jj] =1
    model.addConstrs(
        (x[j] + quicksum(Y[j, k] * w[j, k] for k in K[j]) <= x[jj] + M10 * (1 - PL[j, jj]) + M10 * (o[j]) for j in J for
         jj in J if j != jj), "C10")

    # Constraint 11, guarantee that the part j is placed to the downwards of jj if PB[j,jj] =1
    model.addConstrs(
        (y[j] + quicksum(Y[j, k] * w[j, k] for k in K[j]) <= y[jj] + M11 * (1 - PB[j, jj]) + M11 * (1 - o[j]) for j in J
         for jj in J if j != jj), "C11")

    # Constraint 12, guarantee that the part j is placed to the downwards of jj if PB[j,jj] =1
    model.addConstrs(
        (y[j] + quicksum(Y[j, k] * l[j, k] for k in K[j]) <= y[jj] + M12 * (1 - PB[j, jj]) + M12 * (o[j]) for j in J for
         jj in J if j != jj), "C12")

    # Constraint 13, when part j and jj are allocated on the same batch, there is at least one active positioning relationship
    model.addConstrs(
        (PL[j, jj] + PB[j, jj] + PL[jj, j] + PB[jj, j] >= X[j, i, b] + X[jj, i, b] - 1 for j in J for jj in J for i in I
         for b in B if j < jj), "C13")

    # Constraint 15, height of batch should be lower than that of the machine
    model.addConstrs(
        (H[i, b] <= HM[i] for i in I for b in B), "C15")

    # Constraint 16, calculate the completion time of a batch
    model.addConstrs(
        (C[i, 0] >= P[i, 0] for i in I), "C16")

    model.addConstrs(
        (C[i, b] >= C[i, b - 1] + P[i, b] for i in I for b in list(set(B) - set([B[0]]))), "C16")

    # Constraint 17, calculate the completion time of a part
    model.addConstrs(
        (c[j] >= C[i, b] - M20 * (1 - X[j, i, b]) for j in J for i in I for b in B), "C17")

    # # Constraint 18, decide whether a part is tardy
    model.addConstrs(
        (c[j] - d[j] <= M19 * Delay[j] for j in J), "C18")

    # Constraint 19, eliminate the symmetricity of batchs
    model.addConstrs(
        (Z[i, b - 1] >= Z[i, b] for i in I for b in list(set(B) - set([B[0]]))), "C19")

    # Constraint 20、21、22
    model.addConstrs(
        (area[i, b] == quicksum(X[j, i, b] * a[j] for j in J) for i in I for b in B), "C20")

    model.addConstrs(
        (volume[i, b] == quicksum(X[j, i, b] * v[j] for j in J) for i in I for b in B), "C21")

    model.addConstrs(
        (support[i, b] == quicksum(X[j, i, b] * E[j, i, b] for j in J) for i in I for b in B), "C22")

    # Constraint 14, calculate the height of the batch
    model.addConstrs(
        (H[i, b] >= quicksum(Y[j, k] * h[j, k] for k in K[j]) - M14 * (1 - X[j, i, b]) for i in I for b in B for j in
         J), "C14")

    obj = quicksum(Delay[j] for j in J) + 0.0001 * (10 * quicksum(Z[i, b] for i in I for b in B) \
          + 0.01 * quicksum(x[j] + y[j] for j in J) + 0.1 * quicksum(c[j] for j in J) \
          + 0.1 * quicksum(H[i, b] for i in I for b in B))

    model.setObjective(obj, GRB.MINIMIZE)
    # minimize

    # ---- End of Objective ----

    # ---- Solve ----
    model.setParam(GRB.Param.MIPGap, 0.001)
    model.setParam(GRB.Param.TimeLimit, t_run)

    model.optimize()

    for v in model.getVars():
        if v.X != 0:
            print("%s %f" % (v.Varname, v.X))

    model.write("AM_Scheduling-output.sol")
    return model
