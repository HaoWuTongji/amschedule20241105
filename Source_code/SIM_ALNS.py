"""
Author: Hao Wu (吴昊), Chunlong Yu (余春龙)
Date: 2024-10-08
"""

import math
import random
import copy
import time
import matplotlib.pyplot as plt


from Class_Solution import *
from BinPackingHeu import *


def readInstance(path):
    ## ==== Read data =====
    filePath = path  # get input path for filePath
    lines = []
    with open(filePath) as f:
        lines = f.readlines()  # read all lines in the file

    ## ==== Parameters ====
    count = 0
    numbers1 = [int(x) for x in lines[count].split()]
    types_mac = numbers1[0]  # number of machine types
    types_parts = numbers1[1]  # number of part types

    count = 1
    numbers2 = [int(x) for x in lines[count].split()]
    num_mac = numbers2[0]  # number of machines
    num_parts = numbers2[1]  # number of parts

    v1 = [0 for i in range(num_mac)]  # contour scanning speed of machine
    v2 = [0 for i in range(num_mac)]  # hatch scanning speed of machine
    S1 = [0 for i in range(num_mac)]  # low bound of start up time of machine
    S2 = [0 for i in range(num_mac)]  # upper bound of start up time of machine
    T1 = [0 for i in range(num_mac)]  # environment temperature
    T2 = [0 for i in range(num_mac)]  # building temperature
    D1 = [0 for i in range(num_mac)]  # low bound of post-processing time of machine
    D2 = [0 for i in range(num_mac)]  # upper bound of post-processing time of machine
    L = [0 for i in range(num_mac)]  # platform length of machine
    W = [0 for i in range(num_mac)]  # platform width of machine
    HM = [0 for i in range(num_mac)]  # chamber width of machine
    NL = [0 for i in range(num_mac)]  # number of lasers of machine
    y_slice = [0 for i in range(num_mac)]  # layer thickness of machine
    Dp = [0 for i in range(num_mac)]  # hatching distance of parts of machine
    Ds = [0 for i in range(num_mac)]  # hatching distance of support structure of machine
    Tr = [0 for i in range(num_mac)]  # recoating time for each layer on machine

    v = [0 for i in range(num_parts)]  # part volume
    a = [0 for i in range(num_parts)]  # part surface area

    Kj = dict()  # set of alternative build orientations for part
    h = dict()
    l = dict()
    w = dict()
    s = dict()
    # height, length, and width of and volume of support structure

    ## ====== Machines ======
    count = 3
    count_mac = 0
    for i in range(types_mac):
        [index, num_mac_type, v1[count_mac], v2[count_mac], S1[count_mac], S2[count_mac], T1[count_mac], T2[count_mac], D1[count_mac], D2[count_mac], L[count_mac], W[count_mac], HM[count_mac],
         NL[count_mac], y_slice[count_mac], Dp[count_mac], Ds[count_mac], Tr[count_mac]] = [float(x) for x in lines[count].split()]

        if num_mac_type > 1:
            for j in range(num_mac_type):
                count_mac = count_mac + 1
                [v1[count_mac], v2[count_mac], S1[count_mac], S2[count_mac], T1[count_mac], T2[count_mac], D1[count_mac], D2[count_mac], L[count_mac], W[count_mac], HM[count_mac], NL[count_mac],
                 y_slice[count_mac], Dp[count_mac], Ds[count_mac], Tr[count_mac]] = \
                    [v1[count_mac - 1], v2[count_mac - 1], S1[count_mac - 1], S2[count_mac - 1], T1[count_mac - 1], T2[count_mac - 1], D1[count_mac - 1], D2[count_mac - 1], L[count_mac - 1], W[count_mac - 1],
                     HM[count_mac - 1], NL[count_mac - 1], y_slice[count_mac - 1], Dp[count_mac - 1], Ds[count_mac - 1],
                     Tr[count_mac - 1]]

        count_mac = count_mac + 1
        count = count + 1

    ## ====== Parts ======
    count_part = 0
    for i in range(types_parts):
        count = count + 1

        [index, num_part_type, num_ori, vol, are] = [float(x) for x in lines[count].split()]
        num_ori = int(num_ori)
        num_part_type = int(num_part_type)

        v[count_part] = vol
        a[count_part] = are

        for ii in range(int(num_ori)):
            Kj[count_part, ii] = ii

        for k in range(num_ori):
            count = count + 1
            [ll, ww, hh, ss] = [float(x) for x in lines[count].split()]
            l[count_part, k] = ll
            w[count_part, k] = ww
            h[count_part, k] = hh
            s[count_part, k] = ss
        count_part = count_part + 1

        if num_part_type > 1:
            for j in range(num_part_type - 1):

                v[count_part] = vol
                a[count_part] = are
                for ii in range(int(num_ori)):
                    Kj[count_part, ii] = ii

                for k in range(num_ori):
                    l[count_part, k] = l[count_part - 1, k]
                    w[count_part, k] = w[count_part - 1, k]
                    h[count_part, k] = h[count_part - 1, k]
                    s[count_part, k] = s[count_part - 1, k]
                count_part = count_part + 1

        count = count + 1

    return types_mac, types_parts, num_mac, num_parts, v1, v2, S1, S2, T1, T2, D1, D2, L, W, HM, NL, y_slice, Dp, Ds, Tr, v, a, Kj, h, l, w, s


def FormBatch(J, AP, AM):
    # Given a set of parts(represented by projection area) and machine area, this function returns a set of formed batches in the format of dictionary.

    # ==== Inputs ====
    # J: index of the parts to be batched
    # AP: Area of all parts
    # AM: Area of the machine

    B = dict()
    b = 0
    tot_area = 0
    jobs = []

    for j in J:
        done = False
        while not (done):
            if tot_area + AP[j] < AM:
                # assign the part to batch b
                jobs.append(j)
                tot_area = tot_area + AP[j]
                done = True
            else:
                # close the batch
                B[b] = jobs
                # start a new batch
                jobs = []
                tot_area = 0
                b = b + 1
    # close the last batch
    B[b] = jobs

    return B


def CalBatchProTime(B, i, HP, S2, T1, T2, D2, v, a, VS, v1, v2, Dp, Ds, NL, y_slice, Tr):
    # Given a set of batch and the corresponding machine id, calculate the processing time
    ProTime = 0
    for b in range(len(B)):
        ProTime = ProTime + S2[i] + D2[i] + 0.0838 * ((T2[i] * T2[i]) - (T1[i] * T1[i])) + 2.364 * (T2[i] - T1[i]) + 0.5048 * ((T1[i] * T1[i]) - (T2[i] * T2[i])) - 192.96 * (T1[i] - T2[i]) + (2 * (sum([a[j] for j in B[b]]) / v1[i]) +
                                    ((sum([v[j] for j in B[b]]) / (Dp[i] * v2[i])) +
                                     (sum([VS[j] for j in B[b]]) / (Ds[i] * v2[i]))) / (NL[i] * y_slice[i])
                                    + Tr[i] * max([HP[j] for j in B[b]]) / y_slice[i])

    return ProTime


def GenerateDueDate(path, TF, RDD, RndSeed):
    # Generate DueDate of Parts.

    # ==== Inputs ====
    # TF: Tightness factor
    # RDD: Range of duedate
    # RndSeed: Random seed used to generate duedates
    # TF = 0.3
    # RDD = 0.6
    # RndSeed = 1

    random.seed(RndSeed)

    types_mac, types_parts, num_mac, num_parts, v1, v2, S1, S2, T1, T2, D1, D2, L, W, HM, NL, y_slice, Dp, Ds, Tr, v, a, Kj, h, l, w, s = readInstance(path)

    # Estimate the exptected makespan
    AM = [L[i] * W[i] for i in range(num_mac)]  # Area of machine

    I = range(num_mac)
    J = range(num_parts)

    AP = []  # Area of parts
    HP = []  # Height of parts
    VS = []  # Volume of support for the selected orientation
    K = []  # Selected orientation index
    D = []  # Duedate of parts

    for j in range(num_parts):
        hs = [h[jj, kk] for jj, kk in Kj if jj == j]
        k = hs.index(max(hs))

        K.append(k)
        VS.append(s[j, k])
        HP.append(max(hs))

        AP.append(l[j, k] * w[j, k])
        D.append(0)

    AP_vals = np.array(AP)
    J_sorted = np.argsort(-1 * AP_vals)  # Job sequence with non-increasing area
    # AP_sorted = AP_vals[J_sorted]  # sorted area of parts

    # Initiate the batches on machine
    POM = dict()  # parts on machine
    AAM = dict()  # available area on machine 可用面积
    MCT = dict()  # Machine completion time

    for i in range(num_mac):
        POM[i] = []
        AAM[i] = AM[i]
        MCT[i] = 0

    # Assign parts to batches
    for j in J_sorted:
        MECT = [0 for i in I]  # Machine expected completion time
        for i in I:
            Temp = POM[i] + [j]

            B = FormBatch(Temp, AP, AM[i])

            ProcT = CalBatchProTime(B, i, HP, S2, T1, T2, D2, v, a, VS, v1, v2, Dp, Ds, NL, y_slice, Tr)

            MECT[i] = ProcT

        ii = [i for i in I if MECT[i] == min(MECT)]
        ii = ii[0]  # Selected machine

        # Update parts on machine & machine completion time
        POM[ii].append(j)
        MCT[ii] = MECT[ii]

    Cmax = max([MCT[i] for i in MCT])

    # Generate the duedates for jobs
    D = []
    for j in J:
        ddl = random.randint(int(max(1, Cmax * (1 - TF - RDD / 2))), int(Cmax * (1 - TF + RDD / 2)))
        D.append(ddl)

    return D


def construct_initial_solution(I, J, K, L, W, HM, h, l, w, d):
    """
    Construct the initial feasible solution S0 based on the provided algorithm description.

    Parameters:
    - parts: List of tuples containing part data (part_id, due_date, part_position, part_rotation).
    - machines: List of Machine instances to allocate parts to.
    - K_j: Dictionary mapping part_ids to possible orientations.

    Returns:
    - List of Machine instances with assigned batches.
    """
    # Step 1: Sort parts by due dates in non-decreasing order
    J_sorted = sorted(J, key=lambda j: d[j])  # Sort by due_date

    # Step 2: Initialize variables
    max_batches = len(J_sorted) // len(I) + 1  # Upper bound for the number of batches
    n_b = 0  # Initialize the batch number
    feasible_solution = False
    machines = []

    while not feasible_solution and n_b < max_batches:
        n_b += 1  # Increment batch number
        machines = []

        # Step 3: Create empty batches for each machine
        for m in I:
            """
               Create a list of m Machine instances.

               Parameters:
               - len(I): int, the number of machines to create.

               - List of Machine instances.
               """
            machine = Machine(m)
            machines.append(machine)

            for i in range(n_b):
                machine.add_batch(i)  # Unique batch ID for each machine and batch

        # Step 4: Assign parts to batches
        num_subsets = len(I) * n_b

        if len(J_sorted) < num_subsets:
            raise ValueError("length of J_sorted is not enough for needed subsets.")

        subsets = []

        min_subset_size = len(J_sorted) // num_subsets
        remaining_elements = len(J_sorted) % num_subsets

        start_index = 0

        for i in range(num_subsets):
            current_subset_size = min_subset_size + (1 if remaining_elements > 0 else 0)

            subset = J_sorted[start_index:start_index + current_subset_size]
            subsets.append(subset)

            start_index += current_subset_size
            remaining_elements -= 1

        subset_index = 0

        for machine in machines:
            for batch in machine.batches:
                for j in subsets[subset_index]:
                    # Randomly select the building orientation for the part
                    random.seed()
                    part_direction_id = random.choice(K[j])
                    batch.add_part(j, part_direction_id, (0, 0), "nr")
                subset_index = subset_index + 1

        # Step 5: Execute the BestFitRotate procedure for each batch
        feasible_solution = True  # Assume feasibility

        for machine in machines:
            m = machine.machine_id
            L_mac = L[m]
            W_mac = W[m]
            HM_mac = HM[m]
            for batch in machine.batches:
                b = batch.batch_id
                # Check batch feasibility (implement BestFitRotate check here)
                bin_sizes = []
                h_choosed = []
                parts_info = batch.get_parts_id_ori()
                for (j, k) in parts_info:
                    # get orientation-related info
                    h_choosed.append(h[j, k])
                    l_choosed = l[j, k]
                    w_choosed = w[j, k]
                    bin_sizes.append(l_choosed)
                    bin_sizes.append(w_choosed)

                if max(h_choosed) > HM_mac:
                    feasible_solution = False
                    break
                # call BestFitRotate
                item_coor_list = BestFitRotate(bin_sizes, L_mac)
                # item_coor_list = [bin1_left_bottom_corner_x, bin1_left_bottom_corner_y, bin1_right_top_corner_x,
                # bin1_right_top_corner_y, bin2_left_bottom_corner_x, bin2_left_bottom_corner_y,
                # bin2_right_top_corner_x, bin2_right_top_corner_y,...]
                bins_coor = []
                for i in range(int(len(item_coor_list) / 4)):
                    x1, y1, x2, y2 = item_coor_list[i * 4: (i + 1) * 4]  # get coordinate of part i
                    bins_coor.append([x1, y1, x2, y2])
                    if y2 > W_mac:
                        feasible_solution = False
                        break

            if not feasible_solution:
                break

    # Step 6: Return the initial feasible solution if found
    print("number of batches on each machine:", n_b)
    if feasible_solution:
        return machines
    else:
        return "Infeasible"  # Indicating the problem may be unsolvable


def ALNS(I, J, K, L, W, HM, v1, v2, S1, S2, T1, T2, D1, D2, NL, y_slice, Dp, Ds, Tr, v, a, Kj, h, l, w, s, d):
    """ALNS: Main Function"""
    # record start time
    start_time = time.time()

    random.seed()

    "===========================hyperparameter================================"
    # scale of problem
    n_part = len(J)

    alpha = 0.1  # scaling factor of initial temperature in the Metropolis acceptance criterion
    beta = 0.995  # cooling rate in the Metropolis acceptance criterion
    gamma = 0.1  # perturbation frequency
    # weight adjustment factors
    lamda = 0.9
    w_0 = 33
    w_1 = 9
    w_2 = 13
    w_3 = 0
    # initial temperature in the Metropolis acceptance criterion
    T_0 = alpha * n_part
    # maximum number of iterations
    max_iter = 15000
    # perturbation threshold
    Count = gamma * max_iter

    "===========================initialization================================"
    # call ConstructInitialSolution，get feasible initial solution "ini_machines"
    ini_machines = construct_initial_solution(I, J, K, L, W, HM, h, l, w, d)

    for machine in ini_machines:
        print(f"Machine ID: {machine.machine_id}")
        for batch in machine.batches:
            print(f"Batch ID: {batch.batch_id}")
            for part in batch.parts:
                print(
                    f"Part ID: {part.part_id}, Direction ID: {part.part_direction_id}")
        print()

    # operators weights initialization
    weigh_destory = [1, 1, 1, 1, 1]
    weigh_repair_matrix = [[1, 1, 1, 1],
                           [1, 1, 1, 1],
                           [1, 1, 1, 1],
                           [1, 1, 1, 1],
                           [1, 1, 1, 1]]

    # store current solution "cur_best"
    cur_best = []
    cur_best.append(ini_machines)

    iter = 0
    # count of consecutive rejections
    c = 0
    T = T_0

    # objective value estimation of initial solution "ini_target"
    ini_target = Monte_Carlo(v1, v2, S1, S2, T1, T2, D1, D2, NL, y_slice, Dp, Ds, Tr, v, a, h, s, d, ini_machines)
    print("objective value estimation of initial solution:", ini_target[0])

    # store objective value estimation for all iterations
    target = []
    target.append(ini_target[0])

    # store the best objective value estimation for all iterations
    best_target = []
    best_target.append(ini_target[0])

    # record the time at the best solution is found
    BEST_TIME = 0

    # part of the greatest contribution to objective value estimation
    max_TT_part = ini_target[1]

    # batch of the greatest contribution to objective value estimation
    max_T_batch = ini_target[2]

    # record the best solution "all_best_sol"
    all_best_sol = ini_machines

    # count of get into local optima
    local_best = 0

    "===========================get into loop================================"
    while iter < max_iter:

        iter = iter + 1
        print("\nIteration：", iter)

        copied_cur = copy.deepcopy(cur_best[iter - 1])

        "===========================destory and repair operation================================"
        # select destory operator
        selected_destory = roulette_wheel_selection(weigh_destory)

        new_sol = []

        repair = []

        # Random Part Removal (RR)
        if selected_destory == 0:
            tem_machines, removed_part = remove_random_part(cur_best[iter - 1])

            selected_repair = roulette_wheel_selection(weigh_repair_matrix[selected_destory])
            repair.append(selected_repair)

            # Random Insertion (RI)
            if selected_repair == 0:
                new_machines, insert_mac, insert_batch = insert_random_part(K, L, W, HM, h, l, w, tem_machines, removed_part)
                if new_machines is not "infeasible":
                    new_new_machines = remove_non_part_batch(new_machines)
                    new_sol.append(new_new_machines)
                else:
                    new_sol.append("infeasible")

            # Minimum Height Variance Insertion (MHVI):
            if selected_repair == 1:
                new_machines, insert_mac, insert_batch, insert_ori = Min_H_Var_Insert(K, L, W, HM, h, l, w, tem_machines, removed_part)
                if new_machines is not "infeasible":
                    new_new_machines = remove_non_part_batch(new_machines)
                    new_sol.append(new_new_machines)
                else:
                    new_sol.append("infeasible")

            # Maximum Earliest Due-date Insertion (MEDI)
            if selected_repair == 2:
                new_machines, insert_mac, insert_batch = Max_Earlist_DD_Insert(K, L, W, HM, h, l, w, tem_machines, removed_part, d)
                if new_machines is not "infeasible":
                    new_new_machines = remove_non_part_batch(new_machines)
                    new_sol.append(new_new_machines)
                else:
                    new_sol.append("infeasible")

            # Greedy Insertion (GI)
            if selected_repair == 3:
                new_machines, insert_mac, insert_batch = Min_DR_batch_Insert_HN(K, L, W, HM, h, l, w, v1, v2, S1, S2, T1, T2, D1, D2, NL, y_slice, Dp, Ds, Tr, v, a, s, d, tem_machines, removed_part)
                if new_machines is not "infeasible":
                    new_new_machines = remove_non_part_batch(new_machines)
                    new_sol.append(new_new_machines)
                else:
                    new_sol.append("infeasible")

        # Greedy Part Removal (GR):
        if selected_destory == 1:
            tem_machines, removed_part = remove_max_TT_part(cur_best[iter - 1], max_TT_part)
            selected_repair = roulette_wheel_selection(weigh_repair_matrix[selected_destory])
            repair.append(selected_repair)

            if selected_repair == 0:
                new_machines, insert_mac, insert_batch = insert_random_part(K, L, W, HM, h, l, w, tem_machines, removed_part)
                if new_machines is not "infeasible":
                    new_new_machines = remove_non_part_batch(new_machines)
                    new_sol.append(new_new_machines)
                else:
                    new_sol.append("infeasible")

            if selected_repair == 1:
                new_machines, insert_mac, insert_batch, insert_ori = Min_H_Var_Insert(K, L, W, HM, h, l, w, tem_machines, removed_part)
                if new_machines is not "infeasible":
                    new_new_machines = remove_non_part_batch(new_machines)
                    new_sol.append(new_new_machines)
                else:
                    new_sol.append("infeasible")

            if selected_repair == 2:
                new_machines, insert_mac, insert_batch = Max_Earlist_DD_Insert(K, L, W, HM, h, l, w, tem_machines, removed_part, d)
                if new_machines is not "infeasible":
                    new_new_machines = remove_non_part_batch(new_machines)
                    new_sol.append(new_new_machines)
                else:
                    new_sol.append("infeasible")

            if selected_repair == 3:
                new_machines, insert_mac, insert_batch = Min_DR_batch_Insert_HN(K, L, W, HM, h, l, w, v1, v2, S1, S2,
                                                                                T1, T2, D1, D2, NL, y_slice, Dp, Ds, Tr,
                                                                                v, a, s, d, tem_machines, removed_part)
                if new_machines is not "infeasible":
                    new_new_machines = remove_non_part_batch(new_machines)
                    new_sol.append(new_new_machines)
                else:
                    new_sol.append("infeasible")

        # Random Batch Removal (RBR)
        if selected_destory == 2:
            tem_machines, removed_parts = remove_random_batch(cur_best[iter - 1])

            # eliminate batches that contain no part and renew batch id
            tem_tem_machines = remove_non_part_batch(tem_machines)

            # iteratively insert parts until a complete solution is obtained
            for removed_part in removed_parts:

                selected_repair = roulette_wheel_selection(weigh_repair_matrix[selected_destory])
                repair.append(selected_repair)

                if selected_repair == 0:
                    new_machines, insert_mac, insert_batch = insert_random_part(K, L, W, HM, h, l, w, tem_tem_machines,
                                                                                removed_part)
                    if new_machines is "infeasible":
                        tem_tem_machines = "infeasible"
                        break
                    else:
                        tem_tem_machines = new_machines

                if selected_repair == 1:
                    new_machines, insert_mac, insert_batch, insert_ori = Min_H_Var_Insert(K, L, W, HM, h, l, w,
                                                                                          tem_tem_machines,
                                                                                          removed_part)
                    if new_machines is "infeasible":
                        tem_tem_machines = "infeasible"
                        break
                    else:
                        tem_tem_machines = new_machines

                if selected_repair == 2:
                    new_machines, insert_mac, insert_batch = Max_Earlist_DD_Insert(K, L, W, HM, h, l, w,
                                                                                   tem_tem_machines, removed_part, d)
                    if new_machines is "infeasible":
                        tem_tem_machines = "infeasible"
                        break
                    else:
                        tem_tem_machines = new_machines

                if selected_repair == 3:
                    new_machines, insert_mac, insert_batch = Min_DR_batch_Insert_HN(K, L, W, HM, h, l, w, v1, v2, S1,
                                                                                    S2, T1, T2, D1, D2, NL, y_slice, Dp,
                                                                                    Ds, Tr, v, a, s, d, tem_tem_machines,
                                                                                    removed_part)
                    if new_machines is "infeasible":
                        tem_tem_machines = "infeasible"
                        break
                    else:
                        tem_tem_machines = new_machines

            new_sol.append(tem_tem_machines)

        # Maximum Height Variance Batch Removal (MHVR)
        if selected_destory == 3:
            tem_machines, mac_id, bat_id, removed_parts = remove_max_height_variance_batch(cur_best[iter - 1], h)

            tem_tem_machines = remove_non_part_batch(tem_machines)

            for removed_part in removed_parts:

                selected_repair = roulette_wheel_selection(weigh_repair_matrix[selected_destory])
                repair.append(selected_repair)

                if selected_repair == 0:
                    new_machines, insert_mac, insert_batch = insert_random_part(K, L, W, HM, h, l, w, tem_tem_machines, removed_part)
                    if new_machines is "infeasible":
                        tem_tem_machines = "infeasible"
                        break
                    else:
                        tem_tem_machines = new_machines

                if selected_repair == 1:
                    new_machines, insert_mac, insert_batch, insert_ori = Min_H_Var_Insert(K, L, W, HM, h, l, w, tem_tem_machines, removed_part)
                    if new_machines is "infeasible":
                        tem_tem_machines = "infeasible"
                        break
                    else:
                        tem_tem_machines = new_machines

                if selected_repair == 2:
                    new_machines, insert_mac, insert_batch = Max_Earlist_DD_Insert(K, L, W, HM, h, l, w, tem_tem_machines, removed_part, d)
                    if new_machines is "infeasible":
                        tem_tem_machines = "infeasible"
                        break
                    else:
                        tem_tem_machines = new_machines

                if selected_repair == 3:
                    new_machines, insert_mac, insert_batch = Min_DR_batch_Insert_HN(K, L, W, HM, h, l, w, v1, v2, S1,
                                                                                    S2, T1, T2, D1, D2, NL, y_slice, Dp,
                                                                                    Ds, Tr, v, a, s, d, tem_tem_machines,
                                                                                    removed_part)
                    if new_machines is "infeasible":
                        tem_tem_machines = "infeasible"
                        break
                    else:
                        tem_tem_machines = new_machines

            new_sol.append(tem_tem_machines)

        # Greedy Batch Removal(GBR)
        if selected_destory == 4:

            tem_machines, removed_parts = remove_max_T_batch(cur_best[iter - 1], max_T_batch)

            # 移除这个空批次并更新批次编号
            tem_tem_machines = remove_non_part_batch(tem_machines)

            # 在被移除零件列表中循环，实现逐个嵌套调用过程中的新解
            for removed_part in removed_parts:

                # 轮盘赌选择修复算子,利用修复权重矩阵，改为操作耦合
                selected_repair = roulette_wheel_selection(weigh_repair_matrix[selected_destory])
                repair.append(selected_repair)

                if selected_repair == 0:

                    new_machines, insert_mac, insert_batch = insert_random_part(K, L, W, HM, h, l, w,
                                                                                    tem_tem_machines, removed_part)
                    if new_machines is "infeasible":

                        tem_tem_machines = "infeasible"

                        break
                    else:

                        tem_tem_machines = new_machines

                if selected_repair == 1:

                    new_machines, insert_mac, insert_batch, insert_ori = Min_H_Var_Insert(K, L, W, HM, h, l, w,
                                                                                              tem_tem_machines,
                                                                                              removed_part)
                    if new_machines is "infeasible":

                        tem_tem_machines = "infeasible"

                        break
                    else:

                        tem_tem_machines = new_machines

                if selected_repair == 2:

                    new_machines, insert_mac, insert_batch = Max_Earlist_DD_Insert(K, L, W, HM, h, l, w,
                                                                                       tem_tem_machines, removed_part,
                                                                                       d)
                    if new_machines is "infeasible":

                        tem_tem_machines = "infeasible"

                        break
                    else:

                        tem_tem_machines = new_machines

                if selected_repair == 3:
                    new_machines, insert_mac, insert_batch = Min_DR_batch_Insert_HN(K, L, W, HM, h, l, w, v1, v2, S1,
                                                                                    S2, T1, T2, D1, D2, NL, y_slice, Dp,
                                                                                    Ds, Tr, v, a, s, d, tem_tem_machines,
                                                                                    removed_part)
                    if new_machines is "infeasible":

                        tem_tem_machines = "infeasible"

                        break
                    else:

                        tem_tem_machines = new_machines

            new_sol.append(tem_tem_machines)

        "===========================Acceptance criteria================================"
        # if new solution does not exist (infeasible)
        if new_sol == ["infeasible"]:

            cur_best.append(copied_cur)

            target.append(target[iter - 1])

            best_target.append(best_target[iter - 1])

            c = c + 1

            wei_d = w_3
            wei_r = w_3
        else:
            # new solution exists
            new_solut = new_sol[0]
            # call Monte_Carlo, get objective value estimation of new solution "new_target"
            new_target = Monte_Carlo(v1, v2, S1, S2, T1, T2, D1, D2, NL, y_slice, Dp, Ds, Tr, v, a, h, s, d, new_solut)

            # if new solution is superior
            if new_target[0] < target[iter - 1]:

                cur_best.append(new_solut)

                max_TT_part = new_target[1]

                max_T_batch = new_target[2]

                target.append(new_target[0])

                # if new solution is the historical best
                if new_target[0] < min(best_target):
                    all_best_sol = copy.deepcopy(new_solut)
                    for machine in all_best_sol:
                        print(f"Machine ID: {machine.machine_id}")
                        for batch in machine.batches:
                            print(f"Batch ID: {batch.batch_id}")
                            for part in batch.parts:
                                print(
                                    f"Part ID: {part.part_id}, Direction ID: {part.part_direction_id}, Position: {part.part_position}, Rotation: {part.part_rotation}")
                        print()

                    BEST_TIME = time.time() - start_time

                    best_target.append(new_target[0])

                    wei_d = w_0
                    wei_r = w_0
                else:
                    best_target.append(best_target[iter - 1])

                    wei_d = w_1
                    wei_r = w_1

                c = 0

            else:
                # if new solution is inferior, yet accepted
                random.seed()
                if random.random() <= math.exp(- (new_target[0] - target[iter - 1]) / T):

                    cur_best.append(new_solut)

                    max_TT_part = new_target[1]

                    max_T_batch = new_target[2]

                    target.append(new_target[0])

                    best_target.append(best_target[iter - 1])

                    # compare the new solution with current solution (avoid the mistake of updatation of current solution)
                    if compare_machines(new_solut, cur_best[iter - 1]) == False:
                        c = 0

                        wei_d = w_2
                        wei_r = w_2
                    else:
                        # if the new solution is identical with current solution
                        c = c + 1

                        wei_d = w_3
                        wei_r = w_3
                else:
                    # new solution is rejected

                    c = c + 1

                    wei_d = w_3
                    wei_r = w_3

                    "===========================Perturbation strategy================================"
                    # when the repaired solution is feasible, and the current solution has not been updated for at least perturbation threshold of iterations
                    if c >= Count:
                        # the temperature will be raised
                        T = T / (beta ** c)
                        c = 0
                        local_best = local_best + 1

                        # new solution will be accepted unconditionally
                        cur_best.append(new_solut)

                        max_TT_part = new_target[1]

                        max_T_batch = new_target[2]

                        target.append(new_target[0])

                        best_target.append(best_target[iter - 1])
                    else:

                        cur_best.append(copied_cur)

                        target.append(target[iter - 1])

                        best_target.append(best_target[iter - 1])

        "===========================Record================================"
        print("objective value estimation of the historical best:", best_target[iter])

        "===========================Adaptive mechanism================================"
        # cool
        T = T * beta

        # adjust weights of destroy and repair operators
        weigh_destory[selected_destory] = lamda * weigh_destory[selected_destory] + (1 - lamda) * wei_d

        for sel_repair in repair:
            weigh_repair_matrix[selected_destory][sel_repair] = lamda * weigh_repair_matrix[selected_destory][sel_repair] + (1 - lamda) * wei_r

    "===========================end of loop================================"

    "===========================Outputs================================"

    # ending time
    end_time = time.time()

    # running time
    duration = end_time - start_time
    print("\nrunning time:", duration, "second")

    all_time_best = min(target)
    print("iteration, objective value estimation of the historical best:", target.index(all_time_best), all_time_best)
    print("duration of finding the historical best:", BEST_TIME)

    for machine in all_best_sol:
        print(f"Machine ID: {machine.machine_id}")
        for batch in machine.batches:
            print(f"Batch ID: {batch.batch_id}")
            for part in batch.parts:
                print(
                    f"Part ID: {part.part_id}, Direction ID: {part.part_direction_id}, Position: {part.part_position}, Rotation: {part.part_rotation}")
        print()

    print("perturbation count:", local_best)

    with open("output_sim.txt", "w") as file:

        file.write("running time:")
        file.write(str(duration) + ",\n")

        file.write("iteration of the historical best:")
        file.write(str(target.index(all_time_best)) + ",\n")

        file.write("objective value estimation of the historical best:")
        file.write(str(all_time_best) + ",\n")

        file.write("duration of finding the historical best:")
        file.write(str(BEST_TIME) + ",\n")

        file.write("perturbation count:")
        file.write(str(local_best) + ",\n")

        for machine in all_best_sol:
            file.write(f"Machine ID: {machine.machine_id}\n")
            for batch in machine.batches:
                file.write(f"  Batch ID: {batch.batch_id}\n")
                for part in batch.parts:
                    file.write(
                        f"    Part ID: {part.part_id}, Direction ID: {part.part_direction_id}, Position: {part.part_position}, Rotation: {part.part_rotation}\n"
                    )
                file.write("\n")
            file.write("\n")

        file.write("destory operator weights:")
        file.write(str(weigh_destory) + ",\n")

        file.write("repair operator weights")
        file.write(str(weigh_repair_matrix) + ",\n")

        file.write("list of objective value estimation of current solution:")
        file.write(str(target) + ",\n")

        file.write("list of objective value estimation of the historical best:")
        file.write(str(best_target) + ",\n")

    "============================Drawings==============================="

    x = range(1, len(target) + 1)

    plt.plot(x, target, linestyle='-', color='skyblue')

    plt.plot(x, best_target, linestyle='-', color='red')

    plt.title('Object')
    plt.xlabel('iteration')
    plt.ylabel('value')

    plt.grid(True)

    plt.savefig('tardy_part_count_evaluation.svg', format='svg')

    plt.show()

    return all_best_sol


def roulette_wheel_selection(weights):
    # roulette wheel selection algorithm

    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    roulette = []
    acc_prob = 0
    for weight in normalized_weights:
        acc_prob += weight
        roulette.append(acc_prob)

    random.seed()

    r = random.random()
    for i, prob in enumerate(roulette):
        if r <= prob:
            return i


def Monte_Carlo(v1, v2, S1, S2, T1, T2, D1, D2, NL, y_slice, Dp, Ds, Tr, v, a, h, s, d, machines):

    "calculate each batch's build time expectation"
    bt_mod = []
    ht_mod =[]
    ct_mod = []

    for m in range(get_mac_num(machines)):
        bt_mod_mac = []
        ht_mod.append(0.0838 * (T2[m]**2 - T1[m]**2) + 2.364 * (T2[m] - T1[m]))
        ct_mod.append(0.5048 * (T1[m]**2 - T2[m]**2) - 192.96 * (T1[m] - T2[m]))

        machine = get_machine_by_id(machines, m)

        for i in range(get_bat_num(machines, m)):
            if get_bat_num(machines, m) == 0:
                break
            batch = machine.get_batch_by_id(i)
            part_ori_mac_bat = batch.get_parts_id_ori()

            a_parts_bat = []
            v_parts_bat = []
            s_parts_bat = []
            h_parts_bat = []

            for (p, o) in part_ori_mac_bat:
                a_parts_bat.append(a[p])
                v_parts_bat.append(v[p])
                s_parts_bat.append(s[p, o])
                h_parts_bat.append(h[p, o])

            a_bat_sum = sum(a_parts_bat)
            v_bat_sum = sum(v_parts_bat)
            s_bat_sum = sum(s_parts_bat)
            if h_parts_bat:
                h_bat_max = max(h_parts_bat)
            else:
                h_bat_max = 0

            bt_model = 2 * a_bat_sum / (NL[m] * y_slice[m] * v1[m]) + v_bat_sum / (NL[m] * y_slice[m] * v2[m] * Dp[m]) + s_bat_sum / (NL[m] * y_slice[m] * v2[m] * Ds[m]) + h_bat_max * Tr[m] / y_slice[m]
            bt_mod_mac.append(bt_model)

        bt_mod.append(bt_mod_mac)

    "sample all scenarios"

    num_samples = 500

    np.random.seed(42)

    tardy_part_count = []

    part_delay = {}

    batch_delay = {}

    for j in range(num_samples):

        tardy_part_count_sce = 0

        for m in range(get_mac_num(machines)):
            machine = get_machine_by_id(machines, m)

            B_mac = bt_mod[m]
            S1_mac = S1[m]
            S2_mac = S2[m]
            D1_mac = D1[m]
            D2_mac = D2[m]
            HT_mac = ht_mod[m]
            CT_mac = ct_mod[m]

            P_rand = []
            C_rand = []

            for i in range(get_bat_num(machines, m)):
                batch = machine.get_batch_by_id(i)

                B_bat = B_mac[i]

                # startup time
                st = np.random.uniform(S1_mac, S2_mac, 1)
                # postprocess time
                dt = np.random.uniform(D1_mac, D2_mac, 1)
                # build time
                bt = np.random.normal(B_bat, 0.1 * B_bat, 1)
                # preheat time
                ht = np.random.normal(HT_mac, 0.1 * HT_mac, 1)
                # cool time
                ct = np.random.normal(CT_mac, 0.1 * CT_mac, 1)

                # processing time
                P_rand_value = st + ht + bt + ct + dt
                P_rand.append(P_rand_value)

                # completion time
                if i == 0:
                    C_rand_value = P_rand[0]
                    C_rand.append(C_rand_value)
                else:
                    C_rand_value = P_rand[i] + C_rand[i - 1]
                    C_rand.append(C_rand_value)

                # delay
                Del = []

                for k in batch.get_parts_id():

                    c_rand_value = C_rand[i]

                    if c_rand_value - d[k] > 0:
                        delay = 1
                    else:
                        delay = 0

                    tardy_part_count_sce += delay
                    Del.append(delay)

                for p in range(get_part_num(machines, m, i)):
                    if (m, i, p) not in part_delay:

                        part_delay[(m, i, p)] = 0

                    part_delay[(m, i, p)] += Del[p]

            for b in range(get_bat_num(machines, m)):
                if (m, b) not in batch_delay:

                    batch_delay[(m, b)] = 0

                for p in range(get_part_num(machines, m, b)):
                    batch_delay[(m, b)] += part_delay[(m, b, p)]

        tardy_part_count.append(tardy_part_count_sce)

        "termination conditions"
        if j >= 99:
            # 95% CI
            z_sco = 2.576
            conf_interval_width = z_sco * np.std(tardy_part_count) / np.sqrt(len(tardy_part_count))
            error = conf_interval_width / np.mean(tardy_part_count)

            if error <= 0.05:
                break

    tardy_part_count_mean = np.mean(tardy_part_count)

    key_with_max_value = max(part_delay, key=part_delay.get)

    (mac_id, bat_id, part_index) = key_with_max_value
    machine = get_machine_by_id(machines, mac_id)
    batch = machine.get_batch_by_id(bat_id)

    max_part_id, max_part_or = batch.get_part_by_index(part_index)

    key_with_bat_value = max(batch_delay, key=batch_delay.get)

    (max_mac_id, max_bat_id) = key_with_bat_value

    return tardy_part_count_mean, (mac_id, bat_id, max_part_id, max_part_or), (max_mac_id, max_bat_id)


def Monte_Carlo_Evaluation(v1, v2, S1, S2, T1, T2, D1, D2, NL, y_slice, Dp, Ds, Tr, v, a, h, s, d, machines):
    # Monte Carlo simulation with high precision

    bt_mod = []
    ht_mod =[]
    ct_mod = []

    for m in range(get_mac_num(machines)):
        bt_mod_mac = []
        ht_mod.append(0.0838 * (T2[m]**2 - T1[m]**2) + 2.364 * (T2[m] - T1[m]))
        ct_mod.append(0.5048 * (T1[m]**2 - T2[m]**2) - 192.96 * (T1[m] - T2[m]))

        machine = get_machine_by_id(machines, m)

        for i in range(get_bat_num(machines, m)):
            if get_bat_num(machines, m) == 0:
                break
            batch = machine.get_batch_by_id(i)
            part_ori_mac_bat = batch.get_parts_id_ori()

            a_parts_bat = []
            v_parts_bat = []
            s_parts_bat = []
            h_parts_bat = []

            for (p, o) in part_ori_mac_bat:
                a_parts_bat.append(a[p])
                v_parts_bat.append(v[p])
                s_parts_bat.append(s[p, o])
                h_parts_bat.append(h[p, o])
            a_bat_sum = sum(a_parts_bat)
            v_bat_sum = sum(v_parts_bat)
            s_bat_sum = sum(s_parts_bat)
            if h_parts_bat:
                h_bat_max = max(h_parts_bat)
            else:
                h_bat_max = 0

            bt_model = 2 * a_bat_sum / (NL[m] * y_slice[m] * v1[m]) + v_bat_sum / (NL[m] * y_slice[m] * v2[m] * Dp[m]) + s_bat_sum / (NL[m] * y_slice[m] * v2[m] * Ds[m]) + h_bat_max * Tr[m] / y_slice[m]
            bt_mod_mac.append(bt_model)

        bt_mod.append(bt_mod_mac)

    num_samples = 5000

    np.random.seed(42)

    tardy_part_count = []

    part_delay = {}

    for j in range(num_samples):

        tardy_part_count_sce = 0

        for m in range(get_mac_num(machines)):

            machine = get_machine_by_id(machines, m)

            B_mac = bt_mod[m]
            S1_mac = S1[m]
            S2_mac = S2[m]
            D1_mac = D1[m]
            D2_mac = D2[m]
            HT_mac = ht_mod[m]
            CT_mac = ct_mod[m]

            P_rand = []
            C_rand = []

            for i in range(get_bat_num(machines, m)):

                batch = machine.get_batch_by_id(i)

                B_bat = B_mac[i]

                st = np.random.uniform(S1_mac, S2_mac, 1)

                dt = np.random.uniform(D1_mac, D2_mac, 1)

                bt = np.random.normal(B_bat, 0.1 * B_bat, 1)

                ht = np.random.normal(HT_mac, 0.1 * HT_mac, 1)

                ct = np.random.normal(CT_mac, 0.1 * CT_mac, 1)

                P_rand_value = st + ht + bt + ct + dt
                P_rand.append(P_rand_value)

                if i == 0:

                    C_rand_value = P_rand[0]
                    C_rand.append(C_rand_value)
                else:

                    C_rand_value = P_rand[i] + C_rand[i - 1]
                    C_rand.append(C_rand_value)

                Del = []

                for k in batch.get_parts_id():

                    c_rand_value = C_rand[i]

                    if c_rand_value - d[k] > 0:
                        delay = 1
                    else:
                        delay = 0

                    tardy_part_count_sce += delay
                    Del.append(delay)

                for p in range(get_part_num(machines, m, i)):
                    if (m, i, p) not in part_delay:

                        part_delay[(m, i, p)] = [0, 0, 0]

                        part_id = batch.get_part_by_index(p)[0]
                        part_delay[(m, i, p)][0] = part_id
                        part_delay[(m, i, p)][2] = d[part_id]

                    part_delay[(m, i, p)][1] += Del[p]

        tardy_part_count.append(tardy_part_count_sce)

        if j >= 999:

            z_sco = 2.576
            conf_interval_width = z_sco * np.std(tardy_part_count) / np.sqrt(len(tardy_part_count))
            error = conf_interval_width / np.mean(tardy_part_count)

            if error <= 0.01:
                print("\nsamples:", j + 1)
                break

    tardy_part_count_mean = np.mean(tardy_part_count)

    new_list = list(tardy_part_count)
    print("\nNumber of Scenario:", len(new_list))

    plt.hist(new_list, bins=max(new_list)-min(new_list), density=True, color='skyblue', align='mid', rwidth=0.8)

    ax = plt.gca()

    ax.set_xticks([x + 0.5 for x in range(min(new_list), max(new_list) + 1)])

    ax.set_xticklabels(range(min(new_list), max(new_list) + 1))

    plt.xlabel('Tardy Parts Count')
    plt.ylabel('Frequency')

    plt.axvline(x = sum(new_list) / len(new_list) + 0.5, color='darkblue', linestyle='--',
                label=f'Mean With Uncertainty: {sum(new_list) / len(new_list):.2f} ')

    plt.text(sum(new_list) / len(new_list) + 0.5, 0.08, f'{sum(new_list) / len(new_list):.2f}', color='darkblue', ha='right', va='bottom')
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3)

    plt.savefig('object_distribution_sim.svg', bbox_inches='tight')

    plt.show()

    with open("output1_sim.txt", "w") as file:

        file.write(str(tardy_part_count) + ",\n")

        file.write("objective value estimation")
        file.write(str(tardy_part_count_mean) + ",\n")

        file.write("list of objective value estimation under scenarios")
        file.write(str(part_delay) + ",\n")

    return tardy_part_count_mean, part_delay


"Greedy Insertion (GI)"
def Min_DR_batch_Insert_HN(K, L, W, HM, h, l, w, v1, v2, S1, S2, T1, T2, D1, D2, NL, y_slice, Dp, Ds, Tr, v, a, s, d, machines, part_id):

    all_batch_num = 0
    for mac in range(get_mac_num(machines)):
        all_batch_num += get_bat_num(machines, mac)
    if all_batch_num == 0:
        random.seed()
        m = random.choice(range(get_mac_num(machines)))
        for machine in machines:
            if machine.machine_id == m:
                machine.add_batch(0)
                for batch in machine.batches:
                    if batch.batch_id == 0:
                        random.seed()
                        k = random.choice(K[part_id])
                        batch.add_part(part_id, k, [0, 0], 'nr')
                        return machines, m, 0

    # run Monte Carlo simulation "Monte_Carlo_Tempo_DR" on tempory solution to get the batch with the lowest contribution to objective
    (m, b) = Monte_Carlo_Tempo_DR(v1, v2, S1, S2, T1, T2, D1, D2, NL, y_slice, Dp, Ds, Tr, v, a, h, s, d, machines)[1]

    random.seed()
    k = random.choice(K[part_id])

    for machine in machines:
        if machine.machine_id == m:
            for batch in machine.batches:
                if batch.batch_id == b:
                    batch.add_part(part_id, k, [0, 0], 'nr')

                    parts_id_or = batch.get_parts_id_ori()

                    bin_sizes = []
                    h_p = []
                    l_p = []
                    w_p = []

                    for (n, o) in parts_id_or:
                        l_p.append(l[n, o])
                        w_p.append(w[n, o])
                        h_p.append(h[n, o])
                        # match the put-in form of best-fit
                        bin_sizes.append(l[n, o])
                        bin_sizes.append(w[n, o])

                    batch.parts = []
                    # Simple feasibility check for build volume
                    h_check = all_smaller_than(h_p, HM[m])

                    area_part = [x * y for x, y in zip(l_p, w_p)]
                    area_batch = sum(area_part)
                    if h_check and area_batch <= W[m] * L[m]:
                        # call BestFitRotate
                        item_coor_list = BestFitRotate(bin_sizes, L[m])

                        for i in range(int(len(item_coor_list) / 4)):
                            # get coordinate of part i
                            x1, y1, x2, y2 = item_coor_list[i * 4: (i + 1) * 4]

                            if y2 > W[m]:
                                return "infeasible", m, b
                            (p, o) = parts_id_or[i]

                            tol = 0.05
                            if abs(x2 - x1 - l_p[i]) <= tol and abs(y2 - y1 - w_p[i]) <= tol:
                                P_ro = "nr"  # not rotated
                            else:
                                P_ro = "r"  # rotated
                            batch.add_part(p, o, [x1, y1], P_ro)

                    else:
                        return "infeasible", m, b
    return machines, m, b


def Monte_Carlo_Tempo_DR(v1, v2, S1, S2, T1, T2, D1, D2, NL, y_slice, Dp, Ds, Tr, v, a, h, s, d, machines):
    # MCS for "Greedy Insertion (GI)"

    bt_mod = []

    for m in range(get_mac_num(machines)):
        bt_mod_mac = []
        machine = get_machine_by_id(machines, m)

        for i in range(get_bat_num(machines, m)):
            if get_bat_num(machines, m) == 0:
                break
            batch = machine.get_batch_by_id(i)

            part_ori_mac_bat = batch.get_parts_id_ori()

            a_parts_bat = []
            v_parts_bat = []
            s_parts_bat = []
            h_parts_bat = []

            for (p, o) in part_ori_mac_bat:
                a_parts_bat.append(a[p])
                v_parts_bat.append(v[p])
                s_parts_bat.append(s[p, o])
                h_parts_bat.append(h[p, o])

            a_bat_sum = sum(a_parts_bat)
            v_bat_sum = sum(v_parts_bat)
            s_bat_sum = sum(s_parts_bat)
            if h_parts_bat:
                h_bat_max = max(h_parts_bat)
            else:
                h_bat_max = 0

            bt_model = 2 * a_bat_sum / (NL[m] * y_slice[m] * v1[m]) + v_bat_sum / (NL[m] * y_slice[m] * v2[m] * Dp[m]) + s_bat_sum / (NL[m] * y_slice[m] * v2[m] * Ds[m]) + h_bat_max * Tr[m] / y_slice[m]
            bt_mod_mac.append(bt_model)

        bt_mod.append(bt_mod_mac)

    num_samples = 500

    np.random.seed(42)

    tardy_part_count = []

    part_delay = {}

    batch_delay = {}

    for j in range(num_samples):

        tardy_part_count_sce = 0

        for m in range(get_mac_num(machines)):

            machine = get_machine_by_id(machines, m)

            B_mac = bt_mod[m]
            S1_mac = S1[m]
            S2_mac = S2[m]
            D1_mac = D1[m]
            D2_mac = D2[m]
            HT_mac = 0.0838 * ((T2[m] * T2[m]) - (T1[m] * T1[m])) + 2.364 * (T2[m] - T1[m])
            CT_mac = 0.5048 * ((T1[m] * T1[m]) - (T2[m] * T2[m])) - 192.96 * (T1[m] - T2[m])

            P_rand = []
            C_rand = []

            for i in range(get_bat_num(machines, m)):

                batch = machine.get_batch_by_id(i)

                B_bat = B_mac[i]

                st = np.random.uniform(S1_mac, S2_mac, 1)

                dt = np.random.uniform(D1_mac, D2_mac, 1)

                bt = np.random.normal(B_bat, 0.1 * B_bat, 1)

                ht = np.random.normal(HT_mac, 0.1 * HT_mac, 1)

                ct = np.random.normal(CT_mac, 0.1 * CT_mac, 1)

                P_rand_value = st + ht + bt + ct + dt
                P_rand.append(P_rand_value)

                if i == 0:

                    C_rand_value = P_rand[0]
                    C_rand.append(C_rand_value)
                else:

                    C_rand_value = P_rand[i] + C_rand[i - 1]
                    C_rand.append(C_rand_value)

                Del = []

                for k in batch.get_parts_id():

                    c_rand_value = C_rand[i]

                    duedate = d[k]

                    if c_rand_value - duedate > 0:
                        delay = 1
                    else:
                        delay = 0

                    tardy_part_count_sce += delay
                    Del.append(delay)

                for p in range(get_part_num(machines, m, i)):
                    if (m, i, p) not in part_delay:

                        part_delay[(m, i, p)] = 0

                    part_delay[(m, i, p)] += Del[p]

            for b in range(get_bat_num(machines, m)):
                if (m, b) not in batch_delay:

                    batch_delay[(m, b)] = 0

                for p in range(get_part_num(machines, m, b)):
                    batch_delay[(m, b)] += part_delay[(m, b, p)]

        tardy_part_count.append(tardy_part_count_sce)

        if j >= 99:
            z_sco = 2.576
            conf_interval_width = z_sco * np.std(tardy_part_count) / np.sqrt(len(tardy_part_count))
            error = conf_interval_width / np.mean(tardy_part_count)

            if error <= 0.05:
                break

    tardy_part_count_mean = np.mean(tardy_part_count)

    key_with_min_value = min(batch_delay, key=batch_delay.get)

    (min_mac_id, min_bat_id) = key_with_min_value

    return tardy_part_count_mean, (min_mac_id, min_bat_id)


random.seed()

path = "TestInstances/Calibration/2-20.txt"  # adjust by user

# call readInstance
[types_mac, types_parts, num_mac, num_parts, v1, v2, S1, S2, T1, T2, D1, D2, L, W, HM, NL, y_slice, Dp, Ds, Tr, v, a, Kj, h, l, w, s] = readInstance(path)

# call GenerateDueDate
d = GenerateDueDate(path, 0.3, 0.3, 1)
# d = GenerateDueDate(filePath, TF, RDD, RndSeed)

# Sets of Machines
I = [i for i in range(num_mac)]
# Sets of Parts
J = [i for i in range(num_parts)]
# dict of list of each part's alternative build orientations
K = dict()
for j in range(len(J)):
    K[j] = [k2 for k1, k2 in Kj.keys() if k1 == j]

"RUN!!!"
all_best_sol = ALNS(I, J, K, L, W, HM, v1, v2, S1, S2, T1, T2, D1, D2, NL, y_slice, Dp, Ds, Tr, v, a, Kj, h, l, w, s, d)
"Evaluation"
print(Monte_Carlo_Evaluation(v1, v2, S1, S2, T1, T2, D1, D2, NL, y_slice, Dp, Ds, Tr, v, a, h, s, d, all_best_sol))

