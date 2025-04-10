import math
import random
import joblib

from BinPackingHeu import *


class Machine:
    def __init__(self, machine_id):
        self.machine_id = machine_id
        self.batches = []

    def add_batch(self, batch_id):
        batch = Batch(batch_id)
        self.batches.append(batch)

    def add_part_to_batch(self, batch_id, part_id, part_direction_id, part_position, part_rotation):
        for batch in self.batches:
            if batch.batch_id == batch_id:
                batch.add_part(part_id, part_direction_id, part_position, part_rotation)
                return

        # If batch_id not found, create a new batch and add the part
        self.add_batch(batch_id)
        self.add_part_to_batch(batch_id, part_id, part_direction_id, part_position, part_rotation)

    def get_batch_by_id(self, batch_id):
        for batch in self.batches:
            if batch.batch_id == batch_id:
                return batch
        return None  # 如果未找到指定批次编号的批次，返回None

    def remove_batch_by_id(self, batch_id):
        for batch in self.batches:
            if batch.batch_id == batch_id:
                self.batches.remove(batch)
                return

    def remove_batch_and_get_parts(self, batch_id):
        # 找到机器上的指定批次
        batch = self.get_batch_by_id(batch_id)
        if batch:
            # 获取批次上的零件编号列表
            parts_ids = batch.get_parts_id()
            # 清空这个批次
            batch.parts = []
            # 返回零件编号列表
            return parts_ids
        else:
            # 如果未找到指定批次，则返回空列表
            return []


class Batch:
    def __init__(self, batch_id):
        self.batch_id = batch_id
        self.parts = []

    def add_part(self, part_id, part_direction_id, part_position, part_rotation):
        part = Part(part_id, part_direction_id, part_position, part_rotation)
        self.parts.append(part)

    def get_parts_id_ori(self):
        parts_info = []
        for part in self.parts:
            parts_info.append((part.part_id, part.part_direction_id))
        return parts_info

    def get_parts_id(self):
        parts_ids = []
        for part in self.parts:
            parts_ids.append((part.part_id))
        return parts_ids

    def get_part_by_index(self, index):
        if 0 <= index < len(self.parts):
            part = self.parts[index]
            return (part.part_id, part.part_direction_id)
        else:
            return None


class Part:
    def __init__(self, part_id, part_direction_id, part_position, part_rotation):
        self.part_id = part_id
        self.part_direction_id = part_direction_id
        self.part_position = part_position
        self.part_rotation = part_rotation


def compare_machines(machines1, machines2):

    for machine1, machine2 in zip(machines1, machines2):
        # 判断批次数量是否相同
        if len(machine1.batches) != len(machine2.batches):
            return False

        # 逐个比较每个批次
        for batch1, batch2 in zip(machine1.batches, machine2.batches):
            # 判断批次ID是否相同
            if batch1.batch_id != batch2.batch_id:
                return False

            # 获取批次中的部件信息列表
            parts_info1 = batch1.get_parts_id_ori()
            parts_info2 = batch2.get_parts_id_ori()

            # 判断部件信息列表是否相同
            if parts_info1 != parts_info2:
                return False

    # 如果所有条件都满足，则认为两个机器相同
    return True


def parse_solution(solution):
    # 将字符串转化为实例
    machines = []
    for machine_data in solution:
        machine_id = machine_data[0]
        machine = Machine(machine_id)
        batches_data = machine_data[1:]
        for batch_data in batches_data:
            batch_id = batch_data[0]
            machine.add_batch(batch_id)
            parts_data = batch_data[1:]
            for part_data in parts_data:
                part_id, part_direction_id, part_position, part_rotation = part_data
                machine.add_part_to_batch(batch_id, part_id, part_direction_id, part_position, part_rotation)
        machines.append(machine)
    return machines


def get_machine_by_id(machines, m):
    for machine in machines:
        if machine.machine_id == m:
            return machine
    return None


def get_mac_num(machines):
    return len(machines)


def get_bat_num(machines, m):
    for machine in machines:
        if machine.machine_id == m:
            return len(machine.batches)
    return 0  # 如果未找到指定机器编号m的机器，返回0批次


def get_part_num(machines, m, b):
    for machine in machines:
        if machine.machine_id == m:
            for batch in machine.batches:
                if batch.batch_id == b:
                    return len(batch.parts)
    return 0  # 如果未找到指定机器编号m或批次编号b的零件，返回0个零件


"破环算子"


def remove_random_part(machines):
    # 随机移除零件
    # 使用系统时间作为种子
    random.seed()
    # # 如果解不存在（已规避）
    # if not machines:
    #     return [], None

    machine = random.choice(machines)

    # 如果该机器下没有批次，则循环到有批次的机器为止
    while not machine.batches:
        random.seed()
        machine = random.choice(machines)

    random.seed()
    batch = random.choice(machine.batches)

    # # 算法已规避了批次没有零件的情况
    # if not batch.parts:
    #     return machines, None

    random.seed()
    part = random.choice(batch.parts)
    batch.parts.remove(part)

    return machines, part.part_id


def remove_max_TT_part(machines, max_TT_part):
    # 移除最大目标贡献的零件
    max_TT_machine_id, max_TT_batch_id, max_TT_part_id, max_TT_part_direction_id = max_TT_part

    for machine in machines:
        if machine.machine_id == max_TT_machine_id:
            for batch in machine.batches:
                if batch.batch_id == max_TT_batch_id:
                    for part in batch.parts:
                        if (part.part_id == max_TT_part_id and
                                part.part_direction_id == max_TT_part_direction_id):
                            batch.parts.remove(part)
                            return machines, max_TT_part_id

    return machines, max_TT_part_id


def remove_random_batch(machines):
    # 移除随机批次

    # 使用系统时间作为种子
    random.seed()
    machine = random.choice(machines)

    # 如果该机器下没有批次，则循环到有批次的机器为止
    while not machine.batches:
        random.seed()
        machine = random.choice(machines)

    random.seed()
    batch = random.choice(machine.batches)

    # 移除这个批次上的所有零件,并获取这个批次的零件编号列表
    parts_ids = machine.remove_batch_and_get_parts(batch.batch_id)

    return machines, parts_ids


def remove_max_T_batch(machines, max_T_batch):
    # 移除最大目标贡献的批次
    max_T_machine_id, max_T_batch_id = max_T_batch

    for machine in machines:
        if machine.machine_id == max_T_machine_id:
            for batch in machine.batches:
                if batch.batch_id == max_T_batch_id:
                    # 移除这个批次上的所有零件,并获取这个批次的零件编号列表
                    remove_parts = machine.remove_batch_and_get_parts(batch.batch_id)
                    return machines, remove_parts


def remove_max_height_variance_batch(machines, h):
    # 移除最大高度方差批次
    max_var_machine = []
    max_bat_id = []
    for machine in machines:
        # 存储每个机器上各批次高度方差
        height_var_batch = []
        # 如果机器上有批次
        if machine.batches:
            for batch in machine.batches:
                height_batch = []
                for part in batch.parts:
                    height_batch.append(h[part.part_id, part.part_direction_id])
                mean_h_batch = sum(height_batch) / len(height_batch)
                height_var_part = []
                for part in batch.parts:
                    height_var_part.append(((h[part.part_id, part.part_direction_id] - mean_h_batch) ** 2) / len(height_batch))
                height_var_batch.append(sum(height_var_part))
        else:
            # 否则给该机器上最大批次高度方差置0
            height_var_batch.append(0)

        # 获取该机器上批次高度方差最大值和批次编号
        max_var_batch = max(height_var_batch)
        bat_id = height_var_batch.index(max_var_batch)

        max_var_machine.append(max_var_batch)
        max_bat_id.append(bat_id)

    # 获取所有机器上批次高度方差最大值
    max_var = max(max_var_machine)
    # 获取所在机器id
    mac_id = max_var_machine.index(max_var)
    # 获取批次id
    mac_bat_id = max_bat_id[mac_id]

    parts_ids = []
    for machine in machines:
        if machine.machine_id == mac_id:
            # 移除这个具有最大高度方差的批次上的所有零件,并获取这个批次的零件编号列表
            parts_ids = machine.remove_batch_and_get_parts(mac_bat_id)

    return machines, mac_id, mac_bat_id, parts_ids


"辅助操作算子"


def remove_non_part_batch(machines):
    # 移除没有零件的批次并更新该机器下的批次编号
    for machine in machines:
        for i in range(len(machine.batches) - 1, -1, -1):
            batch = machine.batches[i]
            if not batch.parts:
                # Remove the batch with no parts
                del machine.batches[i]
                # Update batch_id for subsequent batches on the same machine
                for j in range(i, len(machine.batches)):
                    machine.batches[j].batch_id -= 1

    return machines


def all_smaller_than(lst, threshold):
    # 构造初始可行解所用
    return all(x <= threshold for x in lst)


"修复算子"


def insert_random_part(K, L, W, HM, h, l, w, machines, part_id):
    # 随机插入

    # 极端情况，若所有机器上都没有批次，则直接在随机的机器上创建批次0，并插入这个零件
    all_batch_num = 0
    for mac in range(get_mac_num(machines)):
        all_batch_num += get_bat_num(machines, mac)
    if all_batch_num == 0:
        # 使用系统时间作为种子
        random.seed()
        m = random.choice(range(get_mac_num(machines)))
        for machine in machines:
            if machine.machine_id == m:
                machine.add_batch(0)
                for batch in machine.batches:
                    if batch.batch_id == 0:
                        # 使用系统时间作为种子
                        random.seed()
                        # 随机选择零件方向
                        k = random.choice(K[part_id])
                        batch.add_part(part_id, k, [0, 0], 'nr')
                        return machines, m, 0
    # 使用系统时间作为种子
    random.seed()
    # 随机选择零件方向
    k = random.choice(K[part_id])
    # 随机选择批次
    random.seed()
    machine = random.choice(machines)
    # 读机器id
    m = machine.machine_id

    # 读取批次数量
    b_num = get_bat_num(machines, m)

    # 在末尾加上一个空批次,编号等于b_num（如果这个机器下面一个批次都没有，给他创建一个）
    machine.add_batch(b_num)

    # 随机选一个批次
    random.seed()
    batch = random.choice(machine.batches)
    # 读批次id
    b = batch.batch_id

    # 这里加一个模块，使得随机的可能中存在新建立一个批次的可能
    if b != b_num:
        # 把这个末尾批次删了
        machine.remove_batch_by_id(b_num)
        # 插入到批次，其中，位置和旋转预先选了，后续再利用bestfit调整
        machine.add_part_to_batch(b, part_id, k, [0, 0], 'nr')
        # 获取该批次上所有零件的零件编号和方向编号
        parts_id_or = batch.get_parts_id_ori()
        # 装箱算法输入
        bin_sizes = []
        h_p = []
        l_p = []
        w_p = []
        # 获取每个零件的4参数，get each part n's info on batch b
        for (n, o) in parts_id_or:
            l_p.append(l[n, o])
            w_p.append(w[n, o])
            h_p.append(h[n, o])
            # 作为装箱的输入形式，match the put-in form of best-fit
            bin_sizes.append(l[n, o])
            bin_sizes.append(w[n, o])
        # 清空这个批次
        batch.parts = []
        # 基于空间的简单可行性检查，Simple feasibility check for build volume
        h_check = all_smaller_than(h_p, HM[m])
        # 对应位置的值一一相乘，生成新列表area_part
        area_part = [x * y for x, y in zip(l_p, w_p)]
        area_batch = sum(area_part)
        if h_check and area_batch <= W[m] * L[m]:
            # 调用BestFitRotate装箱
            item_coor_list = BestFitRotate(bin_sizes, L[m])

            for i in range(int(len(item_coor_list) / 4)):
                # get coordinate of part i
                x1, y1, x2, y2 = item_coor_list[i * 4: (i + 1) * 4]

                # 判断装箱是否可行
                if y2 > W[m]:
                    return "infeasible", m, b
                # 获取part i的零件编号，方向编号
                (p, o) = parts_id_or[i]

                tol = 0.05
                if abs(x2 - x1 - l_p[i]) <= tol and abs(y2 - y1 - w_p[i]) <= tol:
                    P_ro = "nr"  # not rotated
                else:
                    P_ro = "r"  # rotated
                # 逐个添加零件到批次
                batch.add_part(p, o, [x1, y1], P_ro)

        else:
            return "infeasible", m, b
    else:
        # 把零件放进这个末尾批次
        batch.add_part(part_id, k, [0, 0], 'nr')

    return machines, m, b


def Max_Earlist_DD_Insert(K, L, W, HM, h, l, w, machines, part_id, d):
    # 最大批次内最早交付期限插入

    # 极端情况，若所有机器上都没有批次，则直接在随机的机器上创建批次0，并插入这个零件
    all_batch_num = 0
    for mac in range(get_mac_num(machines)):
        all_batch_num += get_bat_num(machines, mac)
    if all_batch_num == 0:
        # 使用系统时间作为种子
        random.seed()
        m = random.choice(range(get_mac_num(machines)))
        for machine in machines:
            if machine.machine_id == m:
                machine.add_batch(0)
                for batch in machine.batches:
                    if batch.batch_id == 0:
                        # 使用系统时间作为种子
                        random.seed()
                        # 随机选择零件方向
                        k = random.choice(K[part_id])
                        batch.add_part(part_id, k, [0, 0], 'nr')
                        return machines, m, 0
    else:
        EDD_all = []
        bat_id_all = []
        for machine in machines:
            EDD_mac = []
            if machine.batches:
                for batch in machine.batches:
                    DD_batch = []
                    if batch.parts:
                        for part in batch.parts:
                            DD_batch.append(d[part.part_id])
                        EDD_batch = min(DD_batch)
                        EDD_mac.append(EDD_batch)
                    else:
                        EDD_mac.append(0)
            else:
                EDD_mac.append(0)
            max_EDD_mac = max(EDD_mac)
            bat_id_mac = EDD_mac.index(max_EDD_mac)

            EDD_all.append(max_EDD_mac)
            bat_id_all.append(bat_id_mac)

        max_EDD_all = max(EDD_all)
        mac_id = EDD_all.index(max_EDD_all)
        bat_id = bat_id_all[mac_id]

        # 使用系统时间作为种子
        random.seed()
        # 随机选择零件方向
        k = random.choice(K[part_id])
        # 装箱与可行性检查
        # 找到该批次，插入该方向的该零件
        for machine in machines:
            if machine.machine_id == mac_id:
                for batch in machine.batches:
                    if batch.batch_id == bat_id:
                        batch.add_part(part_id, k, [0, 0], 'nr')
                        # 获取该批次上所有零件的零件编号和方向编号
                        parts_id_or = batch.get_parts_id_ori()
                        # 装箱算法输入
                        bin_sizes = []
                        h_p = []
                        l_p = []
                        w_p = []
                        # 获取每个零件的4参数，get each part n's info on batch b
                        for (n, o) in parts_id_or:
                            l_p.append(l[n, o])
                            w_p.append(w[n, o])
                            h_p.append(h[n, o])
                            # 作为装箱的输入形式，match the put-in form of best-fit
                            bin_sizes.append(l[n, o])
                            bin_sizes.append(w[n, o])
                        # 清空这个批次
                        batch.parts = []
                        # 基于空间的简单可行性检查，Simple feasibility check for build volume
                        h_check = all_smaller_than(h_p, HM[mac_id])
                        # 对应位置的值一一相乘，生成新列表area_part
                        area_part = [x * y for x, y in zip(l_p, w_p)]
                        area_batch = sum(area_part)
                        if h_check and area_batch <= W[mac_id] * L[mac_id]:
                            # 调用BestFitRotate装箱
                            item_coor_list = BestFitRotate(bin_sizes, L[mac_id])

                            for i in range(int(len(item_coor_list) / 4)):
                                # get coordinate of part i
                                x1, y1, x2, y2 = item_coor_list[i * 4: (i + 1) * 4]

                                # 判断装箱是否可行
                                if y2 > W[mac_id]:
                                    return "infeasible", mac_id, bat_id
                                # 获取part i的零件编号，方向编号
                                (p, o) = parts_id_or[i]

                                # 判断是否旋转
                                tol = 0.05
                                if abs(x2 - x1 - l_p[i]) <= tol and abs(y2 - y1 - w_p[i]) <= tol:
                                    P_ro = "nr"  # not rotated
                                else:
                                    P_ro = "r"  # rotated
                                # 逐个添加零件到批次
                                batch.add_part(p, o, [x1, y1], P_ro)

                        else:
                            return "infeasible", mac_id, bat_id

    return machines, mac_id, bat_id


def Min_H_Var_Insert(K, L, W, HM, h, l, w, machines, part_id):
    # 最小批次高度方差插入

    # 极端情况，若所有机器上都没有批次，则直接在随机的机器上创建批次0，并插入这个零件
    all_batch_num = 0
    for mac in range(get_mac_num(machines)):
        all_batch_num += get_bat_num(machines, mac)
    if all_batch_num == 0:
        # 使用系统时间作为种子
        random.seed()
        m = random.choice(range(get_mac_num(machines)))
        for machine in machines:
            if machine.machine_id == m:
                machine.add_batch(0)
                for batch in machine.batches:
                    if batch.batch_id == 0:
                        # 使用系统时间作为种子
                        random.seed()
                        # 随机选择零件方向
                        k = random.choice(K[part_id])
                        batch.add_part(part_id, k, [0, 0], 'nr')
                        return machines, m, 0, k

    # 存储各方向上最小方差和所在机器id
    min_k_list = []
    mac_id_list = []

    # 存储各个方向下各机器下最小值对应的批次id（二维数组）
    mac_bat_id_list = []

    # 遍历零件的每个方向
    for k in K[part_id]:
        # 获取该方向的高度
        h_k = h[part_id, k]

        # 存储该方向各机器上最小方差和所在批次id
        min_mac_list = []
        bat_id_list = []

        # 遍历所有机器上的所有批次
        for machine in machines:

            # 存储该方向该机器上批次高度方差
            h_var_mac = []

            # 若机器上有批次
            if machine.batches:

                for batch in machine.batches:
                    # 算法已经保证批次上均有零件

                    # 批次高度列表
                    h_batch = []

                    for part in batch.parts:
                        h_batch.append(h[part.part_id, part.part_direction_id])
                    h_batch.append(h_k)

                    # 计算批次高度均值
                    mean_h_batch = sum(h_batch) / len(h_batch)

                    height_var_part = []

                    for part in batch.parts:
                        height_var_part.append(((h[part.part_id, part.part_direction_id] - mean_h_batch) ** 2) / len(h_batch))

                    h_var_batch = sum(height_var_part)

                    # 该方向该机器下的批次高度方差列表
                    h_var_mac.append(h_var_batch)

            else:
                h_var_mac.append(10000)

            # 获取该机器下最小值和所在批次id
            min_mac = min(h_var_mac)
            bat_id = h_var_mac.index(min_mac)

            min_mac_list.append(min_mac)
            bat_id_list.append(bat_id)

        mac_bat_id_list.append(bat_id_list)

        # 获取该方向下最小值和所在机器id
        min_k = min(min_mac_list)
        mac_id = min_mac_list.index(min_k)

        min_k_list.append(min_k)
        mac_id_list.append(mac_id)

    # 获取全局最小值和所在方向id
    min_all = min(min_k_list)
    k_id = min_k_list.index(min_all)

    # 获取对应的机器id批次id
    m_id = mac_id_list[k_id]
    b_id = mac_bat_id_list[k_id][m_id]

    # 装箱与可行性检查
    # 找到该批次，插入该方向的该零件
    for machine in machines:
        if machine.machine_id == m_id:
            for batch in machine.batches:
                if batch.batch_id == b_id:
                    batch.add_part(part_id, k_id, [0, 0], 'nr')
                    # 获取该批次上所有零件的零件编号和方向编号
                    parts_id_or = batch.get_parts_id_ori()
                    # 装箱算法输入
                    bin_sizes = []
                    h_p = []
                    l_p = []
                    w_p = []
                    # 获取每个零件的4参数，get each part n's info on batch b
                    for (n, o) in parts_id_or:
                        l_p.append(l[n, o])
                        w_p.append(w[n, o])
                        h_p.append(h[n, o])
                        # 作为装箱的输入形式，match the put-in form of best-fit
                        bin_sizes.append(l[n, o])
                        bin_sizes.append(w[n, o])
                    # 清空这个批次
                    batch.parts = []
                    # 基于空间的简单可行性检查，Simple feasibility check for build volume
                    h_check = all_smaller_than(h_p, HM[m_id])
                    # 对应位置的值一一相乘，生成新列表area_part
                    area_part = [x * y for x, y in zip(l_p, w_p)]
                    area_batch = sum(area_part)
                    if h_check and area_batch <= W[m_id] * L[m_id]:
                        # 调用BestFitRotate装箱
                        item_coor_list = BestFitRotate(bin_sizes, L[m_id])

                        for i in range(int(len(item_coor_list) / 4)):
                            # get coordinate of part i
                            x1, y1, x2, y2 = item_coor_list[i * 4: (i + 1) * 4]

                            # 判断装箱是否可行
                            if y2 > W[m_id]:
                                return "infeasible", m_id, b_id, k_id
                            # 获取part i的零件编号，方向编号
                            (p, o) = parts_id_or[i]

                            # 判断是否旋转
                            tol = 0.05
                            if abs(x2 - x1 - l_p[i]) <= tol and abs(y2 - y1 - w_p[i]) <= tol:
                                P_ro = "nr"  # not rotated
                            else:
                                P_ro = "r"  # rotated
                            # 逐个添加零件到批次
                            batch.add_part(p, o, [x1, y1], P_ro)

                    else:
                        return "infeasible", m_id, b_id, k_id
    return machines, m_id, b_id, k_id


# def Min_TT_batch_Insert(K, L, W, HM, h, l, w, v1, v2, S, NL, y_slice, Dp, Ds, Tr, v, a, s, d, train_mean, train_std, machines, part_id):
#     # 最小TT批次插入
#     (m, b) = Monte_Carlo(v1, v2, S, NL, y_slice, Dp, Ds, Tr, v, a, h, s, d, machines, train_mean, train_std)[2]
#
#     # 使用系统时间作为种子
#     random.seed()
#     # 随机选择零件方向
#     k = random.choice(K[part_id])
#
#     # 装箱与可行性检查
#     # 找到该批次，插入该方向的该零件
#     for machine in machines:
#         if machine.machine_id == m:
#             for batch in machine.batches:
#                 if batch.batch_id == b:
#                     batch.add_part(part_id, k, [0, 0], 'nr')
#                     # 获取该批次上所有零件的零件编号和方向编号
#                     parts_id_or = batch.get_parts_id_ori()
#                     # 装箱算法输入
#                     bin_sizes = []
#                     h_p = []
#                     l_p = []
#                     w_p = []
#                     # 获取每个零件的4参数，get each part n's info on batch b
#                     for (n, o) in parts_id_or:
#                         l_p.append(l[n, o])
#                         w_p.append(w[n, o])
#                         h_p.append(h[n, o])
#                         # 作为装箱的输入形式，match the put-in form of best-fit
#                         bin_sizes.append(l[n, o])
#                         bin_sizes.append(w[n, o])
#                     # 清空这个批次
#                     batch.parts = []
#                     # 基于空间的简单可行性检查，Simple feasibility check for build volume
#                     alpha = 0.95
#                     h_check = all_smaller_than(h_p, alpha * HM[m])
#                     # 对应位置的值一一相乘，生成新列表area_part
#                     area_part = [x * y for x, y in zip(l_p, w_p)]
#                     area_batch = sum(area_part)
#                     if h_check and area_batch <= alpha * W[m] * L[m]:
#                         # 调用BestFitRotate装箱
#                         item_coor_list = BestFitRotate(bin_sizes, L[m])
#
#                         for i in range(int(len(item_coor_list) / 4)):
#                             # get coordinate of part i
#                             x1, y1, x2, y2 = item_coor_list[i * 4: (i + 1) * 4]
#
#                             # 判断装箱是否可行
#                             if y2 > W[m]:
#                                 return "infeasible", m, b
#                             # 获取part i的零件编号，方向编号
#                             (p, o) = parts_id_or[i]
#
#                             # 判断是否旋转
#                             tol = 0.05
#                             if abs(x2 - x1 - l_p[i]) <= tol and abs(y2 - y1 - w_p[i]) <= tol:
#                                 P_ro = "nr"  # not rotated
#                             else:
#                                 P_ro = "r"  # rotated
#                             # 逐个添加零件到批次
#                             batch.add_part(p, o, [x1, y1], P_ro)
#
#                     else:
#                         return "infeasible", m, b
#     return machines, m, b



# 示例用法


# # 调用解析函数
# solution = [[0, [0, [2, 1, [195.6999969482422, 0.0], 'r'], [4, 0, [0.0, 0.0], 'r'], [0, 2, [169.0, 0.0], 'nr']], [1, [6, 2, [0.0, 0.0], 'nr'], [7, 2, [43.400001525878906, 0.0], 'r']]], [1, [0, [9, 1, [0.0, 0.0], 'nr'], [3, 0, [167.39999389648438, 0.0], 'nr'], [8, 0, [87.69999694824219, 0.0], 'r']], [1, [5, 0, [77.0, 0.0], 'nr'], [1, 0, [0.0, 0.0], 'nr']]]
# machines = parse_solution(solution)
#
# # 打印解析后的数据
# for machine in machines:
#     print(f"Machine ID: {machine.machine_id}")
#     for batch in machine.batches:
#         print(f"Batch ID: {batch.batch_id}")
#         for part in batch.parts:
#             print(f"Part ID: {part.part_id}, Direction ID: {part.part_direction_id}, Position: {part.part_position}, Rotation: {part.part_rotation}")
#     print()



