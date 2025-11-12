'''
该文件实现 eMPC 控制器（经济型模型预测控制），用于在 EV2Gym 环境中
进行滚动优化调度（包含 V2G 与仅充电 G2V 两种模式）。

Authors: Cesar Diaz-Londono, Stavros Orfanoudakis
'''

import gurobipy as gp
from gurobipy import GRB
from gurobipy import *
import numpy as np

from ev2gym.baselines.mpc.mpc import MPC


class eMPC_V2G(MPC):
    """
    允许双向能量流（V2G）的 eMPC 控制器：
    决策变量含 充电功率/放电功率（连续）+ 充放互斥二元变量。
    目标：最小化购电成本、最大化售电收益（线性目标）。
    """

    def __init__(self, env, control_horizon=10, verbose=False, **kwargs):
        """
        env: EV2Gym 环境
        control_horizon: 预测时域步数 h
        verbose: 调试输出
        """
        super().__init__(env, control_horizon, verbose)

        self.na = self.n_ports                # 状态维度（端口数）
        self.nb = 2 * self.na                 # 控制维度：每端口两路(充/放) → 2n

    def get_action(self, env):
        """
        求解当前步的 V2G 经济 MPC：
        1) 构造预测模型与约束
        2) 用 Gurobi 求解线性/混整规划
        3) 取第 1 步功率归一化为动作返回
        """

        # 若不可行则通过“放松离站目标 SoC”循环重试
        while True:

            t = env.current_step
            # 1) 更新变压器限值/负荷/PV 的预测（用于后续约束）
            self.update_tr_power(t)

            # 2) 重建当前滚动步的状态起点，并生成“终端 SoC 目标”向量（V2G 版含 min SoC）
            self.reconstruct_state(t)
            self.calculate_XF_V2G(t)

            # 3) 构建站内线性模型 Amono/Bmono （V2G：充/放两通道）
            self.v2g_station_models(t)

            # 4) 计算 Gu 与 AU,bU（SoC 演化与容量/终端目标的总不等式约束）
            self.calculate_InequalityConstraints(t)

            # 5) 设置控制量上下界（V2G：LB≥0，UB=[Pch_max, Pdis_max]）
            self.set_power_limits_V2G(t)

            # 调试信息
            if self.verbose:
                self.print_info(t)

            # -------- 目标函数 f^T u（线性）：充电费用(+)、放电收益(-) --------
            f = []
            for i in range(self.control_horizon):
                for j in range(self.n_ports):
                    f.append(self.T * self.ch_prices[t + i])     # 充电项系数
                    f.append(-self.T * self.disch_prices[t + i]) # 放电项系数（负号→收益）

            f = np.array(f).reshape(-1)

            nb = self.nb
            n = self.n_ports
            h = self.control_horizon

            # --------- 建模并添加变量 ----------
            model = gp.Model("optimization_model")

            # 连续控制量 u：按时间步拼接，每步 [P_ch(1), P_dis(1), ..., P_ch(n), P_dis(n)]
            u = model.addMVar(nb*h, vtype=GRB.CONTINUOUS, name="u")

            # 二元变量 Zbin：充/放互斥（Z=1 允许充，Z=0 允许放）
            Zbin = model.addMVar(n*h, vtype=GRB.BINARY, name="Zbin")
            
            # 占位变量（当前版本未进入约束/目标）：SoC 预测、平均 SoC、循环/日历退化
            SoC   = model.addMVar(n * h, vtype=GRB.CONTINUOUS, name="SoC")
            SOCav = model.addMVar(self.EV_number, vtype=GRB.CONTINUOUS, name="SOCav")
            d_cyc = model.addMVar(self.EV_number, vtype=GRB.CONTINUOUS, name="d_cyc")
            d_cal = model.addMVar(self.EV_number, vtype=GRB.CONTINUOUS, name="d_cal")

            # --------- 约束集合 ----------
            # (A) SoC 相关的总约束：不超容量 + 达成终端目标（已由基类拼成 AU,bU）
            model.addConstr((self.AU @ u) <= self.bU, name="constr1")

            # (B) 功率非负与互斥（V2G：充/放各一路，使用二元变量切换）
            # 充电：0 ≤ u_ch ≤ UB_ch * Zbin
            model.addConstrs((0 <= u[j] for j in range(0, nb*h, 2)), name="constr3a")
            model.addConstrs((u[j] <= self.UB[j] * Zbin[j//2]
                              for j in range(0, nb*h, 2)), name="constr3b")

            # 放电：0 ≤ u_dis ≤ UB_dis * (1 - Zbin)
            model.addConstrs((0 <= u[j] for j in range(1, nb*h, 2)), name="constr4a")
            model.addConstrs((u[j] <= self.UB[j]*(1-Zbin[j//2])
                              for j in range(1, nb*h, 2)), name="constr4b")

            # (C) 变压器功率约束：Σ_ports (u_ch - u_dis) + 负荷 + PV ≤ 上限
            for tr_index in range(self.number_of_transformers):
                for i in range(self.control_horizon):
                    model.addConstr((gp.quicksum((u[j] - u[j+1])
                                                 for index, j in enumerate(range(i*self.nb, (i+1)*self.nb, 2))
                                                 if self.cs_transformers[index] == tr_index) +
                                     self.tr_loads[tr_index, i] +
                                     self.tr_pv[tr_index, i] <=
                                     self.tr_power_limit[tr_index, i]),
                                    name=f'constr5_{tr_index}_t{i}')

            # 变压器下界（对称）：Σ_ports (u_ch - u_dis) + 负荷 + PV ≥ -max_limit
            for tr_index in range(self.number_of_transformers):
                for i in range(self.control_horizon):
                    model.addConstr((gp.quicksum((u[j] - u[j+1])
                                                 for index, j in enumerate(range(i*self.nb, (i+1)*self.nb, 2))
                                                 if self.cs_transformers[index] == tr_index) +
                                     self.tr_loads[tr_index, i] +
                                     self.tr_pv[tr_index, i] >=
                                     -self.tr_power_limit[tr_index, :].max()),
                                    name=f'constr5_{tr_index}_t{i}')

            # --------- 目标与参数 ----------
            model.setObjective(f @ u, GRB.MINIMIZE)
            model.setParam('OutputFlag', self.output_flag)  # 0=静默
            # model.params.NonConvex = 2  # 若存在非凸项时可开启

            if self.MIPGap is not None:
                model.params.MIPGap = self.MIPGap
            model.params.TimeLimit = self.time_limit

            # --------- 求解 ----------
            model.optimize()
            self.total_exec_time += model.Runtime

            # --------- 可行性修复：若无解，逐步放松“离站前目标 SoC” ----------
            if model.status == GRB.Status.INF_OR_UNBD or model.status == GRB.Status.INFEASIBLE:
                print(f"INFEASIBLE or Unbounded - step{t} -- Relaxing SoC constraints - try {self.varch2}")
                flagOut = False
                varch = 0

                # 遍历端口与时间，找到“离站边界”，将其上一刻目标 SoC 按最大可充功率×步长下调
                for i in range(n):
                    for j in range(t, t + h + 1):
                        # 检测 j 为离站边界（u 从 1→0）
                        if self.x_final[i, j] == 0 and self.x_final[i, j - 1] > 0 and self.x_next[i] > 0:
                            varch += 1
                            if varch > self.varch2:
                                self.x_final[i, j - 1] = self.x_final[i, j - 1] - self.p_max_MT[i, j - 1] * self.T
                                self.varch2 += 1
                                flagOut = True
                                break
                    if flagOut:
                        break
                    if i == n - 1:
                        self.varch2 = 0
                continue  # 放松后重解

            # --------- 读解并构造动作（只取第 1 步） ----------
            a = np.zeros((nb*h, 1))
            for i in range(2*self.n_ports):
                a[i] = u[i].x  # 第 1 步对应的 2n 个充/放值

            # 归一化到 [-1,1]：充电正、放电负；并防止同端口同步充放
            actions = np.zeros(self.n_ports)
            if self.verbose:
                print(f'Actions:\n {a.reshape(-1,self.n_ports, 2)}')

            e = 0.001  # 数值阈值
            for i in range(0, 2*self.n_ports, 2):
                if a[i] > e and a[i + 1] > e:
                    raise ValueError(f'Charging and discharging at the same time {i} {a[i]} {a[i+1]}')
                elif a[i] > e:
                    actions[i//2] = a[i]/self.max_ch_power[i//2]               # 充电归一化（正）
                elif a[i + 1] > e:
                    actions[i//2] = -a[i+1]/abs(self.max_disch_power[i//2])    # 放电归一化（负）

            if self.verbose:
                print(f'actions: {actions.shape} \n {actions}')

            return actions


class eMPC_G2V(MPC):
    '''
    仅充电（G2V）的 eMPC 控制器：
    无放电通道、无互斥二元变量，线性规划更轻量。
    '''

    def __init__(self, env, control_horizon=10, verbose=False, **kwargs):
        """
        env: EV2Gym 环境
        control_horizon: 预测时域步数 h
        """
        super().__init__(env, control_horizon, verbose)

        self.na = self.n_ports     # 状态维度
        self.nb = self.na          # 控制维度：每端口一路充电 → n

    def get_action(self, env):
        """
        求解当前步的 G2V 经济 MPC：
        1) SoC/容量/终端目标约束 + 变压器限额
        2) 线性目标：最小化充电费用
        3) 取第 1 步充电功率归一化为动作
        """
        t = env.current_step

        # 1) 更新预测（变压器限额/负荷/PV）
        self.update_tr_power(t)

        # 2) 重建状态与 G2V 终端 SoC 目标（无最小 SoC）
        self.reconstruct_state(t)
        self.calculate_XF_G2V(t)

        # 3) 构建 G2V 模型 Amono/Bmono（仅充电通道）
        self.g2v_station_models(t)

        # 4) 计算 Gu 与 AU,bU
        self.calculate_InequalityConstraints(t)

        # 5) 控制量上下界（单通道）
        self.set_power_limits_G2V(t)

        if self.verbose:
            self.print_info(t)

        # -------- 目标函数：仅充电成本 --------
        f = []
        for i in range(self.control_horizon):
            for j in range(self.n_ports):
                f.append(self.T * self.ch_prices[t + i])
        f = np.array(f).reshape(-1)

        nb = self.nb
        n = self.n_ports
        h = self.control_horizon

        # -------- 建模与变量 --------
        model = gp.Model("optimization_model")
        u = model.addMVar(nb*h, vtype=GRB.CONTINUOUS, name="u")  # 充电功率

        # (A) SoC 总约束
        model.addConstr((self.AU @ u) <= self.bU, name="constr1")

        # (B) 功率上下界：0 ≤ u ≤ UB
        model.addConstr((0 <= u), name="constr2a")
        model.addConstr((u <= self.UB), name="constr2b")

        # (C) 变压器功率上/下界：Σ_ports u + 负荷 + PV ≤ 上限；≥ -max_limit
        for tr_index in range(self.number_of_transformers):
            for i in range(self.control_horizon):
                model.addConstr((gp.quicksum(u[j]
                                             for index, j in enumerate(range(i*self.nb, (i+1)*self.nb))
                                             if self.cs_transformers[index] == tr_index) +
                                 self.tr_loads[tr_index, i] +
                                 self.tr_pv[tr_index, i] <=
                                 self.tr_power_limit[tr_index, i]),
                                name=f'constr5_{tr_index}_t{i}')

        for tr_index in range(self.number_of_transformers):
            for i in range(self.control_horizon):
                model.addConstr((gp.quicksum(u[j]
                                             for index, j in enumerate(range(i*self.nb, (i+1)*self.nb))
                                             if self.cs_transformers[index] == tr_index) +
                                 self.tr_loads[tr_index, i] +
                                 self.tr_pv[tr_index, i] >=
                                 -self.tr_power_limit[tr_index, :].max()),
                                name=f'constr5_{tr_index}_t{i}')

        # -------- 目标与参数 --------
        model.setObjective(f @ u, GRB.MINIMIZE)
        model.setParam('OutputFlag', self.output_flag)
        model.params.NonConvex = 2   # 这里原代码设置了非凸开关（本模型线性可不必）
        if self.MIPGap is not None:
            model.params.MIPGap = self.MIPGap
        model.params.TimeLimit = self.time_limit

        # -------- 求解 --------
        model.optimize()
        self.total_exec_time += model.Runtime

        # 不可行时给默认动作（全 0）
        if model.status == GRB.Status.INF_OR_UNBD or model.status == GRB.Status.INFEASIBLE:
            print(f"INFEASIBLE (applying default actions) - step{t} !!!")
            actions = np.ones(self.n_ports) * 0
            return actions

        # -------- 读解并构造动作（只取第 1 步） --------
        a = np.zeros((nb*h, 1))
        cap = np.zeros((nb*h, 1))  # 占位

        for i in range(self.n_ports):
            a[i] = u[i].x  # 第 1 步各端口充电功率

        if self.verbose:
            print(f'Actions:\n {a.reshape(-1,self.n_ports)}')
            print(f'CapF1:\n {cap.reshape(-1,self.n_ports)}')

        # 归一化：动作 = P_ch / P_ch_max
        actions = np.zeros(self.n_ports)
        for i in range(self.n_ports):
            actions[i] = a[i]/self.max_ch_power[i//2]  # 注：此处索引按原实现保留

        if self.verbose:
            print(f'actions: {actions.shape} \n {actions}')

        return actions
