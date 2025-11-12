'''
该文件实现了基于 OCMF (Optimization–Control–Matching Framework)
的模型预测控制算法 (MPC)，包括 V2G 与 G2V 两个版本。
核心思想：在常规 eMPC 的基础上增加一个 CapF1 变量，用于预留功率裕度，
以提高算法在预测误差或电网波动下的可行性与稳健性。
'''

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from mpc import MPC

class OCMF_V2G(MPC):
    '''
    OCMF_V2G：用于 V2G (Vehicle-to-Grid) 场景的模型预测控制。
    允许充电与放电双向功率流，通过 CapF1 引入安全裕度。
    '''

    def __init__(self, env, control_horizon=10, verbose=False, **kwargs):
        '''
        初始化 V2G 模型预测控制器。
        env: EV2Gym 环境对象。
        control_horizon: 控制预测步长。
        verbose: 是否打印调试信息。
        '''
        super().__init__(env, control_horizon, verbose)

        self.na = self.n_ports                # 动作维度（端口数）
        self.nb = 2 * self.na                 # 变量维度（充/放两通道）

    def get_action(self, env):
        '''
        求解给定时刻的最优控制动作（充/放电功率）。
        '''
        t = env.current_step                  # 当前时刻步
        self.update_tr_power(t)               # 更新变压器预测功率限制
        self.reconstruct_state(t)             # 重建当前系统状态
        self.calculate_XF_V2G(t)              # 构建目标 SoC 状态向量
        self.v2g_station_models(t)            # 构建充/放电站动态模型
        self.calculate_InequalityConstraints(t)# 构建不等式约束矩阵
        self.set_power_limits_V2G(t)          # 设定功率上下界

        if self.verbose:
            self.print_info(t)

        # ====== 目标函数系数 ======
        f, f2 = [], []
        for i in range(self.control_horizon):
            for j in range(self.n_ports):
                # f: 常规能量成本
                f.append(self.T * self.ch_prices[t + i])
                f.append(-self.T * self.disch_prices[t + i])
                # f2: CapF1 对应惩罚/激励项
                f2.append(self.T * self.ch_prices[t + i])
                f2.append(self.T * self.disch_prices[t + i] * 2)

        f = np.array(f).reshape(-1)
        f2 = np.array(f2).reshape(-1)

        nb, n, h = self.nb, self.n_ports, self.control_horizon

        # ====== 构建优化模型 ======
        model = gp.Model("optimization_model")
        u = model.addMVar(nb * h, vtype=GRB.CONTINUOUS, name="u")      # 主决策变量：充/放电功率
        CapF1 = model.addMVar(nb * h, vtype=GRB.CONTINUOUS, name="CapF1") # 预留功率裕度变量
        Zbin = model.addMVar(n * h, vtype=GRB.BINARY, name="Zbin")     # 二元变量：充/放模式切换

        # ====== 约束条件 ======
        model.addConstr((self.AU @ u) <= self.bU, name="constr1")      # SoC 动态约束
        model.addConstr((0 <= CapF1), name="constr2a")                 # CapF1 ≥ 0
        model.addConstr((CapF1 <= self.UB), name="constr2b")           # CapF1 ≤ UB

        # 充电约束
        model.addConstrs((CapF1[j] <= u[j] for j in range(0, nb*h, 2)), name="constr3a")
        model.addConstrs((u[j] <= (self.UB[j] - CapF1[j]) * Zbin[j//2]
                          for j in range(0, nb*h, 2)), name="constr3b")

        # 放电约束
        model.addConstrs((CapF1[j] <= u[j] for j in range(1, nb*h, 2)), name="constr4a")
        model.addConstrs((u[j] <= (self.UB[j] - CapF1[j]) * (1 - Zbin[j//2])
                          for j in range(1, nb*h, 2)), name="constr4b")

        # 变压器功率约束：充/放电与负荷、PV 一并计入总功率
        for tr_index in range(self.number_of_transformers):
            for i in range(self.control_horizon):
                model.addConstr((gp.quicksum((u[j] - u[j+1])
                                             for index, j in enumerate(range(i*self.nb, (i+1)*self.nb, 2))
                                             if self.cs_transformers[index] == tr_index) +
                                 self.tr_loads[tr_index, i] + self.tr_pv[tr_index, i] <=
                                 self.tr_power_limit[tr_index, i]),
                                name=f'constr5_{tr_index}_t{i}')

        # 下界（负方向）约束
        for tr_index in range(self.number_of_transformers):
            for i in range(self.control_horizon):
                model.addConstr((gp.quicksum((u[j] - u[j+1])
                                             for index, j in enumerate(range(i*self.nb, (i+1)*self.nb, 2))
                                             if self.cs_transformers[index] == tr_index) +
                                 self.tr_loads[tr_index, i] + self.tr_pv[tr_index, i] >=
                                 -self.tr_power_limit[tr_index, :].max()),
                                name=f'constr5_{tr_index}_t{i}')

        # ====== 目标函数 ======
        model.setObjective(f @ u - f2 @ CapF1, GRB.MINIMIZE)            # 最小化总成本 - 预留激励
        model.setParam('OutputFlag', self.output_flag)
        model.params.NonConvex = 2

        if self.MIPGap is not None:
            model.params.MIPGap = self.MIPGap
        model.params.TimeLimit = self.time_limit

        model.optimize()
        self.total_exec_time += model.Runtime

        # ====== 不可行情况处理 ======
        if model.status in [GRB.Status.INF_OR_UNBD, GRB.Status.INFEASIBLE]:
            print(f"INFEASIBLE (applying default actions) - step {t} !!!")
            actions = np.zeros(self.n_ports)
            return actions

        # ====== 提取最优解并归一化 ======
        a = np.zeros((nb*h, 1))
        for i in range(nb*h):
            a[i] = u[i].x

        actions = np.zeros(self.n_ports)
        e = 1e-3
        for i in range(0, 2*self.n_ports, 2):
            if a[i] > e and a[i+1] > e:
                raise ValueError(f'充电与放电同时为正：端口{i} 值 {a[i]}, {a[i+1]}')
            elif a[i] > e:
                actions[i//2] = a[i] / self.max_ch_power[i//2]
            elif a[i+1] > e:
                actions[i//2] = -a[i+1] / abs(self.max_disch_power[i//2])

        if self.verbose:
            print(f'actions: {actions.shape}\n{actions}')

        return actions