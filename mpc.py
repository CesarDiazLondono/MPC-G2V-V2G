"""
该文件实现了用于“利润与灵活性最大化”的 MPC 基类。
核心职责：将 EV2Gym 环境数据（车辆到离站/SoC/价格/电网约束等）
构造成可滚动求解的预测控制模型（状态-控制矩阵、约束与边界等）。

Authors: Cesar Diaz-Londono, Stavros Orfanoudakis
"""

import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod


class MPC(ABC):

    def __init__(self,
                 env,
                 control_horizon=25,
                 verbose=False,
                 time_limit=0,
                 output_flag=0,
                 MIPGap=None,
                 **kwargs):
        """
        初始化 MPC 基类。

        Args:
            env: EV2Gym 环境实例
            control_horizon: 预测时域步数（滚动优化窗口）
            verbose: 调试打印
            time_limit/output_flag/MIPGap: 给求解器的参数
        """

        self.env = env
        self.n_ports = env.number_of_ports      # 控制的充电端口数（=EVSE 数）
        self.T = env.timescale/60               # 步长（小时）
        self.EV_number = len(env.EVs_profiles)  # 本日将出现的 EV 数量
        assert self.EV_number > 0, "仿真中没有 EV，请重置环境后再试。"

        self.simulation_length = env.simulation_length  # 仿真总步数
        self.t_min = env.timescale                      # 步长（分钟）
        self.control_horizon = control_horizon          # 预测时域步数
        self.total_exec_time = 0

        self.output_flag = output_flag
        self.time_limit = time_limit
        self.MIPGap = MIPGap
        self.verbose = verbose

        if self.verbose:
            np.set_printoptions(linewidth=np.inf)
            print(f'Number of EVs: {self.EV_number}')
            print(f'Number of ports: {self.n_ports}')
            print(f'Simulation length: {self.simulation_length}')
            print(f'Time scale: {self.T}')
            print(f'Prediction horizon: {self.control_horizon}')

        # 假设：每个充电桩只有 1 个端口（简化建模）
        assert env.charging_stations[0].n_ports == 1, "MPC 仅支持单端口桩。"
        # 取桩的充/放电功率上下界（全局假设同质桩）
        Pmax = env.charging_stations[0].get_max_power()
        Pmin = env.charging_stations[0].get_min_power()

        # 初始/目标 SoC（kWh）
        self.Cx0 = np.zeros(self.EV_number)
        self.Cxf = np.zeros(self.EV_number)
        # 到达/离开时间步
        self.arrival_times = np.zeros(self.EV_number, dtype=int)
        self.departure_times = np.zeros(self.EV_number, dtype=int)

        # u：调度矩阵（端口×时间），1 表示该时刻有车连在该端口
        horizon_plus = self.simulation_length + self.control_horizon + 1
        self.u = np.zeros((self.n_ports, horizon_plus))
        # x_init/x_final：初值/目标 SoC 的“定位矩阵”（端口×时间）
        self.x_init  = np.zeros((self.n_ports, horizon_plus))
        self.x_final = np.zeros((self.n_ports, horizon_plus))
        # x_max_batt：容量上限轨迹（端口×时间）
        self.x_max_batt = np.zeros((self.n_ports, horizon_plus))
        # p_max_MT/p_max_MT_dis：每端口每步的充/放功率上限
        self.p_max_MT     = np.zeros((self.n_ports, horizon_plus))
        self.p_max_MT_dis = np.zeros((self.n_ports, horizon_plus))
        # p_min_MT：最小功率（注意：后续 G2V/V2G 配置要谨慎）
        self.p_min_MT = np.zeros((self.n_ports, horizon_plus))
        
        # 记录每辆 EV 连接到哪个端口、最大容量
        self.ev_locations = np.zeros(self.EV_number, dtype=int)
        self.ev_max_batt  = np.zeros(self.EV_number, dtype=int)

        # 逐桩记录最大充/放功率（用于可视化/检查）
        self.max_ch_power   = np.zeros(self.n_ports)
        self.max_disch_power = np.zeros(self.n_ports)
        for i, cs in enumerate(env.charging_stations):
            self.max_ch_power[i]   = cs.get_max_power()
            self.max_disch_power[i] = cs.get_min_power()

        # 基于环境对象，逐 EV 写入调度/参数到上述矩阵
        for index, EV in enumerate(env.EVs_profiles):

            if index == 0:
                # 假设所有 EV 的充/放电效率一致：若为字典，取最大键对应值
                if isinstance(EV.charge_efficiency, dict):
                    key = max(EV.charge_efficiency, key=EV.charge_efficiency.get)                    
                    self.ch_eff   = EV.charge_efficiency[key]
                    self.disch_eff = EV.discharge_efficiency[key]                    
                else:
                    self.ch_eff   = EV.charge_efficiency
                    self.disch_eff = EV.discharge_efficiency                
                # 允许放电的最小 SoC（按容量占比）
                self.min_SoC = EV.min_battery_capacity/EV.battery_capacity

            # 初始/目标容量（kWh）
            self.Cx0[index] = EV.battery_capacity_at_arrival
            self.Cxf[index] = EV.desired_capacity
            # 到达时间
            self.arrival_times[index] = EV.time_of_arrival
            # 离开时间（防越界 +1 让“最后一个在站时刻”可被识别）
            if EV.time_of_departure > self.simulation_length:
                self.departure_times[index] = self.simulation_length
            else:
                self.departure_times[index] = EV.time_of_departure + 1

            # 该 EV 所在端口
            ev_location = EV.location
            self.ev_locations[index] = ev_location
            self.ev_max_batt[index]  = EV.battery_capacity
            
            a, d = self.arrival_times[index], self.departure_times[index]
            # 标记该 EV 在站区间
            self.u[ev_location, a:d]       = 1
            self.x_init[ev_location, a:d]  = self.Cx0[index]
            self.x_final[ev_location, a:d] = self.Cxf[index]
            self.x_max_batt[ev_location, a:d] = EV.battery_capacity

            # 该 EV 可用的最大充/放电功率（受桩与车辆共同限制）
            ev_pmax     = min(Pmax, EV.max_ac_charge_power)
            ev_dis_pmax = min(abs(Pmin), abs(EV.max_discharge_power))
            self.p_max_MT[ev_location, a:d]     = ev_pmax
            self.p_max_MT_dis[ev_location, a:d] = ev_dis_pmax

            # 最小功率（此处原实现存在来回覆盖，实际求解中一般设为 0 更稳）
            ev_pmin = max(abs(Pmin), EV.min_ac_charge_power)
            ev_pmin = 0  # 建模不支持非 0 最小充电功率 → 强制 0
            ev_pmin = Pmin  # 注意：这里又被覆盖为桩最小功率（可能为负），使用时需按模式谨慎处理
            self.p_min_MT[ev_location, a:d] = ev_pmin

        if self.verbose:
            print(f'Initial SoC: {self.Cx0}')
            print(f'Final SoC: {self.Cxf}')
            print(f'Arrival times: {self.arrival_times}')
            print(f'Departure times: {self.departure_times}')
            print(f'Initial conditions: {self.x_init}')
            print(f'Final conditions: {self.x_final}')
            print(f'Pmax: {self.p_max_MT}')

        # 变压器数量、桩-变压器映射，以及预测窗口内的负荷/PV/限值数组
        self.number_of_transformers = env.number_of_transformers
        self.cs_transformers = env.cs_transformers
        self.tr_loads        = np.zeros((self.number_of_transformers, self.control_horizon))
        self.tr_pv           = np.zeros((self.number_of_transformers, self.control_horizon))
        self.tr_power_limit  = np.zeros((self.number_of_transformers, self.control_horizon))
        # tr_cs：把“各步各桩功率”聚合到“各步各变压器”的稀疏选择张量
        self.tr_cs = np.zeros((self.number_of_transformers,
                               self.control_horizon,
                               self.control_horizon*self.n_ports))
        for tr_index in range(self.number_of_transformers):
            for i in range(self.control_horizon):
                self.tr_cs[tr_index, i,
                           i*self.n_ports:(i+1)*self.n_ports] = (np.array(env.cs_transformers) == tr_index)

        if self.verbose:
            print(f'Transformer loads: {self.tr_loads.shape}')
            print(f'{self.tr_loads}')
            print(f'Transformer Power Limit: {self.tr_power_limit.shape}')
            print(f'{self.tr_power_limit}')
            print(f'Transformer to CS: {self.tr_cs.shape}')
            print(f'{self.tr_cs}')

        # 价格（假设所有桩相同）：取整天价格，并在尾部拼接控制时域长度
        self.ch_prices   = abs(env.charge_prices[0, :])
        self.disch_prices = abs(env.discharge_prices[0, :])
        # 尾部扩展：充电价补大数（禁止超窗充电），放电价补 0（无收益）
        self.ch_prices   = np.concatenate((self.ch_prices,   np.ones(self.control_horizon)*100000))
        self.disch_prices = np.concatenate((self.disch_prices, np.zeros(self.control_horizon)))

        self.opti_info = []                  # 存放求解器返回信息
        self.x_next = self.x_init[:, 0].copy()  # 当前滚动步的“初始状态拼接”起点

        if self.verbose:
            print(f'Prices: {self.ch_prices}')
            print(f' Discharge Prices: {self.disch_prices}')

        # 下面是 v2 模型的一些占位/历史缓存（用于统计/绘图）
        self.varch2 = 0
        self.d_cycHist_e2 = []  # 循环退化历史
        self.d_calHist_e2 = []  # 日历退化历史
        self.Xhist_e2 = np.zeros((self.n_ports, self.simulation_length))  # SoC 轨迹
        self.Uhist_e2 = np.zeros((self.n_ports, self.simulation_length))  # 充电功率轨迹
        self.Uhist_e2V = np.zeros((self.n_ports, self.simulation_length)) # 放电功率轨迹（V2G）

    @abstractmethod
    def get_action(self, env):
        """子类需实现：给出当前步的动作（功率指令）。"""
        pass

    def update_tr_power(self, t):
        '''
        根据预测更新接下来控制时域内的“变压器限值/负荷/PV”。
        第一个点用下一时刻的实际观测，其余用 forecast。
        '''
        for i, tr in enumerate(self.env.transformers):
            self.tr_power_limit[i, :] = tr.get_power_limits(step=t, horizon=self.control_horizon)

            self.tr_pv[i, :] = np.zeros(self.control_horizon)
            self.tr_pv[i, 0] = tr.solar_power[tr.current_step+1]
            l = len(tr.pv_generation_forecast[tr.current_step + 2:
                                              tr.current_step+self.control_horizon+1])
            if l >= self.control_horizon - 1:
                l = self.control_horizon - 1
            else:
                l = l + 1
            self.tr_pv[i, 1:l] = tr.pv_generation_forecast[tr.current_step + 2:
                                                           tr.current_step+self.control_horizon]

            self.tr_loads[i, :] = np.zeros(self.control_horizon)
            self.tr_loads[i, 0] = tr.inflexible_load[tr.current_step+1]
            self.tr_loads[i, 1:l] = tr.inflexible_load_forecast[tr.current_step + 2:
                                                                tr.current_step+self.control_horizon]

    def update_tr_power_oracle(self, t):
        '''
        “真值”更新：不使用预测，直接使用全序列（用于对比/上界）。
        '''
        for i, tr in enumerate(self.env.transformers):
            self.tr_power_limit[i, :] = tr.max_power
            self.tr_pv[i, :] = tr.solar_power
            self.tr_loads[i, :] = tr.inflexible_load

    def reconstruct_state(self, t):
        '''
        用环境的实时信息重建当前滚动步的“状态拼接向量” Gxx0。
        若端口无车则该端口状态置 0。
        '''
        counter = 0
        for charger in self.env.charging_stations:
            for ev in charger.evs_connected:
                if ev is None:
                    self.x_next[counter] = 0
                else:
                    self.x_next[counter] = ev.current_capacity
                counter += 1

        self.Gxx0 = self.x_next.copy()

        # 将历史/初始化状态按时域展开，形成控制问题的“状态起点向量”
        if t == 0:
            for i in range(0, self.control_horizon-1):
                self.Gxx0 = np.concatenate((self.Gxx0, self.x_init[:, i].copy()))
        else:
            for i in range(t, t + self.control_horizon-1):
                Gx1 = self.x_init[:, i].copy()
                # 若该端口在 t 时刻有车且上一刻也非空，则用实时 x_next 更新
                for j in range(self.n_ports):
                    if self.x_init[j, t] > 0 and self.x_init[j, t - 1] != 0:
                        Gx1[j] = self.x_next[j].copy()
                self.Gxx0 = np.concatenate((self.Gxx0, Gx1))

        # 预测窗口内的容量上界向量（逐步拼接）
        self.XMAX = np.array([self.x_max_batt[:, t + i]
                              for i in range(self.control_horizon)]).flatten()

    def calculate_XF_G2V(self, t):
        # 生成 G2V 模式下的终端 SoC 目标向量 XF（在“离站边界”落到 x_final）
        self.XF = np.zeros(self.control_horizon * self.n_ports)
        m = self.n_ports
        for j in range(t + 1, t + self.control_horizon + 1):
            for i in range(self.n_ports):
                m += 1
                if self.u[i, j] == 0 and self.u[i, j - 1] == 1:
                    self.XF[m - self.n_ports-1] = self.x_final[i, j - 1]

    def calculate_XF_V2G(self, t):
        # 生成 V2G 模式下的终端 SoC 目标向量 XF：
        # 离站边界→落到 x_final；在站时还需满足 min_SoC 以避免深放
        self.XF = np.zeros(self.control_horizon * self.n_ports)
        m = self.n_ports
        for j in range(t + 1, t + self.control_horizon + 1):
            for i in range(self.n_ports):
                m += 1
                if self.u[i, j] == 0 and self.u[i, j - 1] == 1:
                    self.XF[m - self.n_ports-1] = self.x_final[i, j - 1]
                else:
                    self.XF[m - self.n_ports - 1] = self.x_max_batt[i, j - 1] * self.min_SoC

                # 到站边界：从无→有（新到车），则此处不强制终端目标
                if self.u[i, j] == 1 and self.u[i, j - 1] == 1 and self.u[i, j - 2] == 0:
                    self.XF[m - self.n_ports-1] = 0

    def v2g_station_models(self, t):
        '''
        构建 V2G 的单站状态转移/输入矩阵堆叠：
        Amono：按 u 的对角堆叠；Bmono：两列一组（充电 +ηT、放电 -ηT）
        '''

        self.Amono = np.dstack([np.diag(self.u[:, i])
                                for i in range(t, t + 1 + self.control_horizon)])

        self.Bmono = np.zeros((self.n_ports, self.nb, self.control_horizon+1))
        for j in range(t, t + self.control_horizon+1):
            Bmono2 = []
            bnew = self.T * np.diag(self.u[:, j]).T
            for i in range(self.n_ports):
                Bmono2.append(self.ch_eff * bnew[:, i])   # 充电通道
                Bmono2.append(-self.disch_eff * bnew[:, i])  # 放电通道
            Bmono2 = np.array(Bmono2).T
            self.Bmono[:, :, j - t] = Bmono2

    def g2v_station_models(self, t):
        # 构建 G2V 的单站模型：只有充电通道
        self.Amono = np.dstack([np.diag(self.u[:, i])
                                for i in range(t, t + 1 + self.control_horizon)])
        self.Bmono = self.ch_eff * self.T * np.dstack([np.diag(self.u[:, i])
                                                       for i in range(t, t + 1 + self.control_horizon)])

    def calculate_InequalityConstraints(self, t):
        '''
        计算不等式约束 AU·u <= bU：
        其中 Gu 是把控制序列卷积到状态增量的“块下三角”矩阵。
        '''

        self.Gu = np.zeros((self.control_horizon * self.na,
                            self.control_horizon * self.nb))

        for i in range(self.control_horizon):
            Bbar = self.Bmono[:, :, 0]
            for j in range(i + 1):
                Abar = np.eye(self.n_ports)
                if i == j:
                    self.Gu[i * self.na: (i+1) * self.na, j * self.nb: (j+1) * self.nb] = self.Bmono[:, :, j]
                else:
                    for m in range(j + 1, i + 1):
                        Abar = np.dot(Abar, self.Amono[:, :, m])
                    self.Gu[i * self.na: (i+1) * self.na, j * self.nb: (j+1) * self.nb] = np.dot(Abar, Bbar)
                Bbar = self.Bmono[:, :, j]

        # 由状态上界/终端目标构造 AU、bU（上下界对称拼接）
        self.AU = np.vstack((self.Gu, -self.Gu))
        self.bU = np.concatenate((np.abs(self.XMAX - self.Gxx0),  # 不超过容量上界
                                  -self.XF + self.Gxx0))          # 满足终端目标

    def set_power_limits_V2G(self, t):
        '''
        设置 V2G 模式下的控制量上下界：
        LB 默认 0；UB 拼 [Pch_max, Pdis_max] 对。
        '''
        # 注意：若需最小充/放电功率，可在此对 LB 进行相应设置
        self.LB = np.zeros((self.control_horizon * self.nb, 1), dtype=float)

        self.UB = np.array([[self.p_max_MT[j, i + t], self.p_max_MT_dis[j, i + t]]
                            for i in range(self.control_horizon)
                            for j in range(self.n_ports)], dtype=float)

        self.LB = self.LB.flatten().reshape(-1)
        self.UB = self.UB.flatten().reshape(-1)

    def set_power_limits_G2V(self, t):
        '''
        设置 G2V 模式下的控制量上下界：
        单通道，仅使用每步每端口的充电上限。
        '''
        self.LB = np.zeros((self.control_horizon * self.nb, 1), dtype=float)

        self.UB = np.array([self.p_max_MT[j, i + t]
                            for i in range(self.control_horizon)
                            for j in range(self.n_ports)], dtype=float)

        self.LB = self.LB.flatten().reshape(-1)
        self.UB = self.UB.flatten().reshape(-1)

    def print_info(self, t):
        '''
        调试输出：打印当前滚动步的关键矩阵与边界，便于排查。
        '''
        print(f'-------------------------------------------- \n t: {t}')
        for tr in range(self.number_of_transformers):
            print(f'Transformer {tr}:')
            print(f' - tr_pv: {self.tr_pv[tr, :]}')
            print(f' - tr_loads: {self.tr_loads[tr, :]}')
            print(f' - tr_power_limit: {self.tr_power_limit[tr, :]}')

        print(f'x_next: {self.x_next}')
        print(f'Amono: {self.Amono.shape}')
        print(f'Bmono: {self.Bmono.shape}')
        print(f'Gxx0: {self.Gxx0.shape}')
        print(f'Gxx0:{self.Gxx0}')
        print(f'Gu:{self.Gu.shape}')
        print(f'Gu:{self.Gu}')
        print(f'self.XF: {self.XF.shape}')
        print(f'XF: {self.XF}')
        print(f'self.XMAX: {self.XMAX.shape}')
        print(f'xmax: {self.XMAX}')
        print(f'AU: {self.AU.shape}, BU: {self.bU.shape}')
        print(f'self.LB: {self.LB.shape}')
        print(f'self.UB: {self.UB.shape} ')
        print(f'UB: {self.UB}')
        print(f'u: {self.u[:, t:t+self.control_horizon]}')
        print(f'Initial SoC: {self.Cx0}')
        print(f'Final SoC: {self.Cxf}')
        print(f'Arrival times: {self.arrival_times}')
        print(f'Departure times: {self.departure_times}')
        print(f'P_max_MT: {self.p_max_MT}')
        print(f'Desired Final: {self.x_final}')
