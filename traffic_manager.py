import math
import heapq
from typing import List, Dict, Tuple, Set, Optional, Any

class Conflict:
    """表示两个车辆之间的路径冲突"""
    
    def __init__(self, agent1, agent2, location=None, time_step=None, conflict_type="path"):
        """
        初始化冲突对象
        
        Args:
            agent1: 第一个车辆ID
            agent2: 第二个车辆ID
            location: 冲突位置坐标
            time_step: 冲突时间
            conflict_type: 冲突类型，可以是"path", "connection", "vertex", "edge"等
        """
        self.agent1 = agent1
        self.agent2 = agent2
        self.location = location
        self.time_step = time_step
        self.conflict_type = conflict_type
    
    def __str__(self):
        return f"Conflict({self.conflict_type}): {self.agent1} vs {self.agent2} at {self.location}, time={self.time_step}"


class Constraint:
    """表示ECBS中的约束条件，限制车辆在特定时间不能在特定位置"""
    
    def __init__(self, agent_id, location=None, time_step=None, constraint_type="vertex"):
        """
        初始化约束条件
        
        Args:
            agent_id: 适用的车辆ID
            location: 受约束的位置
            time_step: 约束时间
            constraint_type: 约束类型，可以是"vertex"(点约束)或"edge"(边约束)
        """
        self.agent_id = agent_id
        self.location = location
        self.time_step = time_step
        self.constraint_type = constraint_type
    
    def __str__(self):
        return f"Constraint({self.constraint_type}): Agent {self.agent_id} can't be at {self.location} at time {self.time_step}"


class ConstraintTreeNode:
    """ECBS约束树节点，表示一组约束下的解决方案"""
    
    def __init__(self, constraints=None, solution=None, cost=0, conflicts=None, parent=None):
        """
        初始化约束树节点
        
        Args:
            constraints: 约束列表
            solution: 解决方案（车辆路径字典）
            cost: 解决方案总成本
            conflicts: 解决方案中的冲突列表
            parent: 父节点
        """
        self.constraints = constraints or []
        self.solution = solution or {}
        self.cost = cost
        self.conflicts = conflicts or []
        self.parent = parent
        
        # 节点评估值 - 用于优先队列
        self.conflict_count = len(self.conflicts)
    
    def __lt__(self, other):
        """比较运算符，用于优先队列排序"""
        # 首先比较成本
        if self.cost != other.cost:
            return self.cost < other.cost
        # 如果成本相同，比较冲突数
        return self.conflict_count < other.conflict_count


class SafetyRectangle:
    """表示车辆安全区域的矩形"""
    
    def __init__(self, length=6.0, width=3.0, margin=0.5):
        """
        初始化安全矩形
        
        Args:
            length: 车辆长度
            width: 车辆宽度
            margin: 安全边距
        """
        self.length = length
        self.width = width
        self.margin = margin
        
        # 计算扩展尺寸（包含安全边距）
        self.extended_length = length + 2 * margin
        self.extended_width = width + 2 * margin
        
        # 对角线长度（用于快速冲突检测）
        self.diagonal = math.sqrt(self.extended_length**2 + self.extended_width**2)


class PathConflictDetector:
    """路径冲突检测器"""
    
    def __init__(self, safety_rectangle):
        """
        初始化路径冲突检测器
        
        Args:
            safety_rectangle: 安全矩形对象
        """
        self.safety_rectangle = safety_rectangle
    
    def check_path_conflict(self, agent1, path1, agent2, path2):
        """
        检查两条路径是否有冲突
        
        Args:
            agent1: 第一个车辆ID
            path1: 第一个车辆的路径
            agent2: 第二个车辆ID
            path2: 第二个车辆的路径
            
        Returns:
            Conflict or None: 如果有冲突返回冲突对象，否则返回None
        """
        # 如果路径为空，无冲突
        if not path1 or not path2:
            return None
        
        # 顶点冲突检测 - 两车同时在相同或接近的位置
        conflict = self._check_vertex_conflicts(agent1, path1, agent2, path2)
        if conflict:
            return conflict
        
        # 边冲突检测 - 两车在相同时间段内穿过同一区域
        conflict = self._check_edge_conflicts(agent1, path1, agent2, path2)
        if conflict:
            return conflict
        
        # 追逐冲突检测 - 两车沿着同一路径但可能发生追尾
        conflict = self._check_following_conflicts(agent1, path1, agent2, path2)
        if conflict:
            return conflict
        
        return None
    
    def _check_vertex_conflicts(self, agent1, path1, agent2, path2):
        """检查顶点冲突 - 两车同时在相同位置"""
        min_path_len = min(len(path1), len(path2))
        
        # 假设路径点索引对应时间步
        for t in range(min_path_len):
            p1 = path1[t]
            p2 = path2[t]
            
            # 计算两点之间的距离
            dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            
            # 如果距离小于安全距离，则有冲突
            if dist < self.safety_rectangle.diagonal / 2:
                return Conflict(
                    agent1=agent1,
                    agent2=agent2,
                    location=((p1[0] + p2[0])/2, (p1[1] + p2[1])/2),
                    time_step=t,
                    conflict_type="vertex"
                )
        
        return None
    
    def _check_edge_conflicts(self, agent1, path1, agent2, path2):
        """检查边冲突 - 两车在相邻时间步交叉"""
        min_path_len = min(len(path1), len(path2)) - 1
        
        for t in range(min_path_len):
            a1_from = path1[t]
            a1_to = path1[t+1]
            a2_from = path2[t]
            a2_to = path2[t+1]
            
            # 检查两线段是否相交
            if self._segments_intersect(a1_from, a1_to, a2_from, a2_to):
                # 计算交点
                intersection = self._compute_intersection(a1_from, a1_to, a2_from, a2_to)
                
                return Conflict(
                    agent1=agent1,
                    agent2=agent2,
                    location=intersection,
                    time_step=t + 0.5,  # 交叉发生在中间时间
                    conflict_type="edge"
                )
        
        return None
    
    def _check_following_conflicts(self, agent1, path1, agent2, path2):
        """检查追逐冲突 - 两车沿着相同路径但可能追尾"""
        # 检查是否在同一路径上
        path1_segments = self._extract_path_segments(path1)
        path2_segments = self._extract_path_segments(path2)
        
        # 查找共同的路径段
        common_segments = set(path1_segments) & set(path2_segments)
        
        # 如果没有共同路径段，则无追逐冲突
        if not common_segments:
            return None
        
        # 对于每个共同路径段，计算时间差异
        for segment in common_segments:
            # 找出各自进入该路径段的时间
            t1 = next((i for i, s in enumerate(path1_segments) if s == segment), -1)
            t2 = next((i for i, s in enumerate(path2_segments) if s == segment), -1)
            
            if t1 != -1 and t2 != -1:
                # 计算时间差
                time_diff = abs(t1 - t2)
                
                # 如果时间差太小，有追逐冲突风险
                if time_diff < 3:  # 时间差阈值
                    # 确定哪个车辆在前面
                    if t1 < t2:
                        front_agent, back_agent = agent1, agent2
                        front_time, back_time = t1, t2
                    else:
                        front_agent, back_agent = agent2, agent1
                        front_time, back_time = t2, t1
                    
                    # 创建追逐冲突
                    return Conflict(
                        agent1=front_agent,
                        agent2=back_agent,
                        location=segment,
                        time_step=(front_time + back_time)/2,
                        conflict_type="following"
                    )
        
        return None
    
    def _segments_intersect(self, p1, p2, p3, p4):
        """判断两条线段是否相交"""
        # 简化为2D问题
        p1_2d = (p1[0], p1[1])
        p2_2d = (p2[0], p2[1])
        p3_2d = (p3[0], p3[1])
        p4_2d = (p4[0], p4[1])
        
        # 计算方向
        d1 = self._direction(p3_2d, p4_2d, p1_2d)
        d2 = self._direction(p3_2d, p4_2d, p2_2d)
        d3 = self._direction(p1_2d, p2_2d, p3_2d)
        d4 = self._direction(p1_2d, p2_2d, p4_2d)
        
        # 如果两线段相交，必须方向交替
        return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
               ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))
    
    def _direction(self, p1, p2, p3):
        """计算向量叉积，判断p3相对于从p1到p2的方向"""
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
    
    def _compute_intersection(self, p1, p2, p3, p4):
        """计算两条线段的交点"""
        # 简化为2D问题
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        x3, y3 = p3[0], p3[1]
        x4, y4 = p4[0], p4[1]
        
        # 计算交点公式
        denominator = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
        
        # 如果分母为0，线段平行
        if abs(denominator) < 1e-10:
            # 返回中点作为近似
            return ((x1+x2+x3+x4)/4, (y1+y2+y3+y4)/4)
        
        # 计算交点坐标
        ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denominator
        
        x = x1 + ua * (x2-x1)
        y = y1 + ua * (y2-y1)
        
        return (x, y)
    
    def _extract_path_segments(self, path):
        """从路径提取线段，用于追逐冲突检测"""
        segments = []
        
        for i in range(len(path) - 1):
            # 创建线段表示（简化为起点和终点）
            segment = (
                (round(path[i][0], 1), round(path[i][1], 1)),
                (round(path[i+1][0], 1), round(path[i+1][1], 1))
            )
            segments.append(segment)
        
        return segments


class SpaceTimeReservationTable:
    """时空预留表，记录路径时空占用情况"""
    
    def __init__(self):
        """初始化时空预留表"""
        self.reservations = {}  # 格式: {(x, y, t): [agent_ids]}
        
        # 用于高效查询的空间哈希表
        self.location_reservation = {}  # 格式: {(x, y): {t: [agent_ids]}}
        self.agent_reservation = {}  # 格式: {agent_id: [(x, y, t)]}
    
    def add_reservation(self, agent_id, location, time_step):
        """
        添加预留
        
        Args:
            agent_id: 车辆ID
            location: 位置坐标 (x, y)
            time_step: 时间步
            
        Returns:
            bool: 添加是否成功
        """
        # 对位置和时间进行离散化
        x, y = round(location[0], 1), round(location[1], 1)
        t = round(time_step, 1)
        
        key = (x, y, t)
        
        # 初始化预留列表（如果不存在）
        if key not in self.reservations:
            self.reservations[key] = []
        
        # 添加预留
        self.reservations[key].append(agent_id)
        
        # 更新位置预留映射
        location_key = (x, y)
        if location_key not in self.location_reservation:
            self.location_reservation[location_key] = {}
        if t not in self.location_reservation[location_key]:
            self.location_reservation[location_key][t] = []
        self.location_reservation[location_key][t].append(agent_id)
        
        # 更新车辆预留映射
        if agent_id not in self.agent_reservation:
            self.agent_reservation[agent_id] = []
        self.agent_reservation[agent_id].append(key)
        
        return True
    
    def remove_reservation(self, agent_id, location=None, time_step=None):
        """
        移除预留
        
        Args:
            agent_id: 车辆ID
            location: 位置坐标 (可选)
            time_step: 时间步 (可选)
            
        Returns:
            int: 移除的预留数量
        """
        removed_count = 0
        
        # 如果提供了位置和时间，只移除特定的预留
        if location is not None and time_step is not None:
            x, y = round(location[0], 1), round(location[1], 1)
            t = round(time_step, 1)
            key = (x, y, t)
            
            if key in self.reservations and agent_id in self.reservations[key]:
                self.reservations[key].remove(agent_id)
                removed_count += 1
                
                # 更新位置预留映射
                location_key = (x, y)
                if location_key in self.location_reservation and t in self.location_reservation[location_key]:
                    if agent_id in self.location_reservation[location_key][t]:
                        self.location_reservation[location_key][t].remove(agent_id)
                
                # 更新车辆预留映射
                if agent_id in self.agent_reservation and key in self.agent_reservation[agent_id]:
                    self.agent_reservation[agent_id].remove(key)
        else:
            # 如果没有提供位置和时间，移除该车辆的所有预留
            if agent_id in self.agent_reservation:
                for key in self.agent_reservation[agent_id]:
                    x, y, t = key
                    if key in self.reservations and agent_id in self.reservations[key]:
                        self.reservations[key].remove(agent_id)
                        removed_count += 1
                    
                    # 更新位置预留映射
                    location_key = (x, y)
                    if location_key in self.location_reservation and t in self.location_reservation[location_key]:
                        if agent_id in self.location_reservation[location_key][t]:
                            self.location_reservation[location_key][t].remove(agent_id)
                
                # 清空车辆预留列表
                self.agent_reservation[agent_id] = []
        
        return removed_count
    
    def check_reservation(self, location, time_step):
        """
        检查位置和时间是否已被预留
        
        Args:
            location: 位置坐标 (x, y)
            time_step: 时间步
            
        Returns:
            list: 已预留的车辆ID列表
        """
        x, y = round(location[0], 1), round(location[1], 1)
        t = round(time_step, 1)
        
        key = (x, y, t)
        
        return self.reservations.get(key, []).copy()
    
    def check_path_reservation(self, path, start_time=0, speed=1.0):
        """
        检查路径是否与现有预留冲突
        
        Args:
            path: 路径点列表
            start_time: 开始时间
            speed: 速度
            
        Returns:
            list: 冲突信息列表，每项包含 (location, time_step, agent_ids)
        """
        conflicts = []
        
        current_time = start_time
        
        for i in range(len(path) - 1):
            # 计算段长度
            p1 = path[i]
            p2 = path[i + 1]
            segment_length = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            
            # 计算该段所需时间
            segment_time = segment_length / speed
            
            # 检查沿路径的多个点
            num_checks = max(1, int(segment_length / 0.5))  # 每0.5个单位检查一次
            
            for j in range(num_checks + 1):
                ratio = j / num_checks
                
                # 计算检查点
                x = p1[0] + ratio * (p2[0] - p1[0])
                y = p1[1] + ratio * (p2[1] - p1[1])
                t = current_time + ratio * segment_time
                
                # 检查该点是否有预留
                agents = self.check_reservation((x, y), t)
                
                if agents:
                    conflicts.append(((x, y), t, agents))
            
            # 更新时间
            current_time += segment_time
        
        return conflicts
    
    def reserve_path(self, agent_id, path, start_time=0, speed=1.0):
        """
        为路径添加预留
        
        Args:
            agent_id: 车辆ID
            path: 路径点列表
            start_time: 开始时间
            speed: 速度
            
        Returns:
            int: 添加的预留数量
        """
        reservation_count = 0
        current_time = start_time
        
        for i in range(len(path) - 1):
            # 计算段长度
            p1 = path[i]
            p2 = path[i + 1]
            segment_length = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            
            # 计算该段所需时间
            segment_time = segment_length / speed
            
            # 在路径沿途添加预留
            num_reservations = max(1, int(segment_length / 0.5))  # 每0.5个单位添加一个预留
            
            for j in range(num_reservations + 1):
                ratio = j / num_reservations
                
                # 计算预留点
                x = p1[0] + ratio * (p2[0] - p1[0])
                y = p1[1] + ratio * (p2[1] - p1[1])
                t = current_time + ratio * segment_time
                
                # 添加预留
                self.add_reservation(agent_id, (x, y), t)
                reservation_count += 1
            
            # 更新时间
            current_time += segment_time
        
        return reservation_count


class TrafficManager:
    """交通管理器，处理车辆流量和冲突"""
    
    def __init__(self, env, backbone_network=None):
        """
        初始化交通管理器
        
        Args:
            env: 环境对象
            backbone_network: 主干路径网络，可选
        """
        self.env = env
        self.backbone_network = backbone_network
        self.vehicle_reservations = {}  # 路径占用情况 {path_id: {time: [vehicle_ids]}}
        self.vehicle_paths = {}  # 车辆当前路径 {vehicle_id: path_info}
        
        # 交通规则
        self.rules = {
            'min_vehicle_distance': 10.0,  # 最小车辆间距
            'intersection_priority': 'first_come',  # 路口优先级规则
            'speed_limits': {},  # 路径速度限制 {path_id: speed_limit}
            'path_directions': {},  # 路径方向 {path_id: 'one_way' or 'two_way'}
            'path_lanes': {}  # 路径车道数 {path_id: lanes}
        }
        
        # ECBS参数
        self.suboptimality_bound = 1.5  # ECBS次优界限参数
        
        # 约束树管理
        self.constraint_tree = None
        self.open_list = []  # 按成本排序的节点优先队列
        self.focal_list = []  # 按冲突排序的低冲突解决方案
        
        # 冲突检测组件
        self.safety_rectangle = SafetyRectangle(
            length=6.0,   # 车辆长度
            width=3.0,    # 车辆宽度
            margin=0.5    # 安全边距
        )
        self.conflict_detector = PathConflictDetector(self.safety_rectangle)
        
        # 路径预留系统
        self.reservation_table = SpaceTimeReservationTable()
        
        # 冲突检测参数
        self.has_conflict = False
    
    def set_backbone_network(self, backbone_network):
        """设置主干路径网络"""
        self.backbone_network = backbone_network
        
        # 初始化路径规则
        if backbone_network:
            for path_id, path_data in backbone_network.paths.items():
                # 设置默认速度限制
                self.rules['speed_limits'][path_id] = path_data.get('speed_limit', 1.0)
                
                # 设置默认方向（双向）
                self.rules['path_directions'][path_id] = 'two_way'
                
                # 设置默认车道数（1车道）
                self.rules['path_lanes'][path_id] = 1
    
    def set_path_rule(self, path_id, rule_type, value):
        """设置路径规则"""
        if not path_id:
            return False
            
        if rule_type == 'speed_limit':
            self.rules['speed_limits'][path_id] = value
        elif rule_type == 'direction':
            self.rules['path_directions'][path_id] = value
        elif rule_type == 'lanes':
            self.rules['path_lanes'][path_id] = value
        else:
            return False
            
        return True
    
    def register_vehicle_path(self, vehicle_id, path, start_time=0, speed=1.0):
        """注册车辆路径，用于冲突检测"""
        path_info = {
            'path': path,
            'start_time': start_time,
            'speed': speed,
            'current_index': this_time,
            'progress': 0.0,
            'estimated_completion_time': start_time + self._estimate_path_time(path, speed),
            'reservations': []  # 时间窗口预留
        }
        
        self.vehicle_paths[vehicle_id] = path_info
        
        # 创建路径预留
        self._create_path_reservations(vehicle_id, path_info)
        
        # 更新冲突状态
        self._update_conflict_status()
        
        # 为路径添加空间时间预留
        self.reservation_table.reserve_path(vehicle_id, path, start_time, speed)
        
        return True
    
    def _create_path_reservations(self, vehicle_id, path_info):
        """为车辆路径创建时间窗口预留"""
        path = path_info['path']
        start_time = path_info['start_time']
        speed = path_info['speed']
        
        if not path or len(path) < 2:
            return
            
        # 清除先前的预留
        for reservation in path_info['reservations']:
            path_id = reservation['path_id']
            time_window = reservation['time_window']
            
            if path_id in self.vehicle_reservations:
                for t in range(time_window[0], time_window[1] + 1):
                    if t in self.vehicle_reservations[path_id] and vehicle_id in self.vehicle_reservations[path_id][t]:
                        self.vehicle_reservations[path_id][t].remove(vehicle_id)
        
        path_info['reservations'] = []
        
        # 为路径上每个段创建预留
        current_time = start_time
        
        for i in range(len(path) - 1):
            segment_length = self._calculate_distance(path[i], path[i + 1])
            segment_time = segment_length / speed
            
            # 使用路径ID或生成临时ID
            path_id = f"segment_{i}"
            
            # 查找此段是否属于主干网络的某个路径
            if self.backbone_network:
                for backbone_path_id, backbone_path in self.backbone_network.paths.items():
                    backbone_segments = backbone_path['path']
                    
                    for j in range(len(backbone_segments) - 1):
                        if (self._is_same_point(path[i], backbone_segments[j]) and 
                            self._is_same_point(path[i + 1], backbone_segments[j + 1])):
                            path_id = backbone_path_id
                            break
            
            # 创建时间窗口
            time_window_start = int(current_time)
            time_window_end = int(current_time + segment_time + self.rules['min_vehicle_distance'] / speed)
            
            # 存储预留信息
            path_info['reservations'].append({
                'path_id': path_id,
                'segment_index': i,
                'time_window': (time_window_start, time_window_end)
            })
            
            # 更新全局预留
            if path_id not in self.vehicle_reservations:
                self.vehicle_reservations[path_id] = {}
                
            for t in range(time_window_start, time_window_end + 1):
                if t not in self.vehicle_reservations[path_id]:
                    self.vehicle_reservations[path_id][t] = []
                    
                self.vehicle_reservations[path_id][t].append(vehicle_id)
            
            # 更新时间
            current_time += segment_time
    
    def update_vehicle_position(self, vehicle_id, index, progress):
        """更新车辆在路径上的位置"""
        if vehicle_id not in self.vehicle_paths:
            return False
            
        path_info = self.vehicle_paths[vehicle_id]
        
        # 如果索引或进度发生变化，需要更新预留
        if path_info['current_index'] != index or abs(path_info['progress'] - progress) > 0.1:
            path_info['current_index'] = index
            path_info['progress'] = progress
            
            # 更新预留
            self._create_path_reservations(vehicle_id, path_info)
            
            # 更新冲突状态
            self._update_conflict_status()
        
        return True
    
    def release_vehicle_path(self, vehicle_id):
        """释放车辆路径"""
        if vehicle_id not in self.vehicle_paths:
            return False
            
        path_info = self.vehicle_paths[vehicle_id]
        
        # 清除所有预留
        for reservation in path_info['reservations']:
            path_id = reservation['path_id']
            time_window = reservation['time_window']
            
            if path_id in self.vehicle_reservations:
                for t in range(time_window[0], time_window[1] + 1):
                    if t in self.vehicle_reservations[path_id] and vehicle_id in self.vehicle_reservations[path_id][t]:
                        self.vehicle_reservations[path_id][t].remove(vehicle_id)
        
        # 清除空间时间预留
        self.reservation_table.remove_reservation(vehicle_id)
        
        # 删除路径信息
        del self.vehicle_paths[vehicle_id]
        
        # 更新冲突状态
        self._update_conflict_status()
        
        return True
    
    def _update_conflict_status(self):
        """更新冲突状态"""
        self.has_conflict = False
        
        # 检查每个路径的每个时间窗口是否有冲突
        for path_id, time_windows in self.vehicle_reservations.items():
            for t, vehicles in time_windows.items():
                if len(vehicles) > self.rules.get('path_lanes', {}).get(path_id, 1):
                    self.has_conflict = True
                    break
            
            if self.has_conflict:
                break
    
    def check_path_conflicts(self, vehicle_id, path, start_time=0, speed=1.0):
        """检查路径是否有冲突"""
        # 创建临时路径信息
        temp_path_info = {
            'path': path,
            'start_time': start_time,
            'speed': speed,
            'current_index': 0,
            'progress': 0.0,
            'estimated_completion_time': start_time + self._estimate_path_time(path, speed),
            'reservations': []
        }
        
        # 生成临时预留
        temp_reservations = []
        current_time = start_time
        
        for i in range(len(path) - 1):
            segment_length = self._calculate_distance(path[i], path[i + 1])
            segment_time = segment_length / speed
            
            # 使用路径ID或生成临时ID
            path_id = f"segment_{i}"
            
            # 查找此段是否属于主干网络的某个路径
            if self.backbone_network:
                for backbone_path_id, backbone_path in self.backbone_network.paths.items():
                    backbone_segments = backbone_path['path']
                    
                    for j in range(len(backbone_segments) - 1):
                        if (self._is_same_point(path[i], backbone_segments[j]) and 
                            self._is_same_point(path[i + 1], backbone_segments[j + 1])):
                            path_id = backbone_path_id
                            break
            
            # 创建时间窗口
            time_window_start = int(current_time)
            time_window_end = int(current_time + segment_time + self.rules['min_vehicle_distance'] / speed)
            
            # 存储预留信息
            temp_reservations.append({
                'path_id': path_id,
                'segment_index': i,
                'time_window': (time_window_start, time_window_end)
            })
            
            # 检查是否与现有预留冲突
            has_conflict = False
            max_lanes = self.rules.get('path_lanes', {}).get(path_id, 1)
            
            if path_id in self.vehicle_reservations:
                for t in range(time_window_start, time_window_end + 1):
                    if t in self.vehicle_reservations[path_id]:
                        # 如果是单向路径，检查方向是否冲突
                        if (self.rules.get('path_directions', {}).get(path_id) == 'one_way' and 
                            self._is_direction_conflict(path_id, path[i], path[i + 1])):
                            has_conflict = True
                            break
                        
                        # 检查车道数是否足够
                        if len(self.vehicle_reservations[path_id][t]) >= max_lanes:
                            has_conflict = True
                            break
            
            if has_conflict:
                return True  # 有冲突
            
            # 更新时间
            current_time += segment_time
        
        # 使用ECBS检查器检查更精细的冲突
        for other_vehicle_id, other_path_info in self.vehicle_paths.items():
            if other_vehicle_id != vehicle_id:
                conflict = self.conflict_detector.check_path_conflict(
                    vehicle_id, path,
                    other_vehicle_id, other_path_info['path']
                )
                
                if conflict:
                    return True  # 有冲突
        
        return False  # 无冲突
    
    def suggest_speed_adjustment(self, vehicle_id):
        """为避免冲突建议速度调整"""
        if vehicle_id not in self.vehicle_paths:
            return None
            
        path_info = self.vehicle_paths[vehicle_id]
        
        # 检测即将到来的冲突
        upcoming_conflicts = self._detect_upcoming_conflicts(vehicle_id, path_info)
        
        if not upcoming_conflicts:
            return None  # 无冲突，无需调整
        
        # 根据冲突计算理想的速度调整因子
        speed_factors = []
        
        for conflict in upcoming_conflicts:
            # 计算为避免冲突需要的速度因子
            conflict_path_id = conflict['path_id']
            conflict_time = conflict['time']
            conflict_vehicles = conflict['vehicles']
            
            # 找出当前车辆的预留
            vehicle_reservation = None
            for res in path_info['reservations']:
                if res['path_id'] == conflict_path_id:
                    time_window = res['time_window']
                    if time_window[0] <= conflict_time <= time_window[1]:
                        vehicle_reservation = res
                        break
            
            if not vehicle_reservation:
                continue
            
            # 计算延迟因子（减速）
            delay_factor = 0.8  # 减速20%
            
            # 或者计算加速因子（加速）
            speedup_factor = 1.2  # 加速20%
            
            # 选择更合适的因子
            # 简单策略：如果车辆ID较小（优先级高），则加速；否则减速
            other_vehicle_id = min([v for v in conflict_vehicles if v != vehicle_id], default=None)
            
            if other_vehicle_id and vehicle_id < other_vehicle_id:
                speed_factors.append(speedup_factor)
            else:
                speed_factors.append(delay_factor)
        
        # 选择最保守的速度调整
        if not speed_factors:
            return None
            
        # 如果有减速因子，选择最小的；否则选择最大的加速因子
        delay_factors = [f for f in speed_factors if f < 1.0]
        speedup_factors = [f for f in speed_factors if f > 1.0]
        
        if delay_factors:
            return min(delay_factors)
        elif speedup_factors:
            return max(speedup_factors)
        else:
            return 1.0  # 保持当前速度
    
    def detect_conflicts(self, paths):
        """
        检测路径集合中的冲突
        
        Args:
            paths: 路径字典 {vehicle_id: path}
            
        Returns:
            list: 冲突列表
        """
        conflicts = []
        vehicles = list(paths.keys())
        
        # 检查成对车辆之间的冲突
        for i in range(len(vehicles)):
            for j in range(i+1, len(vehicles)):
                v1 = vehicles[i]
                v2 = vehicles[j]
                path1 = paths[v1]
                path2 = paths[v2]
                
                # 检查不同类型的冲突
                
                # 1. 主干连接点冲突 (入口点)
                conn_conflict = self._check_connection_conflict(v1, path1, v2, path2)
                if conn_conflict:
                    conflicts.append(conn_conflict)
                    continue
                    
                # 2. 主干穿越冲突 (同一路径段)
                trav_conflict = self._check_traversal_conflict(v1, path1, v2, path2)
                if trav_conflict:
                    conflicts.append(trav_conflict)
                    continue
                    
                # 3. 一般路径冲突
                gen_conflict = self.conflict_detector.check_path_conflict(v1, path1, v2, path2)
                if gen_conflict:
                    conflicts.append(gen_conflict)
        
        return conflicts
    
    def _check_connection_conflict(self, v1, path1, v2, path2):
        """检查两个车辆是否在同一主干连接点上产生冲突"""
        # 找出两个路径中的主干连接点
        conn_points1 = self._find_backbone_connections(path1)
        conn_points2 = self._find_backbone_connections(path2)
        
        # 如果其中一个路径没有连接点，则无连接点冲突
        if not conn_points1 or not conn_points2:
            return None
        
        # 检查每对连接点
        for cp1 in conn_points1:
            for cp2 in conn_points2:
                # 如果使用同一个连接点且时间接近
                if self._is_same_connection(cp1['conn'], cp2['conn']):
                    if abs(cp1['time'] - cp2['time']) < 5.0:  # 时间阈值
                        return Conflict(
                            agent1=v1, 
                            agent2=v2,
                            location=cp1['conn']['position'],
                            time_step=min(cp1['time'], cp2['time']),
                            conflict_type="connection"
                        )
        
        return None
    
    def _check_traversal_conflict(self, v1, path1, v2, path2):
        """检查两个车辆是否在主干路径上产生穿越冲突"""
        # 查找每个路径使用的主干路径段
        segments1 = self._identify_backbone_segments(path1)
        segments2 = self._identify_backbone_segments(path2)
        
        # 如果其中一个路径没有主干段，则无穿越冲突
        if not segments1 or not segments2:
            return None
        
        # 检查是否有共同的主干路径段
        for seg1 in segments1:
            for seg2 in segments2:
                # 如果在同一主干路径上
                if seg1['path_id'] == seg2['path_id']:
                    # 检查时间重叠
                    if (seg1['time_start'] <= seg2['time_end'] and 
                        seg2['time_start'] <= seg1['time_end']):
                        
                        # 计算重叠区域
                        overlap_start = max(seg1['time_start'], seg2['time_start'])
                        overlap_end = min(seg1['time_end'], seg2['time_end'])
                        
                        # 创建冲突对象
                        return Conflict(
                            agent1=v1,
                            agent2=v2,
                            location=f"backbone_{seg1['path_id']}",
                            time_step=(overlap_start + overlap_end) / 2,
                            conflict_type="traversal"
                        )
        
        return None
    
    def _find_backbone_connections(self, path):
        """在路径中找出主干网络连接点"""
        if not self.backbone_network or not path:
            return []
        
        connections = []
        current_time = 0
        speed = 1.0  # 假设默认速度
        
        # 检查路径中的每个点
        for i in range(len(path)):
            point = path[i]
            
            # 查找最近的连接点
            conn = self.backbone_network.find_nearest_connection(
                point, 
                max_distance=2.0  # 小范围内查找
            )
            
            if conn:
                connections.append({
                    'conn': conn,
                    'path_index': i,
                    'time': current_time
                })
            
            # 更新时间
            if i < len(path) - 1:
                segment_length = self._calculate_distance(path[i], path[i+1])
                current_time += segment_length / speed
        
        return connections
    
    def _is_same_connection(self, conn1, conn2):
        """判断两个连接点是否为同一个"""
        if not conn1 or not conn2:
            return False
        
        # 比较路径ID和索引
        if conn1.get('path_id') == conn2.get('path_id'):
            # 索引接近即视为同一连接点
            idx1 = conn1.get('path_index', 0)
            idx2 = conn2.get('path_index', 0)
            return abs(idx1 - idx2) <= 2
        
        # 比较位置
        pos1 = conn1.get('position')
        pos2 = conn2.get('position')
        if pos1 and pos2:
            return self._calculate_distance(pos1, pos2) < 3.0
        
        return False
    
    def _identify_backbone_segments(self, path):
        """识别路径中的主干网络段"""
        if not self.backbone_network or not path:
            return []
        
        segments = []
        current_segment = None
        current_time = 0
        speed = 1.0  # 假设默认速度
        
        # 检查路径中的每个点
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i+1]
            
            # 计算此段时间
            segment_length = self._calculate_distance(p1, p2)
            segment_time = segment_length / speed
            
            # 检查此段是否属于主干网络
            backbone_path_id = None
            for path_id, path_data in self.backbone_network.paths.items():
                backbone_path = path_data['path']
                
                for j in range(len(backbone_path) - 1):
                    if (self._is_same_point(p1, backbone_path[j]) and 
                        self._is_same_point(p2, backbone_path[j+1])):
                        backbone_path_id = path_id
                        break
                
                if backbone_path_id:
                    break
            
            # 更新当前段信息
            if backbone_path_id:
                if current_segment and current_segment['path_id'] == backbone_path_id:
                    # 扩展当前段
                    current_segment['end_index'] = i + 1
                    current_segment['time_end'] = current_time + segment_time
                else:
                    # 如果有之前的段，保存它
                    if current_segment:
                        segments.append(current_segment)
                    
                    # 创建新段
                    current_segment = {
                        'path_id': backbone_path_id,
                        'start_index': i,
                        'end_index': i + 1,
                        'time_start': current_time,
                        'time_end': current_time + segment_time
                    }
            elif current_segment:
                # 离开主干网络，保存当前段
                segments.append(current_segment)
                current_segment = None
            
            # 更新时间
            current_time += segment_time
        
        # 保存最后一个段
        if current_segment:
            segments.append(current_segment)
        
        return segments
    
    def suggest_path_adjustment(self, vehicle_id, start, goal):
        """
        为避免冲突建议路径调整 - 使用ECBS寻找最优路径
        
        Args:
            vehicle_id: 车辆ID
            start: 起点
            goal: 终点
            
        Returns:
            list or None: 调整后的路径，如果无法调整则返回None
        """
        # 如果车辆已有路径，将其暂时释放
        had_path = vehicle_id in self.vehicle_paths
        if had_path:
            original_path_info = self.vehicle_paths[vehicle_id]
            self.release_vehicle_path(vehicle_id)
        
        # 获取路径规划器
        planner = None
        if hasattr(self.env, 'path_planner') and self.env.path_planner:
            planner = self.env.path_planner
        
        if not planner:
            if had_path:
                # 恢复原路径
                self.register_vehicle_path(
                    vehicle_id,
                    original_path_info['path'],
                    original_path_info['start_time'],
                    original_path_info['speed']
                )
            return None
        
        # 收集当前所有车辆的路径
        paths = {}
        for v_id, v_path_info in self.vehicle_paths.items():
            paths[v_id] = v_path_info['path']
        
        # 为当前车辆规划基本路径
        initial_path = planner.plan_path(vehicle_id, start, goal, check_conflicts=False)
        
        if not initial_path:
            if had_path:
                # 恢复原路径
                self.register_vehicle_path(
                    vehicle_id,
                    original_path_info['path'],
                    original_path_info['start_time'],
                    original_path_info['speed']
                )
            return None
        
        # 将当前车辆的初始路径添加到路径字典
        paths[vehicle_id] = initial_path
        
        # 使用ECBS解决冲突
        conflict_free_paths = self.resolve_conflicts(paths)
        
        if not conflict_free_paths or vehicle_id not in conflict_free_paths:
            if had_path:
                # 恢复原路径
                self.register_vehicle_path(
                    vehicle_id,
                    original_path_info['path'],
                    original_path_info['start_time'],
                    original_path_info['speed']
                )
            return None
        
        # 返回无冲突的路径
        return conflict_free_paths[vehicle_id]
    

    
    def resolve_conflicts(self, paths, backbone_network=None, path_structures=None):
        """
        使用ECBS风格的约束树搜索解决冲突，保留骨干网络结构
        
        Args:
            paths: 初始路径字典 {vehicle_id: path}
            backbone_network: 骨干网络对象(可选)
            path_structures: 路径结构信息 {vehicle_id: structure_dict}
            
        Returns:
            dict: 无冲突的路径字典，如果无法解决则返回None
        """
        # 初始化约束树根节点
        root = ConstraintTreeNode(
            constraints=[],
            solution=paths.copy(),
            cost=self._calculate_solution_cost(paths)
        )
        
        # 保存路径结构信息 - 新增
        self.path_structures = path_structures or {}
        self.backbone_network = backbone_network
        
        # 检测初始冲突
        root.conflicts = self.detect_conflicts(root.solution)
        root.conflict_count = len(root.conflicts)
        
        # 初始化搜索
        self.open_list = [root]
        self.focal_list = [root]
        
        # 最大迭代次数
        max_iterations = 1000
        iterations = 0
        
        # ECBS搜索
        while self.open_list and iterations < max_iterations:
            iterations += 1
            
            # 从focal list中获取冲突最少的节点
            current = self._get_best_node()
            
            # 检查冲突
            if not current.conflicts:
                # 如果无冲突，找到解决方案
                return current.solution
            
            # 选择第一个冲突
            conflict = current.conflicts[0]
            
            # 为两个车辆生成约束
            constraint1 = self._generate_constraint(conflict.agent1, conflict)
            constraint2 = self._generate_constraint(conflict.agent2, conflict)
            
            # 创建子节点并添加到搜索树
            for constraint in [constraint1, constraint2]:
                # 创建子节点
                child = self._create_child_node(current, constraint)
                
                if child:
                    # 添加到open list
                    heapq.heappush(self.open_list, child)
                    
                    # 更新focal list
                    self._update_focal_list()
        
        # 如果达到最大迭代次数，返回找到的最佳解决方案
        return self._get_best_solution()

    def _apply_constraint(self, agent_id, constraint, solution):
        """
        应用约束并重新规划路径，保留骨干网络结构
        
        Args:
            agent_id: 车辆ID
            constraint: 约束对象
            solution: 当前解决方案
            
        Returns:
            list or None: 满足约束的新路径，如果无法满足则返回None
        """
        # 获取当前路径
        current_path = solution[agent_id]
        
        if not current_path or len(current_path) < 2:
            return None
            
        # 获取起点和终点
        start = current_path[0]
        goal = current_path[-1]
        
        # 获取路径结构信息 - 新增
        path_structure = self.path_structures.get(agent_id, {})
        
        # 根据约束类型处理
        if constraint.constraint_type in ["vertex", "edge", "connection", "traversal"]:
            # 获取冲突位置和时间
            conflict_location = constraint.location
            conflict_time = constraint.time_step
            
            # 如果有骨干网络和路径结构，尝试保留骨干部分
            if self.backbone_network and path_structure:
                # 提取路径各部分
                to_backbone = path_structure.get('to_backbone_path')
                backbone_path = path_structure.get('backbone_path')
                from_backbone = path_structure.get('from_backbone_path')
                
                # 判断冲突在哪个部分
                conflict_part = None
                if to_backbone and self._is_conflict_in_path_segment(constraint, to_backbone):
                    conflict_part = "to_backbone"
                elif backbone_path and self._is_conflict_in_path_segment(constraint, backbone_path):
                    conflict_part = "backbone"
                elif from_backbone and self._is_conflict_in_path_segment(constraint, from_backbone):
                    conflict_part = "from_backbone"
                
                # 只重新规划冲突部分
                if conflict_part == "to_backbone" and backbone_path:
                    # 重新规划到骨干网络的路径
                    entry_point = path_structure.get('entry_point')
                    if entry_point and entry_point.get('position'):
                        # 创建临时检查函数，在规划时避开约束位置
                        def custom_collision_checker(x, y, theta):
                            # 检查是否接近约束位置
                            if constraint.location:
                                dist = self._calculate_distance((x, y), constraint.location)
                                if dist < 2.0:  # 安全距离
                                    return False
                            return True
                        
                        # 获取路径规划器
                        planner = None
                        if hasattr(self.env, 'path_planner') and self.env.path_planner:
                            planner = self.env.path_planner
                        
                        if planner:
                            # 保存原有的碰撞检测器
                            original_checker = None
                            if hasattr(planner, 'collision_checker'):
                                original_checker = planner.collision_checker
                            
                            # 设置临时检测器
                            planner.collision_checker = custom_collision_checker
                            
                            # 重新规划到骨干网络的路径
                            new_to_backbone = planner.plan_path(
                                agent_id, 
                                start, 
                                entry_point['position'],
                                use_backbone=False
                            )
                            
                            # 恢复原有检测器
                            if original_checker:
                                planner.collision_checker = original_checker
                            
                            if new_to_backbone:
                                # 合并新路径
                                complete_path = self._merge_paths(
                                    new_to_backbone,
                                    backbone_path,
                                    from_backbone or []
                                )
                                return complete_path
                
                elif conflict_part == "from_backbone" and backbone_path:
                    # 重新规划从骨干网络到终点的路径
                    exit_point = path_structure.get('exit_point')
                    if exit_point and exit_point.get('position'):
                        # 创建临时检查函数，同上
                        def custom_collision_checker(x, y, theta):
                            if constraint.location:
                                dist = self._calculate_distance((x, y), constraint.location)
                                if dist < 2.0:
                                    return False
                            return True
                        
                        # 获取路径规划器
                        planner = None
                        if hasattr(self.env, 'path_planner') and self.env.path_planner:
                            planner = self.env.path_planner
                        
                        if planner:
                            # 保存原有的碰撞检测器
                            original_checker = None
                            if hasattr(planner, 'collision_checker'):
                                original_checker = planner.collision_checker
                            
                            # 设置临时检测器
                            planner.collision_checker = custom_collision_checker
                            
                            # 重新规划从骨干到终点的路径
                            new_from_backbone = planner.plan_path(
                                agent_id, 
                                exit_point['position'],
                                goal,
                                use_backbone=False
                            )
                            
                            # 恢复原有检测器
                            if original_checker:
                                planner.collision_checker = original_checker
                            
                            if new_from_backbone:
                                # 合并新路径
                                complete_path = self._merge_paths(
                                    to_backbone or [],
                                    backbone_path,
                                    new_from_backbone
                                )
                                return complete_path
                
                elif conflict_part == "backbone":
                    # 尝试找到骨干网络中的替代路径
                    entry_point = path_structure.get('entry_point')
                    exit_point = path_structure.get('exit_point')
                    
                    if entry_point and exit_point and self.backbone_network:
                        # 获取当前使用的骨干路径段ID
                        current_segment = path_structure.get('backbone_segment')
                        
                        # 从骨干网络中查找替代路径
                        if current_segment and ':' in current_segment:
                            start_path_id, end_path_id = current_segment.split(':')
                            
                            # 查找从起点到终点的所有可能路径
                            all_paths = []
                            if hasattr(self.backbone_network, 'find_all_paths'):
                                all_paths = self.backbone_network.find_all_paths(
                                    start_path_id, end_path_id, max_paths=3
                                )
                            
                            # 尝试每条替代路径
                            for path_ids in all_paths:
                                # 跳过当前路径
                                if ':'.join(path_ids) == current_segment:
                                    continue
                                
                                # 构建替代路径
                                alt_backbone_path = []
                                for pid in path_ids:
                                    if pid in self.backbone_network.paths:
                                        path = self.backbone_network.paths[pid]['path']
                                        # 跳过第一个点以避免重复
                                        if alt_backbone_path:
                                            path = path[1:]
                                        alt_backbone_path.extend(path)
                                
                                if alt_backbone_path:
                                    # 验证替代路径是否避开了约束
                                    if not self._is_conflict_in_path_segment(constraint, alt_backbone_path):
                                        # 合并替代路径
                                        complete_path = self._merge_paths(
                                            to_backbone or [],
                                            alt_backbone_path,
                                            from_backbone or []
                                        )
                                        return complete_path
        
        # 如果上面的保留骨干部分的规划失败，回退到完全重新规划
        if hasattr(self.env, 'path_planner') and self.env.path_planner:
            path = self.env.path_planner.plan_path(
                agent_id, 
                start, 
                goal,
                use_backbone=True,  # 仍然使用骨干网络
                check_conflicts=False
            )
            return path
    
        return None

    def _is_conflict_in_path_segment(self, constraint, path_segment):
        """
        判断冲突是否在给定的路径段内
        
        Args:
            constraint: 约束对象
            path_segment: 路径段
            
        Returns:
            bool: 冲突是否在路径段内
        """
        if not path_segment or not constraint.location:
            return False
        
        # 对于点冲突，检查是否有点接近约束位置
        min_dist = float('inf')
        for point in path_segment:
            dist = self._calculate_distance(point, constraint.location)
            min_dist = min(min_dist, dist)
        
        # 如果最小距离小于阈值，认为冲突在此路径段
        return min_dist < 5.0  # 5.0是阈值，可以调整

    def _merge_paths(self, path1, path2, path3):
        """
        合并多个路径段，避免重复点
        
        Args:
            path1, path2, path3: 要合并的路径段
            
        Returns:
            list: 合并后的路径
        """
        if not path1:
            if not path2:
                return path3 or []
            if not path3:
                return path2 or []
            # 合并path2和path3
            if self._is_same_point(path2[-1], path3[0]):
                return path2 + path3[1:]
            return path2 + path3
        
        if not path2:
            if not path3:
                return path1 or []
            # 合并path1和path3
            if self._is_same_point(path1[-1], path3[0]):
                return path1 + path3[1:]
            return path1 + path3
        
        # 合并所有三段
        merged = path1[:]
        if self._is_same_point(path1[-1], path2[0]):
            merged = merged[:-1] + path2
        else:
            merged = merged + path2
        
        if path3:
            if self._is_same_point(merged[-1], path3[0]):
                merged = merged[:-1] + path3
            else:
                merged = merged + path3
        
        return merged
        
    def _is_same_point(self, p1, p2, tolerance=0.5):
        """判断两点是否相同（考虑误差）"""
        if not p1 or not p2:
            return False
            
        # 提取坐标
        x1 = p1[0] if len(p1) > 0 else 0
        y1 = p1[1] if len(p1) > 1 else 0
        x2 = p2[0] if len(p2) > 0 else 0
        y2 = p2[1] if len(p2) > 1 else 0
        
        # 计算距离并判断是否小于容差
        dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        return dist < tolerance
    
    def _is_direction_conflict(self, path_id, start, end):
        """检查方向是否与路径定义的方向冲突"""
        if path_id not in self.rules.get('path_directions', {}):
            return False
            
        direction = self.rules['path_directions'][path_id]
        
        if direction == 'two_way':
            return False  # 双向路径无方向冲突
            
        # 对于单向路径，需要检查车辆行驶方向
        if self.backbone_network and path_id in self.backbone_network.paths:
            backbone_path = self.backbone_network.paths[path_id]['path']
            
            if len(backbone_path) < 2:
                return False
                
            # 检查车辆行驶方向是否与路径定义的方向一致
            backbone_start = backbone_path[0]
            backbone_end = backbone_path[-1]
            
            # 计算方向向量
            veh_dir_x = end[0] - start[0]
            veh_dir_y = end[1] - start[1]
            
            path_dir_x = backbone_end[0] - backbone_start[0]
            path_dir_y = backbone_end[1] - backbone_start[1]
            
            # 计算向量点积，判断是否同向
            dot_product = veh_dir_x * path_dir_x + veh_dir_y * path_dir_y
            
            # 点积为负表示方向相反
            return dot_product < 0
        
        return False
    
    def _detect_upcoming_conflicts(self, vehicle_id, path_info):
        """检测即将到来的冲突"""
        conflicts = []
        
        # 检查每个预留是否有冲突
        for reservation in path_info['reservations']:
            path_id = reservation['path_id']
            time_window = reservation['time_window']
            
            # 检查本车辆已经过了的时间窗口部分
            current_time = self.env.current_time if hasattr(self.env, 'current_time') else 0
            
            # 只检查未来的时间窗口
            for t in range(max(current_time, time_window[0]), time_window[1] + 1):
                if path_id in self.vehicle_reservations and t in self.vehicle_reservations[path_id]:
                    vehicles = self.vehicle_reservations[path_id][t]
                    
                    # 如果预留车辆数超过车道数，则有冲突
                    max_lanes = self.rules.get('path_lanes', {}).get(path_id, 1)
                    
                    if len(vehicles) > max_lanes:
                        conflicts.append({
                            'path_id': path_id,
                            'time': t,
                            'vehicles': vehicles.copy()
                        })
        
        return conflicts
    
    def _estimate_path_time(self, path, speed):
        """估计路径完成时间"""
        if not path or len(path) < 2:
            return 0
            
        total_length = self._calculate_path_length(path)
        return total_length / speed