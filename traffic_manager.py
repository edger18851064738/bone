class TrafficManager:
    """交通管理器，处理车辆流量和冲突"""
    
    def __init__(self, env, backbone_network=None):
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
        
        # 冲突检测参数
        self.conflict_detector = None
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
            'current_index': 0,
            'progress': 0.0,
            'estimated_completion_time': start_time + self._estimate_path_time(path, speed),
            'reservations': []  # 时间窗口预留
        }
        
        self.vehicle_paths[vehicle_id] = path_info
        
        # 创建路径预留
        self._create_path_reservations(vehicle_id, path_info)
        
        # 更新冲突状态
        self._update_conflict_status()
        
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
    
    def suggest_path_adjustment(self, vehicle_id, start, goal):
        """为避免冲突建议路径调整"""
        # 如果车辆已有路径，将其暂时释放
        had_path = vehicle_id in self.vehicle_paths
        if had_path:
            original_path_info = self.vehicle_paths[vehicle_id]
            self.release_vehicle_path(vehicle_id)
        
        # 创建临时规划器
        if hasattr(self.env, 'path_planner') and self.env.path_planner:
            planner = self.env.path_planner
        else:
            # 创建临时规划器，根据实际需要修改
            planner = None
        
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
        
        # 尝试生成几条不同的路径
        num_attempts = 3
        potential_paths = []
        
        for i in range(num_attempts):
            # 调整规划参数，例如在主干网络中选择不同的连接点
            # 具体实现取决于规划器的接口
            path = planner.plan_path(vehicle_id, start, goal)
            
            if path:
                # 检查路径冲突
                has_conflict = self.check_path_conflicts(vehicle_id, path)
                
                potential_paths.append({
                    'path': path,
                    'has_conflict': has_conflict,
                    'length': self._calculate_path_length(path)
                })
        
        # 如果有无冲突的路径，选择最短的
        conflict_free_paths = [p for p in potential_paths if not p['has_conflict']]
        
        if conflict_free_paths:
            best_path = min(conflict_free_paths, key=lambda p: p['length'])
            
            if had_path:
                # 与原路径比较
                original_length = self._calculate_path_length(original_path_info['path'])
                
                # 如果新路径比原路径长太多，可能不值得调整
                if best_path['length'] > original_length * 1.5:
                    # 恢复原路径
                    self.register_vehicle_path(
                        vehicle_id,
                        original_path_info['path'],
                        original_path_info['start_time'],
                        original_path_info['speed']
                    )
                    return None
            
            return best_path['path']
        
        # 如果所有路径都有冲突，或者无法生成路径
        if had_path:
            # 恢复原路径
            self.register_vehicle_path(
                vehicle_id,
                original_path_info['path'],
                original_path_info['start_time'],
                original_path_info['speed']
            )
        
        return None
    
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
    
    def _calculate_path_length(self, path):
        """计算路径总长度"""
        if not path or len(path) < 2:
            return 0
            
        length = 0
        for i in range(len(path) - 1):
            length += self._calculate_distance(path[i], path[i + 1])
        
        return length
    
    def _calculate_distance(self, pos1, pos2):
        """计算两点之间的欧几里得距离"""
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
    
    def _is_same_point(self, pos1, pos2, tolerance=0.1):
        """判断两个点是否相同（考虑误差）"""
        return self._calculate_distance(pos1, pos2) < tolerance
    
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