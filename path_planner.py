import math
import numpy as np
from collections import defaultdict

class PathPlanner:
    """路径规划器，连接车辆与主干路径网络
    
    PathPlanner是车辆与主干路径网络之间的桥梁，负责规划完整路径：
    从车辆当前位置到主干网络入口，在主干网络中的路径，以及从主干网络出口到目标位置。
    
    属性:
        env: 环境对象，提供地图和资源信息
        backbone_network: 主干路径网络对象
        local_planner: 本地路径规划器，用于连接主干网络
        route_cache: 路由缓存，避免重复计算
        traffic_manager: 交通管理器，可选，用于路径冲突检测
    """
    
    def __init__(self, env, backbone_network=None, local_planner=None, traffic_manager=None):
        """初始化路径规划器
        
        Args:
            env: 环境对象
            backbone_network: 主干路径网络对象，可选
            local_planner: 本地路径规划器，用于连接主干网络，可选
            traffic_manager: 交通管理器，可选
        """
        self.env = env
        self.backbone_network = backbone_network
        self.local_planner = local_planner
        self.traffic_manager = traffic_manager
        self.route_cache = {}  # 路由缓存 {(start, goal): path}
        self.vehicle_routes = {}  # 车辆当前路由 {vehicle_id: route_info}
        self.max_connection_distance = 50.0  # 最大连接距离
        self.path_smoothing = True  # 是否平滑路径
        self.use_cache = True  # 是否使用缓存
        self.debug = False  # 调试模式
    
    def set_backbone_network(self, backbone_network):
        """设置主干路径网络
        
        Args:
            backbone_network: 主干路径网络对象
        """
        self.backbone_network = backbone_network
    
    def set_local_planner(self, local_planner):
        """设置本地路径规划器
        
        Args:
            local_planner: 本地路径规划器对象
        """
        self.local_planner = local_planner
    
    def set_traffic_manager(self, traffic_manager):
        """设置交通管理器
        
        Args:
            traffic_manager: 交通管理器对象
        """
        self.traffic_manager = traffic_manager
    
    def plan_path(self, vehicle_id, start, goal, use_backbone=True, check_conflicts=True):
        """规划从起点到终点的完整路径
        
        尝试使用主干路径网络规划路径，如果不可行则使用直接规划。
        
        Args:
            vehicle_id: 车辆ID
            start: 起点坐标 (x, y, theta)
            goal: 终点坐标 (x, y, theta)
            use_backbone: 是否使用主干网络，默认为True
            check_conflicts: 是否检查路径冲突，默认为True
            
        Returns:
            list or None: 路径点列表，规划失败则返回None
        """
        # 首先检查起点和终点是否相同(考虑误差)
        if self._is_same_position(start, goal):
            return [start]
        
        # 检查缓存
        cache_key = (self._position_to_tuple(start), self._position_to_tuple(goal))
        if self.use_cache and cache_key in self.route_cache:
            path = self.route_cache[cache_key].copy()
            
            if self.debug:
                print(f"使用缓存路径: {cache_key} -> {len(path)}点")
                
            # 检查路径冲突
            if check_conflicts and self.traffic_manager:
                if self.traffic_manager.check_path_conflicts(vehicle_id, path):
                    # 路径冲突，尝试调整路径
                    adjusted_path = self.traffic_manager.suggest_path_adjustment(vehicle_id, start, goal)
                    if adjusted_path:
                        return adjusted_path
            
            return path
        
        if use_backbone and self.backbone_network:
            # 使用主干网络规划路径
            path = self._plan_with_backbone(vehicle_id, start, goal)
            
            if path:
                # 保存到缓存
                if self.use_cache:
                    self.route_cache[cache_key] = path.copy()
                
                # 检查路径冲突
                if check_conflicts and self.traffic_manager:
                    if self.traffic_manager.check_path_conflicts(vehicle_id, path):
                        # 路径冲突，尝试调整路径
                        adjusted_path = self.traffic_manager.suggest_path_adjustment(vehicle_id, start, goal)
                        if adjusted_path:
                            return adjusted_path
                
                # 更新车辆路由信息
                self.vehicle_routes[vehicle_id] = {
                    'start': start,
                    'goal': goal,
                    'path': path,
                    'type': 'backbone',
                    'time': self.env.current_time if hasattr(self.env, 'current_time') else 0
                }
                
                return path
        
        # 主干网络规划失败或不使用主干网络，尝试直接规划
        path = self._plan_direct_path(start, goal)
        
        if path:
            # 保存到缓存
            if self.use_cache:
                self.route_cache[cache_key] = path.copy()
            
            # 检查路径冲突
            if check_conflicts and self.traffic_manager:
                if self.traffic_manager.check_path_conflicts(vehicle_id, path):
                    # 路径冲突，尝试调整路径
                    adjusted_path = self.traffic_manager.suggest_path_adjustment(vehicle_id, start, goal)
                    if adjusted_path:
                        return adjusted_path
            
            # 更新车辆路由信息
            self.vehicle_routes[vehicle_id] = {
                'start': start,
                'goal': goal,
                'path': path,
                'type': 'direct',
                'time': self.env.current_time if hasattr(self.env, 'current_time') else 0
            }
            
            return path
        
        # 所有规划方法都失败
        return None
    
    def _plan_with_backbone(self, vehicle_id, start, goal):
        """使用主干网络规划路径
        
        将路径分为三段：起点到主干入口、主干网络中的路径、主干出口到终点
        
        Args:
            vehicle_id: 车辆ID
            start: 起点坐标 (x, y, theta)
            goal: 终点坐标 (x, y, theta)
            
        Returns:
            list or None: 路径点列表，规划失败则返回None
        """
        if not self.backbone_network:
            return None
        
        # 查找最近的主干网络连接点
        start_conn = self.backbone_network.find_nearest_connection(start, self.max_connection_distance)
        goal_conn = self.backbone_network.find_nearest_connection(goal, self.max_connection_distance)
        
        if not start_conn or not goal_conn:
            if self.debug:
                print(f"无法找到合适的主干网络连接点: start_conn={start_conn is not None}, goal_conn={goal_conn is not None}")
            return None
        
        # 构建完整路径：起点 -> 主干入口 -> 主干网络 -> 主干出口 -> 终点
        
        # 1. 从起点到主干网络入口的路径
        path_to_backbone = self._plan_local_path(start, start_conn['position'])
        
        if not path_to_backbone:
            if self.debug:
                print(f"无法规划从起点到主干网络入口的路径")
            return None
        
        # 2. 在主干网络中的路径
        backbone_path = self._plan_backbone_path(start_conn, goal_conn)
        
        if not backbone_path:
            if self.debug:
                print(f"无法在主干网络中规划路径")
            return None
        
        # 3. 从主干网络出口到终点的路径
        path_from_backbone = self._plan_local_path(goal_conn['position'], goal)
        
        if not path_from_backbone:
            if self.debug:
                print(f"无法规划从主干网络出口到终点的路径")
            return None
        
        # 合并三段路径，避免重复点
        complete_path = path_to_backbone[:-1] + backbone_path + path_from_backbone
        
        # 如果需要，平滑路径
        if self.path_smoothing:
            complete_path = self._smooth_path(complete_path)
        
        # 更新主干网络流量
        self._update_backbone_traffic(start_conn, goal_conn, vehicle_id, 1)
        
        if self.debug:
            print(f"成功使用主干网络规划路径: {len(complete_path)}点")
            print(f"  起点到主干入口: {len(path_to_backbone)}点")
            print(f"  主干网络: {len(backbone_path)}点")
            print(f"  主干出口到终点: {len(path_from_backbone)}点")
        
        return complete_path
    
    def _plan_direct_path(self, start, goal):
        """使用本地规划器直接规划路径
        
        Args:
            start: 起点坐标 (x, y, theta)
            goal: 终点坐标 (x, y, theta)
            
        Returns:
            list or None: 路径点列表
        """
        if self.local_planner:
            try:
                # 使用本地规划器规划路径
                path = self.local_planner.plan_path(start, goal)
                
                if path:
                    # 平滑路径
                    if self.path_smoothing:
                        path = self._smooth_path(path)
                    
                    if self.debug:
                        print(f"本地规划器成功规划路径: {len(path)}点")
                    
                    return path
                else:
                    if self.debug:
                        print(f"本地规划器规划路径失败")
            except Exception as e:
                if self.debug:
                    print(f"本地规划器异常: {str(e)}")
        
        # 本地规划器失败或不可用，使用简单直线路径（实际应使用更复杂的避障算法）
        try:
            # 确保起点和终点有角度
            if len(start) < 3:
                s_x, s_y = start[:2]
                s_theta = 0.0
            else:
                s_x, s_y, s_theta = start
                
            if len(goal) < 3:
                g_x, g_y = goal[:2]
                g_theta = 0.0
            else:
                g_x, g_y, g_theta = goal
            
            # 计算方向角度
            theta = math.atan2(g_y - s_y, g_x - s_x)
            
            # 计算距离
            distance = math.sqrt((g_x - s_x) ** 2 + (g_y - s_y) ** 2)
            
            # 点的数量（根据距离调整）
            num_points = max(2, int(distance / 5))  # 每5个单位一个点
            
            # 创建路径点
            path = []
            for i in range(num_points):
                t = i / (num_points - 1)
                x = s_x + t * (g_x - s_x)
                y = s_y + t * (g_y - s_y)
                # 线性插值角度
                angle = s_theta + t * (g_theta - s_theta)
                path.append((x, y, angle))
            
            if self.debug:
                print(f"生成简单直线路径: {len(path)}点")
            
            return path
        except Exception as e:
            if self.debug:
                print(f"直线路径生成异常: {str(e)}")
            
            # 作为最后的手段，只返回起点和终点
            return [start, goal]
    
    def _plan_local_path(self, start, goal):
        """规划本地路径（非主干网络部分）
        
        如果本地规划器可用，使用它；否则使用直线路径
        
        Args:
            start: 起点坐标 (x, y, theta)
            goal: 终点坐标 (x, y, theta)
            
        Returns:
            list: 路径点列表
        """
        return self._plan_direct_path(start, goal)
    
    def _plan_backbone_path(self, start_conn, goal_conn):
        """在主干网络中规划路径
        
        Args:
            start_conn: 起始连接点信息
            goal_conn: 目标连接点信息
            
        Returns:
            list or None: 路径点列表
        """
        # 如果在同一条路径上
        if start_conn['path_id'] == goal_conn['path_id']:
            # 直接获取该路径段
            path_segment = self.backbone_network.get_path_segment(
                start_conn['path_id'],
                start_conn['path_index'],
                goal_conn['path_index']
            )
            
            if self.debug and path_segment:
                print(f"在同一主干路径上的路径段: {len(path_segment)}点")
                
            return path_segment
        
        # 不在同一条路径上，需要路由
        # 获取连接点所在节点的ID
        start_node_id = self._get_node_id_for_connection(start_conn)
        goal_node_id = self._get_node_id_for_connection(goal_conn)
        
        if not start_node_id or not goal_node_id:
            if self.debug:
                print(f"无法确定连接点所属节点: start={start_node_id}, goal={goal_node_id}")
            return None
        
        # 在路径图中查找最优路径
        path_ids = self.backbone_network.find_path(start_node_id, goal_node_id)
        
        if not path_ids:
            if self.debug:
                print(f"在主干网络中无法找到路径: {start_node_id} -> {goal_node_id}")
            return None
        
        # 构建完整路径
        return self._build_path_from_segments(start_conn, goal_conn, path_ids, start_node_id, goal_node_id)
    
    def _get_node_id_for_connection(self, connection):
        """获取连接点所属的节点ID
        
        Args:
            connection: 连接点信息
            
        Returns:
            str or None: 节点ID
        """
        if not self.backbone_network or not connection or 'path_id' not in connection:
            return None
            
        try:
            path_id = connection['path_id']
            path_index = connection['path_index']
            
            if path_id not in self.backbone_network.paths:
                return None
                
            path_data = self.backbone_network.paths[path_id]
            path_length = len(path_data['path'])
            
            # 如果连接点靠近路径起点，使用起点节点ID
            if path_index < path_length / 2:
                return path_data['start']['id']
            else:
                # 否则使用终点节点ID
                return path_data['end']['id']
        except (KeyError, TypeError) as e:
            if self.debug:
                print(f"获取连接点节点ID异常: {str(e)}")
            return None
    
    def _build_path_from_segments(self, start_conn, goal_conn, path_ids, start_node_id, goal_node_id):
        """从路径段构建完整路径
        
        Args:
            start_conn: 起始连接点信息
            goal_conn: 目标连接点信息
            path_ids: 路径ID列表
            start_node_id: 起始节点ID
            goal_node_id: 目标节点ID
            
        Returns:
            list or None: 路径点列表
        """
        if not path_ids:
            return None
            
        # 构建完整路径
        complete_path = []
        
        try:
            # 1. 从起始连接点到第一条路径的对应端点
            first_path_id = path_ids[0]
            first_path_data = self.backbone_network.paths[first_path_id]
            
            # 确定需要连接到第一条路径的哪个端点
            if start_node_id == first_path_data['start']['id']:
                # 连接到起点
                first_segment = self.backbone_network.get_path_segment(
                    start_conn['path_id'],
                    start_conn['path_index'],
                    0  # 路径起点
                )
                
                if first_segment:
                    complete_path.extend(first_segment[:-1])  # 不包括末尾点，避免重复
                
                # 然后添加第一条路径（从起点到终点）
                complete_path.extend(first_path_data['path'])
            else:
                # 连接到终点
                first_segment = self.backbone_network.get_path_segment(
                    start_conn['path_id'],
                    start_conn['path_index'],
                    len(self.backbone_network.paths[start_conn['path_id']]['path']) - 1  # 路径终点
                )
                
                if first_segment:
                    complete_path.extend(first_segment[:-1])  # 不包括末尾点，避免重复
                
                # 然后添加第一条路径（从终点到起点，需要反转）
                first_path = first_path_data['path'].copy()
                first_path.reverse()
                complete_path.extend(first_path)
            
            # 2. 添加中间路径段
            for i in range(1, len(path_ids)):
                current_path_id = path_ids[i]
                current_path_data = self.backbone_network.paths[current_path_id]
                prev_path_id = path_ids[i-1]
                prev_path_data = self.backbone_network.paths[prev_path_id]
                
                # 找出上一条路径的终点和当前路径的起点
                prev_end_node_id = prev_path_data['end']['id']
                curr_start_node_id = current_path_data['start']['id']
                
                # 确定路径方向
                if prev_end_node_id == curr_start_node_id:
                    # 正向添加当前路径
                    complete_path.extend(current_path_data['path'][1:])  # 从第二个点开始，避免重复
                else:
                    # 反向添加当前路径
                    current_path = current_path_data['path'].copy()
                    current_path.reverse()
                    complete_path.extend(current_path[1:])  # 从第二个点开始，避免重复
            
            # 3. 从最后一条路径的对应端点到目标连接点
            last_path_id = path_ids[-1]
            last_path_data = self.backbone_network.paths[last_path_id]
            
            # 确定最后一条路径的哪个端点需要连接到目标
            if goal_node_id == last_path_data['end']['id']:
                # 从终点连接
                last_segment = self.backbone_network.get_path_segment(
                    goal_conn['path_id'],
                    len(self.backbone_network.paths[goal_conn['path_id']]['path']) - 1,  # 路径终点
                    goal_conn['path_index']
                )
                
                if last_segment:
                    complete_path.extend(last_segment[1:])  # 从第二个点开始，避免重复
            else:
                # 从起点连接
                last_segment = self.backbone_network.get_path_segment(
                    goal_conn['path_id'],
                    0,  # 路径起点
                    goal_conn['path_index']
                )
                
                if last_segment:
                    complete_path.extend(last_segment[1:])  # 从第二个点开始，避免重复
            
            if self.debug:
                print(f"成功构建主干网络路径: {len(complete_path)}点")
        except Exception as e:
            if self.debug:
                print(f"构建主干网络路径异常: {str(e)}")
            return None
        
        return complete_path
    
    def _update_backbone_traffic(self, start_conn, goal_conn, vehicle_id, delta=1):
        """更新主干网络的交通流量
        
        Args:
            start_conn: 起始连接点信息
            goal_conn: 目标连接点信息
            vehicle_id: 车辆ID
            delta: 流量变化值，正数表示增加，负数表示减少
        """
        # 清空车辆的路由缓存
        if vehicle_id not in self.route_cache:
            self.route_cache[vehicle_id] = []
        
        # 如果是减少流量
        if delta < 0:
            # 释放所有路径流量
            if vehicle_id in self.vehicle_routes:
                route_info = self.vehicle_routes[vehicle_id]
                
                if route_info['type'] == 'backbone' and 'path_ids' in route_info:
                    for path_id in route_info['path_ids']:
                        self.backbone_network.update_traffic_flow(path_id, delta)
                
                # 清除路由信息
                del self.vehicle_routes[vehicle_id]
            
            return
        
        # 增加流量
        if delta > 0 and start_conn and goal_conn:
            # 如果在同一路径上
            if start_conn['path_id'] == goal_conn['path_id']:
                self.backbone_network.update_traffic_flow(start_conn['path_id'], delta)
                
                # 保存路径信息
                if vehicle_id in self.vehicle_routes:
                    self.vehicle_routes[vehicle_id]['path_ids'] = [start_conn['path_id']]
                
                return
            
            # 不在同一路径上，需要确定完整路径
            start_node_id = self._get_node_id_for_connection(start_conn)
            goal_node_id = self._get_node_id_for_connection(goal_conn)
            
            if start_node_id and goal_node_id:
                path_ids = self.backbone_network.find_path(start_node_id, goal_node_id)
                
                if path_ids:
                    # 更新所有路径的流量
                    for path_id in path_ids:
                        self.backbone_network.update_traffic_flow(path_id, delta)
                    
                    # 保存路径信息
                    if vehicle_id in self.vehicle_routes:
                        self.vehicle_routes[vehicle_id]['path_ids'] = path_ids
    
    def release_path(self, vehicle_id):
        """释放车辆占用的路径流量
        
        Args:
            vehicle_id: 车辆ID
            
        Returns:
            bool: 操作是否成功
        """
        # 更新主干网络流量
        self._update_backbone_traffic(None, None, vehicle_id, -1)
        
        # 清除路由信息
        if vehicle_id in self.vehicle_routes:
            del self.vehicle_routes[vehicle_id]
        
        # 如果有交通管理器，也释放路径
        if self.traffic_manager:
            self.traffic_manager.release_vehicle_path(vehicle_id)
        
        return True
    
    def _smooth_path(self, path):
        """平滑路径，使路径更自然
        
        使用简单的移动平均方法平滑路径。
        
        Args:
            path (list): 原始路径点列表
            
        Returns:
            list: 平滑后的路径点列表
        """
        if not path or len(path) <= 2:
            return path
            
        smoothed = [path[0]]  # 保留起点
        
        # 对中间点进行平滑
        for i in range(1, len(path) - 1):
            prev = path[i-1]
            curr = path[i]
            next_p = path[i+1]
            
            # 简单的三点平均平滑
            x = (prev[0] + curr[0] + next_p[0]) / 3
            y = (prev[1] + curr[1] + next_p[1]) / 3
            
            # 保持原有角度或计算新角度
            if len(curr) > 2:
                theta = curr[2]
            else:
                # 根据移动方向计算角度
                theta = math.atan2(next_p[1] - prev[1], next_p[0] - prev[0])
                
            smoothed.append((x, y, theta))
            
        smoothed.append(path[-1])  # 保留终点
        return smoothed
    
    def _is_same_position(self, pos1, pos2, tolerance=0.1):
        """判断两个位置是否相同（考虑误差）
        
        Args:
            pos1 (tuple): 第一个位置坐标
            pos2 (tuple): 第二个位置坐标
            tolerance (float): 容差，默认0.1
            
        Returns:
            bool: 是否相同
        """
        x1 = pos1[0] if len(pos1) > 0 else 0
        y1 = pos1[1] if len(pos1) > 1 else 0
        x2 = pos2[0] if len(pos2) > 0 else 0
        y2 = pos2[1] if len(pos2) > 1 else 0
        
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) < tolerance
    
    def _position_to_tuple(self, position):
        """将位置坐标转换为元组（用于缓存键）
        
        Args:
            position (tuple): 位置坐标
            
        Returns:
            tuple: 标准化的位置元组
        """
        x = position[0] if len(position) > 0 else 0
        y = position[1] if len(position) > 1 else 0
        theta = position[2] if len(position) > 2 else 0
        
        # 四舍五入以降低精度，避免浮点误差导致缓存失效
        return (round(x, 2), round(y, 2), round(theta, 2))
    
    def clear_cache(self):
        """清除路径缓存"""
        self.route_cache.clear()
    
    def get_route_info(self, vehicle_id):
        """获取车辆的路由信息
        
        Args:
            vehicle_id: 车辆ID
            
        Returns:
            dict or None: 路由信息
        """
        return self.vehicle_routes.get(vehicle_id)
    
    def set_max_connection_distance(self, distance):
        """设置最大连接距离
        
        Args:
            distance (float): 最大连接距离
        """
        self.max_connection_distance = max(1.0, distance)
    
    def set_path_smoothing(self, enabled):
        """设置是否启用路径平滑
        
        Args:
            enabled (bool): 是否启用
        """
        self.path_smoothing = enabled
    
    def set_use_cache(self, enabled):
        """设置是否使用缓存
        
        Args:
            enabled (bool): 是否启用
        """
        self.use_cache = enabled
    
    def set_debug(self, enabled):
        """设置是否启用调试模式
        
        Args:
            enabled (bool): 是否启用
        """
        self.debug = enabled