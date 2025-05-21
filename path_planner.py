import math
import numpy as np
from collections import defaultdict
from RRT import RRTPlanner
class PathPlanner:
    """路径规划器，连接车辆与主干路径网络
    
    PathPlanner是车辆与主干路径网络之间的桥梁，负责规划完整路径：
    从车辆当前位置到主干网络入口，在主干网络中的路径，以及从主干网络出口到目标位置。
    """
    
    def __init__(self, env, backbone_network=None, rrt_planner=None, traffic_manager=None):
        """初始化路径规划器
        
        Args:
            env: 环境对象
            backbone_network: 主干路径网络对象，可选
            rrt_planner: RRT路径规划器，可选
            traffic_manager: 交通管理器，可选
        """
        self.env = env
        self.backbone_network = backbone_network
        if rrt_planner is None:
            try:
                self.rrt_planner = RRTPlanner(
                    env,
                    vehicle_length=6.0,  # 可根据实际需求调整这些参数
                    vehicle_width=3.0,
                    turning_radius=8.0,
                    step_size=0.6
                )
                print("已自动创建RRTPlanner实例")
            except Exception as e:
                print(f"警告: 无法创建RRTPlanner: {e}")
                self.rrt_planner = None
        else:
            self.rrt_planner = rrt_planner

        self.traffic_manager = traffic_manager
        self.route_cache = {}  # 路由缓存 {(start, goal): path}
        self.max_rrt_attempts = 3  # RRT规划最大尝试次数
        
        # 路径验证参数
        self.path_validation_enabled = True
        self.validation_sample_density = 10  # 验证采样密度
        
        # 路径平滑参数
        self.path_smoothing_enabled = True
        self.smoothing_factor = 0.5  # 平滑因子 [0-1]
        
        # 缓存管理
        self.use_cache = True
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        
        # 调试标志
        self.debug = False
    
    def set_backbone_network(self, backbone_network):
        """设置主干路径网络"""
        self.backbone_network = backbone_network
    
    def set_rrt_planner(self, rrt_planner):
        """设置RRT路径规划器"""
        self.rrt_planner = rrt_planner
    
    def set_traffic_manager(self, traffic_manager):
        """设置交通管理器"""
        self.traffic_manager = traffic_manager
    
    def plan_path(self, vehicle_id, start, goal, use_backbone=True, check_conflicts=True):
        """规划从起点到终点的完整路径
        
        Args:
            vehicle_id: 车辆ID
            start: 起点坐标 (x, y, theta)
            goal: 终点坐标 (x, y, theta)
            use_backbone: 是否使用主干网络，默认为True
            check_conflicts: 是否检查路径冲突，默认为True
            
        Returns:
            list or None: 路径点列表，规划失败则返回None
        """
        # 检查起点和终点是否相同(考虑误差)
        if self._is_same_position(start, goal):
            return [start]
        
        # 检查缓存
        if self.use_cache:
            cache_key = (self._position_to_tuple(start), self._position_to_tuple(goal))
            if cache_key in self.route_cache:
                self.cache_hit_count += 1
                if self.debug:
                    print(f"[PathPlanner] 使用缓存路径: {cache_key} -> {len(self.route_cache[cache_key])}点")
                return self.route_cache[cache_key].copy()
            else:
                self.cache_miss_count += 1
        
        # 使用骨干网络辅助规划
        if use_backbone and self.backbone_network:
            path = self._plan_with_backbone_and_rrt(vehicle_id, start, goal)
            
            # 路径验证
            if path and self.path_validation_enabled and not self._validate_path(path):
                if self.debug:
                    print(f"[PathPlanner] 骨干网络辅助路径验证失败，尝试直接RRT规划")
                path = None  # 验证失败，置空，下面尝试直接RRT规划
            
            if path:
                # 保存到缓存
                if self.use_cache:
                    cache_key = (self._position_to_tuple(start), self._position_to_tuple(goal))
                    self.route_cache[cache_key] = path.copy()
                
                # 检查路径冲突
                if check_conflicts and self.traffic_manager:
                    if self.traffic_manager.check_path_conflicts(vehicle_id, path):
                        adjusted_path = self.traffic_manager.suggest_path_adjustment(vehicle_id, start, goal)
                        if adjusted_path:
                            return adjusted_path
                
                return path
        
        # 骨干网络规划失败或不使用骨干网络，直接使用RRT规划
        for attempt in range(self.max_rrt_attempts):
            direct_path = self._plan_direct_path(start, goal)
            
            # 路径验证
            if direct_path and self.path_validation_enabled and not self._validate_path(direct_path):
                if self.debug:
                    print(f"[PathPlanner] 直接RRT路径验证失败，尝试重新规划 (尝试 {attempt+1}/{self.max_rrt_attempts})")
                continue
            
            if direct_path:
                # 保存到缓存
                if self.use_cache:
                    cache_key = (self._position_to_tuple(start), self._position_to_tuple(goal))
                    self.route_cache[cache_key] = direct_path.copy()
                
                # 检查路径冲突
                if check_conflicts and self.traffic_manager:
                    if self.traffic_manager.check_path_conflicts(vehicle_id, direct_path):
                        adjusted_path = self.traffic_manager.suggest_path_adjustment(vehicle_id, start, goal)
                        if adjusted_path:
                            return adjusted_path
                
                return direct_path
        
        # 所有尝试都失败
        if self.debug:
            print(f"[PathPlanner] 路径规划失败: {start} -> {goal}")
        return None
    
    def _plan_with_backbone_and_rrt(self, vehicle_id, start, goal):
        """结合骨干网络和RRT进行规划
        
        Args:
            vehicle_id: 车辆ID
            start: 起点坐标
            goal: 终点坐标
        
        Returns:
            list or None: 路径点列表
        """
        if not self.backbone_network or not self.rrt_planner:
            return None
        
        # 1. 寻找起点附近可通过RRT到达的骨干路径点
        start_candidates = self.backbone_network.find_accessible_points(
            start, self.rrt_planner
        )
        if not start_candidates:
            if self.debug:
                print(f"[PathPlanner] 无法从起点 {start} 到达任何骨干路径点")
            return None  # 无法从起点到达任何骨干路径点
            
        # 2. 寻找终点附近可通过RRT到达的骨干路径点
        goal_candidates = self.backbone_network.find_accessible_points(
            goal, self.rrt_planner
        )
        if not goal_candidates:
            if self.debug:
                print(f"[PathPlanner] 无法从任何骨干路径点到达终点 {goal}")
            return None  # 无法从任何骨干路径点到达终点
            
        # 3. 遍历可能的组合，找出最佳路径
        best_path = None
        best_length = float('inf')
        
        for start_point in start_candidates[:2]:  # 限制尝试次数
            for goal_point in goal_candidates[:2]:
                # 规划三段路径
                
                # a. 从起点到骨干入口点
                path_to_backbone = self._plan_local_path(
                    start, start_point['position']
                )
                if not path_to_backbone:
                    continue
                    
                # b. 在骨干网络中的路径
                backbone_path = self._get_backbone_segment(
                    start_point['path_id'],
                    start_point['path_index'],
                    goal_point['path_id'],
                    goal_point['path_index']
                )
                if not backbone_path:
                    continue
                    
                # c. 从骨干出口点到终点
                path_from_backbone = self._plan_local_path(
                    goal_point['position'], goal
                )
                if not path_from_backbone:
                    continue
                
                # 合并三段路径
                complete_path = self._merge_paths(
                    path_to_backbone, backbone_path, path_from_backbone
                )
                
                # 验证完整路径
                if self.path_validation_enabled and not self._validate_path(complete_path):
                    continue
                
                # 计算路径长度
                path_length = self._calculate_path_length(complete_path)
                if path_length < best_length:
                    best_path = complete_path
                    best_length = path_length
        
        # 如果找到有效路径，进行平滑处理
        if best_path and self.path_smoothing_enabled:
            best_path = self._smooth_path(best_path)
        
        return best_path
    
    def _plan_direct_path(self, start, goal):
        """使用RRT规划器直接规划路径
        
        Args:
            start: 起点坐标
            goal: 终点坐标
            
        Returns:
            list or None: 路径点列表
        """
        if self.rrt_planner:
            try:
                # 使用RRT规划器规划路径
                path = self.rrt_planner.plan_path(start, goal)
                
                if path and self.path_smoothing_enabled:
                    path = self._smooth_path(path)
                
                if self.debug and path:
                    print(f"[PathPlanner] RRT规划成功: {len(path)}点")
                
                return path
            except Exception as e:
                if self.debug:
                    print(f"[PathPlanner] RRT规划器异常: {str(e)}")
                return None
        
        # 如果RRT规划器不可用，使用简单直线路径
        return self._plan_simple_path(start, goal)
    
    def _plan_local_path(self, start, goal):
        """规划本地路径（使用RRT规划器）
        
        Args:
            start: 起点坐标
            goal: 终点坐标
            
        Returns:
            list or None: 路径点列表
        """
        if self.rrt_planner:
            try:
                path = self.rrt_planner.plan_path(start, goal)
                return path
            except Exception as e:
                if self.debug:
                    print(f"[PathPlanner] 本地路径规划异常: {str(e)}")
        
        # 回退到简单路径规划
        return self._plan_simple_path(start, goal)
    
    def _plan_simple_path(self, start, goal):
        """简单直线路径规划
        
        当RRT规划器不可用时的后备方案。
        
        Args:
            start: 起点坐标
            goal: 终点坐标
            
        Returns:
            list: 路径点列表
        """
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
            
            # 起点和终点使用原始角度，中间点使用方向角度
            if i == 0:
                angle = s_theta
            elif i == num_points - 1:
                angle = g_theta
            else:
                angle = theta
                
            path.append((x, y, angle))
        
        return path
    
    def _get_backbone_segment(self, start_path_id, start_index, end_path_id, end_index):
        """获取骨干网络中的路径段
        
        Args:
            start_path_id: 起始路径ID
            start_index: 起始路径索引
            end_path_id: 结束路径ID
            end_index: 结束路径索引
            
        Returns:
            list or None: 路径段点列表
        """
        if not self.backbone_network:
            return None
        
        # 如果在同一路径上
        if start_path_id == end_path_id:
            return self.backbone_network.get_path_segment(start_path_id, start_index, end_index)
        
        # 不在同一路径上，需要进行路由
        # 首先需要找到路径起终点对应的节点ID
        start_node_id = self._get_node_id_for_path(start_path_id, start_index)
        end_node_id = self._get_node_id_for_path(end_path_id, end_index)
        
        if not start_node_id or not end_node_id:
            if self.debug:
                print(f"[PathPlanner] 无法找到路径对应的节点ID: {start_path_id}[{start_index}] -> {end_path_id}[{end_index}]")
            return None
        
        # 在主干网络中查找路径
        path_ids = self.backbone_network.find_path(start_node_id, end_node_id)
        
        if not path_ids:
            if self.debug:
                print(f"[PathPlanner] 在主干网络中无法找到路径: {start_node_id} -> {end_node_id}")
            return None
        
        # 构建完整路径
        full_path = []
        
        # 添加第一段
        if start_index > 0:
            first_segment = self.backbone_network.get_path_segment(start_path_id, start_index, 0)
            if first_segment:
                full_path.extend(first_segment)
        
        # 添加中间段
        for i, path_id in enumerate(path_ids):
            path_data = self.backbone_network.paths.get(path_id)
            if not path_data:
                continue
                
            path = path_data['path']
            
            # 如果不是第一段，跳过第一个点以避免重复
            if i > 0 or start_index > 0:
                path = path[1:]
                
            full_path.extend(path)
        
        # 添加最后一段
        if end_index < len(self.backbone_network.paths.get(end_path_id, {}).get('path', [])) - 1:
            last_segment = self.backbone_network.get_path_segment(
                end_path_id,
                0,
                end_index
            )
            if last_segment and len(last_segment) > 1:
                full_path.extend(last_segment[1:])  # 跳过第一个点以避免重复
        
        return full_path
    
    def _get_node_id_for_path(self, path_id, path_index):
        """获取路径点对应的节点ID
        
        Args:
            path_id: 路径ID
            path_index: 路径索引
            
        Returns:
            str or None: 节点ID
        """
        if not self.backbone_network or path_id not in self.backbone_network.paths:
            return None
            
        path_data = self.backbone_network.paths[path_id]
        path_length = len(path_data['path'])
        
        # 如果索引靠近路径起点，使用起点节点
        if path_index < path_length / 2:
            return path_data['start']['id']
        else:
            # 否则使用终点节点
            return path_data['end']['id']
    
    def _merge_paths(self, path_to_backbone, backbone_path, path_from_backbone):
        """合并三段路径，避免重复点
        
        Args:
            path_to_backbone: 从起点到骨干网络的路径
            backbone_path: 骨干网络中的路径
            path_from_backbone: 从骨干网络到终点的路径
            
        Returns:
            list: 合并后的路径
        """
        if not path_to_backbone or not backbone_path or not path_from_backbone:
            return None
        
        # 合并三段路径，避免重复点
        merged_path = path_to_backbone[:-1]  # 不包括末尾点，避免重复
        
        # 检查第一段末点和第二段起点是否足够接近
        if self._is_same_position(path_to_backbone[-1], backbone_path[0]):
            # 如果接近，跳过第二段的起点
            merged_path.extend(backbone_path[1:-1])
        else:
            merged_path.extend(backbone_path[:-1])
        
        # 检查第二段末点和第三段起点是否足够接近
        if self._is_same_position(backbone_path[-1], path_from_backbone[0]):
            # 如果接近，跳过第三段的起点
            merged_path.extend(path_from_backbone[1:])
        else:
            merged_path.extend(path_from_backbone)
        
        return merged_path
    
    def _validate_path(self, path):
        """验证路径是否有效（不穿越障碍物）
        
        Args:
            path: 路径点列表
            
        Returns:
            bool: 路径是否有效
        """
        if not path or len(path) < 2:
            return False
            
        # 检查路径上的每个线段是否穿越障碍物
        for i in range(len(path) - 1):
            if not self._validate_segment(path[i], path[i+1]):
                return False
                
        return True
    
    def _validate_segment(self, p1, p2):
        """验证路径段是否有效
        
        Args:
            p1: 起点坐标
            p2: 终点坐标
            
        Returns:
            bool: 路径段是否有效
        """
        # 采样检查点
        num_checks = max(10, int(self._calculate_distance(p1, p2) / 2))
        
        for i in range(1, num_checks):
            t = i / num_checks
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            
            # 检查点是否在障碍物内
            if not self._is_valid_position(int(x), int(y)):
                return False
                
        return True
    
    def _is_valid_position(self, x, y):
        """检查位置是否有效（不是障碍物）
        
        Args:
            x, y: 位置坐标
            
        Returns:
            bool: 位置是否有效
        """
        # 检查是否在地图范围内
        if not (0 <= x < self.env.width and 0 <= y < self.env.height):
            return False
            
        # 检查是否是障碍物 (0=可通行, 1=障碍物)
        return self.env.grid[x, y] == 0
    
    def _smooth_path(self, path):
        """平滑路径，使路径更自然
        
        使用简单的移动平均方法平滑路径。
        
        Args:
            path: 原始路径点列表
            
        Returns:
            list: 平滑后的路径点列表
        """
        if not path or len(path) <= 2 or not self.path_smoothing_enabled:
            return path
            
        smoothed = [path[0]]  # 保留起点
        
        # 平滑因子
        alpha = self.smoothing_factor
        
        # 对中间点进行平滑
        for i in range(1, len(path) - 1):
            prev = path[i-1]
            curr = path[i]
            next_p = path[i+1]
            
            # 平滑x和y坐标 - 使用加权平均
            x = curr[0] * (1 - alpha) + (prev[0] + next_p[0]) * alpha / 2
            y = curr[1] * (1 - alpha) + (prev[1] + next_p[1]) * alpha / 2
            
            # 保持原有角度或计算新角度
            if len(curr) > 2:
                # 原有角度与路径方向的加权平均
                dx = next_p[0] - prev[0]
                dy = next_p[1] - prev[1]
                
                if abs(dx) > 0.001 or abs(dy) > 0.001:
                    new_theta = math.atan2(dy, dx)
                    theta = curr[2] * (1 - alpha) + new_theta * alpha
                else:
                    theta = curr[2]
            else:
                # 根据移动方向计算角度
                theta = math.atan2(next_p[1] - prev[1], next_p[0] - prev[0])
                
            # 检查平滑后的点是否有效
            if self._is_valid_position(int(x), int(y)):
                smoothed.append((x, y, theta))
            else:
                # 如果平滑后的点无效，保留原点
                smoothed.append(curr)
            
        smoothed.append(path[-1])  # 保留终点
        return smoothed
    
    def _is_same_position(self, pos1, pos2, tolerance=0.1):
        """判断两个点是否相同（考虑误差）
        
        Args:
            pos1: 第一个点坐标
            pos2: 第二个点坐标
            tolerance: 容差
            
        Returns:
            bool: 是否相同
        """
        return self._calculate_distance(pos1, pos2) < tolerance
    
    def _calculate_distance(self, pos1, pos2):
        """计算两点之间的欧几里得距离
        
        Args:
            pos1: 第一个点坐标
            pos2: 第二个点坐标
            
        Returns:
            float: 距离
        """
        x1 = pos1[0] if len(pos1) > 0 else 0
        y1 = pos1[1] if len(pos1) > 1 else 0
        x2 = pos2[0] if len(pos2) > 0 else 0
        y2 = pos2[1] if len(pos2) > 1 else 0
        
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    
    def _calculate_path_length(self, path):
        """计算路径总长度
        
        Args:
            path: 路径点列表
            
        Returns:
            float: 路径总长度
        """
        if not path or len(path) < 2:
            return 0
            
        length = 0
        for i in range(len(path) - 1):
            length += self._calculate_distance(path[i], path[i + 1])
        
        return length
    
    def _position_to_tuple(self, position):
        """将位置坐标转换为元组（用于缓存键）
        
        Args:
            position: 位置坐标
            
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
        self.cache_hit_count = 0
        self.cache_miss_count = 0
    
    def set_path_validation(self, enabled):
        """设置是否启用路径验证
        
        Args:
            enabled: 是否启用
        """
        self.path_validation_enabled = enabled
    
    def set_path_smoothing(self, enabled, factor=None):
        """设置路径平滑参数
        
        Args:
            enabled: 是否启用平滑
            factor: 平滑因子 [0-1]
        """
        self.path_smoothing_enabled = enabled
        if factor is not None:
            self.smoothing_factor = max(0.0, min(1.0, factor))
    
    def get_cache_stats(self):
        """获取缓存统计信息
        
        Returns:
            dict: 缓存统计信息
        """
        total = self.cache_hit_count + self.cache_miss_count
        hit_rate = self.cache_hit_count / total if total > 0 else 0
        
        return {
            'hit_count': self.cache_hit_count,
            'miss_count': self.cache_miss_count,
            'total': total,
            'hit_rate': hit_rate
        }
    
