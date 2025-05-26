"""
hybridastar.py - 露天矿多车协同调度系统专用混合A*路径规划器
针对大型车辆、转弯半径约束、骨干网络集成等特点优化设计
"""

import math
import numpy as np
import time
import heapq
from typing import List, Dict, Tuple, Set, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field
import threading

@dataclass
class VehicleConfig:
    """车辆配置"""
    length: float = 6.0
    width: float = 3.0
    turning_radius: float = 8.0
    wheelbase: float = 4.5  # 轴距
    max_steer_angle: float = math.pi / 6  # 最大转向角30度

@dataclass
class MotionPrimitive:
    """运动原语"""
    dx: float
    dy: float
    dtheta: float
    cost: float
    steering_angle: float
    direction: int  # 1: forward, -1: backward
    
    def __post_init__(self):
        self.length = math.sqrt(self.dx**2 + self.dy**2)

@dataclass
class HybridNode:
    """混合A*节点"""
    x: float
    y: float
    theta: float
    g_cost: float
    h_cost: float
    f_cost: float
    parent: Optional['HybridNode'] = None
    motion_primitive: Optional[MotionPrimitive] = None
    
    # 网格坐标（用于hash和去重）
    grid_x: int = 0
    grid_y: int = 0
    grid_theta: int = 0
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost
    
    def __eq__(self, other):
        return (self.grid_x == other.grid_x and 
                self.grid_y == other.grid_y and 
                self.grid_theta == other.grid_theta)
    
    def __hash__(self):
        return hash((self.grid_x, self.grid_y, self.grid_theta))

class MotionPrimitiveGenerator:
    """运动原语生成器"""
    
    def __init__(self, vehicle_config: VehicleConfig, step_size: float = 1.0):
        self.vehicle_config = vehicle_config
        self.step_size = step_size
        self.primitives = []
        self._generate_primitives()
    
    def _generate_primitives(self):
        """生成运动原语集合"""
        # 转向角度范围
        steer_angles = [
            -self.vehicle_config.max_steer_angle,
            -self.vehicle_config.max_steer_angle * 0.5,
            0,  # 直行
            self.vehicle_config.max_steer_angle * 0.5,
            self.vehicle_config.max_steer_angle
        ]
        
        directions = [1, -1]  # 前进和后退
        
        for direction in directions:
            for steer_angle in steer_angles:
                primitive = self._compute_primitive(steer_angle, direction)
                if primitive:
                    self.primitives.append(primitive)
    
    def _compute_primitive(self, steer_angle: float, direction: int) -> Optional[MotionPrimitive]:
        """计算单个运动原语"""
        if abs(steer_angle) < 1e-6:
            # 直行
            dx = direction * self.step_size
            dy = 0
            dtheta = 0
            cost = self.step_size
        else:
            # 转弯运动学模型
            wheelbase = self.vehicle_config.wheelbase
            
            # 转弯半径
            turn_radius = wheelbase / math.tan(abs(steer_angle))
            
            # 弧长
            arc_length = self.step_size
            
            # 角度变化
            dtheta = direction * arc_length / turn_radius
            if steer_angle < 0:
                dtheta = -dtheta
            
            # 位移计算
            if abs(dtheta) > 1e-6:
                dx = turn_radius * math.sin(dtheta) * direction
                dy = turn_radius * (1 - math.cos(dtheta))
                if steer_angle < 0:
                    dy = -dy
            else:
                dx = direction * self.step_size
                dy = 0
            
            # 成本：距离 + 转向惩罚
            cost = arc_length + abs(steer_angle) * 0.5
            
            # 后退惩罚
            if direction < 0:
                cost *= 1.5
        
        return MotionPrimitive(
            dx=dx, dy=dy, dtheta=dtheta,
            cost=cost, steering_angle=steer_angle,
            direction=direction
        )

class HeuristicCalculator:
    """启发式函数计算器"""
    
    def __init__(self, env, backbone_network=None):
        self.env = env
        self.backbone_network = backbone_network
        
        # 预计算距离图（可选优化）
        self.distance_map_cache = {}
        self.use_backbone_guidance = backbone_network is not None
    
    def calculate_heuristic(self, node: HybridNode, goal: Tuple[float, float, float],
                          guidance_hints: Dict = None) -> float:
        """计算启发式值"""
        # 基础欧几里得距离
        euclidean_dist = math.sqrt(
            (goal[0] - node.x)**2 + (goal[1] - node.y)**2
        )
        
        # 角度差异惩罚
        angle_diff = abs(self._normalize_angle(goal[2] - node.theta))
        angle_penalty = angle_diff * 2.0
        
        # 障碍物影响
        obstacle_penalty = self._calculate_obstacle_penalty(node.x, node.y, goal)
        
        # 骨干路径引导
        backbone_bonus = 0
        if self.use_backbone_guidance and guidance_hints:
            backbone_bonus = self._calculate_backbone_guidance(node, goal, guidance_hints)
        
        # 综合启发式
        heuristic = euclidean_dist + angle_penalty + obstacle_penalty - backbone_bonus
        
        return max(0, heuristic)
    
    def _normalize_angle(self, angle: float) -> float:
        """角度归一化到[-π, π]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def _calculate_obstacle_penalty(self, x: float, y: float, goal: Tuple) -> float:
        """计算障碍物影响的惩罚"""
        if not hasattr(self.env, 'grid'):
            return 0
        
        penalty = 0
        check_radius = 8  # 检查半径
        
        # 检查周围障碍物密度
        obstacle_count = 0
        total_cells = 0
        
        for dx in range(-check_radius, check_radius + 1):
            for dy in range(-check_radius, check_radius + 1):
                check_x = int(x + dx)
                check_y = int(y + dy)
                
                if (0 <= check_x < self.env.width and 
                    0 <= check_y < self.env.height):
                    total_cells += 1
                    if self.env.grid[check_x, check_y] == 1:
                        distance = math.sqrt(dx*dx + dy*dy)
                        if distance < check_radius:
                            obstacle_count += 1
                            # 距离越近惩罚越大
                            penalty += (check_radius - distance) * 0.5
        
        # 密度惩罚
        if total_cells > 0:
            density = obstacle_count / total_cells
            penalty += density * 10
        
        return penalty
    
    def _calculate_backbone_guidance(self, node: HybridNode, goal: Tuple,
                                   guidance_hints: Dict) -> float:
        """计算骨干路径引导奖励"""
        if not self.backbone_network:
            return 0
        
        bonus = 0
        
        # 检查是否接近骨干路径
        for path_id, path_data in self.backbone_network.backbone_paths.items():
            backbone_path = path_data.get('path', [])
            
            # 采样检查距离
            min_distance = float('inf')
            for i in range(0, len(backbone_path), 5):  # 每5个点检查一次
                bp = backbone_path[i]
                distance = math.sqrt((node.x - bp[0])**2 + (node.y - bp[1])**2)
                min_distance = min(min_distance, distance)
            
            # 距离奖励
            if min_distance < 15:
                proximity_bonus = (15 - min_distance) * 0.8
                
                # 路径质量奖励
                path_quality = path_data.get('quality', 0.5)
                quality_bonus = proximity_bonus * path_quality
                
                bonus = max(bonus, quality_bonus)
        
        # 接口点奖励
        if hasattr(self.backbone_network, 'backbone_interfaces'):
            for interface_id, interface in self.backbone_network.backbone_interfaces.items():
                distance = math.sqrt(
                    (node.x - interface.position[0])**2 + 
                    (node.y - interface.position[1])**2
                )
                
                if distance < 10:
                    interface_bonus = (10 - distance) * 0.5
                    # 可达性奖励
                    accessibility = getattr(interface, 'accessibility_score', 0.5)
                    bonus = max(bonus, interface_bonus * accessibility)
        
        return bonus

class CollisionChecker:
    """碰撞检测器"""
    
    def __init__(self, env, vehicle_config: VehicleConfig):
        self.env = env
        self.vehicle_config = vehicle_config
        
        # 预计算车辆形状
        self._precompute_vehicle_shape()
    
    def _precompute_vehicle_shape(self):
        """预计算车辆形状相对坐标"""
        length = self.vehicle_config.length
        width = self.vehicle_config.width
        
        # 车辆四个角点的相对坐标
        self.vehicle_corners = [
            (length/2, width/2),
            (length/2, -width/2),
            (-length/2, -width/2),
            (-length/2, width/2)
        ]
        
        # 预计算检查点（车辆轮廓上的点）
        self.check_points = []
        
        # 车辆轮廓点
        num_points = 12
        for i in range(num_points):
            if i < 3:  # 前边
                x = length/2
                y = width/2 - i * width/2
            elif i < 6:  # 右边
                x = length/2 - (i-3) * length/3
                y = -width/2
            elif i < 9:  # 后边
                x = -length/2
                y = -width/2 + (i-6) * width/3
            else:  # 左边
                x = -length/2 + (i-9) * length/3
                y = width/2
            
            self.check_points.append((x, y))
    
    def is_collision_free(self, x: float, y: float, theta: float) -> bool:
        """检查指定位置是否无碰撞"""
        if not hasattr(self.env, 'grid'):
            return True
        
        # 边界检查
        margin = max(self.vehicle_config.length, self.vehicle_config.width) / 2
        if (x - margin < 0 or x + margin >= self.env.width or
            y - margin < 0 or y + margin >= self.env.height):
            return False
        
        # 车辆形状碰撞检查
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        
        for rel_x, rel_y in self.check_points:
            # 变换到全局坐标
            global_x = x + rel_x * cos_theta - rel_y * sin_theta
            global_y = y + rel_x * sin_theta + rel_y * cos_theta
            
            # 检查网格
            grid_x, grid_y = int(global_x), int(global_y)
            
            if (0 <= grid_x < self.env.width and 
                0 <= grid_y < self.env.height):
                if self.env.grid[grid_x, grid_y] == 1:
                    return False
            else:
                return False  # 超出边界
        
        return True
    
    def is_path_collision_free(self, path: List[Tuple]) -> bool:
        """检查路径是否无碰撞"""
        for point in path:
            if len(point) >= 3:
                if not self.is_collision_free(point[0], point[1], point[2]):
                    return False
        return True

class HybridAStarPlanner:
    """混合A*路径规划器 - 兼容RRT接口"""
    
    def __init__(self, env, vehicle_length=6.0, vehicle_width=3.0, 
                 turning_radius=8.0, step_size=1.0):
        """
        初始化混合A*规划器
        
        Args:
            env: 环境对象
            vehicle_length: 车辆长度
            vehicle_width: 车辆宽度  
            turning_radius: 转弯半径
            step_size: 步长
        """
        self.env = env
        self.vehicle_config = VehicleConfig(
            length=vehicle_length,
            width=vehicle_width,
            turning_radius=turning_radius
        )
        self.step_size = step_size
        
        # 核心组件
        self.motion_generator = MotionPrimitiveGenerator(self.vehicle_config, step_size)
        self.heuristic_calculator = HeuristicCalculator(env)
        self.collision_checker = CollisionChecker(env, self.vehicle_config)
        
        # 规划参数
        self.grid_resolution = 2.0  # 网格分辨率
        self.angle_resolution = math.pi / 12  # 角度分辨率(15度)
        self.max_iterations = 50000
        self.goal_tolerance = 3.0
        self.angle_tolerance = math.pi / 6
        
        # 骨干网络集成
        self.backbone_network = None
        self.use_backbone_guidance = False
        
        # 统计信息
        self.statistics = {
            'total_calls': 0,
            'successful_plans': 0,
            'average_time': 0,
            'average_iterations': 0,
            'cache_hits': 0,
            'backbone_assisted': 0
        }
        
        # 路径缓存
        self.path_cache = {}
        self.cache_max_size = 1000
        
        print(f"初始化混合A*规划器: 车辆({vehicle_length}x{vehicle_width}), "
              f"转弯半径{turning_radius}, 步长{step_size}")
    
    def set_backbone_network(self, backbone_network):
        """设置骨干路径网络"""
        self.backbone_network = backbone_network
        self.heuristic_calculator.backbone_network = backbone_network
        self.use_backbone_guidance = backbone_network is not None
        print("已设置骨干路径网络")
    
    def plan_path(self, start, goal, max_iterations=None, agent_id=None, quality_threshold=None):
        """
        规划路径 - 兼容RRT接口
        
        Args:
            start: 起点 (x, y, theta)
            goal: 终点 (x, y, theta)  
            max_iterations: 最大迭代次数
            agent_id: 智能体ID
            quality_threshold: 质量阈值
            
        Returns:
            List[Tuple]: 路径点列表
        """
        start_time = time.time()
        self.statistics['total_calls'] += 1
        
        # 参数处理
        if max_iterations is None:
            max_iterations = self.max_iterations
        
        # 输入验证
        if not self._validate_inputs(start, goal):
            return None
        
        # 缓存检查
        cache_key = self._generate_cache_key(start, goal)
        if cache_key in self.path_cache:
            self.statistics['cache_hits'] += 1
            return self.path_cache[cache_key]
        
        # 核心规划算法
        path = self._hybrid_astar_search(start, goal, max_iterations)
        
        # 路径后处理
        if path:
            path = self._post_process_path(path)
            
            # 质量检查
            if quality_threshold and not self._check_path_quality(path, quality_threshold):
                path = self._improve_path_quality(path, quality_threshold)
            
            # 缓存结果
            self._cache_path(cache_key, path)
            
            self.statistics['successful_plans'] += 1
            
            if self.use_backbone_guidance:
                self.statistics['backbone_assisted'] += 1
        
        # 更新统计
        planning_time = time.time() - start_time
        self._update_statistics(planning_time, len(self._get_last_iteration_count()) if hasattr(self, '_last_iteration_count') else 0)
        
        return path
    
    def _hybrid_astar_search(self, start: Tuple, goal: Tuple, max_iterations: int) -> Optional[List]:
        """混合A*搜索算法核心"""
        # 初始化
        start_node = self._create_node(start[0], start[1], start[2], 0)
        goal_pos = goal
        
        open_list = [start_node]
        closed_set = set()
        all_nodes = {self._get_node_key(start_node): start_node}
        
        iteration_count = 0
        
        # 骨干路径引导
        guidance_hints = self._get_backbone_guidance(start, goal)
        
        while open_list and iteration_count < max_iterations:
            iteration_count += 1
            
            # 获取最佳节点
            current_node = heapq.heappop(open_list)
            current_key = self._get_node_key(current_node)
            
            # 检查是否已在closed集合中
            if current_key in closed_set:
                continue
            
            closed_set.add(current_key)
            
            # 目标检查
            if self._is_goal_reached(current_node, goal_pos):
                self._last_iteration_count = iteration_count
                return self._reconstruct_path(current_node)
            
            # 扩展节点
            for primitive in self.motion_generator.primitives:
                successor = self._apply_motion_primitive(current_node, primitive)
                
                if not successor:
                    continue
                
                successor_key = self._get_node_key(successor)
                
                # 跳过已访问的节点
                if successor_key in closed_set:
                    continue
                
                # 碰撞检查
                if not self.collision_checker.is_collision_free(
                    successor.x, successor.y, successor.theta):
                    continue
                
                # 计算成本
                successor.g_cost = current_node.g_cost + primitive.cost
                successor.h_cost = self.heuristic_calculator.calculate_heuristic(
                    successor, goal_pos, guidance_hints
                )
                successor.f_cost = successor.g_cost + successor.h_cost
                successor.parent = current_node
                successor.motion_primitive = primitive
                
                # 检查是否已有更好的路径到达该节点
                if successor_key in all_nodes:
                    existing_node = all_nodes[successor_key]
                    if successor.g_cost < existing_node.g_cost:
                        # 更新现有节点
                        existing_node.g_cost = successor.g_cost
                        existing_node.f_cost = successor.f_cost
                        existing_node.parent = current_node
                        existing_node.motion_primitive = primitive
                        heapq.heappush(open_list, existing_node)
                else:
                    # 添加新节点
                    all_nodes[successor_key] = successor
                    heapq.heappush(open_list, successor)
        
        self._last_iteration_count = iteration_count
        return None  # 搜索失败
    
    def _create_node(self, x: float, y: float, theta: float, g_cost: float) -> HybridNode:
        """创建混合A*节点"""
        # 网格化坐标
        grid_x = int(x / self.grid_resolution)
        grid_y = int(y / self.grid_resolution)
        grid_theta = int(theta / self.angle_resolution)
        
        return HybridNode(
            x=x, y=y, theta=theta,
            g_cost=g_cost, h_cost=0, f_cost=g_cost,
            grid_x=grid_x, grid_y=grid_y, grid_theta=grid_theta
        )
    
    def _get_node_key(self, node: HybridNode) -> Tuple[int, int, int]:
        """获取节点的唯一键值"""
        return (node.grid_x, node.grid_y, node.grid_theta)
    
    def _apply_motion_primitive(self, node: HybridNode, primitive: MotionPrimitive) -> Optional[HybridNode]:
        """应用运动原语"""
        # 计算新状态
        cos_theta = math.cos(node.theta)
        sin_theta = math.sin(node.theta)
        
        new_x = node.x + primitive.dx * cos_theta - primitive.dy * sin_theta
        new_y = node.y + primitive.dx * sin_theta + primitive.dy * cos_theta
        new_theta = node.theta + primitive.dtheta
        
        # 角度归一化
        new_theta = self._normalize_angle(new_theta)
        
        # 边界检查
        margin = max(self.vehicle_config.length, self.vehicle_config.width) / 2
        if (new_x - margin < 0 or new_x + margin >= self.env.width or
            new_y - margin < 0 or new_y + margin >= self.env.height):
            return None
        
        return self._create_node(new_x, new_y, new_theta, 0)
    
    def _is_goal_reached(self, node: HybridNode, goal: Tuple) -> bool:
        """检查是否到达目标"""
        distance = math.sqrt((node.x - goal[0])**2 + (node.y - goal[1])**2)
        
        if distance > self.goal_tolerance:
            return False
        
        if len(goal) > 2:
            angle_diff = abs(self._normalize_angle(node.theta - goal[2]))
            if angle_diff > self.angle_tolerance:
                return False
        
        return True
    
    def _reconstruct_path(self, goal_node: HybridNode) -> List[Tuple]:
        """重建路径"""
        path = []
        current = goal_node
        
        while current:
            path.append((current.x, current.y, current.theta))
            current = current.parent
        
        path.reverse()
        return path
    
    def _get_backbone_guidance(self, start: Tuple, goal: Tuple) -> Dict:
        """获取骨干路径引导信息"""
        guidance = {}
        
        if self.backbone_network and hasattr(self.backbone_network, 'get_sampling_guidance_for_rrt'):
            guidance = self.backbone_network.get_sampling_guidance_for_rrt(start, goal)
        
        return guidance
    
    def _post_process_path(self, path: List[Tuple]) -> List[Tuple]:
        """路径后处理"""
        if not path or len(path) < 3:
            return path
        
        # 路径平滑
        smoothed_path = self._smooth_path(path)
        
        # 捷径优化
        optimized_path = self._shortcut_optimization(smoothed_path)
        
        return optimized_path
    
    def _smooth_path(self, path: List[Tuple], iterations: int = 3) -> List[Tuple]:
        """路径平滑"""
        if len(path) <= 2:
            return path
        
        smoothed = list(path)
        
        for _ in range(iterations):
            new_smoothed = [smoothed[0]]
            
            for i in range(1, len(smoothed) - 1):
                prev = smoothed[i-1]
                curr = smoothed[i]
                next_p = smoothed[i+1]
                
                # 位置平均
                smooth_x = (prev[0] + curr[0] + next_p[0]) / 3
                smooth_y = (prev[1] + curr[1] + next_p[1]) / 3
                
                # 角度平滑（考虑角度连续性）
                smooth_theta = self._smooth_angle(prev[2], curr[2], next_p[2])
                
                # 验证平滑后的点
                if self.collision_checker.is_collision_free(smooth_x, smooth_y, smooth_theta):
                    new_smoothed.append((smooth_x, smooth_y, smooth_theta))
                else:
                    new_smoothed.append(curr)
            
            new_smoothed.append(smoothed[-1])
            smoothed = new_smoothed
        
        return smoothed
    
    def _smooth_angle(self, angle1: float, angle2: float, angle3: float) -> float:
        """角度平滑处理"""
        # 处理角度不连续性
        angles = [angle1, angle2, angle3]
        
        # 标准化角度
        normalized = [self._normalize_angle(a) for a in angles]
        
        # 检查跳跃
        for i in range(1, len(normalized)):
            diff = normalized[i] - normalized[i-1]
            if diff > math.pi:
                normalized[i] -= 2 * math.pi
            elif diff < -math.pi:
                normalized[i] += 2 * math.pi
        
        # 平均
        avg_angle = sum(normalized) / len(normalized)
        return self._normalize_angle(avg_angle)
    
    def _shortcut_optimization(self, path: List[Tuple]) -> List[Tuple]:
        """捷径优化"""
        if len(path) <= 2:
            return path
        
        optimized = [path[0]]
        i = 0
        
        while i < len(path) - 1:
            # 寻找最远的可直达点
            j = len(path) - 1
            found_shortcut = False
            
            while j > i + 1:
                if self._is_straight_path_feasible(path[i], path[j]):
                    optimized.append(path[j])
                    i = j
                    found_shortcut = True
                    break
                j -= 1
            
            if not found_shortcut:
                optimized.append(path[i + 1])
                i += 1
        
        return optimized
    
    def _is_straight_path_feasible(self, start: Tuple, end: Tuple) -> bool:
        """检查直线路径是否可行"""
        # 简化检查：插值检查碰撞
        steps = int(math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2) / 2)
        steps = max(5, min(steps, 20))
        
        for i in range(steps + 1):
            t = i / steps
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + t * (end[1] - start[1])
            theta = start[2] + t * self._normalize_angle(end[2] - start[2])
            
            if not self.collision_checker.is_collision_free(x, y, theta):
                return False
        
        return True
    
    def _check_path_quality(self, path: List[Tuple], threshold: float) -> bool:
        """检查路径质量"""
        if not path or len(path) < 2:
            return False
        
        # 计算路径质量指标
        quality = self._calculate_path_quality(path)
        return quality >= threshold
    
    def _calculate_path_quality(self, path: List[Tuple]) -> float:
        """计算路径质量"""
        if not path or len(path) < 2:
            return 0
        
        # 长度效率
        path_length = self._calculate_path_length(path)
        direct_distance = math.sqrt(
            (path[-1][0] - path[0][0])**2 + (path[-1][1] - path[0][1])**2
        )
        
        length_efficiency = direct_distance / (path_length + 0.1)
        
        # 平滑度
        smoothness = self._calculate_smoothness(path)
        
        # 综合质量
        quality = 0.6 * length_efficiency + 0.4 * smoothness
        
        return min(1.0, quality)
    
    def _calculate_path_length(self, path: List[Tuple]) -> float:
        """计算路径长度"""
        if len(path) < 2:
            return 0
        
        length = 0
        for i in range(len(path) - 1):
            length += math.sqrt(
                (path[i+1][0] - path[i][0])**2 + (path[i+1][1] - path[i][1])**2
            )
        
        return length
    
    def _calculate_smoothness(self, path: List[Tuple]) -> float:
        """计算路径平滑度"""
        if len(path) < 3:
            return 1.0
        
        total_curvature = 0
        
        for i in range(1, len(path) - 1):
            curvature = self._calculate_curvature(path[i-1], path[i], path[i+1])
            total_curvature += curvature
        
        avg_curvature = total_curvature / (len(path) - 2)
        smoothness = math.exp(-avg_curvature * 3)
        
        return min(1.0, smoothness)
    
    def _calculate_curvature(self, p1: Tuple, p2: Tuple, p3: Tuple) -> float:
        """计算曲率"""
        # 使用角度变化计算曲率
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        len_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
        len_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if len_v1 < 1e-6 or len_v2 < 1e-6:
            return 0
        
        cos_angle = (v1[0]*v2[0] + v1[1]*v2[1]) / (len_v1 * len_v2)
        cos_angle = max(-1, min(1, cos_angle))
        
        angle_change = math.acos(cos_angle)
        avg_length = (len_v1 + len_v2) / 2
        
        return angle_change / (avg_length + 1e-6)
    
    def _improve_path_quality(self, path: List[Tuple], threshold: float) -> List[Tuple]:
        """改善路径质量"""
        # 额外的平滑处理
        improved_path = self._smooth_path(path, iterations=5)
        
        # 二次捷径优化
        improved_path = self._shortcut_optimization(improved_path)
        
        return improved_path
    
    def _normalize_angle(self, angle: float) -> float:
        """角度归一化"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def _validate_inputs(self, start: Tuple, goal: Tuple) -> bool:
        """验证输入参数"""
        if not start or not goal:
            return False
        
        if len(start) < 2 or len(goal) < 2:
            return False
        
        # 检查坐标范围
        for pos in [start, goal]:
            if (pos[0] < 0 or pos[0] >= self.env.width or
                pos[1] < 0 or pos[1] >= self.env.height):
                return False
        
        return True
    
    def _generate_cache_key(self, start: Tuple, goal: Tuple) -> str:
        """生成缓存键"""
        start_rounded = (round(start[0], 1), round(start[1], 1), 
                        round(start[2], 2) if len(start) > 2 else 0)
        goal_rounded = (round(goal[0], 1), round(goal[1], 1), 
                       round(goal[2], 2) if len(goal) > 2 else 0)
        
        return f"{start_rounded}:{goal_rounded}"
    
    def _cache_path(self, cache_key: str, path: List[Tuple]):
        """缓存路径"""
        if len(self.path_cache) >= self.cache_max_size:
            # 移除最旧的条目
            oldest_key = next(iter(self.path_cache))
            del self.path_cache[oldest_key]
        
        self.path_cache[cache_key] = path
    
    def _update_statistics(self, planning_time: float, iterations: int):
        """更新统计信息"""
        alpha = 0.1  # 学习率
        
        if self.statistics['average_time'] == 0:
            self.statistics['average_time'] = planning_time
        else:
            self.statistics['average_time'] = (
                (1 - alpha) * self.statistics['average_time'] + 
                alpha * planning_time
            )
        
        if self.statistics['average_iterations'] == 0:
            self.statistics['average_iterations'] = iterations
        else:
            self.statistics['average_iterations'] = (
                (1 - alpha) * self.statistics['average_iterations'] + 
                alpha * iterations
            )
    
    def _get_last_iteration_count(self) -> int:
        """获取最后一次搜索的迭代次数"""
        return getattr(self, '_last_iteration_count', 0)
    
    def get_statistics(self) -> Dict:
        """获取统计信息 - 兼容RRT接口"""
        stats = self.statistics.copy()
        
        # 添加成功率
        if stats['total_calls'] > 0:
            stats['success_rate'] = stats['successful_plans'] / stats['total_calls']
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_calls']
        else:
            stats['success_rate'] = 0
            stats['cache_hit_rate'] = 0
        
        # 添加骨干网络利用率
        if stats['successful_plans'] > 0:
            stats['backbone_utilization'] = stats['backbone_assisted'] / stats['successful_plans']
        else:
            stats['backbone_utilization'] = 0
        
        return stats
    
    def reset_statistics(self):
        """重置统计信息"""
        self.statistics = {
            'total_calls': 0,
            'successful_plans': 0,
            'average_time': 0,
            'average_iterations': 0,
            'cache_hits': 0,
            'backbone_assisted': 0
        }
        
        self.path_cache.clear()


# 向后兼容性 - 模拟RRT接口
class OptimizedRRTPlanner(HybridAStarPlanner):
    """RRT接口兼容包装器"""
    
    def __init__(self, env, vehicle_length=6.0, vehicle_width=3.0, 
                 turning_radius=8.0, step_size=1.0):
        super().__init__(env, vehicle_length, vehicle_width, turning_radius, step_size)
        print("使用混合A*规划器（RRT兼容模式）")


