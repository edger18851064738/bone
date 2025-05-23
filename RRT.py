"""
RRT.py - 优化版本的双向RRT路径规划器
包含性能优化、算法改进和更好的骨干网络集成
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict, deque
import matplotlib
import time
import heapq
from dataclasses import dataclass

# 设置中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

@dataclass
class RRTNode:
    """RRT节点 - 使用dataclass优化内存"""
    x: float
    y: float
    theta: float
    parent: Optional['RRTNode'] = None
    cost: float = 0.0
    from_start: bool = True
    
    def distance_to(self, other: 'RRTNode') -> float:
        """计算到另一个节点的距离"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def angle_diff(self, other: 'RRTNode') -> float:
        """计算角度差"""
        diff = abs(self.theta - other.theta)
        return min(diff, 2 * math.pi - diff)

@dataclass
class PathSegment:
    """路径段 - 用于线段检测优化"""
    start_idx: int
    end_idx: int
    length: float
    quality: float
    is_backbone: bool = False

class OptimizedRRTPlanner:
    """优化版本的RRT路径规划器"""
    
    def __init__(self, env, vehicle_length=5.0, vehicle_width=2.0, 
                 turning_radius=5.0, step_size=0.8, grid_resolution=0.3):
        """
        初始化优化的RRT规划器
        """
        self.env = env
        self.vehicle_length = vehicle_length
        self.vehicle_width = vehicle_width
        self.turning_radius = turning_radius
        self.step_size = step_size
        self.grid_resolution = grid_resolution
        
        # 性能优化参数
        self.max_nodes = 5000  # 限制节点数量防止内存爆炸
        self.early_termination_distance = 3.0  # 早期终止距离
        self.path_cache_size = 100  # 路径缓存大小
        
        # 骨干网络集成
        self.backbone_network = None
        self.backbone_bias = 0.3  # 骨干网络采样偏置
        self.interface_regions = {}  # 接口区域缓存
        
        # 车辆模型和碰撞检查器
        self._init_vehicle_components()
        
        # 双向RRT规划器
        self.bidirectional_rrt = OptimizedBidirectionalRRT(
            collision_checker=self.collision_checker,
            vehicle_model=self.vehicle_model,
            env=env,
            step_size=step_size * 1.2,
            max_steer=math.pi/4,
            goal_bias=0.25,
            vehicle_length=vehicle_length,
            vehicle_width=vehicle_width,
            debug=False
        )
        
        # RS曲线优化器
        try:
            from path_utils import EnhancedReedSheppCurves
            self.rs_curves = EnhancedReedSheppCurves(turning_radius)
        except ImportError:
            self.rs_curves = None
        
        # 路径缓存和统计
        self.path_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 优化的路径评估参数
        self.evaluation_weights = {
            'length': 0.30,
            'smoothness': 0.25,
            'safety': 0.20,
            'backbone_alignment': 0.15,
            'complexity': 0.10
        }
        
        # 调试选项
        self.debug = False
        self.enable_visualization = False
        
        print("初始化优化版RRT规划器")
    
    def _init_vehicle_components(self):
        """初始化车辆模型和碰撞检查器"""
        try:
            from vehicle_model import BicycleModel, CollisionChecker
            
            vehicle_params = {
                'length': self.vehicle_length,
                'width': self.vehicle_width,
                'wheel_base': self.vehicle_length * 0.6,
                'turning_radius': self.turning_radius,
                'step_size': self.step_size
            }
            self.vehicle_model = BicycleModel(vehicle_params)
            self.collision_checker = CollisionChecker(self.env, vehicle_params)
        except ImportError:
            print("警告: 使用简化碰撞检测")
            self.vehicle_model = None
            self.collision_checker = None
    
    def set_backbone_network(self, backbone_network):
        """设置骨干网络并预处理接口区域"""
        self.backbone_network = backbone_network
        self.bidirectional_rrt.set_backbone_network(backbone_network)
        self._preprocess_interface_regions()
        print("已设置骨干网络，并预处理接口区域")
    
    def _preprocess_interface_regions(self):
        """预处理骨干网络接口区域，用于引导采样"""
        if not self.backbone_network or not hasattr(self.backbone_network, 'backbone_interfaces'):
            return
        
        self.interface_regions.clear()
        
        for interface_id, interface in self.backbone_network.backbone_interfaces.items():
            pos = interface.position
            # 创建接口周围的引导区域
            region = {
                'center': (pos[0], pos[1]),
                'radius': 15.0,
                'weight': 1.0 + interface.usage_count * 0.1,  # 使用频率越高权重越大
                'accessibility': self._evaluate_interface_accessibility(interface)
            }
            self.interface_regions[interface_id] = region
    
    def _evaluate_interface_accessibility(self, interface):
        """评估接口的可达性"""
        try:
            # 检查接口周围的障碍物密度
            x, y = int(interface.position[0]), int(interface.position[1])
            obstacle_count = 0
            total_cells = 0
            
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    check_x, check_y = x + dx, y + dy
                    if (0 <= check_x < self.env.width and 
                        0 <= check_y < self.env.height):
                        total_cells += 1
                        if hasattr(self.env, 'grid') and self.env.grid[check_x, check_y] == 1:
                            obstacle_count += 1
            
            return 1.0 - (obstacle_count / max(1, total_cells))
        except:
            return 0.5
    
    def plan_path(self, start, goal, agent_id=None, max_iterations=4000, 
                  use_cache=True, quality_threshold=0.6):
        """
        优化的路径规划主接口
        
        Args:
            start: 起点 (x, y, theta)
            goal: 终点 (x, y, theta)
            agent_id: 代理ID
            max_iterations: 最大迭代次数
            use_cache: 是否使用路径缓存
            quality_threshold: 质量阈值
            
        Returns:
            list: 优化后的路径或None
        """
        start_time = time.time()
        
        # 输入验证
        if not self._validate_inputs(start, goal):
            return None
        
        # 检查缓存
        if use_cache:
            cached_path = self._check_cache(start, goal)
            if cached_path:
                self.cache_hits += 1
                return cached_path
            self.cache_misses += 1
        
        # 如果起终点很近，直接连接
        if self._distance(start, goal) < self.early_termination_distance:
            if self._is_direct_connection_valid(start, goal):
                return [start, goal]
        
        # 使用优化的双向RRT规划
        best_path = None
        best_quality = 0
        
        # 智能重试策略：根据问题复杂度调整尝试次数
        max_attempts = self._determine_max_attempts(start, goal)
        
        for attempt in range(max_attempts):
            try:
                # 调整参数以提高成功率
                adjusted_iterations = max_iterations + attempt * 500
                
                # 规划路径
                raw_path = self.bidirectional_rrt.plan(
                    start=start,
                    goal=goal,
                    time_step=1.0,
                    max_iterations=adjusted_iterations
                )
                
                if raw_path:
                    # 转换为简单路径格式
                    simple_path = [(p[0], p[1], p[2]) for p in raw_path]
                    
                    # 应用优化pipeline
                    optimized_path = self._optimization_pipeline(simple_path, start, goal)
                    
                    if optimized_path:
                        # 评估路径质量
                        quality = self._evaluate_path_quality(optimized_path, start, goal)
                        
                        if quality > best_quality:
                            best_quality = quality
                            best_path = optimized_path
                            
                            # 如果质量足够好，提前结束
                            if quality >= quality_threshold:
                                if self.debug:
                                    print(f"达到质量阈值 {quality:.2f}，提前结束")
                                break
            
            except Exception as e:
                if self.debug:
                    print(f"规划尝试 {attempt + 1} 失败: {e}")
                continue
        
        # 缓存高质量路径
        if best_path and best_quality >= 0.5 and use_cache:
            self._add_to_cache(start, goal, best_path)
        
        # 统计信息
        planning_time = time.time() - start_time
        if self.debug:
            print(f"路径规划完成: 质量={best_quality:.2f}, 耗时={planning_time:.3f}s")
        
        return best_path
    
    def _optimization_pipeline(self, path, start, goal):
        """优化pipeline，按顺序应用各种优化"""
        if not path or len(path) < 2:
            return path
        
        try:
            # 1. 线段检测和简化
            simplified = self._intelligent_line_detection(path)
            
            # 2. 骨干路径对齐（如果可用）
            if self.backbone_network:
                aligned = self._align_with_backbone(simplified)
            else:
                aligned = simplified
            
            # 3. 平滑处理
            smoothed = self._adaptive_smoothing(aligned)
            
            # 4. RS曲线优化关键转弯
            if self.rs_curves:
                final_path = self._optimize_critical_turns(smoothed)
            else:
                final_path = smoothed
            
            # 5. 密度调整
            return self._adjust_path_density(final_path)
        
        except Exception as e:
            if self.debug:
                print(f"优化pipeline失败: {e}")
            return path
    
    def _intelligent_line_detection(self, path):
        """智能线段检测 - 改进版"""
        if len(path) < 3:
            return path
        
        segments = []
        current_segment = [0]  # 当前线段的索引
        
        for i in range(1, len(path) - 1):
            # 检查是否可以继续当前线段
            start_idx = current_segment[0]
            
            if self._can_extend_segment(path, start_idx, i + 1):
                # 可以扩展，继续
                continue
            else:
                # 不能扩展，结束当前线段
                current_segment.append(i)
                if len(current_segment) >= 2:
                    segments.append(PathSegment(
                        start_idx=current_segment[0],
                        end_idx=current_segment[-1],
                        length=self._calculate_segment_length(path, current_segment[0], current_segment[-1]),
                        quality=self._evaluate_segment_quality(path, current_segment[0], current_segment[-1])
                    ))
                current_segment = [i]
        
        # 处理最后一个线段
        current_segment.append(len(path) - 1)
        if len(current_segment) >= 2:
            segments.append(PathSegment(
                start_idx=current_segment[0],
                end_idx=current_segment[-1],
                length=self._calculate_segment_length(path, current_segment[0], current_segment[-1]),
                quality=self._evaluate_segment_quality(path, current_segment[0], current_segment[-1])
            ))
        
        # 构建简化路径
        simplified_path = [path[0]]
        for segment in segments:
            if segment.end_idx > segment.start_idx:
                simplified_path.append(path[segment.end_idx])
        
        return simplified_path
    
    def _can_extend_segment(self, path, start_idx, end_idx, max_error=1.5):
        """检查是否可以扩展线段"""
        if end_idx - start_idx < 2:
            return True
        
        # 计算直线
        start_point = path[start_idx]
        end_point = path[end_idx]
        
        # 检查中间点到直线的最大偏差
        max_deviation = 0
        for i in range(start_idx + 1, end_idx):
            deviation = self._point_to_line_distance(path[i], start_point, end_point)
            max_deviation = max(max_deviation, deviation)
        
        return max_deviation <= max_error
    
    def _point_to_line_distance(self, point, line_start, line_end):
        """计算点到直线的距离"""
        x0, y0 = point[0], point[1]
        x1, y1 = line_start[0], line_start[1]
        x2, y2 = line_end[0], line_end[1]
        
        # 避免除零
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 0.001 and abs(dy) < 0.001:
            return math.sqrt((x0 - x1)**2 + (y0 - y1)**2)
        
        # 点到直线距离公式
        distance = abs((dy) * x0 - (dx) * y0 + x2 * y1 - y2 * x1) / math.sqrt(dx*dx + dy*dy)
        return distance
    
    def _align_with_backbone(self, path):
        """与骨干路径对齐"""
        if not self.backbone_network or not path:
            return path
        
        aligned_path = []
        
        for i, point in enumerate(path):
            # 检查是否接近骨干路径
            closest_backbone_point = self._find_closest_backbone_point(point)
            
            if closest_backbone_point:
                distance = self._distance(point, closest_backbone_point)
                
                # 如果很接近骨干路径，对齐到骨干路径
                if distance < 5.0:
                    aligned_point = self._blend_points(point, closest_backbone_point, 0.7)
                    aligned_path.append(aligned_point)
                else:
                    aligned_path.append(point)
            else:
                aligned_path.append(point)
        
        return aligned_path
    
    def _find_closest_backbone_point(self, point):
        """找到最近的骨干路径点"""
        if not self.backbone_network or not hasattr(self.backbone_network, 'backbone_paths'):
            return None
        
        min_distance = float('inf')
        closest_point = None
        
        for path_data in self.backbone_network.backbone_paths.values():
            backbone_path = path_data.get('path', [])
            
            for bp in backbone_path[::3]:  # 每隔3个点检查一次以提高效率
                distance = self._distance(point, bp)
                if distance < min_distance:
                    min_distance = distance
                    closest_point = bp
        
        return closest_point if min_distance < 10.0 else None
    
    def _blend_points(self, p1, p2, weight):
        """混合两个点"""
        x = p1[0] * (1 - weight) + p2[0] * weight
        y = p1[1] * (1 - weight) + p2[1] * weight
        theta = p1[2] if len(p1) > 2 else 0
        return (x, y, theta)
    
    def _adaptive_smoothing(self, path, iterations=2):
        """自适应平滑处理"""
        if len(path) < 3:
            return path
        
        smoothed = list(path)
        
        for iteration in range(iterations):
            new_smoothed = [smoothed[0]]
            
            for i in range(1, len(smoothed) - 1):
                # 计算局部曲率
                curvature = self._calculate_local_curvature(smoothed, i)
                
                # 根据曲率调整平滑强度
                if curvature > 0.5:  # 高曲率区域，更多平滑
                    smoothing_factor = 0.4
                else:  # 低曲率区域，保持形状
                    smoothing_factor = 0.2
                
                # 加权平均
                prev_point = smoothed[i-1]
                curr_point = smoothed[i]
                next_point = smoothed[i+1]
                
                x = curr_point[0] * (1 - smoothing_factor) + \
                    (prev_point[0] + next_point[0]) * smoothing_factor / 2
                y = curr_point[1] * (1 - smoothing_factor) + \
                    (prev_point[1] + next_point[1]) * smoothing_factor / 2
                theta = curr_point[2] if len(curr_point) > 2 else 0
                
                # 验证平滑后的点
                if self._is_valid_position(int(x), int(y)):
                    new_smoothed.append((x, y, theta))
                else:
                    new_smoothed.append(curr_point)
            
            new_smoothed.append(smoothed[-1])
            smoothed = new_smoothed
        
        return smoothed
    
    def _calculate_local_curvature(self, path, index):
        """计算局部曲率"""
        if index <= 0 or index >= len(path) - 1:
            return 0
        
        p1 = path[index - 1]
        p2 = path[index]
        p3 = path[index + 1]
        
        # 使用三点法计算曲率
        v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        len_v1 = np.linalg.norm(v1)
        len_v2 = np.linalg.norm(v2)
        
        if len_v1 < 0.001 or len_v2 < 0.001:
            return 0
        
        cos_angle = np.dot(v1, v2) / (len_v1 * len_v2)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = math.acos(cos_angle)
        
        avg_length = (len_v1 + len_v2) / 2
        return angle / (avg_length + 0.001)
    
    def _optimize_critical_turns(self, path):
        """优化关键转弯点"""
        if not self.rs_curves or len(path) < 5:
            return path
        
        # 识别关键转弯点
        turn_points = self._identify_critical_turns(path)
        
        optimized = list(path)
        
        for turn_idx in reversed(turn_points):  # 从后往前处理避免索引问题
            # 获取转弯段
            start_idx = max(0, turn_idx - 2)
            end_idx = min(len(path) - 1, turn_idx + 2)
            
            if end_idx - start_idx >= 3:
                turn_segment = path[start_idx:end_idx + 1]
                
                # 使用RS曲线优化
                optimized_segment = self._apply_rs_optimization(turn_segment)
                
                if optimized_segment and len(optimized_segment) > 2:
                    # 替换原始段
                    optimized[start_idx:end_idx + 1] = optimized_segment
        
        return optimized
    
    def _identify_critical_turns(self, path, angle_threshold=math.pi/6):
        """识别关键转弯点"""
        turn_points = []
        
        for i in range(1, len(path) - 1):
            angle = self._calculate_turning_angle(path[i-1], path[i], path[i+1])
            if angle > angle_threshold:
                turn_points.append(i)
        
        return turn_points
    
    def _calculate_turning_angle(self, p1, p2, p3):
        """计算转弯角度"""
        v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        len_v1 = np.linalg.norm(v1)
        len_v2 = np.linalg.norm(v2)
        
        if len_v1 < 0.001 or len_v2 < 0.001:
            return 0
        
        cos_angle = np.dot(v1, v2) / (len_v1 * len_v2)
        cos_angle = np.clip(cos_angle, -1, 1)
        return math.acos(cos_angle)
    
    def _apply_rs_optimization(self, segment):
        """应用RS曲线优化"""
        if not segment or len(segment) < 3:
            return segment
        
        try:
            start_config = segment[0]
            end_config = segment[-1]
            
            # 使用RS曲线生成平滑连接
            rs_path = self.rs_curves.get_path(start_config, end_config)
            
            if rs_path and self._validate_rs_path(rs_path):
                return rs_path
        except:
            pass
        
        return segment
    
    def _validate_rs_path(self, rs_path):
        """验证RS路径的有效性"""
        if not rs_path:
            return False
        
        # 简单的碰撞检测
        for point in rs_path[::2]:  # 每隔一个点检查
            if not self._is_valid_position(int(point[0]), int(point[1])):
                return False
        
        return True
    
    def _adjust_path_density(self, path, target_spacing=2.0):
        """调整路径点密度"""
        if not path or len(path) < 2:
            return path
        
        dense_path = [path[0]]
        
        for i in range(len(path) - 1):
            current = path[i]
            next_point = path[i + 1]
            
            distance = self._distance(current, next_point)
            
            if distance > target_spacing:
                # 需要插入中间点
                num_inserts = int(distance / target_spacing)
                
                for j in range(1, num_inserts + 1):
                    t = j / (num_inserts + 1)
                    
                    x = current[0] + t * (next_point[0] - current[0])
                    y = current[1] + t * (next_point[1] - current[1])
                    theta = current[2] if len(current) > 2 else 0
                    
                    dense_path.append((x, y, theta))
            
            dense_path.append(next_point)
        
        return dense_path
    
    def _evaluate_path_quality(self, path, start, goal):
        """评估路径质量 - 优化版"""
        if not path or len(path) < 2:
            return 0
        
        try:
            # 1. 长度效率
            path_length = sum(self._distance(path[i], path[i+1]) for i in range(len(path)-1))
            direct_distance = self._distance(start, goal)
            length_score = direct_distance / (path_length + 0.1) if path_length > 0 else 0
            
            # 2. 平滑度
            smoothness_score = self._calculate_smoothness_score(path)
            
            # 3. 安全性（基于与障碍物的距离）
            safety_score = self._calculate_safety_score(path)
            
            # 4. 骨干路径对齐度
            backbone_score = self._calculate_backbone_alignment_score(path)
            
            # 5. 复杂度（转弯次数）
            complexity_score = self._calculate_complexity_score(path)
            
            # 加权总分
            total_score = (
                self.evaluation_weights['length'] * length_score +
                self.evaluation_weights['smoothness'] * smoothness_score +
                self.evaluation_weights['safety'] * safety_score +
                self.evaluation_weights['backbone_alignment'] * backbone_score +
                self.evaluation_weights['complexity'] * complexity_score
            )
            
            return min(1.0, max(0.0, total_score))
        
        except Exception as e:
            if self.debug:
                print(f"路径质量评估失败: {e}")
            return 0.5
    
    def _calculate_smoothness_score(self, path):
        """计算平滑度分数"""
        if len(path) < 3:
            return 1.0
        
        total_curvature = 0
        valid_points = 0
        
        for i in range(1, len(path) - 1):
            curvature = self._calculate_local_curvature(path, i)
            total_curvature += curvature
            valid_points += 1
        
        if valid_points == 0:
            return 1.0
        
        avg_curvature = total_curvature / valid_points
        return math.exp(-avg_curvature * 3)  # 曲率越小分数越高
    
    def _calculate_safety_score(self, path):
        """计算安全性分数"""
        if not hasattr(self.env, 'grid'):
            return 1.0
        
        min_clearance = float('inf')
        
        # 采样检查路径上的点
        sample_points = path[::max(1, len(path) // 10)]
        
        for point in sample_points:
            clearance = self._calculate_clearance(point)
            min_clearance = min(min_clearance, clearance)
        
        # 转换为0-1分数
        if min_clearance >= 5:
            return 1.0
        elif min_clearance >= 2:
            return 0.8
        elif min_clearance >= 1:
            return 0.5
        else:
            return 0.2
    
    def _calculate_clearance(self, point):
        """计算到最近障碍物的距离"""
        x, y = int(point[0]), int(point[1])
        
        for radius in range(1, 11):
            obstacle_found = False
            
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx*dx + dy*dy > radius*radius:
                        continue
                    
                    check_x, check_y = x + dx, y + dy
                    
                    if (0 <= check_x < self.env.width and 
                        0 <= check_y < self.env.height and
                        self.env.grid[check_x, check_y] == 1):
                        obstacle_found = True
                        break
                
                if obstacle_found:
                    break
            
            if obstacle_found:
                return radius
        
        return 10  # 最大检查半径
    
    def _calculate_backbone_alignment_score(self, path):
        """计算与骨干路径的对齐分数"""
        if not self.backbone_network:
            return 0.5  # 中性分数
        
        aligned_points = 0
        total_points = len(path)
        
        for point in path[::max(1, len(path) // 20)]:  # 采样检查
            if self._is_near_backbone(point):
                aligned_points += 1
        
        sample_size = len(path[::max(1, len(path) // 20)])
        alignment_ratio = aligned_points / max(1, sample_size)
        
        return alignment_ratio
    
    def _is_near_backbone(self, point, threshold=8.0):
        """检查点是否接近骨干路径"""
        closest_point = self._find_closest_backbone_point(point)
        if closest_point:
            return self._distance(point, closest_point) < threshold
        return False
    
    def _calculate_complexity_score(self, path):
        """计算复杂度分数（越简单分数越高）"""
        if len(path) < 3:
            return 1.0
        
        sharp_turns = 0
        total_segments = len(path) - 2
        
        for i in range(1, len(path) - 1):
            angle = self._calculate_turning_angle(path[i-1], path[i], path[i+1])
            if angle > math.pi / 4:  # 45度以上为急转弯
                sharp_turns += 1
        
        complexity_score = 1.0 - (sharp_turns / max(1, total_segments))
        return max(0, complexity_score)
    
    def _determine_max_attempts(self, start, goal):
        """根据问题复杂度确定最大尝试次数"""
        distance = self._distance(start, goal)
        
        if distance < 20:
            return 1  # 近距离，一次尝试
        elif distance < 50:
            return 2  # 中距离，两次尝试
        else:
            return 3  # 远距离，三次尝试
    
    def _check_cache(self, start, goal):
        """检查路径缓存"""
        cache_key = self._generate_cache_key(start, goal)
        return self.path_cache.get(cache_key)
    
    def _add_to_cache(self, start, goal, path):
        """添加路径到缓存"""
        if len(self.path_cache) >= self.path_cache_size:
            # 移除最旧的缓存项
            oldest_key = next(iter(self.path_cache))
            del self.path_cache[oldest_key]
        
        cache_key = self._generate_cache_key(start, goal)
        self.path_cache[cache_key] = path
    
    def _generate_cache_key(self, start, goal):
        """生成缓存键"""
        return f"{start[0]:.1f},{start[1]:.1f}_{goal[0]:.1f},{goal[1]:.1f}"
    
    def _validate_inputs(self, start, goal):
        """验证输入参数"""
        if not start or not goal or len(start) < 2 or len(goal) < 2:
            return False
        
        # 检查坐标范围
        for pos in [start, goal]:
            if (pos[0] < 0 or pos[0] >= self.env.width or
                pos[1] < 0 or pos[1] >= self.env.height):
                return False
        
        return True
    
    def _is_direct_connection_valid(self, start, goal):
        """检查是否可以直接连接"""
        steps = max(10, int(self._distance(start, goal)))
        
        for i in range(steps + 1):
            t = i / max(1, steps)
            x = start[0] + t * (goal[0] - start[0])
            y = start[1] + t * (goal[1] - start[1])
            
            if not self._is_valid_position(int(x), int(y)):
                return False
        
        return True
    
    def _is_valid_position(self, x, y):
        """检查位置是否有效"""
        if not hasattr(self.env, 'grid'):
            return True
        
        if x < 0 or x >= self.env.width or y < 0 or y >= self.env.height:
            return False
        
        return self.env.grid[x, y] == 0
    
    def _distance(self, p1, p2):
        """计算两点间距离"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _calculate_segment_length(self, path, start_idx, end_idx):
        """计算路径段长度"""
        length = 0
        for i in range(start_idx, end_idx):
            length += self._distance(path[i], path[i + 1])
        return length
    
    def _evaluate_segment_quality(self, path, start_idx, end_idx):
        """评估路径段质量"""
        # 简化的质量评估
        segment_length = self._calculate_segment_length(path, start_idx, end_idx)
        direct_length = self._distance(path[start_idx], path[end_idx])
        
        if direct_length < 0.1:
            return 1.0
        
        return direct_length / (segment_length + 0.1)
    
    def get_statistics(self):
        """获取规划器统计信息"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(1, total_requests)
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': hit_rate,
            'cached_paths': len(self.path_cache),
            'backbone_network_available': self.backbone_network is not None,
            'interface_regions': len(self.interface_regions)
        }
    
    def clear_cache(self):
        """清除缓存"""
        self.path_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    # 保持向后兼容的方法
    def is_path_possible(self, start, goal):
        """检查路径是否可能"""
        return self._is_direct_connection_valid(start, goal)
    
    def draw_vehicle(self, ax, position, size=None, color='blue', alpha=0.7):
        """绘制车辆（保持兼容性）"""
        if size is None:
            size = (self.vehicle_length, self.vehicle_width)
        
        x, y, theta = position
        half_length, half_width = size[0] / 2, size[1] / 2
        
        corners_rel = [
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width]
        ]
        
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        corners = [
            (x + cos_t * rx - sin_t * ry, y + sin_t * rx + cos_t * ry)
            for rx, ry in corners_rel
        ]
        
        polygon = plt.Polygon(corners, closed=True, fill=True,
                             color=color, alpha=alpha)
        ax.add_patch(polygon)
        
        # 方向指示
        ax.plot([x, x + half_length * cos_t], [y, y + half_length * sin_t], 
                color='black', linewidth=1)


class OptimizedBidirectionalRRT:
    """优化的双向RRT算法"""
    
    def __init__(self, collision_checker, vehicle_model, env=None, 
                 step_size=2.0, max_steer=math.pi/4, goal_bias=0.2, 
                 vehicle_length=5.0, vehicle_width=2.0, debug=False):
        
        self.collision_checker = collision_checker
        self.vehicle_model = vehicle_model
        self.env = env
        self.step_size = step_size
        self.max_steer = max_steer
        self.goal_bias = goal_bias
        self.vehicle_length = vehicle_length
        self.vehicle_width = vehicle_width
        self.debug = debug
        
        # 骨干网络支持
        self.backbone_network = None
        self.backbone_bias = 0.3
        
        # 性能优化参数
        self.max_nodes_per_tree = 2500
        self.connection_radius = 15.0
        self.early_termination_threshold = 2.0
        
        # 统计信息
        self.stats = {
            'iterations': 0,
            'nodes_generated': 0,
            'connection_attempts': 0,
            'early_terminations': 0
        }
    
    def set_backbone_network(self, backbone_network):
        """设置骨干网络"""
        self.backbone_network = backbone_network
    
    def plan(self, start, goal, time_step=1.0, max_iterations=4000):
        """
        优化的双向RRT规划
        """
        if self.debug:
            print(f"开始双向RRT规划: {start} -> {goal}")
        
        # 重置统计
        self.stats = {
            'iterations': 0,
            'nodes_generated': 0,
            'connection_attempts': 0,
            'early_terminations': 0
        }
        
        # 创建起点和终点树
        start_tree = [RRTNode(start[0], start[1], start[2], None, 0.0, True)]
        goal_tree = [RRTNode(goal[0], goal[1], goal[2], None, 0.0, False)]
        
        self.start_config = start
        self.goal_config = goal
        
        # 主规划循环
        connection = None
        iterations = 0
        
        while iterations < max_iterations and connection is None:
            iterations += 1
            self.stats['iterations'] = iterations
            
            # 动态调整采样策略
            dynamic_goal_bias = self._calculate_dynamic_bias(iterations, max_iterations)
            
            # 生成采样点
            rand_config = self._intelligent_sampling(dynamic_goal_bias)
            
            # 扩展起点树
            new_start_node = self._extend_tree(start_tree, rand_config, True)
            if new_start_node:
                # 尝试连接到终点树
                connection_node = self._attempt_connection(new_start_node, goal_tree)
                if connection_node:
                    connection = (new_start_node, connection_node)
                    break
            
            # 扩展终点树
            new_goal_node = self._extend_tree(goal_tree, rand_config, False)
            if new_goal_node:
                # 尝试连接到起点树
                connection_node = self._attempt_connection(new_goal_node, start_tree)
                if connection_node:
                    connection = (connection_node, new_goal_node)
                    break
            
            # 树大小控制
            if len(start_tree) > self.max_nodes_per_tree:
                self._prune_tree(start_tree)
            if len(goal_tree) > self.max_nodes_per_tree:
                self._prune_tree(goal_tree)
            
            # 周期性连接尝试
            if iterations % 100 == 0:
                direct_connection = self._try_direct_connection(start_tree, goal_tree)
                if direct_connection:
                    connection = direct_connection
                    break
            
            # 进度报告
            if self.debug and iterations % 500 == 0:
                print(f"  迭代 {iterations}: 起点树{len(start_tree)}个节点, "
                      f"终点树{len(goal_tree)}个节点")
        
        if connection:
            if self.debug:
                print(f"找到连接，总迭代: {iterations}")
            
            # 提取和优化路径
            path_nodes = self._extract_bidirectional_path(connection, start_tree, goal_tree)
            path_nodes = self._post_process_nodes(path_nodes)
            
            # 转换为时间-空间路径
            time_path = [(node.x, node.y, node.theta, i * time_step) 
                        for i, node in enumerate(path_nodes)]
            
            return time_path
        
        # 尝试返回最佳部分路径
        if self.debug:
            print(f"完整路径规划失败，尝试返回部分路径")
        
        return self._generate_best_partial_path(start_tree, goal_tree, time_step)
    
    def _calculate_dynamic_bias(self, current_iter, max_iter):
        """计算动态目标偏置"""
        progress = current_iter / max_iter
        # 开始时低偏置探索，后期高偏置收敛
        return self.goal_bias + 0.4 * progress
    
    def _intelligent_sampling(self, goal_bias):
        """智能采样策略"""
        rand = random.random()
        
        # 目标偏置采样
        if rand < goal_bias:
            return (self.goal_config[0], self.goal_config[1], self.goal_config[2])
        
        # 骨干网络引导采样
        if (self.backbone_network and rand < goal_bias + self.backbone_bias):
            return self._sample_near_backbone()
        
        # 漏斗形采样
        if rand < goal_bias + self.backbone_bias + 0.3:
            return self._funnel_sampling()
        
        # 全局随机采样
        return self._global_sampling()
    
    def _sample_near_backbone(self):
        """在骨干网络附近采样"""
        if (not self.backbone_network or 
            not hasattr(self.backbone_network, 'backbone_paths')):
            return self._global_sampling()
        
        # 随机选择一条骨干路径
        paths = list(self.backbone_network.backbone_paths.values())
        if not paths:
            return self._global_sampling()
        
        random_path = random.choice(paths)
        backbone_path = random_path.get('path', [])
        
        if not backbone_path:
            return self._global_sampling()
        
        # 在骨干路径上随机选择一点
        random_point = random.choice(backbone_path)
        
        # 在该点附近添加噪声
        noise_radius = 10.0
        x = random_point[0] + random.uniform(-noise_radius, noise_radius)
        y = random_point[1] + random.uniform(-noise_radius, noise_radius)
        theta = random.uniform(0, 2 * math.pi)
        
        return (x, y, theta)
    
    def _funnel_sampling(self):
        """漏斗形采样"""
        # 在起点到终点的连线附近采样
        ratio = random.random()
        center_x = self.start_config[0] + ratio * (self.goal_config[0] - self.start_config[0])
        center_y = self.start_config[1] + ratio * (self.goal_config[1] - self.start_config[1])
        
        # 扩散半径随距离增加
        max_spread = 30.0
        spread = max_spread * (1.0 - abs(ratio - 0.5) * 2)
        
        x = center_x + random.uniform(-spread, spread)
        y = center_y + random.uniform(-spread, spread)
        theta = random.uniform(0, 2 * math.pi)
        
        return (x, y, theta)
    
    def _global_sampling(self):
        """全局随机采样"""
        margin = 20
        x_min = max(0, min(self.start_config[0], self.goal_config[0]) - margin)
        x_max = min(self.env.width, max(self.start_config[0], self.goal_config[0]) + margin)
        y_min = max(0, min(self.start_config[1], self.goal_config[1]) - margin)
        y_max = min(self.env.height, max(self.start_config[1], self.goal_config[1]) + margin)
        
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        theta = random.uniform(0, 2 * math.pi)
        
        return (x, y, theta)
    
    def _extend_tree(self, tree, target_config, from_start):
        """扩展树"""
        # 找到最近节点
        nearest_node = min(tree, key=lambda n: self._node_distance(n, target_config))
        
        # 计算新节点位置
        new_x, new_y, new_theta = self._steer(nearest_node, target_config)
        
        # 验证新节点
        if self._is_state_valid(new_x, new_y, new_theta):
            if self._is_path_valid(nearest_node, (new_x, new_y, new_theta)):
                # 计算成本
                new_cost = nearest_node.cost + self._distance((nearest_node.x, nearest_node.y), 
                                                            (new_x, new_y))
                
                # 创建新节点
                new_node = RRTNode(new_x, new_y, new_theta, nearest_node, new_cost, from_start)
                tree.append(new_node)
                self.stats['nodes_generated'] += 1
                
                return new_node
        
        return None
    
    def _steer(self, from_node, to_config):
        """引导函数 - 从一个节点向目标配置移动"""
        dx = to_config[0] - from_node.x
        dy = to_config[1] - from_node.y
        distance = math.sqrt(dx * dx + dy * dy)
        
        if distance < 0.1:
            return from_node.x, from_node.y, from_node.theta
        
        # 限制步长
        if distance > self.step_size:
            ratio = self.step_size / distance
            new_x = from_node.x + dx * ratio
            new_y = from_node.y + dy * ratio
        else:
            new_x = to_config[0]
            new_y = to_config[1]
        
        # 计算新的朝向角
        target_theta = math.atan2(dy, dx)
        angle_diff = (target_theta - from_node.theta + math.pi) % (2 * math.pi) - math.pi
        
        # 限制转向角
        if abs(angle_diff) > self.max_steer:
            angle_diff = math.copysign(self.max_steer, angle_diff)
        
        new_theta = (from_node.theta + angle_diff) % (2 * math.pi)
        
        return new_x, new_y, new_theta
    
    def _attempt_connection(self, node, target_tree):
        """尝试连接到目标树"""
        self.stats['connection_attempts'] += 1
        
        # 在连接半径内查找候选节点
        candidates = []
        for target_node in target_tree:
            distance = self._node_distance(node, (target_node.x, target_node.y, target_node.theta))
            if distance < self.connection_radius:
                candidates.append((target_node, distance))
        
        # 按距离排序
        candidates.sort(key=lambda x: x[1])
        
        # 尝试连接最近的几个候选节点
        for target_node, distance in candidates[:5]:  # 最多尝试5个
            if distance < self.early_termination_threshold:
                # 距离很近，直接连接
                if self._is_path_valid(node, (target_node.x, target_node.y, target_node.theta)):
                    self.stats['early_terminations'] += 1
                    return target_node
            else:
                # 检查是否可以通过引导连接
                if self._can_connect_with_steering(node, target_node):
                    return target_node
        
        return None
    
    def _can_connect_with_steering(self, node1, node2):
        """检查是否可以通过引导连接两个节点"""
        max_steps = int(self._distance((node1.x, node1.y), (node2.x, node2.y)) / (self.step_size * 0.5))
        max_steps = min(max_steps, 10)  # 限制最大步数
        
        current_x, current_y, current_theta = node1.x, node1.y, node1.theta
        
        for _ in range(max_steps):
            # 计算下一步
            next_x, next_y, next_theta = self._steer(
                RRTNode(current_x, current_y, current_theta),
                (node2.x, node2.y, node2.theta)
            )
            
            # 检查有效性
            if not self._is_state_valid(next_x, next_y, next_theta):
                return False
            
            # 检查路径有效性
            if not self._is_path_valid(RRTNode(current_x, current_y, current_theta), 
                                     (next_x, next_y, next_theta)):
                return False
            
            # 更新当前位置
            current_x, current_y, current_theta = next_x, next_y, next_theta
            
            # 检查是否足够接近目标
            if self._distance((current_x, current_y), (node2.x, node2.y)) < 1.0:
                return True
        
        return False
    
    def _try_direct_connection(self, start_tree, goal_tree):
        """尝试直接连接两棵树"""
        min_distance = float('inf')
        best_pair = None
        
        # 采样检查以提高效率
        start_sample = random.sample(start_tree, min(len(start_tree), 50))
        goal_sample = random.sample(goal_tree, min(len(goal_tree), 50))
        
        for start_node in start_sample:
            for goal_node in goal_sample:
                distance = self._node_distance(start_node, (goal_node.x, goal_node.y, goal_node.theta))
                
                if distance < min_distance and distance < self.connection_radius:
                    if self._can_connect_with_steering(start_node, goal_node):
                        min_distance = distance
                        best_pair = (start_node, goal_node)
        
        return best_pair
    
    def _prune_tree(self, tree):
        """修剪树以控制大小"""
        if len(tree) <= self.max_nodes_per_tree:
            return
        
        # 按成本排序，保留成本较低的节点
        tree.sort(key=lambda n: n.cost + random.random() * 0.1)  # 添加随机性避免总是删除相同节点
        
        # 保留前80%的节点
        keep_count = int(self.max_nodes_per_tree * 0.8)
        del tree[keep_count:]
    
    def _extract_bidirectional_path(self, connection, start_tree, goal_tree):
        """提取双向路径"""
        start_node, goal_node = connection
        
        # 从连接点回溯到起点
        start_path = []
        current = start_node
        while current:
            start_path.append(current)
            current = current.parent
        start_path.reverse()
        
        # 从连接点回溯到终点
        goal_path = []
        current = goal_node
        while current:
            goal_path.append(current)
            current = current.parent
        
        # 确定正确的连接方向
        if start_node.from_start:
            return start_path + goal_path
        else:
            return goal_path + start_path
    
    def _post_process_nodes(self, nodes):
        """后处理节点路径"""
        if len(nodes) < 3:
            return nodes
        
        # 简单的路径平滑
        smoothed = [nodes[0]]
        
        i = 0
        while i < len(nodes) - 1:
            # 尝试跳过中间节点
            j = min(len(nodes) - 1, i + 5)
            
            while j > i + 1:
                if self._is_path_valid(nodes[i], (nodes[j].x, nodes[j].y, nodes[j].theta)):
                    smoothed.append(nodes[j])
                    i = j
                    break
                j -= 1
            else:
                smoothed.append(nodes[i + 1])
                i += 1
        
        return smoothed
    
    def _generate_best_partial_path(self, start_tree, goal_tree, time_step):
        """生成最佳部分路径"""
        # 找到最接近目标的起点树节点
        best_node = min(start_tree, 
                       key=lambda n: self._distance((n.x, n.y), (self.goal_config[0], self.goal_config[1])))
        
        best_distance = self._distance((best_node.x, best_node.y), 
                                     (self.goal_config[0], self.goal_config[1]))
        
        if best_distance < 50.0:  # 如果距离合理
            # 提取部分路径
            partial_path = []
            current = best_node
            while current:
                partial_path.append(current)
                current = current.parent
            partial_path.reverse()
            
            # 尝试添加到目标的直线段
            last_node = partial_path[-1]
            if self._is_path_valid(last_node, self.goal_config):
                partial_path.append(RRTNode(self.goal_config[0], self.goal_config[1], self.goal_config[2]))
            
            # 转换为时间路径
            return [(node.x, node.y, node.theta, i * time_step) 
                   for i, node in enumerate(partial_path)]
        
        return None
    
    def _is_state_valid(self, x, y, theta):
        """检查状态有效性"""
        if self.collision_checker:
            return self.collision_checker.is_state_valid(x, y, theta)
        elif self.env and hasattr(self.env, 'grid'):
            grid_x, grid_y = int(x), int(y)
            if 0 <= grid_x < self.env.width and 0 <= grid_y < self.env.height:
                return self.env.grid[grid_x, grid_y] == 0
            return False
        else:
            return True
    
    def _is_path_valid(self, from_node, to_config):
        """检查路径有效性"""
        # 简化的路径检查
        steps = max(3, int(self._distance((from_node.x, from_node.y), to_config) / 0.5))
        
        for i in range(1, steps):
            t = i / steps
            x = from_node.x + t * (to_config[0] - from_node.x)
            y = from_node.y + t * (to_config[1] - from_node.y)
            
            if not self._is_state_valid(x, y, 0):  # 简化角度检查
                return False
        
        return True
    
    def _node_distance(self, node, config):
        """计算节点到配置的距离"""
        dx = node.x - config[0]
        dy = node.y - config[1]
        d_pos = math.sqrt(dx*dx + dy*dy)
        
        # 角度差
        d_theta = abs((node.theta - config[2] + math.pi) % (2 * math.pi) - math.pi)
        
        return d_pos + 0.2 * self.vehicle_length * d_theta
    
    def _distance(self, p1, p2):
        """计算两点距离"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


# 保持向后兼容性
RRTPlanner = OptimizedRRTPlanner
BidirectionalRRT = OptimizedBidirectionalRRT


