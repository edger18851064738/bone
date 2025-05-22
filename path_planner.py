import math
import numpy as np
import time
from collections import defaultdict, OrderedDict
from RRT import RRTPlanner

class SimplifiedPathPlanner:
    """
    简化的路径规划器 - 按照用户设计理念重新实现
    
    设计理念：
    1. 优先检索骨干路径中有无前往该终点的路径
    2. 如果有，选择离起点最近的骨干路径
    3. 规划从起点到骨干路径起点的接入路径
    4. 拼接：接入路径 + 骨干路径 = 完整路径
    5. 如果没有骨干路径，直接点对点规划
    """
    
    def __init__(self, env, backbone_network=None, rrt_planner=None, traffic_manager=None):
        """初始化简化的路径规划器"""
        self.env = env
        self.backbone_network = backbone_network
        self.traffic_manager = traffic_manager
        
        # RRT规划器初始化
        if rrt_planner is None:
            try:
                self.rrt_planner = RRTPlanner(
                    env,
                    vehicle_length=6.0,
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
        
        # 智能缓存系统 - 保留原有的缓存机制
        self.cache_config = {
            'max_size': 500,
            'ttl': 300,  # 5分钟过期
            'quality_threshold': 0.6
        }
        self.route_cache = OrderedDict()  # LRU缓存
        self.cache_metadata = {}
        
        # 路径质量评估器
        self.quality_assessor = PathQualityAssessor(env)
        
        # 性能统计
        self.performance_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'backbone_usage': 0,
            'direct_planning': 0,
            'planning_times': [],
            'quality_scores': [],
            'backbone_success_rate': 0.0
        }
        
        # 规划配置
        self.planning_config = {
            'max_attempts': 3,
            'enable_post_smoothing': True,
            'enable_shortcut_optimization': True,
            'enable_quality_check': True
        }
        
        # 调试选项
        self.debug = False
        self.verbose_logging = False
        
        print("初始化简化的路径规划器")
    
    def set_backbone_network(self, backbone_network):
        """设置骨干路径网络"""
        self.backbone_network = backbone_network
        # 清空缓存
        self._clear_cache()
        print("已设置骨干路径网络")
    
    def set_traffic_manager(self, traffic_manager):
        """设置交通管理器"""
        self.traffic_manager = traffic_manager
    
    def plan_path(self, vehicle_id, start, goal, use_backbone=True, check_conflicts=True, 
                  strategy=None, max_attempts=None):
        """
        主要路径规划接口 - 简化版
        
        Args:
            vehicle_id: 车辆ID
            start: 起点坐标 (x, y, theta)
            goal: 终点坐标 (x, y, theta)
            use_backbone: 是否使用骨干网络（简化后总是尝试使用）
            check_conflicts: 是否检查冲突
            strategy: 兼容参数，不再使用
            max_attempts: 最大尝试次数
            
        Returns:
            tuple: (路径点列表, 路径结构信息) 或 (None, None)
        """
        start_time = time.time()
        self.performance_stats['total_requests'] += 1
        
        # 输入验证
        if not self._validate_inputs(start, goal):
            return None, None
        
        # 检查起点和终点是否相同
        if self._is_same_position(start, goal):
            return [start], {'type': 'direct', 'segments': 1}
        
        # 生成缓存键
        cache_key = self._generate_cache_key(vehicle_id, start, goal)
        
        # 检查缓存
        cached_result = self._check_cache(cache_key)
        if cached_result:
            self.performance_stats['cache_hits'] += 1
            return cached_result
        
        self.performance_stats['cache_misses'] += 1
        
        # 多次尝试规划
        max_attempts = max_attempts or self.planning_config['max_attempts']
        best_path = None
        best_structure = None
        best_quality = 0
        
        for attempt in range(max_attempts):
            try:
                # 核心规划逻辑
                path, structure = self._plan_path_core(vehicle_id, start, goal, attempt)
                
                if path:
                    # 验证路径
                    if self._validate_path(path):
                        # 评估质量
                        quality = self.quality_assessor.evaluate_path(path)
                        
                        if quality > best_quality:
                            best_path = path
                            best_structure = structure
                            best_quality = quality
                            
                            # 质量足够高，提前结束
                            if quality >= 0.9:
                                break
            
            except Exception as e:
                if self.debug:
                    print(f"规划尝试 {attempt + 1} 失败: {e}")
                continue
        
        # 后处理优化
        if best_path:
            best_path = self._post_process_path(best_path)
            if best_structure:
                best_structure['final_quality'] = self.quality_assessor.evaluate_path(best_path)
        
        # 冲突检查
        if best_path and check_conflicts and self.traffic_manager:
            if self.traffic_manager.check_path_conflicts(vehicle_id, best_path):
                # 尝试调整路径
                adjusted_path = self._resolve_path_conflicts(vehicle_id, start, goal, best_path)
                if adjusted_path:
                    best_path = adjusted_path
                    best_structure = self._analyze_path_structure(best_path)
        
        # 缓存高质量结果
        if best_path and best_quality >= self.cache_config['quality_threshold']:
            self._add_to_cache(cache_key, (best_path, best_structure), best_quality)
        
        # 更新统计信息
        planning_time = time.time() - start_time
        self.performance_stats['planning_times'].append(planning_time)
        if best_quality > 0:
            self.performance_stats['quality_scores'].append(best_quality)
        
        if self.verbose_logging:
            print(f"路径规划完成: 车辆{vehicle_id}, 质量={best_quality:.2f}, 耗时={planning_time:.3f}s")
        
        return best_path, best_structure
    
    def plan_path_with_backbone(self, vehicle_id, start, goal):
        """
        使用骨干网络规划路径的专用接口
        按照用户设计理念实现
        """
        return self._plan_path_core(vehicle_id, start, goal, 0)
    
    def _plan_path_core(self, vehicle_id, start, goal, attempt):
        """
        核心路径规划逻辑 - 按照用户设计理念
        
        流程：
        1. 识别目标点是否为特殊点
        2. 如果是，查找骨干路径
        3. 选择最近的骨干路径
        4. 规划接入路径
        5. 拼接完整路径
        6. 如果失败，回退到直接规划
        """
        if self.debug:
            print(f"开始规划路径: {start} -> {goal} (尝试 {attempt + 1})")
        
        # 1. 尝试使用骨干网络
        if self.backbone_network:
            backbone_result = self._try_backbone_planning(start, goal, attempt)
            if backbone_result:
                path, structure = backbone_result
                self.performance_stats['backbone_usage'] += 1
                if self.debug:
                    print(f"骨干网络规划成功，路径长度: {len(path)}")
                return path, structure
        
        # 2. 回退到直接规划
        if self.debug:
            print("回退到直接规划")
        
        direct_result = self._direct_planning(start, goal, attempt)
        if direct_result:
            self.performance_stats['direct_planning'] += 1
            return direct_result, {'type': 'direct', 'method': 'rrt'}
        
        return None, None
    
    def _try_backbone_planning(self, start, goal, attempt):
        """
        尝试使用骨干网络规划路径
        按照用户设计理念：查找 -> 选择 -> 接入 -> 拼接
        """
        try:
            # 1. 识别目标点类型
            target_type, target_id = self.backbone_network.identify_target_point(goal)
            
            if not target_type:
                if self.debug:
                    print("目标不是特殊点，无法使用骨干路径")
                return None
            
            if self.debug:
                print(f"目标识别为: {target_type}_{target_id}")
            
            # 2. 直接使用骨干网络的完整规划方法
            complete_path, structure = self.backbone_network.get_path_from_position_to_target(
                start, target_type, target_id
            )
            
            if complete_path and structure:
                if self.debug:
                    print(f"骨干网络规划成功: 总长度{len(complete_path)}, "
                          f"骨干利用率{structure.get('backbone_utilization', 0):.2f}")
                return complete_path, structure
            
            return None
            
        except Exception as e:
            if self.debug:
                print(f"骨干网络规划失败: {e}")
            return None
    
    def _direct_planning(self, start, goal, attempt):
        """
        直接点对点规划
        """
        if not self.rrt_planner:
            print("RRT规划器不可用")
            return None
        
        try:
            # 根据尝试次数调整参数
            max_iterations = 3000 + attempt * 1000
            
            if self.debug:
                print(f"直接规划: 最大迭代次数 {max_iterations}")
            
            path = self.rrt_planner.plan_path(start, goal, max_iterations=max_iterations)
            
            if path and len(path) >= 2:
                if self.debug:
                    print(f"直接规划成功，路径长度: {len(path)}")
                return path
            
            return None
            
        except Exception as e:
            if self.debug:
                print(f"直接规划失败: {e}")
            return None
    
    def _resolve_path_conflicts(self, vehicle_id, start, goal, current_path):
        """解决路径冲突"""
        if not self.traffic_manager:
            return current_path
        
        try:
            # 尝试获取交通管理器建议的路径调整
            adjusted_path = self.traffic_manager.suggest_path_adjustment(vehicle_id, start, goal)
            
            if adjusted_path:
                if self.debug:
                    print(f"交通管理器建议路径调整，新路径长度: {len(adjusted_path)}")
                return adjusted_path
            
            return current_path
            
        except Exception as e:
            if self.debug:
                print(f"冲突解决失败: {e}")
            return current_path
    
    def _post_process_path(self, path):
        """路径后处理优化"""
        if not path or len(path) < 3:
            return path
        
        optimized = path
        
        # 捷径优化
        if self.planning_config['enable_shortcut_optimization']:
            optimized = self._shortcut_optimization(optimized)
        
        # 平滑处理
        if self.planning_config['enable_post_smoothing']:
            optimized = self._smooth_path(optimized)
        
        return optimized
    
    def _shortcut_optimization(self, path):
        """捷径优化 - 移除不必要的中间点"""
        if len(path) < 3:
            return path
        
        optimized = [path[0]]
        i = 0
        
        while i < len(path) - 1:
            # 寻找最远的可直达点
            j = len(path) - 1
            found_shortcut = False
            
            while j > i + 1:
                if self._is_line_collision_free(path[i], path[j]):
                    optimized.append(path[j])
                    i = j
                    found_shortcut = True
                    break
                j -= 1
            
            if not found_shortcut:
                optimized.append(path[i + 1])
                i += 1
        
        return optimized
    
    def _smooth_path(self, path, iterations=2):
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
                
                # 简单的平均平滑
                x = (prev[0] + curr[0] + next_p[0]) / 3
                y = (prev[1] + curr[1] + next_p[1]) / 3
                theta = curr[2] if len(curr) > 2 else 0
                
                # 验证平滑后的点
                if self._is_valid_position(int(x), int(y)):
                    new_smoothed.append((x, y, theta))
                else:
                    new_smoothed.append(curr)
            
            new_smoothed.append(smoothed[-1])
            smoothed = new_smoothed
        
        return smoothed
    
    def _analyze_path_structure(self, path):
        """分析路径结构"""
        if not path or len(path) < 2:
            return {'type': 'empty'}
        
        return {
            'type': 'analyzed',
            'length': self._calculate_path_length(path),
            'segments': len(path) - 1,
            'complexity': self._calculate_path_complexity(path),
            'quality': self.quality_assessor.evaluate_path(path)
        }
    
    def _calculate_path_complexity(self, path):
        """计算路径复杂度"""
        if len(path) < 3:
            return 0
        
        total_turning = 0
        for i in range(1, len(path) - 1):
            angle = self._calculate_turning_angle(path[i-1], path[i], path[i+1])
            total_turning += angle
        
        # 复杂度 = 总转弯角度 / 路径长度
        length = self._calculate_path_length(path)
        return total_turning / max(1, length) if length > 0 else 0
    
    def _calculate_turning_angle(self, p1, p2, p3):
        """计算转弯角度"""
        v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        len_v1 = np.linalg.norm(v1)
        len_v2 = np.linalg.norm(v2)
        
        if len_v1 < 0.001 or len_v2 < 0.001:
            return 0
        
        cos_angle = np.dot(v1, v2) / (len_v1 * len_v2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        return math.acos(cos_angle)
    
    # 缓存管理方法
    def _generate_cache_key(self, vehicle_id, start, goal):
        """生成缓存键"""
        start_rounded = (round(start[0], 1), round(start[1], 1))
        goal_rounded = (round(goal[0], 1), round(goal[1], 1))
        
        return f"{vehicle_id}:{start_rounded}:{goal_rounded}"
    
    def _check_cache(self, cache_key):
        """检查缓存"""
        if cache_key in self.route_cache:
            # 检查是否过期
            metadata = self.cache_metadata.get(cache_key, {})
            current_time = time.time()
            
            if (current_time - metadata.get('timestamp', 0)) > self.cache_config['ttl']:
                # 过期，删除
                del self.route_cache[cache_key]
                del self.cache_metadata[cache_key]
                return None
            
            # 更新LRU
            self.route_cache.move_to_end(cache_key)
            
            # 更新命中计数
            metadata['hit_count'] = metadata.get('hit_count', 0) + 1
            
            return self.route_cache[cache_key]
        
        return None
    
    def _add_to_cache(self, cache_key, result, quality):
        """添加到缓存"""
        # 检查缓存大小
        if len(self.route_cache) >= self.cache_config['max_size']:
            # 删除最旧的项
            oldest_key = next(iter(self.route_cache))
            del self.route_cache[oldest_key]
            del self.cache_metadata[oldest_key]
        
        # 添加新项
        self.route_cache[cache_key] = result
        self.cache_metadata[cache_key] = {
            'timestamp': time.time(),
            'quality': quality,
            'hit_count': 0
        }
    
    def _clear_cache(self):
        """清空缓存"""
        self.route_cache.clear()
        self.cache_metadata.clear()
    
    # 工具方法
    def _validate_inputs(self, start, goal):
        """验证输入参数"""
        if not start or not goal:
            return False
        
        if len(start) < 2 or len(goal) < 2:
            return False
        
        # 检查坐标是否在地图范围内
        for pos in [start, goal]:
            x, y = pos[0], pos[1]
            if (x < 0 or x >= self.env.width or 
                y < 0 or y >= self.env.height):
                return False
        
        return True
    
    def _is_same_position(self, pos1, pos2, tolerance=0.1):
        """判断两个位置是否相同"""
        return self._calculate_distance(pos1, pos2) < tolerance
    
    def _calculate_distance(self, pos1, pos2):
        """计算两点间距离"""
        x1 = pos1[0] if len(pos1) > 0 else 0
        y1 = pos1[1] if len(pos1) > 1 else 0
        x2 = pos2[0] if len(pos2) > 0 else 0
        y2 = pos2[1] if len(pos2) > 1 else 0
        
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def _calculate_path_length(self, path):
        """计算路径总长度"""
        if not path or len(path) < 2:
            return 0
        
        length = 0
        for i in range(len(path) - 1):
            length += self._calculate_distance(path[i], path[i + 1])
        
        return length
    
    def _is_valid_position(self, x, y):
        """检查位置是否有效"""
        if not hasattr(self.env, 'grid'):
            return True
        
        if x < 0 or x >= self.env.width or y < 0 or y >= self.env.height:
            return False
        
        return self.env.grid[x, y] == 0
    
    def _is_line_collision_free(self, p1, p2):
        """检查直线是否无碰撞"""
        distance = self._calculate_distance(p1, p2)
        steps = max(10, int(distance))
        
        for i in range(steps + 1):
            t = i / max(1, steps)
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            
            if not self._is_valid_position(int(x), int(y)):
                return False
        
        return True
    
    def _validate_path(self, path):
        """验证路径有效性"""
        if not path or len(path) < 2:
            return False
        
        # 基本碰撞检测
        for i in range(len(path) - 1):
            if not self._is_line_collision_free(path[i], path[i+1]):
                return False
        
        return True
    
    def get_performance_stats(self):
        """获取性能统计信息"""
        stats = self.performance_stats.copy()
        
        # 计算平均值
        if stats['planning_times']:
            stats['avg_planning_time'] = sum(stats['planning_times']) / len(stats['planning_times'])
            stats['max_planning_time'] = max(stats['planning_times'])
            stats['min_planning_time'] = min(stats['planning_times'])
        
        if stats['quality_scores']:
            stats['avg_quality_score'] = sum(stats['quality_scores']) / len(stats['quality_scores'])
            stats['max_quality_score'] = max(stats['quality_scores'])
            stats['min_quality_score'] = min(stats['quality_scores'])
        
        # 缓存统计
        total_requests = stats['cache_hits'] + stats['cache_misses']
        if total_requests > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / total_requests
        
        # 骨干网络使用率
        total_planning = stats['backbone_usage'] + stats['direct_planning']
        if total_planning > 0:
            stats['backbone_success_rate'] = stats['backbone_usage'] / total_planning
        
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.performance_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'backbone_usage': 0,
            'direct_planning': 0,
            'planning_times': [],
            'quality_scores': [],
            'backbone_success_rate': 0.0
        }
    
    def set_debug(self, enable):
        """设置调试模式"""
        self.debug = enable
        self.verbose_logging = enable
        print(f"调试模式: {'开启' if enable else '关闭'}")


class PathQualityAssessor:
    """路径质量评估器 - 简化版"""
    
    def __init__(self, env):
        self.env = env
        
        # 质量权重
        self.weights = {
            'length_efficiency': 0.3,
            'smoothness': 0.3,
            'safety': 0.2,
            'complexity': 0.2
        }
    
    def evaluate_path(self, path):
        """综合评估路径质量"""
        if not path or len(path) < 2:
            return 0
        
        scores = {}
        
        # 长度效率
        scores['length_efficiency'] = self._evaluate_length_efficiency(path)
        
        # 平滑度
        scores['smoothness'] = self._evaluate_smoothness(path)
        
        # 安全性
        scores['safety'] = self._evaluate_safety(path)
        
        # 复杂度
        scores['complexity'] = self._evaluate_complexity(path)
        
        # 加权总分
        total_score = sum(
            scores[metric] * self.weights[metric] 
            for metric in scores
        )
        
        return min(1.0, max(0.0, total_score))
    
    def _evaluate_length_efficiency(self, path):
        """评估长度效率"""
        actual_length = self._calculate_path_length(path)
        direct_distance = math.sqrt(
            (path[-1][0] - path[0][0])**2 + 
            (path[-1][1] - path[0][1])**2
        )
        
        if direct_distance < 0.1:
            return 1.0
        
        efficiency = direct_distance / (actual_length + 0.1)
        return min(1.0, efficiency)
    
    def _evaluate_smoothness(self, path):
        """评估路径平滑度"""
        if len(path) < 3:
            return 1.0
        
        total_curvature = 0
        segments = 0
        
        for i in range(1, len(path) - 1):
            curvature = self._calculate_curvature(path[i-1], path[i], path[i+1])
            total_curvature += curvature
            segments += 1
        
        if segments == 0:
            return 1.0
        
        avg_curvature = total_curvature / segments
        smoothness = math.exp(-avg_curvature * 2)
        
        return min(1.0, smoothness)
    
    def _evaluate_safety(self, path):
        """评估路径安全性"""
        # 简化实现：基于路径点的安全间隙
        min_safety = 1.0
        
        for point in path[::max(1, len(path)//10)]:  # 采样检查
            safety = self._calculate_point_safety(point)
            min_safety = min(min_safety, safety)
        
        return min_safety
    
    def _evaluate_complexity(self, path):
        """评估路径复杂度（越简单越好）"""
        if len(path) < 3:
            return 1.0
        
        sharp_turns = 0
        total_segments = len(path) - 2
        
        for i in range(1, len(path) - 1):
            angle = self._calculate_turning_angle(path[i-1], path[i], path[i+1])
            if angle > math.pi / 6:  # 30度以上为急转弯
                sharp_turns += 1
        
        complexity_score = 1.0 - (sharp_turns / max(1, total_segments))
        return max(0, complexity_score)
    
    def _calculate_curvature(self, p1, p2, p3):
        """计算曲率"""
        v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        len_v1 = np.linalg.norm(v1)
        len_v2 = np.linalg.norm(v2)
        
        if len_v1 < 0.001 or len_v2 < 0.001:
            return 0
        
        v1_norm = v1 / len_v1
        v2_norm = v2 / len_v2
        
        dot_product = np.dot(v1_norm, v2_norm)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        angle_change = math.acos(dot_product)
        avg_segment_length = (len_v1 + len_v2) / 2
        
        return angle_change / (avg_segment_length + 0.001)
    
    def _calculate_turning_angle(self, p1, p2, p3):
        """计算转弯角度"""
        v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        len_v1 = np.linalg.norm(v1)
        len_v2 = np.linalg.norm(v2)
        
        if len_v1 < 0.001 or len_v2 < 0.001:
            return 0
        
        cos_angle = np.dot(v1, v2) / (len_v1 * len_v2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        return math.acos(cos_angle)
    
    def _calculate_point_safety(self, point):
        """计算点的安全性"""
        # 基于与障碍物的距离
        clearance = self._calculate_clearance(point)
        
        if clearance >= 5:
            return 1.0
        elif clearance >= 2:
            return 0.8
        elif clearance >= 1:
            return 0.5
        else:
            return 0.2
    
    def _calculate_clearance(self, point):
        """计算到最近障碍物的距离"""
        min_distance = float('inf')
        
        x, y = int(point[0]), int(point[1])
        search_radius = 10
        
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                check_x, check_y = x + dx, y + dy
                
                if (0 <= check_x < self.env.width and 
                    0 <= check_y < self.env.height and 
                    hasattr(self.env, 'grid') and
                    self.env.grid[check_x, check_y] == 1):
                    
                    distance = math.sqrt(dx*dx + dy*dy)
                    min_distance = min(min_distance, distance)
        
        return min_distance if min_distance != float('inf') else 10
    
    def _calculate_path_length(self, path):
        """计算路径总长度"""
        if not path or len(path) < 2:
            return 0
        
        length = 0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            length += math.sqrt(dx*dx + dy*dy)
        
        return length


# 保持向后兼容性
OptimizedPathPlanner = SimplifiedPathPlanner
PathPlanner = SimplifiedPathPlanner