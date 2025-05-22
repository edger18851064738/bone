import math
import numpy as np
import time
from collections import defaultdict, OrderedDict
from RRT import RRTPlanner

class OptimizedPathPlanner:
    """优化后的路径规划器，支持智能缓存、结构化路径规划和多策略路径选择"""
    
    def __init__(self, env, backbone_network=None, rrt_planner=None, traffic_manager=None):
        """初始化路径规划器"""
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
        
        # 智能缓存系统
        self.cache_config = {
            'max_size': 1000,
            'ttl': 300,  # 5分钟过期
            'lru_enabled': True,
            'quality_threshold': 0.6  # 只缓存高质量路径
        }
        self.route_cache = OrderedDict()  # LRU缓存
        self.cache_metadata = {}  # 缓存元数据 {key: {'timestamp', 'quality', 'hit_count'}}
        
        # 路径验证增强
        self.validation_config = {
            'enabled': True,
            'sample_density': 20,  # 每段采样点数
            'safety_margin': 1.5,  # 安全边距
            'multi_pass': True,    # 多次验证
            'dynamic_density': True  # 动态调整采样密度
        }
        
        # 路径质量评估
        self.quality_assessor = PathQualityAssessor(env)
        
        # 多策略路径规划
        self.planning_strategies = {
            'backbone_first': self._plan_backbone_first_strategy,
            'direct_optimized': self._plan_direct_optimized_strategy,
            'hybrid_multi_path': self._plan_hybrid_multi_path_strategy,
            'emergency_fallback': self._plan_emergency_fallback_strategy
        }
        self.default_strategy = 'backbone_first'
        
        # 性能统计
        self.performance_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'backbone_successes': 0,
            'direct_fallbacks': 0,
            'planning_times': [],
            'quality_scores': [],
            'strategy_usage': defaultdict(int)
        }
        
        # 路径优化参数
        self.optimization_config = {
            'post_smoothing': True,
            'shortcut_optimization': True,
            'quality_improvement': True,
            'max_optimization_time': 2.0
        }
        
        # 调试选项
        self.debug = False
        self.verbose_logging = False
    
    def set_backbone_network(self, backbone_network):
        """设置主干路径网络"""
        self.backbone_network = backbone_network
        # 清空缓存，因为骨干网络发生变化
        self._clear_cache()
        
    def set_traffic_manager(self, traffic_manager):
        """设置交通管理器"""
        self.traffic_manager = traffic_manager
    
    def plan_path(self, vehicle_id, start, goal, use_backbone=True, check_conflicts=True, 
                  strategy=None, max_attempts=3):
        """
        规划从起点到终点的完整路径 - 增强版
        
        Args:
            vehicle_id: 车辆ID
            start: 起点坐标 (x, y, theta)
            goal: 终点坐标 (x, y, theta)
            use_backbone: 是否使用主干网络
            check_conflicts: 是否检查路径冲突
            strategy: 规划策略（可选）
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
        cache_key = self._generate_cache_key(vehicle_id, start, goal, use_backbone)
        
        # 检查缓存
        cached_result = self._check_cache(cache_key)
        if cached_result:
            self.performance_stats['cache_hits'] += 1
            planning_time = time.time() - start_time
            self.performance_stats['planning_times'].append(planning_time)
            return cached_result
        
        self.performance_stats['cache_misses'] += 1
        
        # 选择规划策略
        strategy = strategy or self.default_strategy
        self.performance_stats['strategy_usage'][strategy] += 1
        
        # 多次尝试规划
        best_path = None
        best_structure = None
        best_quality = 0
        
        for attempt in range(max_attempts):
            try:
                # 使用选定的策略规划
                path, structure = self.planning_strategies[strategy](
                    vehicle_id, start, goal, use_backbone, attempt
                )
                
                if path:
                    # 验证路径
                    if self._validate_path_comprehensive(path):
                        # 评估路径质量
                        quality = self.quality_assessor.evaluate_path(path)
                        
                        if quality > best_quality:
                            best_path = path
                            best_structure = structure
                            best_quality = quality
                            
                            # 如果质量足够高，提前结束
                            if quality >= 0.9:
                                break
                
            except Exception as e:
                if self.debug:
                    print(f"规划尝试 {attempt + 1} 失败: {e}")
                continue
        
        # 后处理优化
        if best_path and self.optimization_config['post_smoothing']:
            best_path = self._post_process_path(best_path)
            best_quality = self.quality_assessor.evaluate_path(best_path)
        
        # 冲突检查
        if best_path and check_conflicts and self.traffic_manager:
            if self.traffic_manager.check_path_conflicts(vehicle_id, best_path):
                # 尝试获取调整后的路径
                adjusted_path = self.traffic_manager.suggest_path_adjustment(
                    vehicle_id, start, goal
                )
                if adjusted_path:
                    best_path = adjusted_path
                    best_structure = self._analyze_path_structure(best_path)
        
        # 缓存结果
        if best_path and best_quality >= self.cache_config['quality_threshold']:
            self._add_to_cache(cache_key, (best_path, best_structure), best_quality)
        
        # 更新统计信息
        planning_time = time.time() - start_time
        self.performance_stats['planning_times'].append(planning_time)
        
        if best_quality > 0:
            self.performance_stats['quality_scores'].append(best_quality)
        
        if self.verbose_logging:
            print(f"路径规划完成: 车辆{vehicle_id}, 质量={best_quality:.2f}, "
                  f"耗时={planning_time:.3f}s, 策略={strategy}")
        
        return best_path, best_structure
    
    def _plan_backbone_first_strategy(self, vehicle_id, start, goal, use_backbone, attempt):
        """骨干网络优先策略"""
        if not use_backbone or not self.backbone_network:
            return self._plan_direct_optimized_strategy(vehicle_id, start, goal, False, attempt)
        
        # 寻找骨干网络接入点
        start_candidates = self.backbone_network.find_accessible_points(
            start, self.rrt_planner, max_candidates=3 + attempt
        )
        
        goal_candidates = self.backbone_network.find_accessible_points(
            goal, self.rrt_planner, max_candidates=3 + attempt
        )
        
        if not start_candidates or not goal_candidates:
            if self.debug:
                print(f"无法找到骨干网络接入点，回退到直接规划")
            return self._plan_direct_optimized_strategy(vehicle_id, start, goal, False, attempt)
        
        # 尝试所有候选组合
        best_path = None
        best_structure = None
        best_total_cost = float('inf')
        
        max_combinations = min(6, len(start_candidates) * len(goal_candidates))
        combinations_tested = 0
        
        for start_point in start_candidates:
            for goal_point in goal_candidates:
                if combinations_tested >= max_combinations:
                    break
                
                combinations_tested += 1
                
                # 规划三段路径
                result = self._plan_three_segment_path(
                    start, goal, start_point, goal_point
                )
                
                if result:
                    path, structure, total_cost = result
                    
                    if total_cost < best_total_cost:
                        best_total_cost = total_cost
                        best_path = path
                        best_structure = structure
            
            if combinations_tested >= max_combinations:
                break
        
        if best_path:
            self.performance_stats['backbone_successes'] += 1
            return best_path, best_structure
        else:
            self.performance_stats['direct_fallbacks'] += 1
            return self._plan_direct_optimized_strategy(vehicle_id, start, goal, False, attempt)
    
    def _plan_three_segment_path(self, start, goal, start_point, goal_point):
        """规划三段式路径：起点->骨干->骨干内->骨干->终点"""
        try:
            # 第一段：起点到骨干入口
            segment1 = self._plan_local_segment(start, start_point['position'])
            if not segment1:
                return None
            
            # 第二段：骨干网络内路径
            segment2 = self._plan_backbone_segment(start_point, goal_point)
            if not segment2:
                return None
            
            # 第三段：骨干出口到终点
            segment3 = self._plan_local_segment(goal_point['position'], goal)
            if not segment3:
                return None
            
            # 合并路径
            complete_path = self._merge_path_segments([segment1, segment2, segment3])
            
            if not complete_path:
                return None
            
            # 计算总成本
            total_cost = (
                self._calculate_path_cost(segment1) +
                self._calculate_path_cost(segment2) * 0.8 +  # 骨干路径权重较低
                self._calculate_path_cost(segment3)
            )
            
            # 构建结构信息
            structure = {
                'type': 'three_segment',
                'entry_point': start_point,
                'exit_point': goal_point,
                'backbone_segment': f"{start_point.get('path_id', '')}:{goal_point.get('path_id', '')}",
                'to_backbone_path': segment1,
                'backbone_path': segment2,
                'from_backbone_path': segment3,
                'total_cost': total_cost
            }
            
            return complete_path, structure, total_cost
            
        except Exception as e:
            if self.debug:
                print(f"三段路径规划失败: {e}")
            return None
    
    def _plan_local_segment(self, start, goal, max_iterations=2000):
        """规划局部路径段"""
        if not self.rrt_planner:
            return None
        
        # 如果距离很近，尝试直线连接
        distance = self._calculate_distance(start, goal)
        if distance < 5.0 and self._is_line_collision_free(start, goal):
            return [start, goal]
        
        # 使用RRT规划
        return self.rrt_planner.plan_path(start, goal, max_iterations=max_iterations)
    
    def _plan_backbone_segment(self, start_point, goal_point):
        """在骨干网络中规划路径段"""
        if not self.backbone_network:
            return None
        
        start_path_id = start_point.get('path_id')
        start_index = start_point.get('path_index', 0)
        goal_path_id = goal_point.get('path_id')
        goal_index = goal_point.get('path_index', 0)
        
        if not start_path_id or not goal_path_id:
            return None
        
        # 如果在同一条路径上
        if start_path_id == goal_path_id:
            return self.backbone_network.get_path_segment(
                start_path_id, start_index, goal_index
            )
        
        # 跨路径段规划
        compound_path_id = f"{start_path_id}:{goal_path_id}"
        return self.backbone_network.get_path_segment(
            compound_path_id, start_index, goal_index
        )
    
    def _plan_direct_optimized_strategy(self, vehicle_id, start, goal, use_backbone, attempt):
        """优化的直接规划策略"""
        if not self.rrt_planner:
            return None, None
        
        # 根据尝试次数调整参数
        max_iterations = 3000 + attempt * 1000
        
        path = self.rrt_planner.plan_path(
            start, goal, max_iterations=max_iterations
        )
        
        if path:
            structure = {
                'type': 'direct',
                'segments': 1,
                'method': 'rrt',
                'iterations': max_iterations
            }
            return path, structure
        
        return None, None
    
    def _plan_hybrid_multi_path_strategy(self, vehicle_id, start, goal, use_backbone, attempt):
        """混合多路径策略"""
        # 同时尝试多种规划方法
        methods = [
            ('backbone', self._plan_backbone_first_strategy),
            ('direct', self._plan_direct_optimized_strategy)
        ]
        
        best_path = None
        best_structure = None
        best_quality = 0
        
        for method_name, method_func in methods:
            try:
                path, structure = method_func(vehicle_id, start, goal, use_backbone, 0)
                
                if path:
                    quality = self.quality_assessor.evaluate_path(path)
                    
                    if quality > best_quality:
                        best_quality = quality
                        best_path = path
                        best_structure = structure
                        best_structure['method'] = method_name
                        
            except Exception as e:
                if self.debug:
                    print(f"混合策略中的{method_name}方法失败: {e}")
                continue
        
        return best_path, best_structure
    
    def _plan_emergency_fallback_strategy(self, vehicle_id, start, goal, use_backbone, attempt):
        """紧急回退策略"""
        # 使用最简单但可靠的方法
        if self._is_line_collision_free(start, goal):
            return [start, goal], {'type': 'emergency_direct', 'method': 'line'}
        
        # 尝试简化的A*或其他确定性算法
        return self._plan_simple_grid_path(start, goal)
    
    def _plan_simple_grid_path(self, start, goal):
        """简单的网格路径规划（A*简化版）"""
        # 这是一个简化的实现，实际中可以使用更复杂的算法
        try:
            # 转换为网格坐标
            start_grid = (int(start[0]), int(start[1]))
            goal_grid = (int(goal[0]), int(goal[1]))
            
            # 简单的直线路径，但避开障碍物
            path = []
            current = start_grid
            
            while current != goal_grid:
                # 计算下一步方向
                dx = 1 if goal_grid[0] > current[0] else (-1 if goal_grid[0] < current[0] else 0)
                dy = 1 if goal_grid[1] > current[1] else (-1 if goal_grid[1] < current[1] else 0)
                
                next_pos = (current[0] + dx, current[1] + dy)
                
                # 检查是否可通行
                if self._is_valid_position(next_pos[0], next_pos[1]):
                    current = next_pos
                    path.append((float(current[0]), float(current[1]), 0.0))
                else:
                    # 尝试绕行
                    if dx != 0 and self._is_valid_position(current[0] + dx, current[1]):
                        current = (current[0] + dx, current[1])
                        path.append((float(current[0]), float(current[1]), 0.0))
                    elif dy != 0 and self._is_valid_position(current[0], current[1] + dy):
                        current = (current[0], current[1] + dy)
                        path.append((float(current[0]), float(current[1]), 0.0))
                    else:
                        # 无法继续，失败
                        return None, None
                
                # 防止无限循环
                if len(path) > 1000:
                    break
            
            if path:
                # 添加终点
                path.append(goal)
                
                structure = {
                    'type': 'emergency_grid',
                    'method': 'simplified_astar',
                    'segments': len(path) - 1
                }
                
                return path, structure
            
        except Exception as e:
            if self.debug:
                print(f"简单网格路径规划失败: {e}")
        
        return None, None
    
    def _merge_path_segments(self, segments):
        """合并多个路径段，处理重复点"""
        if not segments or not any(segments):
            return None
        
        # 过滤空段
        valid_segments = [seg for seg in segments if seg and len(seg) > 0]
        
        if not valid_segments:
            return None
        
        merged_path = list(valid_segments[0])
        
        for segment in valid_segments[1:]:
            if not segment:
                continue
            
            # 检查连接点是否重复
            if (merged_path and segment and 
                self._is_same_position(merged_path[-1], segment[0])):
                # 跳过重复的连接点
                merged_path.extend(segment[1:])
            else:
                # 直接连接
                merged_path.extend(segment)
        
        return merged_path if len(merged_path) >= 2 else None
    
    def _validate_path_comprehensive(self, path):
        """综合路径验证"""
        if not self.validation_config['enabled'] or not path or len(path) < 2:
            return len(path) >= 2 if path else False
        
        # 基本碰撞检测
        if not self._validate_path_collision(path):
            return False
        
        # 运动学约束检查
        if not self._validate_kinematic_constraints(path):
            return False
        
        # 多次验证（如果启用）
        if self.validation_config['multi_pass']:
            # 使用不同采样密度再次验证
            dense_valid = self._validate_path_collision(
                path, 
                sample_density=self.validation_config['sample_density'] * 2
            )
            if not dense_valid:
                return False
        
        return True
    
    def _validate_path_collision(self, path, sample_density=None):
        """碰撞检测验证"""
        if not path or len(path) < 2:
            return False
        
        sample_density = sample_density or self.validation_config['sample_density']
        
        for i in range(len(path) - 1):
            if not self._validate_segment_collision(path[i], path[i+1], sample_density):
                return False
        
        return True
    
    def _validate_segment_collision(self, p1, p2, sample_density):
        """验证路径段是否无碰撞"""
        distance = self._calculate_distance(p1, p2)
        
        # 动态调整采样密度
        if self.validation_config['dynamic_density']:
            sample_density = max(sample_density, int(distance * 2))
        
        # 沿线段采样检查
        for i in range(sample_density + 1):
            t = i / max(1, sample_density)
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            
            # 检查基本碰撞
            if not self._is_valid_position_with_margin(int(x), int(y)):
                return False
        
        return True
    
    def _is_valid_position_with_margin(self, x, y):
        """带安全边距的位置检查"""
        margin = int(self.validation_config['safety_margin'])
        
        # 检查周围区域
        for dx in range(-margin, margin + 1):
            for dy in range(-margin, margin + 1):
                check_x, check_y = x + dx, y + dy
                
                if (check_x < 0 or check_x >= self.env.width or 
                    check_y < 0 or check_y >= self.env.height):
                    return False
                
                if hasattr(self.env, 'grid') and self.env.grid[check_x, check_y] == 1:
                    return False
        
        return True
    
    def _validate_kinematic_constraints(self, path):
        """验证运动学约束"""
        if not path or len(path) < 3:
            return True
        
        max_turning_rate = math.pi / 4  # 最大转弯率
        
        for i in range(1, len(path) - 1):
            prev = path[i-1]
            curr = path[i]
            next_p = path[i+1]
            
            # 计算转弯角度
            angle = self._calculate_turning_angle(prev, curr, next_p)
            
            # 检查转弯是否过急
            if angle > max_turning_rate:
                return False
        
        return True
    
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
    
    def _post_process_path(self, path):
        """路径后处理优化"""
        if not path or len(path) < 3:
            return path
        
        optimized = path
        
        # 捷径优化
        if self.optimization_config['shortcut_optimization']:
            optimized = self._shortcut_optimization(optimized)
        
        # 平滑处理
        if self.optimization_config['post_smoothing']:
            optimized = self._smooth_path_advanced(optimized)
        
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
    
    def _smooth_path_advanced(self, path, iterations=3):
        """高级路径平滑"""
        if len(path) <= 2:
            return path
        
        smoothed = list(path)
        
        for iteration in range(iterations):
            new_smoothed = [smoothed[0]]
            
            for i in range(1, len(smoothed) - 1):
                prev = smoothed[i-1]
                curr = smoothed[i]
                next_p = smoothed[i+1]
                
                # 自适应平滑权重
                weight = self._calculate_adaptive_smooth_weight(prev, curr, next_p)
                
                # 加权平均
                x = curr[0] * (1 - weight) + (prev[0] + next_p[0]) * weight / 2
                y = curr[1] * (1 - weight) + (prev[1] + next_p[1]) * weight / 2
                
                # 角度处理
                if len(curr) > 2:
                    theta = curr[2]
                else:
                    theta = math.atan2(next_p[1] - prev[1], next_p[0] - prev[0])
                
                # 验证平滑后的点
                if self._is_valid_position(int(x), int(y)):
                    new_smoothed.append((x, y, theta))
                else:
                    new_smoothed.append(curr)
            
            new_smoothed.append(smoothed[-1])
            smoothed = new_smoothed
        
        return smoothed
    
    def _calculate_adaptive_smooth_weight(self, prev, curr, next_p):
        """计算自适应平滑权重"""
        # 基于局部曲率调整权重
        angle = self._calculate_turning_angle(prev, curr, next_p)
        weight = min(0.6, angle / math.pi * 0.8)
        
        return weight
    
    # 缓存管理方法
    def _generate_cache_key(self, vehicle_id, start, goal, use_backbone):
        """生成缓存键"""
        start_rounded = (round(start[0], 1), round(start[1], 1))
        goal_rounded = (round(goal[0], 1), round(goal[1], 1))
        
        return f"{vehicle_id}:{start_rounded}:{goal_rounded}:{use_backbone}"
    
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
            if self.cache_config['lru_enabled']:
                self.route_cache.move_to_end(cache_key)
            
            # 更新命中计数
            metadata['hit_count'] = metadata.get('hit_count', 0) + 1
            
            return self.route_cache[cache_key]
        
        return None
    
    def _add_to_cache(self, cache_key, result, quality):
        """添加到缓存"""
        # 检查缓存大小
        if len(self.route_cache) >= self.cache_config['max_size']:
            if self.cache_config['lru_enabled']:
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
    
    # 分析和评估方法
    def _analyze_path_structure(self, path):
        """分析路径结构"""
        if not path or len(path) < 2:
            return {'type': 'empty'}
        
        structure = {
            'type': 'analyzed',
            'length': self._calculate_path_length(path),
            'segments': len(path) - 1,
            'complexity': self._calculate_path_complexity(path)
        }
        
        # 检查是否使用了骨干网络
        if self.backbone_network:
            backbone_usage = self._detect_backbone_usage(path)
            structure.update(backbone_usage)
        
        return structure
    
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
    
    def _detect_backbone_usage(self, path):
        """检测路径中骨干网络的使用情况"""
        if not self.backbone_network or not path:
            return {'uses_backbone': False}
        
        backbone_points = 0
        total_points = len(path)
        
        for point in path:
            nearest_conn = self.backbone_network.find_nearest_connection_optimized(
                point, max_distance=3.0
            )
            if nearest_conn:
                backbone_points += 1
        
        usage_ratio = backbone_points / max(1, total_points)
        
        return {
            'uses_backbone': usage_ratio > 0.3,
            'backbone_ratio': usage_ratio,
            'backbone_points': backbone_points
        }
    
    def _calculate_path_cost(self, path):
        """计算路径成本"""
        if not path or len(path) < 2:
            return float('inf')
        
        # 基础距离成本
        distance_cost = self._calculate_path_length(path)
        
        # 转弯成本
        turning_cost = 0
        for i in range(1, len(path) - 1):
            angle = self._calculate_turning_angle(path[i-1], path[i], path[i+1])
            turning_cost += angle * 2  # 转弯惩罚
        
        # 复杂度成本
        complexity_cost = self._calculate_path_complexity(path) * 10
        
        return distance_cost + turning_cost + complexity_cost
    
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
        
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    
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
        
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.performance_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'backbone_successes': 0,
            'direct_fallbacks': 0,
            'planning_times': [],
            'quality_scores': [],
            'strategy_usage': defaultdict(int)
        }


class PathQualityAssessor:
    """路径质量评估器"""
    
    def __init__(self, env):
        self.env = env
        
        # 质量权重
        self.weights = {
            'length_efficiency': 0.25,
            'smoothness': 0.20,
            'safety': 0.20,
            'complexity': 0.15,
            'clearance': 0.20
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
        
        # 间隙
        scores['clearance'] = self._evaluate_clearance(path)
        
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
        if not path:
            return 0
        
        min_safety = 1.0
        
        for point in path[::max(1, len(path)//20)]:  # 采样检查
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
    
    def _evaluate_clearance(self, path):
        """评估路径间隙"""
        if not path:
            return 0
        
        min_clearance = float('inf')
        
        for point in path[::max(1, len(path)//10)]:  # 采样检查
            clearance = self._calculate_clearance(point)
            min_clearance = min(min_clearance, clearance)
        
        # 转换为评分
        if min_clearance >= 5:
            return 1.0
        elif min_clearance >= 2:
            return 0.5 + 0.5 * (min_clearance - 2) / 3
        else:
            return 0.5 * min_clearance / 2
    
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
PathPlanner = OptimizedPathPlanner