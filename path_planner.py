import math
import numpy as np
import time
from collections import defaultdict, OrderedDict, deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from RRT import OptimizedRRTPlanner

@dataclass
class PlanningRequest:
    """路径规划请求"""
    vehicle_id: str
    start: Tuple[float, float, float]
    goal: Tuple[float, float, float]
    priority: int = 1
    deadline: float = 0.0
    quality_requirement: float = 0.6
    use_cache: bool = True
    strategy_hints: Dict = None

@dataclass
class PlanningResult:
    """路径规划结果"""
    path: List[Tuple[float, float, float]]
    structure: Dict
    quality_score: float
    planning_time: float
    cache_hit: bool
    rrt_stats: Dict = None

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
        # 初始化代码保持不变
        self.env = env
        self.backbone_network = backbone_network
        self.traffic_manager = traffic_manager
        
        # RRT规划器初始化 - 修复错误处理
        if rrt_planner is None:
            try:
                self.rrt_planner = OptimizedRRTPlanner(
                    env,
                    vehicle_length=6.0,
                    vehicle_width=3.0,
                    turning_radius=8.0,
                    step_size=0.6
                )
                print("已自动创建OptimizedRRTPlanner实例")
            except Exception as e:
                print(f"警告: 无法创建OptimizedRRTPlanner: {e}")
                try:
                    # 修复：导入应该在函数内部进行
                    from RRT import RRTPlanner
                    self.rrt_planner = RRTPlanner(
                        env,
                        vehicle_length=6.0,
                        vehicle_width=3.0,
                        turning_radius=8.0,
                        step_size=0.6
                    )
                    print("回退使用原始RRTPlanner")
                except Exception as e2:
                    print(f"警告: 无法创建任何RRT规划器: {e2}")
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
        
        # 如果有RRT规划器，建立双向连接
        if self.rrt_planner and hasattr(self.rrt_planner, 'set_backbone_network'):
            self.rrt_planner.set_backbone_network(backbone_network)
        
        # 如果骨干网络有RRT集成功能，建立连接
        if backbone_network and hasattr(backbone_network, 'set_rrt_planner'):
            backbone_network.set_rrt_planner(self.rrt_planner)
        
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
        """尝试使用骨干网络规划路径 - 增强版"""
        try:
            # 1. 识别目标点类型
            target_type, target_id = self.backbone_network.identify_target_point(goal)
            
            if not target_type:
                if self.debug:
                    print("目标不是特殊点，无法使用骨干路径")
                return None
            
            if self.debug:
                print(f"目标识别为: {target_type}_{target_id}")
            
            # 2. 优先使用增强的接口系统
            if hasattr(self.backbone_network, 'get_complete_path_via_interface_enhanced'):
                # 获取RRT引导信息
                guidance = None
                if hasattr(self.backbone_network, 'get_sampling_guidance_for_rrt'):
                    guidance = self.backbone_network.get_sampling_guidance_for_rrt(start, goal)
                
                # 准备RRT提示
                rrt_hints = self._prepare_rrt_hints(guidance, start, goal)
                
                complete_path, structure = self.backbone_network.get_complete_path_via_interface_enhanced(
                    start, target_type, target_id, rrt_hints
                )
                
                if complete_path and structure:
                    if self.debug:
                        print(f"增强接口系统规划成功: 总长度{len(complete_path)}, "
                              f"类型{structure.get('type')}, "
                              f"接口{structure.get('interface_id', 'N/A')}")
                    return complete_path, structure
            
            # 3. 回退到标准接口系统
            elif hasattr(self.backbone_network, 'get_path_from_position_to_target_via_interface'):
                complete_path, structure = self.backbone_network.get_path_from_position_to_target_via_interface(
                    start, target_type, target_id
                )
                
                if complete_path and structure:
                    if self.debug:
                        print(f"标准接口系统规划成功: 总长度{len(complete_path)}")
                    return complete_path, structure
            
            # 4. 回退到原始骨干网络方法
            else:
                complete_path, structure = self.backbone_network.get_path_from_position_to_target(
                    start, target_type, target_id
                )
                
                if complete_path and structure:
                    if self.debug:
                        print(f"原始骨干网络规划成功: 总长度{len(complete_path)}")
                    return complete_path, structure
            
            return None
            
        except Exception as e:
            if self.debug:
                print(f"骨干网络规划失败: {e}")
            return None
    
    def _prepare_rrt_hints(self, guidance, start, goal):
        """准备RRT提示信息"""
        hints = {}
        
        if guidance:
            # 采样区域提示
            if guidance.get('priority_regions'):
                hints['priority_sampling_regions'] = guidance['priority_regions']
            
            # 骨干路径提示
            if guidance.get('backbone_hints'):
                hints['backbone_alignment_targets'] = guidance['backbone_hints']
            
            # 质量建议
            avg_backbone_quality = sum(
                hint.get('quality', 0.5) for hint in guidance.get('backbone_hints', [])
            ) / max(1, len(guidance.get('backbone_hints', [])))
            
            if avg_backbone_quality > 0.7:
                hints['smoothing_suggested'] = True
                hints['target_density'] = 2.0
        
        # 距离自适应
        distance = self._calculate_distance(start, goal)
        if distance > 100:
            hints['performance_mode'] = 'exploration'
        elif distance < 20:
            hints['performance_mode'] = 'precision'
        
        return hints
    
    def _direct_planning(self, start, goal, attempt):
        """
        直接点对点规划
        """
        if not self.rrt_planner:
            print("RRT规划器不可用")
            return None
        
        try:
            # 根据尝试次数调整参数
            max_iterations = 4000 + attempt * 1000
            
            if self.debug:
                print(f"直接规划: 最大迭代次数 {max_iterations}")
            
            # 使用优化的RRT规划器
            if hasattr(self.rrt_planner, 'plan_path'):
                path = self.rrt_planner.plan_path(
                    start=start,
                    goal=goal,
                    agent_id='temp',
                    max_iterations=max_iterations,
                    quality_threshold=0.6
                )
            else:
                # 回退到原始接口
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
        
        # RRT统计
        if self.rrt_planner and hasattr(self.rrt_planner, 'get_statistics'):
            stats['rrt_stats'] = self.rrt_planner.get_statistics()
        
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


class AdvancedPathPlanner(SimplifiedPathPlanner):
    """高级路径规划器 - 深度集成优化RRT"""
    
    def __init__(self, env, backbone_network=None, rrt_planner=None, traffic_manager=None):
        super().__init__(env, backbone_network, rrt_planner, traffic_manager)
        
        # 高级规划配置
        self.advanced_config = {
            'adaptive_quality_threshold': True,
            'intelligent_strategy_selection': True,
            'batch_planning_optimization': True,
            'predictive_caching': True,
            'real_time_performance_tuning': True
        }
        
        # 规划请求队列和批处理
        self.planning_queue = deque()
        self.batch_processing = False
        self.max_batch_size = 8
        self.batch_timeout = 0.5  # 秒
        
        # 性能自适应系统
        self.performance_monitor = PlanningPerformanceMonitor()
        self.strategy_selector = IntelligentStrategySelector()
        
        # 预测性缓存
        self.predictive_cache = PredictivePathCache()
        self.cache_hit_threshold = 0.8  # 缓存命中率目标
        
        # 实时统计
        self.real_time_stats = {
            'requests_per_second': 0.0,
            'average_quality': 0.0,
            'backbone_utilization': 0.0,
            'system_load': 0.0
        }
        
        print("初始化高级路径规划器")
    
    def set_rrt_planner(self, rrt_planner):
        """设置RRT规划器并启用高级集成"""
        super().set_rrt_planner(rrt_planner)
        
        if rrt_planner and hasattr(rrt_planner, 'set_backbone_network'):
            # 启用深度集成
            self._enable_advanced_rrt_integration(rrt_planner)
    
    def _enable_advanced_rrt_integration(self, rrt_planner):
        """启用高级RRT集成"""
        # 共享性能监控
        if hasattr(rrt_planner, 'performance_monitor'):
            rrt_planner.performance_monitor = self.performance_monitor
        
        # 启用预测性缓存协调
        if hasattr(rrt_planner, 'path_cache'):
            self.predictive_cache.set_rrt_cache_ref(rrt_planner.path_cache)
        
        # 设置质量反馈循环
        self._setup_quality_feedback_loop(rrt_planner)
        
        print("高级RRT集成已启用")
    
    def _setup_quality_feedback_loop(self, rrt_planner):
        """设置质量反馈循环"""
        # 这里可以设置质量反馈机制
        pass
    
    def plan_path_advanced(self, request: PlanningRequest) -> PlanningResult:
        """高级路径规划接口"""
        start_time = time.time()
        
        # 1. 智能策略选择
        strategy = self.strategy_selector.select_strategy(request, self.real_time_stats)
        
        # 2. 预测性缓存检查
        cached_result = self.predictive_cache.check_predictive_cache(
            request.vehicle_id, request.start, request.goal
        )
        
        if cached_result and cached_result.quality_score >= request.quality_requirement:
            self.performance_monitor.record_cache_hit(request.vehicle_id)
            return cached_result
        
        # 3. 自适应质量阈值
        adaptive_threshold = self._calculate_adaptive_threshold(request, strategy)
        
        # 4. 执行规划
        planning_result = self._execute_planning_with_strategy(request, strategy, adaptive_threshold)
        
        # 5. 结果后处理和缓存
        self._post_process_result(planning_result, request)
        
        # 6. 更新性能监控
        planning_time = time.time() - start_time
        self.performance_monitor.record_planning(
            request.vehicle_id, planning_time, planning_result.quality_score
        )
        
        return planning_result
    
    def _execute_planning_with_strategy(self, request: PlanningRequest, 
                                       strategy: Dict, adaptive_threshold: float) -> PlanningResult:
        """根据策略执行规划"""
        
        if strategy['type'] == 'backbone_priority':
            return self._plan_with_backbone_priority(request, adaptive_threshold)
        elif strategy['type'] == 'direct_rrt':
            return self._plan_with_direct_rrt(request, adaptive_threshold)
        elif strategy['type'] == 'hybrid':
            return self._plan_with_hybrid_strategy(request, adaptive_threshold)
        else:
            # 默认策略
            return self._plan_with_auto_strategy(request, adaptive_threshold)
    
    def _plan_with_backbone_priority(self, request: PlanningRequest, 
                                   threshold: float) -> PlanningResult:
        """骨干网络优先策略"""
        start_time = time.time()
        
        # 1. 尝试骨干网络路径
        if self.backbone_network:
            # 获取RRT采样引导
            guidance = None
            if hasattr(self.backbone_network, 'get_sampling_guidance_for_rrt'):
                guidance = self.backbone_network.get_sampling_guidance_for_rrt(
                    request.start, request.goal
                )
            
            # 使用增强的骨干网络接口
            if hasattr(self.backbone_network, 'get_complete_path_via_interface_enhanced'):
                target_type, target_id = self.backbone_network.identify_target_point(request.goal)
                
                if target_type:
                    rrt_hints = self._prepare_rrt_hints(guidance, request.start, request.goal)
                    
                    backbone_result = self.backbone_network.get_complete_path_via_interface_enhanced(
                        request.start, target_type, target_id, rrt_hints
                    )
                    
                    if backbone_result and backbone_result[0]:
                        path, structure = backbone_result
                        quality = self._evaluate_path_quality(path)
                        
                        if quality >= threshold:
                            return PlanningResult(
                                path=path,
                                structure=structure,
                                quality_score=quality,
                                planning_time=time.time() - start_time,
                                cache_hit=False,
                                rrt_stats=self._get_rrt_stats()
                            )
        
        # 2. 回退到直接RRT规划
        return self._plan_with_direct_rrt(request, threshold)
    
    def _plan_with_direct_rrt(self, request: PlanningRequest, 
                            threshold: float) -> PlanningResult:
        """直接RRT规划策略"""
        start_time = time.time()
        
        if not self.rrt_planner:
            raise ValueError("RRT规划器不可用")
        
        # 获取系统负载调整规划参数
        system_load = self.real_time_stats['system_load']
        max_iterations = self._adjust_iterations_for_load(4000, system_load)
        
        # 执行RRT规划
        if hasattr(self.rrt_planner, 'plan_path'):
            path = self.rrt_planner.plan_path(
                start=request.start,
                goal=request.goal,
                agent_id=request.vehicle_id,
                max_iterations=max_iterations,
                quality_threshold=threshold
            )
        else:
            path = self.rrt_planner.plan_path(request.start, request.goal, max_iterations=max_iterations)
        
        if path:
            # 分析路径结构
            structure = self._analyze_path_structure_advanced(path, request)
            quality = self._evaluate_path_quality(path)
            
            return PlanningResult(
                path=path,
                structure=structure,
                quality_score=quality,
                planning_time=time.time() - start_time,
                cache_hit=False,
                rrt_stats=self._get_rrt_stats()
            )
        
        return None
    
    def _plan_with_hybrid_strategy(self, request: PlanningRequest, 
                                 threshold: float) -> PlanningResult:
        """混合策略 - 同时尝试多种方法"""
        start_time = time.time()
        
        results = []
        
        # 并行尝试不同策略（简化实现，实际可用threading）
        strategies = ['backbone_priority', 'direct_rrt']
        
        for strategy_type in strategies:
            try:
                strategy = {'type': strategy_type}
                result = self._execute_planning_with_strategy(request, strategy, threshold)
                
                if result and result.path:
                    results.append(result)
                    
                    # 如果找到高质量路径，立即返回
                    if result.quality_score >= threshold * 1.2:
                        return result
            
            except Exception as e:
                if self.debug:
                    print(f"策略 {strategy_type} 失败: {e}")
                continue
        
        # 选择最佳结果
        if results:
            best_result = max(results, key=lambda r: r.quality_score)
            best_result.planning_time = time.time() - start_time
            return best_result
        
        return None
    
    def _plan_with_auto_strategy(self, request: PlanningRequest, 
                               threshold: float) -> PlanningResult:
        """自动策略选择"""
        # 默认使用骨干网络优先策略
        return self._plan_with_backbone_priority(request, threshold)
    
    def _calculate_adaptive_threshold(self, request: PlanningRequest, strategy: Dict) -> float:
        """计算自适应质量阈值"""
        base_threshold = request.quality_requirement
        
        # 系统负载调整
        load_factor = 1.0 - self.real_time_stats['system_load'] * 0.2
        
        # 历史性能调整
        vehicle_history = self.performance_monitor.get_vehicle_history(request.vehicle_id)
        if vehicle_history['avg_quality'] > 0.8:
            # 该车辆历史质量高，可以适当提高要求
            history_factor = 1.1
        else:
            history_factor = 0.95
        
        # 策略调整
        strategy_factor = 1.0
        if strategy['type'] == 'backbone_priority':
            strategy_factor = 1.05  # 骨干网络通常质量更高
        elif strategy['type'] == 'direct_rrt':
            strategy_factor = 0.95  # 直接RRT可能质量略低
        
        adaptive_threshold = base_threshold * load_factor * history_factor * strategy_factor
        
        return max(0.3, min(0.95, adaptive_threshold))  # 限制范围
    
    def _adjust_iterations_for_load(self, base_iterations: int, system_load: float) -> int:
        """根据系统负载调整迭代次数"""
        if system_load > 0.9:
            return int(base_iterations * 0.6)  # 高负载时减少迭代
        elif system_load > 0.7:
            return int(base_iterations * 0.8)
        elif system_load < 0.3:
            return int(base_iterations * 1.3)  # 低负载时增加迭代提高质量
        else:
            return base_iterations
    
    def _analyze_path_structure_advanced(self, path: List, request: PlanningRequest) -> Dict:
        """高级路径结构分析"""
        structure = {
            'type': 'rrt_direct',
            'total_length': len(path),
            'path_length_meters': self._calculate_path_length(path),
            'complexity_score': 0.0,
            'backbone_utilization': 0.0,
            'safety_score': 0.0,
            'efficiency_score': 0.0
        }
        
        if len(path) < 2:
            return structure
        
        # 计算复杂度
        structure['complexity_score'] = self._calculate_path_complexity(path)
        
        # 计算骨干网络利用率
        if self.backbone_network:
            structure['backbone_utilization'] = self._calculate_backbone_utilization(path)
        
        # 计算安全性评分
        structure['safety_score'] = self._calculate_safety_score(path)
        
        # 计算效率评分
        direct_distance = math.sqrt(
            (request.goal[0] - request.start[0])**2 + 
            (request.goal[1] - request.start[1])**2
        )
        if structure['path_length_meters'] > 0:
            structure['efficiency_score'] = direct_distance / structure['path_length_meters']
        
        return structure
    
    def _calculate_backbone_utilization(self, path: List) -> float:
        """计算路径的骨干网络利用率"""
        if not self.backbone_network or not hasattr(self.backbone_network, 'backbone_paths'):
            return 0.0
        
        backbone_points = 0
        total_sampled_points = 0
        
        # 采样检查路径点
        sample_step = max(1, len(path) // 20)  # 采样约20个点
        
        for i in range(0, len(path), sample_step):
            point = path[i]
            total_sampled_points += 1
            
            # 检查是否接近任何骨干路径
            if self._is_point_near_backbone(point):
                backbone_points += 1
        
        return backbone_points / max(1, total_sampled_points)
    
    def _is_point_near_backbone(self, point: Tuple, threshold: float = 8.0) -> bool:
        """检查点是否接近骨干路径"""
        for path_data in self.backbone_network.backbone_paths.values():
            backbone_path = path_data.get('path', [])
            
            for bp in backbone_path[::5]:  # 每隔5个点检查一次
                distance = math.sqrt(
                    (point[0] - bp[0])**2 + (point[1] - bp[1])**2
                )
                if distance < threshold:
                    return True
        
        return False
    
    def _calculate_safety_score(self, path: List) -> float:
        """计算安全性评分"""
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
    
    def _evaluate_path_quality(self, path):
        """评估路径质量"""
        if self.quality_assessor:
            return self.quality_assessor.evaluate_path(path)
        return 0.5
    
    def _post_process_result(self, result: PlanningResult, request: PlanningRequest):
        """结果后处理"""
        if not result or not result.path:
            return
        
        # 添加到预测性缓存
        self.predictive_cache.add_path(
            request.vehicle_id, request.start, request.goal, result
        )
        
        # 更新骨干网络反馈
        if self.backbone_network and hasattr(self.backbone_network, 'update_path_feedback'):
            self.backbone_network.update_path_feedback(
                result.path, result.planning_time, result.quality_score, result.cache_hit
            )
        
        # 更新RRT统计
        if self.rrt_planner and hasattr(self.rrt_planner, 'path_cache'):
            rrt_stats = self.rrt_planner.get_statistics()
            result.rrt_stats = rrt_stats
    
    def _get_rrt_stats(self) -> Dict:
        """获取RRT统计信息"""
        if self.rrt_planner and hasattr(self.rrt_planner, 'get_statistics'):
            return self.rrt_planner.get_statistics()
        return {}
    
    def batch_plan_paths(self, requests: List[PlanningRequest]) -> List[PlanningResult]:
        """批量路径规划"""
        if not requests:
            return []
        
        # 按优先级和截止时间排序
        sorted_requests = sorted(
            requests, 
            key=lambda r: (r.priority, r.deadline if r.deadline > 0 else float('inf'))
        )
        
        results = []
        batch_start_time = time.time()
        
        # 批量优化：预热缓存、共享计算等
        self._prepare_batch_planning(sorted_requests)
        
        for request in sorted_requests:
            # 检查时间限制
            if request.deadline > 0 and time.time() > request.deadline:
                results.append(None)  # 超时
                continue
            
            result = self.plan_path_advanced(request)
            results.append(result)
            
            # 批量处理优化：如果已经花费太多时间，降低后续请求的质量要求
            batch_time = time.time() - batch_start_time
            if batch_time > self.batch_timeout:
                self._adjust_remaining_requests_for_time(
                    sorted_requests[len(results):], batch_time
                )
        
        return results
    
    def _prepare_batch_planning(self, requests: List[PlanningRequest]):
        """准备批量规划"""
        # 预热相关的骨干路径
        if self.backbone_network:
            target_points = set()
            for req in requests:
                target_type, target_id = self.backbone_network.identify_target_point(req.goal)
                if target_type:
                    target_points.add((target_type, target_id))
            
            # 预加载相关骨干路径信息
            for target_type, target_id in target_points:
                self.backbone_network.find_paths_to_target(target_type, target_id)
    
    def _adjust_remaining_requests_for_time(self, remaining_requests: List[PlanningRequest], 
                                          elapsed_time: float):
        """为剩余请求调整参数以节省时间"""
        time_pressure_factor = min(2.0, elapsed_time / self.batch_timeout)
        
        for request in remaining_requests:
            # 降低质量要求
            request.quality_requirement *= (1.0 / time_pressure_factor)
            request.quality_requirement = max(0.3, request.quality_requirement)
            
            # 建议使用快速策略
            if not request.strategy_hints:
                request.strategy_hints = {}
            request.strategy_hints['prefer_speed'] = True
    
    def update_real_time_stats(self):
        """更新实时统计"""
        # 这个方法应该定期调用（例如每秒一次）
        current_time = time.time()
        
        # 更新请求频率
        recent_requests = self.performance_monitor.get_recent_request_count(1.0)
        self.real_time_stats['requests_per_second'] = recent_requests
        
        # 更新平均质量
        self.real_time_stats['average_quality'] = self.performance_monitor.get_average_quality()
        
        # 更新骨干网络利用率
        if self.backbone_network:
            backbone_stats = self.backbone_network.get_rrt_performance_stats()
            total_requests = backbone_stats['cache_stats']['total_requests']
            if total_requests > 0:
                self.real_time_stats['backbone_utilization'] = (
                    backbone_stats['cache_stats']['cache_hits'] / total_requests
                )
        
        # 更新系统负载（基于请求频率和质量要求）
        base_load = min(1.0, recent_requests / 10.0)  # 假设10请求/秒为满载
        quality_load = self.real_time_stats['average_quality']  # 质量越高负载越大
        self.real_time_stats['system_load'] = (base_load + quality_load) / 2
    
    def get_advanced_performance_stats(self) -> Dict:
        """获取高级性能统计"""
        stats = super().get_performance_stats()
        
        # 添加高级统计
        stats.update({
            'real_time_stats': self.real_time_stats.copy(),
            'performance_monitor': self.performance_monitor.get_summary(),
            'predictive_cache': self.predictive_cache.get_stats(),
            'strategy_selector': self.strategy_selector.get_stats()
        })
        
        # RRT集成统计
        if self.rrt_planner:
            stats['rrt_integration'] = self._get_rrt_stats()
        
        # 骨干网络集成统计
        if self.backbone_network and hasattr(self.backbone_network, 'get_rrt_performance_stats'):
            stats['backbone_integration'] = self.backbone_network.get_rrt_performance_stats()
        
        return stats


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


class PlanningPerformanceMonitor:
    """规划性能监控器"""
    
    def __init__(self, history_size=1000):
        self.history_size = history_size
        self.vehicle_history = defaultdict(lambda: {
            'planning_times': deque(maxlen=100),
            'quality_scores': deque(maxlen=100),
            'cache_hits': 0,
            'total_requests': 0
        })
        
        self.global_history = {
            'planning_times': deque(maxlen=history_size),
            'quality_scores': deque(maxlen=history_size),
            'request_timestamps': deque(maxlen=history_size)
        }
    
    def record_planning(self, vehicle_id: str, planning_time: float, quality_score: float):
        """记录规划结果"""
        # 车辆级记录
        vh = self.vehicle_history[vehicle_id]
        vh['planning_times'].append(planning_time)
        vh['quality_scores'].append(quality_score)
        vh['total_requests'] += 1
        
        # 全局记录
        self.global_history['planning_times'].append(planning_time)
        self.global_history['quality_scores'].append(quality_score)
        self.global_history['request_timestamps'].append(time.time())
    
    def record_cache_hit(self, vehicle_id: str):
        """记录缓存命中"""
        self.vehicle_history[vehicle_id]['cache_hits'] += 1
    
    def get_vehicle_history(self, vehicle_id: str) -> Dict:
        """获取车辆历史性能"""
        vh = self.vehicle_history[vehicle_id]
        
        if not vh['quality_scores']:
            return {'avg_quality': 0.5, 'avg_planning_time': 1.0, 'cache_hit_rate': 0.0}
        
        return {
            'avg_quality': sum(vh['quality_scores']) / len(vh['quality_scores']),
            'avg_planning_time': sum(vh['planning_times']) / len(vh['planning_times']),
            'cache_hit_rate': vh['cache_hits'] / max(1, vh['total_requests'])
        }
    
    def get_recent_request_count(self, time_window: float) -> int:
        """获取最近时间窗口内的请求数量"""
        current_time = time.time()
        count = 0
        
        for timestamp in reversed(self.global_history['request_timestamps']):
            if current_time - timestamp <= time_window:
                count += 1
            else:
                break
        
        return count
    
    def get_average_quality(self) -> float:
        """获取平均质量"""
        if not self.global_history['quality_scores']:
            return 0.5
        
        return sum(self.global_history['quality_scores']) / len(self.global_history['quality_scores'])
    
    def get_summary(self) -> Dict:
        """获取性能摘要"""
        return {
            'total_vehicles': len(self.vehicle_history),
            'global_avg_quality': self.get_average_quality(),
            'recent_request_rate': self.get_recent_request_count(60.0) / 60.0,  # 每秒请求数
            'total_global_requests': len(self.global_history['planning_times'])
        }


class IntelligentStrategySelector:
    """智能策略选择器"""
    
    def __init__(self):
        self.strategy_performance = defaultdict(lambda: {
            'success_rate': 0.5,
            'avg_quality': 0.5,
            'avg_time': 1.0,
            'usage_count': 0
        })
    
    def select_strategy(self, request: PlanningRequest, system_stats: Dict) -> Dict:
        """选择最佳策略"""
        # 基于系统状态和请求特征选择策略
        if system_stats['system_load'] > 0.8:
            # 高负载时优先选择快速策略
            return {'type': 'direct_rrt', 'reason': 'high_system_load'}
        
        if request.quality_requirement > 0.8:
            # 高质量要求时优先骨干网络
            return {'type': 'backbone_priority', 'reason': 'high_quality_requirement'}
        
        # 基于历史性能选择
        backbone_perf = self.strategy_performance['backbone_priority']
        direct_perf = self.strategy_performance['direct_rrt']
        
        if backbone_perf['success_rate'] > direct_perf['success_rate'] * 1.2:
            return {'type': 'backbone_priority', 'reason': 'historical_performance'}
        
        # 默认混合策略
        return {'type': 'hybrid', 'reason': 'balanced_approach'}
    
    def update_strategy_performance(self, strategy_type: str, success: bool, 
                                  quality: float, planning_time: float):
        """更新策略性能"""
        perf = self.strategy_performance[strategy_type]
        perf['usage_count'] += 1
        
        # 使用指数移动平均更新
        alpha = 0.1
        if success:
            perf['success_rate'] = (1-alpha) * perf['success_rate'] + alpha * 1.0
            perf['avg_quality'] = (1-alpha) * perf['avg_quality'] + alpha * quality
        else:
            perf['success_rate'] = (1-alpha) * perf['success_rate'] + alpha * 0.0
        
        perf['avg_time'] = (1-alpha) * perf['avg_time'] + alpha * planning_time
    
    def get_stats(self) -> Dict:
        """获取策略统计"""
        return dict(self.strategy_performance)


class PredictivePathCache:
    """预测性路径缓存"""
    
    def __init__(self, max_size=500):
        self.max_size = max_size
        self.cache = {}  # {key: PlanningResult}
        self.usage_count = defaultdict(int)
        self.prediction_model = SimplePredictionModel()
    
    def check_predictive_cache(self, vehicle_id: str, start: Tuple, 
                             goal: Tuple) -> Optional[PlanningResult]:
        """检查预测性缓存"""
        # 精确匹配
        exact_key = self._generate_key(vehicle_id, start, goal)
        if exact_key in self.cache:
            self.usage_count[exact_key] += 1
            return self.cache[exact_key]
        
        # 模糊匹配（近似路径）
        fuzzy_result = self._fuzzy_cache_lookup(vehicle_id, start, goal)
        if fuzzy_result:
            return fuzzy_result
        
        return None
    
    def add_path(self, vehicle_id: str, start: Tuple, goal: Tuple, 
                result: PlanningResult):
        """添加路径到缓存"""
        if len(self.cache) >= self.max_size:
            self._evict_least_used()
        
        key = self._generate_key(vehicle_id, start, goal)
        self.cache[key] = result
        
        # 更新预测模型
        self.prediction_model.update(vehicle_id, start, goal, result)
    
    def _generate_key(self, vehicle_id: str, start: Tuple, goal: Tuple) -> str:
        """生成缓存键"""
        return f"{vehicle_id}:{start[0]:.1f},{start[1]:.1f}:{goal[0]:.1f},{goal[1]:.1f}"
    
    def _fuzzy_cache_lookup(self, vehicle_id: str, start: Tuple, 
                          goal: Tuple, tolerance=5.0) -> Optional[PlanningResult]:
        """模糊缓存查找"""
        for key, result in self.cache.items():
            if key.startswith(f"{vehicle_id}:"):
                # 解析缓存的起终点
                parts = key.split(':')
                if len(parts) >= 3:
                    cached_start = [float(x) for x in parts[1].split(',')]
                    cached_goal = [float(x) for x in parts[2].split(',')]
                    
                    # 检查距离容差
                    start_dist = math.sqrt(
                        (start[0] - cached_start[0])**2 + (start[1] - cached_start[1])**2
                    )
                    goal_dist = math.sqrt(
                        (goal[0] - cached_goal[0])**2 + (goal[1] - cached_goal[1])**2
                    )
                    
                    if start_dist < tolerance and goal_dist < tolerance:
                        self.usage_count[key] += 1
                        return result
        
        return None
    
    def _evict_least_used(self):
        """驱逐最少使用的缓存项"""
        if not self.cache:
            return
        
        least_used_key = min(self.usage_count.keys(), key=lambda k: self.usage_count[k])
        del self.cache[least_used_key]
        del self.usage_count[least_used_key]
    
    def set_rrt_cache_ref(self, rrt_cache):
        """设置RRT缓存引用，实现缓存协调"""
        self.rrt_cache_ref = rrt_cache
    
    def get_stats(self) -> Dict:
        """获取缓存统计"""
        total_usage = sum(self.usage_count.values())
        return {
            'cache_size': len(self.cache),
            'total_usage': total_usage,
            'avg_usage_per_item': total_usage / max(1, len(self.cache)),
            'hit_distribution': dict(self.usage_count)
        }


class SimplePredictionModel:
    """简单的路径预测模型"""
    
    def __init__(self):
        self.vehicle_patterns = defaultdict(list)  # 车辆行为模式
    
    def update(self, vehicle_id: str, start: Tuple, goal: Tuple, result: PlanningResult):
        """更新预测模型"""
        pattern = {
            'start': start,
            'goal': goal,
            'quality': result.quality_score,
            'planning_time': result.planning_time,
            'timestamp': time.time()
        }
        
        self.vehicle_patterns[vehicle_id].append(pattern)
        
        # 限制历史长度
        if len(self.vehicle_patterns[vehicle_id]) > 50:
            self.vehicle_patterns[vehicle_id] = self.vehicle_patterns[vehicle_id][-50:]
    
    def predict_next_destination(self, vehicle_id: str, current_position: Tuple) -> Optional[Tuple]:
        """预测下一个目的地"""
        patterns = self.vehicle_patterns.get(vehicle_id, [])
        if len(patterns) < 2:
            return None
        
        # 简单的模式识别：查找相似的起点
        for pattern in reversed(patterns[-10:]):  # 查看最近10个模式
            start_dist = math.sqrt(
                (current_position[0] - pattern['start'][0])**2 + 
                (current_position[1] - pattern['start'][1])**2
            )
            
            if start_dist < 10.0:  # 如果起点相似
                return pattern['goal']
        
        return None


# 保持向后兼容性
OptimizedPathPlanner = AdvancedPathPlanner
PathPlanner = SimplifiedPathPlanner