"""
traffic_manager.py - 优化版交通管理器
集成接口系统、增强ECBS、智能冲突检测
"""

import math
import heapq
import time
import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field
import threading
from concurrent.futures import ThreadPoolExecutor
import weakref

@dataclass
class EnhancedConflict:
    """增强的冲突表示"""
    agent1: str
    agent2: str
    location: Tuple[float, float]
    time_step: float
    conflict_type: str = 'vertex'  # vertex, edge, interface, time_window
    severity: float = 1.0
    confidence: float = 1.0
    resolution_hints: Dict = field(default_factory=dict)
    
    # 接口相关信息
    interface_id: Optional[str] = None
    resource_type: Optional[str] = None  # interface, backbone_segment, loading_point
    
    def __lt__(self, other):
        # 优先级：严重性 -> 置信度 -> 时间
        if abs(self.severity - other.severity) > 0.1:
            return self.severity > other.severity
        if abs(self.confidence - other.confidence) > 0.1:
            return self.confidence > other.confidence
        return self.time_step < other.time_step

@dataclass 
class SmartConstraint:
    """智能约束条件"""
    agent_id: str
    constraint_type: str  # vertex, edge, interface, time_window
    location: Optional[Tuple[float, float]] = None
    time_step: Optional[float] = None
    time_window: Optional[Tuple[float, float]] = None
    interface_id: Optional[str] = None
    priority: int = 1
    
    # 智能约束参数
    flexibility: float = 0.0  # 约束的灵活性 [0,1]
    alternative_options: List = field(default_factory=list)

class InterfaceReservationManager:
    """接口预约管理器"""
    
    def __init__(self):
        self.reservations = {}  # {interface_id: [(vehicle_id, start_time, end_time, priority)]}
        self.reservation_lock = threading.Lock()
        self.default_reservation_duration = 120  # 2分钟默认预约
        
    def reserve_interface(self, interface_id: str, vehicle_id: str, 
                         start_time: float, duration: float = None, 
                         priority: int = 1) -> bool:
        """预约接口使用时间"""
        if duration is None:
            duration = self.default_reservation_duration
        
        end_time = start_time + duration
        
        with self.reservation_lock:
            if interface_id not in self.reservations:
                self.reservations[interface_id] = []
            
            # 检查时间冲突
            for existing_vehicle, existing_start, existing_end, existing_priority in self.reservations[interface_id]:
                if existing_vehicle == vehicle_id:
                    continue  # 同一车辆可以更新预约
                
                # 检查时间重叠
                if not (end_time <= existing_start or start_time >= existing_end):
                    # 有重叠，检查优先级
                    if priority <= existing_priority:
                        return False  # 优先级不够，预约失败
                    else:
                        # 优先级更高，移除低优先级预约
                        self.reservations[interface_id].remove(
                            (existing_vehicle, existing_start, existing_end, existing_priority)
                        )
            
            # 添加新预约
            self.reservations[interface_id].append((vehicle_id, start_time, end_time, priority))
            self.reservations[interface_id].sort(key=lambda x: x[1])  # 按开始时间排序
            
            return True
    
    def release_interface(self, interface_id: str, vehicle_id: str) -> bool:
        """释放接口预约"""
        with self.reservation_lock:
            if interface_id not in self.reservations:
                return False
            
            # 移除该车辆的所有预约
            original_count = len(self.reservations[interface_id])
            self.reservations[interface_id] = [
                res for res in self.reservations[interface_id] 
                if res[0] != vehicle_id
            ]
            
            return len(self.reservations[interface_id]) < original_count
    
    def get_interface_conflicts(self, current_time: float) -> List[EnhancedConflict]:
        """检测接口时间冲突"""
        conflicts = []
        
        with self.reservation_lock:
            for interface_id, reservations in self.reservations.items():
                # 清理过期预约
                self.reservations[interface_id] = [
                    res for res in reservations if res[2] > current_time
                ]
                reservations = self.reservations[interface_id]
                
                # 检查重叠预约
                for i in range(len(reservations)):
                    for j in range(i + 1, len(reservations)):
                        res1 = reservations[i]
                        res2 = reservations[j]
                        
                        # 检查时间重叠
                        if not (res1[2] <= res2[1] or res1[1] >= res2[2]):
                            conflict = EnhancedConflict(
                                agent1=res1[0],
                                agent2=res2[0],
                                location=(0, 0),  # 接口位置待填充
                                time_step=max(res1[1], res2[1]),
                                conflict_type='interface',
                                severity=2.0,  # 接口冲突严重性高
                                interface_id=interface_id,
                                resource_type='interface',
                                resolution_hints={
                                    'can_delay': True,
                                    'alternative_interfaces': [],
                                    'priority_difference': abs(res1[3] - res2[3])
                                }
                            )
                            conflicts.append(conflict)
        
        return conflicts

class EnhancedECBSSolver:
    """增强的ECBS求解器 - 深度集成接口系统"""
    
    def __init__(self, env, backbone_network=None):
        self.env = env
        self.backbone_network = backbone_network
        self.interface_manager = InterfaceReservationManager()
        
        # ECBS参数优化
        self.suboptimality_bound = 1.3  # 降低子最优界限
        self.max_search_time = 20.0
        self.max_nodes_expanded = 800
        self.focal_weight = 1.2
        
        # 启发式权重
        self.heuristic_weights = {
            'distance': 0.4,
            'backbone_alignment': 0.3,
            'interface_availability': 0.2,
            'time_urgency': 0.1
        }
        
        # 性能统计
        self.stats = {
            'nodes_expanded': 0,
            'conflicts_resolved': 0,
            'total_time': 0,
            'cache_hits': 0,
            'interface_conflicts_resolved': 0,
            'backbone_reroutings': 0
        }
        
        # 解决方案缓存
        self.solution_cache = {}
        self.cache_size_limit = 200
    
    def solve(self, initial_paths: Dict[str, List], conflicts: List[EnhancedConflict], 
              current_time: float = 0) -> Dict[str, List]:
        """增强的ECBS求解 - 支持接口冲突"""
        start_time = time.time()
        
        if not conflicts:
            return initial_paths
        
        # 生成缓存键
        cache_key = self._generate_cache_key(initial_paths, conflicts)
        if cache_key in self.solution_cache:
            self.stats['cache_hits'] += 1
            return self.solution_cache[cache_key]
        
        # 按冲突类型和严重性排序
        sorted_conflicts = sorted(conflicts, reverse=True)
        
        # 使用改进的冲突选择策略
        primary_conflict = self._select_primary_conflict(sorted_conflicts, initial_paths)
        
        if not primary_conflict:
            return initial_paths
        
        print(f"ECBS求解主要冲突: {primary_conflict.conflict_type}, "
              f"严重性: {primary_conflict.severity:.2f}")
        
        # 根据冲突类型选择解决策略
        resolved_paths = self._resolve_conflict_by_type(
            initial_paths, primary_conflict, current_time
        )
        
        # 验证解决方案
        if self._validate_solution(resolved_paths, conflicts):
            # 缓存成功的解决方案
            self._cache_solution(cache_key, resolved_paths)
            
            self.stats['conflicts_resolved'] += 1
            if primary_conflict.conflict_type == 'interface':
                self.stats['interface_conflicts_resolved'] += 1
        else:
            # 如果解决方案无效，回退到原路径
            resolved_paths = initial_paths
        
        self.stats['total_time'] += time.time() - start_time
        self.stats['nodes_expanded'] += 1
        
        return resolved_paths
    
    def _select_primary_conflict(self, conflicts: List[EnhancedConflict], 
                                paths: Dict[str, List]) -> Optional[EnhancedConflict]:
        """智能冲突选择策略"""
        if not conflicts:
            return None
        
        # 优先级：接口冲突 > 骨干路径冲突 > 一般冲突
        interface_conflicts = [c for c in conflicts if c.conflict_type == 'interface']
        if interface_conflicts:
            return interface_conflicts[0]
        
        backbone_conflicts = [c for c in conflicts if c.conflict_type == 'backbone']
        if backbone_conflicts:
            return backbone_conflicts[0]
        
        # 选择影响最大的冲突
        best_conflict = None
        best_score = -1
        
        for conflict in conflicts[:5]:  # 只考虑前5个高优先级冲突
            score = self._calculate_conflict_impact(conflict, paths)
            if score > best_score:
                best_score = score
                best_conflict = conflict
        
        return best_conflict
    
    def _calculate_conflict_impact(self, conflict: EnhancedConflict, 
                                  paths: Dict[str, List]) -> float:
        """计算冲突影响度"""
        base_score = conflict.severity * conflict.confidence
        
        # 考虑涉及的路径长度
        path1_length = len(paths.get(conflict.agent1, []))
        path2_length = len(paths.get(conflict.agent2, []))
        length_factor = (path1_length + path2_length) / 200.0
        
        # 考虑冲突位置（路径中的位置）
        position_factor = 1.0 - (conflict.time_step / max(path1_length, path2_length, 1))
        
        return base_score * (1 + length_factor * 0.3 + position_factor * 0.2)
    
    def _resolve_conflict_by_type(self, paths: Dict[str, List], 
                                 conflict: EnhancedConflict, 
                                 current_time: float) -> Dict[str, List]:
        """根据冲突类型选择解决策略"""
        resolved_paths = paths.copy()
        
        if conflict.conflict_type == 'interface':
            resolved_paths = self._resolve_interface_conflict(resolved_paths, conflict, current_time)
        elif conflict.conflict_type == 'backbone':
            resolved_paths = self._resolve_backbone_conflict(resolved_paths, conflict)
        elif conflict.conflict_type in ['vertex', 'edge']:
            resolved_paths = self._resolve_spatial_conflict(resolved_paths, conflict)
        
        return resolved_paths
    
    def _resolve_interface_conflict(self, paths: Dict[str, List], 
                                   conflict: EnhancedConflict, 
                                   current_time: float) -> Dict[str, List]:
        """解决接口冲突"""
        # 获取冲突的两个智能体
        agent1, agent2 = conflict.agent1, conflict.agent2
        
        # 策略1: 时间错开
        if self._try_time_shift_resolution(agent1, agent2, conflict, current_time):
            return paths
        
        # 策略2: 替代接口
        alternative_path = self._find_alternative_interface_path(
            agent1, paths[agent1], conflict.interface_id
        )
        
        if alternative_path:
            paths[agent1] = alternative_path
            return paths
        
        # 策略3: 优先级重新规划
        lower_priority_agent = self._determine_lower_priority_agent(agent1, agent2)
        new_path = self._replan_with_interface_avoidance(
            lower_priority_agent, paths[lower_priority_agent], conflict.interface_id
        )
        
        if new_path:
            paths[lower_priority_agent] = new_path
        
        return paths
    
    def _resolve_backbone_conflict(self, paths: Dict[str, List], 
                                  conflict: EnhancedConflict) -> Dict[str, List]:
        """解决骨干路径冲突"""
        agent1, agent2 = conflict.agent1, conflict.agent2
        
        # 尝试为其中一个智能体寻找替代路径
        for agent in [agent1, agent2]:
            if agent not in paths:
                continue
                
            original_path = paths[agent]
            if len(original_path) < 2:
                continue
            
            start, goal = original_path[0], original_path[-1]
            
            # 尝试规划避开当前骨干路径的新路径
            alternative_path = self._plan_alternative_backbone_path(start, goal, agent)
            
            if alternative_path and self._is_path_better(alternative_path, original_path):
                paths[agent] = alternative_path
                self.stats['backbone_reroutings'] += 1
                break
        
        return paths
    
    def _resolve_spatial_conflict(self, paths: Dict[str, List], 
                                 conflict: EnhancedConflict) -> Dict[str, List]:
        """解决空间冲突（顶点/边冲突）"""
        # 选择重新规划的智能体
        agent_to_replan = self._select_agent_for_replanning(conflict, paths)
        
        if agent_to_replan not in paths:
            return paths
        
        original_path = paths[agent_to_replan]
        if len(original_path) < 2:
            return paths
        
        # 生成空间约束
        spatial_constraints = self._generate_spatial_constraints(conflict, paths)
        
        # 重新规划路径
        new_path = self._replan_with_constraints(
            agent_to_replan, original_path[0], original_path[-1], spatial_constraints
        )
        
        if new_path:
            paths[agent_to_replan] = new_path
        
        return paths
    
    def _try_time_shift_resolution(self, agent1: str, agent2: str, 
                                  conflict: EnhancedConflict, 
                                  current_time: float) -> bool:
        """尝试通过时间错开解决接口冲突"""
        if not conflict.interface_id:
            return False
        
        # 尝试延迟其中一个智能体
        delay_options = [5, 10, 15, 30]  # 延迟选项（秒）
        
        for delay in delay_options:
            # 尝试延迟agent1
            if self.interface_manager.reserve_interface(
                conflict.interface_id, agent1, 
                conflict.time_step + delay, priority=1
            ):
                return True
            
            # 尝试延迟agent2
            if self.interface_manager.reserve_interface(
                conflict.interface_id, agent2,
                conflict.time_step + delay, priority=1
            ):
                return True
        
        return False
    
    def _find_alternative_interface_path(self, agent: str, current_path: List, 
                                        blocked_interface: str) -> Optional[List]:
        """寻找使用替代接口的路径"""
        if not self.backbone_network or len(current_path) < 2:
            return None
        
        start, goal = current_path[0], current_path[-1]
        
        # 识别目标类型
        target_type, target_id = self.backbone_network.identify_target_point(goal)
        if not target_type:
            return None
        
        # 寻找替代接口
        alternative_interfaces = []
        for interface_id, interface in self.backbone_network.backbone_interfaces.items():
            if (interface_id != blocked_interface and 
                interface.backbone_path_id and
                interface.is_available()):
                
                # 检查是否通向同一目标
                backbone_path = self.backbone_network.backbone_paths.get(interface.backbone_path_id)
                if backbone_path:
                    end_point = backbone_path['end_point']
                    if (end_point['type'] == target_type and 
                        end_point['id'] == target_id):
                        
                        distance = math.sqrt(
                            (start[0] - interface.position[0])**2 + 
                            (start[1] - interface.position[1])**2
                        )
                        alternative_interfaces.append((interface, distance))
        
        if not alternative_interfaces:
            return None
        
        # 选择最近的替代接口
        best_interface = min(alternative_interfaces, key=lambda x: x[1])[0]
        
        # 使用替代接口规划路径
        try:
            result = self.backbone_network.get_complete_path_via_interface_enhanced(
                start, target_type, target_id, 
                rrt_hints={'preferred_interface': best_interface.interface_id}
            )
            
            if result and result[0]:
                return result[0]
        
        except Exception as e:
            print(f"替代接口路径规划失败: {e}")
        
        return None
    
    def _plan_alternative_backbone_path(self, start, goal, agent):
        """规划替代的骨干路径"""
        if not self.backbone_network:
            return None
        
        # 尝试使用不同的骨干路径
        target_type, target_id = self.backbone_network.identify_target_point(goal)
        if not target_type:
            return None
        
        # 获取所有可用路径
        available_paths = self.backbone_network.find_paths_to_target(target_type, target_id)
        
        for path_data in available_paths:
            try:
                # 尝试使用这条骨干路径
                result = self.backbone_network.get_path_from_position_to_target(
                    start, target_type, target_id
                )
                
                if result and result[0]:
                    return result[0]
            
            except Exception:
                continue
        
        return None
    
    def _is_path_better(self, new_path: List, old_path: List) -> bool:
        """比较路径质量"""
        if not new_path or not old_path:
            return bool(new_path)
        
        # 简单的长度比较
        new_length = sum(
            math.sqrt((new_path[i+1][0] - new_path[i][0])**2 + 
                     (new_path[i+1][1] - new_path[i][1])**2)
            for i in range(len(new_path) - 1)
        )
        
        old_length = sum(
            math.sqrt((old_path[i+1][0] - old_path[i][0])**2 + 
                     (old_path[i+1][1] - old_path[i][1])**2)
            for i in range(len(old_path) - 1)
        )
        
        # 新路径长度不超过旧路径的120%认为是更好的
        return new_length <= old_length * 1.2
    
    def _generate_cache_key(self, paths: Dict[str, List], 
                           conflicts: List[EnhancedConflict]) -> str:
        """生成解决方案缓存键"""
        # 简化的缓存键生成
        path_hash = hash(tuple(sorted(
            (agent, len(path)) for agent, path in paths.items()
        )))
        
        conflict_hash = hash(tuple(
            (c.agent1, c.agent2, c.conflict_type, round(c.severity, 1))
            for c in conflicts[:3]  # 只考虑前3个冲突
        ))
        
        return f"{path_hash}_{conflict_hash}"
    
    def _cache_solution(self, cache_key: str, solution: Dict[str, List]):
        """缓存解决方案"""
        if len(self.solution_cache) >= self.cache_size_limit:
            # 移除最旧的缓存项
            oldest_key = next(iter(self.solution_cache))
            del self.solution_cache[oldest_key]
        
        self.solution_cache[cache_key] = solution.copy()
    
    def _validate_solution(self, paths: Dict[str, List], 
                          original_conflicts: List[EnhancedConflict]) -> bool:
        """验证解决方案的有效性"""
        # 基本验证：所有路径都存在且有效
        for agent, path in paths.items():
            if not path or len(path) < 1:
                return False
        
        # TODO: 可以添加更严格的冲突检查
        return True

class OptimizedTrafficManager:
    """优化的交通管理器 - 深度集成接口系统"""
    
    def __init__(self, env, backbone_network=None):
        self.env = env
        self.backbone_network = backbone_network
        
        # 核心组件
        self.ecbs_solver = EnhancedECBSSolver(env, backbone_network)
        self.interface_manager = self.ecbs_solver.interface_manager
        
        # 路径管理
        self.active_paths = {}  # {vehicle_id: path_info}
        self.path_reservations = {}  # 时空预留表
        self.path_history = defaultdict(list)  # 路径历史
        
        # 冲突检测参数优化
        self.safety_distance = 6.0  # 减小安全距离提高通行效率
        self.time_discretization = 1.5
        self.prediction_horizon = 50.0  # 预测时间窗口
        
        # 性能优化
        self.parallel_detection = True
        self.max_detection_threads = 4
        self.conflict_cache = {}
        self.cache_expiry_time = 10.0  # 冲突缓存过期时间
        
        # 统计信息
        self.stats = {
            'conflicts_detected': 0,
            'conflicts_resolved': 0,
            'ecbs_calls': 0,
            'resolution_time': 0,
            'backbone_conflicts': 0,
            'interface_conflicts': 0,
            'cache_hits': 0,
            'performance_metrics': {
                'avg_detection_time': 0,
                'avg_resolution_time': 0,
                'peak_vehicle_count': 0
            }
        }
        
        print("初始化优化的交通管理器")
    
    def set_backbone_network(self, backbone_network):
        """设置骨干路径网络"""
        self.backbone_network = backbone_network
        self.ecbs_solver.backbone_network = backbone_network
        print("已更新骨干路径网络引用")
    def register_vehicle_path(self, vehicle_id: str, path: List, 
                             start_time: float = 0, speed: float = 1.0) -> bool:
        """兼容性方法 - 调用增强版本"""
        return self.register_vehicle_path_enhanced(
            vehicle_id, path, None, start_time, speed
        )    
    def register_vehicle_path_enhanced(self, vehicle_id: str, path: List, 
                                     path_structure: Dict = None, 
                                     start_time: float = 0, speed: float = 1.0) -> bool:
        """增强的车辆路径注册"""
        if not path or len(path) < 2:
            return False
        
        # 移除旧路径
        self.release_vehicle_path(vehicle_id)
        
        # 准备路径信息
        path_info = {
            'path': path,
            'structure': path_structure or {},
            'start_time': start_time,
            'speed': speed,
            'registered_time': time.time(),
            'quality_score': path_structure.get('final_quality', 0.5) if path_structure else 0.5
        }
        
        # 处理接口预约
        if path_structure and path_structure.get('interface_id'):
            interface_id = path_structure['interface_id']
            access_time = start_time + path_structure.get('access_length', 0) * 2  # 估算到达时间
            
            success = self.interface_manager.reserve_interface(
                interface_id, vehicle_id, access_time, duration=120, priority=1
            )
            
            if not success:
                print(f"警告: 车辆 {vehicle_id} 接口 {interface_id} 预约失败")
        
        # 注册路径
        self.active_paths[vehicle_id] = path_info
        self._add_path_reservations(vehicle_id, path, start_time, speed)
        
        # 记录路径历史
        self.path_history[vehicle_id].append({
            'timestamp': time.time(),
            'path_length': len(path),
            'structure_type': path_structure.get('type', 'unknown') if path_structure else 'unknown',
            'quality': path_info['quality_score']
        })
        
        # 限制历史记录长度
        if len(self.path_history[vehicle_id]) > 50:
            self.path_history[vehicle_id] = self.path_history[vehicle_id][-50:]
        
        return True
    def check_path_conflicts(self, vehicle_id: str, path: List) -> bool:
        """检查路径是否有冲突"""
        if not path or len(path) < 2:
            return False
        
        # 临时注册路径进行冲突检测
        temp_path_info = {
            'path': path,
            'structure': {},
            'start_time': time.time(),
            'speed': 1.0,
            'registered_time': time.time(),
            'quality_score': 0.5
        }
        
        # 备份当前路径
        original_path = self.active_paths.get(vehicle_id)
        
        # 临时设置新路径
        self.active_paths[vehicle_id] = temp_path_info
        
        # 检测冲突
        conflicts = self.detect_all_conflicts()
        
        # 恢复原路径
        if original_path:
            self.active_paths[vehicle_id] = original_path
        else:
            del self.active_paths[vehicle_id]
        
        # 检查是否有涉及该车辆的冲突
        for conflict in conflicts:
            if conflict.agent1 == vehicle_id or conflict.agent2 == vehicle_id:
                return True
        
        return False    
    def detect_all_conflicts(self, current_time: float = 0) -> List[EnhancedConflict]:
        """综合冲突检测"""
        detection_start = time.time()
        all_conflicts = []
        
        # 1. 空间路径冲突
        spatial_conflicts = self._detect_spatial_conflicts()
        all_conflicts.extend(spatial_conflicts)
        
        # 2. 接口时间冲突
        interface_conflicts = self.interface_manager.get_interface_conflicts(current_time)
        all_conflicts.extend(interface_conflicts)
        
        # 3. 骨干路径访问冲突
        backbone_conflicts = self._detect_backbone_conflicts()
        all_conflicts.extend(backbone_conflicts)
        
        # 4. 预测性冲突
        predictive_conflicts = self._detect_predictive_conflicts(current_time)
        all_conflicts.extend(predictive_conflicts)
        
        # 更新统计信息
        detection_time = time.time() - detection_start
        self.stats['conflicts_detected'] += len(all_conflicts)
        self.stats['performance_metrics']['avg_detection_time'] = (
            self.stats['performance_metrics']['avg_detection_time'] * 0.9 + 
            detection_time * 0.1
        )
        
        # 去重和排序
        unique_conflicts = self._deduplicate_conflicts(all_conflicts)
        return sorted(unique_conflicts, reverse=True)
    
    def _detect_spatial_conflicts(self) -> List[EnhancedConflict]:
        """空间冲突检测"""
        conflicts = []
        
        if len(self.active_paths) < 2:
            return conflicts
        
        # 并行检测优化
        if self.parallel_detection and len(self.active_paths) > 4:
            conflicts = self._parallel_spatial_detection()
        else:
            conflicts = self._sequential_spatial_detection()
        
        return conflicts
    def suggest_path_adjustment(self, vehicle_id: str, start: Tuple, 
                              goal: Tuple) -> Optional[List]:
        """建议路径调整"""
        # 这是一个简化实现，实际应该更复杂
        if not self.backbone_network:
            return None
        
        # 尝试使用骨干网络获取替代路径
        try:
            target_type, target_id = self.backbone_network.identify_target_point(goal)
            if target_type:
                result = self.backbone_network.get_path_from_position_to_target(
                    start, target_type, target_id
                )
                if result and result[0]:
                    return result[0]
        except Exception as e:
            print(f"路径调整建议失败: {e}")
        
        return None
    
    def get_save_data(self) -> Dict:
        """获取保存数据"""
        return {
            'safety_distance': self.safety_distance,
            'time_discretization': self.time_discretization,
            'prediction_horizon': self.prediction_horizon,
            'stats': self.stats.copy(),
            'active_vehicles': len(self.active_paths),
            'total_reservations': len(self.path_reservations),
            'interface_reservations': len(self.interface_manager.reservations),
            'ecbs_settings': {
                'suboptimality_bound': getattr(self.ecbs_solver, 'suboptimality_bound', 1.3),
                'max_search_time': getattr(self.ecbs_solver, 'max_search_time', 20.0),
                'max_nodes_expanded': getattr(self.ecbs_solver, 'max_nodes_expanded', 800)
            }
        }
    
    def restore_from_save_data(self, save_data: Dict):
        """从保存数据恢复状态"""
        try:
            # 恢复基本设置
            self.safety_distance = save_data.get('safety_distance', 6.0)
            self.time_discretization = save_data.get('time_discretization', 1.5)
            self.prediction_horizon = save_data.get('prediction_horizon', 50.0)
            
            # 恢复统计信息
            if 'stats' in save_data:
                self.stats.update(save_data['stats'])
            
            # 恢复ECBS设置
            ecbs_settings = save_data.get('ecbs_settings', {})
            if ecbs_settings and self.ecbs_solver:
                self.ecbs_solver.suboptimality_bound = ecbs_settings.get('suboptimality_bound', 1.3)
                self.ecbs_solver.max_search_time = ecbs_settings.get('max_search_time', 20.0)
                self.ecbs_solver.max_nodes_expanded = ecbs_settings.get('max_nodes_expanded', 800)
            
            print("交通管理器状态已恢复")
            
        except Exception as e:
            print(f"恢复交通管理器状态失败: {e}")
    
    def clear_all_paths(self):
        """清除所有路径（重置时使用）"""
        self.active_paths.clear()
        self.path_reservations.clear()
        
        # 清除接口预约
        with self.interface_manager.reservation_lock:
            self.interface_manager.reservations.clear()
        
        print("已清除所有交通管理数据")
    
    def update_vehicle_priority(self, vehicle_id: str, priority: int):
        """更新车辆优先级"""
        if vehicle_id in self.active_paths:
            path_info = self.active_paths[vehicle_id]
            path_info['priority'] = priority
    
    def get_vehicle_conflicts(self, vehicle_id: str) -> List[EnhancedConflict]:
        """获取特定车辆的冲突"""
        all_conflicts = self.detect_all_conflicts()
        vehicle_conflicts = []
        
        for conflict in all_conflicts:
            if conflict.agent1 == vehicle_id or conflict.agent2 == vehicle_id:
                vehicle_conflicts.append(conflict)
        
        return vehicle_conflicts
    
    def force_replan_vehicle(self, vehicle_id: str) -> bool:
        """强制重新规划车辆路径"""
        if vehicle_id not in self.active_paths:
            return False
        
        # 触发重新规划的逻辑
        path_info = self.active_paths[vehicle_id]
        path_info['needs_replan'] = True
        path_info['replan_reason'] = 'manual_force'
        
        return True    
    def _parallel_spatial_detection(self) -> List[EnhancedConflict]:
        """并行空间冲突检测"""
        conflicts = []
        vehicle_pairs = []
        
        vehicles = list(self.active_paths.keys())
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                vehicle_pairs.append((vehicles[i], vehicles[j]))
        
        # 分批处理
        batch_size = max(1, len(vehicle_pairs) // self.max_detection_threads)
        batches = [vehicle_pairs[i:i + batch_size] 
                  for i in range(0, len(vehicle_pairs), batch_size)]
        
        with ThreadPoolExecutor(max_workers=self.max_detection_threads) as executor:
            future_to_batch = {
                executor.submit(self._detect_batch_conflicts, batch): batch
                for batch in batches
            }
            
            for future in future_to_batch:
                try:
                    batch_conflicts = future.result(timeout=5.0)
                    conflicts.extend(batch_conflicts)
                except Exception as e:
                    print(f"并行冲突检测批次失败: {e}")
        
        return conflicts
    
    def _detect_batch_conflicts(self, vehicle_pairs: List[Tuple[str, str]]) -> List[EnhancedConflict]:
        """检测一批车辆对的冲突"""
        conflicts = []
        
        for vehicle1, vehicle2 in vehicle_pairs:
            if vehicle1 not in self.active_paths or vehicle2 not in self.active_paths:
                continue
            
            path1 = self.active_paths[vehicle1]['path']
            path2 = self.active_paths[vehicle2]['path']
            
            # 顶点冲突
            vertex_conflicts = self._detect_vertex_conflicts(vehicle1, path1, vehicle2, path2)
            conflicts.extend(vertex_conflicts)
            
            # 边冲突
            edge_conflicts = self._detect_edge_conflicts(vehicle1, path1, vehicle2, path2)
            conflicts.extend(edge_conflicts)
        
        return conflicts
    
    def _sequential_spatial_detection(self) -> List[EnhancedConflict]:
        """顺序空间冲突检测"""
        conflicts = []
        vehicles = list(self.active_paths.keys())
        
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                vehicle1, vehicle2 = vehicles[i], vehicles[j]
                path1 = self.active_paths[vehicle1]['path']
                path2 = self.active_paths[vehicle2]['path']
                
                # 顶点冲突
                vertex_conflicts = self._detect_vertex_conflicts(vehicle1, path1, vehicle2, path2)
                conflicts.extend(vertex_conflicts)
                
                # 边冲突  
                edge_conflicts = self._detect_edge_conflicts(vehicle1, path1, vehicle2, path2)
                conflicts.extend(edge_conflicts)
        
        return conflicts
    
    def _detect_backbone_conflicts(self) -> List[EnhancedConflict]:
        """骨干路径访问冲突检测"""
        conflicts = []
        
        if not self.backbone_network:
            return conflicts
        
        # 按骨干路径分组车辆
        backbone_usage = defaultdict(list)
        
        for vehicle_id, path_info in self.active_paths.items():
            structure = path_info.get('structure', {})
            backbone_path_id = structure.get('backbone_path_id')
            
            if backbone_path_id:
                backbone_usage[backbone_path_id].append({
                    'vehicle_id': vehicle_id,
                    'start_time': path_info['start_time'],
                    'access_length': structure.get('access_length', 0),
                    'backbone_length': structure.get('backbone_length', 0)
                })
        
        # 检查每条骨干路径的使用冲突
        for backbone_path_id, users in backbone_usage.items():
            if len(users) > 1:
                backbone_conflicts = self._analyze_backbone_usage_conflicts(
                    backbone_path_id, users
                )
                conflicts.extend(backbone_conflicts)
                self.stats['backbone_conflicts'] += len(backbone_conflicts)
        
        return conflicts
    
    def _analyze_backbone_usage_conflicts(self, backbone_path_id: str, 
                                         users: List[Dict]) -> List[EnhancedConflict]:
        """分析骨干路径使用冲突"""
        conflicts = []
        
        # 按进入时间排序
        users.sort(key=lambda u: u['start_time'] + u['access_length'])
        
        for i in range(len(users) - 1):
            current_user = users[i]
            next_user = users[i + 1]
            
            # 计算时间重叠
            current_end = (current_user['start_time'] + current_user['access_length'] + 
                          current_user['backbone_length'])
            next_start = next_user['start_time'] + next_user['access_length']
            
            if next_start < current_end:
                # 有时间重叠，产生冲突
                conflict = EnhancedConflict(
                    agent1=current_user['vehicle_id'],
                    agent2=next_user['vehicle_id'],
                    location=(0, 0),  # 骨干路径位置
                    time_step=next_start,
                    conflict_type='backbone',
                    severity=1.5,
                    confidence=0.8,
                    resolution_hints={
                        'backbone_path_id': backbone_path_id,
                        'time_overlap': current_end - next_start,
                        'can_delay': True
                    }
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def _detect_predictive_conflicts(self, current_time: float) -> List[EnhancedConflict]:
        """预测性冲突检测"""
        conflicts = []
        
        # 基于车辆运动趋势预测未来可能的冲突
        future_positions = {}
        
        for vehicle_id, path_info in self.active_paths.items():
            path = path_info['path']
            speed = path_info['speed']
            start_time = path_info['start_time']
            
            # 预测未来位置
            prediction_time = current_time + self.prediction_horizon
            predicted_position = self._predict_vehicle_position(
                path, speed, start_time, prediction_time
            )
            
            if predicted_position:
                future_positions[vehicle_id] = predicted_position
        
        # 检查预测位置的冲突
        vehicles = list(future_positions.keys())
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                vehicle1, vehicle2 = vehicles[i], vehicles[j]
                pos1 = future_positions[vehicle1]
                pos2 = future_positions[vehicle2]
                
                distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                
                if distance < self.safety_distance * 1.5:  # 预测冲突使用更大的安全距离
                    conflict = EnhancedConflict(
                        agent1=vehicle1,
                        agent2=vehicle2,
                        location=((pos1[0] + pos2[0])/2, (pos1[1] + pos2[1])/2),
                        time_step=current_time + self.prediction_horizon,
                        conflict_type='predictive',
                        severity=1.0,
                        confidence=0.6,  # 预测冲突置信度较低
                        resolution_hints={'is_prediction': True}
                    )
                    conflicts.append(conflict)
        
        return conflicts
    
    def _predict_vehicle_position(self, path: List, speed: float, 
                                 start_time: float, target_time: float) -> Optional[Tuple]:
        """预测车辆在指定时间的位置"""
        if not path or target_time <= start_time:
            return None
        
        elapsed_time = target_time - start_time
        distance_traveled = speed * elapsed_time
        
        # 沿路径累积距离查找位置
        current_distance = 0
        
        for i in range(len(path) - 1):
            segment_length = math.sqrt(
                (path[i+1][0] - path[i][0])**2 + 
                (path[i+1][1] - path[i][1])**2
            )
            
            if current_distance + segment_length >= distance_traveled:
                # 在这个线段上
                ratio = (distance_traveled - current_distance) / segment_length
                x = path[i][0] + ratio * (path[i+1][0] - path[i][0])
                y = path[i][1] + ratio * (path[i+1][1] - path[i][1])
                return (x, y, 0)
            
            current_distance += segment_length
        
        # 超出路径长度，返回终点
        return path[-1] if path else None
    
    def resolve_conflicts_enhanced(self, conflicts: List[EnhancedConflict], 
                                  current_time: float = 0) -> Dict[str, List]:
        """增强的冲突解决"""
        if not conflicts:
            return {vid: info['path'] for vid, info in self.active_paths.items()}
        
        start_time = time.time()
        
        # 准备当前路径
        current_paths = {vid: info['path'] for vid, info in self.active_paths.items()}
        
        # 使用增强ECBS求解器
        self.stats['ecbs_calls'] += 1
        resolved_paths = self.ecbs_solver.solve(current_paths, conflicts, current_time)
        
        # 更新统计信息
        resolution_time = time.time() - start_time
        self.stats['resolution_time'] += resolution_time
        self.stats['performance_metrics']['avg_resolution_time'] = (
            self.stats['performance_metrics']['avg_resolution_time'] * 0.9 + 
            resolution_time * 0.1
        )
        
        if resolved_paths != current_paths:
            self.stats['conflicts_resolved'] += len(conflicts)
            print(f"ECBS解决了 {len(conflicts)} 个冲突，耗时 {resolution_time:.2f}s")
        
        return resolved_paths
    
    def _detect_vertex_conflicts(self, agent1: str, path1: List, 
                                agent2: str, path2: List) -> List[EnhancedConflict]:
        """检测顶点冲突"""
        conflicts = []
        min_len = min(len(path1), len(path2))
        
        for t in range(min_len):
            p1, p2 = path1[t], path2[t]
            distance = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            
            if distance < self.safety_distance:
                # 计算冲突严重性
                severity = self.safety_distance / (distance + 0.1)
                
                # 检查是否在关键区域
                is_critical = self._is_critical_location((p1[0] + p2[0])/2, (p1[1] + p2[1])/2)
                if is_critical:
                    severity *= 1.5
                
                conflict = EnhancedConflict(
                    agent1=agent1,
                    agent2=agent2,
                    location=((p1[0] + p2[0])/2, (p1[1] + p2[1])/2),
                    time_step=float(t),
                    conflict_type='vertex',
                    severity=severity,
                    confidence=0.9,
                    resolution_hints={
                        'distance': distance,
                        'safety_margin': self.safety_distance - distance,
                        'is_critical': is_critical
                    }
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def _detect_edge_conflicts(self, agent1: str, path1: List, 
                              agent2: str, path2: List) -> List[EnhancedConflict]:
        """检测边冲突"""
        conflicts = []
        min_len = min(len(path1), len(path2)) - 1
        
        for t in range(min_len):
            # 检查交换位置的情况
            if (self._points_close(path1[t], path2[t+1]) and 
                self._points_close(path1[t+1], path2[t])):
                
                conflict = EnhancedConflict(
                    agent1=agent1,
                    agent2=agent2,
                    location=((path1[t][0] + path2[t][0])/2, (path1[t][1] + path2[t][1])/2),
                    time_step=float(t + 0.5),
                    conflict_type='edge',
                    severity=1.8,  # 边冲突通常更严重
                    confidence=0.95,
                    resolution_hints={
                        'swap_detected': True,
                        'alternative_timing': True
                    }
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def _points_close(self, p1: Tuple, p2: Tuple, threshold: float = None) -> bool:
        """检查两点是否接近"""
        if threshold is None:
            threshold = self.safety_distance
        
        distance = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        return distance < threshold
    
    def _is_critical_location(self, x: float, y: float) -> bool:
        """检查是否为关键位置（如装卸点附近）"""
        if not self.env:
            return False
        
        critical_threshold = 15.0
        
        # 检查装载点
        for point in getattr(self.env, 'loading_points', []):
            if math.sqrt((x - point[0])**2 + (y - point[1])**2) < critical_threshold:
                return True
        
        # 检查卸载点
        for point in getattr(self.env, 'unloading_points', []):
            if math.sqrt((x - point[0])**2 + (y - point[1])**2) < critical_threshold:
                return True
        
        return False
    
    def _deduplicate_conflicts(self, conflicts: List[EnhancedConflict]) -> List[EnhancedConflict]:
        """去重冲突"""
        unique_conflicts = []
        seen_pairs = set()
        
        for conflict in conflicts:
            # 创建标准化的智能体对
            pair = tuple(sorted([conflict.agent1, conflict.agent2]))
            key = (pair, conflict.conflict_type, round(conflict.time_step))
            
            if key not in seen_pairs:
                seen_pairs.add(key)
                unique_conflicts.append(conflict)
        
        return unique_conflicts
    
    def release_vehicle_path(self, vehicle_id: str) -> bool:
        """释放车辆路径和相关资源"""
        if vehicle_id not in self.active_paths:
            return False
        
        path_info = self.active_paths[vehicle_id]
        
        # 释放接口预约
        structure = path_info.get('structure', {})
        interface_id = structure.get('interface_id')
        if interface_id:
            self.interface_manager.release_interface(interface_id, vehicle_id)
        
        # 移除路径预留
        self._remove_path_reservations(vehicle_id)
        
        # 移除活动路径
        del self.active_paths[vehicle_id]
        
        return True
    
    def get_comprehensive_stats(self) -> Dict:
        """获取综合统计信息"""
        stats = self.stats.copy()
        
        # 添加ECBS统计
        ecbs_stats = self.ecbs_solver.stats
        stats['ecbs_solver'] = ecbs_stats
        
        # 添加接口管理统计
        interface_stats = {
            'total_reservations': sum(len(reservations) 
                                    for reservations in self.interface_manager.reservations.values()),
            'active_interfaces': len([
                interface_id for interface_id, reservations in self.interface_manager.reservations.items()
                if reservations
            ])
        }
        stats['interface_management'] = interface_stats
        
        # 计算效率指标
        total_conflicts = stats['conflicts_detected']
        if total_conflicts > 0:
            stats['resolution_efficiency'] = stats['conflicts_resolved'] / total_conflicts
        else:
            stats['resolution_efficiency'] = 1.0
        
        # 更新峰值车辆数
        current_vehicle_count = len(self.active_paths)
        stats['performance_metrics']['peak_vehicle_count'] = max(
            stats['performance_metrics']['peak_vehicle_count'],
            current_vehicle_count
        )
        
        return stats
    
    def _add_path_reservations(self, vehicle_id: str, path: List, 
                              start_time: float, speed: float):
        """添加路径到时空预留表"""
        current_time = start_time
        
        for i in range(len(path) - 1):
            segment_length = math.sqrt(
                (path[i+1][0] - path[i][0])**2 + 
                (path[i+1][1] - path[i][1])**2
            )
            segment_time = segment_length / speed
            
            # 离散化时间
            time_slot = int(current_time / self.time_discretization)
            
            # 添加预留
            key = (int(path[i][0]), int(path[i][1]), time_slot)
            
            if key not in self.path_reservations:
                self.path_reservations[key] = []
            
            if vehicle_id not in self.path_reservations[key]:
                self.path_reservations[key].append(vehicle_id)
            
            current_time += segment_time
    
    def _remove_path_reservations(self, vehicle_id: str):
        """从预留表中移除车辆路径"""
        keys_to_remove = []
        
        for key, vehicles in self.path_reservations.items():
            if vehicle_id in vehicles:
                vehicles.remove(vehicle_id)
                
                if not vehicles:
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.path_reservations[key]
    def _select_agent_for_replanning(self, conflict, paths):
        """选择需要重新规划的智能体"""
        agent1, agent2 = conflict.agent1, conflict.agent2
        
        # 优先重新规划优先级较低的智能体
        if agent1 in paths and agent2 in paths:
            path1_length = len(paths[agent1])
            path2_length = len(paths[agent2])
            
            # 选择路径较短的智能体重新规划（影响较小）
            if path1_length <= path2_length:
                return agent1
            else:
                return agent2
        
        return agent1 if agent1 in paths else agent2
    
    def _generate_spatial_constraints(self, conflict, paths):
        """生成空间约束"""
        constraints = []
        
        if conflict.conflict_type == 'vertex':
            # 顶点约束：在特定时间不能占用特定位置
            constraint = SmartConstraint(
                agent_id=conflict.agent1,
                constraint_type='vertex',
                location=conflict.location,
                time_step=conflict.time_step,
                priority=2
            )
            constraints.append(constraint)
        
        elif conflict.conflict_type == 'edge':
            # 边约束：在特定时间不能使用特定边
            constraint = SmartConstraint(
                agent_id=conflict.agent1,
                constraint_type='edge',
                location=conflict.location,
                time_step=conflict.time_step,
                priority=2
            )
            constraints.append(constraint)
        
        return constraints
    
    def _replan_with_constraints(self, agent_id, start, goal, constraints):
        """在约束条件下重新规划路径"""
        if not self.path_planner:
            return None
        
        try:
            # 简化实现：多次尝试规划，每次增加迭代次数
            for attempt in range(3):
                max_iterations = 3000 + attempt * 1000
                
                # 使用路径规划器重新规划
                result = self.path_planner.plan_path(
                    agent_id, start, goal, use_backbone=True, check_conflicts=False
                )
                
                if result:
                    if isinstance(result, tuple):
                        path, structure = result
                        return path
                    else:
                        return result
        
        except Exception as e:
            print(f"约束重新规划失败: {e}")
        
        return None
    
    def _determine_lower_priority_agent(self, agent1, agent2):
        """确定较低优先级的智能体"""
        # 简化实现：基于ID排序
        return agent1 if agent1 < agent2 else agent2
    
    def _replan_with_interface_avoidance(self, agent, current_path, blocked_interface):
        """规划避开特定接口的路径"""
        if not current_path or len(current_path) < 2:
            return None
        
        start, goal = current_path[0], current_path[-1]
        
        # 使用路径规划器重新规划（简化实现）
        if hasattr(self, 'path_planner') and self.path_planner:
            try:
                result = self.path_planner.plan_path(
                    agent, start, goal, use_backbone=True, check_conflicts=False
                )
                
                if result:
                    return result[0] if isinstance(result, tuple) else result
            
            except Exception as e:
                print(f"接口回避规划失败: {e}")
        
        return None
    

# 向后兼容性
TrafficManager = OptimizedTrafficManager