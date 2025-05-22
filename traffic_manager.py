import math
import heapq
import time
import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass

@dataclass
class EnhancedConflict:
    """增强的冲突表示"""
    agent1: str
    agent2: str
    location: Tuple[float, float]
    time_step: float
    conflict_type: str
    severity: float = 1.0  # 冲突严重程度
    resolution_cost: float = 0.0  # 解决成本
    priority: int = 1  # 优先级
    
    def __lt__(self, other):
        # 用于优先队列排序：优先级高的先处理，同优先级按严重程度排序
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.severity > other.severity

@dataclass
class EnhancedConstraint:
    """增强的约束条件"""
    agent_id: str
    location: Tuple[float, float]
    time_step: float
    constraint_type: str
    cost: float = 1.0  # 约束成本
    flexibility: float = 0.0  # 约束灵活性
    
class ConflictDetectionManager:
    """分离的冲突检测管理器"""
    
    def __init__(self, safety_rectangle=None):
        self.safety_rectangle = safety_rectangle or SafetyRectangle()
        
        # 专门的冲突检测器
        self.detectors = {
            'vertex': VertexConflictDetector(self.safety_rectangle),
            'edge': EdgeConflictDetector(self.safety_rectangle),
            'following': FollowingConflictDetector(self.safety_rectangle),
            'connection': ConnectionConflictDetector(self.safety_rectangle),
            'backbone': BackboneConflictDetector(self.safety_rectangle)
        }
        
        # 检测优化
        self.spatial_hash = {}  # 空间哈希表用于加速冲突检测
        self.detection_cache = {}  # 冲突检测缓存
        
    def detect_conflicts(self, paths, time_horizon=100.0):
        """集成多种冲突检测器检测冲突"""
        all_conflicts = []
        
        # 构建空间-时间索引
        self._build_spatial_temporal_index(paths, time_horizon)
        
        # 使用每种检测器
        for detector_type, detector in self.detectors.items():
            try:
                conflicts = detector.detect(paths, self.spatial_hash)
                
                # 为冲突添加类型标记和严重程度评估
                for conflict in conflicts:
                    conflict.conflict_type = detector_type
                    conflict.severity = self._evaluate_conflict_severity(conflict, paths)
                    conflict.resolution_cost = self._estimate_resolution_cost(conflict, paths)
                
                all_conflicts.extend(conflicts)
                
            except Exception as e:
                print(f"冲突检测器 {detector_type} 出错: {e}")
                continue
        
        # 对冲突进行分类和优先级排序
        return self._prioritize_conflicts(all_conflicts)
    
    def _build_spatial_temporal_index(self, paths, time_horizon):
        """构建空间-时间索引"""
        self.spatial_hash = defaultdict(list)  # {(grid_x, grid_y, time_slot): [(agent, path_index)]}
        
        grid_size = 5.0  # 网格大小
        time_slot_size = 2.0  # 时间片大小
        
        for agent_id, path in paths.items():
            if not path:
                continue
                
            current_time = 0.0
            
            for i in range(len(path) - 1):
                # 计算段长度和时间
                segment_length = self._calculate_distance(path[i], path[i+1])
                segment_time = segment_length / 1.0  # 假设速度为1
                
                # 在时间段内添加到空间哈希
                start_time = current_time
                end_time = current_time + segment_time
                
                # 计算网格坐标
                grid_x = int(path[i][0] // grid_size)
                grid_y = int(path[i][1] // grid_size)
                
                # 添加到所有相关时间片
                start_slot = int(start_time // time_slot_size)
                end_slot = int(end_time // time_slot_size)
                
                for time_slot in range(start_slot, end_slot + 1):
                    self.spatial_hash[(grid_x, grid_y, time_slot)].append((agent_id, i))
                
                current_time += segment_time
                
                if current_time > time_horizon:
                    break
    
    def _evaluate_conflict_severity(self, conflict, paths):
        """评估冲突严重程度"""
        base_severity = 1.0
        
        # 基于距离的严重程度
        if hasattr(conflict, 'location') and conflict.location:
            # 距离越近，严重程度越高
            distance_factor = 1.0  # 可以根据实际距离调整
            base_severity *= distance_factor
        
        # 基于时间的严重程度
        if hasattr(conflict, 'time_step'):
            # 时间越早，严重程度越高
            time_factor = max(0.5, 1.0 - conflict.time_step / 100.0)
            base_severity *= time_factor
        
        # 基于冲突类型的严重程度
        type_multipliers = {
            'vertex': 1.5,     # 顶点冲突最严重
            'edge': 1.2,       # 边冲突次之
            'following': 1.0,  # 跟随冲突中等
            'connection': 1.3, # 连接点冲突较严重
            'backbone': 1.4    # 骨干路径冲突严重
        }
        
        type_multiplier = type_multipliers.get(conflict.conflict_type, 1.0)
        base_severity *= type_multiplier
        
        return min(5.0, base_severity)  # 限制最大严重程度
    
    def _estimate_resolution_cost(self, conflict, paths):
        """估计冲突解决成本"""
        base_cost = 1.0
        
        # 基于涉及的路径长度
        if conflict.agent1 in paths and conflict.agent2 in paths:
            path1_length = self._calculate_path_length(paths[conflict.agent1])
            path2_length = self._calculate_path_length(paths[conflict.agent2])
            
            # 路径越长，重新规划成本越高
            length_factor = (path1_length + path2_length) / 100.0
            base_cost *= (1.0 + length_factor)
        
        # 基于冲突类型的解决难度
        type_costs = {
            'vertex': 2.0,
            'edge': 1.5,
            'following': 1.0,
            'connection': 2.5,
            'backbone': 3.0
        }
        
        type_cost = type_costs.get(conflict.conflict_type, 1.0)
        base_cost *= type_cost
        
        return base_cost
    
    def _prioritize_conflicts(self, conflicts):
        """对冲突进行优先级排序"""
        # 按严重程度和解决成本排序
        prioritized = sorted(
            conflicts,
            key=lambda c: (-c.severity, c.resolution_cost, c.time_step)
        )
        
        # 为冲突分配优先级
        for i, conflict in enumerate(prioritized):
            if conflict.severity >= 2.0:
                conflict.priority = 1  # 高优先级
            elif conflict.severity >= 1.0:
                conflict.priority = 2  # 中优先级
            else:
                conflict.priority = 3  # 低优先级
        
        return prioritized
    
    def _calculate_distance(self, p1, p2):
        """计算两点间距离"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _calculate_path_length(self, path):
        """计算路径长度"""
        if not path or len(path) < 2:
            return 0
        
        length = 0
        for i in range(len(path) - 1):
            length += self._calculate_distance(path[i], path[i+1])
        
        return length

class VertexConflictDetector:
    """顶点冲突检测器"""
    
    def __init__(self, safety_rectangle):
        self.safety_rectangle = safety_rectangle
    
    def detect(self, paths, spatial_hash=None):
        """检测顶点冲突"""
        conflicts = []
        
        if spatial_hash:
            # 使用空间哈希加速检测
            conflicts.extend(self._detect_with_spatial_hash(paths, spatial_hash))
        else:
            # 传统检测方法
            conflicts.extend(self._detect_traditional(paths))
        
        return conflicts
    
    def _detect_with_spatial_hash(self, paths, spatial_hash):
        """使用空间哈希检测顶点冲突"""
        conflicts = []
        
        for key, agents_in_cell in spatial_hash.items():
            if len(agents_in_cell) < 2:
                continue
            
            # 检查同一时空格子中的代理
            for i in range(len(agents_in_cell)):
                for j in range(i + 1, len(agents_in_cell)):
                    agent1, idx1 = agents_in_cell[i]
                    agent2, idx2 = agents_in_cell[j]
                    
                    if agent1 == agent2:
                        continue
                    
                    # 详细冲突检查
                    conflict = self._check_detailed_vertex_conflict(
                        agent1, paths[agent1], idx1,
                        agent2, paths[agent2], idx2
                    )
                    
                    if conflict:
                        conflicts.append(conflict)
        
        return conflicts
    
    def _detect_traditional(self, paths):
        """传统顶点冲突检测"""
        conflicts = []
        agents = list(paths.keys())
        
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                agent1, agent2 = agents[i], agents[j]
                path1, path2 = paths[agent1], paths[agent2]
                
                conflict = self._check_path_vertex_conflicts(agent1, path1, agent2, path2)
                if conflict:
                    conflicts.append(conflict)
        
        return conflicts
    
    def _check_detailed_vertex_conflict(self, agent1, path1, idx1, agent2, path2, idx2):
        """详细检查顶点冲突"""
        if not path1 or not path2 or idx1 >= len(path1) or idx2 >= len(path2):
            return None
        
        p1 = path1[idx1]
        p2 = path2[idx2]
        
        # 计算距离
        distance = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        # 如果距离小于安全距离
        if distance < self.safety_rectangle.diagonal / 2:
            return EnhancedConflict(
                agent1=agent1,
                agent2=agent2,
                location=((p1[0] + p2[0])/2, (p1[1] + p2[1])/2),
                time_step=float(max(idx1, idx2)),
                conflict_type='vertex'
            )
        
        return None
    
    def _check_path_vertex_conflicts(self, agent1, path1, agent2, path2):
        """检查两条路径的顶点冲突"""
        min_path_len = min(len(path1), len(path2))
        
        for t in range(min_path_len):
            p1 = path1[t]
            p2 = path2[t]
            
            distance = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            
            if distance < self.safety_rectangle.diagonal / 2:
                return EnhancedConflict(
                    agent1=agent1,
                    agent2=agent2,
                    location=((p1[0] + p2[0])/2, (p1[1] + p2[1])/2),
                    time_step=float(t),
                    conflict_type='vertex'
                )
        
        return None

class EdgeConflictDetector:
    """边冲突检测器"""
    
    def __init__(self, safety_rectangle):
        self.safety_rectangle = safety_rectangle
    
    def detect(self, paths, spatial_hash=None):
        """检测边冲突"""
        conflicts = []
        agents = list(paths.keys())
        
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                agent1, agent2 = agents[i], agents[j]
                path1, path2 = paths[agent1], paths[agent2]
                
                conflict = self._check_edge_conflicts(agent1, path1, agent2, path2)
                if conflict:
                    conflicts.append(conflict)
        
        return conflicts
    
    def _check_edge_conflicts(self, agent1, path1, agent2, path2):
        """检查边冲突"""
        min_path_len = min(len(path1), len(path2)) - 1
        
        for t in range(min_path_len):
            # 检查交换位置的冲突
            if (self._is_same_edge(path1[t], path1[t+1], path2[t+1], path2[t]) or
                self._segments_intersect(path1[t], path1[t+1], path2[t], path2[t+1])):
                
                intersection = self._compute_intersection(
                    path1[t], path1[t+1], path2[t], path2[t+1]
                )
                
                return EnhancedConflict(
                    agent1=agent1,
                    agent2=agent2,
                    location=intersection,
                    time_step=float(t + 0.5),
                    conflict_type='edge'
                )
        
        return None
    
    def _is_same_edge(self, p1, p2, p3, p4):
        """检查是否为相同边（交换）"""
        return (self._points_close(p1, p3) and self._points_close(p2, p4)) or \
               (self._points_close(p1, p4) and self._points_close(p2, p3))
    
    def _points_close(self, p1, p2, threshold=1.0):
        """检查两点是否接近"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) < threshold
    
    def _segments_intersect(self, p1, p2, p3, p4):
        """检查两线段是否相交"""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    def _compute_intersection(self, p1, p2, p3, p4):
        """计算两线段交点"""
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        x3, y3 = p3[0], p3[1]
        x4, y4 = p4[0], p4[1]
        
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-10:
            return ((x1+x2+x3+x4)/4, (y1+y2+y3+y4)/4)
        
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
        
        x = x1 + t*(x2-x1)
        y = y1 + t*(y2-y1)
        
        return (x, y)

class FollowingConflictDetector:
    """跟随冲突检测器"""
    
    def __init__(self, safety_rectangle):
        self.safety_rectangle = safety_rectangle
        self.min_following_distance = safety_rectangle.length * 1.5
    
    def detect(self, paths, spatial_hash=None):
        """检测跟随冲突"""
        conflicts = []
        
        # 构建路径段索引
        path_segments = self._build_path_segment_index(paths)
        
        # 查找共同路径段
        for segment_key, agents_info in path_segments.items():
            if len(agents_info) < 2:
                continue
            
            # 检查在同一路径段上的代理
            conflicts.extend(self._check_following_conflicts_on_segment(
                segment_key, agents_info, paths
            ))
        
        return conflicts
    
    def _build_path_segment_index(self, paths):
        """构建路径段索引"""
        segment_index = defaultdict(list)
        
        for agent_id, path in paths.items():
            if not path or len(path) < 2:
                continue
            
            for i in range(len(path) - 1):
                # 将路径段标准化（起点坐标较小的在前）
                p1, p2 = path[i], path[i+1]
                if (p1[0], p1[1]) > (p2[0], p2[1]):
                    p1, p2 = p2, p1
                
                segment_key = (
                    round(p1[0], 1), round(p1[1], 1),
                    round(p2[0], 1), round(p2[1], 1)
                )
                
                segment_index[segment_key].append((agent_id, i))
        
        return segment_index
    
    def _check_following_conflicts_on_segment(self, segment_key, agents_info, paths):
        """检查路径段上的跟随冲突"""
        conflicts = []
        
        # 计算每个代理在该段的时间
        agent_times = []
        for agent_id, path_index in agents_info:
            path = paths[agent_id]
            
            # 计算到达该段的时间
            arrival_time = self._calculate_arrival_time(path, path_index)
            departure_time = arrival_time + self._calculate_segment_time(
                path[path_index], path[path_index + 1]
            )
            
            agent_times.append((agent_id, arrival_time, departure_time))
        
        # 按到达时间排序
        agent_times.sort(key=lambda x: x[1])
        
        # 检查相邻代理的跟随冲突
        for i in range(len(agent_times) - 1):
            agent1, arrival1, departure1 = agent_times[i]
            agent2, arrival2, departure2 = agent_times[i + 1]
            
            # 如果时间重叠且间距不足
            if arrival2 < departure1:
                time_gap = arrival2 - arrival1
                
                if time_gap < self.min_following_distance / 1.0:  # 假设速度为1
                    conflicts.append(EnhancedConflict(
                        agent1=agent1,
                        agent2=agent2,
                        location=segment_key[:2],  # 使用段起点
                        time_step=(arrival1 + arrival2) / 2,
                        conflict_type='following'
                    ))
        
        return conflicts
    
    def _calculate_arrival_time(self, path, path_index):
        """计算到达路径点的时间"""
        time = 0.0
        for i in range(path_index):
            if i + 1 < len(path):
                time += self._calculate_segment_time(path[i], path[i+1])
        return time
    
    def _calculate_segment_time(self, p1, p2):
        """计算路径段时间"""
        distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        return distance / 1.0  # 假设速度为1

class ConnectionConflictDetector:
    """连接点冲突检测器"""
    
    def __init__(self, safety_rectangle):
        self.safety_rectangle = safety_rectangle
    
    def detect(self, paths, spatial_hash=None):
        """检测连接点冲突"""
        # 这个检测器需要骨干网络信息
        # 目前返回空列表，可以在有骨干网络时实现
        return []

class BackboneConflictDetector:
    """骨干路径冲突检测器"""
    
    def __init__(self, safety_rectangle):
        self.safety_rectangle = safety_rectangle
    
    def detect(self, paths, spatial_hash=None):
        """检测骨干路径冲突"""
        # 这个检测器需要骨干网络信息
        # 目前返回空列表，可以在有骨干网络时实现
        return []

class SafetyRectangle:
    """安全矩形"""
    
    def __init__(self, length=6.0, width=3.0, margin=0.5):
        self.length = length
        self.width = width
        self.margin = margin
        self.extended_length = length + 2 * margin
        self.extended_width = width + 2 * margin
        self.diagonal = math.sqrt(self.extended_length**2 + self.extended_width**2)

class OptimizedECBSSolver:
    """优化的ECBS求解器"""
    
    def __init__(self, env, backbone_network=None):
        self.env = env
        self.backbone_network = backbone_network
        
        # ECBS参数
        self.suboptimality_bound = 1.5
        self.focal_list_factor = 1.0
        
        # 性能优化
        self.max_search_time = 30.0
        self.max_nodes_expanded = 1000
        self.use_focal_search = True
        self.use_path_caching = True
        
        # 路径缓存
        self.path_cache = {}
        
        # 统计信息
        self.stats = {
            'nodes_expanded': 0,
            'conflicts_resolved': 0,
            'cache_hits': 0,
            'total_time': 0
        }
    
    def solve(self, initial_paths, conflicts):
        """ECBS主求解方法"""
        start_time = time.time()
        self.stats['nodes_expanded'] = 0
        
        if not conflicts:
            return initial_paths
        
        # 创建根节点
        root_node = ECBSNode(
            paths=initial_paths.copy(),
            constraints=[],
            cost=self._calculate_solution_cost(initial_paths),
            conflicts=conflicts
        )
        
        # 初始化搜索列表
        open_list = [root_node]
        focal_list = [root_node]
        
        # ECBS搜索循环
        best_solution = None
        
        while open_list and time.time() - start_time < self.max_search_time:
            # 从focal list选择节点
            current_node = self._select_node_from_focal(focal_list, open_list)
            
            if not current_node:
                break
            
            self.stats['nodes_expanded'] += 1
            
            # 检查是否无冲突
            if not current_node.conflicts:
                best_solution = current_node.paths
                break
            
            # 选择要解决的冲突
            conflict = self._select_conflict_to_resolve(current_node.conflicts)
            
            # 生成子节点
            child_nodes = self._generate_child_nodes(current_node, conflict)
            
            # 将子节点加入搜索列表
            for child in child_nodes:
                if child and self._is_valid_node(child):
                    heapq.heappush(open_list, child)
                    
                    # 更新focal list
                    if self.use_focal_search:
                        self._update_focal_list(focal_list, open_list)
            
            # 限制搜索规模
            if self.stats['nodes_expanded'] >= self.max_nodes_expanded:
                break
        
        self.stats['total_time'] = time.time() - start_time
        
        return best_solution or initial_paths
    
    def _select_node_from_focal(self, focal_list, open_list):
        """从focal list选择节点"""
        if not focal_list:
            return None
        
        # 选择冲突数最少的节点
        best_node = min(focal_list, key=lambda n: len(n.conflicts))
        focal_list.remove(best_node)
        
        if best_node in open_list:
            open_list.remove(best_node)
            heapq.heapify(open_list)
        
        return best_node
    
    def _select_conflict_to_resolve(self, conflicts):
        """选择要解决的冲突"""
        if not conflicts:
            return None
        
        # 优先解决高优先级、高严重程度的冲突
        return max(conflicts, key=lambda c: (c.priority, c.severity))
    
    def _generate_child_nodes(self, parent_node, conflict):
        """生成子节点"""
        child_nodes = []
        
        # 为冲突中的每个代理生成约束
        agents = [conflict.agent1, conflict.agent2]
        
        for agent in agents:
            # 创建约束
            constraint = EnhancedConstraint(
                agent_id=agent,
                location=conflict.location,
                time_step=conflict.time_step,
                constraint_type=conflict.conflict_type
            )
            
            # 创建子节点
            child_node = self._create_child_node(parent_node, constraint)
            
            if child_node:
                child_nodes.append(child_node)
        
        return child_nodes
    
    def _create_child_node(self, parent_node, constraint):
        """创建子节点"""
        try:
            # 复制父节点的约束和路径
            new_constraints = parent_node.constraints + [constraint]
            new_paths = parent_node.paths.copy()
            
            # 为受约束的代理重新规划路径
            agent_id = constraint.agent_id
            if agent_id in new_paths:
                # 获取起点和终点
                old_path = new_paths[agent_id]
                if not old_path or len(old_path) < 2:
                    return None
                
                start = old_path[0]
                goal = old_path[-1]
                
                # 重新规划路径（考虑约束）
                new_path = self._replan_with_constraints(
                    agent_id, start, goal, new_constraints
                )
                
                if new_path:
                    new_paths[agent_id] = new_path
                else:
                    return None  # 无法找到满足约束的路径
            
            # 计算新成本
            new_cost = self._calculate_solution_cost(new_paths)
            
            # 检测新冲突
            new_conflicts = self._detect_conflicts_in_solution(new_paths)
            
            # 创建子节点
            child_node = ECBSNode(
                paths=new_paths,
                constraints=new_constraints,
                cost=new_cost,
                conflicts=new_conflicts,
                parent=parent_node
            )
            
            return child_node
            
        except Exception as e:
            if hasattr(self, 'debug') and self.debug:
                print(f"创建子节点失败: {e}")
            return None
    
    def _replan_with_constraints(self, agent_id, start, goal, constraints):
        """在约束条件下重新规划路径"""
        # 检查缓存
        cache_key = self._get_replan_cache_key(agent_id, start, goal, constraints)
        if self.use_path_caching and cache_key in self.path_cache:
            self.stats['cache_hits'] += 1
            return self.path_cache[cache_key]
        
        # 过滤出适用于此代理的约束
        agent_constraints = [c for c in constraints if c.agent_id == agent_id]
        
        # 这里需要调用路径规划器，但考虑约束
        # 简化实现：如果有约束，尝试避开约束位置
        new_path = self._plan_path_avoiding_constraints(start, goal, agent_constraints)
        
        # 缓存结果
        if self.use_path_caching and new_path:
            self.path_cache[cache_key] = new_path
        
        return new_path
    
    def _plan_path_avoiding_constraints(self, start, goal, constraints):
        """规划避开约束的路径"""
        # 简化实现：使用基本的路径规划
        # 实际实现中应该调用enhanced path planner
        
        # 如果没有约束，使用直线路径
        if not constraints:
            return [start, goal]
        
        # 有约束时，尝试简单的绕行
        # 这里应该集成到optimized path planner中
        try:
            # 计算中间点以避开约束
            mid_x = (start[0] + goal[0]) / 2
            mid_y = (start[1] + goal[1]) / 2
            
            # 检查中间点是否受约束影响
            constraint_free = True
            for constraint in constraints:
                if constraint.location:
                    dist = math.sqrt(
                        (mid_x - constraint.location[0])**2 + 
                        (mid_y - constraint.location[1])**2
                    )
                    if dist < 5.0:  # 如果太近，需要调整
                        constraint_free = False
                        break
            
            if constraint_free:
                return [start, (mid_x, mid_y, 0), goal]
            else:
                # 更复杂的绕行逻辑
                offset_x = 10 if goal[0] > start[0] else -10
                offset_y = 10 if goal[1] > start[1] else -10
                
                waypoint = (mid_x + offset_x, mid_y + offset_y, 0)
                return [start, waypoint, goal]
        
        except:
            # 如果出错，返回简单路径
            return [start, goal]
    
    def _get_replan_cache_key(self, agent_id, start, goal, constraints):
        """生成重规划缓存键"""
        constraint_hash = hash(tuple(
            (c.agent_id, c.location, c.time_step, c.constraint_type)
            for c in constraints if c.agent_id == agent_id
        ))
        
        return f"{agent_id}:{start}:{goal}:{constraint_hash}"
    
    def _calculate_solution_cost(self, paths):
        """计算解决方案总成本"""
        total_cost = 0
        for path in paths.values():
            if path and len(path) > 1:
                # 计算路径长度
                for i in range(len(path) - 1):
                    dx = path[i+1][0] - path[i][0]
                    dy = path[i+1][1] - path[i][1]
                    total_cost += math.sqrt(dx*dx + dy*dy)
        
        return total_cost
    
    def _detect_conflicts_in_solution(self, paths):
        """检测解决方案中的冲突"""
        # 使用简化的冲突检测
        conflicts = []
        agents = list(paths.keys())
        
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                agent1, agent2 = agents[i], agents[j]
                path1, path2 = paths[agent1], paths[agent2]
                
                # 简单的顶点冲突检测
                min_len = min(len(path1), len(path2))
                for t in range(min_len):
                    p1, p2 = path1[t], path2[t]
                    dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                    
                    if dist < 3.0:  # 冲突阈值
                        conflicts.append(EnhancedConflict(
                            agent1=agent1,
                            agent2=agent2,
                            location=((p1[0] + p2[0])/2, (p1[1] + p2[1])/2),
                            time_step=float(t),
                            conflict_type='vertex'
                        ))
                        break
        
        return conflicts
    
    def _is_valid_node(self, node):
        """检查节点是否有效"""
        return (node is not None and 
                node.paths and 
                len(node.paths) > 0 and
                node.cost < float('inf'))
    
    def _update_focal_list(self, focal_list, open_list):
        """更新focal list"""
        if not open_list:
            focal_list.clear()
            return
        
        # 获取最优成本
        min_cost = min(node.cost for node in open_list)
        focal_threshold = min_cost * self.suboptimality_bound
        
        # 更新focal list
        focal_list.clear()
        for node in open_list:
            if node.cost <= focal_threshold:
                focal_list.append(node)

class ECBSNode:
    """ECBS搜索树节点"""
    
    def __init__(self, paths, constraints, cost, conflicts, parent=None):
        self.paths = paths
        self.constraints = constraints
        self.cost = cost
        self.conflicts = conflicts
        self.parent = parent
        
        # 用于堆排序
        self.f_value = cost
        self.conflict_count = len(conflicts)
    
    def __lt__(self, other):
        if self.f_value != other.f_value:
            return self.f_value < other.f_value
        return self.conflict_count < other.conflict_count

class OptimizedTrafficManager:
    """优化后的交通管理器"""
    
    def __init__(self, env, backbone_network=None):
        self.env = env
        self.backbone_network = backbone_network
        
        # 冲突检测和解决组件
        self.safety_rectangle = SafetyRectangle()
        self.conflict_detector = ConflictDetectionManager(self.safety_rectangle)
        self.ecbs_solver = OptimizedECBSSolver(env, backbone_network)
        
        # 路径预留系统
        self.reservation_table = SpaceTimeReservationTable()
        
        # 交通规则
        self.traffic_rules = {
            'min_vehicle_distance': 8.0,
            'intersection_priority': 'first_come',
            'speed_limits': {},
            'path_directions': {},
            'path_lanes': {},
            'dynamic_rules': True
        }
        
        # 性能优化
        self.use_prediction = True
        self.prediction_horizon = 50.0
        self.conflict_prediction_cache = {}
        
        # 自适应参数
        self.adaptive_safety_margin = True
        self.base_safety_margin = 2.0
        self.dynamic_priority_adjustment = True
        
        # 统计信息
        self.performance_stats = {
            'conflicts_detected': 0,
            'conflicts_resolved': 0,
            'resolution_time': 0,
            'prediction_accuracy': 0.95,
            'cache_hit_rate': 0.0
        }
    
    def set_backbone_network(self, backbone_network):
        """设置骨干路径网络"""
        self.backbone_network = backbone_network
        self.ecbs_solver.backbone_network = backbone_network
        
        # 初始化骨干路径规则
        if backbone_network:
            for path_id, path_data in backbone_network.paths.items():
                self.traffic_rules['speed_limits'][path_id] = path_data.get('speed_limit', 1.0)
                self.traffic_rules['path_directions'][path_id] = 'two_way'
                self.traffic_rules['path_lanes'][path_id] = path_data.get('capacity', 2)
    
    def register_vehicle_path(self, vehicle_id, path, start_time=0, speed=1.0):
        """注册车辆路径"""
        if not path or len(path) < 2:
            return False
        
        # 在预留表中注册路径
        reservation_count = self.reservation_table.reserve_path(
            vehicle_id, path, start_time, speed
        )
        
        # 预测潜在冲突
        if self.use_prediction:
            self._predict_future_conflicts(vehicle_id, path, start_time, speed)
        
        return reservation_count > 0
    
    def release_vehicle_path(self, vehicle_id):
        """释放车辆路径"""
        return self.reservation_table.remove_reservation(vehicle_id) > 0
    
    def check_path_conflicts(self, vehicle_id, path, start_time=0, speed=1.0):
        """检查路径冲突"""
        if not path or len(path) < 2:
            return False
        
        # 检查预留冲突
        conflicts = self.reservation_table.check_path_reservation(
            path, start_time, speed
        )
        
        if conflicts:
            return True
        
        # 动态安全边距检查
        if self.adaptive_safety_margin:
            safety_margin = self._calculate_dynamic_safety_margin(vehicle_id, path)
            if self._check_dynamic_safety_conflicts(path, safety_margin):
                return True
        
        return False
    
    def detect_conflicts(self, paths):
        """检测路径集合中的冲突"""
        start_time = time.time()
        
        # 使用优化的冲突检测器
        conflicts = self.conflict_detector.detect_conflicts(paths)
        
        self.performance_stats['conflicts_detected'] += len(conflicts)
        self.performance_stats['resolution_time'] += time.time() - start_time
        
        return conflicts
    
    def resolve_conflicts(self, paths, backbone_network=None, path_structures=None):
        """使用ECBS解决冲突"""
        if not paths or len(paths) < 2:
            return paths
        
        start_time = time.time()
        
        # 检测冲突
        conflicts = self.detect_conflicts(paths)
        
        if not conflicts:
            return paths  # 无冲突
        
        # 设置骨干网络（如果提供）
        if backbone_network:
            self.set_backbone_network(backbone_network)
        
        # 使用ECBS求解器解决冲突
        resolved_paths = self.ecbs_solver.solve(paths, conflicts)
        
        # 更新统计信息
        resolution_time = time.time() - start_time
        self.performance_stats['resolution_time'] += resolution_time
        
        if resolved_paths != paths:
            self.performance_stats['conflicts_resolved'] += len(conflicts)
        
        return resolved_paths or paths
    
    def suggest_path_adjustment(self, vehicle_id, start, goal):
        """建议路径调整"""
        # 暂时释放当前车辆的预留
        self.release_vehicle_path(vehicle_id)
        
        # 获取当前所有活动路径
        active_paths = self._get_active_paths()
        
        # 为当前车辆规划新路径
        # 这里需要集成到路径规划器中
        suggested_path = self._plan_conflict_free_path(vehicle_id, start, goal, active_paths)
        
        return suggested_path
    
    def suggest_speed_adjustment(self, vehicle_id):
        """建议速度调整"""
        # 检查车辆的当前路径预留
        upcoming_conflicts = self._get_upcoming_conflicts(vehicle_id)
        
        if not upcoming_conflicts:
            return None
        
        # 计算建议的速度调整
        speed_factors = []
        
        for conflict in upcoming_conflicts:
            # 基于冲突严重程度和时间计算速度因子
            if conflict['time_to_conflict'] < 10.0:  # 即将发生冲突
                if len(conflict['other_agents']) == 1:
                    # 简单避让：减速或加速
                    if vehicle_id < conflict['other_agents'][0]:  # 基于ID优先级
                        speed_factors.append(1.2)  # 加速
                    else:
                        speed_factors.append(0.8)  # 减速
                else:
                    # 多车冲突：保守减速
                    speed_factors.append(0.7)
        
        if speed_factors:
            # 选择最保守的调整
            return min(speed_factors) if any(f < 1.0 for f in speed_factors) else max(speed_factors)
        
        return None
    
    def _predict_future_conflicts(self, vehicle_id, path, start_time, speed):
        """预测未来冲突"""
        if not self.use_prediction:
            return
        
        # 构建预测键
        prediction_key = self._get_prediction_cache_key(vehicle_id, path, start_time)
        
        # 检查缓存
        if prediction_key in self.conflict_prediction_cache:
            return self.conflict_prediction_cache[prediction_key]
        
        # 预测逻辑
        predicted_conflicts = []
        
        # 检查与已注册路径的潜在冲突
        for location, time_slot, agents in self._iterate_reservations():
            if vehicle_id not in agents:  # 排除自己
                # 检查时空重叠
                vehicle_time = self._calculate_vehicle_time_at_location(
                    path, location, start_time, speed
                )
                
                if vehicle_time is not None and abs(vehicle_time - time_slot) < 5.0:
                    predicted_conflicts.append({
                        'location': location,
                        'time': vehicle_time,
                        'other_agents': agents,
                        'confidence': 0.8
                    })
        
        # 缓存预测结果
        self.conflict_prediction_cache[prediction_key] = predicted_conflicts
        
        return predicted_conflicts
    
    def _calculate_dynamic_safety_margin(self, vehicle_id, path):
        """计算动态安全边距"""
        base_margin = self.base_safety_margin
        
        if not self.adaptive_safety_margin:
            return base_margin
        
        # 基于路径复杂度调整
        complexity_factor = self._calculate_path_complexity(path)
        complexity_adjustment = complexity_factor * 0.5
        
        # 基于交通密度调整
        density_factor = self._calculate_local_traffic_density(path)
        density_adjustment = density_factor * 1.0
        
        # 基于车辆历史表现调整
        performance_factor = self._get_vehicle_performance_factor(vehicle_id)
        performance_adjustment = (1.0 - performance_factor) * 0.3
        
        adjusted_margin = base_margin + complexity_adjustment + density_adjustment + performance_adjustment
        
        return max(1.0, min(5.0, adjusted_margin))  # 限制在合理范围内
    
    def _check_dynamic_safety_conflicts(self, path, safety_margin):
        """检查动态安全冲突"""
        # 沿路径检查安全边距
        for i, point in enumerate(path[::5]):  # 每5个点检查一次
            nearby_reservations = self._get_nearby_reservations(point, safety_margin)
            
            if nearby_reservations:
                return True
        
        return False
    
    def _calculate_path_complexity(self, path):
        """计算路径复杂度"""
        if not path or len(path) < 3:
            return 0.0
        
        total_turning = 0.0
        for i in range(1, len(path) - 1):
            # 计算转弯角度
            v1 = np.array([path[i][0] - path[i-1][0], path[i][1] - path[i-1][1]])
            v2 = np.array([path[i+1][0] - path[i][0], path[i+1][1] - path[i][1]])
            
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = math.acos(cos_angle)
                total_turning += angle
        
        # 归一化复杂度
        path_length = self._calculate_path_length(path)
        if path_length > 0:
            return total_turning / path_length
        
        return 0.0
    
    def _calculate_local_traffic_density(self, path):
        """计算路径周围的交通密度"""
        if not path:
            return 0.0
        
        total_density = 0.0
        sample_points = path[::max(1, len(path)//10)]  # 采样10个点
        
        for point in sample_points:
            nearby_count = len(self._get_nearby_reservations(point, 10.0))
            total_density += nearby_count
        
        # 归一化密度
        return min(1.0, total_density / (len(sample_points) * 5))  # 假设5为最大密度
    
    def _get_vehicle_performance_factor(self, vehicle_id):
        """获取车辆性能因子"""
        # 基于历史冲突记录计算
        # 简化实现：返回默认值
        return 0.8  # 0-1之间，1表示性能最好
    
    def _get_nearby_reservations(self, point, radius):
        """获取附近的预留"""
        nearby = []
        
        # 简化实现：检查预留表
        for reservation_key, agents in self.reservation_table.reservations.items():
            x, y, t = reservation_key
            distance = math.sqrt((x - point[0])**2 + (y - point[1])**2)
            
            if distance <= radius:
                nearby.extend(agents)
        
        return nearby
    
    def _get_active_paths(self):
        """获取当前活动路径"""
        # 这需要从调度器或环境中获取
        # 简化实现：返回空字典
        return {}
    
    def _plan_conflict_free_path(self, vehicle_id, start, goal, active_paths):
        """规划无冲突路径"""
        # 这需要调用路径规划器
        # 简化实现：返回直线路径
        return [start, goal]
    
    def _get_upcoming_conflicts(self, vehicle_id):
        """获取即将发生的冲突"""
        upcoming = []
        
        # 检查预测缓存
        for prediction_key, conflicts in self.conflict_prediction_cache.items():
            if vehicle_id in prediction_key:
                for conflict in conflicts:
                    if conflict['time'] - time.time() < 30.0:  # 30秒内
                        conflict['time_to_conflict'] = conflict['time'] - time.time()
                        upcoming.append(conflict)
        
        return upcoming
    
    def _get_prediction_cache_key(self, vehicle_id, path, start_time):
        """生成预测缓存键"""
        path_hash = hash(tuple((round(p[0], 1), round(p[1], 1)) for p in path[::5]))
        return f"{vehicle_id}:{path_hash}:{int(start_time)}"
    
    def _calculate_vehicle_time_at_location(self, path, location, start_time, speed):
        """计算车辆到达某位置的时间"""
        current_time = start_time
        
        for i in range(len(path) - 1):
            # 检查是否经过该位置
            segment_start = path[i]
            segment_end = path[i + 1]
            
            # 简化检查：点是否在线段附近
            if self._point_near_segment(location, segment_start, segment_end, 2.0):
                return current_time
            
            # 更新时间
            segment_length = math.sqrt(
                (segment_end[0] - segment_start[0])**2 + 
                (segment_end[1] - segment_start[1])**2
            )
            current_time += segment_length / speed
        
        return None
    
    def _point_near_segment(self, point, seg_start, seg_end, threshold):
        """检查点是否靠近线段"""
        # 计算点到线段的距离
        A = seg_end[1] - seg_start[1]
        B = seg_start[0] - seg_end[0]
        C = seg_end[0] * seg_start[1] - seg_start[0] * seg_end[1]
        
        distance = abs(A * point[0] + B * point[1] + C) / math.sqrt(A*A + B*B)
        
        return distance <= threshold
    
    def _iterate_reservations(self):
        """迭代所有预留"""
        for key, agents in self.reservation_table.reservations.items():
            x, y, t = key
            yield (x, y), t, agents
    
    def _calculate_path_length(self, path):
        """计算路径长度"""
        if not path or len(path) < 2:
            return 0
        
        length = 0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            length += math.sqrt(dx*dx + dy*dy)
        
        return length
    
    def get_performance_stats(self):
        """获取性能统计"""
        stats = self.performance_stats.copy()
        
        # 计算缓存命中率
        total_predictions = len(self.conflict_prediction_cache)
        if total_predictions > 0:
            # 简化计算
            stats['cache_hit_rate'] = min(0.95, total_predictions / 100.0)
        
        # 添加ECBS统计
        stats.update(self.ecbs_solver.stats)
        
        return stats

class SpaceTimeReservationTable:
    """时空预留表"""
    
    def __init__(self):
        self.reservations = {}  # {(x, y, t): [agent_ids]}
        self.agent_reservations = {}  # {agent_id: [(x, y, t)]}
    
    def reserve_path(self, agent_id, path, start_time=0, speed=1.0):
        """为路径添加预留"""
        if not path or len(path) < 2:
            return 0
        
        reservation_count = 0
        current_time = start_time
        
        # 清除该代理的旧预留
        self.remove_reservation(agent_id)
        
        for i in range(len(path) - 1):
            segment_length = math.sqrt(
                (path[i+1][0] - path[i][0])**2 + 
                (path[i+1][1] - path[i][1])**2
            )
            segment_time = segment_length / speed
            
            # 沿路径段添加预留
            num_reservations = max(1, int(segment_length / 2.0))
            
            for j in range(num_reservations + 1):
                ratio = j / max(1, num_reservations)
                
                x = path[i][0] + ratio * (path[i+1][0] - path[i][0])
                y = path[i][1] + ratio * (path[i+1][1] - path[i][1])
                t = current_time + ratio * segment_time
                
                # 离散化坐标和时间
                key = (round(x, 1), round(y, 1), round(t, 1))
                
                if key not in self.reservations:
                    self.reservations[key] = []
                
                if agent_id not in self.reservations[key]:
                    self.reservations[key].append(agent_id)
                    reservation_count += 1
                
                # 更新代理预留映射
                if agent_id not in self.agent_reservations:
                    self.agent_reservations[agent_id] = []
                
                if key not in self.agent_reservations[agent_id]:
                    self.agent_reservations[agent_id].append(key)
            
            current_time += segment_time
        
        return reservation_count
    
    def remove_reservation(self, agent_id):
        """移除代理的所有预留"""
        if agent_id not in self.agent_reservations:
            return 0
        
        removed_count = 0
        
        for key in self.agent_reservations[agent_id]:
            if key in self.reservations and agent_id in self.reservations[key]:
                self.reservations[key].remove(agent_id)
                removed_count += 1
                
                # 如果该位置没有其他预留，删除键
                if not self.reservations[key]:
                    del self.reservations[key]
        
        # 清空代理预留列表
        self.agent_reservations[agent_id] = []
        
        return removed_count
    
    def check_path_reservation(self, path, start_time=0, speed=1.0):
        """检查路径是否与现有预留冲突"""
        conflicts = []
        current_time = start_time
        
        for i in range(len(path) - 1):
            segment_length = math.sqrt(
                (path[i+1][0] - path[i][0])**2 + 
                (path[i+1][1] - path[i][1])**2
            )
            segment_time = segment_length / speed
            
            # 检查段上的冲突
            num_checks = max(1, int(segment_length / 2.0))
            
            for j in range(num_checks + 1):
                ratio = j / max(1, num_checks)
                
                x = path[i][0] + ratio * (path[i+1][0] - path[i][0])
                y = path[i][1] + ratio * (path[i+1][1] - path[i][1])
                t = current_time + ratio * segment_time
                
                # 检查该时空位置是否被预留
                key = (round(x, 1), round(y, 1), round(t, 1))
                
                if key in self.reservations and self.reservations[key]:
                    conflicts.append(((x, y), t, self.reservations[key]))
            
            current_time += segment_time
        
        return conflicts

# 保持向后兼容性
TrafficManager = OptimizedTrafficManager