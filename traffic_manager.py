import math
import heapq
import time
import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass

@dataclass
class Conflict:
    """简化的冲突表示"""
    agent1: str
    agent2: str
    location: Tuple[float, float]
    time_step: float
    conflict_type: str = 'vertex'
    severity: float = 1.0
    
    def __lt__(self, other):
        return self.severity > other.severity

@dataclass
class Constraint:
    """简化的约束条件"""
    agent_id: str
    location: Tuple[float, float]
    time_step: float
    constraint_type: str = 'vertex'

class SimplifiedECBSSolver:
    """简化的ECBS求解器 - 适配骨干路径系统"""
    
    def __init__(self, env, backbone_network=None):
        self.env = env
        self.backbone_network = backbone_network
        
        # ECBS参数
        self.suboptimality_bound = 1.5
        self.max_search_time = 15.0
        self.max_nodes_expanded = 500
        
        # 统计
        self.stats = {
            'nodes_expanded': 0,
            'conflicts_resolved': 0,
            'total_time': 0
        }
    
    def solve(self, initial_paths, conflicts):
        """简化的ECBS求解"""
        start_time = time.time()
        
        if not conflicts:
            return initial_paths
        
        # 选择最严重的冲突进行解决
        primary_conflict = max(conflicts, key=lambda c: c.severity)
        
        # 为冲突双方生成替代路径
        resolved_paths = initial_paths.copy()
        
        # 简化策略：为冲突中的一个代理重新规划路径
        agent_to_replan = primary_conflict.agent1
        
        if agent_to_replan in initial_paths:
            original_path = initial_paths[agent_to_replan]
            if len(original_path) >= 2:
                start = original_path[0]
                goal = original_path[-1]
                
                # 尝试使用骨干路径重新规划
                new_path = self._replan_avoiding_conflict(
                    agent_to_replan, start, goal, primary_conflict
                )
                
                if new_path:
                    resolved_paths[agent_to_replan] = new_path
                    self.stats['conflicts_resolved'] += 1
        
        self.stats['total_time'] = time.time() - start_time
        self.stats['nodes_expanded'] += 1
        
        return resolved_paths
    
    def _replan_avoiding_conflict(self, agent_id, start, goal, conflict):
        """重新规划避开冲突的路径"""
        if not self.backbone_network:
            return self._simple_detour_path(start, goal, conflict.location)
        
        # 尝试使用骨干路径规划
        try:
            target_type, target_id = self.backbone_network.identify_target_point(goal)
            
            if target_type:
                # 获取骨干路径
                path, structure = self.backbone_network.get_path_from_position_to_target(
                    start, target_type, target_id
                )
                
                if path and self._path_avoids_conflict(path, conflict):
                    return path
        
        except Exception as e:
            print(f"骨干路径重规划失败: {e}")
        
        # 回退到简单绕行
        return self._simple_detour_path(start, goal, conflict.location)
    
    def _simple_detour_path(self, start, goal, conflict_location):
        """生成简单的绕行路径"""
        # 计算绕行点
        mid_x = (start[0] + goal[0]) / 2
        mid_y = (start[1] + goal[1]) / 2
        
        # 向垂直方向偏移以避开冲突
        offset_distance = 15.0
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        
        if abs(dx) > abs(dy):
            # 水平主导，垂直偏移
            waypoint = (mid_x, mid_y + offset_distance, 0)
        else:
            # 垂直主导，水平偏移
            waypoint = (mid_x + offset_distance, mid_y, 0)
        
        # 检查绕行点是否有效
        if self._is_valid_position(waypoint):
            return [start, waypoint, goal]
        else:
            # 尝试反向偏移
            if abs(dx) > abs(dy):
                waypoint = (mid_x, mid_y - offset_distance, 0)
            else:
                waypoint = (mid_x - offset_distance, mid_y, 0)
            
            if self._is_valid_position(waypoint):
                return [start, waypoint, goal]
        
        # 如果都不行，返回直线路径
        return [start, goal]
    
    def _path_avoids_conflict(self, path, conflict):
        """检查路径是否避开冲突"""
        if not path or not conflict.location:
            return False
        
        for point in path:
            distance = math.sqrt(
                (point[0] - conflict.location[0])**2 + 
                (point[1] - conflict.location[1])**2
            )
            if distance < 5.0:  # 如果路径点离冲突位置太近
                return False
        
        return True
    
    def _is_valid_position(self, position):
        """检查位置是否有效"""
        x, y = int(position[0]), int(position[1])
        
        if (x < 0 or x >= self.env.width or 
            y < 0 or y >= self.env.height):
            return False
        
        if hasattr(self.env, 'grid') and self.env.grid[x, y] == 1:
            return False
        
        return True

class OptimizedTrafficManager:
    """优化的交通管理器 - 适配骨干路径系统"""
    
    def __init__(self, env, backbone_network=None):
        self.env = env
        self.backbone_network = backbone_network
        
        # 简化的ECBS求解器
        self.ecbs_solver = SimplifiedECBSSolver(env, backbone_network)
        
        # 路径预留表
        self.active_paths = {}  # {vehicle_id: path}
        self.path_reservations = {}  # {(x, y, t): [vehicle_ids]}
        
        # 冲突检测参数
        self.safety_distance = 8.0
        self.time_discretization = 2.0
        
        # 统计信息
        self.stats = {
            'conflicts_detected': 0,
            'conflicts_resolved': 0,
            'ecbs_calls': 0,
            'resolution_time': 0,
            'backbone_conflicts': 0,  # 骨干路径冲突数
            'access_conflicts': 0     # 接入路径冲突数
        }
        
        print("初始化优化的交通管理器")
    
    def set_backbone_network(self, backbone_network):
        """设置骨干路径网络"""
        self.backbone_network = backbone_network
        self.ecbs_solver.backbone_network = backbone_network
        print("已设置骨干路径网络")
    
    def register_vehicle_path(self, vehicle_id, path, start_time=0, speed=1.0):
        """注册车辆路径"""
        if not path or len(path) < 2:
            return False
        
        # 移除旧路径
        self.release_vehicle_path(vehicle_id)
        
        # 注册新路径
        self.active_paths[vehicle_id] = {
            'path': path,
            'start_time': start_time,
            'speed': speed,
            'registered_time': time.time()
        }
        
        # 添加到时空预留表
        self._add_path_reservations(vehicle_id, path, start_time, speed)
        
        return True
    
    def release_vehicle_path(self, vehicle_id):
        """释放车辆路径"""
        if vehicle_id in self.active_paths:
            # 从预留表中移除
            self._remove_path_reservations(vehicle_id)
            
            # 从活动路径中移除
            del self.active_paths[vehicle_id]
            
            return True
        
        return False
    
    def check_path_conflicts(self, vehicle_id, path, start_time=0, speed=1.0):
        """检查路径是否有冲突"""
        if not path or len(path) < 2:
            return False
        
        # 临时添加路径进行冲突检测
        temp_paths = {vehicle_id: path}
        temp_paths.update({vid: data['path'] for vid, data in self.active_paths.items()})
        
        # 检测冲突
        conflicts = self.detect_conflicts(temp_paths)
        
        # 检查是否涉及当前车辆
        for conflict in conflicts:
            if conflict.agent1 == vehicle_id or conflict.agent2 == vehicle_id:
                return True
        
        return False
    
    def detect_conflicts(self, paths):
        """检测路径集合中的冲突"""
        conflicts = []
        
        if len(paths) < 2:
            return conflicts
        
        agents = list(paths.keys())
        
        # 两两检查车辆路径
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                agent1, agent2 = agents[i], agents[j]
                path1, path2 = paths[agent1], paths[agent2]
                
                # 检测顶点冲突
                vertex_conflicts = self._detect_vertex_conflicts(agent1, path1, agent2, path2)
                conflicts.extend(vertex_conflicts)
                
                # 检测边冲突
                edge_conflicts = self._detect_edge_conflicts(agent1, path1, agent2, path2)
                conflicts.extend(edge_conflicts)
        
        self.stats['conflicts_detected'] += len(conflicts)
        
        # 分类冲突类型
        for conflict in conflicts:
            conflict_location = conflict.location
            if self._is_backbone_location(conflict_location):
                self.stats['backbone_conflicts'] += 1
                conflict.severity *= 1.5  # 骨干路径冲突更严重
            else:
                self.stats['access_conflicts'] += 1
        
        return conflicts
    
    def resolve_conflicts(self, paths, backbone_network=None, path_structures=None):
        """使用ECBS解决冲突"""
        if not paths or len(paths) < 2:
            return paths
        
        start_time = time.time()
        
        # 更新骨干网络引用
        if backbone_network:
            self.set_backbone_network(backbone_network)
        
        # 检测冲突
        conflicts = self.detect_conflicts(paths)
        
        if not conflicts:
            return paths
        
        print(f"检测到 {len(conflicts)} 个冲突，开始ECBS解决")
        
        # 使用ECBS求解器
        self.stats['ecbs_calls'] += 1
        resolved_paths = self.ecbs_solver.solve(paths, conflicts)
        
        # 更新统计
        resolution_time = time.time() - start_time
        self.stats['resolution_time'] += resolution_time
        
        if resolved_paths != paths:
            self.stats['conflicts_resolved'] += len(conflicts)
            print(f"ECBS解决完成，耗时 {resolution_time:.2f}s")
        
        return resolved_paths or paths
    
    def suggest_path_adjustment(self, vehicle_id, start, goal):
        """建议路径调整"""
        if not self.backbone_network:
            return None
        
        # 释放当前路径
        self.release_vehicle_path(vehicle_id)
        
        try:
            # 使用骨干路径规划新路径
            target_type, target_id = self.backbone_network.identify_target_point(goal)
            
            if target_type:
                path, structure = self.backbone_network.get_path_from_position_to_target(
                    start, target_type, target_id
                )
                
                if path:
                    # 检查新路径是否避开冲突
                    if not self.check_path_conflicts(vehicle_id, path):
                        return path
        
        except Exception as e:
            print(f"路径调整建议失败: {e}")
        
        return None
    
    def suggest_speed_adjustment(self, vehicle_id):
        """建议速度调整"""
        if vehicle_id not in self.active_paths:
            return None
        
        # 简化的速度调整策略
        vehicle_path = self.active_paths[vehicle_id]['path']
        
        # 检查是否有即将发生的冲突
        upcoming_conflicts = self._get_upcoming_conflicts(vehicle_id, vehicle_path)
        
        if upcoming_conflicts:
            # 如果有冲突，建议减速
            return 0.7
        else:
            # 如果在骨干路径上，可以加速
            if self._path_uses_backbone(vehicle_path):
                return 1.2
        
        return None
    
    def _detect_vertex_conflicts(self, agent1, path1, agent2, path2):
        """检测顶点冲突"""
        conflicts = []
        
        min_len = min(len(path1), len(path2))
        
        for t in range(min_len):
            p1 = path1[t]
            p2 = path2[t]
            
            distance = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            
            if distance < self.safety_distance:
                conflict = Conflict(
                    agent1=agent1,
                    agent2=agent2,
                    location=((p1[0] + p2[0])/2, (p1[1] + p2[1])/2),
                    time_step=float(t),
                    conflict_type='vertex',
                    severity=self.safety_distance / (distance + 0.1)
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def _detect_edge_conflicts(self, agent1, path1, agent2, path2):
        """检测边冲突"""
        conflicts = []
        
        min_len = min(len(path1), len(path2)) - 1
        
        for t in range(min_len):
            # 检查交换位置的情况
            if (self._points_close(path1[t], path2[t+1]) and 
                self._points_close(path1[t+1], path2[t])):
                
                conflict = Conflict(
                    agent1=agent1,
                    agent2=agent2,
                    location=((path1[t][0] + path2[t][0])/2, (path1[t][1] + path2[t][1])/2),
                    time_step=float(t + 0.5),
                    conflict_type='edge',
                    severity=1.5
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def _points_close(self, p1, p2, threshold=None):
        """检查两点是否接近"""
        if threshold is None:
            threshold = self.safety_distance
        
        distance = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        return distance < threshold
    
    def _is_backbone_location(self, location):
        """检查位置是否在骨干路径上"""
        if not self.backbone_network or not self.backbone_network.paths:
            return False
        
        # 简化检查：查看是否接近任何骨干路径
        for path_data in self.backbone_network.paths.values():
            backbone_path = path_data.get('path', [])
            
            for point in backbone_path[::5]:  # 每隔5个点检查一次
                distance = math.sqrt(
                    (location[0] - point[0])**2 + 
                    (location[1] - point[1])**2
                )
                if distance < 3.0:
                    return True
        
        return False
    
    def _path_uses_backbone(self, path):
        """检查路径是否使用骨干路径"""
        if not self.backbone_network or not path:
            return False
        
        # 简化检查：如果路径的大部分点接近骨干路径，认为使用了骨干路径
        backbone_points = 0
        
        for point in path[::3]:  # 每隔3个点检查一次
            if self._is_backbone_location(point):
                backbone_points += 1
        
        # 如果超过30%的点在骨干路径上，认为使用了骨干路径
        return backbone_points > len(path) * 0.3 / 3
    
    def _get_upcoming_conflicts(self, vehicle_id, path):
        """获取即将发生的冲突"""
        conflicts = []
        
        # 与其他活动路径检查冲突
        for other_id, other_data in self.active_paths.items():
            if other_id == vehicle_id:
                continue
            
            other_path = other_data['path']
            
            # 检查前10个点的冲突
            check_length = min(10, len(path), len(other_path))
            
            for i in range(check_length):
                if self._points_close(path[i], other_path[i]):
                    conflicts.append({
                        'other_vehicle': other_id,
                        'time_step': i,
                        'location': path[i]
                    })
        
        return conflicts
    
    def _add_path_reservations(self, vehicle_id, path, start_time, speed):
        """添加路径到预留表"""
        current_time = start_time
        
        for i in range(len(path) - 1):
            # 计算段时间
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
    
    def _remove_path_reservations(self, vehicle_id):
        """从预留表中移除车辆路径"""
        keys_to_remove = []
        
        for key, vehicles in self.path_reservations.items():
            if vehicle_id in vehicles:
                vehicles.remove(vehicle_id)
                
                if not vehicles:
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.path_reservations[key]
    
    def get_performance_stats(self):
        """获取性能统计"""
        stats = self.stats.copy()
        
        # 添加ECBS统计
        ecbs_stats = self.ecbs_solver.stats
        stats.update({
            'ecbs_nodes_expanded': ecbs_stats['nodes_expanded'],
            'ecbs_total_time': ecbs_stats['total_time'],
            'ecbs_conflicts_resolved': ecbs_stats['conflicts_resolved']
        })
        
        # 计算成功率
        total_conflicts = stats['conflicts_detected']
        if total_conflicts > 0:
            stats['resolution_success_rate'] = stats['conflicts_resolved'] / total_conflicts
        else:
            stats['resolution_success_rate'] = 1.0
        
        # 骨干路径利用情况
        if stats['backbone_conflicts'] + stats['access_conflicts'] > 0:
            stats['backbone_conflict_ratio'] = stats['backbone_conflicts'] / (
                stats['backbone_conflicts'] + stats['access_conflicts']
            )
        else:
            stats['backbone_conflict_ratio'] = 0.0
        
        return stats
    
    def clear_expired_reservations(self, current_time):
        """清理过期的预留"""
        expiry_time = 300  # 5分钟过期
        
        keys_to_remove = []
        
        for vehicle_id, data in list(self.active_paths.items()):
            if current_time - data['registered_time'] > expiry_time:
                keys_to_remove.append(vehicle_id)
        
        for vehicle_id in keys_to_remove:
            self.release_vehicle_path(vehicle_id)
    
    def get_system_status(self):
        """获取系统状态"""
        return {
            'active_vehicles': len(self.active_paths),
            'total_reservations': len(self.path_reservations),
            'backbone_network_available': self.backbone_network is not None,
            'backbone_paths_count': len(self.backbone_network.paths) if self.backbone_network else 0,
            'ecbs_enabled': True,
            'safety_distance': self.safety_distance
        }

# 保持向后兼容性
TrafficManager = OptimizedTrafficManager