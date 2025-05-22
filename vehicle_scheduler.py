import math
import heapq
import time
from typing import List, Dict, Tuple, Set, Optional, Any
from collections import defaultdict

class VehicleTask:
    """车辆任务类 - 增强版"""
    
    def __init__(self, task_id, task_type, start, goal, priority=1, 
                 loading_point_id=None, unloading_point_id=None):
        self.task_id = task_id
        self.task_type = task_type  # 'to_loading', 'to_unloading', 'to_initial'
        self.start = start
        self.goal = goal
        self.priority = priority
        self.status = 'pending'  # 'pending', 'assigned', 'in_progress', 'completed', 'failed'
        self.assigned_vehicle = None
        self.progress = 0.0  # 0.0 - 1.0
        self.path = None
        self.estimated_time = 0
        self.start_time = 0
        self.completion_time = 0
        
        # 新增: 装载点和卸载点ID
        self.loading_point_id = loading_point_id
        self.unloading_point_id = unloading_point_id
        
        # 新增: 保存路径结构
        self.path_structure = {
            'type': 'unknown',  # 'three_segment', 'direct', 'hybrid'
            'entry_point': None,
            'exit_point': None,
            'backbone_segment': None,
            'to_backbone_path': None,
            'backbone_path': None,
            'from_backbone_path': None
        }
        
        # 新增: 质量评估
        self.quality_score = 0.0
        self.backbone_utilization = 0.0  # 骨干网络利用率
    
    def to_dict(self):
        """转换为字典表示"""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'start': self.start,
            'goal': self.goal,
            'priority': self.priority,
            'status': self.status,
            'progress': self.progress,
            'assigned_vehicle': self.assigned_vehicle,
            'estimated_time': self.estimated_time,
            'start_time': self.start_time,
            'completion_time': self.completion_time,
            'loading_point_id': self.loading_point_id,
            'unloading_point_id': self.unloading_point_id,
            'quality_score': self.quality_score,
            'backbone_utilization': self.backbone_utilization
        }
    
    def update_path_structure(self, structure):
        """更新路径结构信息"""
        if structure:
            self.path_structure.update(structure)
            
            # 计算骨干网络利用率
            if self.path and structure.get('backbone_path'):
                backbone_length = self._calculate_path_length(structure['backbone_path'])
                total_length = self._calculate_path_length(self.path)
                self.backbone_utilization = backbone_length / max(total_length, 1.0)
    
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


class VehicleScheduler:
    """车辆调度器，管理任务分配和调度 - 增强版"""
    
    def __init__(self, env, path_planner=None, backbone_network=None, traffic_manager=None):
        self.env = env
        self.path_planner = path_planner
        self.backbone_network = backbone_network
        self.traffic_manager = traffic_manager
        
        self.tasks = {}  # 任务字典 {task_id: task}
        self.task_queues = {}  # 车辆任务队列 {vehicle_id: [task_ids]}
        self.vehicle_statuses = {}  # 车辆状态 {vehicle_id: status}
        self.task_counter = 0  # 任务计数器
        self.mission_templates = {}  # 任务模板 {template_id: [tasks]}
        
        # 统计信息
        self.stats = {
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_distance': 0,
            'total_time': 0,
            'vehicle_utilization': {},
            'backbone_usage_efficiency': 0.0,
            'conflict_resolution_count': 0
        }
        
        # 点位使用情况跟踪
        self.loading_point_usage = {}  # {point_id: usage_count}
        self.unloading_point_usage = {}  # {point_id: usage_count}
        
        # 路径结构缓存和质量评估
        self.path_structure_cache = {}
        self.quality_assessor = PathQualityAssessor() if 'PathQualityAssessor' in globals() else None
    
    def set_path_planner(self, path_planner):
        """设置路径规划器"""
        self.path_planner = path_planner
        if self.path_planner and self.backbone_network:
            self.path_planner.set_backbone_network(self.backbone_network)
    
    def set_backbone_network(self, backbone_network):
        """设置骨干路径网络"""
        self.backbone_network = backbone_network
        if self.path_planner:
            self.path_planner.set_backbone_network(backbone_network)
    
    def set_traffic_manager(self, traffic_manager):
        """设置交通管理器"""
        self.traffic_manager = traffic_manager
    
    def initialize_vehicles(self):
        """初始化车辆状态"""
        for vehicle_id, vehicle in self.env.vehicles.items():
            self.vehicle_statuses[vehicle_id] = {
                'status': 'idle',
                'position': vehicle.get('position', (0, 0, 0)),
                'load': 0,
                'max_load': vehicle.get('max_load', 100),
                'current_task': None,
                'completed_tasks': 0,
                'total_distance': 0,
                'total_time': 0,
                'utilization_rate': 0.0,
                'active_time': 0,
                'idle_time': 0,
                'preferred_loading_point': None,
                'preferred_unloading_point': None,
                'task_queue': []  # 新增: 详细任务队列信息
            }
            
            # 初始化任务队列
            self.task_queues[vehicle_id] = []
    
    def create_mission_template(self, template_id, loading_point=None, unloading_point=None):
        """创建任务模板（装载-卸载-返回循环）"""
        loading_point_id = None
        unloading_point_id = None
        
        # 处理装载点
        if loading_point is not None:
            if isinstance(loading_point, int):
                loading_point_id = loading_point
                if 0 <= loading_point_id < len(self.env.loading_points):
                    loading_point = self.env.loading_points[loading_point_id]
                else:
                    loading_point = None
            else:
                for i, point in enumerate(self.env.loading_points):
                    if point == loading_point:
                        loading_point_id = i
                        break
        
        if loading_point is None and self.env.loading_points:
            loading_point = self.env.loading_points[0]
            loading_point_id = 0
        
        # 处理卸载点
        if unloading_point is not None:
            if isinstance(unloading_point, int):
                unloading_point_id = unloading_point
                if 0 <= unloading_point_id < len(self.env.unloading_points):
                    unloading_point = self.env.unloading_points[unloading_point_id]
                else:
                    unloading_point = None
            else:
                for i, point in enumerate(self.env.unloading_points):
                    if point == unloading_point:
                        unloading_point_id = i
                        break
        
        if unloading_point is None and self.env.unloading_points:
            unloading_point = self.env.unloading_points[0]
            unloading_point_id = 0
        
        if not loading_point or not unloading_point:
            return False
        
        # 创建三段任务模板
        template = [
            {
                'task_type': 'to_loading',
                'goal': (loading_point[0], loading_point[1], 0),
                'priority': 2,
                'loading_point_id': loading_point_id,
                'unloading_point_id': unloading_point_id
            },
            {
                'task_type': 'to_unloading',
                'goal': (unloading_point[0], unloading_point[1], 0),
                'priority': 3,
                'loading_point_id': loading_point_id,
                'unloading_point_id': unloading_point_id
            },
            {
                'task_type': 'to_initial',
                'goal': None,  # 将在分配时设置为车辆初始位置
                'priority': 1,
                'loading_point_id': loading_point_id,
                'unloading_point_id': unloading_point_id
            }
        ]
        
        self.mission_templates[template_id] = template
        return True
    
    def create_mission_with_specific_points(self, template_id, loading_point_id, unloading_point_id):
        """创建使用特定装载点和卸载点的任务模板"""
        if loading_point_id < 0 or loading_point_id >= len(self.env.loading_points):
            return False
        
        if unloading_point_id < 0 or unloading_point_id >= len(self.env.unloading_points):
            return False
        
        loading_point = self.env.loading_points[loading_point_id]
        unloading_point = self.env.unloading_points[unloading_point_id]
        
        return self.create_mission_template(template_id, loading_point, unloading_point)
    
    def assign_mission(self, vehicle_id, template_id):
        """为车辆分配任务模板"""
        if vehicle_id not in self.vehicle_statuses:
            return False
        
        if template_id not in self.mission_templates:
            return False
        
        # 获取模板
        template = self.mission_templates[template_id]
        
        # 获取车辆信息
        vehicle = self.env.vehicles.get(vehicle_id)
        if not vehicle:
            return False
        
        initial_position = vehicle.get('initial_position')
        if not initial_position:
            initial_position = vehicle.get('position', (0, 0, 0))
        
        # 创建任务并添加到队列
        current_position = self.vehicle_statuses[vehicle_id]['position']
        
        for task_template in template:
            task_id = f"task_{self.task_counter}"
            self.task_counter += 1
            
            # 创建任务
            task = VehicleTask(
                task_id,
                task_template['task_type'],
                current_position,
                task_template['goal'] if task_template['goal'] else initial_position,
                task_template['priority'],
                task_template.get('loading_point_id'),
                task_template.get('unloading_point_id')
            )
            
            # 更新下一任务的起点
            current_position = task.goal
            
            # 添加到任务字典
            self.tasks[task_id] = task
            
            # 添加到车辆任务队列
            self.task_queues[vehicle_id].append(task_id)
            
            # 更新车辆状态中的任务队列信息
            self.vehicle_statuses[vehicle_id]['task_queue'].append({
                'task_id': task_id,
                'task_type': task.task_type,
                'start': task.start,
                'goal': task.goal,
                'priority': task.priority
            })
            
            # 更新点位使用情况
            if task.loading_point_id is not None:
                self.loading_point_usage[task.loading_point_id] = \
                    self.loading_point_usage.get(task.loading_point_id, 0) + 1
            
            if task.unloading_point_id is not None:
                self.unloading_point_usage[task.unloading_point_id] = \
                    self.unloading_point_usage.get(task.unloading_point_id, 0) + 1
        
        # 如果车辆空闲，立即开始执行任务
        if self.vehicle_statuses[vehicle_id]['status'] == 'idle':
            self._start_next_task(vehicle_id)
        
        return True
    
    def assign_optimal_mission(self, vehicle_id):
        """基于距离、负载等因素为车辆分配最优任务"""
        if vehicle_id not in self.vehicle_statuses:
            return False
        
        vehicle_pos = self.vehicle_statuses[vehicle_id]['position']
        vehicle_load = self.vehicle_statuses[vehicle_id]['load']
        vehicle_max_load = self.vehicle_statuses[vehicle_id]['max_load']
        
        # 创建唯一的模板ID
        template_id = f"optimal_mission_{vehicle_id}_{self.task_counter}"
        
        # 根据负载情况分配任务
        if vehicle_load < vehicle_max_load * 0.5:
            best_loading_id = self._find_optimal_loading_point(vehicle_id, vehicle_pos)
            best_unloading_id = self._find_optimal_unloading_point(vehicle_id)
        else:
            best_unloading_id = self._find_optimal_unloading_point(vehicle_id, vehicle_pos)
            best_loading_id = self._find_optimal_loading_point(vehicle_id)
        
        # 创建并分配任务模板
        if self.create_mission_with_specific_points(template_id, best_loading_id, best_unloading_id):
            return self.assign_mission(vehicle_id, template_id)
        
        return False
    
    def _find_optimal_loading_point(self, vehicle_id, position=None):
        """寻找最优装载点"""
        if not self.env.loading_points:
            return 0
        
        if position is None and vehicle_id in self.vehicle_statuses:
            position = self.vehicle_statuses[vehicle_id]['position']
        
        # 如果车辆有首选装载点，优先使用
        if (vehicle_id in self.vehicle_statuses and 
            self.vehicle_statuses[vehicle_id]['preferred_loading_point'] is not None):
            return self.vehicle_statuses[vehicle_id]['preferred_loading_point']
        
        best_point_id = 0
        best_score = float('-inf')
        
        for i, point in enumerate(self.env.loading_points):
            # 计算距离分数
            distance = self._calculate_distance(position, point)
            distance_score = 1000 / (distance + 1)
            
            # 计算使用情况分数
            usage_count = self.loading_point_usage.get(i, 0)
            usage_score = 1000 / (usage_count + 1)
            
            # 检查骨干网络可达性加分
            backbone_score = 1.0
            if self.backbone_network:
                accessible_points = self.backbone_network.find_accessible_points(
                    position, None, max_candidates=1
                )
                if accessible_points:
                    backbone_score = 1.2
            
            # 总分
            score = distance_score * 0.6 + usage_score * 0.3 + backbone_score * 0.1
            
            if score > best_score:
                best_score = score
                best_point_id = i
        
        return best_point_id
    
    def _find_optimal_unloading_point(self, vehicle_id, position=None):
        """寻找最优卸载点"""
        if not self.env.unloading_points:
            return 0
        
        if position is None and vehicle_id in self.vehicle_statuses:
            position = self.vehicle_statuses[vehicle_id]['position']
        
        # 如果车辆有首选卸载点，优先使用
        if (vehicle_id in self.vehicle_statuses and 
            self.vehicle_statuses[vehicle_id]['preferred_unloading_point'] is not None):
            return self.vehicle_statuses[vehicle_id]['preferred_unloading_point']
        
        best_point_id = 0
        best_score = float('-inf')
        
        for i, point in enumerate(self.env.unloading_points):
            # 计算距离分数
            distance = self._calculate_distance(position, point)
            distance_score = 1000 / (distance + 1)
            
            # 计算使用情况分数
            usage_count = self.unloading_point_usage.get(i, 0)
            usage_score = 1000 / (usage_count + 1)
            
            # 检查骨干网络可达性
            backbone_score = 1.0
            if self.backbone_network:
                accessible_points = self.backbone_network.find_accessible_points(
                    position, None, max_candidates=1
                )
                if accessible_points:
                    backbone_score = 1.2
            
            # 总分
            score = distance_score * 0.6 + usage_score * 0.3 + backbone_score * 0.1
            
            if score > best_score:
                best_score = score
                best_point_id = i
        
        return best_point_id
    
    def _start_next_task(self, vehicle_id):
        """开始执行车辆的下一个任务"""
        if vehicle_id not in self.task_queues or not self.task_queues[vehicle_id]:
            return False
        
        # 获取下一个任务
        task_id = self.task_queues[vehicle_id][0]
        task = self.tasks[task_id]
        
        # 更新任务状态
        task.status = 'in_progress'
        task.assigned_vehicle = vehicle_id
        task.start_time = self.env.current_time if hasattr(self.env, 'current_time') else 0
        
        # 更新车辆状态
        self.vehicle_statuses[vehicle_id]['status'] = 'moving'
        self.vehicle_statuses[vehicle_id]['current_task'] = task_id
        
        # 规划路径
        if self.path_planner:
            path_result = self._plan_path_with_backbone(
                vehicle_id,
                self.vehicle_statuses[vehicle_id]['position'],
                task.goal
            )
            
            if path_result:
                if isinstance(path_result, tuple) and len(path_result) == 2:
                    # 包含结构信息的结果
                    task.path, structure = path_result
                    task.update_path_structure(structure)
                else:
                    # 只有路径的结果
                    task.path = path_result
                
                # 评估路径质量
                if self.quality_assessor:
                    task.quality_score = self.quality_assessor.evaluate_path(task.path)
                
                # 更新车辆路径
                if vehicle_id in self.env.vehicles and task.path:
                    self.env.vehicles[vehicle_id]['path'] = task.path
                    self.env.vehicles[vehicle_id]['path_index'] = 0
                    self.env.vehicles[vehicle_id]['progress'] = 0.0
                    
                    # 向交通管理器注册路径
                    if self.traffic_manager:
                        self.traffic_manager.register_vehicle_path(
                            vehicle_id,
                            task.path,
                            task.start_time
                        )
        
        return True
    
    def _plan_path_with_backbone(self, vehicle_id, start, goal):
        """使用骨干网络规划路径"""
        if not self.path_planner:
            return None
        
        # 如果有骨干网络且规划器支持，使用增强规划
        if self.backbone_network and hasattr(self.path_planner, 'plan_path'):
            # 确保规划器已设置骨干网络
            if hasattr(self.path_planner, 'set_backbone_network'):
                self.path_planner.set_backbone_network(self.backbone_network)
            
            # 规划路径，启用骨干网络
            result = self.path_planner.plan_path(vehicle_id, start, goal, use_backbone=True)
            
            # 检查结果格式
            if isinstance(result, tuple) and len(result) == 2:
                path, structure = result
                return path, structure
            elif result:
                # 只返回路径，分析结构
                structure = self._analyze_path_structure(result)
                return result, structure
        
        # 回退到普通路径规划
        if hasattr(self.path_planner, 'plan_path'):
            path = self.path_planner.plan_path(vehicle_id, start, goal)
            if path:
                structure = self._analyze_path_structure(path)
                return path, structure
        
        return None
    
    def _analyze_path_structure(self, path):
        """分析路径，识别骨干网络部分"""
        if not self.backbone_network or not path or len(path) < 2:
            return {
                'type': 'direct',
                'entry_point': None,
                'exit_point': None,
                'backbone_segment': None,
                'to_backbone_path': path,
                'backbone_path': None,
                'from_backbone_path': None
            }
        
        # 寻找进入和离开骨干网络的点
        entry_point = None
        entry_index = -1
        exit_point = None
        exit_index = -1
        backbone_segment = None
        
        # 遍历路径点，寻找最接近骨干网络的点
        for i, point in enumerate(path):
            # 查找最近的骨干连接点
            if hasattr(self.backbone_network, 'find_nearest_connection_optimized'):
                nearest_conn = self.backbone_network.find_nearest_connection_optimized(
                    point, max_distance=5.0
                )
            else:
                nearest_conn = None
            
            if nearest_conn:
                if entry_point is None:
                    entry_point = nearest_conn
                    entry_index = i
                    backbone_segment = nearest_conn.get('path_id') or nearest_conn.get('id')
                else:
                    exit_point = nearest_conn
                    exit_index = i
        
        # 根据找到的点划分路径
        if entry_index >= 0 and exit_index > entry_index:
            to_backbone = path[:entry_index+1]
            backbone = path[entry_index:exit_index+1]
            from_backbone = path[exit_index:]
            
            return {
                'type': 'three_segment',
                'entry_point': entry_point,
                'exit_point': exit_point,
                'backbone_segment': backbone_segment,
                'to_backbone_path': to_backbone,
                'backbone_path': backbone,
                'from_backbone_path': from_backbone
            }
        elif entry_index >= 0:
            to_backbone = path[:entry_index+1]
            backbone = path[entry_index:]
            
            return {
                'type': 'hybrid',
                'entry_point': entry_point,
                'exit_point': None,
                'backbone_segment': backbone_segment,
                'to_backbone_path': to_backbone,
                'backbone_path': backbone,
                'from_backbone_path': None
            }
        else:
            return {
                'type': 'direct',
                'entry_point': None,
                'exit_point': None,
                'backbone_segment': None,
                'to_backbone_path': path,
                'backbone_path': None,
                'from_backbone_path': None
            }
    
    def update(self, time_delta):
        """更新所有车辆状态和任务进度"""
        for vehicle_id, status in self.vehicle_statuses.items():
            self._update_vehicle(vehicle_id, time_delta)
    
    def _update_vehicle(self, vehicle_id, time_delta):
        """更新单个车辆状态"""
        if vehicle_id not in self.vehicle_statuses:
            return
        
        status = self.vehicle_statuses[vehicle_id]
        
        # 更新时间统计
        if status['status'] == 'idle':
            status['idle_time'] += time_delta
        else:
            status['active_time'] += time_delta
        
        # 计算利用率
        total_time = status['active_time'] + status['idle_time']
        if total_time > 0:
            status['utilization_rate'] = status['active_time'] / total_time
        
        # 更新车辆位置和状态
        vehicle = self.env.vehicles.get(vehicle_id)
        if not vehicle:
            return
        
        # 获取当前任务
        current_task_id = status['current_task']
        if not current_task_id or current_task_id not in self.tasks:
            # 如果没有当前任务，但有待执行任务，开始下一个任务
            if self.task_queues.get(vehicle_id) and status['status'] == 'idle':
                self._start_next_task(vehicle_id)
            return
        
        current_task = self.tasks[current_task_id]
        
        # 根据状态更新
        if status['status'] == 'moving':
            self._update_moving_vehicle(vehicle_id, vehicle, current_task, time_delta)
        elif status['status'] == 'loading':
            self._update_loading_vehicle(vehicle_id, vehicle, current_task, time_delta)
        elif status['status'] == 'unloading':
            self._update_unloading_vehicle(vehicle_id, vehicle, current_task, time_delta)
    
    def _update_moving_vehicle(self, vehicle_id, vehicle, current_task, time_delta):
        """更新移动中的车辆"""
        if 'path' in vehicle and vehicle['path']:
            path_index = vehicle.get('path_index', 0)
            path = vehicle['path']
            
            if path_index >= len(path):
                # 到达目标
                self._handle_task_completion(vehicle_id, current_task.task_id)
            else:
                # 更新路径进度
                progress = vehicle.get('progress', 0.0)
                speed = 1.0  # 基础速度
                
                # 根据路径质量调整速度
                if current_task.quality_score > 0:
                    speed *= (0.8 + 0.4 * current_task.quality_score)
                
                progress += (speed * time_delta * 0.05)
                
                if progress >= 1.0:
                    # 移动到下一个路径点
                    path_index += 1
                    progress = 0.0
                    
                    if path_index >= len(path):
                        # 到达目标
                        status = self.vehicle_statuses[vehicle_id]
                        status['position'] = current_task.goal
                        vehicle['position'] = current_task.goal
                        self._handle_task_completion(vehicle_id, current_task.task_id)
                    else:
                        # 更新位置到新的路径点
                        status = self.vehicle_statuses[vehicle_id]
                        status['position'] = path[path_index]
                        vehicle['position'] = path[path_index]
                        vehicle['path_index'] = path_index
                        vehicle['progress'] = progress
                        
                        # 检查骨干网络转换
                        self._check_backbone_transition(vehicle_id, current_task, path_index)
                else:
                    # 在当前路径段内插值计算位置
                    self._interpolate_position(vehicle_id, vehicle, path, path_index, progress)
                
                # 更新任务进度
                path_length = len(path)
                current_task.progress = min(1.0, (path_index + progress) / path_length)
    
    def _interpolate_position(self, vehicle_id, vehicle, path, path_index, progress):
        """在路径段内插值计算位置"""
        current_point = path[path_index]
        next_point = path[path_index + 1] if path_index + 1 < len(path) else current_point
        
        # 线性插值
        x = current_point[0] + (next_point[0] - current_point[0]) * progress
        y = current_point[1] + (next_point[1] - current_point[1]) * progress
        
        # 计算朝向角度
        dx = next_point[0] - current_point[0]
        dy = next_point[1] - current_point[1]
        if abs(dx) > 0.001 or abs(dy) > 0.001:
            theta = math.atan2(dy, dx)
        else:
            theta = current_point[2] if len(current_point) > 2 else 0
        
        # 更新位置
        status = self.vehicle_statuses[vehicle_id]
        status['position'] = (x, y, theta)
        vehicle['position'] = status['position']
        vehicle['progress'] = progress
    
    def _update_loading_vehicle(self, vehicle_id, vehicle, current_task, time_delta):
        """更新正在装载的车辆"""
        loading_time = 50  # 装载时间
        vehicle['loading_progress'] = vehicle.get('loading_progress', 0) + time_delta
        
        if vehicle['loading_progress'] >= loading_time:
            # 装载完成
            status = self.vehicle_statuses[vehicle_id]
            status['load'] = status['max_load']
            vehicle['load'] = status['max_load']
            
            # 完成当前任务
            self._complete_task(vehicle_id, current_task.task_id)
            
            # 开始下一个任务
            self._start_next_task(vehicle_id)
    
    def _update_unloading_vehicle(self, vehicle_id, vehicle, current_task, time_delta):
        """更新正在卸载的车辆"""
        unloading_time = 30  # 卸载时间
        vehicle['unloading_progress'] = vehicle.get('unloading_progress', 0) + time_delta
        
        if vehicle['unloading_progress'] >= unloading_time:
            # 卸载完成
            status = self.vehicle_statuses[vehicle_id]
            status['load'] = 0
            vehicle['load'] = 0
            
            # 完成当前任务
            self._complete_task(vehicle_id, current_task.task_id)
            
            # 开始下一个任务
            self._start_next_task(vehicle_id)
    
    def _check_backbone_transition(self, vehicle_id, task, path_index):
        """检查车辆是否进入或离开骨干网络"""
        if not task.path_structure or task.path_structure['type'] != 'three_segment':
            return
        
        # 计算路径段边界
        to_backbone_len = len(task.path_structure.get('to_backbone_path', []))
        backbone_len = len(task.path_structure.get('backbone_path', []))
        
        entry_index = to_backbone_len - 1 if to_backbone_len > 0 else -1
        exit_index = to_backbone_len + backbone_len - 1 if backbone_len > 0 else -1
        
        # 检查进入骨干网络
        if entry_index >= 0 and path_index == entry_index:
            if self.backbone_network and task.path_structure.get('backbone_segment'):
                # 更新骨干网络流量
                self._update_backbone_traffic(task.path_structure['backbone_segment'], 1)
        
        # 检查离开骨干网络
        if exit_index >= 0 and path_index == exit_index:
            if self.backbone_network and task.path_structure.get('backbone_segment'):
                # 更新骨干网络流量
                self._update_backbone_traffic(task.path_structure['backbone_segment'], -1)
    
    def _update_backbone_traffic(self, segment_id, change):
        """更新骨干网络交通流量"""
        if hasattr(self.backbone_network, 'update_traffic_flow'):
            self.backbone_network.update_traffic_flow(segment_id, change)
    
    def _handle_task_completion(self, vehicle_id, task_id):
        """处理任务完成"""
        if task_id not in self.tasks or vehicle_id not in self.vehicle_statuses:
            return
        
        task = self.tasks[task_id]
        status = self.vehicle_statuses[vehicle_id]
        
        # 根据任务类型执行不同操作
        if task.task_type == 'to_loading':
            # 到达装载点，开始装载
            status['status'] = 'loading'
            self.env.vehicles[vehicle_id]['status'] = 'loading'
            self.env.vehicles[vehicle_id]['loading_progress'] = 0
            
            # 记录使用的装载点
            if task.loading_point_id is not None:
                status['preferred_loading_point'] = task.loading_point_id
        
        elif task.task_type == 'to_unloading':
            # 到达卸载点，开始卸载
            status['status'] = 'unloading'
            self.env.vehicles[vehicle_id]['status'] = 'unloading'
            self.env.vehicles[vehicle_id]['unloading_progress'] = 0
            
            # 记录使用的卸载点
            if task.unloading_point_id is not None:
                status['preferred_unloading_point'] = task.unloading_point_id
        
        elif task.task_type == 'to_initial':
            # 返回起点，完成循环
            self._complete_task(vehicle_id, task_id)
            
            # 更新完成循环计数
            self.env.vehicles[vehicle_id]['completed_cycles'] = \
                self.env.vehicles[vehicle_id].get('completed_cycles', 0) + 1
            
            # 处理下一轮任务
            self._handle_cycle_completion(vehicle_id)
    
    def _handle_cycle_completion(self, vehicle_id):
        """处理循环完成"""
        if vehicle_id in self.task_queues:
            # 移除已完成的任务
            if self.task_queues[vehicle_id]:
                self.task_queues[vehicle_id].pop(0)
            
            # 如果还有任务模板，添加新循环
            if self.mission_templates:
                template_id = list(self.mission_templates.keys())[0]
                self.assign_mission(vehicle_id, template_id)
            
            # 开始下一个任务
            if self.task_queues[vehicle_id]:
                self._start_next_task(vehicle_id)
            else:
                # 没有更多任务，车辆空闲
                status = self.vehicle_statuses[vehicle_id]
                status['status'] = 'idle'
                status['current_task'] = None
                self.env.vehicles[vehicle_id]['status'] = 'idle'
    
    def _complete_task(self, vehicle_id, task_id):
        """完成任务并更新统计信息"""
        if task_id not in self.tasks:
            return
        
        task = self.tasks[task_id]
        
        # 更新任务状态
        task.status = 'completed'
        task.progress = 1.0
        task.completion_time = self.env.current_time if hasattr(self.env, 'current_time') else 0
        
        # 更新统计信息
        self.stats['completed_tasks'] += 1
        
        # 计算路径长度
        if task.path:
            path_length = self._calculate_distance_path(task.path)
            self.stats['total_distance'] += path_length
            self.vehicle_statuses[vehicle_id]['total_distance'] += path_length
        
        # 更新车辆完成任务计数
        self.vehicle_statuses[vehicle_id]['completed_tasks'] += 1
        
        # 从任务队列中移除
        if self.task_queues[vehicle_id] and self.task_queues[vehicle_id][0] == task_id:
            self.task_queues[vehicle_id].pop(0)
            
            # 同步更新车辆状态中的任务队列
            if self.vehicle_statuses[vehicle_id]['task_queue']:
                self.vehicle_statuses[vehicle_id]['task_queue'].pop(0)
        
        # 释放交通管理器中的路径
        if self.traffic_manager:
            self.traffic_manager.release_vehicle_path(vehicle_id)
        
        # 清理骨干网络上的流量
        if (self.backbone_network and task.path_structure and 
            task.path_structure.get('backbone_segment')):
            self._update_backbone_traffic(task.path_structure['backbone_segment'], -1)
    
    def get_vehicle_info(self, vehicle_id):
        """获取车辆详细信息"""
        if vehicle_id not in self.vehicle_statuses:
            return None
        
        info = self.vehicle_statuses[vehicle_id].copy()
        
        # 添加当前任务信息
        if info['current_task'] and info['current_task'] in self.tasks:
            task = self.tasks[info['current_task']]
            task_info = task.to_dict()
            
            # 添加路径结构信息
            if task.path_structure:
                task_info['path_structure'] = {}
                for key, value in task.path_structure.items():
                    if key in ['entry_point', 'exit_point', 'backbone_segment', 'type']:
                        task_info['path_structure'][key] = value
                    elif key in ['to_backbone_path', 'backbone_path', 'from_backbone_path'] and value:
                        task_info['path_structure'][key] = len(value)
            
            info['current_task_info'] = task_info
        
        return info
    
    def get_stats(self):
        """获取调度统计信息"""
        # 更新最新统计数据
        for vehicle_id, status in self.vehicle_statuses.items():
            self.stats['vehicle_utilization'][vehicle_id] = status['utilization_rate']
        
        # 计算全局利用率
        if self.vehicle_statuses:
            total_utilization = sum(self.stats['vehicle_utilization'].values())
            self.stats['average_utilization'] = total_utilization / len(self.vehicle_statuses)
        
        # 计算骨干网络使用效率
        if self.tasks:
            backbone_tasks = [t for t in self.tasks.values() 
                            if t.backbone_utilization > 0]
            if backbone_tasks:
                avg_backbone_util = sum(t.backbone_utilization for t in backbone_tasks) / len(backbone_tasks)
                self.stats['backbone_usage_efficiency'] = avg_backbone_util
        
        return self.stats
    
    def _calculate_distance(self, pos1, pos2):
        """计算两点之间的欧几里得距离"""
        x1 = pos1[0] if len(pos1) > 0 else 0
        y1 = pos1[1] if len(pos1) > 1 else 0
        x2 = pos2[0] if len(pos2) > 0 else 0
        y2 = pos2[1] if len(pos2) > 1 else 0
        
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    
    def _calculate_distance_path(self, path):
        """计算路径总长度"""
        if not path or len(path) < 2:
            return 0
        
        length = 0
        for i in range(len(path) - 1):
            length += self._calculate_distance(path[i], path[i + 1])
        
        return length


class ECBSVehicleScheduler(VehicleScheduler):
    """ECBS增强版车辆调度器 - 基于优化后的VehicleScheduler"""
    
    def __init__(self, env, path_planner=None, traffic_manager=None, backbone_network=None):
        super().__init__(env, path_planner, backbone_network, traffic_manager)
        
        # ECBS特有属性
        self.vehicle_priorities = {}  # 车辆优先级
        self.task_priorities = {}     # 任务优先级
        self.conflict_counts = {}     # 车辆冲突计数
        
        # 冲突解决策略
        self.conflict_resolution_strategy = 'ecbs'
        
        # 批量规划参数
        self.batch_planning = True
        self.max_batch_size = 10
        self.planning_horizon = 100.0
        
        # 性能优化
        self.parallel_execution = True
        self.max_parallel_vehicles = 5
    
    def initialize_vehicles(self):
        """初始化车辆状态，并设置初始优先级"""
        super().initialize_vehicles()
        
        # 根据车辆ID设置初始优先级
        for i, vehicle_id in enumerate(self.vehicle_statuses.keys()):
            self.vehicle_priorities[vehicle_id] = len(self.vehicle_statuses) - i
            self.conflict_counts[vehicle_id] = 0
    
    def assign_tasks_batch(self, tasks_batch):
        """使用ECBS原理批量分配任务"""
        if not tasks_batch:
            return {}
        
        # 按优先级排序任务
        sorted_tasks = sorted(
            tasks_batch, 
            key=lambda t: self.task_priorities.get(t.task_id, 1),
            reverse=True
        )
        
        # 初始化分配结果
        assignments = {}
        
        # 为每个任务找到最佳车辆
        for task in sorted_tasks:
            best_vehicle = self._find_best_vehicle_for_task(task)
            
            if best_vehicle:
                assignments[best_vehicle] = task.task_id
                self.assign_task(best_vehicle, task.task_id)
        
        # 如果设置了批量规划，为所有分配的车辆规划协调路径
        if self.batch_planning and assignments:
            vehicle_ids = list(assignments.keys())
            self._plan_coordinated_paths(vehicle_ids)
        
        return assignments
    
    def assign_task(self, vehicle_id, task_id):
        """分配单个任务给车辆"""
        if vehicle_id not in self.vehicle_statuses or task_id not in self.tasks:
            return False
        
        # 标记任务为已分配
        task = self.tasks[task_id]
        task.status = 'assigned'
        task.assigned_vehicle = vehicle_id
        
        # 添加任务到车辆队列
        if vehicle_id not in self.task_queues:
            self.task_queues[vehicle_id] = []
        
        self.task_queues[vehicle_id].append(task_id)
        
        # 如果车辆空闲，立即开始任务
        if self.vehicle_statuses[vehicle_id]['status'] == 'idle':
            self._start_next_task(vehicle_id)
        
        return True
    
    def _find_best_vehicle_for_task(self, task):
        """为任务找到最佳车辆"""
        best_vehicle = None
        best_score = float('-inf')
        
        for vehicle_id, status in self.vehicle_statuses.items():
            # 如果车辆任务过多，跳过
            if vehicle_id in self.task_queues and len(self.task_queues[vehicle_id]) > 3:
                continue
            
            # 计算综合评分
            distance = self._calculate_distance(status['position'], task.start)
            distance_score = 1000 / (distance + 10)
            
            # 负载匹配度
            load_factor = self._calculate_load_factor(task, status)
            
            # 优先级因素
            priority_factor = self.vehicle_priorities.get(vehicle_id, 1) / 10.0
            
            # 冲突历史因素
            conflict_factor = 1.0 / (1.0 + 0.1 * self.conflict_counts.get(vehicle_id, 0))
            
            # 点位偏好匹配
            preference_factor = self._calculate_preference_factor(task, status)
            
            # 总分
            score = (distance_score * 0.3 + load_factor * 0.25 + 
                    priority_factor * 0.2 + conflict_factor * 0.15 + 
                    preference_factor * 0.1)
            
            # 空闲车辆加分
            if status['status'] == 'idle':
                score *= 1.5
            
            if score > best_score:
                best_score = score
                best_vehicle = vehicle_id
        
        return best_vehicle
    
    def _calculate_load_factor(self, task, status):
        """计算负载匹配因子"""
        load_ratio = status['load'] / status['max_load']
        
        if task.task_type == 'to_unloading' and load_ratio < 0.5:
            return 0.5  # 空车去卸载点不合适
        elif task.task_type == 'to_loading' and load_ratio > 0.5:
            return 0.5  # 满载车去装载点不合适
        else:
            return 1.0
    
    def _calculate_preference_factor(self, task, status):
        """计算点位偏好匹配因子"""
        factor = 1.0
        
        if (task.task_type == 'to_loading' and 
            status['preferred_loading_point'] is not None and
            task.loading_point_id == status['preferred_loading_point']):
            factor = 1.5
        
        if (task.task_type == 'to_unloading' and 
            status['preferred_unloading_point'] is not None and
            task.unloading_point_id == status['preferred_unloading_point']):
            factor = 1.5
        
        return factor
    
    def _plan_coordinated_paths(self, vehicle_ids):
        """为多个车辆规划协调的无冲突路径"""
        if not self.traffic_manager or not vehicle_ids:
            return False
        
        # 收集车辆路径需求
        paths = {}
        vehicle_tasks = {}
        path_structures = {}
        
        for vehicle_id in vehicle_ids:
            task_id = self.vehicle_statuses[vehicle_id]['current_task']
            if not task_id or task_id not in self.tasks:
                continue
            
            task = self.tasks[task_id]
            vehicle_tasks[vehicle_id] = task
            
            # 规划初始路径
            path_result = self._plan_structured_path(vehicle_id, task.start, task.goal)
            
            if path_result:
                if isinstance(path_result, tuple):
                    path, structure = path_result
                    paths[vehicle_id] = path
                    path_structures[vehicle_id] = structure
                else:
                    paths[vehicle_id] = path_result
                    path_structures[vehicle_id] = self._analyze_path_structure(path_result)
        
        # 使用交通管理器解决冲突
        if len(paths) > 1:
            conflict_free_paths = self.traffic_manager.resolve_conflicts(
                paths, 
                backbone_network=self.backbone_network,
                path_structures=path_structures
            )
            
            if conflict_free_paths:
                # 更新任务和车辆的路径
                for vehicle_id, path in conflict_free_paths.items():
                    if vehicle_id in vehicle_tasks:
                        task = vehicle_tasks[vehicle_id]
                        task.path = path
                        task.update_path_structure(path_structures.get(vehicle_id, {}))
                        
                        # 注册路径到交通管理器
                        self.traffic_manager.register_vehicle_path(
                            vehicle_id, path,
                            self.env.current_time if hasattr(self.env, 'current_time') else 0
                        )
                        
                        # 更新车辆路径
                        if vehicle_id in self.env.vehicles:
                            self.env.vehicles[vehicle_id]['path'] = path
                            self.env.vehicles[vehicle_id]['path_index'] = 0
                            self.env.vehicles[vehicle_id]['progress'] = 0.0
                
                return True
        
        return False
    
    def _plan_structured_path(self, vehicle_id, start, goal):
        """规划具有结构的路径"""
        if not self.path_planner or not self.backbone_network:
            # 没有骨干网络，使用普通规划
            if self.path_planner:
                path = self.path_planner.plan_path(vehicle_id, start, goal)
                return path, {'type': 'direct', 'to_backbone_path': path}
            return None
        
        # 使用增强的路径规划器
        result = self.path_planner.plan_path(vehicle_id, start, goal, use_backbone=True)
        
        if isinstance(result, tuple):
            return result
        elif result:
            structure = self._analyze_path_structure(result)
            return result, structure
        
        return None
    
    def update(self, time_delta):
        """更新所有车辆状态，包括ECBS冲突检测和解决"""
        # 首先更新车辆状态
        super().update(time_delta)
        
        # 检查和解决执行中的冲突
        if self.traffic_manager:
            self._check_and_resolve_execution_conflicts()
        
        # 选择下一批任务
        self._select_next_batch_tasks()
        
        return True
    
    def _check_and_resolve_execution_conflicts(self):
        """检查并解决执行中的路径冲突"""
        # 收集当前活动的车辆和路径
        active_vehicles = []
        active_paths = {}
        path_structures = {}
        
        for vehicle_id, status in self.vehicle_statuses.items():
            if status['status'] == 'moving' and status['current_task']:
                active_vehicles.append(vehicle_id)
                
                task_id = status['current_task']
                if task_id in self.tasks and self.tasks[task_id].path:
                    # 获取剩余路径
                    if vehicle_id in self.env.vehicles:
                        vehicle = self.env.vehicles[vehicle_id]
                        path_index = vehicle.get('path_index', 0)
                        path = self.tasks[task_id].path[path_index:]
                        
                        active_paths[vehicle_id] = path
                        
                        # 保存路径结构信息
                        task = self.tasks[task_id]
                        if task.path_structure:
                            path_structures[vehicle_id] = task.path_structure
        
        # 如果有多个活动车辆，检查冲突
        if len(active_vehicles) > 1:
            conflicts = self.traffic_manager.detect_conflicts(active_paths)
            
            if conflicts:
                # 记录冲突车辆
                for conflict in conflicts:
                    self.conflict_counts[conflict.agent1] = \
                        self.conflict_counts.get(conflict.agent1, 0) + 1
                    self.conflict_counts[conflict.agent2] = \
                        self.conflict_counts.get(conflict.agent2, 0) + 1
                
                # 统计冲突解决次数
                self.stats['conflict_resolution_count'] += len(conflicts)
                
                # 使用ECBS解决冲突
                if self.conflict_resolution_strategy == 'ecbs':
                    new_paths = self.traffic_manager.resolve_conflicts(
                        active_paths,
                        backbone_network=self.backbone_network,
                        path_structures=path_structures
                    )
                    
                    if new_paths:
                        self._apply_conflict_resolution(new_paths, active_vehicles)
    
    def _apply_conflict_resolution(self, new_paths, active_vehicles):
        """应用冲突解决结果"""
        for vehicle_id, path in new_paths.items():
            if vehicle_id in active_vehicles:
                task_id = self.vehicle_statuses[vehicle_id]['current_task']
                if task_id in self.tasks:
                    # 更新任务路径
                    self.tasks[task_id].path = path
                    self.tasks[task_id].update_path_structure(
                        self._analyze_path_structure(path)
                    )
                    
                    # 更新车辆路径
                    self.env.vehicles[vehicle_id]['path'] = path
                    self.env.vehicles[vehicle_id]['path_index'] = 0
                    self.env.vehicles[vehicle_id]['progress'] = 0.0
                    
                    # 重新注册路径
                    self.traffic_manager.release_vehicle_path(vehicle_id)
                    self.traffic_manager.register_vehicle_path(
                        vehicle_id, path,
                        self.env.current_time if hasattr(self.env, 'current_time') else 0
                    )
    
    def _select_next_batch_tasks(self):
        """选择下一批要执行的任务"""
        # 寻找空闲车辆
        available_vehicles = [
            vehicle_id for vehicle_id, status in self.vehicle_statuses.items()
            if (status['status'] == 'idle' and 
                vehicle_id in self.task_queues and 
                self.task_queues[vehicle_id])
        ]
        
        if not available_vehicles:
            return
        
        # 按优先级排序
        sorted_vehicles = sorted(
            available_vehicles,
            key=lambda v: self.vehicle_priorities.get(v, 1),
            reverse=True
        )
        
        # 限制同时执行的车辆数量
        batch_vehicles = sorted_vehicles[:self.max_parallel_vehicles]
        
        # 为选定车辆启动任务
        if len(batch_vehicles) > 1 and self.batch_planning:
            self._plan_coordinated_paths(batch_vehicles)
        else:
            for vehicle_id in batch_vehicles:
                self._start_next_task(vehicle_id)
    
    def create_ecbs_mission_template(self, template_id, loading_point=None, unloading_point=None):
        """创建带有ECBS特性的任务模板"""
        if not super().create_mission_template(template_id, loading_point, unloading_point):
            return False
        
        # 增强任务优先级设置
        template = self.mission_templates[template_id]
        
        for task in template:
            if task['task_type'] == 'to_loading':
                task['priority'] = 2
            elif task['task_type'] == 'to_unloading':
                task['priority'] = 3
            elif task['task_type'] == 'to_initial':
                task['priority'] = 1
        
        return True
    
    def assign_mission(self, vehicle_id, template_id):
        """以ECBS感知方式分配任务"""
        if not super().assign_mission(vehicle_id, template_id):
            return False
        
        # 更新任务优先级
        current_prio = self.vehicle_priorities.get(vehicle_id, 1)
        
        if vehicle_id in self.task_queues:
            for task_id in self.task_queues[vehicle_id]:
                if task_id in self.tasks:
                    self.tasks[task_id].priority *= current_prio / 10.0
                    self.task_priorities[task_id] = self.tasks[task_id].priority
        
        return True
    
    def get_vehicle_info(self, vehicle_id):
        """获取车辆详细信息，包括ECBS特有信息"""
        info = super().get_vehicle_info(vehicle_id)
        
        if not info:
            return None
        
        # 添加ECBS特有信息
        info['priority'] = self.vehicle_priorities.get(vehicle_id, 1)
        info['conflict_count'] = self.conflict_counts.get(vehicle_id, 0)
        
        return info
    
    def get_stats(self):
        """获取调度统计信息，包括ECBS特有信息"""
        stats = super().get_stats()
        
        # 添加ECBS特有信息
        stats['conflict_resolution'] = {
            'strategy': self.conflict_resolution_strategy,
            'total_conflicts': sum(self.conflict_counts.values()),
            'conflicts_by_vehicle': self.conflict_counts.copy(),
            'resolution_count': self.stats.get('conflict_resolution_count', 0)
        }
        
        return stats


# 简化的路径质量评估器
class PathQualityAssessor:
    """路径质量评估器"""
    
    def __init__(self):
        self.weights = {
            'length_efficiency': 0.3,
            'smoothness': 0.3,
            'backbone_usage': 0.4
        }
    
    def evaluate_path(self, path):
        """评估路径质量"""
        if not path or len(path) < 2:
            return 0
        
        # 长度效率
        length_score = self._evaluate_length_efficiency(path)
        
        # 平滑度
        smoothness_score = self._evaluate_smoothness(path)
        
        # 骨干网络使用（简化）
        backbone_score = 0.5  # 默认值
        
        # 综合评分
        total_score = (
            length_score * self.weights['length_efficiency'] +
            smoothness_score * self.weights['smoothness'] +
            backbone_score * self.weights['backbone_usage']
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
        
        total_angle_change = 0
        for i in range(1, len(path) - 1):
            angle_change = self._calculate_angle_change(path[i-1], path[i], path[i+1])
            total_angle_change += angle_change
        
        # 归一化
        avg_angle_change = total_angle_change / max(1, len(path) - 2)
        smoothness = math.exp(-avg_angle_change)
        
        return min(1.0, smoothness)
    
    def _calculate_angle_change(self, p1, p2, p3):
        """计算角度变化"""
        v1 = [p2[0] - p1[0], p2[1] - p1[1]]
        v2 = [p3[0] - p2[0], p3[1] - p2[1]]
        
        len1 = math.sqrt(v1[0]**2 + v1[1]**2)
        len2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if len1 < 0.001 or len2 < 0.001:
            return 0
        
        cos_angle = (v1[0]*v2[0] + v1[1]*v2[1]) / (len1 * len2)
        cos_angle = max(-1, min(1, cos_angle))
        
        return math.acos(cos_angle)
    
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