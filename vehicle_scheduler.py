"""
vehicle_scheduler.py - 优化版车辆调度器
修复移动逻辑，简化复杂功能，专注核心调度
"""

import math
import time
import threading
from typing import List, Dict, Tuple, Set, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class VehicleStatus(Enum):
    IDLE = "idle"
    MOVING = "moving"
    LOADING = "loading"
    UNLOADING = "unloading"
    WAITING = "waiting"
    MAINTENANCE = "maintenance"

@dataclass
class VehicleTask:
    """车辆任务类"""
    task_id: str
    task_type: str
    start: Tuple[float, float, float]
    goal: Tuple[float, float, float]
    priority: int = 1
    status: TaskStatus = TaskStatus.PENDING
    
    # 任务属性
    loading_point_id: Optional[int] = None
    unloading_point_id: Optional[int] = None
    deadline: Optional[float] = None
    estimated_duration: float = 0
    
    # 分配信息
    assigned_vehicle: Optional[str] = None
    assignment_time: float = 0
    start_time: float = 0
    completion_time: float = 0
    
    # 路径信息
    path: Optional[List] = None
    path_structure: Dict = field(default_factory=dict)
    quality_score: float = 0.0
    
    # 性能指标
    planning_time: float = 0
    execution_progress: float = 0
    actual_duration: float = 0
    
    def update_progress(self, progress: float):
        """更新执行进度"""
        self.execution_progress = max(0, min(1.0, progress))
        if self.execution_progress >= 1.0:
            self.status = TaskStatus.COMPLETED
            self.completion_time = time.time()
            self.actual_duration = self.completion_time - self.start_time

@dataclass
class VehicleState:
    """车辆状态"""
    vehicle_id: str
    status: VehicleStatus = VehicleStatus.IDLE
    position: Tuple[float, float, float] = (0, 0, 0)
    
    # 基本属性
    max_load: float = 100
    current_load: float = 0
    speed: float = 1.0
    
    # 任务相关
    current_task: Optional[str] = None
    completed_tasks: int = 0
    
    # 性能统计
    total_distance: float = 0
    total_time: float = 0
    idle_time: float = 0
    utilization_rate: float = 0
    
    # 效率统计
    backbone_usage_count: int = 0
    direct_path_count: int = 0
    interface_efficiency: float = 0.5
    
    def update_utilization(self, time_delta: float):
        """更新利用率"""
        if self.status != VehicleStatus.IDLE:
            self.total_time += time_delta
        else:
            self.idle_time += time_delta
        
        total_elapsed = self.total_time + self.idle_time
        if total_elapsed > 0:
            self.utilization_rate = self.total_time / total_elapsed

class SimplifiedVehicleScheduler:
    """简化版车辆调度器 - 专注核心功能"""
    
    def __init__(self, env, path_planner=None, backbone_network=None, traffic_manager=None):
        self.env = env
        self.path_planner = path_planner
        self.backbone_network = backbone_network
        self.traffic_manager = traffic_manager
        
        # 数据存储
        self.tasks = {}  # {task_id: VehicleTask}
        self.vehicle_states = {}  # {vehicle_id: VehicleState}
        self.mission_templates = {}  # {template_id: mission_config}
        
        # 任务管理
        self.task_counter = 0
        self.active_assignments = {}  # {vehicle_id: [task_ids]}
        
        # 统计信息
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'average_completion_time': 0,
            'total_distance': 0,
            'backbone_utilization_rate': 0,
            'vehicle_utilization': {}
        }
        
        print("初始化简化版车辆调度器")
    
    def initialize_vehicles(self):
        """初始化车辆状态"""
        for vehicle_id, vehicle_data in self.env.vehicles.items():
            position = vehicle_data.get('position', (0, 0, 0))
            max_load = vehicle_data.get('max_load', 100)
            
            self.vehicle_states[vehicle_id] = VehicleState(
                vehicle_id=vehicle_id,
                position=position,
                max_load=max_load,
                current_load=vehicle_data.get('load', 0)
            )
            
            self.active_assignments[vehicle_id] = []
        
        print(f"初始化了 {len(self.vehicle_states)} 个车辆状态")
    
    def create_enhanced_mission_template(self, template_id: str, 
                                       loading_point_id: int = None, 
                                       unloading_point_id: int = None) -> bool:
        """创建任务模板"""
        if not self.env.loading_points or not self.env.unloading_points:
            return False
        
        # 智能选择装载点和卸载点
        if loading_point_id is None:
            loading_point_id = 0
        
        if unloading_point_id is None:
            unloading_point_id = 0
        
        # 验证点位有效性
        if (loading_point_id >= len(self.env.loading_points) or 
            unloading_point_id >= len(self.env.unloading_points)):
            return False
        
        loading_point = self.env.loading_points[loading_point_id]
        unloading_point = self.env.unloading_points[unloading_point_id]
        
        # 创建任务模板
        template = {
            'loading_point_id': loading_point_id,
            'unloading_point_id': unloading_point_id,
            'loading_position': loading_point,
            'unloading_position': unloading_point,
            'tasks': [
                {
                    'task_type': 'to_loading',
                    'goal': loading_point,
                    'priority': 2,
                    'estimated_duration': 180
                },
                {
                    'task_type': 'to_unloading',
                    'goal': unloading_point,
                    'priority': 3,
                    'estimated_duration': 150
                },
                {
                    'task_type': 'to_initial',
                    'goal': None,
                    'priority': 1,
                    'estimated_duration': 120
                }
            ]
        }
        
        self.mission_templates[template_id] = template
        return True
    
    # 兼容性方法
    def create_mission_template(self, template_id: str) -> bool:
        """兼容性方法"""
        return self.create_enhanced_mission_template(template_id)
    
    def assign_mission_intelligently(self, vehicle_id: str, template_id: str = None) -> bool:
        """智能分配任务"""
        if vehicle_id not in self.vehicle_states:
            return False
        
        # 自动选择最佳模板
        if template_id is None:
            template_id = "default"
            if template_id not in self.mission_templates:
                self.create_enhanced_mission_template(template_id)
        
        if template_id not in self.mission_templates:
            return False
        
        template = self.mission_templates[template_id]
        vehicle_state = self.vehicle_states[vehicle_id]
        
        # 生成任务
        created_tasks = []
        current_position = vehicle_state.position
        
        for task_template in template['tasks']:
            task_id = f"task_{self.task_counter}"
            self.task_counter += 1
            
            # 确定目标位置
            if task_template['goal'] is None:
                # 返回初始位置
                goal = self.env.vehicles[vehicle_id].get('position', current_position)
            else:
                goal = task_template['goal']
            
            # 创建任务
            task = VehicleTask(
                task_id=task_id,
                task_type=task_template['task_type'],
                start=current_position,
                goal=goal,
                priority=task_template['priority'],
                loading_point_id=template.get('loading_point_id'),
                unloading_point_id=template.get('unloading_point_id'),
                estimated_duration=task_template['estimated_duration']
            )
            
            self.tasks[task_id] = task
            created_tasks.append(task_id)
            
            # 更新下一任务的起点
            current_position = goal
        
        # 分配任务到车辆
        self.active_assignments[vehicle_id].extend(created_tasks)
        
        # 如果车辆空闲，立即开始第一个任务
        if vehicle_state.status == VehicleStatus.IDLE:
            self._start_next_task(vehicle_id)
        
        self.stats['total_tasks'] += len(created_tasks)
        
        print(f"为车辆 {vehicle_id} 分配了 {len(created_tasks)} 个任务")
        return True
    
    # 兼容性方法
    def assign_mission(self, vehicle_id: str, template_id: str) -> bool:
        """兼容性方法"""
        return self.assign_mission_intelligently(vehicle_id, template_id)
    
    def _start_next_task(self, vehicle_id: str) -> bool:
        """开始车辆的下一个任务"""
        if vehicle_id not in self.active_assignments:
            return False
        
        assignments = self.active_assignments[vehicle_id]
        if not assignments:
            return False
        
        task_id = assignments[0]
        if task_id not in self.tasks:
            assignments.remove(task_id)
            return self._start_next_task(vehicle_id)
        
        task = self.tasks[task_id]
        vehicle_state = self.vehicle_states[vehicle_id]
        
        # 更新任务状态
        task.status = TaskStatus.IN_PROGRESS
        task.assigned_vehicle = vehicle_id
        task.start_time = time.time()
        
        # 更新车辆状态
        vehicle_state.status = VehicleStatus.MOVING
        vehicle_state.current_task = task_id
        
        # 同步到环境
        if vehicle_id in self.env.vehicles:
            self.env.vehicles[vehicle_id]['status'] = 'moving'
        
        # 规划路径
        success = self._plan_task_path(task, vehicle_state)
        
        if success:
            print(f"车辆 {vehicle_id} 开始任务 {task_id} ({task.task_type})")
            return True
        else:
            # 路径规划失败，标记任务失败
            task.status = TaskStatus.FAILED
            vehicle_state.status = VehicleStatus.IDLE
            self.stats['failed_tasks'] += 1
            
            assignments.remove(task_id)
            return self._start_next_task(vehicle_id)
    
    def _plan_task_path(self, task: VehicleTask, vehicle: VehicleState) -> bool:
        """为任务规划路径"""
        if not self.path_planner:
            return False
        
        planning_start = time.time()
        
        try:
            result = self.path_planner.plan_path(
                vehicle.vehicle_id, 
                task.start, 
                task.goal,
                use_backbone=True,
                check_conflicts=True
            )
            
            if result:
                if isinstance(result, tuple) and len(result) == 2:
                    path, structure = result
                    task.path = path
                    task.path_structure = structure
                    task.quality_score = structure.get('final_quality', 0.5)
                    
                    # 注册到交通管理器
                    if self.traffic_manager:
                        self.traffic_manager.register_vehicle_path_enhanced(
                            vehicle.vehicle_id, path, structure, task.start_time
                        )
                    
                    # 更新车辆路径信息
                    if vehicle.vehicle_id in self.env.vehicles:
                        env_vehicle = self.env.vehicles[vehicle.vehicle_id]
                        env_vehicle['path'] = path
                        env_vehicle['path_index'] = 0
                        env_vehicle['progress'] = 0.0
                        env_vehicle['path_structure'] = structure
                    
                    # 更新统计
                    if structure.get('type') in ['interface_assisted', 'backbone_only']:
                        vehicle.backbone_usage_count += 1
                    else:
                        vehicle.direct_path_count += 1
                else:
                    task.path = result
                    task.path_structure = {'type': 'direct'}
                    vehicle.direct_path_count += 1
                
                task.planning_time = time.time() - planning_start
                return True
        
        except Exception as e:
            print(f"任务 {task.task_id} 路径规划失败: {e}")
        
        return False

    def update(self, time_delta: float):
        """更新调度器状态"""
        # 更新所有车辆状态
        for vehicle_id, vehicle_state in self.vehicle_states.items():
            self._update_vehicle_state(vehicle_id, vehicle_state, time_delta)
        
        # 更新性能统计
        self._update_performance_stats(time_delta)
    
    def _update_vehicle_state(self, vehicle_id: str, vehicle_state: VehicleState, 
                             time_delta: float):
        """更新单个车辆状态"""
        # 更新利用率
        vehicle_state.update_utilization(time_delta)
        
        # 同步环境中的车辆信息
        if vehicle_id in self.env.vehicles:
            env_vehicle = self.env.vehicles[vehicle_id]
            
            # 只在车辆不在移动时同步位置
            if vehicle_state.status != VehicleStatus.MOVING:
                vehicle_state.position = env_vehicle.get('position', vehicle_state.position)
            
            vehicle_state.current_load = env_vehicle.get('load', vehicle_state.current_load)
            
            # 更新状态
            env_status = env_vehicle.get('status', 'idle')
            status_mapping = {
                'idle': VehicleStatus.IDLE,
                'moving': VehicleStatus.MOVING,
                'loading': VehicleStatus.LOADING,
                'unloading': VehicleStatus.UNLOADING
            }
            
            # 只在没有当前任务时从环境同步状态
            if not vehicle_state.current_task:
                vehicle_state.status = status_mapping.get(env_status, VehicleStatus.IDLE)
        
        # 处理当前任务进度
        if vehicle_state.current_task:
            self._update_task_progress(vehicle_state.current_task, vehicle_state, time_delta)
    
    def _update_task_progress(self, task_id: str, vehicle_state: VehicleState, 
                             time_delta: float):
        """更新任务进度 - 核心移动逻辑"""
        if task_id not in self.tasks:
            return
        
        task = self.tasks[task_id]
        
        # 根据车辆状态更新任务进度
        if vehicle_state.status == VehicleStatus.MOVING and task.path:
            # 车辆移动逻辑
            if vehicle_state.vehicle_id in self.env.vehicles:
                env_vehicle = self.env.vehicles[vehicle_state.vehicle_id]
                path = task.path
                path_length = len(path)
                
                if path_length > 1:
                    # 获取当前进度（0-1之间的浮点数）
                    current_progress = env_vehicle.get('progress', 0.0)
                    
                    # 计算移动距离
                    speed = vehicle_state.speed
                    
                    # 计算路径总长度
                    total_path_distance = self._calculate_path_total_distance(path)
                    
                    if total_path_distance > 0:
                        # 计算进度增量
                        distance_per_step = speed * time_delta
                        progress_increment = distance_per_step / total_path_distance
                        
                        # 更新进度
                        new_progress = min(1.0, current_progress + progress_increment)
                        env_vehicle['progress'] = new_progress
                        
                        # 根据进度计算新位置
                        new_position, new_index = self._calculate_position_from_progress(
                            path, new_progress
                        )
                        
                        # 更新环境中的车辆位置和索引
                        env_vehicle['position'] = new_position
                        env_vehicle['path_index'] = new_index
                        
                        # 更新车辆状态中的位置
                        vehicle_state.position = new_position
                        
                        # 更新任务进度
                        task.update_progress(new_progress)
                        
                        # 检查是否到达目标（进度>=95%认为到达）
                        if new_progress >= 0.95:
                            self._handle_task_arrival(task, vehicle_state)
        
        elif vehicle_state.status in [VehicleStatus.LOADING, VehicleStatus.UNLOADING]:
            # 装载/卸载过程中的进度更新
            operation_time = 60 if vehicle_state.status == VehicleStatus.LOADING else 40
            elapsed_time = time.time() - task.start_time
            
            progress = min(1.0, elapsed_time / operation_time)
            task.update_progress(progress)
            
            if progress >= 1.0:
                self._complete_current_task(vehicle_state.vehicle_id)

    def _calculate_path_total_distance(self, path: List) -> float:
        """计算路径总长度"""
        if not path or len(path) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(path) - 1):
            distance = math.sqrt(
                (path[i+1][0] - path[i][0])**2 + 
                (path[i+1][1] - path[i][1])**2
            )
            total_distance += distance
        
        return total_distance

    def _calculate_position_from_progress(self, path: List, progress: float) -> Tuple[Tuple, int]:
        """根据进度计算位置"""
        if not path or len(path) < 2:
            return path[0] if path else (0, 0, 0), 0
        
        if progress <= 0:
            return path[0], 0
        
        if progress >= 1.0:
            return path[-1], len(path) - 1
        
        # 计算总长度
        total_distance = self._calculate_path_total_distance(path)
        target_distance = total_distance * progress
        
        # 沿路径查找目标位置
        current_distance = 0.0
        
        for i in range(len(path) - 1):
            segment_distance = math.sqrt(
                (path[i+1][0] - path[i][0])**2 + 
                (path[i+1][1] - path[i][1])**2
            )
            
            if current_distance + segment_distance >= target_distance:
                # 在这个线段上
                remaining = target_distance - current_distance
                ratio = remaining / segment_distance if segment_distance > 0 else 0
                
                # 插值计算位置
                x = path[i][0] + ratio * (path[i+1][0] - path[i][0])
                y = path[i][1] + ratio * (path[i+1][1] - path[i][1])
                theta = path[i][2] if len(path[i]) > 2 else 0
                
                return (x, y, theta), i
            
            current_distance += segment_distance
        
        # 如果没找到，返回终点
        return path[-1], len(path) - 1

    def _handle_task_arrival(self, task: VehicleTask, vehicle_state: VehicleState):
        """处理任务到达目标"""
        if task.task_type == 'to_loading':
            # 到达装载点，开始装载
            vehicle_state.status = VehicleStatus.LOADING
            if vehicle_state.vehicle_id in self.env.vehicles:
                self.env.vehicles[vehicle_state.vehicle_id]['status'] = 'loading'
                self.env.vehicles[vehicle_state.vehicle_id]['progress'] = 0.0
            
            # 更新装载量
            vehicle_state.current_load = vehicle_state.max_load
            if vehicle_state.vehicle_id in self.env.vehicles:
                self.env.vehicles[vehicle_state.vehicle_id]['load'] = vehicle_state.max_load
            
            # 重置任务开始时间用于装载计时
            task.start_time = time.time()
            
            print(f"车辆 {vehicle_state.vehicle_id} 到达装载点，开始装载")
        
        elif task.task_type == 'to_unloading':
            # 到达卸载点，开始卸载
            vehicle_state.status = VehicleStatus.UNLOADING
            if vehicle_state.vehicle_id in self.env.vehicles:
                self.env.vehicles[vehicle_state.vehicle_id]['status'] = 'unloading'
                self.env.vehicles[vehicle_state.vehicle_id]['progress'] = 0.0
            
            # 重置任务开始时间用于卸载计时
            task.start_time = time.time()
            
            print(f"车辆 {vehicle_state.vehicle_id} 到达卸载点，开始卸载")
        
        elif task.task_type == 'to_initial':
            # 返回起点，完成一个循环
            self._complete_current_task(vehicle_state.vehicle_id)
            
            # 更新完成循环数
            if vehicle_state.vehicle_id in self.env.vehicles:
                env_vehicle = self.env.vehicles[vehicle_state.vehicle_id]
                env_vehicle['completed_cycles'] = env_vehicle.get('completed_cycles', 0) + 1
            
            print(f"车辆 {vehicle_state.vehicle_id} 完成一个工作循环")
            
            # 自动分配下一轮任务
            self._schedule_next_mission_cycle(vehicle_state.vehicle_id)

    def _complete_current_task(self, vehicle_id: str):
        """完成当前任务"""
        if vehicle_id not in self.vehicle_states:
            return
        
        vehicle_state = self.vehicle_states[vehicle_id]
        
        if not vehicle_state.current_task:
            return
        
        task_id = vehicle_state.current_task
        if task_id not in self.tasks:
            return
        
        task = self.tasks[task_id]
        
        # 完成任务
        task.status = TaskStatus.COMPLETED
        task.completion_time = time.time()
        task.actual_duration = task.completion_time - task.start_time
        
        # 处理装载/卸载后的状态变化
        if task.task_type == 'to_unloading':
            # 卸载后清空载重
            vehicle_state.current_load = 0
            if vehicle_id in self.env.vehicles:
                self.env.vehicles[vehicle_id]['load'] = 0
        
        # 从分配列表中移除
        if vehicle_id in self.active_assignments:
            if task_id in self.active_assignments[vehicle_id]:
                self.active_assignments[vehicle_id].remove(task_id)
        
        # 更新统计
        self.stats['completed_tasks'] += 1
        
        if task.path:
            path_distance = self._calculate_path_total_distance(task.path)
            self.stats['total_distance'] += path_distance
            vehicle_state.total_distance += path_distance
        
        # 释放交通管理器中的路径
        if self.traffic_manager:
            self.traffic_manager.release_vehicle_path(vehicle_id)
        
        # 清理车辆状态
        vehicle_state.current_task = None
        vehicle_state.status = VehicleStatus.IDLE
        vehicle_state.completed_tasks += 1
        
        if vehicle_id in self.env.vehicles:
            self.env.vehicles[vehicle_id]['status'] = 'idle'
            self.env.vehicles[vehicle_id]['progress'] = 0.0
            self.env.vehicles[vehicle_id]['path'] = None
            self.env.vehicles[vehicle_id]['path_index'] = 0
        
        print(f"任务 {task_id} 完成，车辆 {vehicle_id}，"
              f"用时 {task.actual_duration:.1f}s，质量 {task.quality_score:.2f}")
        
        # 开始下一个任务
        self._start_next_task(vehicle_id)
    
    def _schedule_next_mission_cycle(self, vehicle_id: str):
        """安排下一轮任务循环"""
        # 如果车辆没有更多任务，自动分配新的任务循环
        if (vehicle_id in self.active_assignments and 
            not self.active_assignments[vehicle_id]):
            
            # 使用默认模板自动分配
            self.assign_mission_intelligently(vehicle_id, "default")
    
    def _update_performance_stats(self, time_delta: float):
        """更新性能统计"""
        # 更新车辆利用率统计
        for vehicle_id, vehicle_state in self.vehicle_states.items():
            self.stats['vehicle_utilization'][vehicle_id] = vehicle_state.utilization_rate
        
        # 计算骨干网络利用率
        total_backbone_usage = sum(vehicle.backbone_usage_count 
                                 for vehicle in self.vehicle_states.values())
        total_path_usage = sum(vehicle.backbone_usage_count + vehicle.direct_path_count 
                             for vehicle in self.vehicle_states.values())
        
        if total_path_usage > 0:
            self.stats['backbone_utilization_rate'] = total_backbone_usage / total_path_usage
    
    def get_comprehensive_stats(self) -> Dict:
        """获取综合统计信息"""
        stats = self.stats.copy()
        
        # 添加实时状态
        stats['real_time'] = {
            'active_vehicles': len([v for v in self.vehicle_states.values() 
                                  if v.status != VehicleStatus.IDLE]),
            'idle_vehicles': len([v for v in self.vehicle_states.values() 
                                if v.status == VehicleStatus.IDLE]),
            'active_tasks': len([t for t in self.tasks.values() 
                               if t.status == TaskStatus.IN_PROGRESS]),
            'current_time': time.time()
        }
        
        return stats
    
    def get_stats(self) -> Dict:
        """兼容性方法"""
        return self.get_comprehensive_stats()
    
    def set_backbone_network(self, backbone_network):
        """设置骨干路径网络"""
        self.backbone_network = backbone_network
        print("已设置骨干路径网络到车辆调度器")

class SimplifiedECBSVehicleScheduler(SimplifiedVehicleScheduler):
    """ECBS增强版车辆调度器"""
    
    def __init__(self, env, path_planner=None, traffic_manager=None, backbone_network=None):
        super().__init__(env, path_planner, backbone_network, traffic_manager)
        
        # ECBS特有设置
        self.conflict_detection_interval = 15.0
        self.last_conflict_check = 0
        self.ecbs_enabled = True
        
        # 冲突统计
        self.conflict_stats = {
            'total_conflicts_detected': 0,
            'conflicts_resolved': 0,
            'ecbs_calls': 0
        }
        
        print("初始化ECBS增强版车辆调度器")
    
    def update(self, time_delta: float):
        """更新调度器状态 - 增加ECBS冲突检测"""
        # 调用父类更新
        super().update(time_delta)
        
        # ECBS冲突检测和解决
        current_time = time.time()
        if current_time - self.last_conflict_check > self.conflict_detection_interval:
            self._ecbs_conflict_resolution()
            self.last_conflict_check = current_time
    
    def _ecbs_conflict_resolution(self):
        """ECBS冲突检测和解决"""
        if not self.traffic_manager or not self.ecbs_enabled:
            return
        
        # 检测冲突
        conflicts = self.traffic_manager.detect_all_conflicts()
        
        if conflicts:
            self.conflict_stats['total_conflicts_detected'] += len(conflicts)
            self.conflict_stats['ecbs_calls'] += 1
            
            print(f"ECBS检测到 {len(conflicts)} 个冲突，开始解决...")
            
            # 解决冲突
            resolved_paths = self.traffic_manager.resolve_conflicts_enhanced(conflicts)
            
            if resolved_paths:
                self.conflict_stats['conflicts_resolved'] += len(resolved_paths)
                print(f"ECBS成功解决了 {len(resolved_paths)} 个车辆的路径冲突")

# 向后兼容性
VehicleScheduler = SimplifiedVehicleScheduler
OptimizedVehicleScheduler = SimplifiedVehicleScheduler