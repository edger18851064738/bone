"""
vehicle_scheduler.py - 优化版车辆调度器
深度集成接口系统、智能调度、预测性优化
"""

import math
import heapq
import time
import threading
from typing import List, Dict, Tuple, Set, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
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
class EnhancedVehicleTask:
    """增强的车辆任务类"""
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
    
    # 接口相关
    reserved_interfaces: List[str] = field(default_factory=list)
    interface_access_times: Dict[str, float] = field(default_factory=dict)
    
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
    
    def reserve_interface(self, interface_id: str, access_time: float):
        """预约接口"""
        if interface_id not in self.reserved_interfaces:
            self.reserved_interfaces.append(interface_id)
        self.interface_access_times[interface_id] = access_time
    
    def get_performance_metrics(self) -> Dict:
        """获取任务性能指标"""
        return {
            'planning_time': self.planning_time,
            'execution_time': self.actual_duration,
            'quality_score': self.quality_score,
            'deadline_met': self.deadline is None or self.completion_time <= self.deadline,
            'interface_count': len(self.reserved_interfaces),
            'path_type': self.path_structure.get('type', 'unknown')
        }

@dataclass
class VehicleState:
    """增强的车辆状态"""
    vehicle_id: str
    status: VehicleStatus = VehicleStatus.IDLE
    position: Tuple[float, float, float] = (0, 0, 0)
    
    # 基本属性
    max_load: float = 100
    current_load: float = 0
    speed: float = 1.0
    
    # 任务相关
    current_task: Optional[str] = None
    task_queue: List[str] = field(default_factory=list)
    completed_tasks: int = 0
    
    # 性能统计
    total_distance: float = 0
    total_time: float = 0
    idle_time: float = 0
    utilization_rate: float = 0
    
    # 偏好和历史
    preferred_loading_points: List[int] = field(default_factory=list)
    preferred_unloading_points: List[int] = field(default_factory=list)
    performance_history: List[Dict] = field(default_factory=list)
    
    # 接口使用统计
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
    
    def add_performance_record(self, task_metrics: Dict):
        """添加性能记录"""
        record = {
            'timestamp': time.time(),
            'task_metrics': task_metrics,
            'position': self.position,
            'utilization': self.utilization_rate
        }
        
        self.performance_history.append(record)
        
        # 限制历史记录长度
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

class IntelligentTaskAssigner:
    """智能任务分配器"""
    
    def __init__(self, env, backbone_network):
        self.env = env
        self.backbone_network = backbone_network
        
        # 分配策略权重
        self.assignment_weights = {
            'distance': 0.25,
            'vehicle_utilization': 0.20,
            'interface_efficiency': 0.20,
            'load_compatibility': 0.15,
            'historical_performance': 0.10,
            'deadline_urgency': 0.10
        }
        
        # 智能学习参数
        self.learning_rate = 0.1
        self.performance_memory = defaultdict(list)
    
    def assign_task_to_vehicle(self, task: EnhancedVehicleTask, 
                              available_vehicles: Dict[str, VehicleState]) -> Optional[str]:
        """智能任务分配"""
        if not available_vehicles:
            return None
        
        best_vehicle = None
        best_score = -float('inf')
        
        for vehicle_id, vehicle_state in available_vehicles.items():
            score = self._calculate_assignment_score(task, vehicle_state)
            
            if score > best_score:
                best_score = score
                best_vehicle = vehicle_id
        
        # 记录分配决策
        if best_vehicle:
            self._record_assignment_decision(task, best_vehicle, best_score)
        
        return best_vehicle
    
    def _calculate_assignment_score(self, task: EnhancedVehicleTask, 
                                   vehicle: VehicleState) -> float:
        """计算分配评分"""
        scores = {}
        
        # 1. 距离评分
        distance = self._calculate_distance(vehicle.position, task.start)
        scores['distance'] = 1000 / (distance + 10)
        
        # 2. 车辆利用率评分（低利用率优先）
        scores['vehicle_utilization'] = 1.0 - vehicle.utilization_rate
        
        # 3. 接口效率评分
        scores['interface_efficiency'] = vehicle.interface_efficiency
        
        # 4. 负载兼容性评分
        scores['load_compatibility'] = self._calculate_load_compatibility(task, vehicle)
        
        # 5. 历史性能评分
        scores['historical_performance'] = self._get_historical_performance_score(
            vehicle.vehicle_id, task.task_type
        )
        
        # 6. 截止时间紧迫性评分
        scores['deadline_urgency'] = self._calculate_deadline_urgency(task, vehicle)
        
        # 加权总分
        total_score = sum(
            scores[metric] * self.assignment_weights[metric]
            for metric in scores
        )
        
        # 空闲车辆加分
        if vehicle.status == VehicleStatus.IDLE:
            total_score *= 1.3
        
        # 任务队列长度惩罚
        queue_penalty = len(vehicle.task_queue) * 0.1
        total_score *= (1.0 - queue_penalty)
        
        return total_score
    
    def _calculate_load_compatibility(self, task: EnhancedVehicleTask, 
                                    vehicle: VehicleState) -> float:
        """计算负载兼容性"""
        load_ratio = vehicle.current_load / vehicle.max_load
        
        if task.task_type == 'to_loading':
            # 去装载：空车更合适
            return 1.0 - load_ratio
        elif task.task_type == 'to_unloading':
            # 去卸载：满载更合适
            return load_ratio
        else:
            # 其他任务：中等负载
            return 1.0 - abs(load_ratio - 0.5) * 2
    
    def _get_historical_performance_score(self, vehicle_id: str, task_type: str) -> float:
        """获取历史性能评分"""
        key = f"{vehicle_id}_{task_type}"
        history = self.performance_memory.get(key, [])
        
        if not history:
            return 0.5  # 默认中等评分
        
        # 计算最近的平均性能
        recent_performances = history[-10:]  # 最近10次
        avg_performance = sum(recent_performances) / len(recent_performances)
        
        return min(1.0, max(0.0, avg_performance))
    
    def _calculate_deadline_urgency(self, task: EnhancedVehicleTask, 
                                   vehicle: VehicleState) -> float:
        """计算截止时间紧迫性"""
        if not task.deadline:
            return 0.5  # 无截止时间
        
        current_time = time.time()
        time_remaining = task.deadline - current_time
        
        if time_remaining <= 0:
            return 1.0  # 已过期，最高紧迫性
        
        # 估算完成时间
        estimated_completion_time = self._estimate_task_completion_time(task, vehicle)
        
        if estimated_completion_time > time_remaining:
            return 0.8  # 可能无法按时完成
        else:
            urgency = 1.0 - (time_remaining - estimated_completion_time) / time_remaining
            return max(0.0, min(1.0, urgency))
    
    def _estimate_task_completion_time(self, task: EnhancedVehicleTask, 
                                      vehicle: VehicleState) -> float:
        """估算任务完成时间"""
        # 简化估算：距离 / 速度 + 操作时间
        distance = self._calculate_distance(vehicle.position, task.start)
        travel_time = distance / vehicle.speed
        
        # 添加任务类型相关的操作时间
        operation_time = {
            'to_loading': 60,  # 装载时间
            'to_unloading': 40,  # 卸载时间
            'to_initial': 0    # 返回起点
        }.get(task.task_type, 30)
        
        return travel_time + operation_time
    
    def _record_assignment_decision(self, task: EnhancedVehicleTask, 
                                   vehicle_id: str, score: float):
        """记录分配决策用于学习"""
        decision_record = {
            'timestamp': time.time(),
            'task_type': task.task_type,
            'vehicle_id': vehicle_id,
            'score': score,
            'task_id': task.task_id
        }
        
        # 这里可以添加机器学习逻辑来优化分配策略
    
    def update_performance_feedback(self, vehicle_id: str, task_type: str, 
                                   performance_score: float):
        """更新性能反馈"""
        key = f"{vehicle_id}_{task_type}"
        self.performance_memory[key].append(performance_score)
        
        # 限制历史长度
        if len(self.performance_memory[key]) > 50:
            self.performance_memory[key] = self.performance_memory[key][-50:]
    
    def _calculate_distance(self, pos1: Tuple, pos2: Tuple) -> float:
        """计算距离"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

class PredictiveScheduler:
    """预测性调度器"""
    
    def __init__(self, env, backbone_network):
        self.env = env
        self.backbone_network = backbone_network
        
        # 预测参数
        self.prediction_horizon = 300.0  # 5分钟预测窗口
        self.update_interval = 30.0      # 30秒更新一次
        
        # 预测模型
        self.demand_predictor = self._init_demand_predictor()
        self.congestion_predictor = self._init_congestion_predictor()
        
        # 预测结果缓存
        self.last_prediction_time = 0
        self.cached_predictions = {}
    
    def _init_demand_predictor(self):
        """初始化需求预测器"""
        return {
            'loading_demand': defaultdict(float),
            'unloading_demand': defaultdict(float),
            'historical_patterns': []
        }
    
    def _init_congestion_predictor(self):
        """初始化拥堵预测器"""
        return {
            'interface_congestion': defaultdict(float),
            'backbone_congestion': defaultdict(float),
            'congestion_history': []
        }
    
    def predict_optimal_assignments(self, pending_tasks: List[EnhancedVehicleTask],
                                   vehicle_states: Dict[str, VehicleState]) -> Dict[str, List[str]]:
        """预测最优任务分配"""
        current_time = time.time()
        
        # 更新预测
        if current_time - self.last_prediction_time > self.update_interval:
            self._update_predictions(current_time)
            self.last_prediction_time = current_time
        
        # 基于预测进行分配
        assignments = {}
        
        # 按优先级和预测价值排序任务
        sorted_tasks = self._sort_tasks_by_predicted_value(pending_tasks, current_time)
        
        available_vehicles = {vid: state for vid, state in vehicle_states.items() 
                            if state.status == VehicleStatus.IDLE or len(state.task_queue) < 3}
        
        for task in sorted_tasks:
            best_vehicle = self._predict_best_vehicle_for_task(task, available_vehicles, current_time)
            
            if best_vehicle:
                if best_vehicle not in assignments:
                    assignments[best_vehicle] = []
                assignments[best_vehicle].append(task.task_id)
                
                # 更新可用车辆状态
                available_vehicles[best_vehicle].task_queue.append(task.task_id)
        
        return assignments
    
    def _update_predictions(self, current_time: float):
        """更新预测模型"""
        # 更新需求预测
        self._update_demand_predictions(current_time)
        
        # 更新拥堵预测
        self._update_congestion_predictions(current_time)
    
    def _update_demand_predictions(self, current_time: float):
        """更新需求预测"""
        # 基于历史模式预测未来需求
        time_of_day = (current_time % 86400) / 3600  # 一天中的小时
        
        # 简化的周期性需求模型
        for i, loading_point in enumerate(self.env.loading_points):
            # 假设需求有周期性变化
            base_demand = 1.0
            time_factor = 1.0 + 0.3 * math.sin(2 * math.pi * time_of_day / 24)
            
            predicted_demand = base_demand * time_factor
            self.demand_predictor['loading_demand'][i] = predicted_demand
        
        for i, unloading_point in enumerate(self.env.unloading_points):
            base_demand = 1.0
            time_factor = 1.0 + 0.3 * math.cos(2 * math.pi * time_of_day / 24)
            
            predicted_demand = base_demand * time_factor
            self.demand_predictor['unloading_demand'][i] = predicted_demand
    
    def _update_congestion_predictions(self, current_time: float):
        """更新拥堵预测"""
        if not self.backbone_network:
            return
        
        # 预测接口拥堵
        for interface_id, interface in self.backbone_network.backbone_interfaces.items():
            # 基于使用频率预测拥堵
            usage_rate = interface.usage_count / max(1, current_time / 3600)  # 每小时使用次数
            congestion_score = min(1.0, usage_rate / 10.0)  # 假设10次/小时为拥堵
            
            self.congestion_predictor['interface_congestion'][interface_id] = congestion_score
    
    def _sort_tasks_by_predicted_value(self, tasks: List[EnhancedVehicleTask], 
                                      current_time: float) -> List[EnhancedVehicleTask]:
        """按预测价值排序任务"""
        def task_value(task):
            # 基础优先级
            base_value = task.priority
            
            # 截止时间紧迫性
            urgency_value = 0
            if task.deadline:
                time_remaining = task.deadline - current_time
                urgency_value = max(0, 10 - time_remaining / 60)  # 时间越短价值越高
            
            # 需求预测价值
            demand_value = 0
            if task.loading_point_id is not None:
                demand_value += self.demand_predictor['loading_demand'].get(task.loading_point_id, 1.0)
            if task.unloading_point_id is not None:
                demand_value += self.demand_predictor['unloading_demand'].get(task.unloading_point_id, 1.0)
            
            return base_value + urgency_value + demand_value
        
        return sorted(tasks, key=task_value, reverse=True)
    
    def _predict_best_vehicle_for_task(self, task: EnhancedVehicleTask, 
                                      available_vehicles: Dict[str, VehicleState], 
                                      current_time: float) -> Optional[str]:
        """预测任务的最佳车辆"""
        best_vehicle = None
        best_predicted_performance = -float('inf')
        
        for vehicle_id, vehicle_state in available_vehicles.items():
            # 预测性能
            predicted_performance = self._predict_task_performance(
                task, vehicle_state, current_time
            )
            
            if predicted_performance > best_predicted_performance:
                best_predicted_performance = predicted_performance
                best_vehicle = vehicle_id
        
        return best_vehicle
    
    def _predict_task_performance(self, task: EnhancedVehicleTask, 
                                 vehicle: VehicleState, current_time: float) -> float:
        """预测任务执行性能"""
        # 基础距离评分
        distance = math.sqrt(
            (vehicle.position[0] - task.start[0])**2 + 
            (vehicle.position[1] - task.start[1])**2
        )
        distance_score = 100 / (distance + 10)
        
        # 车辆历史性能
        historical_score = vehicle.interface_efficiency * 100
        
        # 预测拥堵影响
        congestion_penalty = 0
        if task.goal and self.backbone_network:
            target_type, target_id = self.backbone_network.identify_target_point(task.goal)
            if target_type:
                # 查找相关接口的拥堵情况
                relevant_interfaces = [
                    interface_id for interface_id, interface in self.backbone_network.backbone_interfaces.items()
                    if interface.backbone_path_id and target_type in interface.backbone_path_id
                ]
                
                if relevant_interfaces:
                    avg_congestion = sum(
                        self.congestion_predictor['interface_congestion'].get(iid, 0)
                        for iid in relevant_interfaces
                    ) / len(relevant_interfaces)
                    
                    congestion_penalty = avg_congestion * 20
        
        return distance_score + historical_score - congestion_penalty

class OptimizedVehicleScheduler:
    """优化的车辆调度器 - 全面集成新系统"""
    
    def __init__(self, env, path_planner=None, traffic_manager=None, backbone_network=None):
        self.env = env
        self.path_planner = path_planner
        self.traffic_manager = traffic_manager
        self.backbone_network = backbone_network
        
        # 核心组件
        self.task_assigner = IntelligentTaskAssigner(env, backbone_network)
        self.predictive_scheduler = PredictiveScheduler(env, backbone_network)
        
        # 数据存储
        self.tasks = {}  # {task_id: EnhancedVehicleTask}
        self.vehicle_states = {}  # {vehicle_id: VehicleState}
        self.mission_templates = {}  # {template_id: mission_config}
        
        # 任务管理
        self.task_counter = 0
        self.active_assignments = {}  # {vehicle_id: [task_ids]}
        self.pending_tasks = deque()
        
        # 性能优化
        self.batch_planning = True
        self.max_batch_size = 8
        self.parallel_processing = True
        self.max_workers = 4
        
        # 调度策略
        self.scheduling_strategy = 'predictive'  # 'greedy', 'optimal', 'predictive'
        self.replan_interval = 60.0  # 重新规划间隔
        self.last_global_replan = 0
        
        # 统计信息
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'average_completion_time': 0,
            'total_distance': 0,
            'backbone_utilization_rate': 0,
            'interface_efficiency': 0,
            'vehicle_utilization': {},
            'performance_trends': defaultdict(list),
            'optimization_calls': 0,
            'batch_planning_savings': 0
        }
        
        # 线程安全
        self.lock = threading.Lock()
        
        print("初始化优化版车辆调度器")
    
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
                                       unloading_point_id: int = None,
                                       mission_config: Dict = None) -> bool:
        """创建增强的任务模板"""
        if not self.env.loading_points or not self.env.unloading_points:
            return False
        
        # 智能选择装载点和卸载点
        if loading_point_id is None:
            loading_point_id = self._select_optimal_loading_point()
        
        if unloading_point_id is None:
            unloading_point_id = self._select_optimal_unloading_point()
        
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
            'config': mission_config or {},
            'tasks': [
                {
                    'task_type': 'to_loading',
                    'goal': loading_point,
                    'priority': 2,
                    'estimated_duration': 180  # 3分钟
                },
                {
                    'task_type': 'to_unloading',
                    'goal': unloading_point,
                    'priority': 3,
                    'estimated_duration': 150  # 2.5分钟
                },
                {
                    'task_type': 'to_initial',
                    'goal': None,  # 将在分配时确定
                    'priority': 1,
                    'estimated_duration': 120  # 2分钟
                }
            ]
        }
        
        self.mission_templates[template_id] = template
        return True
    
    def assign_mission_intelligently(self, vehicle_id: str, template_id: str = None) -> bool:
        """智能分配任务"""
        if vehicle_id not in self.vehicle_states:
            return False
        
        # 自动选择最佳模板
        if template_id is None:
            template_id = self._select_best_template_for_vehicle(vehicle_id)
        
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
                goal = self.env.vehicles[vehicle_id].get('initial_position', current_position)
            else:
                goal = task_template['goal']
            
            # 创建增强任务
            task = EnhancedVehicleTask(
                task_id=task_id,
                task_type=task_template['task_type'],
                start=current_position,
                goal=goal,
                priority=task_template['priority'],
                loading_point_id=template.get('loading_point_id'),
                unloading_point_id=template.get('unloading_point_id'),
                estimated_duration=task_template['estimated_duration']
            )
            
            # 设置截止时间
            if 'deadline_offset' in task_template:
                task.deadline = time.time() + task_template['deadline_offset']
            
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
        
        print(f"为车辆 {vehicle_id} 智能分配了 {len(created_tasks)} 个任务")
        return True
    
    def _select_optimal_loading_point(self) -> int:
        """选择最优装载点"""
        if not self.env.loading_points:
            return 0
        
        # 基于当前使用情况和预测需求选择
        best_point = 0
        best_score = -1
        
        for i, point in enumerate(self.env.loading_points):
            # 计算当前使用率
            current_usage = sum(
                1 for task in self.tasks.values()
                if task.loading_point_id == i and task.status in [TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS]
            )
            
            # 获取预测需求
            predicted_demand = self.predictive_scheduler.demand_predictor['loading_demand'].get(i, 1.0)
            
            # 计算评分（需求高但使用率低的点评分高）
            usage_factor = 1.0 / (current_usage + 1)
            demand_factor = predicted_demand
            
            score = usage_factor * demand_factor
            
            if score > best_score:
                best_score = score
                best_point = i
        
        return best_point
    
    def _select_optimal_unloading_point(self) -> int:
        """选择最优卸载点"""
        if not self.env.unloading_points:
            return 0
        
        # 类似装载点的选择逻辑
        best_point = 0
        best_score = -1
        
        for i, point in enumerate(self.env.unloading_points):
            current_usage = sum(
                1 for task in self.tasks.values()
                if task.unloading_point_id == i and task.status in [TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS]
            )
            
            predicted_demand = self.predictive_scheduler.demand_predictor['unloading_demand'].get(i, 1.0)
            
            usage_factor = 1.0 / (current_usage + 1)
            demand_factor = predicted_demand
            
            score = usage_factor * demand_factor
            
            if score > best_score:
                best_score = score
                best_point = i
        
        return best_point
    
    def _select_best_template_for_vehicle(self, vehicle_id: str) -> str:
        """为车辆选择最佳模板"""
        if not self.mission_templates:
            # 创建默认模板
            self.create_enhanced_mission_template("default")
        
        if len(self.mission_templates) == 1:
            return list(self.mission_templates.keys())[0]
        
        # 基于车辆历史性能和当前状态选择模板
        vehicle_state = self.vehicle_states[vehicle_id]
        best_template = None
        best_score = -1
        
        for template_id, template in self.mission_templates.items():
            score = self._evaluate_template_for_vehicle(template, vehicle_state)
            
            if score > best_score:
                best_score = score
                best_template = template_id
        
        return best_template or list(self.mission_templates.keys())[0]
    
    def _evaluate_template_for_vehicle(self, template: Dict, vehicle: VehicleState) -> float:
        """评估模板对车辆的适合度"""
        score = 1.0
        
        # 基于车辆偏好
        loading_id = template['loading_point_id']
        unloading_id = template['unloading_point_id']
        
        if loading_id in vehicle.preferred_loading_points:
            score += 0.3
        
        if unloading_id in vehicle.preferred_unloading_points:
            score += 0.3
        
        # 基于距离
        loading_pos = template['loading_position']
        distance = math.sqrt(
            (vehicle.position[0] - loading_pos[0])**2 + 
            (vehicle.position[1] - loading_pos[1])**2
        )
        
        distance_score = 100 / (distance + 10)
        score += distance_score * 0.001  # 归一化距离影响
        
        return score
    
    def _start_next_task(self, vehicle_id: str) -> bool:
        """开始车辆的下一个任务"""
        if vehicle_id not in self.active_assignments:
            return False
        
        assignments = self.active_assignments[vehicle_id]
        if not assignments:
            return False
        
        task_id = assignments[0]
        if task_id not in self.tasks:
            # 移除无效任务
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
            
            # 移除失败的任务并尝试下一个
            assignments.remove(task_id)
            return self._start_next_task(vehicle_id)
    
    def _plan_task_path(self, task: EnhancedVehicleTask, vehicle: VehicleState) -> bool:
        """为任务规划路径"""
        if not self.path_planner:
            return False
        
        planning_start = time.time()
        
        try:
            # 使用增强的路径规划接口
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
                    
                    # 处理接口预约
                    if structure.get('interface_id'):
                        interface_id = structure['interface_id']
                        access_time = time.time() + structure.get('access_length', 0) * 2
                        task.reserve_interface(interface_id, access_time)
                        
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
    def update_backbone_visualization(self):
        """更新骨干路径可视化"""
        if self.backbone_visualizer:
            self.removeItem(self.backbone_visualizer)
        
        if self.backbone_network and self.show_options['backbone']:
            self.backbone_visualizer = BackbonePathVisualization(self.backbone_network)
            self.addItem(self.backbone_visualizer)
    def set_backbone_network(self, backbone_network):
        """设置骨干路径网络"""
        self.backbone_network = backbone_network
        self.update_backbone_visualization()
        self.update_interface_display()    
    def update(self, time_delta: float):
        """更新调度器状态"""
        current_time = time.time()
        
        # 更新所有车辆状态
        for vehicle_id, vehicle_state in self.vehicle_states.items():
            self._update_vehicle_state(vehicle_id, vehicle_state, time_delta)
        
        # 处理待分配任务
        self._process_pending_tasks()
        
        # 批量路径规划优化
        if self.batch_planning:
            self._optimize_batch_planning()
        
        # 全局重新规划
        if current_time - self.last_global_replan > self.replan_interval:
            self._global_replan()
            self.last_global_replan = current_time
        
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
            vehicle_state.status = status_mapping.get(env_status, VehicleStatus.IDLE)
        
        # 处理当前任务进度
        if vehicle_state.current_task:
            self._update_task_progress(vehicle_state.current_task, vehicle_state, time_delta)
    
    def _update_task_progress(self, task_id: str, vehicle_state: VehicleState, 
                             time_delta: float):
        """更新任务进度"""
        if task_id not in self.tasks:
            return
        
        task = self.tasks[task_id]
        
        # 根据车辆状态更新任务进度
        if vehicle_state.status == VehicleStatus.MOVING and task.path:
            # 基于路径进度更新
            if vehicle_state.vehicle_id in self.env.vehicles:
                env_vehicle = self.env.vehicles[vehicle_state.vehicle_id]
                path_index = env_vehicle.get('path_index', 0)
                path_length = len(task.path)
                
                if path_length > 0:
                    progress = path_index / path_length
                    task.update_progress(progress)
                    
                    # 检查是否到达目标
                    if progress >= 1.0:
                        self._handle_task_arrival(task, vehicle_state)
        
        elif vehicle_state.status in [VehicleStatus.LOADING, VehicleStatus.UNLOADING]:
            # 装载/卸载过程中的进度更新
            operation_time = 60 if vehicle_state.status == VehicleStatus.LOADING else 40
            elapsed_time = time.time() - task.start_time
            
            progress = min(1.0, elapsed_time / operation_time)
            task.update_progress(progress)
            
            if progress >= 1.0:
                self._complete_current_task(vehicle_state.vehicle_id)
    
    def _handle_task_arrival(self, task: EnhancedVehicleTask, vehicle_state: VehicleState):
        """处理任务到达目标"""
        if task.task_type == 'to_loading':
            # 到达装载点，开始装载
            vehicle_state.status = VehicleStatus.LOADING
            if vehicle_state.vehicle_id in self.env.vehicles:
                self.env.vehicles[vehicle_state.vehicle_id]['status'] = 'loading'
                self.env.vehicles[vehicle_state.vehicle_id]['loading_progress'] = 0
            
            # 更新偏好
            if (task.loading_point_id is not None and 
                task.loading_point_id not in vehicle_state.preferred_loading_points):
                vehicle_state.preferred_loading_points.append(task.loading_point_id)
        
        elif task.task_type == 'to_unloading':
            # 到达卸载点，开始卸载
            vehicle_state.status = VehicleStatus.UNLOADING
            if vehicle_state.vehicle_id in self.env.vehicles:
                self.env.vehicles[vehicle_state.vehicle_id]['status'] = 'unloading'
                self.env.vehicles[vehicle_state.vehicle_id]['unloading_progress'] = 0
            
            # 更新偏好
            if (task.unloading_point_id is not None and 
                task.unloading_point_id not in vehicle_state.preferred_unloading_points):
                vehicle_state.preferred_unloading_points.append(task.unloading_point_id)
        
        elif task.task_type == 'to_initial':
            # 返回起点，完成一个循环
            self._complete_current_task(vehicle_state.vehicle_id)
            
            # 更新完成循环数
            if vehicle_state.vehicle_id in self.env.vehicles:
                env_vehicle = self.env.vehicles[vehicle_state.vehicle_id]
                env_vehicle['completed_cycles'] = env_vehicle.get('completed_cycles', 0) + 1
            
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
        
        # 释放接口预约
        for interface_id in task.reserved_interfaces:
            if self.traffic_manager:
                self.traffic_manager.interface_manager.release_interface(interface_id, vehicle_id)
        
        # 记录性能
        metrics = task.get_performance_metrics()
        vehicle_state.add_performance_record(metrics)
        
        # 更新车辆效率
        if task.path_structure.get('type') in ['interface_assisted', 'backbone_only']:
            # 更新接口使用效率
            efficiency_update = task.quality_score * 0.1
            vehicle_state.interface_efficiency = (
                vehicle_state.interface_efficiency * 0.9 + efficiency_update
            )
        
        # 从分配列表中移除
        if vehicle_id in self.active_assignments:
            if task_id in self.active_assignments[vehicle_id]:
                self.active_assignments[vehicle_id].remove(task_id)
        
        # 更新统计
        self.stats['completed_tasks'] += 1
        
        if task.path:
            path_distance = sum(
                math.sqrt((task.path[i+1][0] - task.path[i][0])**2 + 
                         (task.path[i+1][1] - task.path[i][1])**2)
                for i in range(len(task.path) - 1)
            )
            self.stats['total_distance'] += path_distance
            vehicle_state.total_distance += path_distance
        
        # 释放交通管理器中的路径
        if self.traffic_manager:
            self.traffic_manager.release_vehicle_path(vehicle_id)
        
        # 更新任务分配器的性能反馈
        performance_score = min(1.0, task.quality_score + (1.0 if metrics['deadline_met'] else -0.5))
        self.task_assigner.update_performance_feedback(
            vehicle_id, task.task_type, performance_score
        )
        
        # 清理车辆状态
        vehicle_state.current_task = None
        vehicle_state.status = VehicleStatus.IDLE
        vehicle_state.completed_tasks += 1
        
        if vehicle_id in self.env.vehicles:
            self.env.vehicles[vehicle_id]['status'] = 'idle'
        
        print(f"任务 {task_id} 完成，车辆 {vehicle_id}，"
              f"用时 {task.actual_duration:.1f}s，质量 {task.quality_score:.2f}")
        
        # 开始下一个任务
        self._start_next_task(vehicle_id)
    
    def _schedule_next_mission_cycle(self, vehicle_id: str):
        """安排下一轮任务循环"""
        # 如果车辆没有更多任务，自动分配新的任务循环
        if (vehicle_id in self.active_assignments and 
            not self.active_assignments[vehicle_id]):
            
            # 智能选择最佳模板
            self.assign_mission_intelligently(vehicle_id)
    
    def _process_pending_tasks(self):
        """处理待分配任务"""
        if not self.pending_tasks:
            return
        
        # 使用预测性调度
        if self.scheduling_strategy == 'predictive':
            assignments = self.predictive_scheduler.predict_optimal_assignments(
                list(self.pending_tasks), self.vehicle_states
            )
            
            for vehicle_id, task_ids in assignments.items():
                for task_id in task_ids:
                    if vehicle_id in self.active_assignments:
                        self.active_assignments[vehicle_id].append(task_id)
                        
                        # 从待处理队列中移除
                        task_to_remove = None
                        for task in self.pending_tasks:
                            if task.task_id == task_id:
                                task_to_remove = task
                                break
                        
                        if task_to_remove:
                            self.pending_tasks.remove(task_to_remove)
        
        # 启动空闲车辆的任务
        for vehicle_id, vehicle_state in self.vehicle_states.items():
            if (vehicle_state.status == VehicleStatus.IDLE and 
                vehicle_id in self.active_assignments and
                self.active_assignments[vehicle_id] and
                not vehicle_state.current_task):
                
                self._start_next_task(vehicle_id)
    
    def _optimize_batch_planning(self):
        """批量规划优化"""
        if not self.batch_planning or len(self.vehicle_states) < 2:
            return
        
        # 收集需要重新规划的车辆
        vehicles_to_replan = []
        
        for vehicle_id, vehicle_state in self.vehicle_states.items():
            if (vehicle_state.status == VehicleStatus.MOVING and 
                vehicle_state.current_task in self.tasks):
                
                task = self.tasks[vehicle_state.current_task]
                
                # 检查是否需要重新规划
                if self._should_replan_vehicle(vehicle_id, task):
                    vehicles_to_replan.append(vehicle_id)
        
        # 批量重新规划
        if len(vehicles_to_replan) >= 2:
            self._batch_replan_vehicles(vehicles_to_replan)
            self.stats['batch_planning_savings'] += len(vehicles_to_replan) - 1
    
    def _should_replan_vehicle(self, vehicle_id: str, task: EnhancedVehicleTask) -> bool:
        """判断是否需要重新规划车辆路径"""
        # 检查路径质量
        if task.quality_score < 0.6:
            return True
        
        # 检查是否有新的冲突
        if self.traffic_manager:
            conflicts = self.traffic_manager.detect_all_conflicts()
            
            for conflict in conflicts:
                if conflict.agent1 == vehicle_id or conflict.agent2 == vehicle_id:
                    return True
        
        return False
    
    def _batch_replan_vehicles(self, vehicle_ids: List[str]):
        """批量重新规划车辆路径"""
        if not self.traffic_manager:
            return
        
        print(f"批量重新规划 {len(vehicle_ids)} 个车辆的路径")
        
        # 收集当前路径
        current_paths = {}
        for vehicle_id in vehicle_ids:
            if vehicle_id in self.env.vehicles and 'path' in self.env.vehicles[vehicle_id]:
                current_paths[vehicle_id] = self.env.vehicles[vehicle_id]['path']
        
        if len(current_paths) < 2:
            return
        
        # 检测冲突
        conflicts = self.traffic_manager.detect_all_conflicts()
        
        if conflicts:
            # 解决冲突
            resolved_paths = self.traffic_manager.resolve_conflicts_enhanced(conflicts)
            
            # 应用新路径
            for vehicle_id, new_path in resolved_paths.items():
                if vehicle_id in vehicle_ids and new_path != current_paths.get(vehicle_id):
                    self._apply_new_path(vehicle_id, new_path)
    
    def _apply_new_path(self, vehicle_id: str, new_path: List):
        """应用新路径"""
        if vehicle_id not in self.vehicle_states:
            return
        
        vehicle_state = self.vehicle_states[vehicle_id]
        
        if vehicle_state.current_task in self.tasks:
            task = self.tasks[vehicle_state.current_task]
            task.path = new_path
            task.path_structure = {'type': 'conflict_resolved', 'optimized': True}
            
            # 更新环境中的车辆路径
            if vehicle_id in self.env.vehicles:
                self.env.vehicles[vehicle_id]['path'] = new_path
                self.env.vehicles[vehicle_id]['path_index'] = 0
                self.env.vehicles[vehicle_id]['progress'] = 0.0
                self.env.vehicles[vehicle_id]['path_structure'] = task.path_structure
    
    def _global_replan(self):
        """全局重新规划"""
        self.stats['optimization_calls'] += 1
        
        # 预测性重新调度
        if self.scheduling_strategy == 'predictive':
            # 更新预测模型
            current_time = time.time()
            self.predictive_scheduler._update_predictions(current_time)
            
            # 重新评估任务分配
            pending_and_future_tasks = []
            
            # 收集所有未完成的任务
            for task in self.tasks.values():
                if task.status in [TaskStatus.PENDING, TaskStatus.ASSIGNED]:
                    pending_and_future_tasks.append(task)
            
            if pending_and_future_tasks:
                # 预测性重新分配
                new_assignments = self.predictive_scheduler.predict_optimal_assignments(
                    pending_and_future_tasks, self.vehicle_states
                )
                
                # 应用新的分配
                self._apply_new_assignments(new_assignments)
    
    def _apply_new_assignments(self, new_assignments: Dict[str, List[str]]):
        """应用新的任务分配"""
        for vehicle_id, task_ids in new_assignments.items():
            if vehicle_id in self.active_assignments:
                # 只添加新的任务，不移除已在执行的任务
                current_assignments = set(self.active_assignments[vehicle_id])
                new_task_ids = [tid for tid in task_ids if tid not in current_assignments]
                
                self.active_assignments[vehicle_id].extend(new_task_ids)
    
    def _update_performance_stats(self, time_delta: float):
        """更新性能统计"""
        # 更新车辆利用率统计
        for vehicle_id, vehicle_state in self.vehicle_states.items():
            self.stats['vehicle_utilization'][vehicle_id] = vehicle_state.utilization_rate
        
        # 计算平均完成时间
        completed_tasks = [task for task in self.tasks.values() 
                          if task.status == TaskStatus.COMPLETED]
        
        if completed_tasks:
            total_duration = sum(task.actual_duration for task in completed_tasks)
            self.stats['average_completion_time'] = total_duration / len(completed_tasks)
        
        # 计算骨干网络利用率
        total_backbone_usage = sum(vehicle.backbone_usage_count 
                                 for vehicle in self.vehicle_states.values())
        total_path_usage = sum(vehicle.backbone_usage_count + vehicle.direct_path_count 
                             for vehicle in self.vehicle_states.values())
        
        if total_path_usage > 0:
            self.stats['backbone_utilization_rate'] = total_backbone_usage / total_path_usage
        
        # 计算接口效率
        interface_efficiencies = [vehicle.interface_efficiency 
                                for vehicle in self.vehicle_states.values()]
        if interface_efficiencies:
            self.stats['interface_efficiency'] = sum(interface_efficiencies) / len(interface_efficiencies)
        
        # 记录性能趋势
        current_time = time.time()
        self.stats['performance_trends']['completion_rate'].append({
            'timestamp': current_time,
            'value': self.stats['completed_tasks'] / max(1, self.stats['total_tasks'])
        })
        
        self.stats['performance_trends']['utilization'].append({
            'timestamp': current_time,
            'value': sum(self.stats['vehicle_utilization'].values()) / max(1, len(self.stats['vehicle_utilization']))
        })
        
        # 限制趋势数据长度
        for trend_name, trend_data in self.stats['performance_trends'].items():
            if len(trend_data) > 100:
                self.stats['performance_trends'][trend_name] = trend_data[-100:]
    
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
            'pending_tasks': len([t for t in self.tasks.values() 
                                if t.status == TaskStatus.PENDING]),
            'current_time': time.time()
        }
        
        # 添加车辆详细统计
        stats['vehicle_details'] = {}
        for vehicle_id, vehicle_state in self.vehicle_states.items():
            stats['vehicle_details'][vehicle_id] = {
                'status': vehicle_state.status.value,
                'utilization_rate': vehicle_state.utilization_rate,
                'completed_tasks': vehicle_state.completed_tasks,
                'total_distance': vehicle_state.total_distance,
                'backbone_usage_rate': (
                    vehicle_state.backbone_usage_count / 
                    max(1, vehicle_state.backbone_usage_count + vehicle_state.direct_path_count)
                ),
                'interface_efficiency': vehicle_state.interface_efficiency
            }
        
        # 添加任务分类统计
        task_status_counts = defaultdict(int)
        task_type_counts = defaultdict(int)
        
        for task in self.tasks.values():
            task_status_counts[task.status.value] += 1
            task_type_counts[task.task_type] += 1
        
        stats['task_breakdown'] = {
            'by_status': dict(task_status_counts),
            'by_type': dict(task_type_counts)
        }
        
        # 性能指标
        if self.stats['total_tasks'] > 0:
            stats['efficiency_metrics'] = {
                'task_success_rate': self.stats['completed_tasks'] / self.stats['total_tasks'],
                'task_failure_rate': self.stats['failed_tasks'] / self.stats['total_tasks'],
                'average_task_quality': sum(
                    task.quality_score for task in self.tasks.values() 
                    if task.status == TaskStatus.COMPLETED
                ) / max(1, self.stats['completed_tasks'])
            }
        
        return stats
    
    def get_vehicle_info(self, vehicle_id: str) -> Optional[Dict]:
        """获取车辆详细信息"""
        if vehicle_id not in self.vehicle_states:
            return None
        
        vehicle_state = self.vehicle_states[vehicle_id]
        
        # 基本信息
        info = {
            'vehicle_id': vehicle_id,
            'status': vehicle_state.status.value,
            'position': vehicle_state.position,
            'current_load': vehicle_state.current_load,
            'max_load': vehicle_state.max_load,
            'utilization_rate': vehicle_state.utilization_rate,
            'completed_tasks': vehicle_state.completed_tasks,
            'total_distance': vehicle_state.total_distance
        }
        
        # 当前任务信息
        if vehicle_state.current_task and vehicle_state.current_task in self.tasks:
            current_task = self.tasks[vehicle_state.current_task]
            info['current_task'] = {
                'task_id': current_task.task_id,
                'task_type': current_task.task_type,
                'progress': current_task.execution_progress,
                'quality_score': current_task.quality_score,
                'path_structure': current_task.path_structure,
                'reserved_interfaces': current_task.reserved_interfaces
            }
        
        # 任务队列信息
        info['task_queue'] = []
        if vehicle_id in self.active_assignments:
            for task_id in self.active_assignments[vehicle_id]:
                if task_id in self.tasks:
                    task = self.tasks[task_id]
                    info['task_queue'].append({
                        'task_id': task_id,
                        'task_type': task.task_type,
                        'priority': task.priority,
                        'status': task.status.value
                    })
        
        # 性能历史
        info['performance_history'] = vehicle_state.performance_history[-10:]  # 最近10条记录
        
        # 偏好和效率
        info['preferences'] = {
            'loading_points': vehicle_state.preferred_loading_points,
            'unloading_points': vehicle_state.preferred_unloading_points
        }
        
        info['efficiency_metrics'] = {
            'backbone_usage_count': vehicle_state.backbone_usage_count,
            'direct_path_count': vehicle_state.direct_path_count,
            'backbone_usage_rate': (
                vehicle_state.backbone_usage_count / 
                max(1, vehicle_state.backbone_usage_count + vehicle_state.direct_path_count)
            ),
            'interface_efficiency': vehicle_state.interface_efficiency
        }
        
        return info

# ECBS增强版调度器
class ECBSEnhancedVehicleScheduler(OptimizedVehicleScheduler):
    """ECBS增强版车辆调度器"""
    
    def __init__(self, env, path_planner=None, traffic_manager=None, backbone_network=None):
        super().__init__(env, path_planner, traffic_manager, backbone_network)
        
        # ECBS特有设置
        self.conflict_detection_interval = 10.0  # 10秒检测一次冲突
        self.last_conflict_check = 0
        self.ecbs_enabled = True
        
        # 冲突统计
        self.conflict_stats = {
            'total_conflicts_detected': 0,
            'conflicts_resolved': 0,
            'ecbs_calls': 0,
            'resolution_success_rate': 0.0
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
        
        # 收集所有活动车辆的路径
        active_paths = {}
        for vehicle_id, vehicle_state in self.vehicle_states.items():
            if (vehicle_state.status == VehicleStatus.MOVING and 
                vehicle_state.current_task in self.tasks):
                
                task = self.tasks[vehicle_state.current_task]
                if task.path:
                    # 获取剩余路径
                    if vehicle_id in self.env.vehicles:
                        env_vehicle = self.env.vehicles[vehicle_id]
                        path_index = env_vehicle.get('path_index', 0)
                        remaining_path = task.path[path_index:]
                        
                        if len(remaining_path) > 1:
                            active_paths[vehicle_id] = remaining_path
        
        if len(active_paths) < 2:
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
                # 应用解决方案
                resolution_count = 0
                for vehicle_id, new_path in resolved_paths.items():
                    if vehicle_id in active_paths and new_path != active_paths[vehicle_id]:
                        self._apply_new_path(vehicle_id, new_path)
                        resolution_count += 1
                
                self.conflict_stats['conflicts_resolved'] += resolution_count
                
                if resolution_count > 0:
                    print(f"ECBS成功解决了 {resolution_count} 个车辆的路径冲突")
        
        # 更新成功率
        if self.conflict_stats['total_conflicts_detected'] > 0:
            self.conflict_stats['resolution_success_rate'] = (
                self.conflict_stats['conflicts_resolved'] / 
                self.conflict_stats['total_conflicts_detected']
            )
    
    def get_comprehensive_stats(self) -> Dict:
        """获取包含ECBS统计的综合信息"""
        stats = super().get_comprehensive_stats()
        
        # 添加ECBS统计
        stats['ecbs_stats'] = self.conflict_stats.copy()
        
        # 添加交通管理器统计
        if self.traffic_manager:
            traffic_stats = self.traffic_manager.get_comprehensive_stats()
            stats['traffic_management'] = traffic_stats
        
        return stats

# 向后兼容性
VehicleScheduler = OptimizedVehicleScheduler
SimplifiedVehicleScheduler = OptimizedVehicleScheduler
SimplifiedECBSVehicleScheduler = ECBSEnhancedVehicleScheduler