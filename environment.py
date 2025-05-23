"""
environment.py - 优化版环境管理系统
提供统一的环境状态管理、组件集成、数据持久化
"""

import numpy as np
import math
import random
import json
import time
import threading
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from PyQt5.QtGui import QColor
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnvironmentState(Enum):
    """环境状态枚举"""
    UNINITIALIZED = "uninitialized"
    LOADING = "loading"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    SAVING = "saving"
    ERROR = "error"

@dataclass
class EnvironmentConfig:
    """环境配置类"""
    width: int = 500
    height: int = 500
    grid_resolution: float = 1.0
    
    # 性能配置
    max_vehicles: int = 50
    collision_detection_enabled: bool = True
    spatial_indexing_enabled: bool = True
    
    # 仿真配置
    default_time_step: float = 0.5
    auto_save_interval: float = 300.0  # 5分钟自动保存
    
    # 安全配置
    vehicle_safety_margin: float = 2.0
    max_speed: float = 10.0
    
    def validate(self) -> bool:
        """验证配置有效性"""
        return (self.width > 0 and self.height > 0 and 
                self.grid_resolution > 0 and self.max_vehicles > 0)

@dataclass
class VehicleInfo:
    """车辆信息数据类"""
    vehicle_id: str
    position: Tuple[float, float, float]
    initial_position: Tuple[float, float, float]
    goal: Optional[Tuple[float, float, float]] = None
    
    # 基本属性
    vehicle_type: str = "dump_truck"
    max_load: float = 100
    current_load: float = 0
    speed: float = 1.0
    
    # 状态信息
    status: str = 'idle'
    path: Optional[List] = None
    path_index: int = 0
    progress: float = 0.0
    
    # 任务信息
    completed_cycles: int = 0
    path_structure: Dict = field(default_factory=dict)
    
    # 性能统计
    total_distance: float = 0
    total_time: float = 0
    efficiency_metrics: Dict = field(default_factory=dict)
    
    # 显示属性
    color: Optional[QColor] = None
    
    def __post_init__(self):
        if self.color is None:
            self.color = QColor(
                random.randint(100, 255), 
                random.randint(100, 255), 
                random.randint(100, 255)
            )

class ComponentManager:
    """组件管理器 - 统一管理系统组件"""
    
    def __init__(self):
        self.components = {}
        self.component_dependencies = {}
        self.initialization_order = []
        
    def register_component(self, name: str, component: Any, 
                          dependencies: List[str] = None):
        """注册组件"""
        self.components[name] = component
        self.component_dependencies[name] = dependencies or []
        self._update_initialization_order()
    
    def get_component(self, name: str) -> Optional[Any]:
        """获取组件"""
        return self.components.get(name)
    
    def remove_component(self, name: str) -> bool:
        """移除组件"""
        if name in self.components:
            del self.components[name]
            del self.component_dependencies[name]
            self._update_initialization_order()
            return True
        return False
    
    def _update_initialization_order(self):
        """更新初始化顺序（拓扑排序）"""
        # 简化的拓扑排序
        visited = set()
        temp_visited = set()
        self.initialization_order = []
        
        def visit(name):
            if name in temp_visited:
                raise ValueError(f"循环依赖检测到: {name}")
            if name in visited:
                return
            
            temp_visited.add(name)
            for dep in self.component_dependencies.get(name, []):
                if dep in self.components:
                    visit(dep)
            temp_visited.remove(name)
            visited.add(name)
            self.initialization_order.append(name)
        
        for name in self.components:
            if name not in visited:
                visit(name)
    
    def initialize_all(self):
        """按依赖顺序初始化所有组件"""
        for name in self.initialization_order:
            component = self.components[name]
            if hasattr(component, 'initialize'):
                try:
                    component.initialize()
                    logger.info(f"组件 {name} 初始化成功")
                except Exception as e:
                    logger.error(f"组件 {name} 初始化失败: {e}")
                    raise
    
    def shutdown_all(self):
        """关闭所有组件"""
        # 反向顺序关闭
        for name in reversed(self.initialization_order):
            component = self.components[name]
            if hasattr(component, 'shutdown'):
                try:
                    component.shutdown()
                    logger.info(f"组件 {name} 已关闭")
                except Exception as e:
                    logger.error(f"组件 {name} 关闭失败: {e}")
    
    def get_save_data(self) -> Dict:
        """获取所有组件的保存数据"""
        save_data = {}
        for name, component in self.components.items():
            if hasattr(component, 'get_save_data'):
                try:
                    save_data[name] = component.get_save_data()
                except Exception as e:
                    logger.warning(f"组件 {name} 保存数据失败: {e}")
        return save_data
    
    def restore_from_save_data(self, save_data: Dict):
        """从保存数据恢复所有组件"""
        for name in self.initialization_order:
            if name in save_data and name in self.components:
                component = self.components[name]
                if hasattr(component, 'restore_from_save_data'):
                    try:
                        component.restore_from_save_data(save_data[name])
                        logger.info(f"组件 {name} 状态已恢复")
                    except Exception as e:
                        logger.warning(f"组件 {name} 状态恢复失败: {e}")

class SpatialIndex:
    """空间索引 - 优化碰撞检测性能"""
    
    def __init__(self, width: int, height: int, cell_size: float = 20.0):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.grid_width = int(math.ceil(width / cell_size))
        self.grid_height = int(math.ceil(height / cell_size))
        
        # 使用字典存储每个网格单元中的对象
        self.grid = defaultdict(set)
        self.object_positions = {}  # 对象ID到网格位置的映射
    
    def _get_grid_pos(self, x: float, y: float) -> Tuple[int, int]:
        """获取网格坐标"""
        grid_x = int(x / self.cell_size)
        grid_y = int(y / self.cell_size)
        return (max(0, min(grid_x, self.grid_width - 1)),
                max(0, min(grid_y, self.grid_height - 1)))
    
    def add_object(self, obj_id: str, x: float, y: float):
        """添加对象到空间索引"""
        grid_pos = self._get_grid_pos(x, y)
        
        # 如果对象已存在，先移除
        if obj_id in self.object_positions:
            self.remove_object(obj_id)
        
        self.grid[grid_pos].add(obj_id)
        self.object_positions[obj_id] = grid_pos
    
    def remove_object(self, obj_id: str):
        """从空间索引中移除对象"""
        if obj_id in self.object_positions:
            old_pos = self.object_positions[obj_id]
            self.grid[old_pos].discard(obj_id)
            del self.object_positions[obj_id]
    
    def get_nearby_objects(self, x: float, y: float, radius: float) -> List[str]:
        """获取附近的对象"""
        nearby_objects = set()
        
        # 计算需要检查的网格范围
        grid_radius = int(math.ceil(radius / self.cell_size))
        center_x, center_y = self._get_grid_pos(x, y)
        
        for dx in range(-grid_radius, grid_radius + 1):
            for dy in range(-grid_radius, grid_radius + 1):
                grid_x = center_x + dx
                grid_y = center_y + dy
                
                if (0 <= grid_x < self.grid_width and 
                    0 <= grid_y < self.grid_height):
                    nearby_objects.update(self.grid[(grid_x, grid_y)])
        
        return list(nearby_objects)
    
    def clear(self):
        """清空空间索引"""
        self.grid.clear()
        self.object_positions.clear()

class OptimizedOpenPitMineEnv:
    """优化版露天矿环境 - 统一管理、高性能、易扩展"""
    
    def __init__(self, config: EnvironmentConfig = None):
        """初始化环境"""
        # 配置管理
        self.config = config or EnvironmentConfig()
        if not self.config.validate():
            raise ValueError("环境配置无效")
        
        # 基本属性
        self.width = self.config.width
        self.height = self.config.height
        self.grid_resolution = self.config.grid_resolution
        self.map_size = (self.width, self.height)
        
        # 状态管理
        self.state = EnvironmentState.UNINITIALIZED
        self.state_lock = threading.Lock()
        
        # 地图数据
        self.grid = np.zeros((self.width, self.height), dtype=np.uint8)
        self.obstacle_points = []
        
        # 关键点位置
        self.loading_points = []
        self.unloading_points = []
        self.parking_areas = []
        
        # 车辆管理 - 使用新的VehicleInfo类
        self.vehicles = OrderedDict()  # {vehicle_id: VehicleInfo}
        
        # 空间索引（性能优化）
        self.spatial_index = None
        if self.config.spatial_indexing_enabled:
            self.spatial_index = SpatialIndex(self.width, self.height)
        
        # 组件管理
        self.component_manager = ComponentManager()
        
        # 仿真状态
        self.current_time = 0.0
        self.time_step = self.config.default_time_step
        self.running = False
        self.paused = False
        
        # 统计信息
        self.stats = {
            'total_vehicles': 0,
            'active_vehicles': 0,
            'collision_checks': 0,
            'performance_metrics': {
                'avg_fps': 0,
                'memory_usage': 0,
                'cpu_usage': 0
            }
        }
        
        # 事件系统
        self.event_listeners = defaultdict(list)
        
        # 自动保存
        self.last_auto_save = time.time()
        self.auto_save_enabled = False
        
        # 错误处理
        self.error_count = 0
        self.last_error_time = 0
        
        logger.info(f"环境初始化完成: {self.width}x{self.height}")
        self.state = EnvironmentState.READY
    
    # ==================== 状态管理 ====================
    
    def set_state(self, new_state: EnvironmentState):
        """安全地设置环境状态"""
        with self.state_lock:
            old_state = self.state
            self.state = new_state
            self._emit_event('state_changed', {
                'old_state': old_state,
                'new_state': new_state,
                'timestamp': time.time()
            })
    
    def get_state(self) -> EnvironmentState:
        """获取当前环境状态"""
        return self.state
    
    def is_ready(self) -> bool:
        """检查环境是否就绪"""
        return self.state in [EnvironmentState.READY, EnvironmentState.RUNNING, EnvironmentState.PAUSED]
    
    def is_running(self) -> bool:
        """检查模拟是否正在运行"""
        return self.state == EnvironmentState.RUNNING and not self.paused
    
    # ==================== 事件系统 ====================
    
    def add_event_listener(self, event_type: str, callback: Callable):
        """添加事件监听器"""
        self.event_listeners[event_type].append(callback)
    
    def remove_event_listener(self, event_type: str, callback: Callable):
        """移除事件监听器"""
        if callback in self.event_listeners[event_type]:
            self.event_listeners[event_type].remove(callback)
    
    def _emit_event(self, event_type: str, data: Any = None):
        """发出事件"""
        for callback in self.event_listeners[event_type]:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"事件监听器错误 {event_type}: {e}")
    
    # ==================== 组件集成 ====================
    
    def register_component(self, name: str, component: Any, dependencies: List[str] = None):
        """注册系统组件"""
        self.component_manager.register_component(name, component, dependencies)
        logger.info(f"注册组件: {name}")
    
    def get_component(self, name: str) -> Optional[Any]:
        """获取组件"""
        return self.component_manager.get_component(name)
    
    def set_backbone_network(self, backbone_network):
        """设置骨干路径网络"""
        self.register_component('backbone_network', backbone_network)
    
    def set_vehicle_scheduler(self, scheduler):
        """设置车辆调度器"""
        self.register_component('vehicle_scheduler', scheduler, ['backbone_network'])
    
    def set_traffic_manager(self, traffic_manager):
        """设置交通管理器"""
        self.register_component('traffic_manager', traffic_manager, ['backbone_network'])
    
    def set_path_planner(self, path_planner):
        """设置路径规划器"""
        self.register_component('path_planner', path_planner, ['backbone_network'])
    
    # ==================== 地图管理 ====================
    
    def add_obstacle_point(self, x: int, y: int) -> bool:
        """添加单个障碍物点"""
        try:
            # 坐标验证和修正
            x = max(0, min(int(x), self.width - 1))
            y = max(0, min(int(y), self.height - 1))
            
            # 设置网格障碍物
            self.grid[x, y] = 1
            
            # 添加到障碍点列表
            if (x, y) not in self.obstacle_points:
                self.obstacle_points.append((x, y))
                
                # 更新空间索引
                if self.spatial_index:
                    self.spatial_index.add_object(f"obstacle_{x}_{y}", float(x), float(y))
            
            self._emit_event('obstacle_added', {'x': x, 'y': y})
            return True
            
        except Exception as e:
            logger.error(f"添加障碍物失败 ({x}, {y}): {e}")
            return False
    
    def add_obstacle(self, x: int, y: int, width: int = 1, height: int = 1) -> bool:
        """添加矩形障碍物区域"""
        try:
            success_count = 0
            total_count = width * height
            
            x1 = max(0, min(x, self.width - 1))
            y1 = max(0, min(y, self.height - 1))
            x2 = max(0, min(x + width, self.width))
            y2 = max(0, min(y + height, self.height))
            
            for i in range(x1, x2):
                for j in range(y1, y2):
                    if self.add_obstacle_point(i, j):
                        success_count += 1
            
            success_rate = success_count / total_count if total_count > 0 else 0
            logger.info(f"添加障碍物区域: {success_count}/{total_count} ({success_rate:.1%})")
            
            return success_rate > 0.8  # 80%以上成功率认为成功
            
        except Exception as e:
            logger.error(f"添加障碍物区域失败: {e}")
            return False
    
    def remove_obstacle_point(self, x: int, y: int) -> bool:
        """移除障碍物点"""
        try:
            x = max(0, min(int(x), self.width - 1))
            y = max(0, min(int(y), self.height - 1))
            
            if self.grid[x, y] == 1:
                self.grid[x, y] = 0
                
                if (x, y) in self.obstacle_points:
                    self.obstacle_points.remove((x, y))
                
                # 更新空间索引
                if self.spatial_index:
                    self.spatial_index.remove_object(f"obstacle_{x}_{y}")
                
                self._emit_event('obstacle_removed', {'x': x, 'y': y})
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"移除障碍物失败 ({x}, {y}): {e}")
            return False
    
    def clear_obstacles(self):
        """清除所有障碍物"""
        self.grid.fill(0)
        self.obstacle_points.clear()
        if self.spatial_index:
            # 只清除障碍物对象
            obstacle_ids = [obj_id for obj_id in self.spatial_index.object_positions 
                           if obj_id.startswith('obstacle_')]
            for obj_id in obstacle_ids:
                self.spatial_index.remove_object(obj_id)
        
        self._emit_event('obstacles_cleared', {})
        logger.info("已清除所有障碍物")
    
    # ==================== 关键点管理 ====================
    
    def add_loading_point(self, position: Tuple[float, float, float], capacity: int = 1) -> int:
        """添加装载点"""
        try:
            if len(position) < 2:
                raise ValueError("位置坐标不完整")
            
            x, y = float(position[0]), float(position[1])
            theta = float(position[2]) if len(position) > 2 else 0.0
            
            # 坐标验证
            if not (0 <= x < self.width and 0 <= y < self.height):
                raise ValueError(f"坐标超出范围: ({x}, {y})")
            
            # 检查是否为障碍物
            if self.grid[int(x), int(y)] == 1:
                raise ValueError(f"位置 ({x}, {y}) 是障碍物")
            
            # 添加装载点
            point = (x, y, theta)
            self.loading_points.append(point)
            index = len(self.loading_points) - 1
            
            # 更新空间索引
            if self.spatial_index:
                self.spatial_index.add_object(f"loading_{index}", x, y)
            
            self._emit_event('loading_point_added', {
                'index': index, 'position': point, 'capacity': capacity
            })
            
            logger.info(f"添加装载点 {index}: ({x:.1f}, {y:.1f})")
            return index
            
        except Exception as e:
            logger.error(f"添加装载点失败: {e}")
            return -1
    
    def add_unloading_point(self, position: Tuple[float, float, float], capacity: int = 1) -> int:
        """添加卸载点"""
        try:
            if len(position) < 2:
                raise ValueError("位置坐标不完整")
            
            x, y = float(position[0]), float(position[1])
            theta = float(position[2]) if len(position) > 2 else 0.0
            
            # 坐标验证
            if not (0 <= x < self.width and 0 <= y < self.height):
                raise ValueError(f"坐标超出范围: ({x}, {y})")
            
            # 检查是否为障碍物
            if self.grid[int(x), int(y)] == 1:
                raise ValueError(f"位置 ({x}, {y}) 是障碍物")
            
            # 添加卸载点
            point = (x, y, theta)
            self.unloading_points.append(point)
            index = len(self.unloading_points) - 1
            
            # 更新空间索引
            if self.spatial_index:
                self.spatial_index.add_object(f"unloading_{index}", x, y)
            
            self._emit_event('unloading_point_added', {
                'index': index, 'position': point, 'capacity': capacity
            })
            
            logger.info(f"添加卸载点 {index}: ({x:.1f}, {y:.1f})")
            return index
            
        except Exception as e:
            logger.error(f"添加卸载点失败: {e}")
            return -1
    
    def add_parking_area(self, position: Tuple[float, float, float], capacity: int = 5) -> int:
        """添加停车区"""
        try:
            if len(position) < 2:
                raise ValueError("位置坐标不完整")
            
            x, y = float(position[0]), float(position[1])
            theta = float(position[2]) if len(position) > 2 else 0.0
            
            # 坐标验证
            if not (0 <= x < self.width and 0 <= y < self.height):
                raise ValueError(f"坐标超出范围: ({x}, {y})")
            
            # 检查是否为障碍物
            if self.grid[int(x), int(y)] == 1:
                raise ValueError(f"位置 ({x}, {y}) 是障碍物")
            
            # 添加停车区
            point = (x, y, theta)
            self.parking_areas.append(point)
            index = len(self.parking_areas) - 1
            
            # 更新空间索引
            if self.spatial_index:
                self.spatial_index.add_object(f"parking_{index}", x, y)
            
            self._emit_event('parking_area_added', {
                'index': index, 'position': point, 'capacity': capacity
            })
            
            logger.info(f"添加停车区 {index}: ({x:.1f}, {y:.1f})")
            return index
            
        except Exception as e:
            logger.error(f"添加停车区失败: {e}")
            return -1
    
    def get_nearest_point(self, position: Tuple[float, float], point_type: str = "loading") -> Tuple[Optional[Tuple], int]:
        """获取最近的特定类型点位"""
        try:
            x, y = float(position[0]), float(position[1])
            
            # 选择点位列表
            if point_type == "loading":
                points = self.loading_points
            elif point_type == "unloading":
                points = self.unloading_points
            elif point_type == "parking":
                points = self.parking_areas
            else:
                raise ValueError(f"未知的点位类型: {point_type}")
            
            if not points:
                return None, -1
            
            # 查找最近点
            min_dist = float('inf')
            nearest_idx = -1
            
            for i, point in enumerate(points):
                dist = math.sqrt((x - point[0])**2 + (y - point[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = i
            
            if nearest_idx != -1:
                return points[nearest_idx], nearest_idx
            else:
                return None, -1
                
        except Exception as e:
            logger.error(f"获取最近点位失败: {e}")
            return None, -1
    
    # ==================== 车辆管理 ====================
    
    def add_vehicle(self, vehicle_id: str, position: Tuple[float, float, float], 
                   goal: Tuple[float, float, float] = None, 
                   vehicle_type: str = "dump_truck", max_load: float = 100) -> bool:
        """添加车辆"""
        try:
            # 检查车辆数量限制
            if len(self.vehicles) >= self.config.max_vehicles:
                raise ValueError(f"车辆数量超出限制: {self.config.max_vehicles}")
            
            # 检查车辆ID是否已存在
            if vehicle_id in self.vehicles:
                raise ValueError(f"车辆ID已存在: {vehicle_id}")
            
            # 位置验证
            if len(position) < 2:
                raise ValueError("位置坐标不完整")
            
            x, y = float(position[0]), float(position[1])
            theta = float(position[2]) if len(position) > 2 else 0.0
            
            if not (0 <= x < self.width and 0 <= y < self.height):
                raise ValueError(f"车辆位置超出范围: ({x}, {y})")
            
            # 碰撞检测
            if self.config.collision_detection_enabled:
                if self.check_collision((x, y, theta)):
                    raise ValueError(f"车辆位置有碰撞: ({x}, {y})")
            
            # 创建车辆信息
            vehicle_info = VehicleInfo(
                vehicle_id=vehicle_id,
                position=(x, y, theta),
                initial_position=(x, y, theta),
                goal=goal,
                vehicle_type=vehicle_type,
                max_load=max_load
            )
            
            # 添加到车辆字典
            self.vehicles[vehicle_id] = vehicle_info
            
            # 更新空间索引
            if self.spatial_index:
                self.spatial_index.add_object(f"vehicle_{vehicle_id}", x, y)
            
            # 更新统计
            self.stats['total_vehicles'] = len(self.vehicles)
            
            self._emit_event('vehicle_added', {
                'vehicle_id': vehicle_id, 'vehicle_info': vehicle_info
            })
            
            logger.info(f"添加车辆 {vehicle_id}: ({x:.1f}, {y:.1f})")
            return True
            
        except Exception as e:
            logger.error(f"添加车辆失败 {vehicle_id}: {e}")
            return False
    
    def remove_vehicle(self, vehicle_id: str) -> bool:
        """移除车辆"""
        try:
            if vehicle_id not in self.vehicles:
                return False
            
            # 从空间索引中移除
            if self.spatial_index:
                self.spatial_index.remove_object(f"vehicle_{vehicle_id}")
            
            # 移除车辆
            vehicle_info = self.vehicles[vehicle_id]
            del self.vehicles[vehicle_id]
            
            # 更新统计
            self.stats['total_vehicles'] = len(self.vehicles)
            
            self._emit_event('vehicle_removed', {
                'vehicle_id': vehicle_id, 'vehicle_info': vehicle_info
            })
            
            logger.info(f"移除车辆 {vehicle_id}")
            return True
            
        except Exception as e:
            logger.error(f"移除车辆失败 {vehicle_id}: {e}")
            return False
    
    def update_vehicle_position(self, vehicle_id: str, position: Tuple[float, float, float]) -> bool:
        """更新车辆位置"""
        try:
            if vehicle_id not in self.vehicles:
                return False
            
            x, y, theta = float(position[0]), float(position[1]), float(position[2])
            
            # 碰撞检测
            if self.config.collision_detection_enabled:
                if self.check_collision((x, y, theta), exclude_vehicle=vehicle_id):
                    return False
            
            # 更新位置
            vehicle_info = self.vehicles[vehicle_id]
            old_position = vehicle_info.position
            vehicle_info.position = (x, y, theta)
            
            # 更新空间索引
            if self.spatial_index:
                self.spatial_index.add_object(f"vehicle_{vehicle_id}", x, y)
            
            # 计算距离
            if old_position:
                distance = math.sqrt(
                    (x - old_position[0])**2 + (y - old_position[1])**2
                )
                vehicle_info.total_distance += distance
            
            self._emit_event('vehicle_position_updated', {
                'vehicle_id': vehicle_id,
                'old_position': old_position,
                'new_position': (x, y, theta)
            })
            
            return True
            
        except Exception as e:
            logger.error(f"更新车辆位置失败 {vehicle_id}: {e}")
            return False
    
    def get_vehicle_info(self, vehicle_id: str) -> Optional[VehicleInfo]:
        """获取车辆信息"""
        return self.vehicles.get(vehicle_id)
    
    def get_vehicles_in_area(self, center: Tuple[float, float], radius: float) -> List[str]:
        """获取区域内的车辆"""
        try:
            if self.spatial_index:
                # 使用空间索引加速查询
                nearby_objects = self.spatial_index.get_nearby_objects(
                    center[0], center[1], radius
                )
                vehicle_ids = [obj_id.replace('vehicle_', '') 
                              for obj_id in nearby_objects 
                              if obj_id.startswith('vehicle_')]
                return vehicle_ids
            else:
                # 暴力搜索
                vehicles_in_area = []
                for vehicle_id, vehicle_info in self.vehicles.items():
                    pos = vehicle_info.position
                    distance = math.sqrt(
                        (pos[0] - center[0])**2 + (pos[1] - center[1])**2
                    )
                    if distance <= radius:
                        vehicles_in_area.append(vehicle_id)
                return vehicles_in_area
                
        except Exception as e:
            logger.error(f"获取区域车辆失败: {e}")
            return []
    
    # ==================== 碰撞检测 ====================
    
    def check_collision(self, position: Tuple[float, float, float], 
                       vehicle_dim: Tuple[float, float] = (6, 3),
                       exclude_vehicle: str = None) -> bool:
        """优化的碰撞检测"""
        try:
            if not self.config.collision_detection_enabled:
                return False
            
            self.stats['collision_checks'] += 1
            
            x, y, theta = float(position[0]), float(position[1]), float(position[2])
            length, width = vehicle_dim
            
            # 快速边界检查
            margin = max(length, width) / 2
            if (x - margin < 0 or x + margin >= self.width or 
                y - margin < 0 or y + margin >= self.height):
                return True
            
            # 中心点障碍物检查
            ix, iy = int(x), int(y)
            if (0 <= ix < self.width and 0 <= iy < self.height and 
                self.grid[ix, iy] == 1):
                return True
            
            # 详细车辆形状检查
            if self._detailed_collision_check(x, y, theta, length, width):
                return True
            
            # 与其他车辆的碰撞检测
            if self._check_vehicle_collisions(position, exclude_vehicle):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"碰撞检测失败: {e}")
            return True  # 保守策略：出错时认为有碰撞
    
    def _detailed_collision_check(self, x: float, y: float, theta: float, 
                                 length: float, width: float) -> bool:
        """详细的车辆形状碰撞检测"""
        half_length, half_width = length / 2, width / 2
        cos_theta, sin_theta = math.cos(theta), math.sin(theta)
        
        # 车辆四个角点
        corners_rel = [
            (half_length, half_width),
            (half_length, -half_width),
            (-half_length, -half_width),
            (-half_length, half_width)
        ]
        
        # 检查角点
        for dx, dy in corners_rel:
            # 旋转后的绝对坐标
            abs_x = int(x + dx * cos_theta - dy * sin_theta)
            abs_y = int(y + dx * sin_theta + dy * cos_theta)
            
            # 边界检查
            if (abs_x < 0 or abs_x >= self.width or 
                abs_y < 0 or abs_y >= self.height):
                return True
            
            # 障碍物检查
            if self.grid[abs_x, abs_y] == 1:
                return True
        
        return False
    
    def _check_vehicle_collisions(self, position: Tuple[float, float, float], 
                                 exclude_vehicle: str = None) -> bool:
        """检查与其他车辆的碰撞"""
        x, y = position[0], position[1]
        safety_radius = self.config.vehicle_safety_margin
        
        # 获取附近的车辆
        nearby_vehicles = self.get_vehicles_in_area((x, y), safety_radius * 2)
        
        for vehicle_id in nearby_vehicles:
            if vehicle_id == exclude_vehicle:
                continue
            
            vehicle_info = self.vehicles[vehicle_id]
            other_pos = vehicle_info.position
            
            # 计算距离
            distance = math.sqrt(
                (x - other_pos[0])**2 + (y - other_pos[1])**2
            )
            
            if distance < safety_radius:
                return True
        
        return False
    
    # ==================== 仿真控制 ====================
    
    def start(self):
        """开始仿真"""
        if self.state == EnvironmentState.READY or self.state == EnvironmentState.PAUSED:
            self.running = True
            self.paused = False
            self.set_state(EnvironmentState.RUNNING)
            self._emit_event('simulation_started', {'time': self.current_time})
            logger.info("仿真已开始")
    
    def pause(self):
        """暂停仿真"""
        if self.state == EnvironmentState.RUNNING:
            self.paused = True
            self.set_state(EnvironmentState.PAUSED)
            self._emit_event('simulation_paused', {'time': self.current_time})
            logger.info("仿真已暂停")
    
    def resume(self):
        """恢复仿真"""
        if self.state == EnvironmentState.PAUSED:
            self.paused = False
            self.set_state(EnvironmentState.RUNNING)
            self._emit_event('simulation_resumed', {'time': self.current_time})
            logger.info("仿真已恢复")
    
    def stop(self):
        """停止仿真"""
        self.running = False
        self.paused = False
        self.set_state(EnvironmentState.READY)
        self._emit_event('simulation_stopped', {'time': self.current_time})
        logger.info("仿真已停止")
    
    def reset(self):
        """重置环境状态"""
        try:
            self.stop()
            
            # 重置车辆位置和状态
            for vehicle_id, vehicle_info in self.vehicles.items():
                vehicle_info.position = vehicle_info.initial_position
                vehicle_info.current_load = 0
                vehicle_info.status = 'idle'
                vehicle_info.path = None
                vehicle_info.path_index = 0
                vehicle_info.progress = 0.0
                vehicle_info.completed_cycles = 0
                vehicle_info.total_distance = 0
                vehicle_info.total_time = 0
                
                # 更新空间索引
                if self.spatial_index:
                    pos = vehicle_info.position
                    self.spatial_index.add_object(f"vehicle_{vehicle_id}", pos[0], pos[1])
            
            # 重置时间
            self.current_time = 0.0
            
            # 重置统计
            self.stats['collision_checks'] = 0
            self.stats['active_vehicles'] = 0
            
            self._emit_event('environment_reset', {'time': self.current_time})
            logger.info("环境已重置")
            
            return True
            
        except Exception as e:
            logger.error(f"环境重置失败: {e}")
            self.set_state(EnvironmentState.ERROR)
            return False
    
    def update(self, time_delta: float = None):
        """更新环境状态"""
        try:
            if not self.is_running():
                return False
            
            if time_delta is None:
                time_delta = self.time_step
            
            # 更新时间
            self.current_time += time_delta
            
            # 更新车辆统计
            active_count = len([v for v in self.vehicles.values() 
                              if v.status != 'idle'])
            self.stats['active_vehicles'] = active_count
            
            # 更新车辆时间统计
            for vehicle_info in self.vehicles.values():
                if vehicle_info.status != 'idle':
                    vehicle_info.total_time += time_delta
            
            # 自动保存检查
            if (self.auto_save_enabled and 
                time.time() - self.last_auto_save > self.config.auto_save_interval):
                self._auto_save()
            
            self._emit_event('environment_updated', {
                'time': self.current_time, 'time_delta': time_delta
            })
            
            return True
            
        except Exception as e:
            logger.error(f"环境更新失败: {e}")
            self._handle_error(e)
            return False
    
    # ==================== 数据持久化 ====================
    
    def save_to_file(self, filename: str) -> bool:
        """保存环境到文件"""
        try:
            self.set_state(EnvironmentState.SAVING)
            
            # 构建保存数据
            save_data = {
                "metadata": {
                    "version": "2.0_optimized",
                    "timestamp": time.time(),
                    "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "environment_state": self.state.value
                },
                "config": {
                    "width": self.config.width,
                    "height": self.config.height,
                    "grid_resolution": self.config.grid_resolution,
                    "max_vehicles": self.config.max_vehicles,
                    "collision_detection_enabled": self.config.collision_detection_enabled,
                    "spatial_indexing_enabled": self.config.spatial_indexing_enabled
                },
                "environment": {
                    "obstacles": [{"x": x, "y": y} for x, y in self.obstacle_points],
                    "loading_points": [
                        {"x": p[0], "y": p[1], "theta": p[2]}
                        for p in self.loading_points
                    ],
                    "unloading_points": [
                        {"x": p[0], "y": p[1], "theta": p[2]}
                        for p in self.unloading_points
                    ],
                    "parking_areas": [
                        {"x": p[0], "y": p[1], "theta": p[2]}
                        for p in self.parking_areas
                    ]
                },
                "vehicles": {
                    vehicle_id: {
                        "position": vehicle_info.position,
                        "initial_position": vehicle_info.initial_position,
                        "goal": vehicle_info.goal,
                        "vehicle_type": vehicle_info.vehicle_type,
                        "max_load": vehicle_info.max_load,
                        "current_load": vehicle_info.current_load,
                        "status": vehicle_info.status,
                        "completed_cycles": vehicle_info.completed_cycles,
                        "total_distance": vehicle_info.total_distance,
                        "total_time": vehicle_info.total_time,
                        "path_structure": vehicle_info.path_structure,
                        "efficiency_metrics": vehicle_info.efficiency_metrics
                    }
                    for vehicle_id, vehicle_info in self.vehicles.items()
                },
                "simulation": {
                    "current_time": self.current_time,
                    "time_step": self.time_step,
                    "running": self.running,
                    "paused": self.paused
                },
                "statistics": self.stats.copy(),
                "components": self.component_manager.get_save_data()
            }
            
            # 写入文件
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"环境已保存到: {filename}")
            self.set_state(EnvironmentState.READY)
            
            self._emit_event('environment_saved', {'filename': filename})
            return True
            
        except Exception as e:
            logger.error(f"保存环境失败: {e}")
            self.set_state(EnvironmentState.ERROR)
            return False
    
    def load_from_file(self, filename: str) -> bool:
        """从文件加载环境"""
        try:
            self.set_state(EnvironmentState.LOADING)
            
            # 读取文件
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 验证文件格式
            if not self._validate_save_file(data):
                raise ValueError("保存文件格式无效")
            
            # 清空当前环境
            self._clear_environment()
            
            # 加载配置
            config_data = data.get("config", {})
            self.config = EnvironmentConfig(**config_data)
            self.width = self.config.width
            self.height = self.config.height
            self.grid_resolution = self.config.grid_resolution
            
            # 重新初始化网格和空间索引
            self.grid = np.zeros((self.width, self.height), dtype=np.uint8)
            if self.config.spatial_indexing_enabled:
                self.spatial_index = SpatialIndex(self.width, self.height)
            else:
                self.spatial_index = None
            
            # 加载环境数据
            env_data = data.get("environment", {})
            
            # 加载障碍物
            for obstacle in env_data.get("obstacles", []):
                self.add_obstacle_point(obstacle["x"], obstacle["y"])
            
            # 加载关键点
            for point_data in env_data.get("loading_points", []):
                self.add_loading_point((point_data["x"], point_data["y"], point_data["theta"]))
            
            for point_data in env_data.get("unloading_points", []):
                self.add_unloading_point((point_data["x"], point_data["y"], point_data["theta"]))
            
            for point_data in env_data.get("parking_areas", []):
                self.add_parking_area((point_data["x"], point_data["y"], point_data["theta"]))
            
            # 加载车辆
            vehicles_data = data.get("vehicles", {})
            for vehicle_id, vehicle_data in vehicles_data.items():
                self._load_vehicle_from_data(vehicle_id, vehicle_data)
            
            # 加载仿真状态
            sim_data = data.get("simulation", {})
            self.current_time = sim_data.get("current_time", 0.0)
            self.time_step = sim_data.get("time_step", self.config.default_time_step)
            self.running = sim_data.get("running", False)
            self.paused = sim_data.get("paused", False)
            
            # 加载统计信息
            if "statistics" in data:
                self.stats.update(data["statistics"])
            
            # 恢复组件状态
            if "components" in data:
                self.component_manager.restore_from_save_data(data["components"])
            
            logger.info(f"环境已从 {filename} 加载")
            self.set_state(EnvironmentState.READY)
            
            self._emit_event('environment_loaded', {'filename': filename})
            return True
            
        except Exception as e:
            logger.error(f"加载环境失败: {e}")
            self.set_state(EnvironmentState.ERROR)
            return False
    
    def _validate_save_file(self, data: Dict) -> bool:
        """验证保存文件格式"""
        required_keys = ["metadata", "config", "environment"]
        return all(key in data for key in required_keys)
    
    def _clear_environment(self):
        """清空环境数据"""
        self.obstacle_points.clear()
        self.loading_points.clear()
        self.unloading_points.clear()
        self.parking_areas.clear()
        self.vehicles.clear()
        
        if self.spatial_index:
            self.spatial_index.clear()
    
    def _load_vehicle_from_data(self, vehicle_id: str, vehicle_data: Dict):
        """从数据加载车辆"""
        try:
            # 创建车辆信息对象
            vehicle_info = VehicleInfo(
                vehicle_id=vehicle_id,
                position=tuple(vehicle_data["position"]),
                initial_position=tuple(vehicle_data["initial_position"]),
                goal=tuple(vehicle_data["goal"]) if vehicle_data.get("goal") else None,
                vehicle_type=vehicle_data.get("vehicle_type", "dump_truck"),
                max_load=vehicle_data.get("max_load", 100),
                current_load=vehicle_data.get("current_load", 0),
                status=vehicle_data.get("status", "idle"),
                completed_cycles=vehicle_data.get("completed_cycles", 0),
                path_structure=vehicle_data.get("path_structure", {}),
                total_distance=vehicle_data.get("total_distance", 0),
                total_time=vehicle_data.get("total_time", 0),
                efficiency_metrics=vehicle_data.get("efficiency_metrics", {})
            )
            
            # 添加到车辆字典
            self.vehicles[vehicle_id] = vehicle_info
            
            # 更新空间索引
            if self.spatial_index:
                pos = vehicle_info.position
                self.spatial_index.add_object(f"vehicle_{vehicle_id}", pos[0], pos[1])
            
        except Exception as e:
            logger.warning(f"加载车辆 {vehicle_id} 失败: {e}")
    
    def _auto_save(self):
        """自动保存"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"auto_save_{timestamp}.json"
            if self.save_to_file(filename):
                self.last_auto_save = time.time()
                logger.info(f"自动保存完成: {filename}")
        except Exception as e:
            logger.error(f"自动保存失败: {e}")
    
    # ==================== 错误处理 ====================
    
    def _handle_error(self, error: Exception):
        """统一的错误处理"""
        self.error_count += 1
        self.last_error_time = time.time()
        
        # 错误恢复策略
        if self.error_count > 10:  # 如果错误太多，停止仿真
            self.stop()
            self.set_state(EnvironmentState.ERROR)
            logger.critical(f"错误过多，环境已停止: {error}")
        else:
            logger.error(f"环境错误 #{self.error_count}: {error}")
        
        self._emit_event('error_occurred', {
            'error': str(error),
            'error_count': self.error_count,
            'timestamp': time.time()
        })
    
    # ==================== 性能监控和统计 ====================
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计信息"""
        return {
            'environment': {
                'state': self.state.value,
                'running_time': self.current_time,
                'total_vehicles': len(self.vehicles),
                'active_vehicles': self.stats['active_vehicles'],
                'collision_checks': self.stats['collision_checks'],
                'error_count': self.error_count
            },
            'spatial_index': {
                'enabled': self.spatial_index is not None,
                'objects_count': len(self.spatial_index.object_positions) if self.spatial_index else 0
            },
            'components': {
                'registered_count': len(self.component_manager.components),
                'component_names': list(self.component_manager.components.keys())
            },
            'memory': {
                'vehicles_size': len(self.vehicles),
                'obstacles_size': len(self.obstacle_points),
                'loading_points_size': len(self.loading_points),
                'unloading_points_size': len(self.unloading_points)
            }
        }
    
    def enable_auto_save(self, enabled: bool = True):
        """启用/禁用自动保存"""
        self.auto_save_enabled = enabled
        if enabled:
            self.last_auto_save = time.time()
        logger.info(f"自动保存 {'启用' if enabled else '禁用'}")
    
    # ==================== 兼容性方法 ====================
    
    def set_changed_callback(self, callback: Callable):
        """设置变化回调（兼容性）"""
        self.add_event_listener('environment_updated', lambda data: callback(self))
        self.add_event_listener('vehicle_added', lambda data: callback(self))
        self.add_event_listener('vehicle_removed', lambda data: callback(self))
        self.add_event_listener('vehicle_position_updated', lambda data: callback(self))
    
    # ==================== 清理和关闭 ====================
    
    def shutdown(self):
        """关闭环境，清理资源"""
        try:
            self.stop()
            self.component_manager.shutdown_all()
            
            # 清理事件监听器
            self.event_listeners.clear()
            
            # 清理空间索引
            if self.spatial_index:
                self.spatial_index.clear()
            
            logger.info("环境已关闭")
            
        except Exception as e:
            logger.error(f"环境关闭失败: {e}")

# ==================== 向后兼容性 ====================

# 为了保持向后兼容性，提供原始类名的别名
OpenPitMineEnv = OptimizedOpenPitMineEnv