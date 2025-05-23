import numpy as np
import math
import random
import json
from collections import defaultdict
from PyQt5.QtGui import QColor

class OpenPitMineEnv:
    """露天矿环境模型，提供地图数据和资源信息
    
    环境模型主要维护以下信息：
    1. 网格地图和障碍物
    2. 关键点位置（装载点、卸载点、停车区）
    3. 车辆信息和状态
    4. 环境状态（时间、运行状态等）
    """
    
    def __init__(self, width=500, height=500, grid_resolution=1.0):
        """初始化环境
        
        Args:
            width (int): 环境宽度
            height (int): 环境高度
            grid_resolution (float): 网格分辨率
        """
        # 环境尺寸
        self.width = width
        self.height = height
        self.grid_resolution = grid_resolution
        self.map_size = (width, height)
        
        # 创建网格地图 (0-可通行 1-障碍物)
        self.grid = np.zeros((width, height), dtype=np.uint8)
        self.backbone_network = None
        self.vehicle_scheduler = None
        self.traffic_manager = None
        
        # 保存的状态数据
        self._saved_backbone_data = None
        self._saved_interface_states = None
        self._saved_scheduler_states = None
        self._saved_traffic_states = None
        # 关键点位置
        self.loading_points = []     # 装载点 [(x, y, theta), ...]
        self.unloading_points = []   # 卸载点 [(x, y, theta), ...]
        self.parking_areas = []      # 停车区 [(x, y, theta), ...]
        
        # 车辆信息
        self.vehicles = {}           # 车辆信息字典 {vehicle_id: vehicle_data}
        
        # 环境计时器和状态
        self.current_time = 0.0
        self.time_step = 0.5  # 默认时间步长
        self.running = False
        self.paused = False
        
        # 障碍物点列表
        self.obstacle_points = []    # 障碍物散点 [(x, y), ...]
        
        # 变化回调函数
        self.changed_callback = None
        
    def add_obstacle_point(self, x, y):
        """添加单个障碍物点
        
        Args:
            x (int): 障碍物x坐标
            y (int): 障碍物y坐标
            
        Returns:
            bool: 添加是否成功
        """
        # 确保坐标在范围内
        x = max(0, min(int(x), self.width - 1))
        y = max(0, min(int(y), self.height - 1))
        
        # 设置网格中的障碍物
        self.grid[x, y] = 1
        
        # 添加到障碍点列表中
        if (x, y) not in self.obstacle_points:
            self.obstacle_points.append((x, y))
        
        # 通知环境变化
        self._notify_change()
        
        return True
        
    def add_obstacle(self, x, y, width=1, height=1):
        """添加障碍物区域
        
        Args:
            x (int): 障碍物左上角x坐标
            y (int): 障碍物左上角y坐标
            width (int): 障碍物宽度
            height (int): 障碍物高度
            
        Returns:
            bool: 添加是否成功
        """
        # 确保坐标在范围内
        x1 = max(0, min(x, self.width-1))
        y1 = max(0, min(y, self.height-1))
        x2 = max(0, min(x + width, self.width))
        y2 = max(0, min(y + height, self.height))
        
        # 将区域转换为散点添加
        for i in range(x1, x2):
            for j in range(y1, y2):
                self.add_obstacle_point(i, j)
        
        return True
        
    def add_loading_point(self, position, capacity=1):
        """添加装载点
        
        Args:
            position (tuple): 装载点位置 (x, y) 或 (x, y, theta)
            capacity (int): 同时装载容量
            
        Returns:
            int: 添加的装载点索引
        """
        # 确保position至少有x,y两个值
        if len(position) < 2:
            return -1
            
        x, y = position[0], position[1]
        theta = position[2] if len(position) > 2 else 0.0
        
        # 确保坐标在范围内
        if not (0 <= x < self.width and 0 <= y < self.height):
            return -1
            
        # 检查位置是否为障碍物
        if self.grid[int(x), int(y)] == 1:
            return -1
            
        # 添加到列表
        self.loading_points.append((x, y, theta))
        
        # 通知环境变化
        self._notify_change()
        
        return len(self.loading_points) - 1
        
    def add_unloading_point(self, position, capacity=1):
        """添加卸载点
        
        Args:
            position (tuple): 卸载点位置 (x, y) 或 (x, y, theta)
            capacity (int): 同时卸载容量
            
        Returns:
            int: 添加的卸载点索引
        """
        # 确保position至少有x,y两个值
        if len(position) < 2:
            return -1
            
        x, y = position[0], position[1]
        theta = position[2] if len(position) > 2 else 0.0
        
        # 确保坐标在范围内
        if not (0 <= x < self.width and 0 <= y < self.height):
            return -1
            
        # 检查位置是否为障碍物
        if self.grid[int(x), int(y)] == 1:
            return -1
            
        # 添加到列表
        self.unloading_points.append((x, y, theta))
        
        # 通知环境变化
        self._notify_change()
        
        return len(self.unloading_points) - 1
        
    def add_parking_area(self, position, capacity=5):
        """添加停车区
        
        Args:
            position (tuple): 停车区中心位置 (x, y) 或 (x, y, theta)
            capacity (int): 停车容量
            
        Returns:
            int: 添加的停车区索引
        """
        # 确保position至少有x,y两个值
        if len(position) < 2:
            return -1
            
        x, y = position[0], position[1]
        theta = position[2] if len(position) > 2 else 0.0
        
        # 确保坐标在范围内
        if not (0 <= x < self.width and 0 <= y < self.height):
            return -1
            
        # 检查位置是否为障碍物
        if self.grid[int(x), int(y)] == 1:
            return -1
            
        # 添加到列表
        self.parking_areas.append((x, y, theta))
        
        # 通知环境变化
        self._notify_change()
        
        return len(self.parking_areas) - 1
        
    def add_vehicle(self, vehicle_id, position, goal=None, vehicle_type="dump_truck", max_load=100):
        """添加车辆
        
        Args:
            vehicle_id: 车辆ID
            position (tuple): 车辆初始位置 (x, y, theta)
            goal (tuple, optional): 车辆目标位置 (x, y, theta)
            vehicle_type (str): 车辆类型
            max_load (float): 最大载重
            
        Returns:
            bool: 添加是否成功
        """
        # 确保position至少有x,y,theta三个值
        if len(position) < 3:
            x, y = position[0], position[1]
            theta = 0.0
        else:
            x, y, theta = position
            
        # 确保坐标在范围内
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
            
        # 创建车辆数据
        vehicle = {
            'position': (x, y, theta),
            'initial_position': (x, y, theta),  # 保存初始位置用于任务循环
            'goal': goal,
            'type': vehicle_type,
            'max_load': max_load,
            'load': 0,  # 当前载重
            'status': 'idle',  # 'idle', 'moving', 'loading', 'unloading'
            'path': None,  # 当前规划路径
            'path_index': 0,  # 当前在路径上的索引
            'progress': 0.0,  # 当前路径段的进度(0.0-1.0)
            'completed_cycles': 0,  # 完成的任务循环数
            'color': QColor(random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))  # 随机颜色
        }
        
        # 添加车辆
        self.vehicles[vehicle_id] = vehicle
        
        # 通知环境变化
        self._notify_change()
        
        return True
    
    def remove_vehicle(self, vehicle_id):
        """移除车辆
        
        Args:
            vehicle_id: 要移除的车辆ID
            
        Returns:
            bool: 是否成功移除
        """
        if vehicle_id in self.vehicles:
            del self.vehicles[vehicle_id]
            # 通知环境变化
            self._notify_change()
            return True
        return False
        
    def check_collision(self, position, vehicle_dim=(6, 3)):
        """检查位置是否碰撞
        
        Args:
            position (tuple): 要检查的位置 (x, y, theta)
            vehicle_dim (tuple): 车辆尺寸 (长, 宽)
            
        Returns:
            bool: 是否有碰撞
        """
        x, y, theta = position
        length, width = vehicle_dim
        
        # 转换为整数坐标
        ix, iy = int(x), int(y)
        
        # 快速检查：如果中心点在边界外或是障碍物，则碰撞
        if (ix < 0 or ix >= self.width or 
            iy < 0 or iy >= self.height or 
            self.grid[ix, iy] == 1):
            return True
            
        # 详细检查：根据车辆尺寸和方向检查碰撞
        # 计算车辆四个角点相对于中心的偏移
        half_length = length / 2
        half_width = width / 2
        
        # 计算旋转矩阵
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        
        # 四个角点相对于中心的偏移(未旋转)
        corners_rel = [
            (half_length, half_width),   # 右前
            (half_length, -half_width),  # 左前
            (-half_length, -half_width), # 左后
            (-half_length, half_width)   # 右后
        ]
        
        # 应用旋转并计算绝对坐标
        corners_abs = []
        for dx, dy in corners_rel:
            # 旋转
            rotated_dx = dx * cos_theta - dy * sin_theta
            rotated_dy = dx * sin_theta + dy * cos_theta
            
            # 计算绝对坐标
            abs_x = int(x + rotated_dx)
            abs_y = int(y + rotated_dy)
            
            # 检查是否在边界内
            if (abs_x < 0 or abs_x >= self.width or 
                abs_y < 0 or abs_y >= self.height):
                return True
                
            # 检查是否是障碍物
            if self.grid[abs_x, abs_y] == 1:
                return True
                
            corners_abs.append((abs_x, abs_y))
        
        # 检查边界框内的所有点
        # 获取边界框
        min_x = min(x for x, _ in corners_abs)
        max_x = max(x for x, _ in corners_abs)
        min_y = min(y for _, y in corners_abs)
        max_y = max(y for _, y in corners_abs)
        
        # 确保边界在网格范围内
        min_x = max(0, min_x)
        max_x = min(self.width - 1, max_x)
        min_y = max(0, min_y)
        max_y = min(self.height - 1, max_y)
        
        # 检查边界框内的点
        for check_x in range(int(min_x), int(max_x) + 1):
            for check_y in range(int(min_y), int(max_y) + 1):
                if self.grid[check_x, check_y] == 1:
                    # 如果是障碍物，检查点是否在车辆多边形内
                    if self._point_in_polygon(check_x, check_y, corners_abs):
                        return True
        
        # 额外检查：检查是否与其他车辆碰撞
        for other_id, other_vehicle in self.vehicles.items():
            other_pos = other_vehicle['position']
            # 计算与其他车辆的距离
            dist = math.sqrt((x - other_pos[0])**2 + (y - other_pos[1])**2)
            # 简单的距离检查
            if dist < (length + width) / 2:  # 近似车辆为圆形
                return True
                
        return False
    
    def _point_in_polygon(self, x, y, polygon):
        """判断点是否在多边形内
        
        Args:
            x, y: 点坐标
            polygon: 多边形顶点列表
            
        Returns:
            bool: 点是否在多边形内
        """
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
            
        return inside
        
    def get_nearest_point(self, position, point_type="loading"):
        """获取最近的特定类型点位
        
        Args:
            position (tuple): 当前位置 (x, y) 或 (x, y, theta)
            point_type (str): 点位类型 "loading", "unloading", "parking"
            
        Returns:
            tuple: 最近点的位置和索引 ((x, y, theta), index)
        """
        x, y = position[0], position[1]
        
        if point_type == "loading":
            points = self.loading_points
        elif point_type == "unloading":
            points = self.unloading_points
        elif point_type == "parking":
            points = self.parking_areas
        else:
            return None, -1
            
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
    
    def update_vehicle_position(self, vehicle_id, position):
        """更新车辆位置
        
        Args:
            vehicle_id: 车辆ID
            position (tuple): 新位置 (x, y, theta)
            
        Returns:
            bool: 更新是否成功
        """
        if vehicle_id not in self.vehicles:
            return False
            
        # 检查新位置是否有碰撞
        if self.check_collision(position):
            return False
            
        # 更新位置
        self.vehicles[vehicle_id]['position'] = position
        
        # 通知环境变化
        self._notify_change()
        
        return True
    
    def update(self, time_delta):
        """更新环境状态
        
        Args:
            time_delta (float): 时间步长
            
        Returns:
            bool: 更新是否成功
        """
        if not self.running or self.paused:
            return False
            
        # 更新环境时间
        self.current_time += time_delta
        
        # 通知环境变化
        self._notify_change()
        
        return True
        
    def reset(self):
        """重置环境状态
        
        Returns:
            bool: 重置是否成功
        """
        # 保留地图和关键点，重置车辆状态
        for vehicle_id, vehicle in self.vehicles.items():
            # 重置位置到初始位置
            vehicle['position'] = vehicle['initial_position']
            # 重置其他状态
            vehicle['load'] = 0
            vehicle['status'] = 'idle'
            vehicle['path'] = None
            vehicle['path_index'] = 0
            vehicle['progress'] = 0.0
            vehicle['completed_cycles'] = 0
            
        # 重置时间
        self.current_time = 0.0
        self.running = False
        self.paused = False
        
        # 通知环境变化
        self._notify_change()
        
        return True
    
    def start(self):
        """开始模拟"""
        self.running = True
        self.paused = False
        # 通知环境变化
        self._notify_change()
        
    def pause(self):
        """暂停模拟"""
        self.paused = True
        # 通知环境变化
        self._notify_change()
        
    def resume(self):
        """恢复模拟"""
        self.paused = False
        # 通知环境变化
        self._notify_change()
        
    def stop(self):
        """停止模拟"""
        self.running = False
        # 通知环境变化
        self._notify_change()
        
    def set_changed_callback(self, callback):
        """设置环境变化回调函数
        
        Args:
            callback: 回调函数，接受环境对象作为参数
        """
        self.changed_callback = callback
        
    def _notify_change(self):
        """通知环境变化"""
        if self.changed_callback:
            self.changed_callback(self)
            
    def save_to_file(self, filename):
        """保存环境到文件 - 包含接口系统状态
        
        Args:
            filename (str): 文件路径
            
        Returns:
            bool: 保存是否成功
        """
        try:
            # 构建基础数据字典
            data = {
                "width": self.width,
                "height": self.height,
                "resolution": self.grid_resolution,
                "obstacles": [{"x": x, "y": y} for x, y in self.obstacle_points],  # 散点障碍物
                "loading_points": [
                    {"x": point[0], "y": point[1], "theta": point[2] if len(point) > 2 else 0}
                    for point in self.loading_points
                ],
                "unloading_points": [
                    {"x": point[0], "y": point[1], "theta": point[2] if len(point) > 2 else 0}
                    for point in self.unloading_points
                ],
                "parking_areas": [
                    {"x": point[0], "y": point[1], "theta": point[2] if len(point) > 2 else 0}
                    for point in self.parking_areas
                ],
                "vehicles": [
                    {
                        "id": vehicle_id,
                        "x": vehicle["position"][0],
                        "y": vehicle["position"][1],
                        "theta": vehicle["position"][2],
                        "type": vehicle["type"],
                        "max_load": vehicle["max_load"],
                        "status": vehicle.get("status", "idle"),
                        "load": vehicle.get("load", 0),
                        "completed_cycles": vehicle.get("completed_cycles", 0),
                        # 保存路径结构信息
                        "path_structure": vehicle.get("path_structure", {}),
                        # 保存当前路径（如果有的话）
                        "current_path_length": len(vehicle.get("path", [])),
                        "path_index": vehicle.get("path_index", 0),
                        "progress": vehicle.get("progress", 0.0)
                    }
                    for vehicle_id, vehicle in self.vehicles.items()
                ],
                # 保存当前环境时间和状态
                "current_time": getattr(self, 'current_time', 0.0),
                "running": getattr(self, 'running', False),
                "paused": getattr(self, 'paused', False)
            }
            
            # 保存骨干网络信息（如果存在）
            backbone_data = self._save_backbone_network_data()
            if backbone_data:
                data["backbone_network"] = backbone_data
            
            # 保存接口状态信息（如果存在）
            interface_data = self._save_interface_states()
            if interface_data:
                data["interface_states"] = interface_data
            
            # 保存调度器状态（如果存在）
            scheduler_data = self._save_scheduler_states()
            if scheduler_data:
                data["scheduler_states"] = scheduler_data
            
            # 保存交通管理器状态（如果存在）
            traffic_data = self._save_traffic_manager_states()
            if traffic_data:
                data["traffic_manager"] = traffic_data
            
            # 添加保存时间戳和版本信息
            import time
            data["save_metadata"] = {
                "timestamp": time.time(),
                "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "version": "v1.0_interface_system",
                "total_vehicles": len(self.vehicles),
                "total_loading_points": len(self.loading_points),
                "total_unloading_points": len(self.unloading_points)
            }
            
            # 保存到文件
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"环境保存成功: {filename}")
            print(f"- 车辆数量: {len(self.vehicles)}")
            print(f"- 装载点: {len(self.loading_points)}")
            print(f"- 卸载点: {len(self.unloading_points)}")
            
            if backbone_data:
                print(f"- 骨干路径: {backbone_data.get('total_paths', 0)} 条")
                print(f"- 骨干接口: {len(interface_data) if interface_data else 0} 个")
                
            return True
            
        except Exception as e:
            print(f"保存环境失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _save_backbone_network_data(self):
        """保存骨干网络数据"""
        # 检查是否有骨干网络引用
        backbone_network = getattr(self, 'backbone_network', None)
        
        if not backbone_network or not hasattr(backbone_network, 'backbone_paths'):
            return None
        
        try:
            backbone_data = {
                "total_paths": len(backbone_network.backbone_paths),
                "interface_spacing": getattr(backbone_network, 'interface_spacing', 10),
                "generation_stats": getattr(backbone_network, 'stats', {}),
                "paths": {}
            }
            
            # 保存每条骨干路径的基本信息（不保存完整路径点以节省空间）
            for path_id, path_data in backbone_network.backbone_paths.items():
                backbone_data["paths"][path_id] = {
                    "id": path_id,
                    "start_point_type": path_data["start_point"]["type"],
                    "start_point_id": path_data["start_point"]["id"],
                    "end_point_type": path_data["end_point"]["type"],
                    "end_point_id": path_data["end_point"]["id"],
                    "length": path_data.get("length", 0),
                    "quality": path_data.get("quality", 0),
                    "usage_count": path_data.get("usage_count", 0),
                    "path_points_count": len(path_data.get("path", [])),
                    "created_time": path_data.get("created_time", 0)
                }
            
            return backbone_data
            
        except Exception as e:
            print(f"保存骨干网络数据失败: {e}")
            return None

    def _save_interface_states(self):
        """保存接口状态数据"""
        # 检查是否有骨干网络和接口
        backbone_network = getattr(self, 'backbone_network', None)
        
        if (not backbone_network or 
            not hasattr(backbone_network, 'backbone_interfaces') or
            not backbone_network.backbone_interfaces):
            return None
        
        try:
            interface_states = {}
            
            for interface_id, interface in backbone_network.backbone_interfaces.items():
                interface_states[interface_id] = {
                    "interface_id": interface.interface_id,
                    "position": interface.position,
                    "direction": interface.direction,
                    "backbone_path_id": interface.backbone_path_id,
                    "path_index": interface.path_index,
                    "access_difficulty": interface.access_difficulty,
                    "usage_count": interface.usage_count,
                    "is_occupied": interface.is_occupied,
                    "occupied_by": interface.occupied_by,
                    "reservation_time": interface.reservation_time
                }
            
            return interface_states
            
        except Exception as e:
            print(f"保存接口状态失败: {e}")
            return None

    def _save_scheduler_states(self):
        """保存调度器状态数据"""
        # 检查是否有调度器引用
        scheduler = getattr(self, 'vehicle_scheduler', None)
        
        if not scheduler:
            return None
        
        try:
            scheduler_data = {
                "total_tasks": len(getattr(scheduler, 'tasks', {})),
                "completed_tasks": len([t for t in getattr(scheduler, 'tasks', {}).values() 
                                    if t.status == 'completed']),
                "active_tasks": len([t for t in getattr(scheduler, 'tasks', {}).values() 
                                if t.status in ['assigned', 'in_progress']]),
                "vehicle_statuses": {},
                "mission_templates": list(getattr(scheduler, 'mission_templates', {}).keys()),
                "stats": getattr(scheduler, 'stats', {})
            }
            
            # 保存车辆状态摘要
            if hasattr(scheduler, 'vehicle_statuses'):
                for vehicle_id, status in scheduler.vehicle_statuses.items():
                    scheduler_data["vehicle_statuses"][vehicle_id] = {
                        "status": status.get("status", "idle"),
                        "current_task": status.get("current_task"),
                        "completed_tasks": status.get("completed_tasks", 0),
                        "total_distance": status.get("total_distance", 0),
                        "utilization_rate": status.get("utilization_rate", 0),
                        "backbone_usage_count": status.get("backbone_usage_count", 0),
                        "direct_path_count": status.get("direct_path_count", 0),
                        "task_queue_length": len(status.get("task_queue", []))
                    }
            
            return scheduler_data
            
        except Exception as e:
            print(f"保存调度器状态失败: {e}")
            return None

    def _save_traffic_manager_states(self):
        """保存交通管理器状态数据"""
        # 检查是否有交通管理器引用
        traffic_manager = getattr(self, 'traffic_manager', None)
        
        if not traffic_manager:
            return None
        
        try:
            traffic_data = {
                "active_vehicles": len(getattr(traffic_manager, 'active_paths', {})),
                "total_reservations": len(getattr(traffic_manager, 'path_reservations', {})),
                "interface_reservations": len(getattr(traffic_manager, 'interface_reservations', {})),
                "safety_distance": getattr(traffic_manager, 'safety_distance', 8.0),
                "stats": getattr(traffic_manager, 'stats', {}),
                "ecbs_stats": {}
            }
            
            # 保存ECBS统计信息
            if hasattr(traffic_manager, 'ecbs_solver') and traffic_manager.ecbs_solver:
                traffic_data["ecbs_stats"] = getattr(traffic_manager.ecbs_solver, 'stats', {})
            
            return traffic_data
            
        except Exception as e:
            print(f"保存交通管理器状态失败: {e}")
            return None

    def set_backbone_network(self, backbone_network):
        """设置骨干网络引用"""
        self.backbone_network = backbone_network
    
    def set_vehicle_scheduler(self, scheduler):
        """设置车辆调度器引用"""
        self.vehicle_scheduler = scheduler
    
    def set_traffic_manager(self, traffic_manager):
        """设置交通管理器引用"""
        self.traffic_manager = traffic_manager
    
    def save_to_file(self, filename):
        """保存环境到文件"""
        try:
            # 构建基础数据字典
            data = {
                "width": self.width,
                "height": self.height,
                "resolution": self.grid_resolution,
                "obstacles": [{"x": x, "y": y} for x, y in self.obstacle_points],
                "loading_points": [
                    {"x": point[0], "y": point[1], "theta": point[2] if len(point) > 2 else 0}
                    for point in self.loading_points
                ],
                "unloading_points": [
                    {"x": point[0], "y": point[1], "theta": point[2] if len(point) > 2 else 0}
                    for point in self.unloading_points
                ],
                "parking_areas": [
                    {"x": point[0], "y": point[1], "theta": point[2] if len(point) > 2 else 0}
                    for point in self.parking_areas
                ],
                "vehicles": [
                    {
                        "id": vehicle_id,
                        "x": vehicle["position"][0],
                        "y": vehicle["position"][1],
                        "theta": vehicle["position"][2],
                        "type": vehicle["type"],
                        "max_load": vehicle["max_load"],
                        "status": vehicle.get("status", "idle"),
                        "load": vehicle.get("load", 0),
                        "completed_cycles": vehicle.get("completed_cycles", 0),
                        "path_structure": vehicle.get("path_structure", {}),
                        "path_index": vehicle.get("path_index", 0),
                        "progress": vehicle.get("progress", 0.0)
                    }
                    for vehicle_id, vehicle in self.vehicles.items()
                ],
                "current_time": self.current_time,
                "running": self.running,
                "paused": self.paused
            }
            
            # 保存组件状态
            if self.backbone_network:
                data["backbone_network"] = self.backbone_network.get_save_data()
            
            if self.vehicle_scheduler:
                data["scheduler_states"] = self.vehicle_scheduler.get_save_data()
            
            if self.traffic_manager:
                data["traffic_manager"] = self.traffic_manager.get_save_data()
            
            # 添加保存时间戳和版本信息
            import time
            data["save_metadata"] = {
                "timestamp": time.time(),
                "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "version": "v1.0_interface_system",
                "total_vehicles": len(self.vehicles)
            }
            
            # 写入文件
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"保存环境失败: {str(e)}")
            return False

    def restore_interface_states(self, backbone_network):
        """恢复接口状态到骨干网络"""
        if not self._saved_interface_states or not backbone_network:
            return False
        
        try:
            restored_count = 0
            
            for interface_id, state_data in self._saved_interface_states.items():
                if interface_id in backbone_network.backbone_interfaces:
                    interface = backbone_network.backbone_interfaces[interface_id]
                    
                    # 恢复状态
                    interface.usage_count = state_data.get("usage_count", 0)
                    interface.is_occupied = state_data.get("is_occupied", False)
                    interface.occupied_by = state_data.get("occupied_by")
                    interface.reservation_time = state_data.get("reservation_time")
                    
                    restored_count += 1
            
            print(f"恢复了 {restored_count} 个接口的状态")
            return True
            
        except Exception as e:
            print(f"恢复接口状态失败: {e}")
            return False

    def get_saved_states(self):
        """获取保存的状态数据，供其他组件使用"""
        return {
            'backbone_data': getattr(self, '_saved_backbone_data', None),
            'interface_states': getattr(self, '_saved_interface_states', None),
            'scheduler_states': getattr(self, '_saved_scheduler_states', None),
            'traffic_states': getattr(self, '_saved_traffic_states', None)
        }
    def _get_obstacle_list(self):
        """将点障碍物转换为矩形列表"""
        obstacles = []
        
        # 将散点障碍物转换为小矩形
        for x, y in self.obstacle_points:
            obstacles.append({
                "x": x,
                "y": y,
                "width": 1,
                "height": 1
            })
        
        return obstacles    
    def to_grid_coordinates(self, x, y):
        """将真实坐标转换为网格坐标
        
        Args:
            x, y (float): 真实坐标
            
        Returns:
            tuple: 网格坐标 (grid_x, grid_y)
        """
        grid_x = int(x / self.grid_resolution)
        grid_y = int(y / self.grid_resolution)
        
        # 确保在有效范围内
        grid_x = max(0, min(grid_x, self.width - 1))
        grid_y = max(0, min(grid_y, self.height - 1))
        
        return grid_x, grid_y
    
    def from_grid_coordinates(self, grid_x, grid_y):
        """将网格坐标转换为真实坐标
        
        Args:
            grid_x, grid_y (int): 网格坐标
            
        Returns:
            tuple: 真实坐标 (x, y)
        """
        x = (grid_x + 0.5) * self.grid_resolution
        y = (grid_y + 0.5) * self.grid_resolution
        
        return x, y
    
    def is_running(self):
        """检查模拟是否正在运行
        
        Returns:
            bool: 是否运行中
        """
        return self.running and not self.paused
    def load_from_file(self, filename):
        """从文件加载环境
        
        Args:
            filename (str): 环境文件路径
            
        Returns:
            bool: 加载是否成功
        """
        try:
            # 读取文件
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 重置环境
            self.__init__(
                width=data.get("width", 500),
                height=data.get("height", 500),
                grid_resolution=data.get("resolution", 1.0)
            )
            
            # 清空障碍点列表
            self.obstacle_points = []
            self.grid = np.zeros((self.width, self.height), dtype=np.uint8)
            
            # 加载障碍物
            for obstacle in data.get("obstacles", []):
                if isinstance(obstacle, dict):
                    x = obstacle.get("x", 0)
                    y = obstacle.get("y", 0)
                    if "width" in obstacle and "height" in obstacle:
                        # 矩形障碍物
                        self.add_obstacle(x, y, obstacle["width"], obstacle["height"])
                    else:
                        # 单点障碍物
                        self.add_obstacle_point(x, y)
            
            # 加载装载点
            self.loading_points = []
            for point in data.get("loading_points", []):
                if isinstance(point, dict):
                    x = point.get("x", 0)
                    y = point.get("y", 0)
                    theta = point.get("theta", 0.0)
                    self.add_loading_point((x, y, theta))
                elif isinstance(point, list) and len(point) >= 2:
                    row, col = point[0], point[1]
                    theta = 0.0 if len(point) <= 2 else point[2]
                    self.add_loading_point((col, row, theta))
            
            # 加载卸载点
            self.unloading_points = []
            for point in data.get("unloading_points", []):
                if isinstance(point, dict):
                    x = point.get("x", 0)
                    y = point.get("y", 0)
                    theta = point.get("theta", 0.0)
                    self.add_unloading_point((x, y, theta))
                elif isinstance(point, list) and len(point) >= 2:
                    row, col = point[0], point[1]
                    theta = 0.0 if len(point) <= 2 else point[2]
                    self.add_unloading_point((col, row, theta))
            
            # 加载停车区
            self.parking_areas = []
            for point in data.get("parking_areas", []):
                if isinstance(point, dict):
                    x = point.get("x", 0)
                    y = point.get("y", 0)
                    theta = point.get("theta", 0.0)
                    capacity = point.get("capacity", 5)
                    self.add_parking_area((x, y, theta), capacity)
                elif isinstance(point, list) and len(point) >= 2:
                    row, col = point[0], point[1]
                    theta = 0.0 if len(point) <= 2 else point[2]
                    capacity = 5 if len(point) <= 3 else point[3]
                    self.add_parking_area((col, row, theta), capacity)
            
            # 加载车辆信息
            self.vehicles = {}
            for vehicle in data.get("vehicles", []):
                if isinstance(vehicle, dict):
                    vehicle_id = str(vehicle.get("id", f"v_{len(self.vehicles) + 1}"))
                    x = vehicle.get("x", 0)
                    y = vehicle.get("y", 0)
                    theta = vehicle.get("theta", 0.0)
                    v_type = vehicle.get("type", "dump_truck")
                    max_load = vehicle.get("max_load", 100)
                    
                    # 添加车辆并设置其他属性
                    self.add_vehicle(vehicle_id, (x, y, theta), None, v_type, max_load)
                    if vehicle_id in self.vehicles:
                        self.vehicles[vehicle_id].update({
                            "status": vehicle.get("status", "idle"),
                            "load": vehicle.get("load", 0),
                            "completed_cycles": vehicle.get("completed_cycles", 0),
                            "path_structure": vehicle.get("path_structure", {}),
                            "path_index": vehicle.get("path_index", 0),
                            "progress": vehicle.get("progress", 0.0)
                        })
            
            # 加载组件状态
            self._saved_backbone_data = data.get("backbone_network")
            self._saved_scheduler_states = data.get("scheduler_states")
            self._saved_traffic_states = data.get("traffic_manager")
            
            # 设置环境状态
            self.current_time = data.get("current_time", 0.0)
            self.running = data.get("running", False)
            self.paused = data.get("paused", False)
            
            return True
            
        except Exception as e:
            print(f"加载环境失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False