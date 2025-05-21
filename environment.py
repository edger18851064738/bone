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
        """保存环境到文件
        
        Args:
            filename (str): 文件路径
            
        Returns:
            bool: 保存是否成功
        """
        try:
            # 构建数据字典
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
                        "max_load": vehicle["max_load"]
                    }
                    for vehicle_id, vehicle in self.vehicles.items()
                ]
            }
            
            # 保存到文件
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
                
            return True
            
        except Exception as e:
            print(f"保存环境失败: {str(e)}")
            return False
    
    def load_from_file(self, filename):
        """从文件加载环境
        
        Args:
            filename (str): 文件路径
            
        Returns:
            bool: 加载是否成功
        """
        try:
            # 读取文件
            with open(filename, 'r') as f:
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
                if "width" in obstacle and "height" in obstacle:
                    # 旧格式：矩形障碍物
                    self.add_obstacle(
                        obstacle["x"],
                        obstacle["y"],
                        obstacle["width"],
                        obstacle["height"]
                    )
                else:
                    # 新格式：散点障碍物
                    self.add_obstacle_point(obstacle["x"], obstacle["y"])
            
            # 加载装载点
            for point in data.get("loading_points", []):
                self.add_loading_point(
                    (point["x"], point["y"], point.get("theta", 0))
                )
            
            # 加载卸载点
            for point in data.get("unloading_points", []):
                self.add_unloading_point(
                    (point["x"], point["y"], point.get("theta", 0))
                )
            
            # 加载停车区
            for area in data.get("parking_areas", []):
                self.add_parking_area(
                    (area["x"], area["y"], area.get("theta", 0)),

                    area.get("capacity", 5)
                )
            
            # 加载车辆
            for vehicle in data.get("vehicles", []):
                self.add_vehicle(
                    vehicle["id"],
                    (vehicle["x"], vehicle["y"], vehicle.get("theta", 0)),
                    None,
                    vehicle.get("type", "dump_truck"),
                    vehicle.get("max_load", 100)
                )
            
            # 通知环境变化
            self._notify_change()
            
            return True
            
        except Exception as e:
            print(f"加载环境失败: {str(e)}")
            return False
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