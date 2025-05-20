import numpy as np
import math
from collections import defaultdict
from PyQt5.QtCore import (Qt, QTimer, pyqtSignal, QObject, QThread, QRectF, 
                         QPointF, QLineF, QSizeF, QPropertyAnimation, QVariantAnimation,QSize)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QComboBox, QCheckBox, QFileDialog, QSlider,
                            QGroupBox, QTabWidget, QSpinBox, QDoubleSpinBox, QProgressBar,
                            QMessageBox, QTextEdit, QSplitter, QAction, QStatusBar, QToolBar,
                            QMenu, QDockWidget, QGraphicsScene, QGraphicsView, QGraphicsItem,
                            QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsPolygonItem, 
                            QGraphicsPathItem, QGraphicsLineItem, QGraphicsTextItem, QGraphicsItemGroup,
                            QGridLayout, QFrame, QStyleFactory, QScrollArea)
from PyQt5.QtGui import (QIcon, QFont, QPixmap, QPen, QBrush, QColor, QPainter, QPainterPath,
                        QTransform, QPolygonF, QLinearGradient, QRadialGradient, QPalette)

class OpenPitMineEnv:
    """露天矿环境模型，提供地图数据和资源信息"""
    
    def __init__(self, width=500, height=500, grid_resolution=1.0):
        """初始化环境
        
        Args:
            width (int): 环境宽度
            height (int): 环境高度
            grid_resolution (float): 网格分辨率
        """
        self.width = width
        self.height = height
        self.grid_resolution = grid_resolution
        
        # 创建网格地图 (0-可通行 1-障碍物)
        self.grid = np.zeros((width, height), dtype=np.uint8)
        
        # 关键点位置
        self.loading_points = []     # 装载点
        self.unloading_points = []   # 卸载点
        self.parking_areas = []      # 停车区
        
        # 车辆信息
        self.vehicles = {}           # 车辆信息字典 {vehicle_id: vehicle_data}
        
        # 地形信息
        self.elevation_map = None    # 高程图，可选
        self.slope_map = None        # 坡度图，可选
        
        # 环境计时器
        self.current_time = 0.0
        self.time_step = 0.5  # 默认时间步长
        
        # 规划器引用
        self.planner = None
        
        # 环境状态
        self.running = False
        self.paused = False
        
        # 可视化相关属性
        self.scene_rect = QRectF(0, 0, width, height)
        self.changed_callback = None  # 环境变化时的回调函数
        
    def add_obstacle(self, x, y, width, height):
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
        
        # 设置障碍物区域
        self.grid[x1:x2, y1:y2] = 1
        
        # 通知环境变化
        self._notify_change()
        
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
            
        # 创建装载点
        loading_point = {
            'position': (x, y, theta),
            'capacity': capacity,
            'current_usage': 0,
            'waiting_vehicles': []
        }
        
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
            
        # 创建卸载点
        unloading_point = {
            'position': (x, y, theta),
            'capacity': capacity,
            'current_usage': 0,
            'waiting_vehicles': []
        }
        
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
            
        # 创建停车区
        parking_area = {
            'position': (x, y, theta),
            'capacity': capacity,
            'current_usage': 0,
            'parked_vehicles': []
        }
        
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
    
    def get_vehicle_at_position(self, position, radius=5.0):
        """获取指定位置附近的车辆
        
        Args:
            position (tuple): 位置 (x, y) 或 (x, y, theta)
            radius (float): 搜索半径
            
        Returns:
            list: 车辆ID列表
        """
        x, y = position[0], position[1]
        
        vehicles_found = []
        
        for vehicle_id, vehicle in self.vehicles.items():
            vx, vy = vehicle['position'][0], vehicle['position'][1]
            dist = math.sqrt((x - vx)**2 + (y - vy)**2)
            
            if dist <= radius:
                vehicles_found.append(vehicle_id)
                
        return vehicles_found
    
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
        
        # 在这里添加更新逻辑，例如车辆移动、任务处理等
        # 这部分通常由车辆调度器等其他组件处理
        
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
        
    def update_terrain(self, terrain_file):
        """更新地形数据
        
        Args:
            terrain_file (str): 地形数据文件路径
            
        Returns:
            bool: 更新是否成功
        """
        try:
            # 读取地形文件并更新高程图
            # 具体实现取决于文件格式，这里只是示例
            data = np.loadtxt(terrain_file)
            
            # 确保尺寸匹配
            if data.shape != (self.width, self.height):
                # 调整尺寸
                from scipy.ndimage import zoom
                zoom_factor = (self.width / data.shape[0], self.height / data.shape[1])
                data = zoom(data, zoom_factor)
            
            # 更新高程图
            self.elevation_map = data
            
            # 计算坡度图
            gradient_x, gradient_y = np.gradient(self.elevation_map)
            self.slope_map = np.sqrt(gradient_x**2 + gradient_y**2)
            
            # 通知环境变化
            self._notify_change()
            
            return True
        except Exception as e:
            print(f"更新地形失败: {str(e)}")
            return False
    
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
            import json
            
            # 构建数据字典
            data = {
                "width": self.width,
                "height": self.height,
                "resolution": self.grid_resolution,
                "obstacles": self._get_obstacle_list(),
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
    
    def _get_obstacle_list(self):
        """将网格障碍物转换为矩形列表
        
        Returns:
            list: 障碍物矩形列表
        """
        obstacles = []
        
        # 简单实现：用连通区域分析
        visited = np.zeros_like(self.grid, dtype=bool)
        
        for x in range(self.width):
            for y in range(self.height):
                if self.grid[x, y] == 1 and not visited[x, y]:
                    # 找到一个新障碍物，用BFS寻找连通区域
                    min_x, min_y = x, y
                    max_x, max_y = x, y
                    
                    # BFS
                    queue = [(x, y)]
                    visited[x, y] = True
                    
                    while queue:
                        cx, cy = queue.pop(0)
                        
                        # 更新边界
                        min_x = min(min_x, cx)
                        min_y = min(min_y, cy)
                        max_x = max(max_x, cx)
                        max_y = max(max_y, cy)
                        
                        # 检查四邻域
                        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                            nx, ny = cx + dx, cy + dy
                            if (0 <= nx < self.width and 0 <= ny < self.height and
                                self.grid[nx, ny] == 1 and not visited[nx, ny]):
                                visited[nx, ny] = True
                                queue.append((nx, ny))
                    
                    # 添加矩形障碍物
                    obstacles.append({
                        "x": min_x,
                        "y": min_y,
                        "width": max_x - min_x + 1,
                        "height": max_y - min_y + 1
                    })
        
        return obstacles
    
    def load_from_file(self, filename):
        """从文件加载环境
        
        Args:
            filename (str): 文件路径
            
        Returns:
            bool: 加载是否成功
        """
        try:
            import json
            
            # 读取文件
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # 重置环境
            self.__init__(
                width=data.get("width", 500),
                height=data.get("height", 500),
                grid_resolution=data.get("resolution", 1.0)
            )
            
            # 加载障碍物
            for obstacle in data.get("obstacles", []):
                self.add_obstacle(
                    obstacle["x"],
                    obstacle["y"],
                    obstacle["width"],
                    obstacle["height"]
                )
            
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
    
    def get_qt_coordinates(self, pos):
        """获取PyQt坐标系中的坐标（可能需要坐标变换）
        
        Args:
            pos (tuple): 环境中的坐标 (x, y) 或 (x, y, theta)
            
        Returns:
            QPointF: PyQt坐标系中的点
        """
        # 在这个简单实现中，我们直接使用相同的坐标系
        # 如果GUI使用不同的坐标系，您可能需要进行变换
        return QPointF(pos[0], pos[1])
    
    def is_running(self):
        """检查模拟是否正在运行
        
        Returns:
            bool: 是否运行中
        """
        return self.running and not self.paused

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QPen, QBrush, QColor, QPainterPath
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsPathItem, QGraphicsTextItem,QGraphicsItemGroup

class EnvironmentGraphicsScene(QGraphicsScene):
    """用于可视化环境的图形场景"""
    
    def __init__(self, env=None, parent=None):
        super().__init__(parent)
        self.env = env
        
        # 图形项字典
        self.obstacle_items = {}     # 障碍物图形项
        self.loading_items = {}      # 装载点图形项
        self.unloading_items = {}    # 卸载点图形项
        self.parking_items = {}      # 停车区图形项
        self.vehicle_items = {}      # 车辆图形项
        self.grid_items = []         # 网格线图形项
        
        # 配色方案
        self.colors = {
            'background': QColor(255, 255, 255),
            'grid': QColor(220, 220, 220),
            'obstacle': QColor(80, 80, 90),
            'loading': QColor(0, 180, 0),  # 绿色
            'unloading': QColor(180, 0, 0),  # 红色
            'parking': QColor(0, 0, 180),  # 蓝色
            'vehicle_idle': QColor(128, 128, 128),  # 灰色
            'vehicle_moving': QColor(0, 123, 255),  # 蓝色
            'vehicle_loading': QColor(40, 167, 69),  # 绿色
            'vehicle_unloading': QColor(220, 53, 69)  # 红色
        }
        
        # 初始化场景
        if env:
            self.set_environment(env)
    
    def set_environment(self, env):
        """设置环境对象，并初始化场景"""
        self.env = env
        
        # 设置回调函数，以便在环境变化时更新场景
        env.set_changed_callback(self.update_scene)
        
        # 初始化场景
        self.init_scene()
    
    def init_scene(self):
        """初始化场景，绘制所有元素"""
        # 清除现有项
        self.clear()
        self.obstacle_items.clear()
        self.loading_items.clear()
        self.unloading_items.clear()
        self.parking_items.clear()
        self.vehicle_items.clear()
        self.grid_items.clear()
        
        if not self.env:
            return
            
        # 设置场景大小
        self.setSceneRect(0, 0, self.env.width, self.env.height)
        
        # 绘制背景
        self.setBackgroundBrush(QBrush(self.colors['background']))
        
        # 绘制网格
        self.draw_grid()
        
        # 绘制障碍物
        self.draw_obstacles()
        
        # 绘制装载点
        self.draw_loading_points()
        
        # 绘制卸载点
        self.draw_unloading_points()
        
        # 绘制停车区
        self.draw_parking_areas()
        
        # 绘制车辆
        self.draw_vehicles()
    
    def draw_grid(self):
        """绘制网格线"""
        # 设置网格参数
        grid_size = 10  # 网格大小
        
        # 网格线样式
        grid_pen = QPen(self.colors['grid'])
        grid_pen.setWidth(0)  # 细线
        
        # 主网格线样式（每5条网格线）
        major_grid_pen = QPen(self.colors['grid'].darker(120))
        major_grid_pen.setWidth(0)
        
        # 清除现有网格线
        for item in self.grid_items:
            self.removeItem(item)
        self.grid_items.clear()
        
        # 绘制垂直线
        for x in range(0, self.env.width + 1, grid_size):
            if x % (grid_size * 5) == 0:
                # 主网格线
                line = self.addLine(x, 0, x, self.env.height, major_grid_pen)
            else:
                # 次网格线
                line = self.addLine(x, 0, x, self.env.height, grid_pen)
            
            line.setZValue(-100)  # 确保网格在最底层
            self.grid_items.append(line)
        
        # 绘制水平线
        for y in range(0, self.env.height + 1, grid_size):
            if y % (grid_size * 5) == 0:
                # 主网格线
                line = self.addLine(0, y, self.env.width, y, major_grid_pen)
            else:
                # 次网格线
                line = self.addLine(0, y, self.env.width, y, grid_pen)
            
            line.setZValue(-100)  # 确保网格在最底层
            self.grid_items.append(line)
        
        # 添加坐标标签
        label_interval = 50  # 每50个单位显示一个标签
        
        for x in range(0, self.env.width + 1, label_interval):
            label = self.addText(str(x))
            label.setPos(x, 5)
            label.setDefaultTextColor(self.colors['grid'].darker(200))
            label.setZValue(-90)
            self.grid_items.append(label)
        
        for y in range(0, self.env.height + 1, label_interval):
            label = self.addText(str(y))
            label.setPos(5, y)
            label.setDefaultTextColor(self.colors['grid'].darker(200))
            label.setZValue(-90)
            self.grid_items.append(label)
    
    def draw_obstacles(self):
        """绘制障碍物"""
        # 障碍物样式
        obstacle_brush = QBrush(self.colors['obstacle'])
        obstacle_pen = QPen(self.colors['obstacle'].darker(120))
        obstacle_pen.setWidth(0)
        
        # 清除现有障碍物图形项
        for item in self.obstacle_items.values():
            self.removeItem(item)
        self.obstacle_items.clear()
        
        # 遍历环境网格，标记障碍物
        # 为了效率，先找到连续的障碍物区域
        obstacles = self.env._get_obstacle_list()
        
        for i, obstacle in enumerate(obstacles):
            x, y = obstacle['x'], obstacle['y']
            width, height = obstacle['width'], obstacle['height']
            
            # 创建矩形项
            rect = self.addRect(x, y, width, height, obstacle_pen, obstacle_brush)
            rect.setZValue(-50)  # 确保在网格之上
            
            # 存储到字典
            self.obstacle_items[i] = rect
    
    def draw_loading_points(self):
        """绘制装载点"""
        # 装载点样式
        loading_brush = QBrush(self.colors['loading'])
        loading_pen = QPen(self.colors['loading'].darker(120))
        loading_pen.setWidth(1)
        
        # 清除现有装载点图形项
        for item in self.loading_items.values():
            if isinstance(item, dict):
                for subitem in item.values():
                    self.removeItem(subitem)
            else:
                self.removeItem(item)
        self.loading_items.clear()
        
        # 绘制装载点
        for i, point in enumerate(self.env.loading_points):
            x, y = point[0], point[1]
            
            # 创建装载点图形
            radius = 4.0
            loading_item = self.addEllipse(
                x - radius, y - radius, radius * 2, radius * 2,
                loading_pen, loading_brush
            )
            loading_item.setZValue(5)
            
            # 添加标签
            text = self.addText(f"装载点{i+1}")
            text.setPos(x - 20, y - 15)
            text.setDefaultTextColor(self.colors['loading'].darker(120))
            text.setZValue(5)
            
            # 存储到字典
            self.loading_items[i] = {'ellipse': loading_item, 'label': text}
    
    def draw_unloading_points(self):
        """绘制卸载点"""
        # 卸载点样式
        unloading_brush = QBrush(self.colors['unloading'])
        unloading_pen = QPen(self.colors['unloading'].darker(120))
        unloading_pen.setWidth(1)
        
        # 清除现有卸载点图形项
        for item in self.unloading_items.values():
            if isinstance(item, dict):
                for subitem in item.values():
                    self.removeItem(subitem)
            else:
                self.removeItem(item)
        self.unloading_items.clear()
        
        # 绘制卸载点
        for i, point in enumerate(self.env.unloading_points):
            x, y = point[0], point[1]
            
            # 创建卸载点图形
            radius = 4.0
            unloading_item = self.addEllipse(
                x - radius, y - radius, radius * 2, radius * 2,
                unloading_pen, unloading_brush
            )
            unloading_item.setZValue(5)
            
            # 添加标签
            text = self.addText(f"卸载点{i+1}")
            text.setPos(x - 20, y - 15)
            text.setDefaultTextColor(self.colors['unloading'].darker(120))
            text.setZValue(5)
            
            # 存储到字典
            self.unloading_items[i] = {'ellipse': unloading_item, 'label': text}
    
    def draw_parking_areas(self):
        """绘制停车区"""
        # 停车区样式
        parking_brush = QBrush(self.colors['parking'])
        parking_pen = QPen(self.colors['parking'].darker(120))
        parking_pen.setWidth(1)
        
        # 清除现有停车区图形项
        for item in self.parking_items.values():
            if isinstance(item, dict):
                for subitem in item.values():
                    self.removeItem(subitem)
            else:
                self.removeItem(item)
        self.parking_items.clear()
        
        # 绘制停车区
        for i, point in enumerate(self.env.parking_areas):
            x, y = point[0], point[1]
            
            # 创建停车区图形
            radius = 4.0
            parking_item = self.addEllipse(
                x - radius, y - radius, radius * 2, radius * 2,
                parking_pen, parking_brush
            )
            parking_item.setZValue(5)
            
            # 添加标签
            text = self.addText(f"停车区{i+1}")
            text.setPos(x - 20, y - 15)
            text.setDefaultTextColor(self.colors['parking'].darker(120))
            text.setZValue(5)
            
            # 存储到字典
            self.parking_items[i] = {'ellipse': parking_item, 'label': text}
    
    def draw_vehicles(self):
        """绘制车辆"""
        # 清除现有车辆图形项
        for item in self.vehicle_items.values():
            if isinstance(item, dict):
                for subitem in item.values():
                    self.removeItem(subitem)
            else:
                self.removeItem(item)
        self.vehicle_items.clear()
        
        # 为每个车辆创建图形项
        for vehicle_id, vehicle_data in self.env.vehicles.items():
            # 创建并添加车辆图形项
            vehicle_item = VehicleGraphicsItem(vehicle_id, vehicle_data)
            self.addItem(vehicle_item)
            
            # 存储到字典
            self.vehicle_items[vehicle_id] = vehicle_item
    
    def update_scene(self, env=None):
        """更新场景，响应环境变化"""
        if env:
            self.env = env
            
        # 更新车辆位置和状态
        self.update_vehicles()
        
        # 可以根据需要更新其他元素
    
    def update_vehicles(self):
        """更新车辆位置和状态"""
        if not self.env:
            return
            
        # 移除不再存在的车辆
        vehicle_ids = set(self.env.vehicles.keys())
        item_ids = set(self.vehicle_items.keys())
        
        for vehicle_id in item_ids - vehicle_ids:
            item = self.vehicle_items.pop(vehicle_id)
            self.removeItem(item)
        
        # 更新或添加车辆
        for vehicle_id, vehicle_data in self.env.vehicles.items():
            if vehicle_id in self.vehicle_items:
                # 更新现有车辆
                self.vehicle_items[vehicle_id].update_data(vehicle_data)
            else:
                # 添加新车辆
                vehicle_item = VehicleGraphicsItem(vehicle_id, vehicle_data)
                self.addItem(vehicle_item)
                self.vehicle_items[vehicle_id] = vehicle_item

class VehicleGraphicsItem(QGraphicsItemGroup):
    """车辆图形项 - 改进版"""
    def __init__(self, vehicle_id, vehicle_data, parent=None):
        super().__init__(parent)
        self.vehicle_id = vehicle_id
        self.vehicle_data = vehicle_data
        self.position = vehicle_data['position']
        self.vehicle_length = 4.0  # 车辆长度
        self.vehicle_width = 2.0   # 车辆宽度
        
        # 创建车辆图形
        self.vehicle_body = QGraphicsPolygonItem(self)
        self.vehicle_label = QGraphicsTextItem(str(vehicle_id), self)
        self.status_label = QGraphicsTextItem("", self)
        
        # 设置车辆颜色 - 使用新配色
        status = vehicle_data.get('status', 'idle')
        color = VEHICLE_COLORS.get(status, VEHICLE_COLORS['idle'])
        
        # 设置样式
        self.vehicle_body.setBrush(QBrush(color))
        self.vehicle_body.setPen(QPen(Qt.black, 0.5))
        
        # 改进标签样式
        self.vehicle_label.setDefaultTextColor(Qt.white)
        font = QFont("Arial", 3, QFont.Bold)  # 增大字体
        self.vehicle_label.setFont(font)
        
        # 状态标签
        status_text = {
            'idle': '空闲',
            'loading': '装载中',
            'unloading': '卸载中',
            'moving': '移动中'
        }.get(status, '')
        
        self.status_label.setPlainText(status_text)
        self.status_label.setDefaultTextColor(Qt.black)
        self.status_label.setFont(QFont("Arial", 2))
        
        # 设置Z值，确保车辆显示在上层
        self.setZValue(10)
        
        # 更新车辆位置和朝向
        self.update_position()
    
    def update_position(self):
        """更新车辆位置和朝向"""
        # 确保position是一个有效的元组，包含三个值：x, y, theta
        if not self.position or len(self.position) < 3:
            return
            
        x, y, theta = self.position
        
        # 创建车辆多边形
        polygon = QPolygonF()
        
        # 计算车辆四个角点的相对坐标 - 修改为默认沿Y轴方向
        half_length = self.vehicle_length / 2
        half_width = self.vehicle_width / 2
        corners_relative = [
            QPointF(half_length, half_width),     # 前右
            QPointF(half_length, -half_width),    # 前左
            QPointF(-half_length, -half_width),   # 后左
            QPointF(-half_length, half_width)     # 后右
        ]
        
        # 创建变换矩阵
        transform = QTransform()
        # 由于默认方向改为Y轴，这里不需要额外加90度
        transform.rotate(theta * 180 / math.pi)
        
        # 应用旋转，然后添加到多边形
        for corner in corners_relative:
            rotated_corner = transform.map(corner)
            polygon.append(QPointF(x + rotated_corner.x(), y + rotated_corner.y()))
        
        # 设置多边形
        self.vehicle_body.setPolygon(polygon)
        
        # 更新标签位置 - 调整位置使其更合理
        self.vehicle_label.setPos(x - 1.5, y - 1.5)
        
        # 状态标签显示在车辆下方
        self.status_label.setPos(x - 5, y + 3.5)

    def update_data(self, vehicle_data):
        """更新车辆数据"""
        self.vehicle_data = vehicle_data
        
        # 保存旧位置用于调试
        old_position = self.position
        
        # 更新位置
        self.position = vehicle_data['position']
        
        # 仅在位置发生变化时打印调试信息
        if old_position != self.position:
            # 可以在这里打印位置变化调试信息
            pass
        
        # 更新状态标签
        status = vehicle_data.get('status', 'idle')
        status_text = {
            'idle': '空闲',
            'loading': '装载中',
            'unloading': '卸载中',
            'moving': '移动中'
        }.get(status, '')
        
        # 根据车辆状态更新颜色 - 使用配色
        color = VEHICLE_COLORS.get(status, VEHICLE_COLORS['idle'])
        
        self.vehicle_body.setBrush(QBrush(color))
        self.status_label.setPlainText(status_text)
        
        # 更新车辆位置
        self.update_position()

import sys
import random
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel

class MineSimulationDemo(QMainWindow):
    """露天矿模拟演示应用"""
    
    def __init__(self):
        super().__init__()
        
        # 创建环境
        self.env = OpenPitMineEnv(width=800, height=600)
        
        # 创建图形场景
        self.scene = EnvironmentGraphicsScene(self.env)
        
        # 创建图形视图
        self.view = QGraphicsView()
        self.view.setScene(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        
        # 创建布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.addWidget(self.view)
        
        # 控制按钮
        control_layout = QHBoxLayout()
        
        self.init_button = QPushButton("初始化环境")
        self.init_button.clicked.connect(self.init_environment)
        control_layout.addWidget(self.init_button)
        
        self.start_button = QPushButton("开始模拟")
        self.start_button.clicked.connect(self.start_simulation)
        control_layout.addWidget(self.start_button)
        
        self.pause_button = QPushButton("暂停")
        self.pause_button.clicked.connect(self.pause_simulation)
        control_layout.addWidget(self.pause_button)
        
        self.reset_button = QPushButton("重置")
        self.reset_button.clicked.connect(self.reset_simulation)
        control_layout.addWidget(self.reset_button)
        
        main_layout.addLayout(control_layout)
        
        # 设置窗口
        self.setWindowTitle("露天矿多车调度系统")
        self.resize(1000, 800)
        
        # 创建更新定时器
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_simulation)
        self.update_timer.setInterval(50)  # 20 FPS
        
        # 初始化示例环境
        self.init_environment()
    
    def init_environment(self):
        """初始化环境"""
        # 重置环境
        self.env.reset()
        
        # 添加一些障碍物
        self.env.add_obstacle(100, 100, 50, 50)
        self.env.add_obstacle(300, 200, 80, 30)
        self.env.add_obstacle(500, 300, 60, 60)
        
        # 添加装载点
        self.env.add_loading_point((50, 50, 0), capacity=2)
        self.env.add_loading_point((450, 450, 0), capacity=2)
        
        # 添加卸载点
        self.env.add_unloading_point((450, 50, 0), capacity=3)
        self.env.add_unloading_point((50, 450, 0), capacity=3)
        
        # 添加停车区
        self.env.add_parking_area((200, 300, 0), capacity=5)
        
        # 添加车辆
        for i in range(3):
            self.env.add_vehicle(
                i+1,
                (250 + i*20, 250, 0),
                None,
                "dump_truck",
                100
            )
        
        # 更新场景
        self.scene.init_scene()
        
        # 调整视图以显示整个场景
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
    
    def start_simulation(self):
        """开始模拟"""
        self.env.start()
        self.update_timer.start()
    
    def pause_simulation(self):
        """暂停模拟"""
        self.env.pause()
    
    def reset_simulation(self):
        """重置模拟"""
        self.update_timer.stop()
        self.env.reset()
        self.scene.init_scene()
    
    def update_simulation(self):
        """更新模拟"""
        # 更新环境
        self.env.update(0.1)  # 时间步长0.1
        
        # 模拟车辆移动（在实际应用中，这应该由车辆调度器处理）
        self.simulate_vehicle_movement()
    
    def simulate_vehicle_movement(self):
        """模拟车辆移动（示例）"""
        for vehicle_id, vehicle in self.env.vehicles.items():
            # 随机变化状态
            if random.random() < 0.01:  # 1%的概率改变状态
                states = ['idle', 'moving', 'loading', 'unloading']
                vehicle['status'] = random.choice(states)
            
            # 如果状态是移动中，更新位置
            if vehicle['status'] == 'moving':
                x, y, theta = vehicle['position']
                
                # 简单的随机移动
                dx = random.uniform(-2, 2)
                dy = random.uniform(-2, 2)
                
                # 更新位置，确保不超出范围
                new_x = max(0, min(self.env.width, x + dx))
                new_y = max(0, min(self.env.height, y + dy))
                
                # 计算新朝向
                if abs(dx) > 0.001 or abs(dy) > 0.001:
                    new_theta = math.atan2(dy, dx)
                else:
                    new_theta = theta
                
                # 避免碰撞
                new_position = (new_x, new_y, new_theta)
                if not self.env.check_collision(new_position):
                    vehicle['position'] = new_position
    
    def resizeEvent(self, event):
        """窗口大小改变事件"""
        super().resizeEvent(event)
        # 调整视图以显示整个场景
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
# ======== 样式和主题配置 ========
# 定义应用全局样式
GLOBAL_STYLESHEET = """
QMainWindow {
    background-color: #f8f9fa;
}

QTabWidget::pane {
    border: 1px solid #d3d3d3;
    background-color: white;
    border-radius: 4px;
}

QTabBar::tab {
    background-color: #f1f1f1;
    border: 1px solid #d3d3d3;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    padding: 8px 16px;
    font-weight: bold;
    min-width: 120px;
}

QTabBar::tab:selected {
    background-color: white;
    border-bottom: 1px solid white;
}

QGroupBox {
    font-weight: bold;
    border: 1px solid #d3d3d3;
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 20px;
    background-color: #ffffff;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 5px 8px;
    background-color: #f8f9fa;
    border-radius: 2px;
}

QPushButton {
    background-color: #007bff;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 4px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #0069d9;
}

QPushButton:pressed {
    background-color: #005cbf;
}

QPushButton:disabled {
    background-color: #6c757d;
    color: #f8f9fa;
}

QLabel {
    font-size: 12px;
}

QComboBox {
    border: 1px solid #d3d3d3;
    border-radius: 4px;
    padding: 5px;
    min-width: 150px;
    background-color: white;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox QAbstractItemView {
    border: 1px solid #d3d3d3;
    border-radius: 4px;
    selection-background-color: #007bff;
    selection-color: white;
}

QSpinBox, QDoubleSpinBox {
    border: 1px solid #d3d3d3;
    border-radius: 4px;
    padding: 5px;
    min-width: 100px;
}

QTextEdit {
    border: 1px solid #d3d3d3;
    border-radius: 4px;
    background-color: white;
}

QProgressBar {
    border: 1px solid #d3d3d3;
    border-radius: 4px;
    background-color: #f8f9fa;
    text-align: center;
    font-weight: bold;
    color: black;
    height: 20px;
}

QProgressBar::chunk {
    background-color: #28a745;
    border-radius: 3px;
}

QSlider::groove:horizontal {
    border: 1px solid #d3d3d3;
    height: 8px;
    background: #f8f9fa;
    border-radius: 4px;
}

QSlider::handle:horizontal {
    background: #007bff;
    border: 1px solid #007bff;
    width: 18px;
    height: 18px;
    border-radius: 9px;
    margin: -5px 0;
}

QSlider::handle:horizontal:hover {
    background: #0069d9;
}

QStatusBar {
    background-color: #f8f9fa;
    border-top: 1px solid #d3d3d3;
    color: #495057;
}

QToolBar {
    background-color: #f8f9fa;
    border-bottom: 1px solid #d3d3d3;
    spacing: 6px;
}

QMenuBar {
    background-color: #f8f9fa;
}

QMenuBar::item {
    padding: 6px 16px;
}

QMenuBar::item:selected {
    background-color: #007bff;
    color: white;
}

QMenu {
    background-color: white;
    border: 1px solid #d3d3d3;
}

QMenu::item {
    padding: 6px 24px 6px 24px;
}

QMenu::item:selected {
    background-color: #007bff;
    color: white;
}

QCheckBox {
    padding: 5px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
}

QFrame[frameShape="4"] { /* QFrame::HLine */
    color: #d3d3d3;
    height: 1px;
}

QDockWidget {
    titlebar-close-icon: url(close.png);
    titlebar-normal-icon: url(undock.png);
}

QDockWidget::title {
    text-align: center;
    background-color: #f8f9fa;
    padding: 6px;
}

QGraphicsView {
    border: 1px solid #d3d3d3;
    background-color: white;
}
"""
# 车辆显示颜色配置
VEHICLE_COLORS = {
    'idle': QColor(128, 128, 128),      # 灰色
    'loading': QColor(40, 167, 69),     # 绿色
    'unloading': QColor(220, 53, 69),   # 红色
    'moving': QColor(0, 123, 255)       # 蓝色
}
# 应用程序入口
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(GLOBAL_STYLESHEET)
    
    window = MineSimulationDemo()
    window.show()
    
    sys.exit(app.exec_())