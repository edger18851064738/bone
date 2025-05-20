import math
import numpy as np
from typing import Tuple, Dict, List, Optional

class BicycleModel:
    """自行车运动学模型 - 优化版本"""
    
    def __init__(self, vehicle_params):
        """
        初始化自行车模型
        
        Args:
            vehicle_params: 车辆参数，包含:
                - wheel_base: 轴距
                - turning_radius: 最小转弯半径
                - length: 车辆长度
                - width: 车辆宽度
        """
        self.wheel_base = vehicle_params.get('wheel_base', 3.0)
        self.turning_radius = vehicle_params.get('turning_radius', 5.0)
        self.vehicle_length = vehicle_params.get('length', 5.0)
        self.vehicle_width = vehicle_params.get('width', 2.5)
        
        # 计算一些常量
        self._min_turn_radius_inv = 1.0 / max(0.1, self.turning_radius)
        
        # 缓存最近使用的运动结果
        self._motion_cache = {}
        self._cache_size = 1000  # 缓存大小限制
        self._cache_hits = 0
        self._cache_misses = 0
    
    def bicycle_model(self, x, y, theta, steering_angle, distance):
        """
        使用自行车运动模型计算下一个状态，优化版本
        
        Args:
            x, y, theta: 当前位置和朝向
            steering_angle: 转向角（弧度）
            distance: 移动距离
            
        Returns:
            tuple: 新的位置和朝向 (x_new, y_new, theta_new)
        """
        # 生成缓存键 - 使用截断的坐标减少内存占用
        cache_key = (round(x, 2), round(y, 2), round(theta, 2), 
                    round(steering_angle, 2), round(distance, 2))
        
        # 检查缓存
        if cache_key in self._motion_cache:
            self._cache_hits += 1
            return self._motion_cache[cache_key]
        
        self._cache_misses += 1
        
        # 如果转向角接近于0，则直线运动 (避免数值不稳定)
        if abs(steering_angle) < 1e-3:
            next_x = x + distance * math.cos(theta)
            next_y = y + distance * math.sin(theta)
            next_theta = theta
        else:
            # 计算转弯半径 (限制不小于最小转弯半径)
            effective_steering = max(min(steering_angle, math.atan(self.wheel_base * self._min_turn_radius_inv)), 
                                    -math.atan(self.wheel_base * self._min_turn_radius_inv))
            
            turning_radius = self.wheel_base / math.tan(abs(effective_steering))
            
            # 旋转中心
            if effective_steering > 0:  # 左转
                cx = x - turning_radius * math.sin(theta)
                cy = y + turning_radius * math.cos(theta)
                turn_direction = 1
            else:  # 右转
                cx = x + turning_radius * math.sin(theta)
                cy = y - turning_radius * math.cos(theta)
                turn_direction = -1
                
            # 计算旋转角度
            beta = distance / turning_radius
            # 计算新的朝向
            next_theta = (theta + turn_direction * beta) % (2 * math.pi)
            
            # 计算新的位置 (优化三角函数计算)
            next_x = cx + turning_radius * math.sin(next_theta)
            next_y = cy - turning_radius * math.cos(next_theta)
        
        # 归一化角度到[-π, π]
        next_theta = self.normalize_angle(next_theta)
        
        # 存储到缓存
        result = (next_x, next_y, next_theta)
        self._motion_cache[cache_key] = result
        
        # 如果缓存过大，删除一些旧条目
        if len(self._motion_cache) > self._cache_size:
            # 随机删除20%的缓存项
            keys_to_remove = list(self._motion_cache.keys())[:int(self._cache_size * 0.2)]
            for key in keys_to_remove:
                del self._motion_cache[key]
        
        return result
    
    def motion_with_collision_check(self, x, y, theta, steering_angle, distance, collision_checker):
        """
        使用自行车运动模型计算下一个状态，同时进行碰撞检测
        
        Args:
            x, y, theta: 当前位置和朝向
            steering_angle: 转向角（弧度）
            distance: 移动距离
            collision_checker: 碰撞检测函数
            
        Returns:
            tuple: 新的位置和朝向 (x_new, y_new, theta_new)
        """
        # 分段移动，每段都进行碰撞检测
        num_segments = max(1, min(3, int(distance / 0.5)))  # 自适应分段数
        segment_distance = distance / num_segments
        current_x, current_y, current_theta = x, y, theta
        
        for _ in range(num_segments):
            # 计算下一个中间点
            next_x, next_y, next_theta = self.bicycle_model(
                current_x, current_y, current_theta,
                steering_angle, segment_distance
            )
            
            # 检查中间点是否有效
            if not collision_checker(next_x, next_y, next_theta):
                # 如果无效，返回当前状态
                return current_x, current_y, current_theta
            
            # 更新当前状态
            current_x, current_y, current_theta = next_x, next_y, next_theta
        
        return current_x, current_y, current_theta
    
    def get_vehicle_corners(self, x, y, theta):
        """
        获取车辆四个角点的坐标
        
        Args:
            x, y, theta: 车辆位置和朝向
            
        Returns:
            List[Tuple[float, float]]: 角点坐标列表
        """
        # 车辆矩形四个角点的相对坐标
        half_length = self.vehicle_length / 2
        half_width = self.vehicle_width / 2
        corners = [
            (half_length, half_width),   # 右前
            (half_length, -half_width),  # 左前
            (-half_length, -half_width), # 左后
            (-half_length, half_width)   # 右后
        ]
        
        # 旋转并平移角点到世界坐标系
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        
        rotated_corners = []
        for corner in corners:
            corner_x = x + corner[0] * cos_theta - corner[1] * sin_theta
            corner_y = y + corner[0] * sin_theta + corner[1] * cos_theta
            rotated_corners.append((corner_x, corner_y))
        
        return rotated_corners
    
    @staticmethod
    def normalize_angle(angle):
        """归一化角度到[-π, π]范围"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


class CollisionChecker:
    """碰撞检测 - 优化版本"""
    
    def __init__(self, env, vehicle_params, grid_resolution=0.3):
        """
        初始化碰撞检测器
        
        Args:
            env: 环境对象
            vehicle_params: 车辆参数
            grid_resolution: 网格分辨率
        """
        self.env = env
        self.vehicle_length = vehicle_params.get('length', 5.0)
        self.vehicle_width = vehicle_params.get('width', 2.5)
        self.grid_resolution = grid_resolution
        
        # 创建一个车辆模型用于获取车辆几何形状
        self.vehicle_model = BicycleModel(vehicle_params)
        
        # 碰撞检测缓存
        self._collision_cache = {}
        self._cache_size = 5000  # 碰撞缓存大小
        self._cache_hits = 0
        self._cache_misses = 0
        
        # 预计算常用几何数据
        self._diagonal_length = math.sqrt(self.vehicle_length**2 + self.vehicle_width**2)
        self._half_length = self.vehicle_length / 2
        self._half_width = self.vehicle_width / 2
        
        # 创建占用网格 - 用于快速初步碰撞检测
        self.initialize_occupancy_grid()
    
    def initialize_occupancy_grid(self):
        """
        初始化占用网格 - 一种空间哈希数据结构，加速碰撞检测
        """
        # 创建占用网格 - 使用环境网格简化，但分辨率可能不同
        self.occupancy_grid = {}
        
        # 将环境网格中的障碍物加入到占用网格
        for x in range(self.env.width):
            for y in range(self.env.height):
                if self.env.grid[x, y] == 1:  # 障碍物
                    # 计算对应的网格单元
                    cell_x = int(x / self.grid_resolution)
                    cell_y = int(y / self.grid_resolution)
                    cell_key = (cell_x, cell_y)
                    
                    # 将障碍物单元添加到占用网格
                    if cell_key not in self.occupancy_grid:
                        self.occupancy_grid[cell_key] = 1
    
    def is_state_valid(self, x, y, theta):
        """
        检查状态是否有效（没有碰撞）- 优化版本
        
        Args:
            x, y, theta: 位置和朝向
            
        Returns:
            bool: 状态是否有效
        """
        # 生成缓存键
        cache_key = (round(x, 1), round(y, 1), round(theta, 2))
        
        # 检查缓存
        if cache_key in self._collision_cache:
            self._cache_hits += 1
            return self._collision_cache[cache_key]
        
        self._cache_misses += 1
        
        # 快速边界检查
        if not (0 <= x < self.env.width and 0 <= y < self.env.height):
            self._collision_cache[cache_key] = False
            return False
        
        # 快速中心点检查 - 如果车辆中心点在障碍物上，直接返回无效
        if self.env.grid[int(x), int(y)] == 1:
            self._collision_cache[cache_key] = False
            return False
        
        # 快速半径检查 - 使用一个保守的圆形区域做初步判断
        if self.quick_radius_check(x, y):
            # 如果快速检查通过，进行详细的碰撞检测
            is_valid = self.detailed_collision_check(x, y, theta)
            
            # 更新缓存
            self._collision_cache[cache_key] = is_valid
            
            # 如果缓存过大，删除一些旧条目
            if len(self._collision_cache) > self._cache_size:
                # 随机删除20%的缓存项
                keys_to_remove = list(self._collision_cache.keys())[:int(self._cache_size * 0.2)]
                for key in keys_to_remove:
                    del self._collision_cache[key]
            
            return is_valid
        
        # 如果快速半径检查失败，返回有效
        self._collision_cache[cache_key] = True
        return True
    
    def quick_radius_check(self, x, y):
        """
        快速半径检查 - 检查车辆周围是否有障碍物
        
        Args:
            x, y: 位置
            
        Returns:
            bool: 如果附近有障碍物返回True，需要进一步检查
        """
        # 使用车辆对角线长度作为保守半径
        radius = self._diagonal_length / 2
        
        # 获取边界
        left = max(0, int(x - radius))
        right = min(self.env.width - 1, int(x + radius))
        bottom = max(0, int(y - radius))
        top = min(self.env.height - 1, int(y + radius))
        
        # 检查边界框内是否有障碍物
        for cx in range(left, right + 1):
            for cy in range(bottom, top + 1):
                if self.env.grid[cx, cy] == 1:
                    # 计算到中心的距离
                    dx = cx - x
                    dy = cy - y
                    dist_sq = dx*dx + dy*dy
                    
                    # 如果距离小于半径，需要详细检查
                    if dist_sq < radius*radius:
                        return True
        
        # 如果没有找到障碍物，不需要详细检查
        return False
    
    def detailed_collision_check(self, x, y, theta):
        """
        详细的碰撞检测
        
        Args:
            x, y, theta: 位置和朝向
            
        Returns:
            bool: 是否无碰撞
        """
        # 获取车辆四个角点
        corners = self.vehicle_model.get_vehicle_corners(x, y, theta)
        
        # 检查每个角点
        for corner_x, corner_y in corners:
            # 边界检查
            if not (0 <= corner_x < self.env.width and 0 <= corner_y < self.env.height):
                return False
            
            # 障碍物检查
            if self.env.grid[int(corner_x), int(corner_y)] == 1:
                return False
        
        # 检查车辆边缘
        num_samples = 5  # 每条边采样点数
        
        # 遍历四条边
        for i in range(4):
            x1, y1 = corners[i]
            x2, y2 = corners[(i+1) % 4]
            
            # 在边上采样点
            for j in range(num_samples):
                t = (j + 1) / (num_samples + 1)  # 不检查端点
                sx = x1 + t * (x2 - x1)
                sy = y1 + t * (y2 - y1)
                
                # 边界检查
                if not (0 <= sx < self.env.width and 0 <= sy < self.env.height):
                    return False
                
                # 障碍物检查
                if self.env.grid[int(sx), int(sy)] == 1:
                    return False
        
        # 如果所有检查都通过，则状态有效
        return True
    
    def is_intermediate_valid(self, x1, y1, x2, y2, samples=5):
        """
        检查两点之间的路径是否有效（不穿墙）
        
        Args:
            x1, y1: 起点坐标
            x2, y2: 终点坐标
            samples: 检查点数量
            
        Returns:
            bool: 路径是否有效
        """
        # 先做一个快速的线段检测
        # 如果起点和终点都很接近，可以跳过详细检查
        dx = x2 - x1
        dy = y2 - y1
        dist_sq = dx*dx + dy*dy
        
        if dist_sq < 1.0:  # 如果距离小于1，直接返回有效
            return True
        
        # 在两点之间采样检查点
        for i in range(samples):
            t = (i + 1) / (samples + 1)  # 不检查端点
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            
            # 边界和障碍物检查
            if not (0 <= x < self.env.width and 0 <= y < self.env.height):
                return False
            if self.env.grid[int(x), int(y)] == 1:
                return False
        
        return True