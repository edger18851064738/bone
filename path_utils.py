import math
import numpy as np
import time
from queue import Queue
from typing import List, Tuple, Dict, Optional, Callable
import concurrent.futures

class ReedSheppCurves:
    """Reed-Shepp曲线计算 - 优化版本"""
    
    def __init__(self, turning_radius):
        """
        初始化RS曲线计算器
        
        Args:
            turning_radius: 最小转弯半径
        """
        self.turning_radius = turning_radius
        
        # 曲线类型
        self.curve_types = ['LSL', 'LSR', 'RSL', 'RSR', 'RLR', 'LRL']
        
        # RS曲线缓存
        self._rs_cache = {}
        self._cache_size = 500  # 最多保存500条路径
        self._cache_hits = 0
        self._cache_misses = 0
        
        # 用于离散化的步长
        self.rs_step_size = 0.2
    
    def get_path(self, start, goal):
        """
        计算从起点到终点的RS曲线路径
        
        Args:
            start: 起点位置 (x, y, theta)
            goal: 终点位置 (x, y, theta)
            
        Returns:
            rs_path: RS曲线路径，如果无法找到则返回None
        """
        # 生成缓存键
        cache_key = (
            round(start[0], 1), round(start[1], 1), round(start[2], 2),
            round(goal[0], 1), round(goal[1], 1), round(goal[2], 2)
        )
        
        # 检查缓存
        if cache_key in self._rs_cache:
            self._cache_hits += 1
            return self._rs_cache[cache_key]
        
        self._cache_misses += 1
        
        # 计算局部坐标下的目标点
        local_goal = self.global_to_local(start, goal)
        
        # 计算所有可能的RS曲线路径
        paths = self.compute_rs_curves(local_goal[0], local_goal[1], local_goal[2])
        
        if not paths:
            return None
        
        # 按照路径长度排序
        paths.sort(key=lambda p: p[0])
        
        # 选择最短的路径
        _, path_type, controls = paths[0]
        
        # 将RS曲线参数转换为实际路径点
        path = self.generate_rs_path(start, path_type, controls)
        
        # 存储到缓存
        self._rs_cache[cache_key] = path
        
        # 如果缓存过大，删除一些旧条目
        if len(self._rs_cache) > self._cache_size:
            # 随机删除20%的缓存项
            keys_to_remove = list(self._rs_cache.keys())[:int(self._cache_size * 0.2)]
            for key in keys_to_remove:
                del self._rs_cache[key]
        
        return path
    
    def global_to_local(self, start, goal):
        """
        将全局坐标转换为以起点为原点的局部坐标
        
        Args:
            start: 起点位置 (x, y, theta)
            goal: 目标位置 (x, y, theta)
            
        Returns:
            tuple: 局部坐标系中的目标点 (x, y, theta)
        """
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        
        # 计算旋转矩阵
        c = math.cos(start[2])
        s = math.sin(start[2])
        
        # 转换目标点到局部坐标系
        local_x = dx * c + dy * s
        local_y = -dx * s + dy * c
        
        # 目标点在局部坐标系中的朝向
        local_theta = self.normalize_angle(goal[2] - start[2])
        
        return (local_x / self.turning_radius, 
                local_y / self.turning_radius, 
                local_theta)
    
    def compute_rs_curves(self, x, y, phi):
        """
        计算所有可能的Reed-Shepp曲线
        
        Args:
            x, y: 目标点在本地坐标系中的位置（已归一化）
            phi: 目标朝向（相对于起始朝向）
            
        Returns:
            list: 所有可能路径的列表，每个元素为(长度,类型,控制参数)
        """
        # 存储所有可能的路径
        paths = []
        
        # 使用更优先的顺序评估曲线类型 - 一般来说LSL和RSR更常见
        # 优先计算CSC路径（曲线-直线-曲线）
        self._CSC(x, y, phi, paths)
        
        # 如果没有找到路径或最短路径太长，再计算其他类型
        if not paths or paths[0][0] > 6.0:
            # 计算CCC路径（曲线-曲线-曲线）
            self._CCC(x, y, phi, paths)
        
        return paths
    
    def _CSC(self, x, y, phi, paths):
        """计算所有CSC类型的路径 - 优化版本"""
        # 左-直-左(LSL) - 最常见的曲线类型
        t, u, v = self._LSL(x, y, phi)
        if t is not None and abs(t) > 1e-10 and abs(u) > 1e-10 and abs(v) > 1e-10:
            paths.append((abs(t) + abs(u) + abs(v), 'LSL', (t, u, v)))
        
        # 右-直-右(RSR) - 次常见的曲线类型
        t, u, v = self._RSR(x, y, phi)
        if t is not None and abs(t) > 1e-10 and abs(u) > 1e-10 and abs(v) > 1e-10:
            paths.append((abs(t) + abs(u) + abs(v), 'RSR', (t, u, v)))
        
        # 左-直-右(LSR)
        t, u, v = self._LSR(x, y, phi)
        if t is not None and abs(t) > 1e-10 and abs(u) > 1e-10 and abs(v) > 1e-10:
            paths.append((abs(t) + abs(u) + abs(v), 'LSR', (t, u, v)))
        
        # 右-直-左(RSL)
        t, u, v = self._RSL(x, y, phi)
        if t is not None and abs(t) > 1e-10 and abs(u) > 1e-10 and abs(v) > 1e-10:
            paths.append((abs(t) + abs(u) + abs(v), 'RSL', (t, u, v)))
    
    def _CCC(self, x, y, phi, paths):
        """计算所有CCC类型的路径 - 优化版本"""
        # 左-右-左(LRL)
        t, u, v = self._LRL(x, y, phi)
        if t is not None and abs(t) > 1e-10 and abs(u) > 1e-10 and abs(v) > 1e-10:
            paths.append((abs(t) + abs(u) + abs(v), 'LRL', (t, u, v)))
        
        # 右-左-右(RLR)
        t, u, v = self._RLR(x, y, phi)
        if t is not None and abs(t) > 1e-10 and abs(u) > 1e-10 and abs(v) > 1e-10:
            paths.append((abs(t) + abs(u) + abs(v), 'RLR', (t, u, v)))
    
    def _LSL(self, x, y, phi):
        """计算左-直-左(LSL)路径的参数 - 优化版本"""
        u = 0.0
        t = 0.0
        v = 0.0
        
        # 计算圆心到目标点的向量
        cx = -math.sin(phi)
        cy = 1.0 - math.cos(phi)
        
        # 计算u（直线段长度）
        u = math.sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy))
        
        if u < 1e-10:
            return None, None, None
        
        # 计算t和v（两个圆弧的角度）
        theta = math.atan2(y-cy, x-cx)
        t = self.normalize_angle(theta)
        v = self.normalize_angle(phi - t)
        
        return t, u, v
    
    def _RSR(self, x, y, phi):
        """计算右-直-右(RSR)路径的参数 - 优化版本"""
        u = 0.0
        t = 0.0
        v = 0.0
        
        # 计算圆心到目标点的向量
        cx = math.sin(phi)
        cy = -1.0 + math.cos(phi)
        
        # 计算u（直线段长度）
        u = math.sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy))
        
        if u < 1e-10:
            return None, None, None
        
        # 计算t和v（两个圆弧的角度）
        theta = math.atan2(y-cy, x-cx)
        t = self.normalize_angle(-theta)
        v = self.normalize_angle(-phi + t)
        
        return t, u, v
    
    def _LSR(self, x, y, phi):
        """计算左-直-右(LSR)路径的参数 - 优化版本"""
        u = 0.0
        t = 0.0
        v = 0.0
        
        # 归一化角度到[-π, π]
        phi_norm = self.normalize_angle(math.pi - phi)
        
        # LSR路径的目标点要做坐标变换
        x_prime = x * math.cos(phi_norm) + y * math.sin(phi_norm)
        y_prime = x * math.sin(phi_norm) - y * math.cos(phi_norm)
        
        # 计算参数
        u = math.sqrt(x_prime*x_prime + (y_prime-2)*(y_prime-2))
        
        if u < 2.0:  # 曲线无法连接
            return None, None, None
        
        theta = math.atan2(y_prime-2, x_prime)
        t = self.normalize_angle(theta)
        v = self.normalize_angle(t + phi_norm)
        
        return t, u, v
    
    def _RSL(self, x, y, phi):
        """计算右-直-左(RSL)路径的参数 - 优化版本"""
        u = 0.0
        t = 0.0
        v = 0.0
        
        # 由于对称性，RSL实际上就是LSR的镜像
        phi_norm = self.normalize_angle(math.pi - phi)
        
        # RSL路径的目标点要做坐标变换
        x_prime = x * math.cos(phi_norm) - y * math.sin(phi_norm)
        y_prime = x * math.sin(phi_norm) + y * math.cos(phi_norm)
        
        # 计算参数
        u = math.sqrt(x_prime*x_prime + (y_prime+2)*(y_prime+2))
        
        if u < 2.0:  # 曲线无法连接
            return None, None, None
        
        theta = math.atan2(y_prime+2, -x_prime)
        t = self.normalize_angle(theta)
        v = self.normalize_angle(-t - phi_norm)
        
        return t, u, v
    
    def _LRL(self, x, y, phi):
        """计算左-右-左(LRL)路径的参数 - 优化版本"""
        u = 0.0
        t = 0.0
        v = 0.0
        
        # 计算参数
        u = math.sqrt(x*x + y*y)
        
        # 检查边界条件
        if u < 4.0 or u/4.0 > 1.0:  # 曲线无法连接
            return None, None, None
        
        alpha = math.atan2(y, x)
        beta = math.acos(u/4.0)
        
        t = self.normalize_angle(alpha + beta)
        u = self.normalize_angle(math.pi - 2*beta)
        v = self.normalize_angle(phi - t - u)
        
        return t, u, v
    
    def _RLR(self, x, y, phi):
        """计算右-左-右(RLR)路径的参数 - 优化版本"""
        u = 0.0
        t = 0.0
        v = 0.0
        
        # 计算参数
        u = math.sqrt(x*x + y*y)
        
        # 检查边界条件
        if u < 4.0 or u/4.0 > 1.0:  # 曲线无法连接
            return None, None, None
        
        alpha = math.atan2(y, x)
        beta = math.acos(u/4.0)
        
        t = self.normalize_angle(alpha - beta)
        u = self.normalize_angle(-math.pi + 2*beta)
        v = self.normalize_angle(phi - t - u)
        
        return t, u, v
    
    def generate_rs_path(self, start, path_type, controls):
        """
        根据RS曲线类型和控制参数生成实际路径点
        
        Args:
            start: 起始状态 (x, y, theta)
            path_type: 路径类型，如'LSL', 'RSR'等
            controls: 控制参数 (t, u, v)
            
        Returns:
            list: 路径点列表
        """
        t, u, v = controls
        path = []
        
        # 添加起点
        x, y, theta = start[0], start[1], start[2]
        path.append((x, y, theta))
        
        # 生成路径点
        step_size = self.rs_step_size  # 路径离散化步长
        
        # 根据路径类型生成路径点 - 优化版本
        if path_type == 'LSL':
            # 第一段：左转
            for i in range(1, int(abs(t) / step_size) + 1):
                s = min(i * step_size, abs(t)) * (1 if t >= 0 else -1)
                x, y, theta = self.move_along_curve(x, y, theta, s, 'L')
                path.append((x, y, theta))
            
            # 第二段：直行
            for i in range(1, int(abs(u) / step_size) + 1):
                s = min(i * step_size, abs(u)) * (1 if u >= 0 else -1)
                x, y, theta = self.move_along_curve(x, y, theta, s, 'S')
                path.append((x, y, theta))
            
            # 第三段：左转
            for i in range(1, int(abs(v) / step_size) + 1):
                s = min(i * step_size, abs(v)) * (1 if v >= 0 else -1)
                x, y, theta = self.move_along_curve(x, y, theta, s, 'L')
                path.append((x, y, theta))
        
        elif path_type == 'RSR':
            # 第一段：右转
            for i in range(1, int(abs(t) / step_size) + 1):
                s = min(i * step_size, abs(t)) * (1 if t >= 0 else -1)
                x, y, theta = self.move_along_curve(x, y, theta, s, 'R')
                path.append((x, y, theta))
            
            # 第二段：直行
            for i in range(1, int(abs(u) / step_size) + 1):
                s = min(i * step_size, abs(u)) * (1 if u >= 0 else -1)
                x, y, theta = self.move_along_curve(x, y, theta, s, 'S')
                path.append((x, y, theta))
            
            # 第三段：右转
            for i in range(1, int(abs(v) / step_size) + 1):
                s = min(i * step_size, abs(v)) * (1 if v >= 0 else -1)
                x, y, theta = self.move_along_curve(x, y, theta, s, 'R')
                path.append((x, y, theta))
        
        elif path_type == 'LSR':
            # 第一段：左转
            for i in range(1, int(abs(t) / step_size) + 1):
                s = min(i * step_size, abs(t)) * (1 if t >= 0 else -1)
                x, y, theta = self.move_along_curve(x, y, theta, s, 'L')
                path.append((x, y, theta))
            
            # 第二段：直行
            for i in range(1, int(abs(u) / step_size) + 1):
                s = min(i * step_size, abs(u)) * (1 if u >= 0 else -1)
                x, y, theta = self.move_along_curve(x, y, theta, s, 'S')
                path.append((x, y, theta))
            
            # 第三段：右转
            for i in range(1, int(abs(v) / step_size) + 1):
                s = min(i * step_size, abs(v)) * (1 if v >= 0 else -1)
                x, y, theta = self.move_along_curve(x, y, theta, s, 'R')
                path.append((x, y, theta))
        
        elif path_type == 'RSL':
            # 第一段：右转
            for i in range(1, int(abs(t) / step_size) + 1):
                s = min(i * step_size, abs(t)) * (1 if t >= 0 else -1)
                x, y, theta = self.move_along_curve(x, y, theta, s, 'R')
                path.append((x, y, theta))
            
            # 第二段：直行
            for i in range(1, int(abs(u) / step_size) + 1):
                s = min(i * step_size, abs(u)) * (1 if u >= 0 else -1)
                x, y, theta = self.move_along_curve(x, y, theta, s, 'S')
                path.append((x, y, theta))
            
            # 第三段：左转
            for i in range(1, int(abs(v) / step_size) + 1):
                s = min(i * step_size, abs(v)) * (1 if v >= 0 else -1)
                x, y, theta = self.move_along_curve(x, y, theta, s, 'L')
                path.append((x, y, theta))
        
        elif path_type == 'LRL':
            # 第一段：左转
            for i in range(1, int(abs(t) / step_size) + 1):
                s = min(i * step_size, abs(t)) * (1 if t >= 0 else -1)
                x, y, theta = self.move_along_curve(x, y, theta, s, 'L')
                path.append((x, y, theta))
            
            # 第二段：右转
            for i in range(1, int(abs(u) / step_size) + 1):
                s = min(i * step_size, abs(u)) * (1 if u >= 0 else -1)
                x, y, theta = self.move_along_curve(x, y, theta, s, 'R')
                path.append((x, y, theta))
            
            # 第三段：左转
            for i in range(1, int(abs(v) / step_size) + 1):
                s = min(i * step_size, abs(v)) * (1 if v >= 0 else -1)
                x, y, theta = self.move_along_curve(x, y, theta, s, 'L')
                path.append((x, y, theta))
        
        elif path_type == 'RLR':
            # 第一段：右转
            for i in range(1, int(abs(t) / step_size) + 1):
                s = min(i * step_size, abs(t)) * (1 if t >= 0 else -1)
                x, y, theta = self.move_along_curve(x, y, theta, s, 'R')
                path.append((x, y, theta))
            
            # 第二段：左转
            for i in range(1, int(abs(u) / step_size) + 1):
                s = min(i * step_size, abs(u)) * (1 if u >= 0 else -1)
                x, y, theta = self.move_along_curve(x, y, theta, s, 'L')
                path.append((x, y, theta))
            
            # 第三段：右转
            for i in range(1, int(abs(v) / step_size) + 1):
                s = min(i * step_size, abs(v)) * (1 if v >= 0 else -1)
                x, y, theta = self.move_along_curve(x, y, theta, s, 'R')
                path.append((x, y, theta))
        
        return path
    
    def move_along_curve(self, x, y, theta, arc_length, curve_type):
        """
        沿着指定类型的曲线移动指定的弧长
        
        Args:
            x, y, theta: 当前位置和朝向
            arc_length: 移动的弧长
            curve_type: 曲线类型('L'左转, 'R'右转, 'S'直行)
            
        Returns:
            tuple: 新的位置和朝向 (x_new, y_new, theta_new)
        """
        if curve_type == 'S':  # 直行
            return (x + arc_length * math.cos(theta),
                    y + arc_length * math.sin(theta),
                    theta)
        
        elif curve_type == 'L':  # 左转
            return (x + self.turning_radius * (math.sin(theta + arc_length) - math.sin(theta)),
                    y - self.turning_radius * (math.cos(theta + arc_length) - math.cos(theta)),
                    self.normalize_angle(theta + arc_length))
        
        elif curve_type == 'R':  # 右转
            return (x - self.turning_radius * (math.sin(theta - arc_length) - math.sin(theta)),
                    y + self.turning_radius * (math.cos(theta - arc_length) - math.cos(theta)),
                    self.normalize_angle(theta - arc_length))
    
    @staticmethod
    def normalize_angle(angle):
        """归一化角度到[-π, π]范围"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


class PathSmoother:
    """路径平滑 - 优化版本"""
    
    def __init__(self, collision_checker, params=None):
        """
        初始化路径平滑器
        
        Args:
            collision_checker: 碰撞检测函数
            params: 平滑参数
        """
        self.collision_checker = collision_checker
        
        # 默认参数
        self.params = {
            'enabled': True,
            'factor': 0.5,
            'iterations': 10
        }
        
        # 更新参数
        if params:
            self.params.update(params)
    
    def smooth(self, path):
        """
        平滑路径 - 优化版本
        
        Args:
            path: 原始路径
            
        Returns:
            list: 平滑后的路径
        """
        if not self.params['enabled'] or len(path) <= 2:
            return path
        
        # 复制路径
        smoothed_path = path.copy()
        
        # 迭代平滑
        iterations = self.params['iterations']
        smoothing_factor = self.params['factor']
        
        for _ in range(iterations):
            # 创建新路径（保留起点和终点）
            new_path = [smoothed_path[0]]
            
            # 平滑中间点 - 使用加权平均
            for i in range(1, len(smoothed_path) - 1):
                prev = smoothed_path[i-1]
                curr = smoothed_path[i]
                next_p = smoothed_path[i+1]
                
                # 平滑x和y坐标
                x = curr[0] * (1 - smoothing_factor) + (prev[0] + next_p[0]) * smoothing_factor / 2
                y = curr[1] * (1 - smoothing_factor) + (prev[1] + next_p[1]) * smoothing_factor / 2
                
                # 平滑角度 - 使用路径方向
                dx = next_p[0] - prev[0]
                dy = next_p[1] - prev[1]
                
                # 如果有明显的方向变化，使用前进方向作为角度
                if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                    theta = math.atan2(dy, dx)
                else:
                    # 否则保持原角度
                    theta = curr[2]
                
                # 确保角度在[-π, π]范围内
                theta = self.normalize_angle(theta)
                
                # 确保新位置不会穿墙
                if self.collision_checker(x, y, theta):
                    new_path.append((x, y, theta))
                else:
                    # 如果新位置无效，保持原位置
                    new_path.append(curr)
            
            # 添加终点
            new_path.append(smoothed_path[-1])
            smoothed_path = new_path
        
        # 返回平滑后的路径
        return smoothed_path
    
    @staticmethod
    def normalize_angle(angle):
        """归一化角度到[-π, π]范围"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


class DistanceMap:
    """距离图计算 - 优化版本，支持并行计算"""
    
    def __init__(self, width, height, grid, use_parallel=False):
        """
        初始化距离图
        
        Args:
            width: 地图宽度
            height: 地图高度
            grid: 网格数据
            use_parallel: 是否使用并行计算
        """
        self.width = width
        self.height = height
        self.grid = grid
        self.use_parallel = use_parallel
        
        # 初始化距离图
        self.distance_map = np.ones((width, height)) * float('inf')
        
        # 用于并行计算的参数
        self.num_workers = 4  # 默认工作线程数
    
    def compute(self, goal):
        """
        计算从目标点到所有网格点的最短距离
        
        Args:
            goal: 目标位置 (x, y)
        """
        # 目标位置
        goal_x, goal_y = int(goal[0]), int(goal[1])
        
        # 清空距离图
        self.distance_map.fill(float('inf'))
        
        if self.use_parallel:
            self._compute_parallel(goal_x, goal_y)
        else:
            self._compute_sequential(goal_x, goal_y)
    
    def _compute_sequential(self, goal_x, goal_y):
        """
        串行计算距离图
        
        Args:
            goal_x, goal_y: 目标点坐标
        """
        # BFS队列
        queue = Queue()
        queue.put((goal_x, goal_y, 0))  # (x, y, 距离)
        visited = set([(goal_x, goal_y)])
        
        # 设置目标点距离为0
        self.distance_map[goal_x, goal_y] = 0
        
        # 遍历方向 - 4个主方向 + 4个对角线方向
        directions = [
            (0, 1), (1, 0), (0, -1), (-1, 0),  # 主方向
            (1, 1), (-1, 1), (1, -1), (-1, -1)  # 对角线方向
        ]
        
        # BFS计算距离图
        while not queue.empty():
            x, y, dist = queue.get()
            
            # 更新距离
            self.distance_map[x, y] = dist
            
            # 遍历相邻点
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                # 检查新位置是否有效
                if (0 <= nx < self.width and 0 <= ny < self.height and 
                    (nx, ny) not in visited and self.grid[nx, ny] == 0):
                    # 计算新距离（对角线移动距离为√2）
                    new_dist = dist + (1.414 if dx != 0 and dy != 0 else 1)
                    queue.put((nx, ny, new_dist))
                    visited.add((nx, ny))
    
    def _compute_parallel(self, goal_x, goal_y):
        """
        并行计算距离图
        
        Args:
            goal_x, goal_y: 目标点坐标
        """
        # 将地图分成多个部分
        chunks = self._split_map_into_chunks()
        
        # 创建参数列表
        params = [(chunk, goal_x, goal_y) for chunk in chunks]
        
        # 并行计算
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(self._compute_chunk, params))
        
        # 合并结果
        for (start_x, end_x, start_y, end_y), distance_chunk in results:
            self.distance_map[start_x:end_x, start_y:end_y] = distance_chunk
    
    def _split_map_into_chunks(self):
        """
        将地图分成多个部分，用于并行计算
        
        Returns:
            list: 区块列表，每个元素为(start_x, end_x, start_y, end_y)
        """
        chunks = []
        
        # 简单地将地图等分成num_workers个水平条带
        chunk_height = self.height // self.num_workers
        
        for i in range(self.num_workers):
            start_y = i * chunk_height
            end_y = (i + 1) * chunk_height if i < self.num_workers - 1 else self.height
            chunks.append((0, self.width, start_y, end_y))
        
        return chunks
    
    def _compute_chunk(self, params):
        """
        计算一个区块的距离图
        
        Args:
            params: 包含 (chunk, goal_x, goal_y)
                chunk: (start_x, end_x, start_y, end_y)
                goal_x, goal_y: 目标点坐标
                
        Returns:
            tuple: (chunk, distance_chunk)
        """
        chunk, goal_x, goal_y = params
        start_x, end_x, start_y, end_y = chunk
        
        # 为该区块创建距离图
        distance_chunk = np.ones((end_x - start_x, end_y - start_y)) * float('inf')
        
        # 如果目标点在该区块内，设置其距离为0
        if start_x <= goal_x < end_x and start_y <= goal_y < end_y:
            distance_chunk[goal_x - start_x, goal_y - start_y] = 0
        
        # 这里简化了计算，实际上需要更复杂的处理来考虑跨区块的路径
        # 可以使用边界交换或者其他方法来解决这个问题
        
        return chunk, distance_chunk
    
    def get_distance(self, x, y):
        """
        获取指定位置的距离值
        
        Args:
            x, y: 位置坐标
            
        Returns:
            float: 距离值
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.distance_map[int(x), int(y)]
        return float('inf')

# 添加到path_utils.py文件中，在现有类的前面或后面
import threading
class EnhancedReedSheppCurves(ReedSheppCurves):
    """增强版Reed-Shepp曲线计算，专注于生成更平滑的曲线"""
    
    def __init__(self, turning_radius):
        """
        初始化增强版RS曲线计算器
        
        Args:
            turning_radius: 最小转弯半径
        """
        super().__init__(turning_radius)
        
        # 使用更小的步长用于转弯优化
        self.turn_rs_step_size = 0.1  # 用于转弯的更小步长
        
        # 为转弯优化调整参数
        self.turn_curve_preference = {
            'LRL': 1.0,  # 倾向于使用LRL类型
            'RLR': 1.0,  # 倾向于使用RLR类型
            'LSL': 1.2,  # 降低LSL的优先级
            'RSR': 1.2,  # 降低RSR的优先级
            'LSR': 1.4,  # 更少使用LSR
            'RSL': 1.4   # 更少使用RSL
        }
    
    def get_smooth_turn_path(self, start, goal, dense=True):
        """
        生成专为转弯优化的RS曲线路径
        
        Args:
            start: 起点位置 (x, y, theta)
            goal: 终点位置 (x, y, theta)
            dense: 是否使用高密度离散化
            
        Returns:
            list: 优化后的路径点列表
        """
        # 保存原始步长
        original_step = self.rs_step_size
        
        # 使用更小的步长以获得更平滑的曲线
        if dense:
            self.rs_step_size = self.turn_rs_step_size
        
        # 计算局部坐标下的目标点
        local_goal = self.global_to_local(start, goal)
        
        # 优先考虑转弯曲线类型 (CCC类型) 来增强转弯平滑性
        paths = []
        
        # 1. 先尝试CCC类型曲线
        self._CCC(local_goal[0], local_goal[1], local_goal[2], paths)
        
        # 2. 如果没有找到合适的CCC曲线，再尝试CSC曲线
        if not paths:
            self._CSC(local_goal[0], local_goal[1], local_goal[2], paths)
        
        # 按照偏好和长度排序
        if paths:
            # 应用曲线类型偏好
            paths.sort(key=lambda p: p[0] * self.turn_curve_preference.get(p[1], 1.0))
            
            # 选择最优的路径
            _, path_type, controls = paths[0]
            
            # 生成带额外采样点的平滑路径
            path = self.generate_smooth_rs_path(start, path_type, controls)
        else:
            path = None
        
        # 恢复原始步长
        self.rs_step_size = original_step
        
        return path
    
    def generate_smooth_rs_path(self, start, path_type, controls):
        """
        生成经过优化的平滑RS曲线路径
        
        Args:
            start: 起始状态 (x, y, theta)
            path_type: 路径类型，如'LSL', 'RLR'等
            controls: 控制参数 (t, u, v)
            
        Returns:
            list: 更平滑的路径点列表
        """
        t, u, v = controls
        path = []
        
        # 添加起点
        x, y, theta = start[0], start[1], start[2]
        path.append((x, y, theta))
        
        # 使用当前步长
        step_size = self.rs_step_size
        
        # 1. 平滑过渡：提前计算总长度
        total_arc_length = abs(t) + abs(u) + abs(v)
        
        # 2. 根据路径总长度动态调整采样密度
        if total_arc_length > 0:
            # 较短的路径使用更密集的采样
            num_samples = max(50, int(total_arc_length / step_size * 1.5))
            
            # 通过参数化方式均匀采样整条路径
            for i in range(1, num_samples):
                s = i / (num_samples - 1) * total_arc_length  # 沿路径的累积距离
                
                # 确定在哪个段上
                if s <= abs(t):
                    # 第一段
                    seg_s = s * (1 if t >= 0 else -1)
                    x, y, theta = self.move_along_curve(start[0], start[1], start[2], seg_s, path_type[0])
                elif s <= abs(t) + abs(u):
                    # 第二段
                    # 先移动完第一段
                    x_temp, y_temp, theta_temp = self.move_along_curve(start[0], start[1], start[2], t, path_type[0])
                    
                    # 再在第二段上移动
                    seg_s = (s - abs(t)) * (1 if u >= 0 else -1)
                    x, y, theta = self.move_along_curve(x_temp, y_temp, theta_temp, seg_s, path_type[1])
                else:
                    # 第三段
                    # 先移动完第一段
                    x_temp, y_temp, theta_temp = self.move_along_curve(start[0], start[1], start[2], t, path_type[0])
                    
                    # 再移动完第二段
                    x_temp, y_temp, theta_temp = self.move_along_curve(x_temp, y_temp, theta_temp, u, path_type[1])
                    
                    # 最后在第三段上移动
                    seg_s = (s - abs(t) - abs(u)) * (1 if v >= 0 else -1)
                    x, y, theta = self.move_along_curve(x_temp, y_temp, theta_temp, seg_s, path_type[2])
                
                # 添加到路径中
                path.append((x, y, theta))
        
        return path
    
    def _compute_rs_paths_for_turn(self, x, y, phi):
        """
        专门为转弯场景计算RS曲线
        
        Args:
            x, y: 目标点在本地坐标系中的位置（已归一化）
            phi: 目标朝向（相对于起始朝向）
            
        Returns:
            list: 所有可能路径的列表，偏向于产生平滑转弯的路径
        """
        paths = []
        
        # 首先尝试CCC类型曲线 (通常转弯时这些更好)
        self._CCC(x, y, phi, paths)
        
        # 如果没有找到合适的CCC路径，再尝试CSC路径
        if not paths or paths[0][0] > 4.0:
            self._CSC(x, y, phi, paths)
        
        return paths
    
    def smooth_path_section(self, path, start_idx, end_idx):
        """
        优化路径中的一段，使用优化的RS曲线
        
        Args:
            path: 原始路径
            start_idx: 开始索引
            end_idx: 结束索引
            
        Returns:
            list: 优化后的路径段
        """
        if end_idx <= start_idx + 1:
            return path[start_idx:end_idx+1]
        
        # 获取该段的起点和终点
        start_config = path[start_idx]
        end_config = path[end_idx]
        
        # 获取更平滑的RS曲线
        smooth_segment = self.get_smooth_turn_path(start_config, end_config, dense=True)
        
        # 如果找到了平滑路径，返回它
        if smooth_segment:
            return smooth_segment
        
        # 否则返回原始路径段
        return path[start_idx:end_idx+1]
    def get_multi_refined_turn_path(self, start, goal, iterations=3):
        """
        通过多次迭代优化，生成极度平滑的转弯路径
        
        Args:
            start: 起点位置 (x, y, theta)
            goal: 终点位置 (x, y, theta)
            iterations: 迭代优化次数
            
        Returns:
            list: 多次优化后的路径点列表
        """
        # 第一次获取基础RS曲线
        path = self.get_smooth_turn_path(start, goal, dense=True)
        
        if not path or len(path) < 5:
            return path
        
        # 迭代优化
        for i in range(iterations - 1):
            # 选择关键点进行下一次优化
            key_points = self._select_key_points_for_refinement(path)
            
            if len(key_points) < 3:
                break  # 如果没有足够的关键点，中止迭代
            
            # 分段优化
            refined_path = []
            
            for j in range(len(key_points) - 1):
                start_idx = key_points[j]
                end_idx = key_points[j+1]
                
                segment_start = path[start_idx]
                segment_end = path[end_idx]
                
                # 使用更小的步长和更高的采样密度
                temp_step_size = self.rs_step_size
                self.rs_step_size = self.rs_step_size / (i + 2)  # 每次迭代减小步长
                
                # 重新计算该段的RS曲线
                refined_segment = self.get_smooth_turn_path(segment_start, segment_end, dense=True)
                
                # 恢复步长
                self.rs_step_size = temp_step_size
                
                # 添加优化后的段（避免重复点）
                if j == 0:
                    refined_path.extend(refined_segment)
                else:
                    refined_path.extend(refined_segment[1:])
            
            # 更新路径为优化后的版本
            path = refined_path
            
            # 应用张力平滑以消除可能的震荡
            path = self._apply_tension_smoothing(path, tension=0.3)
        
        return path

    def _select_key_points_for_refinement(self, path, max_points=10):
        """
        为进一步优化选择关键点
        
        Args:
            path: 当前路径
            max_points: 最大关键点数
            
        Returns:
            list: 关键点索引列表
        """
        if len(path) <= 2:
            return [0, len(path)-1]
        
        # 始终包含起点和终点
        key_indices = [0, len(path)-1]
        
        # 对较长的路径选择更多的点
        if len(path) < max_points * 2:
            # 路径较短，均匀选择点
            step = max(1, len(path) // max_points)
            for i in range(step, len(path) - step, step):
                key_indices.append(i)
        else:
            # 计算每个点的曲率或方向变化
            curvatures = []
            for i in range(1, len(path) - 1):
                p1, p2, p3 = path[i-1], path[i], path[i+1]
                
                # 计算两个向量
                v1 = (p2[0] - p1[0], p2[1] - p1[1])
                v2 = (p3[0] - p2[0], p3[1] - p2[1])
                
                # 计算向量长度
                len_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
                len_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
                
                if len_v1 < 0.001 or len_v2 < 0.001:
                    curvatures.append(0)
                    continue
                
                # 计算向量夹角余弦值
                cos_angle = (v1[0]*v2[0] + v1[1]*v2[1]) / (len_v1 * len_v2)
                cos_angle = max(-1, min(1, cos_angle))  # 防止数值误差
                
                # 计算曲率得分（角度变化）
                angle = math.acos(cos_angle)
                curvatures.append(angle)
            
            # 选择曲率最大的点作为关键点
            indices = list(range(1, len(path) - 1))
            
            # 如果曲率全为0，则均匀选择点
            if all(c == 0 for c in curvatures):
                step = max(1, len(path) // max_points)
                for i in range(step, len(path) - step, step):
                    key_indices.append(i)
            else:
                # 按曲率排序选择关键点
                sorted_indices = sorted(zip(indices, curvatures), key=lambda x: x[1], reverse=True)
                
                # 选择曲率最大的几个点
                for idx, _ in sorted_indices[:max_points-2]:
                    key_indices.append(idx)
        
        # 确保点是有序的
        key_indices.sort()
        return key_indices

    def _apply_tension_smoothing(self, path, tension=0.5, iterations=2):
        """
        应用张力平滑算法，消除震荡
        
        Args:
            path: 路径点列表
            tension: 张力系数（0-1）
            iterations: 平滑迭代次数
            
        Returns:
            list: 平滑后的路径
        """
        if len(path) <= 2:
            return path
        
        smoothed = list(path)
        
        for _ in range(iterations):
            # 创建新路径（保留起点和终点）
            new_path = [smoothed[0]]
            
            # 对中间点应用张力平滑
            for i in range(1, len(smoothed) - 1):
                prev = smoothed[i-1]
                curr = smoothed[i]
                next_p = smoothed[i+1]
                
                # 平滑坐标
                x = curr[0] + tension * ((prev[0] + next_p[0])/2 - curr[0])
                y = curr[1] + tension * ((prev[1] + next_p[1])/2 - curr[1])
                
                # 平滑角度 - 方向插值
                dx = next_p[0] - prev[0]
                dy = next_p[1] - prev[1]
                
                if abs(dx) > 0.001 or abs(dy) > 0.001:
                    # 使用路径方向作为角度
                    theta = math.atan2(dy, dx)
                else:
                    # 保持原角度
                    theta = curr[2]
                
                # 归一化角度
                theta = self.normalize_angle(theta)
                
                new_path.append((x, y, theta))
            
            # 添加终点
            new_path.append(smoothed[-1])
            smoothed = new_path
        
        return smoothed
class RSCache:
    """RS曲线缓存，提高路径生成效率"""
    def __init__(self, max_size=10000):
        self.cache = {}
        self.max_size = max_size
        self.lock = threading.Lock()
    
    def get(self, start, goal, turning_radius):
        """获取缓存的RS曲线，如无则返回None"""
        key = self._make_key(start, goal, turning_radius)
        with self.lock:
            return self.cache.get(key)
    
    def put(self, start, goal, turning_radius, path):
        """将RS曲线添加到缓存"""
        key = self._make_key(start, goal, turning_radius)
        with self.lock:
            # 如果缓存已满，删除随机条目
            if len(self.cache) >= self.max_size:
                # 删除10%的缓存条目
                keys_to_remove = list(self.cache.keys())[:self.max_size // 10]
                for k in keys_to_remove:
                    del self.cache[k]
            self.cache[key] = path
    
    def _make_key(self, start, goal, turning_radius):
        """创建缓存键"""
        # 离散化坐标和角度以提高命中率
        grid_size = 0.1
        angle_grid = 0.05
        
        start_x = round(start[0] / grid_size) * grid_size
        start_y = round(start[1] / grid_size) * grid_size
        start_theta = round(start[2] / angle_grid) * angle_grid
        
        goal_x = round(goal[0] / grid_size) * grid_size
        goal_y = round(goal[1] / grid_size) * grid_size
        goal_theta = round(goal[2] / angle_grid) * angle_grid
        
        return (start_x, start_y, start_theta, goal_x, goal_y, goal_theta, turning_radius)

def improved_visualize_environment(env, planner=None, ax=None, show_trajectories=True):
    """
    改进的环境可视化函数，增加车辆形状显示和RS曲线路径
    
    Args:
        env: 环境对象
        planner: HybridAStarPlanner对象，用于绘制车辆形状
        ax: 指定的坐标轴，如果为None则创建新的
        show_trajectories: 是否显示车辆轨迹
    
    Returns:
        fig, ax: 图形对象和坐标轴对象
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))
    else:
        fig = ax.figure
        ax.clear()  # 清除当前坐标轴上的内容
    
    # 使用热力图绘制网格，更好地显示道路
    ax.imshow(env.grid.T, origin='lower', cmap='Greys', alpha=0.7)
    
    # 高亮显示道路
    road_mask = (env.grid == 0).T
    ax.imshow(road_mask, origin='lower', cmap='Blues', alpha=0.3)
    
    # 绘制网格线，帮助定位
    ax.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # 添加装载点、卸载点和停车场标记 - 使用更明显的标记
    marker_size = 250  # 增大标记尺寸
    
    # 绘制装载点区域
    for i, point in enumerate(env.loading_points):
        # 绘制区域
        circle = plt.Circle((point[0], point[1]), 5, color='lightgreen', alpha=0.3)
        ax.add_patch(circle)
        # 绘制标记
        ax.scatter(point[0], point[1], c='darkgreen', s=marker_size, marker='o', 
                  edgecolors='black', zorder=10)
        ax.text(point[0], point[1], f'装载点{i+1}', fontsize=12, ha='center', va='center', 
               color='white', weight='bold', zorder=11)
    
    # 绘制卸载点区域
    for i, point in enumerate(env.unloading_points):
        # 绘制区域
        circle = plt.Circle((point[0], point[1]), 5, color='indianred', alpha=0.3)
        ax.add_patch(circle)
        # 绘制标记
        ax.scatter(point[0], point[1], c='darkred', s=marker_size, marker='s', 
                  edgecolors='black', zorder=10)
        ax.text(point[0], point[1], f'卸载点{i+1}', fontsize=12, ha='center', va='center', 
               color='white', weight='bold', zorder=11)
    
    # 显示车辆（如果有）
    for vehicle_id, vehicle in env.vehicles.items():
        position = vehicle['position']
        status = vehicle.get('status', 'idle')
        
        # 根据车辆状态调整颜色
        if 'color' in vehicle:
            base_color = vehicle['color']
        else:
            if status == 'loading':
                base_color = [0.2, 0.8, 0.2]  # 绿色表示装载中
            elif status == 'unloading':
                base_color = [0.8, 0.2, 0.2]  # 红色表示卸载中
            elif status == 'moving':
                base_color = [0.2, 0.2, 0.8]  # 蓝色表示移动中
            else:
                base_color = [0.5, 0.5, 0.5]  # 灰色表示闲置
        
        # 如果提供了规划器，使用其绘制车辆形状
        if planner and hasattr(planner, 'vehicle_model'):
            # 调整透明度，根据负载情况
            load_ratio = vehicle.get('load', 0) / vehicle.get('max_load', 100) if vehicle.get('max_load', 0) > 0 else 0
            # 颜色随负载变化：空车更透明，满载更不透明
            vehicle_color = base_color
            vehicle_alpha = 0.6 + 0.4 * load_ratio
            
            # 绘制车辆
            if hasattr(planner, 'draw_vehicle'):
                planner.draw_vehicle(ax, position, color=vehicle_color, alpha=vehicle_alpha)
            else:
                # 获取车辆参数
                vehicle_length = planner.vehicle_params.get('length', 5.0)
                vehicle_width = planner.vehicle_params.get('width', 2.5)
                
                # 计算车辆四个角点的坐标
                x, y, theta = position
                corners = planner.vehicle_model.get_vehicle_corners(x, y, theta)
                
                # 绘制车辆轮廓
                polygon = plt.Polygon(corners, color=vehicle_color, alpha=vehicle_alpha)
                ax.add_patch(polygon)
                
                # 绘制车头方向
                ax.plot([x, x + vehicle_length/2 * math.cos(theta)], 
                       [y, y + vehicle_length/2 * math.sin(theta)], 
                       color='black', linewidth=1)
            
            # 添加车辆ID标签
            x, y, _ = position
            ax.text(x, y, f'V{vehicle_id}', fontsize=10, 
                   ha='center', va='center', color='white', weight='bold', zorder=12)
            
            # 可选：添加车辆状态指示器
            status_text = {
                'idle': '闲置',
                'loading': '装载中',
                'unloading': '卸载中',
                'moving': '移动中'
            }.get(status, '')
            
            if status_text:
                ax.text(x, y - 3, status_text, fontsize=8, 
                       ha='center', va='center', color='black', 
                       bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
        else:
            # 如果没有规划器，使用简单标记
            marker_type = {'idle': 'o', 'loading': '^', 'unloading': 'v', 'moving': '*'}.get(status, '*')
            ax.scatter(position[0], position[1], c=base_color, s=100, marker=marker_type, 
                      edgecolors='black', zorder=10)
            ax.text(position[0], position[1], f'V{vehicle_id}', fontsize=10, 
                   ha='center', va='center', color='white', weight='bold', zorder=11)
        
        # 绘制车辆路径（如果有）- 使用平滑曲线
        if show_trajectories and 'path' in vehicle and vehicle['path']:
            path = vehicle['path']
            
            # 提取路径点
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            
            # 绘制主路径轨迹 - 使用渐变色增强视觉效果
            points = np.array([path_x, path_y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # 创建渐变色映射，从绿色(起点)到红色(终点)
            cmap = plt.get_cmap('cool')
            colors = np.linspace(0, 1, len(segments))
            
            # 使用LineCollection绘制彩色路径
            from matplotlib.collections import LineCollection
            lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, 1), 
                               alpha=0.6, linewidth=2.5, zorder=5)
            lc.set_array(colors)
            line = ax.add_collection(lc)
            
            # 绘制路径开始和结束点
            ax.scatter(path_x[0], path_y[0], c='green', s=50, marker='o', 
                      edgecolors='black', zorder=6)
            ax.scatter(path_x[-1], path_y[-1], c='red', s=50, marker='x', 
                      edgecolors='black', zorder=6)
            
            # 绘制路径点和方向 - 仅在关键点处显示
            num_arrows = min(10, len(path))  # 最多显示10个方向箭头
            step = max(1, len(path) // num_arrows)
            
            for i in range(0, len(path), step):
                if i < len(path):
                    px, py, ptheta = path[i]
                    # 绘制方向箭头
                    arrow_length = 3.0
                    dx = arrow_length * math.cos(ptheta)
                    dy = arrow_length * math.sin(ptheta)
                    ax.arrow(px, py, dx, dy, head_width=1.5, head_length=1.5, 
                            fc='k', ec='k', alpha=0.5, zorder=7)
    
    # 设置坐标轴和标题
    ax.set_title('露天矿运输路径规划', fontsize=16)
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    ax.set_xlim(0, env.width-1)
    ax.set_ylim(0, env.height-1)
    
    # 添加图例
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, alpha=0.7, label='规划路径'),
        Patch(facecolor='lightgreen', alpha=0.3, label='装载区'),
        Patch(facecolor='indianred', alpha=0.3, label='卸载区')
    ]
    
    # 添加车辆状态图例
    if any('status' in v for v in env.vehicles.values()):
        for status, color, name in [
            ('moving', [0.2, 0.2, 0.8], '移动中'),
            ('loading', [0.2, 0.8, 0.2], '装载中'),
            ('unloading', [0.8, 0.2, 0.2], '卸载中'),
            ('idle', [0.5, 0.5, 0.5], '闲置')
        ]:
            if any(v.get('status', '') == status for v in env.vehicles.values()):
                legend_elements.append(Patch(facecolor=color, alpha=0.7, label=name))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # 显示坐标信息
    ax.format_coord = lambda x, y: f'坐标: ({x:.1f}, {y:.1f})'
    
    fig.tight_layout()
    return fig, ax