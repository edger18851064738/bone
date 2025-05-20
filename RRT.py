"""
RRT.py - 优化版本的双向RRT路径规划器
包含线段检测和RS曲线优化功能，维持与GUI.py的接口兼容性
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import matplotlib

# 设置中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
class RRTPlanner:
    """双向RRT路径规划器，具有RS曲线优化功能，保持与GUI兼容的接口"""
    
    def __init__(self, env, vehicle_length=5.0, vehicle_width=2.0, 
                 turning_radius=5.0, step_size=0.8, grid_resolution=0.3):
        """
        初始化RRT规划器
        
        参数:
            env: 环境对象
            vehicle_length: 车辆长度
            vehicle_width: 车辆宽度
            turning_radius: 最小转弯半径
            step_size: 步长
            grid_resolution: 网格分辨率
        """
        self.env = env
        self.vehicle_length = vehicle_length
        self.vehicle_width = vehicle_width
        self.turning_radius = turning_radius
        self.step_size = step_size
        self.grid_resolution = grid_resolution
        
        # 多代理规划储存
        self.agents = {}  # 存储所有注册的代理
        
        # 导入必要的类
        try:
            from vehicle_model import BicycleModel, CollisionChecker
            
            # 创建车辆模型和碰撞检查器
            vehicle_params = {
                'length': vehicle_length,
                'width': vehicle_width,
                'wheel_base': vehicle_length * 0.6,
                'turning_radius': turning_radius,
                'step_size': step_size
            }
            self.vehicle_model = BicycleModel(vehicle_params)
            self.collision_checker = CollisionChecker(env, vehicle_params)
        except ImportError as e:
            print(f"警告: 无法导入车辆模型，将使用简化碰撞检测: {e}")
            self.vehicle_model = None
            self.collision_checker = None
        
        # 创建双向RRT规划器
        self.bidirectional_rrt = BidirectionalRRT(
            collision_checker=self.collision_checker,
            vehicle_model=self.vehicle_model,
            env=env,
            step_size=step_size * 1.5,  # 使用较大步长加速收敛
            max_steer=math.pi/4,
            goal_bias=0.2,
            vehicle_length=vehicle_length,
            vehicle_width=vehicle_width,
            debug=False
        )
        
        # 导入RS曲线优化器
        try:
            from path_utils import EnhancedReedSheppCurves
            self.rs_curves = EnhancedReedSheppCurves(turning_radius)
        except ImportError as e:
            print(f"警告: 无法导入RS曲线优化器，将使用简化路径优化: {e}")
            self.rs_curves = None
        
        # 调试标志
        self.debug = False
        
        # 路径评估参数 - 优化的权重配置
        self.evaluation_weights = {
            'length': 0.35,      # 路径长度权重
            'smoothness': 0.30,  # 平滑度权重
            'turns': 0.20,       # 转弯次数权重
            'curvature': 0.10,   # 最大曲率权重
            'alignment': 0.05    # 终点对齐权重
        }
        
        # 创建评估缓存，提高性能
        self.path_evaluations = {}
        
        # 路径优化参数
        self.smoothing_params = {
            'min_segment_length': 4,    # 最小线段长度
            'max_error': 1.0,           # 最大误差
            'angle_threshold': 0.3,     # 转弯角度阈值
            'turn_optimization_iters': 3 # 转弯优化迭代次数
        }
    
    def register_agent(self, agent_id, start, goal, priority=1):
        """
        注册一个车辆到规划系统
        
        参数:
            agent_id: 车辆ID
            start: 起始位置 (x, y, theta)
            goal: 目标位置 (x, y, theta)
            priority: 优先级（数值越高优先级越高）
            
        返回:
            bool: 注册是否成功
        """
        if self.debug:
            print(f"注册车辆 {agent_id}，起点:{start}，终点:{goal}，优先级:{priority}")
            
        self.agents[agent_id] = {
            'start': start,
            'goal': goal,
            'priority': priority,
            'path': None
        }
        return True
    
    def plan_paths_for_all(self, max_iterations=6000):
        """
        为所有注册的代理规划路径
        
        参数:
            max_iterations: 每个代理的最大迭代次数
            
        返回:
            dict: 包含每个代理ID和对应路径的字典
        """
        if self.debug:
            print(f"为 {len(self.agents)} 个车辆规划路径")
            
        # 按优先级排序代理
        sorted_agents = sorted(self.agents.items(), 
                              key=lambda x: x[1]['priority'], 
                              reverse=True)
        
        paths = {}
        existing_paths = []  # 已规划路径列表，用于冲突检查
        
        # 为每个代理规划路径
        for agent_id, agent_data in sorted_agents:
            if self.debug:
                print(f"为车辆 {agent_id} 规划路径，优先级: {agent_data['priority']}")
                
            # 规划单个路径
            path = self.plan_path(
                agent_data['start'], 
                agent_data['goal'], 
                agent_id,
                max_iterations
            )
            
            if path:
                # 保存路径
                paths[agent_id] = path
                self.agents[agent_id]['path'] = path
                existing_paths.append((agent_id, path))
                
                if self.debug:
                    print(f"  车辆 {agent_id} 路径规划成功，路径长度: {len(path)}")
            else:
                if self.debug:
                    print(f"  车辆 {agent_id} 路径规划失败")
                
        return paths
    
    def plan_path(self, start, goal, agent_id=None, max_iterations=6000, num_attempts=2):
        """
        规划从起点到终点的路径，使用RRT算法并进行RS曲线优化
        
        参数:
            start: 起点位置 (x, y, theta)
            goal: 终点位置 (x, y, theta)
            agent_id: 代理ID（可选）
            max_iterations: 最大迭代次数
            num_attempts: 尝试次数
            
        返回:
            list: 优化后的路径点列表 [(x, y, theta), ...] 或 None
        """
        best_path = None
        best_score = float('-inf')
        
        if self.debug:
            print(f"开始进行{num_attempts}次路径规划尝试")
        
        # 清除评估缓存
        self.path_evaluations = {}
        
        for attempt in range(num_attempts):
            # 设置RRT规划器的调试标志
            self.bidirectional_rrt.debug = self.debug
            
            # 使用双向RRT规划初始路径
            time_path = self.bidirectional_rrt.plan(
                start=start,
                goal=goal,
                time_step=1.0,
                max_iterations=max_iterations
            )
            
            if time_path:
                # 将时间路径转换为简单路径（移除时间组件）
                path = [(p[0], p[1], p[2]) for p in time_path]
                
                # 应用线段检测和RS曲线优化
                refined_path = self.refine_path_with_line_detection(path)
                
                if refined_path:
                    # 优化转弯
                    final_path = self.optimize_turns_with_rs_curves(refined_path)
                    
                    # 评估路径质量
                    path_score = self._evaluate_path(final_path, start, goal)
                    
                    if self.debug:
                        print(f"  尝试 {attempt + 1}: 路径长度={len(final_path)}, 评分={path_score:.2f}")
                    
                    # 更新最佳路径
                    if path_score > best_score:
                        best_score = path_score
                        best_path = final_path
                        
                        if self.debug:
                            print(f"  发现更好的路径！新的最高评分：{best_score:.2f}")
                else:
                    if self.debug:
                        print(f"  尝试 {attempt + 1}: 路径优化失败")
        
        if best_path:
            if self.debug:
                print(f"完成路径规划，最终选择评分为 {best_score:.2f} 的路径")
            return best_path
        else:
            if self.debug:
                print("所有尝试均失败，未能找到有效路径")
            return None
    
    def optimize_turns_with_rs_curves(self, path, turn_threshold=0.3):
        """
        优化路径中的转弯部分，使用RS曲线使其更加平滑
        
        参数:
            path: 初步优化后的路径 [(x, y, theta), ...]
            turn_threshold: 识别转弯的角度阈值（弧度）
            
        返回:
            list: 优化后的路径
        """
        if len(path) < 5 or self.rs_curves is None:
            return path
            
        # 1. 识别路径中的转弯部分
        turn_sections = self._identify_turn_sections(path, turn_threshold)
        
        if self.debug:
            print(f"找到 {len(turn_sections)} 个转弯部分需要优化")
        
        # 2. 优化每个转弯部分
        optimized_path = list(path)  # 创建副本
        
        # 追踪在优化过程中索引的偏移
        offset = 0
        
        for orig_start, orig_end in turn_sections:
            # 调整索引以考虑之前优化带来的变化
            section_start = orig_start + offset
            section_end = orig_end + offset
            
            # 获取转弯部分
            turn_section = optimized_path[section_start:section_end+1]
            
            # 如果转弯部分太短，跳过
            if len(turn_section) < 3:
                continue
                
            # 使用增强版RS曲线优化转弯部分
            # 确保从转弯前开始优化并延伸到转弯后
            extension = 2  # 向前和向后多取几个点
            extended_start = max(0, section_start - extension)
            extended_end = min(len(optimized_path) - 1, section_end + extension)
            
            # 获取扩展后的部分
            extended_section = optimized_path[extended_start:extended_end+1]
            
            # 创建平滑的RS曲线通过整个转弯部分
            start_config = extended_section[0]
            end_config = extended_section[-1]
            
            # 使用多次迭代优化的方法
            smooth_curve = self.rs_curves.get_multi_refined_turn_path(
                start_config, end_config, 
                iterations=self.smoothing_params['turn_optimization_iters']
            )
            
            # 检查RS曲线是否有效
            if smooth_curve and self._check_rs_segment_valid(smooth_curve):
                # 计算索引变化
                old_len = extended_end - extended_start + 1
                new_len = len(smooth_curve)
                offset += (new_len - old_len)
                
                # 用平滑曲线替换原始段
                optimized_path[extended_start:extended_end+1] = smooth_curve
                
                if self.debug:
                    print(f"  优化了转弯部分 {extended_start}-{extended_end}，"
                        f"替换了 {old_len} 个点为 {new_len} 个点")
        
        return optimized_path
    
    def _identify_turn_sections(self, path, angle_threshold=0.3):
        """
        识别路径中的转弯部分
        
        参数:
            path: 路径
            angle_threshold: 转弯角度阈值
            
        返回:
            list: 转弯部分的索引列表 [(start_idx, end_idx), ...]
        """
        turn_sections = []
        in_turn = False
        turn_start = 0
        
        # 计算每个点的方向变化
        for i in range(1, len(path)-1):
            # 计算前一段和后一段的方向向量
            v1 = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
            v2 = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
            
            # 计算向量长度
            len_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
            len_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            # 跳过极短的段
            if len_v1 < 0.01 or len_v2 < 0.01:
                continue
            
            # 计算夹角
            cos_angle = (v1[0]*v2[0] + v1[1]*v2[1]) / (len_v1 * len_v2)
            cos_angle = max(-1, min(1, cos_angle))  # 防止数值误差
            angle = math.acos(cos_angle)
            
            # 判断是否为转弯点
            if angle > angle_threshold:
                # 如果不在转弯中，标记转弯开始
                if not in_turn:
                    in_turn = True
                    turn_start = max(0, i-2)  # 往前多取两个点以确保平滑过渡
            else:
                # 如果在转弯中且方向变化小，可能是转弯结束
                if in_turn:
                    # 连续几个点方向变化小，确认转弯结束
                    if i > turn_start + 2:  # 确保转弯段有足够长度
                        turn_end = min(len(path)-1, i+2)  # 往后多取两个点
                        
                        # 只保留那些超过最小长度的转弯段
                        if turn_end - turn_start >= 3:
                            turn_sections.append((turn_start, turn_end))
                    
                    in_turn = False
        
        # 处理可能的最后一个转弯段
        if in_turn and turn_start < len(path)-3:
            turn_sections.append((turn_start, len(path)-1))
        
        # 合并相邻或重叠的转弯段
        if len(turn_sections) > 1:
            merged_sections = [turn_sections[0]]
            
            for current in turn_sections[1:]:
                previous = merged_sections[-1]
                
                # 如果当前段与前一段重叠或相邻，合并它们
                if current[0] <= previous[1] + 2:
                    merged_sections[-1] = (previous[0], max(previous[1], current[1]))
                else:
                    merged_sections.append(current)
            
            turn_sections = merged_sections
        
        return turn_sections
    
    def refine_path_with_line_detection(self, path):
        """
        通过检测线段和转弯点，并使用RS曲线连接这些关键点来优化路径
        
        参数:
            path: 原始RRT路径 [(x, y, theta), ...]
            
        返回:
            list: 平滑后的路径 [(x, y, theta), ...]
        """
        if len(path) < 5:  # 太短无法处理
            return path
        
        # 1. 检测路径中的线段
        line_segments = self._detect_line_segments(
            path, 
            max_error=self.smoothing_params['max_error'], 
            min_points=self.smoothing_params['min_segment_length']
        )
        
        if self.debug:
            print(f"检测到 {len(line_segments)} 个线段")
        
        # 2. 提取关键点（线段的起点和终点）
        key_points = self._extract_key_points_from_segments(path, line_segments)
        
        # 3. 使用RS曲线连接关键点
        return self._connect_key_points_with_rs(path, key_points)
    
    def _detect_line_segments(self, path, max_error=1.0, min_points=4):
        """
        使用分段线性拟合检测路径中的线段
        
        参数:
            path: 原始路径
            max_error: 最大允许误差
            min_points: 一个段所需的最小点数
            
        返回:
            list: 检测到的线段，每个元素为 (start_idx, end_idx)
        """
        segments = []
        start_idx = 0
        
        while start_idx < len(path) - min_points:
            # 尝试找到最长的可能线段
            end_idx = start_idx + min_points
            max_deviation = 0
            
            while end_idx < len(path):
                # 创建从起点到当前终点的线
                p_start = path[start_idx]
                p_end = path[end_idx]
                
                # 计算线方程 Ax + By + C = 0
                A = p_end[1] - p_start[1]  # y2 - y1
                B = p_start[0] - p_end[0]  # x1 - x2
                C = p_end[0]*p_start[1] - p_start[0]*p_end[1]  # x2*y1 - x1*y2
                norm = math.sqrt(A*A + B*B)
                
                if norm < 0.001:  # 避免除以零
                    break
                    
                # 检查中间点到线的距离
                max_dev = 0
                for i in range(start_idx + 1, end_idx):
                    # 点到线距离公式
                    dist = abs(A*path[i][0] + B*path[i][1] + C) / norm
                    max_dev = max(max_dev, dist)
                
                # 如果最大偏差在阈值内，继续扩展线段
                if max_dev <= max_error:
                    max_deviation = max_dev
                    end_idx += 1
                else:
                    # 超出误差阈值，结束当前线段
                    break
            
            # 如果找到足够长的线段
            if end_idx - start_idx >= min_points:
                segments.append((start_idx, end_idx - 1))
                start_idx = end_idx - 1  # 从当前终点开始下一个线段
            else:
                # 没有找到足够长的线段，向前移动
                start_idx += 1
        
        return segments
    
    def _extract_key_points_from_segments(self, path, line_segments):
        """
        从检测到的线段中提取关键点
        
        参数:
            path: 原始路径
            line_segments: 线段列表
            
        返回:
            list: 关键点索引列表
        """
        key_indices = set([0, len(path) - 1])  # 始终包括起点和终点
        
        # 添加每个线段的起点和终点
        for start_idx, end_idx in line_segments:
            key_indices.add(start_idx)
            key_indices.add(end_idx)
        
        # 处理线段之间的间隙 - 如果有显著间隙，添加转弯点
        sorted_segments = sorted(line_segments, key=lambda x: x[0])
        for i in range(len(sorted_segments) - 1):
            curr_end = sorted_segments[i][1]
            next_start = sorted_segments[i+1][0]
            
            # 如果线段之间有间隙
            if next_start - curr_end > 3:
                # 在间隙中添加转弯点
                gap_indices = range(curr_end + 1, next_start)
                turning_point = self._find_max_turning_point(path, gap_indices)
                if turning_point is not None:
                    key_indices.add(turning_point)
        
        # 转换为有序列表
        key_indices = sorted(list(key_indices))
        return key_indices
    
    def _find_max_turning_point(self, path, indices):
        """
        在给定索引范围内找到最大转弯点
        
        参数:
            path: 原始路径
            indices: 索引范围
            
        返回:
            int: 最大转弯点的索引，如果没有有效点则返回None
        """
        if not indices:
            return None
            
        max_angle = -1
        max_turning_idx = None
        
        for i in indices:
            if i <= 0 or i >= len(path) - 1:
                continue
                
            # 计算前后点之间的方向变化
            v1 = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
            v2 = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
            
            # 避免零向量
            len_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
            len_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if len_v1 < 0.001 or len_v2 < 0.001:
                continue
                
            # 计算角度
            cos_angle = (v1[0]*v2[0] + v1[1]*v2[1]) / (len_v1 * len_v2)
            cos_angle = max(-1, min(1, cos_angle))  # 防止数值误差
            angle = math.acos(cos_angle)
            
            if angle > max_angle:
                max_angle = angle
                max_turning_idx = i
        
        return max_turning_idx
    
    def _connect_key_points_with_rs(self, path, key_indices):
        """
        使用RS曲线连接关键点
        
        参数:
            path: 原始路径
            key_indices: 关键点索引列表
            
        返回:
            list: 优化后的路径
        """
        if len(key_indices) < 2:
            return path
            
        refined_path = []
        
        # 处理每对相邻的关键点
        for i in range(len(key_indices) - 1):
            start_idx = key_indices[i]
            end_idx = key_indices[i+1]
            
            start_config = path[start_idx]
            end_config = path[end_idx]
            
            # 根据点之间的距离选择连接策略
            dist = math.sqrt((end_config[0]-start_config[0])**2 + 
                           (end_config[1]-start_config[1])**2)
            
            if dist < 2.0:
                # 点非常接近，使用简单线性连接
                segment = [start_config]
                if i < len(key_indices) - 2:  # 不是最后一段
                    # 添加一些使用线性插值的中间点
                    num_steps = max(2, (end_idx - start_idx) // 2)
                    for j in range(1, num_steps):
                        t = j / num_steps
                        x = start_config[0] + t * (end_config[0] - start_config[0])
                        y = start_config[1] + t * (end_config[1] - start_config[1])
                        theta = self._interpolate_angle(start_config[2], end_config[2], t)
                        segment.append((x, y, theta))
                segment.append(end_config)
            else:
                # 较远的点，使用RS曲线 (如果可用)
                if self.rs_curves:
                    # 对于线段，设置适当的起点和终点方向
                    if i > 0 and i < len(key_indices) - 2:
                        # 计算线的方向角
                        dx = end_config[0] - start_config[0]
                        dy = end_config[1] - start_config[1]
                        line_theta = math.atan2(dy, dx)
                        
                        # 创建具有正确方向的配置
                        adjusted_start = (start_config[0], start_config[1], line_theta)
                        adjusted_end = (end_config[0], end_config[1], line_theta)
                        
                        # 尝试使用RS曲线
                        rs_segment = self.rs_curves.get_path(adjusted_start, adjusted_end)
                    else:
                        # 第一段或最后一段，保持原始方向
                        rs_segment = self.rs_curves.get_path(start_config, end_config)
                    
                    # 检查RS曲线是否有效
                    if rs_segment and self._check_rs_segment_valid(rs_segment):
                        segment = rs_segment
                    else:
                        # 如果RS曲线无效，使用原始路径段
                        segment = path[start_idx:end_idx+1]
                else:
                    # 没有RS曲线优化器，使用原始路径段
                    segment = path[start_idx:end_idx+1]
            
            # 添加到结果路径
            if i == 0:  # 第一段，添加所有点
                refined_path.extend(segment)
            else:  # 后续段，跳过第一个点（避免重复）
                refined_path.extend(segment[1:])
        
        return refined_path
    
    def _check_rs_segment_valid(self, segment, check_density=5):
        """
        检查RS曲线段是否有效（无碰撞）
        
        参数:
            segment: RS曲线段
            check_density: 每单位距离的检查点数
            
        返回:
            bool: 段是否有效
        """
        if not self.collision_checker:
            return True  # 如果没有碰撞检查器，假设所有路径都有效
            
        # 检查每个点
        for point in segment:
            x, y, theta = point
            if not self.collision_checker.is_state_valid(x, y, theta):
                return False
        
        # 检查相邻点之间的中间点
        for i in range(len(segment)-1):
            p1 = segment[i]
            p2 = segment[i+1]
            
            # 计算距离
            dist = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
            
            # 如果点很接近，跳过中间检查
            if dist < 0.5:
                continue
                
            # 确定检查点数量
            num_checks = max(2, int(dist * check_density))
            
            # 检查中间点
            for j in range(1, num_checks):
                t = j / num_checks
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                
                # 角度插值
                angle_diff = (p2[2] - p1[2] + math.pi) % (2 * math.pi) - math.pi
                theta = (p1[2] + angle_diff * t) % (2 * math.pi)
                
                if not self.collision_checker.is_state_valid(x, y, theta):
                    return False
        
        return True
    
    def _interpolate_angle(self, theta1, theta2, t):
        """
        角度插值
        
        参数:
            theta1, theta2: 起始和结束角度
            t: 插值参数 [0,1]
            
        返回:
            float: 插值后的角度
        """
        # 将角度差规范化到 [-pi, pi]
        diff = (theta2 - theta1 + math.pi) % (2 * math.pi) - math.pi
        
        # 线性插值
        return (theta1 + diff * t) % (2 * math.pi)
    
    def _evaluate_path(self, path, start=None, goal=None):
        """
        评估路径质量，考虑多个指标
        
        参数:
            path: 待评估的路径
            start: 起点位置（可选）
            goal: 终点位置（可选）
            
        返回:
            float: 路径质量评分
        """
        # 缓存检查 - 使用路径的哈希值作为键
        path_tuple = tuple(tuple(p) for p in path)
        if path_tuple in self.path_evaluations:
            return self.path_evaluations[path_tuple]
        
        if len(path) < 2:
            return float('-inf')
        
        # 1. 计算路径长度（越短越好）
        path_length = 0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            path_length += math.sqrt(dx*dx + dy*dy)
        
        # 2. 计算平滑度（相邻段方向变化越小越好）
        smoothness = 0
        for i in range(1, len(path)-1):
            v1 = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
            v2 = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
            
            # 计算向量夹角
            len_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
            len_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if len_v1 > 0.01 and len_v2 > 0.01:  # 避免除零
                cos_angle = (v1[0]*v2[0] + v1[1]*v2[1]) / (len_v1 * len_v2)
                cos_angle = max(-1, min(1, cos_angle))  # 防止数值误差
                angle = math.acos(cos_angle)
                smoothness += angle
        
        # 3. 计算转弯次数（越少越好）
        turn_count = 0
        turn_threshold = self.smoothing_params['angle_threshold']
        for i in range(1, len(path)-1):
            v1 = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
            v2 = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
            
            len_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
            len_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if len_v1 > 0.01 and len_v2 > 0.01:
                cos_angle = (v1[0]*v2[0] + v1[1]*v2[1]) / (len_v1 * len_v2)
                cos_angle = max(-1, min(1, cos_angle))
                angle = math.acos(cos_angle)
                
                if angle > turn_threshold:
                    turn_count += 1
        
        # 4. 计算最大曲率（越小越好）
        max_curvature = 0
        for i in range(1, len(path)-1):
            # 使用三点法估算曲率
            x1, y1 = path[i-1][0], path[i-1][1]
            x2, y2 = path[i][0], path[i][1]
            x3, y3 = path[i+1][0], path[i+1][1]
            
            # 计算两个向量
            v1 = (x2-x1, y2-y1)
            v2 = (x3-x2, y3-y2)
            
            # 计算向量长度
            len_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
            len_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if len_v1 > 0.01 and len_v2 > 0.01:
                # 计算夹角
                cos_angle = (v1[0]*v2[0] + v1[1]*v2[1]) / (len_v1 * len_v2)
                cos_angle = max(-1, min(1, cos_angle))
                angle = math.acos(cos_angle)
                
                # 估算曲率（夹角/路径长度）
                curvature = angle / ((len_v1 + len_v2) / 2)
                max_curvature = max(max_curvature, curvature)
        
        # 5. 计算与目标点的方向对齐程度（如果提供了目标）
        alignment_score = 0
        if goal is not None:
            end_alignment = abs(path[-1][2] - goal[2])  # theta差异
            end_alignment = min(end_alignment, 2*math.pi - end_alignment)  # 取最小角度差
            alignment_score = -end_alignment  # 角度差越小越好
        
        # 归一化各指标
        length_score = -path_length / 100  # 路径越短越好
        smoothness_score = -smoothness / len(path)  # 平均每个点的方向变化
        turn_score = -turn_count / 5  # 转弯次数归一化
        curvature_score = -max_curvature * 10  # 曲率分数
        
        # 计算总分（权重之和为1）
        total_score = (
            self.evaluation_weights['length'] * length_score +
            self.evaluation_weights['smoothness'] * smoothness_score +
            self.evaluation_weights['turns'] * turn_score +
            self.evaluation_weights['curvature'] * curvature_score +
            self.evaluation_weights['alignment'] * alignment_score
        )
        
        # 缓存评估结果
        self.path_evaluations[path_tuple] = total_score
        
        return total_score
    
    def draw_vehicle(self, ax, position, size=None, color='blue', alpha=0.7):
        """在给定的matplotlib轴上绘制车辆形状"""
        if size is None:
            size = (self.vehicle_length, self.vehicle_width)
                
        x, y, theta = position
        
        # 计算车辆角点
        if self.vehicle_model and hasattr(self.vehicle_model, 'get_vehicle_corners'):
            corners = self.vehicle_model.get_vehicle_corners(x, y, theta)
        else:
            # 使用简化方法计算角点
            half_length = size[0] / 2
            half_width = size[1] / 2
            
            # 计算四个角点的相对坐标
            corners_rel = [
                [half_length, half_width],
                [half_length, -half_width],
                [-half_length, -half_width],
                [-half_length, half_width]
            ]
            
            # 应用旋转和平移
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            corners = [
                (x + cos_t * rx - sin_t * ry, y + sin_t * rx + cos_t * ry)
                for rx, ry in corners_rel
            ]
        
        # 绘制车辆轮廓
        polygon = plt.Polygon(corners, closed=True, fill=True,
                             color=color, alpha=alpha)
        ax.add_patch(polygon)
        
        # 添加一条显示车辆朝向的线
        length = size[0]
        ax.plot([x, x + length/2 * math.cos(theta)], [y, y + length/2 * math.sin(theta)], 
                color='black', linewidth=1)

    def visualize_path_refinement(self, path, refined_path, env=None):
        """
        可视化原始路径和优化后的路径
        
        参数:
            path: 原始路径
            refined_path: 优化后的路径
            env: 环境（可选）
        """
        plt.figure(figsize=(12, 12))
        
        # 绘制环境（如果提供）
        if env and hasattr(env, 'grid'):
            plt.imshow(env.grid.T, origin='lower', cmap='gray_r', alpha=0.7)
        
        # 绘制原始路径
        x_coords = [p[0] for p in path]
        y_coords = [p[1] for p in path]
        plt.plot(x_coords, y_coords, 'r-', linewidth=1.5, label='原始RRT路径')
        
        # 绘制优化后的路径
        x_refined = [p[0] for p in refined_path]
        y_refined = [p[1] for p in refined_path]
        plt.plot(x_refined, y_refined, 'b-', linewidth=2.5, label='优化后的路径')
        
        # 绘制起点和终点
        plt.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='起点')
        plt.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10, label='终点')
        
        # 在优化后路径上的几个点绘制车辆
        num_vehicles = min(10, len(refined_path))
        indices = [int(i * (len(refined_path)-1) / (num_vehicles-1)) for i in range(num_vehicles)]
        for idx in indices:
            self.draw_vehicle(plt.gca(), refined_path[idx], color='cyan', alpha=0.3)
        
        plt.legend()
        plt.title('路径优化对比')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    
    def increase_path_density(self, path, min_dist=2.0):
        """
        增加路径点密度，使运动更加平滑
        
        参数:
            path: 原始路径
            min_dist: 相邻点之间的最小距离
            
        返回:
            list: 增加密度后的路径
        """
        if not path or len(path) < 2:
            return path
        
        dense_path = []
        dense_path.append(path[0])  # 添加起点
        
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            
            # 计算两点之间的距离
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dist = math.sqrt(dx * dx + dy * dy)
            
            # 如果距离大于阈值，添加中间点
            if dist > min_dist:
                # 计算需要添加的点数
                num_points = max(1, int(dist / min_dist))
                
                for j in range(1, num_points):
                    t = j / num_points
                    # 线性插值计算位置
                    x = p1[0] + t * dx
                    y = p1[1] + t * dy
                    
                    # 使用路径方向作为朝向
                    if abs(dx) > 0.01 or abs(dy) > 0.01:
                        theta = math.atan2(dy, dx)
                    else:
                        theta = p1[2]  # 保持原有朝向
                    
                    # 归一化角度
                    while theta > math.pi:
                        theta -= 2 * math.pi
                    while theta < -math.pi:
                        theta += 2 * math.pi
                    
                    # 添加插值点
                    dense_path.append((x, y, theta))
            
            dense_path.append(p2)  # 添加原始路径中的下一个点
        
        return dense_path


class BidirectionalRRT:
    """双向RRT算法实现"""
    
    class Node:
        """RRT树节点"""
        def __init__(self, x, y, theta, parent=None, from_start=True):
            self.x = x
            self.y = y
            self.theta = theta
            self.parent = parent
            self.from_start = from_start  # True表示起点树，False表示终点树
    
    def __init__(self, collision_checker, vehicle_model, env=None, 
                 step_size=2.0, max_steer=math.pi/4,
                 goal_bias=0.2, vehicle_length=5.0, vehicle_width=2.0,
                 debug=False):
        """
        初始化双向RRT规划器
        
        参数:
            collision_checker: 碰撞检查器对象
            vehicle_model: 车辆模型对象
            env: 环境对象（可选）
            step_size: RRT扩展步长
            max_steer: 最大转向角
            goal_bias: 采样目标点的概率
            vehicle_length: 车辆长度
            vehicle_width: 车辆宽度
            debug: 启用调试输出
        """
        self.collision_checker = collision_checker
        self.vehicle_model = vehicle_model
        self.env = env
        self.step_size = step_size
        self.max_steer = max_steer
        self.goal_bias = goal_bias
        self.vehicle_length = vehicle_length
        self.vehicle_width = vehicle_width
        self.debug = debug
        
        # 性能统计
        self.stats = {
            'iterations': 0,
            'nodes_generated': 0,
            'connection_attempts': 0
        }
    
    def plan(self, start, goal, time_step=1.0, max_iterations=3000):
        """
        使用双向RRT规划路径
        
        参数:
            start: 起点位置 (x, y, theta)
            goal: 终点位置 (x, y, theta)
            time_step: 路径点之间的时间步长
            max_iterations: 最大迭代次数
            
        返回:
            list: 时间-空间路径，每个点为 (x, y, theta, t)；失败则返回None
        """
        if self.debug:
            print(f"开始双向RRT规划，从 {start} 到 {goal}，最大迭代 {max_iterations}")
        
        # 重置统计信息
        self.stats = {
            'iterations': 0,
            'nodes_generated': 0,
            'connection_attempts': 0
        }
        
        # 创建起点和终点树
        start_tree = [self.Node(start[0], start[1], start[2], None, True)]
        goal_tree = [self.Node(goal[0], goal[1], goal[2], None, False)]
        
        # 设置采样的起点和终点
        self.start = start
        self.goal = goal
        
        # 尝试找到连接点
        connection = None
        
        # 主循环
        iterations = 0
        while iterations < max_iterations and connection is None:
            iterations += 1
            self.stats['iterations'] = iterations
            
            # 动态调整目标偏置
            adaptive_goal_bias = min(0.5, self.goal_bias + 0.3 * (iterations / max_iterations))
            
            # 生成随机配置
            rand_config = self._random_config(adaptive_goal_bias)
            
            # 从起点树向随机配置扩展
            new_start_node = self._extend_tree(start_tree, rand_config)
            if new_start_node:
                # 尝试连接到终点树
                goal_connect_node = self._try_connect(new_start_node, goal_tree)
                if goal_connect_node:
                    connection = (new_start_node, goal_connect_node)
                    break
            
            # 从终点树向随机配置扩展
            new_goal_node = self._extend_tree(goal_tree, rand_config)
            if new_goal_node:
                # 尝试连接到起点树
                start_connect_node = self._try_connect(new_goal_node, start_tree)
                if start_connect_node:
                    connection = (start_connect_node, new_goal_node)
                    break
            
            # 周期性尝试直接连接树
            if iterations % 50 == 0:
                closest_pair = self._find_closest_nodes(start_tree, goal_tree)
                if closest_pair:
                    start_node, goal_node = closest_pair
                    if self._check_path(start_node, goal_node):
                        connection = (start_node, goal_node)
                        break
            
            # 进度报告
            if self.debug and iterations % 500 == 0:
                print(f"完成 {iterations} 次迭代，起点树: {len(start_tree)} 个节点，"
                      f"终点树: {len(goal_tree)} 个节点")
        
        # 检查是否找到路径
        if connection:
            if self.debug:
                print(f"双向RRT在 {iterations} 次迭代后找到路径")
            
            # 提取完整路径
            path_nodes = self._extract_path(connection[0], connection[1], 
                                          start_tree, goal_tree)
            
            # 平滑路径
            path_nodes = self._smooth_path(path_nodes)
            
            # 转换为时间-空间路径
            time_path = [(p.x, p.y, p.theta, i * time_step) 
                        for i, p in enumerate(path_nodes)]
            
            return time_path
        
        # 如果没有找到完整路径，尝试返回最接近目标的部分路径
        if self.debug:
            print(f"双向RRT在 {max_iterations} 次迭代后未能找到完整路径")
        
        # 找到起点树中最接近目标的节点
        closest_to_goal = min(start_tree, 
                             key=lambda n: self._distance((n.x, n.y), (goal[0], goal[1])))
        
        # 如果足够接近目标，返回部分路径
        if self._distance((closest_to_goal.x, closest_to_goal.y), (goal[0], goal[1])) < 20.0:
            if self.debug:
                print("返回最接近目标的部分路径")
            
            # 提取部分路径
            partial_path = self._extract_partial_path(closest_to_goal, goal)
            time_path = [(p.x, p.y, p.theta, i * time_step) for i, p in enumerate(partial_path)]
            
            return time_path
        
        return None  # 未找到路径
    
    def _random_config(self, goal_bias):
        """
        生成随机配置，使用自适应采样策略
        
        参数:
            goal_bias: 采样目标点的概率
            
        返回:
            tuple: 随机配置 (x, y, theta)
        """
        # 目标偏置 - 直接采样目标点
        if random.random() < goal_bias:
            return (self.goal[0], self.goal[1], self.goal[2])
        
        # 漏斗形采样区域
        if random.random() < 0.7:
            # 沿起点到终点方向采样
            ratio = random.random()
            center_x = self.start[0] + ratio * (self.goal[0] - self.start[0])
            center_y = self.start[1] + ratio * (self.goal[1] - self.start[1])
            
            # 接近终点时减小扩散范围
            spread = max(5.0, 40.0 * (1.0 - ratio))
            x = center_x + random.uniform(-spread, spread)
            y = center_y + random.uniform(-spread, spread)
            
            # 角度偏向目标方向
            dx = self.goal[0] - self.start[0]
            dy = self.goal[1] - self.start[1]
            base_angle = math.atan2(dy, dx) if (dx != 0 or dy != 0) else 0
            theta = base_angle + random.uniform(-math.pi/2, math.pi/2)
        else:
            # 全局采样
            margin = 50  # 扩展区域
            x_min = min(self.start[0], self.goal[0]) - margin
            x_max = max(self.start[0], self.goal[0]) + margin
            y_min = min(self.start[1], self.goal[1]) - margin
            y_max = max(self.start[1], self.goal[1]) + margin
            
            # 确保坐标在环境范围内(如果环境有定义)
            if self.env:
                if hasattr(self.env, 'width') and hasattr(self.env, 'height'):
                    x_min = max(0, x_min)
                    x_max = min(self.env.width, x_max)
                    y_min = max(0, y_min)
                    y_max = min(self.env.height, y_max)
            
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            theta = random.uniform(0, 2 * math.pi)
        
        return (x, y, theta)
    
    def _extend_tree(self, tree, rand_config):
        """
        向随机配置扩展树
        
        参数:
            tree: 要扩展的树
            rand_config: 随机配置
            
        返回:
            Node: 如果扩展成功则返回新节点，否则返回None
        """
        # 找到最近节点
        nearest = min(tree, key=lambda n: self._node_distance(n, rand_config))
        
        # 计算方向和距离
        dx = rand_config[0] - nearest.x
        dy = rand_config[1] - nearest.y
        dist = math.sqrt(dx*dx + dy*dy)
        
        # 如果距离太小，返回
        if dist < 0.1:
            return None
        
        # 确定步长 - 在障碍物附近减小步长
        obstacle_dist = self._get_obstacle_distance(nearest.x, nearest.y)
        adaptive_step = min(dist, max(self.step_size * 0.5, 
                                    min(self.step_size * 2.5, obstacle_dist * 0.7)))
        
        # 计算新位置
        ratio = adaptive_step / dist
        new_x = nearest.x + dx * ratio
        new_y = nearest.y + dy * ratio
        
        # 计算新朝向 - 平滑转向
        target_theta = math.atan2(dy, dx)
        angle_diff = (target_theta - nearest.theta + math.pi) % (2 * math.pi) - math.pi
        
        # 限制转向角
        if abs(angle_diff) > self.max_steer:
            angle_diff = math.copysign(self.max_steer, angle_diff)
        
        new_theta = (nearest.theta + angle_diff) % (2 * math.pi)
        
        # 检查状态和路径有效性
        if self._is_state_valid(new_x, new_y, new_theta):
            if self._check_intermediate_path(nearest, new_x, new_y, new_theta):
                # 创建新节点并添加到树中
                new_node = self.Node(new_x, new_y, new_theta, nearest, nearest.from_start)
                tree.append(new_node)
                self.stats['nodes_generated'] += 1
                return new_node
        
        return None
    
    def _is_state_valid(self, x, y, theta):
        """
        检查状态是否有效（无碰撞）
        
        参数:
            x, y, theta: 位置和朝向
            
        返回:
            bool: 状态是否有效
        """
        if self.collision_checker:
            return self.collision_checker.is_state_valid(x, y, theta)
        elif self.env and hasattr(self.env, 'is_state_valid'):
            return self.env.is_state_valid(x, y, theta)
        elif self.env and hasattr(self.env, 'grid'):
            # 简单的网格碰撞检测
            grid_x, grid_y = int(x), int(y)
            if 0 <= grid_x < self.env.width and 0 <= grid_y < self.env.height:
                return self.env.grid[grid_x, grid_y] == 0
            return False
        else:
            # 如果没有碰撞检测方法,假设所有状态有效(不建议)
            return True
    
    def _try_connect(self, node, target_tree):
        """
        尝试连接节点到目标树
        
        参数:
            node: 要连接的节点
            target_tree: 目标树
            
        返回:
            Node: 如果连接成功则返回目标树中的连接节点，否则返回None
        """
        self.stats['connection_attempts'] += 1
        
        # 找到目标树中最近的节点
        nearest = min(target_tree, key=lambda n: self._node_distance(n, (node.x, node.y, node.theta)))
        
        # 计算距离
        dist = self._node_distance(node, (nearest.x, nearest.y, nearest.theta))
        
        # 根据距离决定连接策略
        if dist < self.step_size * 2:  # 如果距离近，尝试直接连接
            if self._check_path(node, nearest):
                return nearest
        elif dist < self.step_size * 6:  # 如果距离适中，尝试多步连接
            # 计算步数
            steps = max(1, min(3, int(dist / self.step_size)))
            
            # 检查带中间步骤的路径
            start = (node.x, node.y, node.theta)
            end = (nearest.x, nearest.y, nearest.theta)
            
            if self._check_path_with_steps(start, end, steps):
                return nearest
        
        return None
    
    def _check_path(self, start_node, end_node):
        """
        检查两个节点之间的路径是否有效
        
        参数:
            start_node: 起始节点
            end_node: 终止节点
            
        返回:
            bool: 路径是否有效
        """
        # 计算距离
        dx = end_node.x - start_node.x
        dy = end_node.y - start_node.y
        dist = math.sqrt(dx*dx + dy*dy)
        
        # 确定检查点数量
        num_checks = max(3, int(dist / (self.step_size * 0.5)))
        
        # 检查中间点
        for i in range(1, num_checks):
            ratio = i / num_checks
            x = start_node.x + dx * ratio
            y = start_node.y + dy * ratio
            
            # 平滑插值角度
            angle_diff = (end_node.theta - start_node.theta + math.pi) % (2 * math.pi) - math.pi
            theta = (start_node.theta + angle_diff * ratio) % (2 * math.pi)
            
            if not self._is_state_valid(x, y, theta):
                return False
        
        return True
    
    def _check_path_with_steps(self, start, end, steps):
        """
        检查多步路径
        
        参数:
            start: 起始配置 (x, y, theta)
            end: 终止配置 (x, y, theta)
            steps: 步数
            
        返回:
            bool: 路径是否有效
        """
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        d_theta = (end[2] - start[2] + math.pi) % (2 * math.pi) - math.pi
        
        for i in range(1, steps + 1):
            ratio = i / (steps + 1)
            x = start[0] + dx * ratio
            y = start[1] + dy * ratio
            theta = (start[2] + d_theta * ratio) % (2 * math.pi)
            
            if not self._is_state_valid(x, y, theta):
                return False
        
        return True
    
    def _check_intermediate_path(self, node, new_x, new_y, new_theta):
        """
        检查从节点到新位置的中间路径是否有效
        
        参数:
            node: 起始节点
            new_x, new_y, new_theta: 新位置
            
        返回:
            bool: 路径是否有效
        """
        # 简单线性插值检查
        dx = new_x - node.x
        dy = new_y - node.y
        dist = math.sqrt(dx*dx + dy*dy)
        
        # 确定检查点数量
        num_checks = max(2, int(dist / (self.step_size * 0.3)))
        
        # 检查中间点
        for i in range(1, num_checks):
            ratio = i / num_checks
            x = node.x + dx * ratio
            y = node.y + dy * ratio
            
            # 角度插值
            angle_diff = (new_theta - node.theta + math.pi) % (2 * math.pi) - math.pi
            theta = (node.theta + angle_diff * ratio) % (2 * math.pi)
            
            if not self._is_state_valid(x, y, theta):
                return False
        
        return True
    
    def _find_closest_nodes(self, tree1, tree2):
        """
        找到两棵树之间最接近的节点对
        
        参数:
            tree1, tree2: 要搜索的树
            
        返回:
            tuple: (node1, node2) 节点对，如果未找到则返回None
        """
        min_dist = float('inf')
        closest_pair = None
        
        # 为了效率，只检查一部分节点
        max_sample_size = 100
        sample_size1 = min(len(tree1), max_sample_size)
        sample_size2 = min(len(tree2), max_sample_size)
        
        tree1_sample = random.sample(tree1, sample_size1) if len(tree1) > sample_size1 else tree1
        tree2_sample = random.sample(tree2, sample_size2) if len(tree2) > sample_size2 else tree2
        
        # 找到最接近的节点对
        for n1 in tree1_sample:
            for n2 in tree2_sample:
                dist = self._node_distance(n1, (n2.x, n2.y, n2.theta))
                if dist < min_dist:
                    min_dist = dist
                    closest_pair = (n1, n2)
        
        # 只有当距离低于阈值时才返回
        if min_dist < self.step_size * 6:
            return closest_pair
        return None
    
    def _extract_path(self, start_node, end_node, start_tree, goal_tree):
        """
        提取完整路径
        
        参数:
            start_node, end_node: 连接节点
            start_tree, goal_tree: 树
            
        返回:
            list: 完整路径节点
        """
        # 确定哪个节点属于哪棵树
        if start_node.from_start:
            # 从起点树到连接点的路径
            path_to_connector = self._get_path_to_node(start_node, start_tree)
            # 从连接点到终点的路径
            path_from_connector = self._get_path_to_node(end_node, goal_tree, reverse=True)
        else:
            # 从起点树到连接点的路径
            path_to_connector = self._get_path_to_node(end_node, start_tree)
            # 从连接点到终点的路径
            path_from_connector = self._get_path_to_node(start_node, goal_tree, reverse=True)
        
        # 合并路径
        complete_path = path_to_connector + path_from_connector
        return complete_path
    
    def _get_path_to_node(self, node, tree, reverse=False):
        """
        获取从根到节点的路径
        
        参数:
            node: 目标节点
            tree: 包含节点的树
            reverse: 如果为True，返回从节点到根的路径
            
        返回:
            list: 路径节点
        """
        path = []
        current = node
        
        # 回溯到根
        while current:
            path.append(current)
            current = current.parent
        
        # 如果从叶到根，不需要反转
        # 如果从根到叶，需要反转
        if not reverse:
            path.reverse()
        
        return path
    
    def _extract_partial_path(self, closest_node, goal):
        """
        当找不到完整路径时提取部分路径
        
        参数:
            closest_node: 最接近目标的节点
            goal: 目标配置
            
        返回:
            list: 部分路径节点
        """
        # 获取从根到最近节点的路径
        partial_path = self._get_path_to_node(closest_node, None, reverse=False)
        
        # 尝试添加最后一段到目标
        try_connect_to_goal = True
        
        if try_connect_to_goal:
            # 创建目标节点
            goal_node = self.Node(goal[0], goal[1], goal[2])
            
            # 检查从最近节点到目标的路径是否有效
            if self._check_path(closest_node, goal_node):
                partial_path.append(goal_node)
        
        return partial_path
    
    def _smooth_path(self, path):
        """
        通过移除冗余节点平滑路径
        
        参数:
            path: 原始路径
            
        返回:
            list: 平滑后的路径
        """
        if len(path) <= 2:
            return path
        
        # 初始化平滑路径
        smoothed = [path[0]]
        i = 0
        
        while i < len(path) - 1:
            current = path[i]
            
            # 尝试跳过中间节点
            max_look_ahead = min(len(path) - 1, i + 10)
            advanced = False
            
            for j in range(max_look_ahead, i, -1):
                if j < len(path) and self._check_path(current, path[j]):
                    smoothed.append(path[j])
                    i = j
                    advanced = True
                    break
            
            if not advanced:
                # 如果不能跳过，添加下一个节点
                i += 1
                if i < len(path):
                    smoothed.append(path[i])
        
        # 确保目标在路径中
        if smoothed[-1] != path[-1]:
            smoothed.append(path[-1])
        
        return smoothed
    
    def _node_distance(self, node, config):
        """
        计算节点和配置之间的距离
        
        参数:
            node: 节点
            config: 配置 (x, y, theta)
            
        返回:
            float: 综合距离（位置 + 方向）
        """
        dx = node.x - config[0]
        dy = node.y - config[1]
        d_pos = math.sqrt(dx*dx + dy*dy)
        
        # 角度差，归一化到 [-π,π]
        d_theta = abs((node.theta - config[2] + math.pi) % (2 * math.pi) - math.pi)
        
        # 综合距离，位置为主导，角度为次要因素
        return d_pos + 0.3 * self.vehicle_length * d_theta
    
    def _distance(self, p1, p2):
        """
        计算两点之间的欧几里得距离
        
        参数:
            p1, p2: 点 (x, y)
            
        返回:
            float: 距离
        """
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _get_obstacle_distance(self, x, y):
        """
        获取到最近障碍物的距离（如果可用）
        
        参数:
            x, y: 位置
            
        返回:
            float: 到最近障碍物的距离，或默认值
        """
        try:
            if self.collision_checker and hasattr(self.collision_checker, 'get_obstacle_distance'):
                return self.collision_checker.get_obstacle_distance(x, y)
            elif self.env and hasattr(self.env, 'get_distance_to_obstacles'):
                return self.env.get_distance_to_obstacles(x, y)
            else:
                return self.vehicle_length * 3  # 默认安全距离
        except:
            return self.vehicle_length * 3


if __name__ == "__main__":
    # 简单的测试代码
    from environment import OpenPitMineEnv
    
    # 创建测试环境
    print("创建测试环境...")
    env = OpenPitMineEnv(map_size=(100, 100))
    
    # 添加障碍物
    obstacles = [
        (20, 20, 30, 10),   # 主障碍区域
        (60, 50, 20, 20),   # 障碍区2
        (30, 70, 40, 10),   # 障碍区3
        (10, 40, 5, 30)     # 狭窄通道
    ]
    
    for obs in obstacles:
        env.add_obstacle(obs)
    
    # 添加装载点和卸载点
    env.add_loading_point((20, 40))
    env.add_unloading_point((70, 85))
    
    # 创建RRT规划器
    print("初始化RRT规划器...")
    planner = RRTPlanner(
        env, 
        vehicle_length=6.0, 
        vehicle_width=3.0, 
        turning_radius=8.0,
        step_size=0.6,
        grid_resolution=0.25
    )
    
    # 启用调试模式
    planner.debug = True
    
    # 规划一条路径
    start = (5, 5, 0)
    goal = (80, 80, math.pi/4)
    
    print(f"规划从 {start} 到 {goal} 的路径...")
    path = planner.plan_path(start, goal)
    
    if path:
        print(f"成功找到路径，点数: {len(path)}")
        
        # 可视化
        planner.visualize_path_refinement(path, path, env)
    else:
        print("路径规划失败")