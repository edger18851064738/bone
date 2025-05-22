import math
import numpy as np
from collections import defaultdict
from scipy.spatial import KDTree  # 用于加速空间查询
import time

from RRT import RRTPlanner  # 导入RRT规划器

class OptimizedBackbonePathNetwork:
    """优化后的主干路径网络，支持智能连接点生成和高效路径管理"""
    
    def __init__(self, env):
        """初始化主干路径网络"""
        self.env = env
        self.paths = {}               # 路径字典 {path_id: path_data}
        self.nodes = {}               # 节点字典 {node_id: node_data}
        self.path_graph = {}          # 路径连接图，用于路由
        self.connections = {}         # 连接点字典
        
        # 空间索引优化
        self.connection_kdtree = None  # 连接点KD树
        self.path_point_kdtree = None  # 路径点KD树
        self.spatial_index_dirty = True  # 空间索引是否需要重建
        
        # 路径质量评估
        self.path_quality_cache = {}  # 路径质量缓存
        self.quality_weights = {
            'length': 0.3,
            'smoothness': 0.25,
            'turning_count': 0.2,
            'clearance': 0.15,
            'traffic_compatibility': 0.1
        }
        
        # 层次结构管理
        self.path_hierarchy = {
            'primary': [],     # 主要路径
            'secondary': [],   # 次要路径
            'auxiliary': []    # 辅助路径
        }
        
        # 性能统计
        self.performance_stats = {
            'generation_time': 0,
            'optimization_time': 0,
            'query_time': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # 规划器缓存
        self.planner = None
    
    def generate_network(self, connection_spacing=10, quality_threshold=0.6):
        """
        生成完整的主干路径网络
        
        Args:
            connection_spacing: 连接点间距
            quality_threshold: 路径质量阈值
            
        Returns:
            dict: 路径字典
        """
        start_time = time.time()
        
        print("开始生成优化的骨干路径网络...")
        
        # 1. 识别关键点
        key_points = self._identify_key_points()
        print(f"识别到 {len(key_points)} 个关键点")
        
        # 2. 生成关键点间路径
        self._generate_paths_between_key_points(key_points, quality_threshold)
        print(f"生成了 {len(self.paths)} 条初始路径")
        
        # 3. 智能生成连接点
        self._generate_intelligent_connection_points(connection_spacing)
        print(f"生成了 {len(self.connections)} 个连接点")
        
        # 4. 优化所有路径
        self._optimize_all_paths_advanced()
        print("路径优化完成")
        
        # 5. 建立路径层次结构
        self._build_path_hierarchy()
        print("路径层次结构建立完成")
        
        # 6. 构建路径连接图
        self._build_path_graph()
        print("路径连接图构建完成")
        
        # 7. 构建空间索引
        self._build_spatial_indexes()
        print("空间索引构建完成")
        
        self.performance_stats['generation_time'] = time.time() - start_time
        print(f"骨干网络生成完成，耗时: {self.performance_stats['generation_time']:.2f}秒")
        
        return self.paths
    
    def _identify_key_points(self):
        """识别所有关键点"""
        key_points = []
        
        # 添加装载点
        for i, point in enumerate(self.env.loading_points):
            pos = self._ensure_3d_point(point)
            key_points.append({
                "position": pos, 
                "type": "loading_point",
                "id": f"L{i}",
                "priority": 3,  # 高优先级
                "capacity": 5   # 假设容量
            })
        
        # 添加卸载点
        for i, point in enumerate(self.env.unloading_points):
            pos = self._ensure_3d_point(point)
            key_points.append({
                "position": pos, 
                "type": "unloading_point",
                "id": f"U{i}",
                "priority": 3,  # 高优先级
                "capacity": 5
            })
        
        # 添加停车区
        parking_areas = getattr(self.env, 'parking_areas', [])
        for i, point in enumerate(parking_areas):
            pos = self._ensure_3d_point(point)
            key_points.append({
                "position": pos, 
                "type": "parking",
                "id": f"P{i}",
                "priority": 1,  # 低优先级
                "capacity": 10
            })
        
        return key_points
    
    def _generate_paths_between_key_points(self, key_points, quality_threshold=0.6):
        """生成关键点之间的高质量路径"""
        if not self.planner:
            self.planner = self._create_planner()
        
        path_candidates = []  # 存储所有候选路径
        
        # 为每对关键点生成多条候选路径
        for i, start_point in enumerate(key_points):
            for j, end_point in enumerate(key_points):
                if i != j:
                    start_pos = start_point["position"]
                    end_pos = end_point["position"]
                    path_id = f"{start_point['id']}_{end_point['id']}"
                    
                    print(f"规划路径: {path_id}")
                    
                    # 尝试生成多条候选路径
                    candidates = self._generate_path_candidates(
                        start_pos, end_pos, num_candidates=3
                    )
                    
                    best_path = None
                    best_quality = 0
                    
                    for candidate in candidates:
                        if candidate:
                            quality = self._evaluate_path_quality(candidate)
                            if quality > best_quality and quality >= quality_threshold:
                                best_quality = quality
                                best_path = candidate
                    
                    if best_path:
                        # 存储路径信息
                        self.paths[path_id] = {
                            'start': start_point,
                            'end': end_point,
                            'path': best_path,
                            'length': self._calculate_path_length(best_path),
                            'capacity': self._estimate_path_capacity(best_path),
                            'traffic_flow': 0,
                            'speed_limit': self._calculate_speed_limit(best_path),
                            'quality_score': best_quality,
                            'optimized': False,
                            'hierarchy_level': self._determine_hierarchy_level(start_point, end_point),
                            'last_updated': time.time()
                        }
                        print(f"  成功生成高质量路径，质量评分: {best_quality:.2f}")
                    else:
                        print(f"  无法生成满足质量要求的路径")
    
    def _generate_path_candidates(self, start, goal, num_candidates=3):
        """生成多条候选路径"""
        candidates = []
        
        for attempt in range(num_candidates):
            # 调整RRT参数以获得不同的路径
            max_iterations = 3000 + attempt * 1000
            path = self.planner.plan_path(start, goal, max_iterations=max_iterations)
            
            if path and len(path) > 1:
                candidates.append(path)
        
        return candidates
    
    def _evaluate_path_quality(self, path):
        """
        综合评估路径质量
        
        Args:
            path: 路径点列表
            
        Returns:
            float: 质量评分 (0-1)
        """
        if not path or len(path) < 2:
            return 0
        
        # 检查缓存
        path_key = self._get_path_cache_key(path)
        if path_key in self.path_quality_cache:
            self.performance_stats['cache_hits'] += 1
            return self.path_quality_cache[path_key]
        
        self.performance_stats['cache_misses'] += 1
        
        # 1. 长度评分 (越短越好)
        length = self._calculate_path_length(path)
        direct_distance = self._calculate_distance(path[0], path[-1])
        length_score = direct_distance / (length + 0.1)  # 避免除零
        length_score = min(1.0, length_score)
        
        # 2. 平滑度评分
        smoothness_score = self._evaluate_path_smoothness(path)
        
        # 3. 转弯次数评分
        turning_score = self._evaluate_turning_complexity(path)
        
        # 4. 障碍物间隙评分
        clearance_score = self._evaluate_path_clearance(path)
        
        # 5. 交通兼容性评分
        traffic_score = self._evaluate_traffic_compatibility(path)
        
        # 综合评分
        quality_score = (
            self.quality_weights['length'] * length_score +
            self.quality_weights['smoothness'] * smoothness_score +
            self.quality_weights['turning_count'] * turning_score +
            self.quality_weights['clearance'] * clearance_score +
            self.quality_weights['traffic_compatibility'] * traffic_score
        )
        
        # 缓存结果
        self.path_quality_cache[path_key] = quality_score
        
        return quality_score
    
    def _evaluate_path_smoothness(self, path):
        """评估路径平滑度"""
        if len(path) < 3:
            return 1.0
        
        total_curvature = 0
        valid_segments = 0
        
        for i in range(1, len(path) - 1):
            prev = path[i-1]
            curr = path[i]
            next_p = path[i+1]
            
            # 计算曲率
            curvature = self._calculate_curvature(prev, curr, next_p)
            total_curvature += curvature
            valid_segments += 1
        
        if valid_segments == 0:
            return 1.0
        
        avg_curvature = total_curvature / valid_segments
        # 将曲率转换为平滑度评分 (曲率越小越平滑)
        smoothness = math.exp(-avg_curvature * 2)  # 指数衰减
        
        return min(1.0, smoothness)
    
    def _calculate_curvature(self, p1, p2, p3):
        """计算三点的曲率"""
        # 使用向量叉积计算曲率
        v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        # 计算向量长度
        len_v1 = np.linalg.norm(v1)
        len_v2 = np.linalg.norm(v2)
        
        if len_v1 < 0.001 or len_v2 < 0.001:
            return 0
        
        # 归一化向量
        v1_norm = v1 / len_v1
        v2_norm = v2 / len_v2
        
        # 计算角度变化
        dot_product = np.dot(v1_norm, v2_norm)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        angle_change = math.acos(dot_product)
        
        # 曲率 = 角度变化 / 平均段长度
        avg_segment_length = (len_v1 + len_v2) / 2
        curvature = angle_change / (avg_segment_length + 0.001)
        
        return curvature
    
    def _evaluate_turning_complexity(self, path):
        """评估转弯复杂度"""
        if len(path) < 3:
            return 1.0
        
        sharp_turns = 0
        total_segments = len(path) - 2
        
        for i in range(1, len(path) - 1):
            prev = path[i-1]
            curr = path[i]
            next_p = path[i+1]
            
            # 计算转弯角度
            angle = self._calculate_turning_angle(prev, curr, next_p)
            
            # 如果转弯角度大于45度，认为是急转弯
            if angle > math.pi / 4:  # 45度
                sharp_turns += 1
        
        # 转弯评分：急转弯越少越好
        turning_score = 1.0 - (sharp_turns / max(1, total_segments))
        
        return max(0, turning_score)
    
    def _calculate_turning_angle(self, p1, p2, p3):
        """计算转弯角度"""
        v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        len_v1 = np.linalg.norm(v1)
        len_v2 = np.linalg.norm(v2)
        
        if len_v1 < 0.001 or len_v2 < 0.001:
            return 0
        
        cos_angle = np.dot(v1, v2) / (len_v1 * len_v2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        return math.acos(cos_angle)
    
    def _evaluate_path_clearance(self, path):
        """评估路径与障碍物的间隙"""
        if not path:
            return 0
        
        min_clearance = float('inf')
        clearance_samples = 0
        
        # 沿路径采样检查间隙
        for i in range(0, len(path), max(1, len(path) // 20)):  # 采样20个点
            point = path[i]
            clearance = self._calculate_clearance_at_point(point)
            min_clearance = min(min_clearance, clearance)
            clearance_samples += 1
        
        if clearance_samples == 0:
            return 0
        
        # 将间隙转换为评分
        # 假设安全间隙为3个单位，理想间隙为6个单位
        if min_clearance >= 6:
            return 1.0
        elif min_clearance >= 3:
            return 0.5 + 0.5 * (min_clearance - 3) / 3
        else:
            return 0.5 * min_clearance / 3
    
    def _calculate_clearance_at_point(self, point):
        """计算某点到最近障碍物的距离"""
        min_distance = float('inf')
        
        x, y = int(point[0]), int(point[1])
        
        # 在周围区域搜索障碍物
        search_radius = 10
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                check_x, check_y = x + dx, y + dy
                
                if (0 <= check_x < self.env.width and 
                    0 <= check_y < self.env.height and 
                    self.env.grid[check_x, check_y] == 1):  # 障碍物
                    
                    distance = math.sqrt(dx*dx + dy*dy)
                    min_distance = min(min_distance, distance)
        
        return min_distance if min_distance != float('inf') else 10
    
    def _evaluate_traffic_compatibility(self, path):
        """评估路径的交通兼容性"""
        if not path or len(path) < 2:
            return 1.0
        
        # 检查路径是否适合双向交通
        bidirectional_score = 0.8  # 假设大部分路径支持双向
        
        # 检查路径宽度（基于间隙）
        avg_clearance = 0
        for i in range(0, len(path), max(1, len(path) // 10)):
            clearance = self._calculate_clearance_at_point(path[i])
            avg_clearance += clearance
        
        avg_clearance /= min(10, len(path))
        
        # 宽度评分：间隙越大，交通兼容性越好
        width_score = min(1.0, avg_clearance / 8)  # 8个单位为理想宽度
        
        return (bidirectional_score + width_score) / 2
    
    def _get_path_cache_key(self, path):
        """生成路径的缓存键"""
        # 使用路径的起点、终点和几个中间点生成哈希
        if len(path) < 2:
            return str(path)
        
        key_points = [path[0], path[-1]]
        if len(path) > 4:
            key_points.extend([path[len(path)//3], path[2*len(path)//3]])
        
        # 简化坐标并生成字符串
        simplified = []
        for point in key_points:
            simplified.append(f"{point[0]:.1f},{point[1]:.1f}")
        
        return "|".join(simplified)
    
    def _generate_intelligent_connection_points(self, spacing=10):
        """智能生成连接点"""
        self.connections = {}
        connection_id = 0
        
        for path_id, path_data in self.paths.items():
            path = path_data['path']
            if len(path) < 2:
                continue
            
            # 添加起点和终点
            start_id = f"conn_{connection_id}"
            connection_id += 1
            self.connections[start_id] = {
                'position': path[0],
                'type': 'endpoint',
                'paths': [path_id],
                'priority': path_data['start'].get('priority', 1),
                'capacity': 3,
                'quality_score': path_data.get('quality_score', 0.5)
            }
            
            end_id = f"conn_{connection_id}"
            connection_id += 1
            self.connections[end_id] = {
                'position': path[-1],
                'type': 'endpoint',
                'paths': [path_id],
                'priority': path_data['end'].get('priority', 1),
                'capacity': 3,
                'quality_score': path_data.get('quality_score', 0.5)
            }
            
            # 在路径上智能添加中间连接点
            self._add_intermediate_connections(path_id, path, spacing, connection_id)
    
    def _add_intermediate_connections(self, path_id, path, spacing, start_connection_id):
        """在路径上添加中间连接点"""
        if len(path) < spacing * 2:
            return
        
        connection_id = start_connection_id
        
        # 基于路径特征添加连接点
        for i in range(spacing, len(path) - spacing, spacing):
            point = path[i]
            
            # 检查是否为适合的连接点位置
            if self._is_good_connection_point(path, i):
                conn_id = f"conn_{connection_id}"
                connection_id += 1
                
                self.connections[conn_id] = {
                    'position': point,
                    'type': 'intermediate',
                    'paths': [path_id],
                    'path_index': i,
                    'priority': 2,
                    'capacity': 2,
                    'quality_score': self._evaluate_connection_quality(path, i)
                }
    
    def _is_good_connection_point(self, path, index):
        """判断是否为好的连接点位置"""
        if index <= 0 or index >= len(path) - 1:
            return False
        
        point = path[index]
        
        # 检查间隙 - 需要足够空间
        clearance = self._calculate_clearance_at_point(point)
        if clearance < 4:  # 最小间隙要求
            return False
        
        # 检查是否在相对直的路段上
        if index > 0 and index < len(path) - 1:
            angle = self._calculate_turning_angle(path[index-1], point, path[index+1])
            if angle > math.pi / 6:  # 避免在急转弯处设置连接点
                return False
        
        return True
    
    def _evaluate_connection_quality(self, path, index):
        """评估连接点质量"""
        if index <= 0 or index >= len(path) - 1:
            return 0
        
        point = path[index]
        
        # 间隙评分
        clearance = self._calculate_clearance_at_point(point)
        clearance_score = min(1.0, clearance / 6)
        
        # 位置评分 - 靠近路径中间的点评分更高
        position_ratio = index / len(path)
        position_score = 1.0 - abs(position_ratio - 0.5) * 2
        
        # 平滑度评分
        angle = self._calculate_turning_angle(path[index-1], point, path[index+1])
        smoothness_score = 1.0 - angle / math.pi
        
        return (clearance_score + position_score + smoothness_score) / 3
    
    def _build_path_hierarchy(self):
        """建立路径层次结构"""
        self.path_hierarchy = {'primary': [], 'secondary': [], 'auxiliary': []}
        
        for path_id, path_data in self.paths.items():
            quality = path_data.get('quality_score', 0.5)
            start_priority = path_data['start'].get('priority', 1)
            end_priority = path_data['end'].get('priority', 1)
            
            # 根据质量和点的重要性分类
            importance = (quality + (start_priority + end_priority) / 6) / 2
            
            if importance >= 0.8:
                self.path_hierarchy['primary'].append(path_id)
                path_data['hierarchy_level'] = 'primary'
            elif importance >= 0.5:
                self.path_hierarchy['secondary'].append(path_id)
                path_data['hierarchy_level'] = 'secondary'
            else:
                self.path_hierarchy['auxiliary'].append(path_id)
                path_data['hierarchy_level'] = 'auxiliary'
    
    def _optimize_all_paths_advanced(self):
        """高级路径优化"""
        start_time = time.time()
        
        for path_id, path_data in self.paths.items():
            if not path_data['optimized']:
                original_path = path_data['path']
                
                # 多级优化
                optimized_path = self._multi_level_optimization(original_path)
                
                if optimized_path and len(optimized_path) >= 2:
                    path_data['path'] = optimized_path
                    path_data['optimized'] = True
                    path_data['length'] = self._calculate_path_length(optimized_path)
                    path_data['speed_limit'] = self._calculate_speed_limit(optimized_path)
                    path_data['quality_score'] = self._evaluate_path_quality(optimized_path)
        
        self.performance_stats['optimization_time'] = time.time() - start_time
    
    def _multi_level_optimization(self, path):
        """多级路径优化"""
        if not path or len(path) < 3:
            return path
        
        # 第一级：路径简化
        simplified = self._simplify_path_douglas_peucker(path, epsilon=0.8)
        
        # 第二级：平滑处理
        smoothed = self._advanced_smooth_path(simplified)
        
        # 第三级：局部优化
        optimized = self._local_path_optimization(smoothed)
        
        return optimized
    
    def _simplify_path_douglas_peucker(self, path, epsilon=0.5):
        """使用Douglas-Peucker算法简化路径"""
        if len(path) <= 2:
            return path
        
        # 找到距离线段最远的点
        max_distance = 0
        index = 0
        
        start = path[0]
        end = path[-1]
        
        for i in range(1, len(path) - 1):
            distance = self._point_line_distance(path[i], start, end)
            if distance > max_distance:
                max_distance = distance
                index = i
        
        # 如果最大距离大于阈值，递归简化
        if max_distance > epsilon:
            # 递归简化两部分
            first_part = self._simplify_path_douglas_peucker(path[:index+1], epsilon)
            second_part = self._simplify_path_douglas_peucker(path[index:], epsilon)
            
            # 合并结果
            return first_part[:-1] + second_part
        else:
            # 距离小于阈值，只保留端点
            return [path[0], path[-1]]
    
    def _advanced_smooth_path(self, path, iterations=3):
        """高级路径平滑"""
        if len(path) <= 2:
            return path
        
        smoothed = list(path)
        
        for iteration in range(iterations):
            new_smoothed = [smoothed[0]]  # 保留起点
            
            for i in range(1, len(smoothed) - 1):
                prev = smoothed[i-1]
                curr = smoothed[i]
                next_p = smoothed[i+1]
                
                # 自适应平滑权重
                weight = self._calculate_smooth_weight(prev, curr, next_p)
                
                # 加权平均
                x = curr[0] * (1 - weight) + (prev[0] + next_p[0]) * weight / 2
                y = curr[1] * (1 - weight) + (prev[1] + next_p[1]) * weight / 2
                
                # 保持角度或重新计算
                if len(curr) > 2:
                    theta = curr[2]
                else:
                    theta = math.atan2(next_p[1] - prev[1], next_p[0] - prev[0])
                
                # 检查平滑后的点是否有效
                if self._is_valid_position(int(x), int(y)):
                    new_smoothed.append((x, y, theta))
                else:
                    new_smoothed.append(curr)
            
            new_smoothed.append(smoothed[-1])  # 保留终点
            smoothed = new_smoothed
        
        return smoothed
    
    def _calculate_smooth_weight(self, prev, curr, next_p):
        """计算自适应平滑权重"""
        # 基于曲率计算权重 - 曲率越大，平滑权重越大
        curvature = self._calculate_curvature(prev, curr, next_p)
        weight = min(0.8, curvature * 0.3)  # 限制最大权重
        
        return weight
    
    def _local_path_optimization(self, path):
        """局部路径优化"""
        if len(path) < 4:
            return path
        
        optimized = list(path)
        
        # 尝试连接距离较近的非相邻点以缩短路径
        for i in range(len(optimized) - 3):
            for j in range(i + 2, min(i + 6, len(optimized))):  # 检查后续几个点
                if self._can_connect_directly(optimized[i], optimized[j]):
                    # 可以直接连接，移除中间点
                    optimized = optimized[:i+1] + optimized[j:]
                    break
        
        return optimized
    
    def _can_connect_directly(self, p1, p2):
        """检查两点是否可以直接连接"""
        # 检查直线路径是否无碰撞
        return self._is_line_collision_free(p1, p2)
    
    def _is_line_collision_free(self, p1, p2):
        """检查直线是否无碰撞"""
        # 简化的碰撞检测
        steps = int(self._calculate_distance(p1, p2) / 0.5)  # 每0.5单位检查一次
        
        for i in range(steps + 1):
            t = i / max(1, steps)
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            
            if not self._is_valid_position(int(x), int(y)):
                return False
        
        return True
    
    def _build_spatial_indexes(self):
        """构建空间索引用于加速查询"""
        if not self.connections:
            return
        
        # 构建连接点KD树
        connection_points = []
        connection_ids = []
        
        for conn_id, conn_data in self.connections.items():
            pos = conn_data['position']
            connection_points.append([pos[0], pos[1]])
            connection_ids.append(conn_id)
        
        if connection_points:
            self.connection_kdtree = KDTree(connection_points)
            self.connection_ids = connection_ids
        
        # 构建路径点KD树
        path_points = []
        path_info = []  # (path_id, point_index)
        
        for path_id, path_data in self.paths.items():
            path = path_data['path']
            for i, point in enumerate(path):
                path_points.append([point[0], point[1]])
                path_info.append((path_id, i))
        
        if path_points:
            self.path_point_kdtree = KDTree(path_points)
            self.path_point_info = path_info
        
        self.spatial_index_dirty = False
    
    def find_nearest_connection_optimized(self, position, max_distance=5.0, max_candidates=5):
        """使用空间索引优化的最近连接点查找"""
        start_time = time.time()
        
        if self.spatial_index_dirty or self.connection_kdtree is None:
            self._build_spatial_indexes()
        
        if self.connection_kdtree is None:
            self.performance_stats['query_time'] += time.time() - start_time
            return None
        
        # 使用KD树查找候选点
        query_point = [position[0], position[1]]
        distances, indices = self.connection_kdtree.query(
            query_point, 
            k=min(max_candidates, len(self.connection_ids)),
            distance_upper_bound=max_distance
        )
        
        # 处理单个结果的情况
        if not hasattr(distances, '__len__'):
            distances = [distances]
            indices = [indices]
        
        best_connection = None
        best_score = -1
        
        for dist, idx in zip(distances, indices):
            if idx >= len(self.connection_ids) or dist > max_distance:
                continue
            
            conn_id = self.connection_ids[idx]
            conn_data = self.connections[conn_id]
            
            # 计算综合评分
            score = self._calculate_connection_score(conn_data, dist)
            
            if score > best_score:
                best_score = score
                best_connection = conn_data.copy()
                best_connection['id'] = conn_id
                best_connection['distance'] = dist
        
        self.performance_stats['query_time'] += time.time() - start_time
        return best_connection
    
    def _calculate_connection_score(self, conn_data, distance):
        """计算连接点综合评分"""
        # 距离评分 (越近越好)
        distance_score = 1.0 / (1.0 + distance / 10.0)
        
        # 质量评分
        quality_score = conn_data.get('quality_score', 0.5)
        
        # 优先级评分
        priority_score = conn_data.get('priority', 1) / 3.0
        
        # 容量评分
        capacity_score = min(1.0, conn_data.get('capacity', 1) / 5.0)
        
        # 综合评分
        return (distance_score * 0.4 + quality_score * 0.3 + 
                priority_score * 0.2 + capacity_score * 0.1)
    
    def find_accessible_points(self, position, rrt_planner, max_candidates=5, 
                              sampling_step=10, max_distance=20.0):
        """优化版可达点查找"""
        start_time = time.time()
        accessible_points = []
        
        # 首先使用优化的连接点查找
        nearest_connections = self.find_nearest_connection_optimized(
            position, max_distance, max_candidates * 2
        )
        
        if nearest_connections:
            # 将单个结果转换为列表
            if not isinstance(nearest_connections, list):
                nearest_connections = [nearest_connections]
            
            for conn in nearest_connections[:max_candidates]:
                if rrt_planner and rrt_planner.is_path_possible(position, conn['position']):
                    accessible_points.append({
                        'conn_id': conn['id'],
                        'path_id': conn.get('paths', [None])[0],
                        'path_index': conn.get('path_index', 0),
                        'position': conn['position'],
                        'distance': conn['distance'],
                        'type': 'connection',
                        'quality': conn.get('quality_score', 0.5)
                    })
        
        # 如果连接点不足，使用路径点KD树查找
        if len(accessible_points) < max_candidates and self.path_point_kdtree:
            additional_needed = max_candidates - len(accessible_points)
            
            query_point = [position[0], position[1]]
            distances, indices = self.path_point_kdtree.query(
                query_point,
                k=min(additional_needed * 3, len(self.path_point_info)),
                distance_upper_bound=max_distance
            )
            
            if not hasattr(distances, '__len__'):
                distances = [distances]
                indices = [indices]
            
            for dist, idx in zip(distances, indices):
                if idx >= len(self.path_point_info) or dist > max_distance:
                    continue
                
                path_id, point_idx = self.path_point_info[idx]
                if path_id not in self.paths:
                    continue
                
                point = self.paths[path_id]['path'][point_idx]
                
                if rrt_planner and rrt_planner.is_path_possible(position, point):
                    accessible_points.append({
                        'conn_id': None,
                        'path_id': path_id,
                        'path_index': point_idx,
                        'position': point,
                        'distance': dist,
                        'type': 'path_point',
                        'quality': self.paths[path_id].get('quality_score', 0.5)
                    })
                    
                    if len(accessible_points) >= max_candidates:
                        break
        
        # 按质量和距离排序
        accessible_points.sort(key=lambda x: (-x['quality'], x['distance']))
        
        self.performance_stats['query_time'] += time.time() - start_time
        return accessible_points[:max_candidates]
    
    # 保留原有的其他方法，确保兼容性
    def _ensure_3d_point(self, point):
        """确保点坐标有三个元素 (x, y, theta)"""
        if not point:
            return (0, 0, 0)
        if len(point) >= 3:
            return (point[0], point[1], point[2])
        elif len(point) == 2:
            return (point[0], point[1], 0)
        else:
            return (0, 0, 0)
    
    def _calculate_distance(self, pos1, pos2):
        """计算两点之间的欧几里得距离"""
        x1 = pos1[0] if len(pos1) > 0 else 0
        y1 = pos1[1] if len(pos1) > 1 else 0
        x2 = pos2[0] if len(pos2) > 0 else 0
        y2 = pos2[1] if len(pos2) > 1 else 0
        
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    
    def _calculate_path_length(self, path):
        """计算路径总长度"""
        if not path or len(path) < 2:
            return 0
        
        length = 0
        for i in range(len(path) - 1):
            length += self._calculate_distance(path[i], path[i + 1])
        
        return length
    
    def _point_line_distance(self, point, line_start, line_end):
        """计算点到线段的距离"""
        x0, y0 = point[0], point[1]
        x1, y1 = line_start[0], line_start[1]
        x2, y2 = line_end[0], line_end[1]
        
        l2 = (x2 - x1) ** 2 + (y2 - y1) ** 2
        
        if l2 == 0:
            return ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5
        
        t = max(0, min(1, ((x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)) / l2))
        
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        
        return ((x0 - px) ** 2 + (y0 - py) ** 2) ** 0.5
    
    def _is_valid_position(self, x, y):
        """检查位置是否有效（不是障碍物）"""
        if not hasattr(self.env, 'grid'):
            return True
        
        if x < 0 or x >= self.env.width or y < 0 or y >= self.env.height:
            return False
        
        return self.env.grid[x, y] == 0
    
    def _create_planner(self):
        """创建RRT规划器用于生成初始路径"""
        if hasattr(self.env, 'rrt_planner') and self.env.rrt_planner:
            return self.env.rrt_planner
        
        try:
            from RRT import RRTPlanner
            return RRTPlanner(
                self.env, 
                vehicle_length=5.0,
                vehicle_width=2.0,
                turning_radius=5.0,
                step_size=0.8
            )
        except Exception as e:
            print(f"警告: 无法创建RRTPlanner: {e}")
            return None
    
    def _calculate_speed_limit(self, path):
        """基于路径质量计算速度限制"""
        if not path or len(path) < 2:
            return 1.0
        
        # 基于路径平滑度和质量计算速度限制
        quality = self._evaluate_path_quality(path)
        
        # 质量越高，允许的速度越高
        speed_limit = 0.3 + 0.7 * quality
        
        return min(1.0, max(0.3, speed_limit))
    
    def _estimate_path_capacity(self, path):
        """基于路径特征估计容量"""
        if not path:
            return 1
        
        length = self._calculate_path_length(path)
        base_capacity = max(1, int(length / 20))  # 每20单位长度1个容量
        
        # 基于路径质量调整容量
        quality = self._evaluate_path_quality(path)
        capacity_multiplier = 0.5 + quality
        
        return max(1, int(base_capacity * capacity_multiplier))
    
    def get_performance_stats(self):
        """获取性能统计信息"""
        return self.performance_stats.copy()
    
    def get_quality_report(self):
        """获取路径质量报告"""
        if not self.paths:
            return {}
        
        qualities = [data.get('quality_score', 0) for data in self.paths.values()]
        
        return {
            'total_paths': len(self.paths),
            'average_quality': sum(qualities) / len(qualities),
            'min_quality': min(qualities),
            'max_quality': max(qualities),
            'high_quality_paths': len([q for q in qualities if q >= 0.8]),
            'medium_quality_paths': len([q for q in qualities if 0.5 <= q < 0.8]),
            'low_quality_paths': len([q for q in qualities if q < 0.5]),
            'hierarchy_distribution': {
                level: len(paths) for level, paths in self.path_hierarchy.items()
            }
        }

# 保持向后兼容性的类别名
BackbonePathNetwork = OptimizedBackbonePathNetwork