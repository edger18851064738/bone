import math
import numpy as np
from collections import defaultdict

from RRT import RRTPlanner  # 导入RRT规划器

class BackbonePathNetwork:
    """主干路径网络，管理预计算的最优路径
    
    主干路径网络是一种高效的路径管理方法，通过预先计算和优化关键点之间的路径，
    减少实时路径规划的计算负担，并提供更加稳定和可预测的车辆行驶模式。
    
    属性:
        env: 环境对象，提供地图和资源信息
        paths: 路径字典 {path_id: path_data}
        nodes: 节点字典 {node_id: node_data}
        path_graph: 路径连接图，用于路由
    """
    
    def __init__(self, env):
        """初始化主干路径网络
        
        Args:
            env: 环境对象，包含地图数据、关键点和资源信息
        """
        self.env = env
        self.paths = {}               # 路径字典 {path_id: path_data}
        self.nodes = {}               # 节点字典 {node_id: node_data}
        self.path_graph = {}          # 路径连接图，用于路由
        self.connections = {}
        # 规划器缓存
        self.planner = None
    
    def generate_network(self):
        """生成完整的主干路径网络
        
        步骤:
        1. 识别关键点（装载点、卸载点、停车区等）
        2. 为关键点对生成路径
        3. 优化路径（平滑、简化等）
        4. 创建路径连接图用于路由
        
        Returns:
            dict: 路径字典 {path_id: path_data}
        """
        # 1. 识别关键点（装载点、卸载点、停车区等）
        key_points = self._identify_key_points()
        
        # 2. 为关键点对生成路径
        self._generate_paths_between_key_points(key_points)
        
        # 3. 优化路径（平滑、简化等）
        self._optimize_all_paths()
        
        # 4. 创建路径连接图用于路由
        self._build_path_graph()
        self._generate_connection_points()        
        return self.paths

    def _generate_connection_points(self):
        """创建路径连接点"""
        self.connections = {}
        connection_id = 0
        
        # 从每条路径中添加起点和终点作为连接点
        for path_id, path_data in self.paths.items():
            # 起点
            start_pos = path_data['start']['position']
            start_id = f"conn_{connection_id}"
            connection_id += 1
            self.connections[start_id] = {
                'position': start_pos,
                'type': 'endpoint',
                'paths': [path_id]
            }
            
            # 终点
            end_pos = path_data['end']['position']
            end_id = f"conn_{connection_id}"
            connection_id += 1
            self.connections[end_id] = {
                'position': end_pos,
                'type': 'endpoint',
                'paths': [path_id]
            }
    def find_nearest_connection(self, position, max_distance=5.0):
        """查找最靠近给定位置的连接点
        
        Args:
            position: 位置坐标 (x, y) 或 (x, y, theta)
            max_distance: 最大搜索距离
            
        Returns:
            dict: 连接点信息，如果没找到则返回None
        """
        nearest_connection = None
        min_dist = float('inf')
        
        # 遍历所有连接点
        for conn_id, conn_data in self.connections.items():
            conn_pos = conn_data['position']
            dist = self._calculate_distance(position, conn_pos)
            
            if dist < min_dist and dist <= max_distance:
                min_dist = dist
                nearest_connection = conn_data.copy()  # 复制一份避免修改原始数据
                nearest_connection['id'] = conn_id
                nearest_connection['distance'] = dist
        
        return nearest_connection

    def _is_turning_point(self, prev, curr, next_point):
        """判断是否是转弯点，用于添加连接点"""
        # 提取坐标
        x0, y0 = prev[0], prev[1]
        x1, y1 = curr[0], curr[1]
        x2, y2 = next_point[0], next_point[1]
        
        # 计算方向向量
        dx1, dy1 = x1 - x0, y1 - y0
        dx2, dy2 = x2 - x1, y2 - y1
        
        # 归一化向量
        len1 = math.sqrt(dx1*dx1 + dy1*dy1)
        len2 = math.sqrt(dx2*dx2 + dy2*dy2)
        
        if len1 < 0.001 or len2 < 0.001:
            return False
            
        dx1, dy1 = dx1/len1, dy1/len1
        dx2, dy2 = dx2/len2, dy2/len2
        
        # 计算向量夹角余弦值
        cos_angle = dx1*dx2 + dy1*dy2
        
        # 如果夹角大于30度，认为是转弯点
        return cos_angle < 0.866  # cos(30度) ≈ 0.866
    def _is_valid_position(self, x, y):
        """检查位置是否有效（不是障碍物）"""
        if not hasattr(self.env, 'grid'):
            return True
            
        # 检查是否在地图范围内
        if x < 0 or x >= self.env.width or y < 0 or y >= self.env.height:
            return False
            
        # 检查是否是障碍物 (0=可通行, 1=障碍物)
        return self.env.grid[x, y] == 0
    def _identify_key_points(self):
        """识别所有关键点"""
        key_points = []
        
        # 添加装载点
        for i, point in enumerate(self.env.loading_points):
            pos = self._ensure_3d_point(point)
            key_points.append({
                "position": pos, 
                "type": "loading_point",
                "id": f"L{i}"
            })
        
        # 添加卸载点
        for i, point in enumerate(self.env.unloading_points):
            pos = self._ensure_3d_point(point)
            key_points.append({
                "position": pos, 
                "type": "unloading_point",
                "id": f"U{i}"
            })
        
        # 添加停车区 - 确保正确访问
        parking_areas = getattr(self.env, 'parking_areas', [])
        for i, point in enumerate(parking_areas):
            pos = self._ensure_3d_point(point)
            key_points.append({
                "position": pos, 
                "type": "parking",
                "id": f"P{i}"
            })
        
        # 打印调试信息以验证点是否找到
        print(f"识别到 {len(key_points)} 个关键点:")
        for kp in key_points:
            print(f"  {kp['id']} ({kp['type']}): {kp['position']}")
        
        return key_points
    
    def _ensure_3d_point(self, point):
        """确保点坐标有三个元素 (x, y, theta)"""
        if not point:
            return (0, 0, 0)
            
        if len(point) >= 3:
            return (point[0], point[1], point[2])
        elif len(point) == 2:
            return (point[0], point[1], 0)
        else:
            # 处理异常情况
            return (0, 0, 0)
        
    def _generate_paths_between_key_points(self, key_points):
        """生成关键点之间的路径"""
        # 使用RRT规划器生成初始路径
        if not self.planner:
            self.planner = self._create_planner()
        
        for i, start_point in enumerate(key_points):
            for j, end_point in enumerate(key_points):
                if i != j:  # 不为同一点生成路径
                    start_pos = start_point["position"]
                    end_pos = end_point["position"]
                    path_id = f"{start_point['id']}_{end_point['id']}"
                    
                    # 使用RRT规划器生成初始路径
                    print(f"规划路径: {path_id} (从 {start_point['id']} 到 {end_point['id']})")
                    path = self.planner.plan_path(start_pos, end_pos, max_iterations=6000)
                    
                    if path:
                        # 存储路径信息
                        self.paths[path_id] = {
                            'start': start_point,
                            'end': end_point,
                            'path': path,
                            'length': self._calculate_path_length(path),
                            'capacity': self._estimate_path_capacity(path),
                            'traffic_flow': 0,  # 当前流量
                            'speed_limit': self._calculate_speed_limit(path),
                            'optimized': False  # 标记为未优化
                        }
                        print(f"  路径规划成功，路径长度: {len(path)}")
                    else:
                        print(f"  路径规划失败: {path_id}")
    
    def _optimize_all_paths(self):
        """优化所有路径"""
        for path_id, path_data in self.paths.items():
            if not path_data['optimized']:
                path_data['path'] = self._optimize_path(path_data['path'])
                path_data['optimized'] = True
                # 更新路径属性
                path_data['length'] = self._calculate_path_length(path_data['path'])
                path_data['speed_limit'] = self._calculate_speed_limit(path_data['path'])
    
    def _optimize_path(self, path):
        """优化单个路径（平滑和简化）"""
        if not path or len(path) < 2:
            return path
            
        # 1. 路径平滑处理
        smoothed_path = self._smooth_path(path)
        
        # 2. 路径简化（去除冗余点）
        simplified_path = self._simplify_path(smoothed_path)
        
        return simplified_path
    
    def _smooth_path(self, path):
        """路径平滑处理，使用简单的移动平均法"""
        if len(path) <= 2:
            return path
            
        smoothed = [path[0]]  # 保留起点
        
        # 对中间点进行平滑
        for i in range(1, len(path) - 1):
            prev = path[i-1]
            curr = path[i]
            next_p = path[i+1]
            
            # 简单的三点平均平滑
            x = (prev[0] + curr[0] + next_p[0]) / 3
            y = (prev[1] + curr[1] + next_p[1]) / 3
            
            # 保持原有角度或计算新角度
            if len(curr) > 2:
                theta = curr[2]
            else:
                # 根据移动方向计算角度
                theta = math.atan2(next_p[1] - prev[1], next_p[0] - prev[0])
                
            smoothed.append((x, y, theta))
            
        smoothed.append(path[-1])  # 保留终点
        return smoothed
    
    def _simplify_path(self, path):
        """路径简化，去除冗余点
        使用Ramer-Douglas-Peucker算法简化路径，减少路径点数量。
        """
        if len(path) <= 2:
            return path
            
        # 设置简化阈值
        epsilon = 0.5
        
        # 查找最大距离点
        max_dist = 0
        index = 0
        
        start = path[0]
        end = path[-1]
        
        for i in range(1, len(path) - 1):
            dist = self._point_line_distance(path[i], start, end)
            if dist > max_dist:
                max_dist = dist
                index = i
        
        # 如果最大距离大于阈值，则递归简化
        if max_dist > epsilon:
            # 递归简化两部分
            first_part = self._simplify_path(path[:index+1])
            second_part = self._simplify_path(path[index:])
            
            # 合并结果，避免重复中间点
            return first_part[:-1] + second_part
        else:
            # 距离小于阈值，只保留端点
            return [path[0], path[-1]]
    
    def _point_line_distance(self, point, line_start, line_end):
        """计算点到线段的距离"""
        x0, y0 = point[0], point[1]
        x1, y1 = line_start[0], line_start[1]
        x2, y2 = line_end[0], line_end[1]
        
        # 线段长度的平方
        l2 = (x2 - x1) ** 2 + (y2 - y1) ** 2
        
        # 如果线段实际上是一个点，则返回到该点的距离
        if l2 == 0:
            return ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5
            
        # 计算点在线段上的投影位置
        t = max(0, min(1, ((x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)) / l2))
        
        # 投影点
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        
        # 返回点到投影点的距离
        return ((x0 - px) ** 2 + (y0 - py) ** 2) ** 0.5
    
    def _build_path_graph(self):
        """构建路径连接图，用于路由"""
        # 清空现有图
        self.path_graph = {}
        
        # 为每个节点创建条目
        for path_id, path_data in self.paths.items():
            start_id = path_data['start']['id']
            end_id = path_data['end']['id']
            
            # 添加起点到图
            if start_id not in self.path_graph:
                self.path_graph[start_id] = {}
            
            # 添加连接信息
            self.path_graph[start_id][end_id] = {
                'path_id': path_id,
                'length': path_data['length'],
                'capacity': path_data['capacity'],
                'traffic_flow': path_data['traffic_flow']
            }
    
    # 新增方法：查找骨干路径上最近点
    def find_nearest_backbone_point(self, position, max_distance=20.0):
        """查找最靠近给定位置的骨干路径点
        
        Args:
            position: 位置坐标 (x, y) 或 (x, y, theta)
            max_distance: 最大搜索距离
            
        Returns:
            dict: {'path_id': path_id, 'path_index': idx, 'position': point, 'distance': distance}
            或 None (如果没找到)
        """
        nearest_info = None
        min_dist = float('inf')
        
        # 遍历所有骨干路径点
        for path_id, path_data in self.paths.items():
            path = path_data['path']
            
            for idx, point in enumerate(path):
                dist = self._calculate_distance(position, point)
                if dist < min_dist and dist <= max_distance:
                    min_dist = dist
                    nearest_info = {
                        'path_id': path_id,
                        'path_index': idx,
                        'position': point,
                        'distance': dist
                    }
        
        return nearest_info
     
    # 新增方法：获取可达点
    def find_accessible_points(self, position, rrt_planner, max_candidates=5, sampling_step=10, max_distance=20.0):
        """
        查找可通过RRT从当前位置到达的骨干路径点 - 增强版
        
        Args:
            position: 起点位置
            rrt_planner: RRT路径规划器
            max_candidates: 最大候选点数
            sampling_step: 路径采样步长
            max_distance: 最大搜索距离
            
        Returns:
            list: 可达点信息列表，按距离排序
        """
        accessible_points = []
        
        # 首先检查连接点 - 更好的入口点
        connections = self.connections.copy()
        for conn_id, conn_data in connections.items():
            conn_pos = conn_data['position']
            distance = self._calculate_distance(position, conn_pos)
            
            # 如果距离在阈值内，检查是否可达
            if distance < max_distance:
                if rrt_planner.is_path_possible(position, conn_pos):
                    # 创建详细的连接点信息
                    path_id = None
                    path_index = 0
                    
                    # 查找对应的路径ID和索引
                    if 'paths' in conn_data and conn_data['paths']:
                        path_id = conn_data['paths'][0]  # 使用第一条相关路径
                        
                        # 查找连接点在路径中的索引
                        if path_id in self.paths:
                            path = self.paths[path_id]['path']
                            for i, point in enumerate(path):
                                if self._is_same_point(point, conn_pos):
                                    path_index = i
                                    break
                    
                    # 添加到可达点列表
                    accessible_points.append({
                        'conn_id': conn_id,
                        'path_id': path_id,
                        'path_index': path_index,
                        'position': conn_pos,
                        'distance': distance,
                        'type': 'connection'
                    })
        
        # 如果连接点不足，再检查路径点
        if len(accessible_points) < max_candidates:
            # 遍历所有骨干路径
            for path_id, path_data in self.paths.items():
                path = path_data['path']
                
                # 采样路径点（避免检查过多点）
                for idx in range(0, len(path), sampling_step):
                    target = path[idx]
                    distance = self._calculate_distance(position, target)
                    
                    # 如果距离在阈值内，检查是否可达
                    if distance < max_distance and rrt_planner.is_path_possible(position, target):
                        accessible_points.append({
                            'conn_id': None,
                            'path_id': path_id,
                            'path_index': idx,
                            'position': target,
                            'distance': distance,
                            'type': 'path_point'
                        })
                        
                        # 找到足够多的候选点后停止
                        if len(accessible_points) >= max_candidates:
                            break
                
                if len(accessible_points) >= max_candidates:
                    break
        
        # 按距离排序
        return sorted(accessible_points, key=lambda x: x['distance'])

    def get_path_segment(self, path_id, start_index, end_index):
        """
        获取路径的一个段 - 增强版，支持不同路径之间的段
        
        Args:
            path_id (str): 路径ID或路径ID列表(用':'分隔)
            start_index (int): 起始索引
            end_index (int): 结束索引
            
        Returns:
            list or None: 路径段点列表，如果路径不存在则返回None
        """
        if ':' in path_id:
            # 多段路径处理
            path_ids = path_id.split(':')
            if len(path_ids) != 2:
                return None
            
            start_path_id, end_path_id = path_ids
            
            # 获取两条路径
            if start_path_id not in self.paths or end_path_id not in self.paths:
                return None
            
            start_path = self.paths[start_path_id]['path']
            end_path = self.paths[end_path_id]['path']
            
            # 处理同一路径的情况
            if start_path_id == end_path_id:
                return self._get_single_path_segment(start_path_id, start_index, end_index)
            
            # 在主干网络中查找路径
            connecting_paths = self.find_path(start_path_id, end_path_id)
            
            if not connecting_paths:
                return None
            
            # 构建完整路径
            full_path = []
            
            # 添加第一段 - 从start_index到路径末尾
            if start_index < len(start_path):
                full_path.extend(start_path[start_index:])
            
            # 添加中间连接路径 - 跳过第一条路径和最后一条路径
            for i, conn_path_id in enumerate(connecting_paths[1:-1], 1):
                if conn_path_id in self.paths:
                    conn_path = self.paths[conn_path_id]['path']
                    # 如果不是第一段，跳过第一个点以避免重复
                    if i > 1:
                        conn_path = conn_path[1:]
                    full_path.extend(conn_path)
            
            # 添加最后一段 - 从路径开始到end_index
            if end_index < len(end_path):
                end_segment = end_path[:end_index+1]
                # 如果不是第一段且第一个点重复，跳过
                if full_path and self._is_same_point(full_path[-1], end_segment[0]):
                    end_segment = end_segment[1:]
                full_path.extend(end_segment)
            
            return full_path
        else:
            # 单一路径处理
            return self._get_single_path_segment(path_id, start_index, end_index)

    def _get_single_path_segment(self, path_id, start_index, end_index):
        """获取单一路径的段"""
        if path_id not in self.paths:
            return None
            
        path = self.paths[path_id]['path']
        
        if not path:
            return None
            
        # 确保索引有效
        start_index = max(0, min(start_index, len(path) - 1))
        end_index = max(0, min(end_index, len(path) - 1))
        
        # 根据索引大小决定方向
        if start_index <= end_index:
            return path[start_index:end_index + 1]
        else:
            # 如果起始索引大于结束索引，则需要反转路径段
            return list(reversed(path[end_index:start_index + 1]))

    def _is_same_point(self, p1, p2, tolerance=0.5):
        """判断两点是否相同（考虑误差）"""
        if not p1 or not p2:
            return False
            
        # 提取坐标
        x1 = p1[0] if len(p1) > 0 else 0
        y1 = p1[1] if len(p1) > 1 else 0
        x2 = p2[0] if len(p2) > 0 else 0
        y2 = p2[1] if len(p2) > 1 else 0
        
        # 计算距离并判断是否小于容差
        dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        return dist < tolerance
    
    def find_path(self, start_id, end_id):
        """在主干网络中查找从起点到终点的最佳路径
        
        使用Dijkstra算法找到从起点节点到终点节点的最优路径。
        
        Args:
            start_id (str): 起点节点ID
            end_id (str): 终点节点ID
            
        Returns:
            list or None: 路径ID列表，如果没有找到路径则返回None
        """
        if start_id not in self.path_graph or end_id not in self.path_graph:
            return None
            
        # 使用Dijkstra算法查找最短路径
        # 初始化
        distances = {node: float('inf') for node in self.path_graph}
        distances[start_id] = 0
        previous = {node: None for node in self.path_graph}
        unvisited = list(self.path_graph.keys())
        
        while unvisited:
            # 找到距离最小的未访问节点
            current = min(unvisited, key=lambda node: distances[node])
            
            # 如果当前节点是终点或无法到达更多节点，则结束
            if current == end_id or distances[current] == float('inf'):
                break
                
            unvisited.remove(current)
            
            # 检查当前节点的所有邻居
            for neighbor, edge_data in self.path_graph[current].items():
                # 计算通过当前节点到达邻居的距离
                # 可以考虑路径长度、流量等因素
                weight = edge_data['length'] * (1 + edge_data['traffic_flow'] / max(1, edge_data['capacity']))
                distance = distances[current] + weight
                
                # 如果找到更短的路径，则更新
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current
        
        # 重建路径
        if end_id not in previous or previous[end_id] is None:
            return None  # 没有找到路径
            
        # 构建节点ID序列
        path = []
        current = end_id
        while current:
            path.append(current)
            current = previous[current]
        
        # 反转以获得从起点到终点的顺序
        path.reverse()
        
        # 转换为路径ID序列
        path_ids = []
        for i in range(len(path) - 1):
            start_node = path[i]
            end_node = path[i + 1]
            path_ids.append(self.path_graph[start_node][end_node]['path_id'])
        
        return path_ids
    
    def get_path_segment(self, path_id, start_index, end_index):
        """获取路径的一个段
        
        从路径中提取指定索引范围的段。
        
        Args:
            path_id (str): 路径ID
            start_index (int): 起始索引
            end_index (int): 结束索引
            
        Returns:
            list or None: 路径段点列表，如果路径不存在则返回None
        """
        if path_id not in self.paths:
            return None
            
        path = self.paths[path_id]['path']
        
        if not path:
            return None
            
        # 确保索引有效
        start_index = max(0, min(start_index, len(path) - 1))
        end_index = max(0, min(end_index, len(path) - 1))
        
        # 根据索引大小决定方向
        if start_index <= end_index:
            return path[start_index:end_index + 1]
        else:
            # 如果起始索引大于结束索引，则需要反转路径段
            return list(reversed(path[end_index:start_index + 1]))
    
    def _calculate_distance(self, pos1, pos2):
        """计算两点之间的欧几里得距离"""
        # 确保坐标至少有x,y两个值
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
    
    def _calculate_speed_limit(self, path):
        """计算路径的速度限制
        
        基于路径的弯曲程度计算适合的速度限制。
        
        Args:
            path (list): 路径点列表
            
        Returns:
            float: 速度限制 (0.3-1.0)
        """
        # 简单实现：基于路径的弯曲程度计算速度限制
        # 更复杂的实现可以考虑坡度、路面类型等
        
        if not path or len(path) < 3:
            return 1.0  # 默认最高速度
        
        # 计算各段的角度变化
        angle_changes = []
        for i in range(1, len(path) - 1):
            prev = path[i - 1]
            curr = path[i]
            next_p = path[i + 1]
            
            # 计算前一段的方向
            dx1 = curr[0] - prev[0]
            dy1 = curr[1] - prev[1]
            
            # 计算后一段的方向
            dx2 = next_p[0] - curr[0]
            dy2 = next_p[1] - curr[1]
            
            # 计算角度变化（使用点积和叉积）
            if (dx1 == 0 and dy1 == 0) or (dx2 == 0 and dy2 == 0):
                continue  # 跳过重合点
                
            # 归一化向量
            mag1 = (dx1 ** 2 + dy1 ** 2) ** 0.5
            mag2 = (dx2 ** 2 + dy2 ** 2) ** 0.5
            
            dx1, dy1 = dx1 / mag1, dy1 / mag1
            dx2, dy2 = dx2 / mag2, dy2 / mag2
            
            # 计算点积（夹角余弦）
            dot_product = dx1 * dx2 + dy1 * dy2
            
            # 确保数值在有效范围内（浮点误差可能导致越界）
            dot_product = max(-1.0, min(1.0, dot_product))
            
            # 角度变化（弧度）
            angle_change = math.acos(dot_product)
            angle_changes.append(angle_change)
        
        # 无角度变化
        if not angle_changes:
            return 1.0
            
        # 计算最大角度变化和平均角度变化
        max_change = max(angle_changes)
        avg_change = sum(angle_changes) / len(angle_changes)
        
        # 基于最大角度变化和平均角度变化计算速度限制
        # 角度变化越大，速度限制越低
        max_speed = 1.0
        min_speed = 0.3
        
        # 使用最大角度变化作为主要因素，平均角度变化作为次要因素
        speed_limit = max_speed - (max_change / math.pi) * 0.5 - (avg_change / math.pi) * 0.2
        
        # 确保速度在有效范围内
        return max(min_speed, min(max_speed, speed_limit))
    
    def _estimate_path_capacity(self, path):
        """估计路径的通行能力
        
        基于路径长度和弯曲程度估计路径的最大车辆容量。
        
        Args:
            path (list): 路径点列表
            
        Returns:
            int: 估计的路径容量
        """
        # 简单实现：基于路径长度和弯曲程度估计
        # 可以根据需要改进
        
        # 基本容量（每单位长度的车辆数）
        base_capacity = 0.05  # 每20单位长度1辆车
        
        # 路径长度
        length = self._calculate_path_length(path)
        
        # 基于长度的基础容量
        capacity = length * base_capacity
        
        # 考虑弯曲程度的影响
        if len(path) > 2:
            # 计算平均角度变化
            total_angle_change = 0
            count = 0
            
            for i in range(1, len(path) - 1):
                prev = path[i - 1]
                curr = path[i]
                next_p = path[i + 1]
                
                # 计算角度变化（简化版）
                dx1 = curr[0] - prev[0]
                dy1 = curr[1] - prev[1]
                dx2 = next_p[0] - curr[0]
                dy2 = next_p[1] - curr[1]
                
                if (dx1 == 0 and dy1 == 0) or (dx2 == 0 and dy2 == 0):
                    continue
                
                # 计算夹角余弦值
                dot_product = dx1 * dx2 + dy1 * dy2
                mag1 = (dx1 ** 2 + dy1 ** 2) ** 0.5
                mag2 = (dx2 ** 2 + dy2 ** 2) ** 0.5
                
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = max(-1.0, min(1.0, cos_angle))  # 防止数值误差
                
                angle_change = math.acos(cos_angle)
                total_angle_change += angle_change
                count += 1
            
            # 计算平均角度变化
            if count > 0:
                avg_angle_change = total_angle_change / count
                
                # 弯曲程度系数（越弯曲，容量越低）
                curvature_factor = 1.0 - (avg_angle_change / math.pi) * 0.5
                capacity *= curvature_factor
        
        # 最小容量
        return max(1, int(capacity))
    


    def _create_planner(self):
        """创建RRT规划器用于生成初始路径
        
        使用双向RRT算法进行路径规划。
        
        Returns:
            RRTPlanner: 路径规划器
        """
        # 如果有RRT规划器可用，则使用它
        if hasattr(self.env, 'rrt_planner') and self.env.rrt_planner:
            return self.env.rrt_planner
            
        # 否则创建新的RRTPlanner (不再使用SimplePlanner作为后备)
        try:
            return RRTPlanner(
                self.env, 
                vehicle_length=5.0,  # 可根据实际需求调整这些参数
                vehicle_width=2.0,
                turning_radius=5.0,
                step_size=0.8
            )
        except Exception as e:
            print(f"警告: 无法创建RRTPlanner，将使用SimplePlanner作为后备: {e}")
            return SimplePlanner(self.env)
    
    def update_traffic_flow(self, path_id, delta=1):
        """更新路径的交通流量
        
        增加或减少路径的交通流量，用于交通管理。
        
        Args:
            path_id (str): 路径ID
            delta (int, optional): 流量变化值，正数表示增加，负数表示减少
        """
        if path_id in self.paths:
            self.paths[path_id]['traffic_flow'] += delta
            # 确保流量不为负
            self.paths[path_id]['traffic_flow'] = max(0, self.paths[path_id]['traffic_flow'])
            
            # 更新路径图
            self._update_path_graph_edge(path_id)
    
    def _update_path_graph_edge(self, path_id):
        """更新路径图中的边权重
        
        根据路径流量更新路径图中的边权重。
        
        Args:
            path_id (str): 路径ID
        """
        if path_id in self.paths:
            path_data = self.paths[path_id]
            start_id = path_data['start']['id']
            end_id = path_data['end']['id']
            
            if start_id in self.path_graph and end_id in self.path_graph[start_id]:
                self.path_graph[start_id][end_id]['traffic_flow'] = path_data['traffic_flow']
    
    def visualize(self, scene=None):
        """可视化主干路径网络
        
        如果提供场景对象，则在场景中绘制路径网络。
        
        Args:
            scene: 可视化场景对象（可选）
        """
        # 实现取决于可视化系统
        pass


class SimplePlanner:
    """简单的路径规划器，用于临时替代 PathPlanner"""
    
    def __init__(self, env):
        """初始化简单规划器
        
        Args:
            env: 环境对象
        """
        self.env = env
    
    def plan_path(self, start, goal, max_iterations=1000):
        """规划从起点到终点的路径
        
        简单实现，仅生成直线路径并避开障碍物。
        
        Args:
            start (tuple): 起点坐标
            goal (tuple): 终点坐标
            max_iterations: 最大迭代次数（此实现未使用）
            
        Returns:
            list: 路径点列表
        """
        # 提取坐标
        start_x = start[0] if len(start) > 0 else 0
        start_y = start[1] if len(start) > 1 else 0
        goal_x = goal[0] if len(goal) > 0 else 0
        goal_y = goal[1] if len(goal) > 1 else 0
        
        # 计算方向角度
        theta = math.atan2(goal_y - start_y, goal_x - start_x)
        
        # 路径点数量（根据距离确定）
        distance = ((goal_x - start_x) ** 2 + (goal_y - start_y) ** 2) ** 0.5
        num_points = max(2, int(distance / 5))  # 每5个单位一个点
        
        # 生成路径点
        path = []
        for i in range(num_points):
            t = i / (num_points - 1)
            x = start_x + t * (goal_x - start_x)
            y = start_y + t * (goal_y - start_y)
            path.append((x, y, theta))
        
        return path
        
    def is_path_possible(self, start, goal):
        """检查是否可以在起点和终点之间规划路径
        
        简单实现，检查两点连线是否穿越障碍物。
        
        Args:
            start: 起点坐标
            goal: 终点坐标
            
        Returns:
            bool: 是否可能有路径
        """
        # 提取坐标
        start_x = int(start[0]) if len(start) > 0 else 0
        start_y = int(start[1]) if len(start) > 1 else 0
        goal_x = int(goal[0]) if len(goal) > 0 else 0
        goal_y = int(goal[1]) if len(goal) > 1 else 0
        
        # 检查两点是否在障碍物上
        if (not self._is_valid_position(start_x, start_y) or 
            not self._is_valid_position(goal_x, goal_y)):
            return False
        
        # 使用Bresenham算法检查线段是否穿越障碍物
        x, y = start_x, start_y
        dx = abs(goal_x - start_x)
        dy = abs(goal_y - start_y)
        sx = 1 if start_x < goal_x else -1
        sy = 1 if start_y < goal_y else -1
        err = dx - dy
        
        while x != goal_x or y != goal_y:
            if not self._is_valid_position(x, y):
                return False
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return True
    
    def _is_valid_position(self, x, y):
        """检查位置是否有效（不是障碍物）"""
        if not hasattr(self.env, 'grid'):
            return True
            
        # 检查是否在地图范围内
        if x < 0 or x >= self.env.width or y < 0 or y >= self.env.height:
            return False
            
        # 检查是否是障碍物 (0=可通行, 1=障碍物)
        return self.env.grid[x, y] == 0