

import math
import time
import threading
from collections import defaultdict, OrderedDict
from RRT import RRTPlanner

class BackboneInterface:
    """骨干路径接口点"""
    def __init__(self, interface_id, position, direction, backbone_path_id, 
                 path_index, access_difficulty=0.0):
        self.interface_id = interface_id
        self.position = position  # (x, y, theta)
        self.direction = direction  # 接口进入方向
        self.backbone_path_id = backbone_path_id
        self.path_index = path_index  # 在骨干路径中的索引
        self.access_difficulty = access_difficulty  # 接入难度评估
        self.usage_count = 0  # 使用次数统计
        self.is_occupied = False  # 是否被占用
        self.occupied_by = None  # 被哪个车辆占用
        self.reservation_time = None  # 预约时间
        
    def reserve(self, vehicle_id, duration=30):
        """预约接口"""
        self.is_occupied = True
        self.occupied_by = vehicle_id
        self.reservation_time = time.time() + duration
        
    def release(self):
        """释放接口"""
        self.is_occupied = False
        self.occupied_by = None
        self.reservation_time = None
        
    def is_available(self):
        """检查接口是否可用"""
        if not self.is_occupied:
            return True
        
        # 检查预约是否过期
        if self.reservation_time and time.time() > self.reservation_time:
            self.release()
            return True
            
        return False


class SimplifiedBackbonePathNetwork:
    """
    简化的骨干路径网络 - 带接口系统优化版
    """
    
    def __init__(self, env):
        self.env = env
        self.backbone_paths = {}  # 骨干路径字典 {path_id: path_data}
        self.special_points = {   # 特殊点分类
            'loading': [],
            'unloading': [],
            'parking': []
        }
        
        # 骨干接口系统
        self.backbone_interfaces = {}  # {interface_id: BackboneInterface}
        self.path_interfaces = defaultdict(list)  # {path_id: [interface_ids]}
        self.interface_spacing = 10  # 接口间距（路径点数）
        
        # 空间索引用于快速查找接口
        self.interface_spatial_index = {}  # 简化的空间索引
        
        # 路径查找索引
        self.paths_to_target = defaultdict(list)  # {(target_type, target_id): [path_ids]}
        self.paths_from_source = defaultdict(list)  # {(source_type, source_id): [path_ids]}
        
        # 规划器
        self.planner = None
        
        # 性能统计
        self.stats = {
            'total_paths': 0,
            'total_interfaces': 0,
            'interface_usage': defaultdict(int),
            'generation_time': 0,
            'average_path_length': 0,
            'path_usage_count': defaultdict(int),
            'total_usage': 0
        }
        
        print("初始化带接口系统的骨干路径网络")
    
    def generate_backbone_network(self, quality_threshold=0.6, interface_spacing=10):
        """
        生成骨干路径网络并创建接口点
        """
        self.interface_spacing = interface_spacing
        
        start_time = time.time()
        print("开始生成骨干路径网络...")
        
        try:
            # 1. 读取和分类特殊点
            self._load_special_points()
            print(f"特殊点统计: 装载点{len(self.special_points['loading'])}个, "
                  f"卸载点{len(self.special_points['unloading'])}个, "
                  f"停车点{len(self.special_points['parking'])}个")
            
            # 2. 创建规划器
            if not self.planner:
                self.planner = self._create_planner()
                
            if not self.planner:
                print("无法创建路径规划器，骨干网络生成失败")
                return False
            
            # 3. 生成骨干路径
            self._generate_backbone_paths(quality_threshold)
            
            # 4. 生成骨干接口点
            self._generate_backbone_interfaces()
            
            # 5. 建立查找索引
            self._build_path_indexes()
            self._build_interface_spatial_index()
            
            # 6. 统计信息
            generation_time = time.time() - start_time
            self.stats['generation_time'] = generation_time
            self.stats['total_paths'] = len(self.backbone_paths)
            self.stats['total_interfaces'] = len(self.backbone_interfaces)
            
            if self.backbone_paths:
                total_length = sum(path_data['length'] for path_data in self.backbone_paths.values())
                self.stats['average_path_length'] = total_length / len(self.backbone_paths)
            
            print(f"骨干路径网络生成完成!")
            print(f"- 总路径数: {len(self.backbone_paths)}")
            print(f"- 总接口数: {len(self.backbone_interfaces)}")
            print(f"- 接口间距: {self.interface_spacing} 个路径点")
            print(f"- 生成耗时: {generation_time:.2f}秒")
            print(f"- 平均路径长度: {self.stats['average_path_length']:.1f}")
            
            return True
            
        except Exception as e:
            print(f"生成骨干路径网络失败: {e}")
            return False
    
    def _generate_backbone_interfaces(self):
        """为每条骨干路径生成接口点"""
        print("正在生成骨干路径接口点...")
        
        total_interfaces = 0
        
        for path_id, path_data in self.backbone_paths.items():
            path = path_data['path']
            if len(path) < self.interface_spacing:
                continue
                
            interfaces_for_path = []
            
            # 从路径起点开始，每隔interface_spacing个点设置一个接口
            for i in range(0, len(path), self.interface_spacing):
                if i >= len(path):
                    break
                    
                # 计算接口方向
                direction = self._calculate_interface_direction(path, i)
                
                # 创建接口
                interface_id = f"{path_id}_interface_{i // self.interface_spacing}"
                interface = BackboneInterface(
                    interface_id=interface_id,
                    position=path[i],
                    direction=direction,
                    backbone_path_id=path_id,
                    path_index=i,
                    access_difficulty=self._evaluate_interface_access_difficulty(path, i)
                )
                
                # 存储接口
                self.backbone_interfaces[interface_id] = interface
                interfaces_for_path.append(interface_id)
                total_interfaces += 1
            
            # 确保路径终点也有接口
            if len(path) > 0:
                last_index = len(path) - 1
                last_interface_id = f"{path_id}_interface_end"
                if last_interface_id not in self.backbone_interfaces:
                    direction = self._calculate_interface_direction(path, last_index)
                    last_interface = BackboneInterface(
                        interface_id=last_interface_id,
                        position=path[last_index],
                        direction=direction,
                        backbone_path_id=path_id,
                        path_index=last_index,
                        access_difficulty=self._evaluate_interface_access_difficulty(path, last_index)
                    )
                    
                    self.backbone_interfaces[last_interface_id] = last_interface
                    interfaces_for_path.append(last_interface_id)
                    total_interfaces += 1
            
            self.path_interfaces[path_id] = interfaces_for_path
        
        print(f"成功生成 {total_interfaces} 个骨干接口点")
    
    def _calculate_interface_direction(self, path, index):
        """计算接口的方向角"""
        if index < len(path) - 1:
            # 使用当前点到下一点的方向
            dx = path[index + 1][0] - path[index][0]
            dy = path[index + 1][1] - path[index][1]
            if abs(dx) > 0.001 or abs(dy) > 0.001:
                return math.atan2(dy, dx)
        
        # 如果是最后一个点或者方向向量为零，使用点本身的朝向
        return path[index][2] if len(path[index]) > 2 else 0.0
    
    def _evaluate_interface_access_difficulty(self, path, index):
        """评估接口的接入难度"""
        difficulty = 0.0
        
        # 基于周围障碍物密度
        x, y = int(path[index][0]), int(path[index][1])
        obstacle_count = 0
        search_radius = 5
        
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                check_x, check_y = x + dx, y + dy
                if (0 <= check_x < self.env.width and 
                    0 <= check_y < self.env.height and
                    hasattr(self.env, 'grid') and
                    self.env.grid[check_x, check_y] == 1):
                    obstacle_count += 1
        
        difficulty += obstacle_count * 0.1
        
        # 基于路径曲率（转弯越急难度越高）
        if index > 0 and index < len(path) - 1:
            curvature = self._calculate_path_curvature(path, index)
            difficulty += curvature * 5
        
        return difficulty
    
    def _calculate_path_curvature(self, path, index):
        """计算路径在指定点的曲率"""
        if index <= 0 or index >= len(path) - 1:
            return 0.0
        
        p1 = path[index - 1]
        p2 = path[index]
        p3 = path[index + 1]
        
        # 使用三点法计算曲率
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        len_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
        len_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if len_v1 < 0.001 or len_v2 < 0.001:
            return 0.0
        
        cos_angle = (v1[0]*v2[0] + v1[1]*v2[1]) / (len_v1 * len_v2)
        cos_angle = max(-1, min(1, cos_angle))
        angle = math.acos(cos_angle)
        
        # 曲率 = 角度变化 / 平均段长
        avg_length = (len_v1 + len_v2) / 2
        return angle / (avg_length + 0.001)
    
    def _build_interface_spatial_index(self):
        """建立接口的空间索引"""
        self.interface_spatial_index = {}
        
        # 简化的网格索引
        grid_size = 20  # 网格大小
        
        for interface_id, interface in self.backbone_interfaces.items():
            x, y = interface.position[0], interface.position[1]
            grid_x = int(x // grid_size)
            grid_y = int(y // grid_size)
            grid_key = (grid_x, grid_y)
            
            if grid_key not in self.interface_spatial_index:
                self.interface_spatial_index[grid_key] = []
            
            self.interface_spatial_index[grid_key].append(interface_id)
    
    def find_nearest_interface(self, position, target_type, target_id, max_distance=50, debug=True):
        """
        找到最近的可用骨干接口 - 带调试版本
        """
        if debug:
            candidates = self.debug_find_interface(position, target_type, target_id, max_distance)
            if not candidates:
                return None
        else:
            # 原有逻辑
            target_paths = self.find_paths_to_target(target_type, target_id)
            if not target_paths:
                return None
            
            nearby_interfaces = self._find_nearby_interfaces(position, max_distance)
            
            candidate_interfaces = []
            for interface_id in nearby_interfaces:
                interface = self.backbone_interfaces[interface_id]
                if any(path_data['id'] == interface.backbone_path_id for path_data in target_paths):
                    if interface.is_available():
                        candidate_interfaces.append(interface)
            
            if not candidate_interfaces:
                return None
            
            candidates = candidate_interfaces
        
        # 评估接口并选择最佳的
        best_interface = None
        best_score = float('inf')
        
        for interface in candidates:
            score = self._evaluate_interface_score(position, interface, target_type, target_id)
            if debug:
                print(f"接口 {interface.interface_id} 评分: {score:.2f}")
            if score < best_score:
                best_score = score
                best_interface = interface
        
        if debug and best_interface:
            print(f"✅ 选择接口: {best_interface.interface_id}")
        
        return best_interface
        
    def _find_nearby_interfaces(self, position, max_distance):
        """使用空间索引查找附近的接口"""
        nearby_interfaces = []
        x, y = position[0], position[1]
        grid_size = 20
        
        # 计算搜索半径（网格数）
        search_radius = max(1, int(max_distance // grid_size)) + 1
        center_grid_x = int(x // grid_size)
        center_grid_y = int(y // grid_size)
        
        # 搜索周围的网格
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                grid_key = (center_grid_x + dx, center_grid_y + dy)
                if grid_key in self.interface_spatial_index:
                    for interface_id in self.interface_spatial_index[grid_key]:
                        interface = self.backbone_interfaces[interface_id]
                        distance = self._calculate_distance(position, interface.position)
                        if distance <= max_distance:
                            nearby_interfaces.append(interface_id)
        
        return nearby_interfaces
    
    def _evaluate_interface_score(self, vehicle_position, interface, target_type, target_id):
        """
        评估接口的综合得分（越小越好）
        """
        # 1. 距离成本
        distance = self._calculate_distance(vehicle_position, interface.position)
        distance_cost = distance
        
        # 2. 接入难度成本
        access_difficulty_cost = interface.access_difficulty * 10
        
        # 3. 角度对齐成本
        vehicle_heading = vehicle_position[2] if len(vehicle_position) > 2 else 0
        interface_heading = interface.direction
        angle_diff = abs(vehicle_heading - interface_heading)
        angle_diff = min(angle_diff, 2*math.pi - angle_diff)
        angle_cost = angle_diff * 5
        
        # 4. 使用频率成本（避免热点）
        usage_cost = interface.usage_count * 2
        
        # 5. 剩余骨干路径长度（越长越好，所以成本为负）
        remaining_path_cost = -self._calculate_remaining_path_length(interface, target_type, target_id)
        
        total_score = (distance_cost + access_difficulty_cost + angle_cost + 
                      usage_cost + remaining_path_cost * 0.1)
        
        return total_score
    
    def _calculate_remaining_path_length(self, interface, target_type, target_id):
        """计算从接口到目标的剩余路径长度"""
        backbone_path_data = self.backbone_paths.get(interface.backbone_path_id)
        if not backbone_path_data:
            return 0
        
        backbone_path = backbone_path_data['path']
        remaining_length = 0
        
        # 从接口位置到路径终点的长度
        for i in range(interface.path_index, len(backbone_path) - 1):
            remaining_length += self._calculate_distance(backbone_path[i], backbone_path[i + 1])
        
        return remaining_length
    
    def get_path_from_position_to_target_via_interface(self, current_position, target_type, target_id):
        """
        通过骨干接口获取从当前位置到目标的完整路径
        
        Args:
            current_position: 当前位置
            target_type: 目标类型
            target_id: 目标ID
            
        Returns:
            tuple: (完整路径, 路径结构信息)
        """
        # 1. 查找最佳骨干接口
        best_interface = self.find_nearest_interface(current_position, target_type, target_id)
        
        if not best_interface:
            print(f"未找到到 {target_type}_{target_id} 的可用骨干接口")
            return None, None
        
        # 2. 预约接口
        # 这里可以添加车辆ID参数来正确预约
        best_interface.reserve("vehicle_temp", duration=60)  # 预约1分钟
        
        # 3. 规划从当前位置到接口的接入路径
        interface_position = best_interface.position
        
        # 如果当前位置就在接口附近，直接使用骨干路径
        if self._calculate_distance(current_position, interface_position) < 3.0:
            print(f"当前位置接近骨干接口 {best_interface.interface_id}，直接使用骨干路径")
            
            # 获取从接口到目标的骨干路径段
            backbone_segment = self._get_backbone_segment_from_interface(best_interface, target_type, target_id)
            
            return backbone_segment, {
                'type': 'backbone_only',
                'interface_id': best_interface.interface_id,
                'backbone_path_id': best_interface.backbone_path_id,
                'backbone_utilization': 1.0,
                'access_length': 0,
                'backbone_length': len(backbone_segment) if backbone_segment else 0,
                'total_length': len(backbone_segment) if backbone_segment else 0
            }
        
        # 4. 规划接入路径
        print(f"规划到骨干接口 {best_interface.interface_id} 的接入路径")
        access_path = self.planner.plan_path(current_position, interface_position, max_iterations=3000)
        
        if not access_path or len(access_path) < 2:
            print("接入路径规划失败")
            best_interface.release()  # 释放接口预约
            return None, None
        
        # 5. 获取骨干路径段
        backbone_segment = self._get_backbone_segment_from_interface(best_interface, target_type, target_id)
        
        if not backbone_segment:
            print("获取骨干路径段失败")
            best_interface.release()
            return None, None
        
        # 6. 拼接路径
        complete_path = self._merge_paths(access_path, backbone_segment)
        
        if not complete_path:
            print("路径拼接失败")
            best_interface.release()
            return None, None
        
        # 7. 构建路径结构信息
        structure = {
            'type': 'interface_assisted',
            'interface_id': best_interface.interface_id,
            'backbone_path_id': best_interface.backbone_path_id,
            'access_path': access_path,
            'backbone_path': backbone_segment,
            'backbone_utilization': len(backbone_segment) / len(complete_path),
            'access_length': len(access_path),
            'backbone_length': len(backbone_segment),
            'total_length': len(complete_path)
        }
        
        # 8. 更新使用统计
        best_interface.usage_count += 1
        self.stats['interface_usage'][best_interface.interface_id] += 1
        self.stats['total_usage'] += 1
        
        print(f"接口辅助路径生成成功: 总长度{len(complete_path)}, "
              f"骨干利用率{structure['backbone_utilization']:.2f}, "
              f"使用接口{best_interface.interface_id}")
        
        return complete_path, structure
    
    def _get_backbone_segment_from_interface(self, interface, target_type, target_id):
        """从接口获取到目标的骨干路径段"""
        backbone_path_data = self.backbone_paths.get(interface.backbone_path_id)
        if not backbone_path_data:
            return None
        
        backbone_path = backbone_path_data['path']
        
        # 从接口位置开始到路径终点的段
        if interface.path_index < len(backbone_path):
            return backbone_path[interface.path_index:]
        
        return None
    
    def release_interface(self, interface_id):
        """释放接口"""
        if interface_id in self.backbone_interfaces:
            self.backbone_interfaces[interface_id].release()
    
    def get_interface_statistics(self):
        """获取接口使用统计"""
        stats = {
            'total_interfaces': len(self.backbone_interfaces),
            'available_interfaces': sum(1 for i in self.backbone_interfaces.values() if i.is_available()),
            'occupied_interfaces': sum(1 for i in self.backbone_interfaces.values() if i.is_occupied),
            'interface_usage': dict(self.stats['interface_usage']),
            'most_used_interface': None,
            'least_used_interface': None
        }
        
        if self.backbone_interfaces:
            most_used = max(self.backbone_interfaces.values(), key=lambda x: x.usage_count)
            least_used = min(self.backbone_interfaces.values(), key=lambda x: x.usage_count)
            stats['most_used_interface'] = {
                'id': most_used.interface_id,
                'usage_count': most_used.usage_count
            }
            stats['least_used_interface'] = {
                'id': least_used.interface_id,
                'usage_count': least_used.usage_count
            }
        
        return stats
    
    # 保持原有接口兼容性
    def get_path_from_position_to_target(self, current_position, target_type, target_id):
        """兼容原有接口，内部调用新的接口系统"""
        return self.get_path_from_position_to_target_via_interface(
            current_position, target_type, target_id
        )
    
    # ... 其他原有方法保持不变 ...
    def _load_special_points(self):
        """载入和分类特殊点"""
        # 装载点
        self.special_points['loading'] = []
        for i, point in enumerate(self.env.loading_points):
            self.special_points['loading'].append({
                'id': i,
                'type': 'loading',
                'position': self._ensure_3d_point(point),
                'capacity': 5  # 默认容量
            })
        
        # 卸载点
        self.special_points['unloading'] = []
        for i, point in enumerate(self.env.unloading_points):
            self.special_points['unloading'].append({
                'id': i,
                'type': 'unloading', 
                'position': self._ensure_3d_point(point),
                'capacity': 5
            })
        
        # 停车点
        self.special_points['parking'] = []
        parking_areas = getattr(self.env, 'parking_areas', [])
        for i, point in enumerate(parking_areas):
            self.special_points['parking'].append({
                'id': i,
                'type': 'parking',
                'position': self._ensure_3d_point(point),
                'capacity': 10
            })
    
    def _generate_backbone_paths(self, quality_threshold):
        """生成特殊点之间的骨干路径"""
        path_count = 0
        
        print("生成装载点 ↔ 卸载点路径...")
        # 装载点 → 卸载点 (双向)
        for loading_point in self.special_points['loading']:
            for unloading_point in self.special_points['unloading']:
                # 正向路径
                path_id = f"L{loading_point['id']}_to_U{unloading_point['id']}"
                if self._generate_single_path(loading_point, unloading_point, path_id, quality_threshold):
                    path_count += 1
                
                # 反向路径
                reverse_path_id = f"U{unloading_point['id']}_to_L{loading_point['id']}"
                if self._generate_single_path(unloading_point, loading_point, reverse_path_id, quality_threshold):
                    path_count += 1
        
        print("生成装载点 ↔ 停车点路径...")
        # 装载点 → 停车点 (双向)
        for loading_point in self.special_points['loading']:
            for parking_point in self.special_points['parking']:
                # 正向路径
                path_id = f"L{loading_point['id']}_to_P{parking_point['id']}"
                if self._generate_single_path(loading_point, parking_point, path_id, quality_threshold):
                    path_count += 1
                
                # 反向路径
                reverse_path_id = f"P{parking_point['id']}_to_L{loading_point['id']}"
                if self._generate_single_path(parking_point, loading_point, reverse_path_id, quality_threshold):
                    path_count += 1
        
        print("生成卸载点 ↔ 停车点路径...")
        # 卸载点 → 停车点 (双向)
        for unloading_point in self.special_points['unloading']:
            for parking_point in self.special_points['parking']:
                # 正向路径
                path_id = f"U{unloading_point['id']}_to_P{parking_point['id']}"
                if self._generate_single_path(unloading_point, parking_point, path_id, quality_threshold):
                    path_count += 1
                
                # 反向路径
                reverse_path_id = f"P{parking_point['id']}_to_U{unloading_point['id']}"
                if self._generate_single_path(parking_point, unloading_point, reverse_path_id, quality_threshold):
                    path_count += 1
        
        print(f"成功生成 {path_count} 条骨干路径")
    
    def _generate_single_path(self, start_point, end_point, path_id, quality_threshold):
        """生成单条骨干路径"""
        try:
            start_pos = start_point['position']
            end_pos = end_point['position']
            
            # 使用RRT规划器生成路径
            path = self.planner.plan_path(start_pos, end_pos, max_iterations=5000)
            
            if not path or len(path) < 2:
                print(f"路径 {path_id} 规划失败")
                return False
            
            # 评估路径质量
            quality = self._evaluate_path_quality(path)
            if quality < quality_threshold:
                print(f"路径 {path_id} 质量不达标: {quality:.2f}")
                return False
            
            # 存储路径
            self.backbone_paths[path_id] = {
                'id': path_id,
                'start_point': start_point,
                'end_point': end_point,
                'path': path,
                'length': self._calculate_path_length(path),
                'quality': quality,
                'usage_count': 0,
                'created_time': time.time()
            }
            
            return True
            
        except Exception as e:
            print(f"生成路径 {path_id} 失败: {e}")
            return False
    
    def _build_path_indexes(self):
        """建立路径查找索引 - 修复版本"""
        self.paths_to_target.clear()
        self.paths_from_source.clear()
        
        print("开始建立路径索引...")
        
        for path_id, path_data in self.backbone_paths.items():
            start_point = path_data['start_point']
            end_point = path_data['end_point']
            
            # 按终点建立索引
            target_key = (end_point['type'], end_point['id'])
            self.paths_to_target[target_key].append(path_data)  # 注意：这里存储的是path_data，不是path_id
            
            # 按起点建立索引
            source_key = (start_point['type'], start_point['id'])
            self.paths_from_source[source_key].append(path_data)
            
            print(f"索引路径 {path_id}: {source_key} -> {target_key}")
        
        print(f"路径索引建立完成，目标索引: {len(self.paths_to_target)} 个")
    
    def find_paths_to_target(self, target_type, target_id):
        """查找到指定目标的所有骨干路径"""
        target_key = (target_type, target_id)
        return self.paths_to_target.get(target_key, [])
    
    def identify_target_point(self, target_position):
        """识别目标位置是否为特殊点"""
        tolerance = 2.0  # 位置容差
        
        # 检查是否为装载点
        for point in self.special_points['loading']:
            if self._calculate_distance(target_position, point['position']) < tolerance:
                return 'loading', point['id']
        
        # 检查是否为卸载点
        for point in self.special_points['unloading']:
            if self._calculate_distance(target_position, point['position']) < tolerance:
                return 'unloading', point['id']
        
        # 检查是否为停车点
        for point in self.special_points['parking']:
            if self._calculate_distance(target_position, point['position']) < tolerance:
                return 'parking', point['id']
        
        return None, None
    
    def _merge_paths(self, access_path, backbone_path):
        """合并接入路径和骨干路径"""
        if not access_path or not backbone_path:
            return None
        
        # 移除重复的连接点
        merged_path = list(access_path)
        
        # 如果接入路径的终点和骨干路径的起点很接近，跳过骨干路径的起点
        if (len(access_path) > 0 and len(backbone_path) > 0 and
            self._calculate_distance(access_path[-1], backbone_path[0]) < 1.0):
            merged_path.extend(backbone_path[1:])
        else:
            merged_path.extend(backbone_path)
        
        return merged_path
    
    def _evaluate_path_quality(self, path):
        """评估路径质量"""
        if not path or len(path) < 2:
            return 0.0
        
        # 简化的质量评估
        # 1. 长度效率
        path_length = self._calculate_path_length(path)
        direct_distance = self._calculate_distance(path[0], path[-1])
        
        if direct_distance < 0.1:
            length_efficiency = 1.0
        else:
            length_efficiency = min(1.0, direct_distance / path_length)
        
        # 2. 平滑度
        smoothness = self._evaluate_path_smoothness(path)
        
        # 综合评分
        quality = length_efficiency * 0.6 + smoothness * 0.4
        
        return quality
    
    def _evaluate_path_smoothness(self, path):
        """评估路径平滑度"""
        if len(path) < 3:
            return 1.0
        
        total_angle_change = 0.0
        for i in range(1, len(path) - 1):
            angle_change = self._calculate_angle_change(path[i-1], path[i], path[i+1])
            total_angle_change += angle_change
        
        # 归一化
        avg_angle_change = total_angle_change / max(1, len(path) - 2)
        smoothness = math.exp(-avg_angle_change * 2)
        
        return min(1.0, smoothness)
    
    def _calculate_angle_change(self, p1, p2, p3):
        """计算角度变化"""
        v1 = [p2[0] - p1[0], p2[1] - p1[1]]
        v2 = [p3[0] - p2[0], p3[1] - p2[1]]
        
        len1 = math.sqrt(v1[0]**2 + v1[1]**2)
        len2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if len1 < 0.001 or len2 < 0.001:
            return 0.0
        
        cos_angle = (v1[0]*v2[0] + v1[1]*v2[1]) / (len1 * len2)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        
        return math.acos(cos_angle)
    
    def _calculate_distance(self, pos1, pos2):
        """计算两点间距离"""
        x1 = pos1[0] if len(pos1) > 0 else 0
        y1 = pos1[1] if len(pos1) > 1 else 0
        x2 = pos2[0] if len(pos2) > 0 else 0
        y2 = pos2[1] if len(pos2) > 1 else 0
        
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def _calculate_path_length(self, path):
        """计算路径长度"""
        if not path or len(path) < 2:
            return 0.0
        
        length = 0.0
        for i in range(len(path) - 1):
            length += self._calculate_distance(path[i], path[i+1])
        
        return length
    
    def _ensure_3d_point(self, point):
        """确保点坐标有三个元素"""
        if not point:
            return (0, 0, 0)
        if len(point) >= 3:
            return (point[0], point[1], point[2])
        elif len(point) == 2:
            return (point[0], point[1], 0)
        else:
            return (0, 0, 0)
    
    def _create_planner(self):
        """创建RRT规划器"""
        try:
            return RRTPlanner(
                self.env,
                vehicle_length=6.0,
                vehicle_width=3.0,
                turning_radius=8.0,
                step_size=0.8
            )
        except Exception as e:
            print(f"警告: 无法创建RRTPlanner: {e}")
            return None
    
    # 兼容原始接口的方法
    @property
    def paths(self):
        """兼容原始接口"""
        return self.backbone_paths
    
    @property
    def connections(self):
        """兼容原始接口 - 返回空字典"""
        return {}
    def debug_network_status(self):
        """调试网络状态"""
        print("=== 骨干网络调试信息 ===")
        print(f"骨干路径数量: {len(self.backbone_paths)}")
        
        for path_id, path_data in self.backbone_paths.items():
            print(f"路径 {path_id}:")
            print(f"  起点: {path_data['start_point']['type']}_{path_data['start_point']['id']}")
            print(f"  终点: {path_data['end_point']['type']}_{path_data['end_point']['id']}")
            print(f"  路径长度: {len(path_data.get('path', []))} 个点")
            print(f"  接口数量: {len(self.path_interfaces.get(path_id, []))}")
        
        print(f"\n接口总数: {len(self.backbone_interfaces)}")
        print(f"特殊点数量:")
        print(f"  装载点: {len(self.special_points.get('loading', []))}")
        print(f"  卸载点: {len(self.special_points.get('unloading', []))}")
        
        # 显示到目标的路径索引
        print(f"\n路径到目标索引:")
        for target_key, paths in self.paths_to_target.items():
            print(f"  {target_key}: {len(paths)} 条路径")

    def debug_find_interface(self, position, target_type, target_id, max_distance=50):
        """调试接口查找过程"""
        print(f"\n=== 调试接口查找 ===")
        print(f"当前位置: {position}")
        print(f"目标: {target_type}_{target_id}")
        
        # 1. 检查能到达目标的骨干路径
        target_paths = self.find_paths_to_target(target_type, target_id)
        print(f"能到达目标的路径数量: {len(target_paths)}")
        for path_data in target_paths:
            print(f"  路径: {path_data['id']}")
        
        if not target_paths:
            print("❌ 没有找到到目标的骨干路径！")
            return None
        
        # 2. 查找附近的接口
        nearby_interfaces = self._find_nearby_interfaces(position, max_distance)
        print(f"附近接口数量: {len(nearby_interfaces)}")
        
        if not nearby_interfaces:
            print("❌ 没有找到附近的接口！")
            return None
        
        # 3. 筛选能到达目标的接口
        candidate_interfaces = []
        for interface_id in nearby_interfaces:
            interface = self.backbone_interfaces[interface_id]
            print(f"检查接口: {interface_id}")
            print(f"  骨干路径: {interface.backbone_path_id}")
            print(f"  是否可用: {interface.is_available()}")
            
            # 检查接口是否在能到达目标的路径上
            path_match = any(path_data['id'] == interface.backbone_path_id for path_data in target_paths)
            print(f"  路径匹配: {path_match}")
            
            if path_match and interface.is_available():
                candidate_interfaces.append(interface)
                print(f"  ✅ 接口可用")
            else:
                print(f"  ❌ 接口不可用")
        
        print(f"候选接口数量: {len(candidate_interfaces)}")
        return candidate_interfaces

# 保持向后兼容性
OptimizedBackbonePathNetwork = SimplifiedBackbonePathNetwork
BackbonePathNetwork = SimplifiedBackbonePathNetwork