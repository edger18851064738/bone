import math
import time
import threading
from collections import defaultdict, OrderedDict
from RRT import RRTPlanner

class SimplifiedBackbonePathNetwork:
    """
    简化的骨干路径网络 - 按照用户设计理念重新实现
    骨干网络是特殊点之间的完整路径集合，不包含复杂的网络结构
    
    设计理念：
    1. 读取地图中的特殊点（装载点、卸载点、停车点）
    2. 规划特殊点之间的完整骨干路径（不同类型点之间）
    3. 提供简单的查找和拼接接口
    """
    
    def __init__(self, env):
        self.env = env
        self.backbone_paths = {}  # 骨干路径字典 {path_id: path_data}
        self.special_points = {   # 特殊点分类
            'loading': [],
            'unloading': [],
            'parking': []
        }
        
        # 路径查找索引
        self.paths_to_target = defaultdict(list)  # {(target_type, target_id): [path_ids]}
        self.paths_from_source = defaultdict(list)  # {(source_type, source_id): [path_ids]}
        
        # 规划器
        self.planner = None
        
        # 性能统计
        self.stats = {
            'total_paths': 0,
            'generation_time': 0,
            'average_path_length': 0,
            'path_usage_count': defaultdict(int),
            'total_usage': 0
        }
        
        print("初始化简化的骨干路径网络")
    
    def generate_network(self, connection_spacing=None, quality_threshold=0.6):
        """
        生成骨干路径网络 - 兼容原接口
        只生成不同类型特殊点之间的完整路径
        
        Args:
            connection_spacing: 兼容参数，不再使用
            quality_threshold: 路径质量阈值
        """
        return self.generate_backbone_network(quality_threshold)
    
    def generate_backbone_network(self, quality_threshold=0.6):
        """
        生成骨干路径网络
        只生成不同类型特殊点之间的完整路径
        """
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
            
            # 4. 建立查找索引
            self._build_path_indexes()
            
            # 5. 统计信息
            generation_time = time.time() - start_time
            self.stats['generation_time'] = generation_time
            self.stats['total_paths'] = len(self.backbone_paths)
            
            if self.backbone_paths:
                total_length = sum(path_data['length'] for path_data in self.backbone_paths.values())
                self.stats['average_path_length'] = total_length / len(self.backbone_paths)
            
            print(f"骨干路径网络生成完成!")
            print(f"- 总路径数: {len(self.backbone_paths)}")
            print(f"- 生成耗时: {generation_time:.2f}秒")
            print(f"- 平均路径长度: {self.stats['average_path_length']:.1f}")
            
            return True
            
        except Exception as e:
            print(f"生成骨干路径网络失败: {e}")
            return False
    
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
        """建立路径查找索引"""
        self.paths_to_target.clear()
        self.paths_from_source.clear()
        
        for path_id, path_data in self.backbone_paths.items():
            start_point = path_data['start_point']
            end_point = path_data['end_point']
            
            # 按终点建立索引
            target_key = (end_point['type'], end_point['id'])
            self.paths_to_target[target_key].append(path_id)
            
            # 按起点建立索引
            source_key = (start_point['type'], start_point['id'])
            self.paths_from_source[source_key].append(path_id)
    
    def find_paths_to_target(self, target_type, target_id):
        """
        查找到指定目标的所有骨干路径
        
        Args:
            target_type: 目标类型 ('loading', 'unloading', 'parking')
            target_id: 目标ID
            
        Returns:
            list: 骨干路径数据列表
        """
        target_key = (target_type, target_id)
        path_ids = self.paths_to_target.get(target_key, [])
        
        paths = []
        for path_id in path_ids:
            if path_id in self.backbone_paths:
                paths.append(self.backbone_paths[path_id])
        
        return paths
    
    def find_nearest_backbone_path(self, current_position, target_type, target_id):
        """
        找到离当前位置最近的到目标的骨干路径
        
        Args:
            current_position: 当前位置 (x, y, theta)
            target_type: 目标类型
            target_id: 目标ID
            
        Returns:
            dict: 最近的骨干路径数据，None如果没找到
        """
        # 获取所有到目标的骨干路径
        candidate_paths = self.find_paths_to_target(target_type, target_id)
        
        if not candidate_paths:
            return None
        
        # 找到离当前位置最近的骨干路径起点
        best_path = None
        min_distance = float('inf')
        
        for path_data in candidate_paths:
            start_pos = path_data['start_point']['position']
            distance = self._calculate_distance(current_position, start_pos)
            
            if distance < min_distance:
                min_distance = distance
                best_path = path_data
        
        return best_path
    
    def get_path_from_position_to_target(self, current_position, target_type, target_id):
        """
        获取从当前位置到目标的完整路径（包含接入路径和骨干路径）
        
        Args:
            current_position: 当前位置
            target_type: 目标类型
            target_id: 目标ID
            
        Returns:
            tuple: (完整路径, 路径结构信息)
        """
        # 1. 查找最近的骨干路径
        backbone_path_data = self.find_nearest_backbone_path(current_position, target_type, target_id)
        
        if not backbone_path_data:
            print(f"未找到到 {target_type}_{target_id} 的骨干路径")
            return None, None
        
        # 2. 规划从当前位置到骨干路径起点的接入路径
        backbone_start_pos = backbone_path_data['start_point']['position']
        
        # 如果当前位置就是骨干路径起点，直接返回骨干路径
        if self._calculate_distance(current_position, backbone_start_pos) < 2.0:
            print(f"当前位置接近骨干路径起点，直接使用骨干路径 {backbone_path_data['id']}")
            return backbone_path_data['path'], {
                'type': 'backbone_only',
                'backbone_path_id': backbone_path_data['id'],
                'backbone_utilization': 1.0
            }
        
        # 规划接入路径
        print(f"规划接入路径到骨干路径 {backbone_path_data['id']}")
        access_path = self.planner.plan_path(current_position, backbone_start_pos, max_iterations=3000)
        
        if not access_path or len(access_path) < 2:
            print("接入路径规划失败")
            return None, None
        
        # 3. 拼接路径
        complete_path = self._merge_paths(access_path, backbone_path_data['path'])
        
        if not complete_path:
            print("路径拼接失败")
            return None, None
        
        # 4. 构建路径结构信息
        structure = {
            'type': 'backbone_assisted',
            'access_path': access_path,
            'backbone_path': backbone_path_data['path'],
            'backbone_path_id': backbone_path_data['id'],
            'backbone_utilization': len(backbone_path_data['path']) / len(complete_path),
            'access_length': len(access_path),
            'backbone_length': len(backbone_path_data['path']),
            'total_length': len(complete_path)
        }
        
        # 5. 更新使用统计
        self.backbone_paths[backbone_path_data['id']]['usage_count'] += 1
        self.stats['path_usage_count'][backbone_path_data['id']] += 1
        self.stats['total_usage'] += 1
        
        print(f"完整路径生成成功: 总长度{len(complete_path)}, 骨干利用率{structure['backbone_utilization']:.2f}")
        
        return complete_path, structure
    
    def identify_target_point(self, target_position):
        """
        识别目标位置是否为特殊点
        
        Args:
            target_position: 目标位置
            
        Returns:
            tuple: (target_type, target_id) 或 (None, None)
        """
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
        """
        合并接入路径和骨干路径
        
        Args:
            access_path: 接入路径
            backbone_path: 骨干路径
            
        Returns:
            list: 合并后的完整路径
        """
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
    
    def get_statistics(self):
        """获取统计信息"""
        stats = self.stats.copy()
        
        if self.backbone_paths:
            # 路径使用情况
            stats['total_usage'] = sum(path_data['usage_count'] for path_data in self.backbone_paths.values())
            
            # 最常用的路径
            if self.backbone_paths:
                most_used_path = max(self.backbone_paths.items(), 
                                   key=lambda x: x[1]['usage_count'])
                stats['most_used_path'] = {
                    'id': most_used_path[0],
                    'usage_count': most_used_path[1]['usage_count']
                }
            
            # 质量分布
            qualities = [path_data['quality'] for path_data in self.backbone_paths.values()]
            stats['average_quality'] = sum(qualities) / len(qualities)
            stats['min_quality'] = min(qualities)
            stats['max_quality'] = max(qualities)
            
            # 路径类型统计
            loading_to_unloading = len([p for p in self.backbone_paths.keys() if 'L' in p and 'U' in p])
            loading_to_parking = len([p for p in self.backbone_paths.keys() if 'L' in p and 'P' in p])
            unloading_to_parking = len([p for p in self.backbone_paths.keys() if 'U' in p and 'P' in p])
            
            stats['path_type_distribution'] = {
                'loading_to_unloading': loading_to_unloading,
                'loading_to_parking': loading_to_parking,
                'unloading_to_parking': unloading_to_parking
            }
        
        return stats
    
    def get_path_info(self, path_id):
        """获取指定路径的详细信息"""
        if path_id not in self.backbone_paths:
            return None
        
        path_data = self.backbone_paths[path_id]
        
        return {
            'id': path_data['id'],
            'start_point': {
                'type': path_data['start_point']['type'],
                'id': path_data['start_point']['id'],
                'position': path_data['start_point']['position']
            },
            'end_point': {
                'type': path_data['end_point']['type'],
                'id': path_data['end_point']['id'],
                'position': path_data['end_point']['position']
            },
            'length': path_data['length'],
            'quality': path_data['quality'],
            'usage_count': path_data['usage_count'],
            'path_points': len(path_data['path'])
        }
    
    def list_all_paths(self):
        """列出所有骨干路径的基本信息"""
        paths_info = []
        
        for path_id, path_data in self.backbone_paths.items():
            paths_info.append({
                'id': path_id,
                'start': f"{path_data['start_point']['type']}_{path_data['start_point']['id']}",
                'end': f"{path_data['end_point']['type']}_{path_data['end_point']['id']}",
                'length': round(path_data['length'], 1),
                'quality': round(path_data['quality'], 2),
                'usage': path_data['usage_count']
            })
        
        # 按使用次数排序
        paths_info.sort(key=lambda x: x['usage'], reverse=True)
        
        return paths_info
    
    # 兼容原始接口的方法
    @property
    def paths(self):
        """兼容原始接口"""
        return self.backbone_paths
    
    @property
    def connections(self):
        """兼容原始接口 - 返回空字典"""
        return {}
    
    def find_nearest_connection_optimized(self, position, max_distance=5.0, max_candidates=5):
        """兼容原始接口 - 简化实现"""
        # 在简化版本中，不再需要连接点概念
        # 返回None表示没有找到连接点
        return None
    
    def find_accessible_points(self, position, rrt_planner, max_candidates=5, 
                            sampling_step=10, max_distance=20.0):
        """兼容原始接口 - 简化实现"""
        # 在简化版本中，不再需要可达点概念
        # 返回空列表
        return []


# 保持向后兼容性
OptimizedBackbonePathNetwork = SimplifiedBackbonePathNetwork
BackbonePathNetwork = SimplifiedBackbonePathNetwork