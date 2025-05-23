import math
import time
import threading
import numpy as np
from collections import defaultdict, OrderedDict
from RRT import OptimizedRRTPlanner

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


class EnhancedBackboneInterface(BackboneInterface):
    """增强的骨干路径接口 - 支持RRT预处理"""
    
    def __init__(self, interface_id, position, direction, backbone_path_id, 
                 path_index, access_difficulty=0.0):
        super().__init__(interface_id, position, direction, backbone_path_id, 
                        path_index, access_difficulty)
        
        # 新增RRT集成属性
        self.rrt_sampling_weight = 1.0  # RRT采样权重
        self.accessibility_score = 0.5  # 可达性评分
        self.usage_efficiency = 0.0     # 使用效率
        self.last_quality_score = 0.0   # 最近路径质量
        
        # 性能统计
        self.rrt_cache_hits = 0
        self.total_planning_attempts = 0
        self.average_planning_time = 0.0
        
        # 区域影响
        self.influence_radius = 15.0    # 影响半径
        self.sampling_hotspot = False   # 是否为采样热点
    
    def update_rrt_statistics(self, planning_time, path_quality, cache_hit=False):
        """更新RRT相关统计信息"""
        self.total_planning_attempts += 1
        if cache_hit:
            self.rrt_cache_hits += 1
        
        # 更新平均规划时间
        alpha = 0.1  # 学习率
        if self.average_planning_time == 0:
            self.average_planning_time = planning_time
        else:
            self.average_planning_time = (1-alpha) * self.average_planning_time + alpha * planning_time
        
        # 更新质量评分
        self.last_quality_score = path_quality
        
        # 更新使用效率
        if self.total_planning_attempts > 0:
            self.usage_efficiency = self.rrt_cache_hits / self.total_planning_attempts
    
    def calculate_sampling_priority(self):
        """计算RRT采样优先级"""
        # 基础权重
        priority = self.rrt_sampling_weight
        
        # 可达性加权
        priority *= (0.5 + self.accessibility_score)
        
        # 使用频率加权（使用越多优先级越高，但有上限）
        usage_factor = min(2.0, 1.0 + self.usage_count * 0.1)
        priority *= usage_factor
        
        # 效率加权
        if self.usage_efficiency > 0.7:
            priority *= 1.2  # 高效接口优先
        elif self.usage_efficiency < 0.3:
            priority *= 0.8  # 低效接口降权
        
        return priority
    
    def get_influence_region(self):
        """获取接口影响区域（用于RRT采样）"""
        return {
            'center': (self.position[0], self.position[1]),
            'radius': self.influence_radius,
            'priority': self.calculate_sampling_priority(),
            'direction_bias': self.direction,
            'quality_hint': self.last_quality_score
        }


class SimplifiedBackbonePathNetwork:
    """
    完整的简化骨干路径网络 - 带接口系统优化版
    保持所有原有属性和方法的兼容性
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
        
        # RRT集成增强
        self.rrt_integration = {
            'preprocessing_enabled': True,
            'adaptive_sampling': True,
            'quality_feedback': True,
            'cache_coordination': True
        }
        
        # 预处理数据
        self.sampling_regions = {}      # RRT采样区域
        self.path_quality_map = {}      # 路径质量映射
        self.access_heatmap = None      # 可达性热力图
        
        # 性能缓存
        self.rrt_planner_ref = None     # RRT规划器引用
        self.path_cache_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'quality_improvements': 0
        }
        
        print("初始化带接口系统的骨干路径网络")
    
    def set_rrt_planner(self, rrt_planner):
        """设置RRT规划器引用，启用深度集成"""
        self.rrt_planner_ref = rrt_planner
        if rrt_planner:
            rrt_planner.set_backbone_network(self)
            self._initialize_rrt_integration()
    
    def _initialize_rrt_integration(self):
        """初始化RRT集成"""
        print("初始化RRT深度集成...")
        
        # 预处理接口区域
        self._preprocess_sampling_regions()
        
        # 构建可达性热力图
        self._build_accessibility_heatmap()
        
        # 启用质量反馈循环
        self._setup_quality_feedback()
        
        print(f"RRT集成完成: {len(self.sampling_regions)}个采样区域")
    
    def _preprocess_sampling_regions(self):
        """预处理RRT采样区域"""
        self.sampling_regions.clear()
        
        if not hasattr(self, 'backbone_interfaces'):
            return
        
        for interface_id, interface in self.backbone_interfaces.items():
            # 升级为增强接口
            if not isinstance(interface, EnhancedBackboneInterface):
                enhanced_interface = self._upgrade_interface(interface)
                self.backbone_interfaces[interface_id] = enhanced_interface
                interface = enhanced_interface
            
            # 计算接口影响区域
            region = interface.get_influence_region()
            self.sampling_regions[interface_id] = region
        
        # 添加骨干路径中点作为采样区域
        self._add_backbone_midpoint_regions()
    
    def _upgrade_interface(self, old_interface):
        """升级接口为增强版本"""
        enhanced = EnhancedBackboneInterface(
            old_interface.interface_id,
            old_interface.position,
            old_interface.direction,
            old_interface.backbone_path_id,
            old_interface.path_index,
            old_interface.access_difficulty
        )
        
        # 传递统计数据
        enhanced.usage_count = old_interface.usage_count
        enhanced.is_occupied = old_interface.is_occupied
        enhanced.occupied_by = old_interface.occupied_by
        enhanced.reservation_time = old_interface.reservation_time
        
        return enhanced
    
    def _add_backbone_midpoint_regions(self):
        """添加骨干路径中点作为采样区域"""
        for path_id, path_data in self.backbone_paths.items():
            path = path_data.get('path', [])
            if len(path) > 10:  # 只处理较长的路径
                # 在路径中点添加采样区域
                mid_index = len(path) // 2
                mid_point = path[mid_index]
                
                region_id = f"{path_id}_midpoint"
                self.sampling_regions[region_id] = {
                    'center': (mid_point[0], mid_point[1]),
                    'radius': 12.0,
                    'priority': 0.8,
                    'direction_bias': mid_point[2] if len(mid_point) > 2 else 0,
                    'quality_hint': path_data.get('quality', 0.5)
                }
    
    def _build_accessibility_heatmap(self):
        """构建可达性热力图"""
        if not self.env:
            return
        
        # 简化的热力图：基于距离障碍物的距离
        grid_size = 10  # 热力图网格大小
        width_cells = self.env.width // grid_size
        height_cells = self.env.height // grid_size
        
        self.access_heatmap = np.zeros((width_cells, height_cells))
        
        for i in range(width_cells):
            for j in range(height_cells):
                # 计算网格中心点
                center_x = i * grid_size + grid_size // 2
                center_y = j * grid_size + grid_size // 2
                
                # 计算可达性评分
                accessibility = self._calculate_point_accessibility(center_x, center_y)
                self.access_heatmap[i, j] = accessibility
    
    def _calculate_point_accessibility(self, x, y):
        """计算点的可达性评分"""
        # 基于周围障碍物密度
        obstacle_count = 0
        total_cells = 0
        check_radius = 5
        
        for dx in range(-check_radius, check_radius + 1):
            for dy in range(-check_radius, check_radius + 1):
                check_x, check_y = x + dx, y + dy
                if (0 <= check_x < self.env.width and 
                    0 <= check_y < self.env.height):
                    total_cells += 1
                    if hasattr(self.env, 'grid') and self.env.grid[check_x, check_y] == 1:
                        obstacle_count += 1
        
        if total_cells == 0:
            return 0
        
        return 1.0 - (obstacle_count / total_cells)
    
    def _setup_quality_feedback(self):
        """设置质量反馈循环"""
        # 这里可以设置质量反馈机制
        pass
    
    def generate_backbone_network(self, quality_threshold=0.4, interface_spacing=8):
        """
        生成骨干路径网络并创建接口点 - 完整版本
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
            
            # 4. 生成骨干接口点 - 新的简化方法
            self._generate_backbone_interfaces_simplified()
            
            # 5. 建立查找索引
            self._build_path_indexes()
            self._build_interface_spatial_index()
            
            # 6. 如果有RRT集成，初始化相关系统
            if self.rrt_planner_ref:
                self._initialize_rrt_integration()
            
            # 7. 统计信息
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
            import traceback
            traceback.print_exc()
            return False
    
    def _generate_backbone_interfaces_simplified(self):
        """简化的接口生成方法 - 在每条骨干路径上均匀分布接口"""
        print("正在生成骨干路径接口点...")
        
        total_interfaces = 0
        
        for path_id, path_data in self.backbone_paths.items():
            path = path_data['path']
            path_length = len(path)
            
            if path_length < 2:
                print(f"⚠️ 路径 {path_id} 太短({path_length}个点)，跳过接口生成")
                continue
                
            interfaces_for_path = []
            interface_count = 0
            
            # 从路径起点开始，每隔interface_spacing个点设置一个接口
            for i in range(0, path_length, self.interface_spacing):
                if i >= path_length:
                    break
                    
                # 计算接口方向
                direction = self._calculate_interface_direction(path, i)
                
                # 创建增强接口
                interface_id = f"{path_id}_if_{interface_count}"
                interface = EnhancedBackboneInterface(
                    interface_id=interface_id,
                    position=path[i],
                    direction=direction,
                    backbone_path_id=path_id,
                    path_index=i,
                    access_difficulty=self._evaluate_interface_access_difficulty(path, i)
                )
                
                # 计算可达性评分
                interface.accessibility_score = self._calculate_point_accessibility(
                    int(path[i][0]), int(path[i][1])
                )
                
                # 存储接口
                self.backbone_interfaces[interface_id] = interface
                interfaces_for_path.append(interface_id)
                total_interfaces += 1
                interface_count += 1
            
            # 确保路径终点也有接口（如果终点不在间距点上）
            last_index = path_length - 1
            if last_index > 0 and last_index % self.interface_spacing != 0:
                last_interface_id = f"{path_id}_if_end"
                direction = self._calculate_interface_direction(path, last_index)
                last_interface = EnhancedBackboneInterface(
                    interface_id=last_interface_id,
                    position=path[last_index],
                    direction=direction,
                    backbone_path_id=path_id,
                    path_index=last_index,
                    access_difficulty=self._evaluate_interface_access_difficulty(path, last_index)
                )
                
                last_interface.accessibility_score = self._calculate_point_accessibility(
                    int(path[last_index][0]), int(path[last_index][1])
                )
                
                self.backbone_interfaces[last_interface_id] = last_interface
                interfaces_for_path.append(last_interface_id)
                total_interfaces += 1
            
            self.path_interfaces[path_id] = interfaces_for_path
            print(f"   路径 {path_id}: 生成 {len(interfaces_for_path)} 个接口")
        
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
        找到最近的可用骨干接口 - 简化版本
        """
        if debug:
            print(f"\n🔍 查找从 {position} 到 {target_type}_{target_id} 的接口")
        
        # 1. 找到所有通向目标的骨干路径
        target_key = (target_type, target_id)
        target_paths = self.paths_to_target.get(target_key, [])
        
        if not target_paths:
            if debug:
                print(f"❌ 没有找到通向 {target_type}_{target_id} 的骨干路径")
            return None
        
        if debug:
            print(f"📍 找到 {len(target_paths)} 条通向目标的骨干路径")
        
        # 2. 收集这些路径上的所有可用接口
        candidate_interfaces = []
        
        for path_data in target_paths:
            path_id = path_data['id']
            
            if path_id not in self.path_interfaces:
                continue
                
            for interface_id in self.path_interfaces[path_id]:
                interface = self.backbone_interfaces[interface_id]
                
                # 检查接口是否可用
                if not interface.is_available():
                    continue
                
                # 计算距离
                distance = self._calculate_distance(position, interface.position)
                if distance <= max_distance:
                    candidate_interfaces.append((interface, distance, path_id))
        
        if not candidate_interfaces:
            if debug:
                print(f"❌ 在距离 {max_distance} 内没有找到可用接口")
            return None
        
        if debug:
            print(f"🎯 找到 {len(candidate_interfaces)} 个候选接口")
        
        # 3. 选择最佳接口
        best_interface = None
        best_score = -float('inf')
        
        for interface, distance, path_id in candidate_interfaces:
            # 计算到目标的剩余路径长度
            backbone_path = self.backbone_paths[path_id]['path']
            remaining_length = len(backbone_path) - interface.path_index
            
            # 综合评分：距离越近越好，剩余路径越长越好，可达性越高越好
            distance_score = 100 / (distance + 1)
            remaining_score = remaining_length * 0.5
            accessibility_score = interface.accessibility_score * 20
            quality_score = interface.last_quality_score * 10
            
            total_score = distance_score + remaining_score + accessibility_score + quality_score
            
            if debug:
                print(f"   接口 {interface.interface_id}: 距离={distance:.1f}, "
                      f"剩余={remaining_length}, 可达性={interface.accessibility_score:.2f}, "
                      f"评分={total_score:.1f}")
            
            if total_score > best_score:
                best_score = total_score
                best_interface = interface
        
        if best_interface and debug:
            print(f"✅ 选择接口: {best_interface.interface_id} (评分: {best_score:.1f})")
        
        return best_interface
    
    def get_complete_path_via_interface_enhanced(self, start, target_type, target_id, 
                                               rrt_hints=None):
        """增强版路径获取 - 集成RRT提示"""
        self.path_cache_stats['total_requests'] += 1
        
        # 获取基础路径
        base_result = self.get_path_from_position_to_target_via_interface(
            start, target_type, target_id
        )
        
        if not base_result or not base_result[0]:
            return base_result
        
        path, structure = base_result
        
        # 如果有RRT提示，进行路径优化
        if rrt_hints and self.rrt_planner_ref:
            optimized_path = self._apply_rrt_hints(path, rrt_hints)
            if optimized_path:
                path = optimized_path
                structure['rrt_optimized'] = True
                self.path_cache_stats['quality_improvements'] += 1
        
        # 更新接口统计
        interface_id = structure.get('interface_id')
        if interface_id in self.backbone_interfaces:
            interface = self.backbone_interfaces[interface_id]
            if hasattr(interface, 'update_rrt_statistics'):
                quality = self._evaluate_path_quality(path)
                interface.update_rrt_statistics(0.1, quality, False)  # 假设时间
        
        return path, structure
    
    def _apply_rrt_hints(self, path, hints):
        """应用RRT提示优化路径"""
        try:
            # 这里可以应用RRT规划器的优化建议
            if 'smoothing_suggested' in hints:
                return self._smooth_path_with_rrt(path)
            
            if 'density_adjustment' in hints:
                return self._adjust_path_density_smart(path, hints['target_density'])
            
        except Exception as e:
            print(f"RRT提示应用失败: {e}")
        
        return path
    
    def _smooth_path_with_rrt(self, path):
        """使用RRT优化器平滑路径"""
        if self.rrt_planner_ref and hasattr(self.rrt_planner_ref, '_adaptive_smoothing'):
            return self.rrt_planner_ref._adaptive_smoothing(path)
        return path
    
    def _adjust_path_density_smart(self, path, target_density):
        """智能调整路径密度"""
        if self.rrt_planner_ref and hasattr(self.rrt_planner_ref, '_adjust_path_density'):
            return self.rrt_planner_ref._adjust_path_density(path, target_density)
        return path
    
    def get_sampling_guidance_for_rrt(self, start, goal):
        """为RRT提供采样引导信息"""
        guidance = {
            'priority_regions': [],
            'avoid_regions': [],
            'backbone_hints': [],
            'interface_targets': []
        }
        
        # 添加相关的采样区域
        for region_id, region in self.sampling_regions.items():
            # 计算与起终点的相关性
            relevance = self._calculate_region_relevance(region, start, goal)
            
            if relevance > 0.3:
                guidance['priority_regions'].append({
                    'region': region,
                    'relevance': relevance,
                    'id': region_id
                })
        
        # 排序并限制数量
        guidance['priority_regions'].sort(key=lambda x: x['relevance'], reverse=True)
        guidance['priority_regions'] = guidance['priority_regions'][:10]
        
        # 添加骨干路径提示
        target_type, target_id = self.identify_target_point(goal)
        if target_type:
            relevant_paths = self.find_paths_to_target(target_type, target_id)
            for path_data in relevant_paths[:3]:  # 最多3条提示路径
                guidance['backbone_hints'].append({
                    'path_id': path_data['id'],
                    'quality': path_data.get('quality', 0.5),
                    'length': path_data.get('length', 0)
                })
        
        return guidance
    
    def _calculate_region_relevance(self, region, start, goal):
        """计算区域与起终点的相关性"""
        center = region['center']
        
        # 计算到起点和终点的距离
        dist_to_start = math.sqrt((center[0] - start[0])**2 + (center[1] - start[1])**2)
        dist_to_goal = math.sqrt((center[0] - goal[0])**2 + (center[1] - goal[1])**2)
        
        # 计算起终点直线距离
        direct_distance = math.sqrt((goal[0] - start[0])**2 + (goal[1] - start[1])**2)
        
        # 相关性基于区域是否在合理的路径范围内
        max_detour = direct_distance * 1.5  # 允许50%的绕行
        total_distance = dist_to_start + dist_to_goal
        
        if total_distance <= max_detour:
            # 基础相关性
            base_relevance = 1.0 - (total_distance - direct_distance) / (max_detour - direct_distance)
            
            # 区域优先级加权
            priority_weight = region.get('priority', 1.0)
            
            return base_relevance * priority_weight
        
        return 0.0
    
    def update_path_feedback(self, path, planning_time, quality_score, used_cache=False):
        """更新路径反馈信息"""
        if used_cache:
            self.path_cache_stats['cache_hits'] += 1
        
        # 更新相关接口的统计信息
        used_interfaces = self._identify_path_interfaces(path)
        
        for interface_id in used_interfaces:
            if interface_id in self.backbone_interfaces:
                interface = self.backbone_interfaces[interface_id]
                if hasattr(interface, 'update_rrt_statistics'):
                    interface.update_rrt_statistics(planning_time, quality_score, used_cache)
    
    def _identify_path_interfaces(self, path):
        """识别路径使用的接口"""
        used_interfaces = []
        
        if not path:
            return used_interfaces
        
        # 检查路径点是否接近接口
        for interface_id, interface in self.backbone_interfaces.items():
            interface_pos = interface.position
            
            for path_point in path[::5]:  # 每隔5个点检查一次
                distance = math.sqrt(
                    (path_point[0] - interface_pos[0])**2 + 
                    (path_point[1] - interface_pos[1])**2
                )
                
                if distance < 5.0:  # 如果路径接近接口
                    used_interfaces.append(interface_id)
                    break
        
        return used_interfaces
    
    def get_path_from_position_to_target_via_interface(self, current_position, target_type, target_id):
        """
        通过骨干接口获取从当前位置到目标的完整路径
        """
        # 1. 查找最佳骨干接口
        best_interface = self.find_nearest_interface(current_position, target_type, target_id)
        
        if not best_interface:
            print(f"未找到到 {target_type}_{target_id} 的可用骨干接口")
            return None, None
        
        # 2. 预约接口
        best_interface.reserve("vehicle_temp", duration=60)
        
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
    
    def get_rrt_performance_stats(self):
        """获取RRT相关的性能统计"""
        stats = {
            'cache_stats': self.path_cache_stats.copy(),
            'interface_performance': {},
            'sampling_region_count': len(self.sampling_regions),
            'heatmap_available': self.access_heatmap is not None
        }
        
        # 接口性能统计
        for interface_id, interface in self.backbone_interfaces.items():
            if hasattr(interface, 'total_planning_attempts'):
                stats['interface_performance'][interface_id] = {
                    'usage_count': interface.usage_count,
                    'planning_attempts': interface.total_planning_attempts,
                    'cache_hit_rate': interface.usage_efficiency,
                    'avg_planning_time': interface.average_planning_time,
                    'last_quality': interface.last_quality_score,
                    'accessibility_score': getattr(interface, 'accessibility_score', 0.5)
                }
        
        return stats
    
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
    
    # 保持原有接口兼容性的方法
    def get_path_from_position_to_target(self, current_position, target_type, target_id):
        """兼容原有接口，内部调用新的接口系统"""
        return self.get_path_from_position_to_target_via_interface(
            current_position, target_type, target_id
        )
    
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
        """生成单条骨干路径 - 增加重试机制"""
        try:
            start_pos = start_point['position']
            end_pos = end_point['position']
            
            # 多次尝试增加成功率
            for attempt in range(3):
                max_iterations = 3000 + attempt * 1000
                
                path = self.planner.plan_path(start_pos, end_pos, max_iterations=max_iterations)
                
                if path and len(path) >= 2:
                    # 评估路径质量
                    quality = self._evaluate_path_quality(path)
                    if quality >= quality_threshold:
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
                        
                        print(f"✅ 路径 {path_id} 生成成功 (尝试 {attempt+1}, 质量: {quality:.2f})")
                        return True
                    else:
                        print(f"⚠️ 路径 {path_id} 质量不达标: {quality:.2f} < {quality_threshold} (尝试 {attempt+1})")
                else:
                    print(f"❌ 路径 {path_id} 规划失败 (尝试 {attempt+1})")
            
            return False
            
        except Exception as e:
            print(f"生成路径 {path_id} 失败: {e}")
            return False
    
    def _build_path_indexes(self):
        """建立路径查找索引"""
        self.paths_to_target.clear()
        self.paths_from_source.clear()
        
        print("开始建立路径索引...")
        
        for path_id, path_data in self.backbone_paths.items():
            start_point = path_data['start_point']
            end_point = path_data['end_point']
            
            # 按终点建立索引
            target_key = (end_point['type'], end_point['id'])
            self.paths_to_target[target_key].append(path_data)
            
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
            # 使用新的优化RRT规划器
            planner = OptimizedRRTPlanner(
                self.env,
                vehicle_length=6.0,
                vehicle_width=3.0,
                turning_radius=8.0,
                step_size=0.8
            )
            
            # 设置双向引用
            planner.set_backbone_network(self)
            
            return planner
        except Exception as e:
            print(f"警告: 无法创建OptimizedRRTPlanner: {e}")
            # 回退到原始RRT规划器
            try:
                from RRT import RRTPlanner
                return RRTPlanner(
                    self.env,
                    vehicle_length=6.0,
                    vehicle_width=3.0,
                    turning_radius=8.0,
                    step_size=0.8
                )
            except Exception as e2:
                print(f"警告: 无法创建任何RRT规划器: {e2}")
                return None
    
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

    def debug_interface_system(self):
        """调试接口系统"""
        print("\n=== 接口系统调试信息 ===")
        print(f"骨干路径数量: {len(self.backbone_paths)}")
        print(f"接口总数: {len(self.backbone_interfaces)}")
        print(f"接口间距设置: {self.interface_spacing}")
        print(f"采样区域数量: {len(self.sampling_regions)}")
        
        # 按路径显示接口分布
        for path_id, interface_ids in self.path_interfaces.items():
            if path_id in self.backbone_paths:
                path_length = len(self.backbone_paths[path_id]['path'])
                print(f"\n路径 {path_id}:")
                print(f"   路径长度: {path_length} 个点")
                print(f"   接口数量: {len(interface_ids)} 个")
                
                for i, interface_id in enumerate(interface_ids):
                    if interface_id in self.backbone_interfaces:
                        interface = self.backbone_interfaces[interface_id]
                        accessibility = getattr(interface, 'accessibility_score', 0.5)
                        print(f"   - {interface_id}: 索引{interface.path_index}, "
                              f"位置({interface.position[0]:.1f}, {interface.position[1]:.1f}), "
                              f"可达性:{accessibility:.2f}")
        
        # 显示RRT集成状态
        if self.rrt_planner_ref:
            print(f"\nRRT集成状态: ✅ 已启用")
            rrt_stats = self.rrt_planner_ref.get_statistics() if hasattr(self.rrt_planner_ref, 'get_statistics') else {}
            print(f"RRT缓存命中率: {rrt_stats.get('cache_hit_rate', 0):.1%}")
        else:
            print(f"\nRRT集成状态: ❌ 未启用")
        
        # 显示路径到目标的索引
        print(f"\n路径到目标索引:")
        for target_key, path_data_list in self.paths_to_target.items():
            path_ids = [p['id'] for p in path_data_list]
            print(f"   {target_key}: {path_ids}")
    
    # ===== 保持向后兼容性的属性和方法 =====
    
    @property
    def paths(self):
        """兼容原始接口"""
        return self.backbone_paths
    
    @property
    def connections(self):
        """兼容原始接口 - 返回空字典"""
        return {}


# 保持向后兼容性
OptimizedBackbonePathNetwork = SimplifiedBackbonePathNetwork
BackbonePathNetwork = SimplifiedBackbonePathNetwork