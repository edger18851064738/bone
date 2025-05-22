import math
import numpy as np
import time
import threading
from collections import defaultdict, OrderedDict
from scipy.spatial import KDTree

try:
    from scipy.spatial import cKDTree
    CKDTREE_AVAILABLE = True
except ImportError:
    from scipy.spatial import KDTree as cKDTree
    CKDTREE_AVAILABLE = False

from RRT import RRTPlanner

class OptimizedBackbonePathNetwork:
    """优化后的主干路径网络，支持智能连接点生成、高效路径管理和实时监控"""
    
    def __init__(self, env):
        """初始化主干路径网络"""
        self.env = env
        self.paths = {}               # 路径字典 {path_id: path_data}
        self.nodes = {}               # 节点字典 {node_id: node_data}
        self.path_graph = {}          # 路径连接图，用于路由
        self.connections = {}         # 连接点字典
        
        # 高级空间索引系统
        self.advanced_spatial_index = {
            'connection_kdtree': None,
            'path_kdtree': None,
            'grid_index': {},  # 网格索引
            'dirty': True
        }
        
        # 缓存优化系统
        self.cache_config = {
            'max_cache_size': 1000,
            'ttl': 300,  # 5分钟过期
            'cleanup_interval': 60  # 1分钟清理一次
        }
        self.query_cache = OrderedDict()  # LRU缓存
        self.cache_timestamps = {}
        
        # 网络拓扑分析
        self.topology_metrics = {
            'connectivity_graph': {},
            'shortest_paths': {},
            'centrality_scores': {},
            'last_analysis': 0
        }
        
        # 空间索引优化（兼容原有代码）
        self.connection_kdtree = None
        self.path_point_kdtree = None
        self.spatial_index_dirty = True
        
        # 路径质量评估
        self.path_quality_cache = {}
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
            'cache_misses': 0,
            'start_time': time.time(),
            'health_issues': []
        }
        
        # 规划器缓存
        self.planner = None
        
        # 实时监控
        self.monitoring_enabled = False
        self.monitor_thread = None
        
        # 自适应参数
        self.adaptive_enabled = True
        
        print(f"初始化优化的骨干路径网络 - KDTree可用: {CKDTREE_AVAILABLE}")
    
    def generate_network(self, connection_spacing=10, quality_threshold=0.6):
        """生成完整的主干路径网络"""
        start_time = time.time()
        
        print("开始生成优化的骨干路径网络...")
        
        try:
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
            
            # 7. 构建高级空间索引
            self._build_spatial_indexes()
            print("空间索引构建完成")
            
            # 8. 分析拓扑结构
            self.analyze_network_topology()
            
            self.performance_stats['generation_time'] = time.time() - start_time
            print(f"骨干网络生成完成，耗时: {self.performance_stats['generation_time']:.2f}秒")
            
            return self.paths
            
        except Exception as e:
            print(f"生成骨干网络时出错: {e}")
            return {}
    
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
                "priority": 3,
                "capacity": 5
            })
        
        # 添加卸载点
        for i, point in enumerate(self.env.unloading_points):
            pos = self._ensure_3d_point(point)
            key_points.append({
                "position": pos, 
                "type": "unloading_point",
                "id": f"U{i}",
                "priority": 3,
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
                "priority": 1,
                "capacity": 10
            })
        
        return key_points
    
    def _generate_paths_between_key_points(self, key_points, quality_threshold=0.6):
        """生成关键点之间的高质量路径"""
        if not self.planner:
            self.planner = self._create_planner()
        
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
                            'utilization': 0.0,
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
            max_iterations = 3000 + attempt * 1000
            if self.planner:
                path = self.planner.plan_path(start, goal, max_iterations=max_iterations)
                if path and len(path) > 1:
                    candidates.append(path)
        
        return candidates
    
    def _generate_intelligent_connection_points(self, spacing=10):
        """智能生成连接点"""
        self.connections = {}
        connection_id = 0
        
        for path_id, path_data in self.paths.items():
            path = path_data['path']
            if len(path) < 2:
                continue
            
            # 添加起点和终点连接点
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
        
        for i in range(spacing, len(path) - spacing, spacing):
            point = path[i]
            
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
    
    def _build_spatial_indexes(self):
        """构建优化的空间索引系统 - 修复版"""
        try:
            if not self.connections:
                if self.debug:
                    print("❌ 没有连接点可建立索引")
                return
            
            # 构建连接点KD树
            connection_points = []
            connection_ids = []
            
            for conn_id, conn_data in self.connections.items():
                if (conn_data and 'position' in conn_data and 
                    conn_data['position'] is not None and 
                    len(conn_data['position']) >= 2):
                    
                    pos = conn_data['position']
                    connection_points.append([float(pos[0]), float(pos[1])])
                    connection_ids.append(conn_id)
            
            if connection_points:
                # 使用最佳可用的KDTree实现
                if CKDTREE_AVAILABLE:
                    self.advanced_spatial_index['connection_kdtree'] = cKDTree(
                        connection_points, 
                        leafsize=16,
                        balanced_tree=True,
                        compact_nodes=True
                    )
                else:
                    self.advanced_spatial_index['connection_kdtree'] = cKDTree(connection_points)
                    
                self.connection_ids = connection_ids
                # 保持向后兼容
                self.connection_kdtree = self.advanced_spatial_index['connection_kdtree']
                
                if self.debug:
                    print(f"✅ 连接点KD树构建完成，{len(connection_points)}个点")
            else:
                if self.debug:
                    print("❌ 没有有效的连接点位置")
            
            # 构建路径点KD树
            path_points = []
            path_info = []
            
            for path_id, path_data in self.paths.items():
                if not path_data or 'path' not in path_data:
                    continue
                    
                path = path_data['path']
                if not path:
                    continue
                    
                # 每5个点采样一个，避免索引过大
                for i, point in enumerate(path[::5]):
                    if point and len(point) >= 2:
                        try:
                            path_points.append([float(point[0]), float(point[1])])
                            path_info.append((path_id, i * 5))
                        except (ValueError, TypeError):
                            continue
            
            if path_points:
                if CKDTREE_AVAILABLE:
                    self.advanced_spatial_index['path_kdtree'] = cKDTree(
                        path_points,
                        leafsize=20,
                        balanced_tree=True
                    )
                else:
                    self.advanced_spatial_index['path_kdtree'] = cKDTree(path_points)
                    
                self.path_point_info = path_info
                # 保持向后兼容
                self.path_point_kdtree = self.advanced_spatial_index['path_kdtree']
                
                if self.debug:
                    print(f"✅ 路径点KD树构建完成，{len(path_points)}个点")
            
            # 构建网格索引
            self._build_grid_index()
            
            self.advanced_spatial_index['dirty'] = False
            self.spatial_index_dirty = False
            
            print("✅ 高级空间索引构建完成")
            
        except Exception as e:
            print(f"❌ 构建空间索引失败: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            self._build_simple_spatial_index()
    # 添加调试属性
    @property 
    def debug(self):
        return getattr(self, '_debug', False)

    @debug.setter
    def debug(self, value):
        self._debug = value
    def _build_grid_index(self):
        """构建网格索引用于快速区域查询"""
        grid_size = 20.0
        self.advanced_spatial_index['grid_index'] = {}
        
        for conn_id, conn_data in self.connections.items():
            if 'position' not in conn_data:
                continue
                
            pos = conn_data['position']
            grid_x = int(pos[0] // grid_size)
            grid_y = int(pos[1] // grid_size)
            grid_key = (grid_x, grid_y)
            
            if grid_key not in self.advanced_spatial_index['grid_index']:
                self.advanced_spatial_index['grid_index'][grid_key] = {
                    'connections': [],
                    'paths': set()
                }
            
            self.advanced_spatial_index['grid_index'][grid_key]['connections'].append(conn_id)
            
            # 添加相关路径
            for path_id in conn_data.get('paths', []):
                self.advanced_spatial_index['grid_index'][grid_key]['paths'].add(path_id)
    
    def find_nearest_connection_optimized(self, position, max_distance=5.0, max_candidates=5):
        """超级优化的最近连接点查找 - 修复版"""
        start_time = time.time()
        
        # 输入验证
        if not position or len(position) < 2:
            if self.debug:
                print("❌ 位置信息无效")
            return None
        
        # 检查缓存
        cache_key = f"conn_{position[0]:.1f}_{position[1]:.1f}_{max_distance}"
        if self._check_cache(cache_key):
            cached_result = self._get_cache(cache_key)
            if cached_result:
                self.performance_stats['cache_hits'] += 1
                return cached_result
        
        self.performance_stats['cache_misses'] += 1
        
        # 重建索引（如果需要）
        if self.advanced_spatial_index.get('dirty', True):
            self._build_spatial_indexes()
        
        result = None
        
        try:
            # 使用KD树查找
            if self.advanced_spatial_index.get('connection_kdtree') is not None:
                result = self._kdtree_nearest_connection(position, max_distance, max_candidates)
            else:
                # 回退到网格查找
                result = self._grid_nearest_connection(position, max_distance, max_candidates)
        except Exception as e:
            if self.debug:
                print(f"优化查找失败，使用简单方法: {e}")
            result = self._simple_nearest_connection(position, max_distance)
        
        # 验证结果的有效性
        if result and isinstance(result, dict):
            # 确保所有必要字段都存在
            if not all(key in result for key in ['id', 'position']):
                if self.debug:
                    print("❌ 查找结果缺少必要字段")
                result = None
        elif result and isinstance(result, list):
            # 过滤掉无效的结果
            valid_results = []
            for item in result:
                if (item and isinstance(item, dict) and 
                    'id' in item and 'position' in item and 
                    item['position'] is not None):
                    valid_results.append(item)
            result = valid_results[0] if valid_results else None
        
        # 缓存结果
        if result:
            self._add_to_cache(cache_key, result)
        
        # 更新性能统计
        query_time = time.time() - start_time
        self.performance_stats['query_time'] += query_time
        
        return result
    
    def _kdtree_nearest_connection(self, position, max_distance, max_candidates):
        """使用KD树查找最近连接点 - 修复版"""
        kdtree = self.advanced_spatial_index.get('connection_kdtree')
        
        if not kdtree or not hasattr(self, 'connection_ids'):
            if self.debug:
                print("❌ KD树或连接点ID列表不可用")
            return None
        
        if not self.connection_ids:
            if self.debug:
                print("❌ 连接点ID列表为空")
            return None
        
        query_point = [position[0], position[1]]
        
        try:
            distances, indices = kdtree.query(
                query_point,
                k=min(max_candidates * 2, len(self.connection_ids)),
                distance_upper_bound=max_distance
            )
            
            # 处理单个结果
            if not hasattr(distances, '__len__'):
                distances = [distances]
                indices = [indices]
            
            best_connections = []
            
            for dist, idx in zip(distances, indices):
                # 检查索引和距离的有效性
                if (idx >= len(self.connection_ids) or 
                    dist > max_distance or 
                    np.isinf(dist) or 
                    np.isnan(dist)):
                    continue
                
                conn_id = self.connection_ids[idx]
                if conn_id not in self.connections:
                    continue
                
                conn_data = self.connections[conn_id]
                if not conn_data or 'position' not in conn_data:
                    continue
                
                # 创建安全的连接点信息
                safe_conn_data = {
                    'id': conn_id,
                    'position': conn_data['position'],
                    'distance': float(dist),
                    'type': conn_data.get('type', 'unknown'),
                    'paths': conn_data.get('paths', []),
                    'path_index': conn_data.get('path_index', 0),
                    'priority': conn_data.get('priority', 1),
                    'capacity': conn_data.get('capacity', 1),
                    'quality_score': conn_data.get('quality_score', 0.5)
                }
                
                # 计算综合评分
                score = self._calculate_connection_score(safe_conn_data, dist)
                safe_conn_data['score'] = score
                
                best_connections.append(safe_conn_data)
            
            # 按评分排序
            best_connections.sort(key=lambda x: -x.get('score', 0))
            
            return best_connections[0] if best_connections else None
            
        except Exception as e:
            if self.debug:
                print(f"KD树查询失败: {e}")
            return None
    
    def find_accessible_points(self, position, rrt_planner, max_candidates=5, 
                            sampling_step=10, max_distance=20.0):
        """优化版可达点查找 - 修复版"""
        start_time = time.time()
        accessible_points = []
        
        try:
            # 首先使用优化的连接点查找
            nearest_connections = self.find_nearest_connection_optimized(
                position, max_distance, max_candidates * 2
            )
            
            if nearest_connections:
                # 将单个结果转换为列表
                if not isinstance(nearest_connections, list):
                    nearest_connections = [nearest_connections]
                
                # 过滤掉 None 值和无效连接
                valid_connections = []
                for conn in nearest_connections:
                    if (conn and isinstance(conn, dict) and 
                        'id' in conn and 'position' in conn and 
                        conn['position'] is not None):
                        valid_connections.append(conn)
                
                for conn in valid_connections[:max_candidates]:
                    if rrt_planner and rrt_planner.is_path_possible(position, conn['position']):
                        # 确保所有必要的字段都存在
                        point_info = {
                            'conn_id': conn['id'],
                            'path_id': conn.get('paths', [None])[0] if conn.get('paths') else None,
                            'path_index': conn.get('path_index', 0),
                            'position': conn['position'],
                            'distance': conn.get('distance', 0),
                            'type': 'connection',
                            'quality': conn.get('quality_score', 0.5)
                        }
                        accessible_points.append(point_info)
            
            # 如果连接点不足，使用路径点KD树查找
            if len(accessible_points) < max_candidates and hasattr(self, 'path_point_kdtree') and self.path_point_kdtree:
                additional_needed = max_candidates - len(accessible_points)
                
                query_point = [position[0], position[1]]
                try:
                    distances, indices = self.path_point_kdtree.query(
                        query_point,
                        k=min(additional_needed * 3, len(self.path_point_info)),
                        distance_upper_bound=max_distance
                    )
                    
                    if not hasattr(distances, '__len__'):
                        distances = [distances]
                        indices = [indices]
                    
                    for dist, idx in zip(distances, indices):
                        if (idx < len(self.path_point_info) and 
                            dist <= max_distance and 
                            not np.isinf(dist)):
                            
                            path_id, point_idx = self.path_point_info[idx]
                            if (path_id in self.paths and 
                                point_idx < len(self.paths[path_id]['path'])):
                                
                                point = self.paths[path_id]['path'][point_idx]
                                
                                if rrt_planner and rrt_planner.is_path_possible(position, point):
                                    point_info = {
                                        'conn_id': None,
                                        'path_id': path_id,
                                        'path_index': point_idx,
                                        'position': point,
                                        'distance': dist,
                                        'type': 'path_point',
                                        'quality': self.paths[path_id].get('quality_score', 0.5)
                                    }
                                    accessible_points.append(point_info)
                                    
                                    if len(accessible_points) >= max_candidates:
                                        break
                                    
                except Exception as e:
                    if self.debug:
                        print(f"路径点查询失败: {e}")
            
            # 按质量和距离排序，过滤掉无效项
            valid_points = [p for p in accessible_points if p and isinstance(p, dict)]
            valid_points.sort(key=lambda x: (-x.get('quality', 0), x.get('distance', float('inf'))))
            
            # 更新性能统计
            if hasattr(self, 'performance_stats'):
                self.performance_stats['query_time'] += time.time() - start_time
            
            return valid_points[:max_candidates]
            
        except Exception as e:
            if self.debug:
                print(f"find_accessible_points 出错: {e}")
            return []
    
    def analyze_network_topology(self):
        """分析网络拓扑结构"""
        current_time = time.time()
        
        # 检查是否需要重新分析
        if (current_time - self.topology_metrics['last_analysis'] < 300 and 
            self.topology_metrics['connectivity_graph']):
            return self.topology_metrics
        
        print("分析骨干网络拓扑结构...")
        
        try:
            # 构建连通图
            self._build_connectivity_graph()
            
            # 计算最短路径
            self._calculate_shortest_paths()
            
            # 计算中心性指标
            self._calculate_centrality_scores()
            
            self.topology_metrics['last_analysis'] = current_time
            print("拓扑结构分析完成")
            
        except Exception as e:
            print(f"拓扑分析失败: {e}")
        
        return self.topology_metrics
    
    def suggest_optimal_route(self, start_pos, end_pos, preferences=None):
        """基于拓扑分析建议最优路由"""
        if not self.topology_metrics.get('connectivity_graph'):
            self.analyze_network_topology()
        
        preferences = preferences or {}
        
        # 找到最近的起始和结束路径
        start_candidates = self.find_accessible_points(start_pos, None, max_candidates=3)
        end_candidates = self.find_accessible_points(end_pos, None, max_candidates=3)
        
        if not start_candidates or not end_candidates:
            return None
        
        best_route = None
        best_score = float('-inf')
        
        for start_point in start_candidates:
            for end_point in end_candidates:
                start_path = start_point.get('path_id')
                end_path = end_point.get('path_id')
                
                if start_path and end_path:
                    route_score = self._evaluate_route(
                        start_path, end_path, preferences
                    )
                    
                    if route_score > best_score:
                        best_score = route_score
                        best_route = {
                            'start_point': start_point,
                            'end_point': end_point,
                            'route': self._get_route_details(start_path, end_path),
                            'score': route_score
                        }
        
        return best_route
    
    def enable_realtime_monitoring(self, interval=30):
        """启用实时监控"""
        if self.monitoring_enabled:
            return
        
        self.monitoring_enabled = True
        
        def monitor_loop():
            while self.monitoring_enabled:
                try:
                    # 检查网络健康状况
                    self._check_network_health()
                    
                    # 自适应优化
                    if self.adaptive_enabled:
                        self._adaptive_optimization()
                    
                    # 清理缓存
                    self._cleanup_cache()
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    print(f"监控循环错误: {e}")
                    time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"启用实时监控，间隔{interval}秒")
    
    def disable_realtime_monitoring(self):
        """禁用实时监控"""
        self.monitoring_enabled = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
            self.monitor_thread = None
        print("实时监控已禁用")
    
    # 缓存管理方法
    def _check_cache(self, cache_key):
        """检查缓存是否有效"""
        if cache_key not in self.query_cache:
            return False
        
        timestamp = self.cache_timestamps.get(cache_key, 0)
        current_time = time.time()
        
        if current_time - timestamp > self.cache_config['ttl']:
            del self.query_cache[cache_key]
            if cache_key in self.cache_timestamps:
                del self.cache_timestamps[cache_key]
            return False
        
        return True
    
    def _get_cache(self, cache_key):
        """获取缓存结果"""
        if cache_key in self.query_cache:
            # 更新LRU
            self.query_cache.move_to_end(cache_key)
            return self.query_cache[cache_key]
        return None
    
    def _add_to_cache(self, cache_key, result):
        """添加到缓存"""
        if len(self.query_cache) >= self.cache_config['max_cache_size']:
            self._cleanup_cache()
        
        self.query_cache[cache_key] = result
        self.cache_timestamps[cache_key] = time.time()
    
    def _cleanup_cache(self):
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = []
        
        for key, timestamp in self.cache_timestamps.items():
            if current_time - timestamp > self.cache_config['ttl']:
                expired_keys.append(key)
        
        for key in expired_keys:
            if key in self.query_cache:
                del self.query_cache[key]
            if key in self.cache_timestamps:
                del self.cache_timestamps[key]
        
        # 如果还是太多，删除最旧的一半
        if len(self.query_cache) >= self.cache_config['max_cache_size']:
            remove_count = len(self.query_cache) // 2
            for _ in range(remove_count):
                if self.query_cache:
                    oldest_key = next(iter(self.query_cache))
                    del self.query_cache[oldest_key]
                    if oldest_key in self.cache_timestamps:
                        del self.cache_timestamps[oldest_key]
    
    # 路径质量评估方法
    def _evaluate_path_quality(self, path):
        """综合评估路径质量"""
        if not path or len(path) < 2:
            return 0
        
        path_key = self._get_path_cache_key(path)
        if path_key in self.path_quality_cache:
            return self.path_quality_cache[path_key]
        
        # 1. 长度评分
        length = self._calculate_path_length(path)
        direct_distance = self._calculate_distance(path[0], path[-1])
        length_score = direct_distance / (length + 0.1) if length > 0 else 0
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
        
        self.path_quality_cache[path_key] = quality_score
        return quality_score
    
    # 拓扑分析辅助方法
    def _build_connectivity_graph(self):
        """构建连通图"""
        graph = defaultdict(set)
        
        for conn_data in self.connections.values():
            connected_paths = conn_data.get('paths', [])
            
            for i, path1 in enumerate(connected_paths):
                for j, path2 in enumerate(connected_paths):
                    if i != j and path1 in self.paths and path2 in self.paths:
                        graph[path1].add(path2)
                        graph[path2].add(path1)
        
        self.topology_metrics['connectivity_graph'] = dict(graph)
    
    def _calculate_shortest_paths(self):
        """计算关键点之间的最短路径"""
        graph = self.topology_metrics['connectivity_graph']
        paths = {}
        all_paths = list(self.paths.keys())
        
        for start_path in all_paths:
            paths[start_path] = {}
            
            # BFS找最短路径
            queue = [(start_path, 0, [start_path])]
            visited = {start_path}
            
            while queue:
                current_path, distance, path_route = queue.pop(0)
                paths[start_path][current_path] = {
                    'distance': distance,
                    'route': path_route
                }
                
                for neighbor in graph.get(current_path, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, distance + 1, path_route + [neighbor]))
        
        self.topology_metrics['shortest_paths'] = paths
    
    def _calculate_centrality_scores(self):
        """计算中心性评分"""
        graph = self.topology_metrics['connectivity_graph']
        centrality = {}
        
        for path_id in self.paths.keys():
            # 度中心性
            degree = len(graph.get(path_id, []))
            
            # 接近中心性的简化计算
            shortest_paths = self.topology_metrics['shortest_paths'].get(path_id, {})
            total_distance = sum(
                data.get('distance', float('inf')) 
                for data in shortest_paths.values()
            )
            
            closeness = 1.0 / (total_distance + 1)
            
            # 综合中心性评分
            centrality[path_id] = {
                'degree': degree,
                'closeness': closeness,
                'combined': degree * 0.6 + closeness * 0.4
            }
        
        self.topology_metrics['centrality_scores'] = centrality
    
    # 监控和优化方法
    def _check_network_health(self):
        """检查网络健康状况"""
        issues = []
        
        # 检查高负载路径
        for path_id, path_data in self.paths.items():
            utilization = path_data.get('utilization', 0)
            if utilization > 0.9:
                issues.append(f"路径 {path_id} 负载过高: {utilization:.2f}")
        
        # 检查低质量路径
        low_quality_paths = [
            path_id for path_id, data in self.paths.items()
            if data.get('quality_score', 1.0) < 0.3
        ]
        
        if low_quality_paths:
            issues.append(f"发现 {len(low_quality_paths)} 条低质量路径")
        
        self.performance_stats['health_issues'] = issues
        if issues:
            print(f"网络健康检查发现 {len(issues)} 个问题")
    
    def _adaptive_optimization(self):
        """自适应优化"""
        # 动态调整缓存大小
        query_rate = len(self.query_cache) / max(1, time.time() - self.performance_stats.get('start_time', time.time()))
        
        if query_rate > 10:  # 高查询率
            self.cache_config['max_cache_size'] = min(2000, int(self.cache_config['max_cache_size'] * 1.2))
        elif query_rate < 2:  # 低查询率
            self.cache_config['max_cache_size'] = max(500, int(self.cache_config['max_cache_size'] * 0.8))
        
        # 动态调整空间索引
        if len(self.connections) > 1000 and self.advanced_spatial_index['dirty']:
            self._build_spatial_indexes()
    
    def update_traffic_flow(self, path_id, flow_change):
        """更新路径交通流量"""
        if path_id in self.paths:
            current_flow = self.paths[path_id].get('traffic_flow', 0)
            capacity = self.paths[path_id].get('capacity', 1)
            
            new_flow = max(0, current_flow + flow_change)
            self.paths[path_id]['traffic_flow'] = new_flow
            self.paths[path_id]['utilization'] = new_flow / max(capacity, 1)
    
    def get_path_segment(self, path_id, start_index, end_index):
        """获取路径段"""
        if path_id not in self.paths:
            return None
        
        path = self.paths[path_id]['path']
        if start_index < 0 or end_index >= len(path) or start_index >= end_index:
            return None
        
        return path[start_index:end_index + 1]
    
    # 工具方法
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
    
    def _calculate_distance(self, pos1, pos2):
        """计算两点间距离"""
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
    
    def _create_planner(self):
        """创建RRT规划器"""
        if hasattr(self.env, 'rrt_planner') and self.env.rrt_planner:
            return self.env.rrt_planner
        
        try:
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
    
    def get_performance_stats(self):
        """获取性能统计信息"""
        stats = self.performance_stats.copy()
        
        # 计算缓存命中率
        total_requests = stats['cache_hits'] + stats['cache_misses']
        if total_requests > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / total_requests
        else:
            stats['cache_hit_rate'] = 0.0
        
        return stats
    
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
    
    def _simple_nearest_connection(self, position, max_distance):
        """简单的最近连接点查找 - 修复版"""
        if not self.connections:
            return None
        
        best_connection = None
        best_distance = float('inf')
        
        for conn_id, conn_data in self.connections.items():
            try:
                if not conn_data or 'position' not in conn_data:
                    continue
                
                conn_position = conn_data['position']
                if not conn_position or len(conn_position) < 2:
                    continue
                    
                dist = self._calculate_distance(position, conn_position)
                
                if dist <= max_distance and dist < best_distance:
                    best_distance = dist
                    # 创建安全的连接点信息
                    best_connection = {
                        'id': conn_id,
                        'position': conn_position,
                        'distance': float(dist),
                        'type': conn_data.get('type', 'unknown'),
                        'paths': conn_data.get('paths', []),
                        'path_index': conn_data.get('path_index', 0),
                        'priority': conn_data.get('priority', 1),
                        'capacity': conn_data.get('capacity', 1),
                        'quality_score': conn_data.get('quality_score', 0.5)
                    }
            except Exception as e:
                if self.debug:
                    print(f"处理连接点 {conn_id} 时出错: {e}")
                continue
        
        return best_connection

    
    def _build_simple_spatial_index(self):
        """简单的空间索引实现"""
        self.connection_kdtree = None
        self.path_point_kdtree = None
        print("使用简单空间索引")
    
    def _calculate_connection_score(self, conn_data, distance):
        """计算连接点综合评分"""
        distance_score = 1.0 / (1.0 + distance / 10.0)
        quality_score = conn_data.get('quality_score', 0.5)
        priority_score = conn_data.get('priority', 1) / 3.0
        capacity_score = min(1.0, conn_data.get('capacity', 1) / 5.0)
        
        return (distance_score * 0.4 + quality_score * 0.3 + 
                priority_score * 0.2 + capacity_score * 0.1)
    
    # 其他必要的辅助方法的简化实现
    def _grid_nearest_connection(self, position, max_distance, max_candidates):
        """使用网格索引查找最近连接点"""
        return self._simple_nearest_connection(position, max_distance)
    
    def _determine_hierarchy_level(self, start_point, end_point):
        """确定路径层次等级"""
        start_priority = start_point.get('priority', 1)
        end_priority = end_point.get('priority', 1)
        avg_priority = (start_priority + end_priority) / 2
        
        if avg_priority >= 2.5:
            return 'primary'
        elif avg_priority >= 1.5:
            return 'secondary'
        else:
            return 'auxiliary'
    
    def _build_path_hierarchy(self):
        """建立路径层次结构"""
        self.path_hierarchy = {'primary': [], 'secondary': [], 'auxiliary': []}
        
        for path_id, path_data in self.paths.items():
            hierarchy_level = path_data.get('hierarchy_level', 'auxiliary')
            if hierarchy_level in self.path_hierarchy:
                self.path_hierarchy[hierarchy_level].append(path_id)
    
    def _build_path_graph(self):
        """构建路径连接图"""
        self.path_graph = {}
        
        for path_id in self.paths.keys():
            self.path_graph[path_id] = []
            
        # 通过连接点建立路径间的连接关系
        for conn_data in self.connections.values():
            connected_paths = conn_data.get('paths', [])
            for i, path1 in enumerate(connected_paths):
                for j, path2 in enumerate(connected_paths):
                    if i != j and path1 in self.path_graph and path2 not in self.path_graph[path1]:
                        self.path_graph[path1].append(path2)
    
    def _optimize_all_paths_advanced(self):
        """高级路径优化"""
        start_time = time.time()
        
        for path_id, path_data in self.paths.items():
            if not path_data.get('optimized', False):
                original_path = path_data['path']
                
                # 执行多级优化
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
        
        # 简化版本的优化
        return self._smooth_path(path)
    
    def _smooth_path(self, path, iterations=2):
        """路径平滑"""
        if len(path) <= 2:
            return path
        
        smoothed = list(path)
        
        for _ in range(iterations):
            new_smoothed = [smoothed[0]]
            
            for i in range(1, len(smoothed) - 1):
                prev = smoothed[i-1]
                curr = smoothed[i]
                next_p = smoothed[i+1]
                
                # 简单的平均平滑
                x = (prev[0] + curr[0] + next_p[0]) / 3
                y = (prev[1] + curr[1] + next_p[1]) / 3
                theta = curr[2] if len(curr) > 2 else 0
                
                new_smoothed.append((x, y, theta))
            
            new_smoothed.append(smoothed[-1])
            smoothed = new_smoothed
        
        return smoothed
    
    # 其他简化实现的方法
    def _evaluate_path_smoothness(self, path):
        """评估路径平滑度"""
        if len(path) < 3:
            return 1.0
        
        total_angle_change = 0
        for i in range(1, len(path) - 1):
            angle_change = self._calculate_angle_change(path[i-1], path[i], path[i+1])
            total_angle_change += angle_change
        
        avg_angle_change = total_angle_change / max(1, len(path) - 2)
        return max(0, 1.0 - avg_angle_change / math.pi)
    
    def _calculate_angle_change(self, p1, p2, p3):
        """计算角度变化"""
        v1 = [p2[0] - p1[0], p2[1] - p1[1]]
        v2 = [p3[0] - p2[0], p3[1] - p2[1]]
        
        len1 = math.sqrt(v1[0]**2 + v1[1]**2)
        len2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if len1 < 0.001 or len2 < 0.001:
            return 0
        
        cos_angle = (v1[0]*v2[0] + v1[1]*v2[1]) / (len1 * len2)
        cos_angle = max(-1, min(1, cos_angle))
        
        return math.acos(cos_angle)
    
    def _evaluate_turning_complexity(self, path):
        """评估转弯复杂度"""
        if len(path) < 3:
            return 1.0
        
        sharp_turns = 0
        for i in range(1, len(path) - 1):
            angle = self._calculate_angle_change(path[i-1], path[i], path[i+1])
            if angle > math.pi / 4:  # 45度以上为急转弯
                sharp_turns += 1
        
        total_segments = len(path) - 2
        return max(0, 1.0 - (sharp_turns / max(1, total_segments)))
    
    def _evaluate_path_clearance(self, path):
        """评估路径间隙"""
        # 简化实现：假设所有路径都有基本的安全间隙
        return 0.7
    
    def _evaluate_traffic_compatibility(self, path):
        """评估交通兼容性"""
        # 简化实现：基于路径长度的兼容性评估
        length = self._calculate_path_length(path)
        return min(1.0, length / 100.0)
    
    def _get_path_cache_key(self, path):
        """生成路径缓存键"""
        if len(path) < 2:
            return str(path)
        
        key_points = [path[0], path[-1]]
        if len(path) > 4:
            key_points.extend([path[len(path)//3], path[2*len(path)//3]])
        
        simplified = []
        for point in key_points:
            simplified.append(f"{point[0]:.1f},{point[1]:.1f}")
        
        return "|".join(simplified)
    
    def _is_good_connection_point(self, path, index):
        """判断是否为好的连接点位置"""
        if index <= 0 or index >= len(path) - 1:
            return False
        
        # 简化判断：每隔一定距离就是好的连接点
        return True
    
    def _evaluate_connection_quality(self, path, index):
        """评估连接点质量"""
        if index <= 0 or index >= len(path) - 1:
            return 0
        
        # 基于位置的简化评分
        position_ratio = index / len(path)
        return 1.0 - abs(position_ratio - 0.5) * 2
    
    def _calculate_speed_limit(self, path):
        """基于路径质量计算速度限制"""
        if not path or len(path) < 2:
            return 1.0
        
        quality = self._evaluate_path_quality(path)
        return 0.3 + 0.7 * quality
    
    def _estimate_path_capacity(self, path):
        """基于路径特征估计容量"""
        if not path:
            return 1
        
        length = self._calculate_path_length(path)
        base_capacity = max(1, int(length / 20))
        
        quality = self._evaluate_path_quality(path)
        capacity_multiplier = 0.5 + quality
        
        return max(1, int(base_capacity * capacity_multiplier))
    
    def _evaluate_route(self, start_path, end_path, preferences):
        """评估路由质量"""
        # 简化实现
        return 0.7
    
    def _get_route_details(self, start_path, end_path):
        """获取路由详细信息"""
        return {
            'path_sequence': [start_path, end_path],
            'total_hops': 1,
            'estimated_length': 100,
            'quality_breakdown': {},
            'bottlenecks': []
        }
    def get_path_segment(self, path_id, start_index, end_index):
        """获取路径段 - 修复版"""
        try:
            # 检查路径是否存在
            if path_id not in self.paths:
                print(f"⚠️ 路径ID {path_id} 不存在于骨干网络中")
                return None
            
            path_data = self.paths[path_id]
            if not path_data or 'path' not in path_data:
                print(f"⚠️ 路径 {path_id} 数据无效")
                return None
            
            path = path_data['path']
            if not path or len(path) == 0:
                print(f"⚠️ 路径 {path_id} 为空")
                return None
            
            # 修正索引范围
            max_index = len(path) - 1
            start_index = max(0, min(start_index, max_index))
            end_index = max(start_index, min(end_index, max_index))
            
            # 确保有效的段
            if start_index == end_index:
                return [path[start_index]]
            elif start_index < end_index:
                return path[start_index:end_index + 1]
            else:
                # 如果索引顺序颠倒，反转
                return path[end_index:start_index + 1][::-1]
                
        except Exception as e:
            print(f"❌ get_path_segment 出错: {e}")
            return None

# 保持向后兼容性的类别名
BackbonePathNetwork = OptimizedBackbonePathNetwork