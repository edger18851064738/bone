class VehicleTask:
    """车辆任务类"""
    
    def __init__(self, task_id, task_type, start, goal, priority=1):
        self.task_id = task_id
        self.task_type = task_type  # 'to_loading', 'to_unloading', 'to_initial'
        self.start = start
        self.goal = goal
        self.priority = priority
        self.status = 'pending'  # 'pending', 'assigned', 'in_progress', 'completed', 'failed'
        self.assigned_vehicle = None
        self.progress = 0.0  # 0.0 - 1.0
        self.path = None
        self.estimated_time = 0
        self.start_time = 0
        self.completion_time = 0
    
    def to_dict(self):
        """转换为字典表示"""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'start': self.start,
            'goal': self.goal,
            'priority': self.priority,
            'status': self.status,
            'progress': self.progress,
            'assigned_vehicle': self.assigned_vehicle,
            'estimated_time': self.estimated_time,
            'start_time': self.start_time,
            'completion_time': self.completion_time
        }


class VehicleTask:
    """车辆任务类 - 增强版"""
    
    def __init__(self, task_id, task_type, start, goal, priority=1, 
                 loading_point_id=None, unloading_point_id=None):
        self.task_id = task_id
        self.task_type = task_type  # 'to_loading', 'to_unloading', 'to_initial'
        self.start = start
        self.goal = goal
        self.priority = priority
        self.status = 'pending'  # 'pending', 'assigned', 'in_progress', 'completed', 'failed'
        self.assigned_vehicle = None
        self.progress = 0.0  # 0.0 - 1.0
        self.path = None
        self.estimated_time = 0
        self.start_time = 0
        self.completion_time = 0
        
        # 新增: 装载点和卸载点ID
        self.loading_point_id = loading_point_id  # 指定使用的装载点ID
        self.unloading_point_id = unloading_point_id  # 指定使用的卸载点ID
        
        # 新增: 保存路径结构
        self.path_structure = {
            'entry_point': None,  # 进入骨干网络的点
            'exit_point': None,   # 离开骨干网络的点
            'backbone_segment': None,  # 使用的骨干路径段
            'to_backbone_path': None,  # 从起点到骨干的路径
            'backbone_path': None,     # 骨干网络中的路径
            'from_backbone_path': None  # 从骨干到终点的路径
        }
    
    def to_dict(self):
        """转换为字典表示"""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'start': self.start,
            'goal': self.goal,
            'priority': self.priority,
            'status': self.status,
            'progress': self.progress,
            'assigned_vehicle': self.assigned_vehicle,
            'estimated_time': self.estimated_time,
            'start_time': self.start_time,
            'completion_time': self.completion_time,
            'loading_point_id': self.loading_point_id,
            'unloading_point_id': self.unloading_point_id
        }


class VehicleScheduler:
    """车辆调度器，管理任务分配和调度 - 增强版"""
    
    def __init__(self, env, path_planner=None, backbone_network=None):
        self.env = env
        self.path_planner = path_planner
        self.backbone_network = backbone_network  # 新增: 骨干路径网络
        self.tasks = {}  # 任务字典 {task_id: task}
        self.task_queues = {}  # 车辆任务队列 {vehicle_id: [task_ids]}
        self.vehicle_statuses = {}  # 车辆状态 {vehicle_id: status}
        self.task_counter = 0  # 任务计数器
        self.mission_templates = {}  # 任务模板 {template_id: [tasks]}
        
        # 统计信息
        self.stats = {
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_distance': 0,
            'total_time': 0,
            'vehicle_utilization': {}  # {vehicle_id: utilization_rate}
        }
        
        # 新增: 点位使用情况跟踪
        self.loading_point_usage = {}  # {point_id: [vehicle_ids]}
        self.unloading_point_usage = {}  # {point_id: [vehicle_ids]}
        
        # 新增: 路径结构缓存
        self.path_structure_cache = {}  # {(start, goal): structure}
    
    def set_path_planner(self, path_planner):
        """设置路径规划器"""
        self.path_planner = path_planner
    
    def set_backbone_network(self, backbone_network):
        """设置骨干路径网络"""
        self.backbone_network = backbone_network
    
    def initialize_vehicles(self):
        """初始化车辆状态"""
        for vehicle_id, vehicle in self.env.vehicles.items():
            self.vehicle_statuses[vehicle_id] = {
                'status': 'idle',  # 'idle', 'moving', 'loading', 'unloading'
                'position': vehicle.get('position', (0, 0, 0)),
                'load': 0,
                'max_load': vehicle.get('max_load', 100),
                'current_task': None,
                'completed_tasks': 0,
                'total_distance': 0,
                'total_time': 0,
                'utilization_rate': 0.0,
                'active_time': 0,
                'idle_time': 0,
                'preferred_loading_point': None,  # 新增: 首选装载点
                'preferred_unloading_point': None  # 新增: 首选卸载点
            }
            
            # 初始化任务队列
            self.task_queues[vehicle_id] = []
    
    def create_mission_template(self, template_id, loading_point=None, unloading_point=None):
        """创建任务模板（装载-卸载-返回循环）"""
        loading_point_id = None
        unloading_point_id = None
        
        # 如果提供了特定装载点
        if loading_point is not None:
            # 检查是否提供了ID
            if isinstance(loading_point, int):
                loading_point_id = loading_point
                if 0 <= loading_point_id < len(self.env.loading_points):
                    loading_point = self.env.loading_points[loading_point_id]
                else:
                    loading_point = None
            else:
                # 查找点的ID
                for i, point in enumerate(self.env.loading_points):
                    if point == loading_point:
                        loading_point_id = i
                        break
        
        # 如果没有指定或找不到指定的装载点，使用默认的
        if loading_point is None and self.env.loading_points:
            loading_point = self.env.loading_points[0]
            loading_point_id = 0
        
        # 类似处理卸载点
        if unloading_point is not None:
            if isinstance(unloading_point, int):
                unloading_point_id = unloading_point
                if 0 <= unloading_point_id < len(self.env.unloading_points):
                    unloading_point = self.env.unloading_points[unloading_point_id]
                else:
                    unloading_point = None
            else:
                for i, point in enumerate(self.env.unloading_points):
                    if point == unloading_point:
                        unloading_point_id = i
                        break
        
        if unloading_point is None and self.env.unloading_points:
            unloading_point = self.env.unloading_points[0]
            unloading_point_id = 0
        
        if not loading_point or not unloading_point:
            return False
        
        # 创建三段任务模板 - 增加点位ID信息
        template = [
            {
                'task_type': 'to_loading',
                'goal': (loading_point[0], loading_point[1], 0),
                'priority': 1,
                'loading_point_id': loading_point_id,
                'unloading_point_id': unloading_point_id
            },
            {
                'task_type': 'to_unloading',
                'goal': (unloading_point[0], unloading_point[1], 0),
                'priority': 2,
                'loading_point_id': loading_point_id,
                'unloading_point_id': unloading_point_id
            },
            {
                'task_type': 'to_initial',
                'goal': None,  # 将在分配时设置为车辆初始位置
                'priority': 1,
                'loading_point_id': loading_point_id,
                'unloading_point_id': unloading_point_id
            }
        ]
        
        self.mission_templates[template_id] = template
        return True
    
    def create_mission_with_specific_points(self, template_id, loading_point_id, unloading_point_id):
        """创建使用特定装载点和卸载点的任务模板"""
        # 验证点位ID是否有效
        if loading_point_id < 0 or loading_point_id >= len(self.env.loading_points):
            return False
        
        if unloading_point_id < 0 or unloading_point_id >= len(self.env.unloading_points):
            return False
        
        # 获取对应的点位
        loading_point = self.env.loading_points[loading_point_id]
        unloading_point = self.env.unloading_points[unloading_point_id]
        
        # 使用原有方法创建模板
        return self.create_mission_template(template_id, loading_point, unloading_point)
    
    def assign_mission(self, vehicle_id, template_id):
        """为车辆分配任务模板"""
        if vehicle_id not in self.vehicle_statuses:
            return False
        
        if template_id not in self.mission_templates:
            return False
        
        # 获取模板
        template = self.mission_templates[template_id]
        
        # 获取车辆初始位置
        vehicle = self.env.vehicles.get(vehicle_id)
        if not vehicle:
            return False
        
        initial_position = vehicle.get('initial_position')
        if not initial_position:
            initial_position = vehicle.get('position', (0, 0, 0))
        
        # 创建任务并添加到队列
        current_position = self.vehicle_statuses[vehicle_id]['position']
        
        for task_template in template:
            task_id = f"task_{self.task_counter}"
            self.task_counter += 1
            
            # 创建任务 - 包含点位ID信息
            task = VehicleTask(
                task_id,
                task_template['task_type'],
                current_position,  # 起点设为前一任务的终点
                task_template['goal'] if task_template['goal'] else initial_position,
                task_template['priority'],
                task_template.get('loading_point_id'),
                task_template.get('unloading_point_id')
            )
            
            # 更新下一任务的起点
            current_position = task.goal
            
            # 添加到任务字典
            self.tasks[task_id] = task
            
            # 添加到车辆任务队列
            self.task_queues[vehicle_id].append(task_id)
            
            # 更新点位使用情况
            if task.loading_point_id is not None:
                if task.loading_point_id not in self.loading_point_usage:
                    self.loading_point_usage[task.loading_point_id] = []
                if vehicle_id not in self.loading_point_usage[task.loading_point_id]:
                    self.loading_point_usage[task.loading_point_id].append(vehicle_id)
            
            if task.unloading_point_id is not None:
                if task.unloading_point_id not in self.unloading_point_usage:
                    self.unloading_point_usage[task.unloading_point_id] = []
                if vehicle_id not in self.unloading_point_usage[task.unloading_point_id]:
                    self.unloading_point_usage[task.unloading_point_id].append(vehicle_id)
        
        # 如果车辆空闲，立即开始执行任务
        if self.vehicle_statuses[vehicle_id]['status'] == 'idle':
            self._start_next_task(vehicle_id)
        
        return True
    
    def assign_optimal_mission(self, vehicle_id):
        """基于距离、负载等因素为车辆分配最优的装载点和卸载点任务"""
        if vehicle_id not in self.vehicle_statuses:
            return False
        
        vehicle_pos = self.vehicle_statuses[vehicle_id]['position']
        vehicle_load = self.vehicle_statuses[vehicle_id]['load']
        vehicle_max_load = self.vehicle_statuses[vehicle_id]['max_load']
        
        # 创建唯一的模板ID
        template_id = f"optimal_mission_{vehicle_id}_{self.task_counter}"
        
        # 如果车辆负载低，分配前往装载点的任务
        if vehicle_load < vehicle_max_load * 0.5:
            # 寻找最佳装载点 - 考虑距离和使用情况
            best_loading_id = self._find_optimal_loading_point(vehicle_id, vehicle_pos)
            best_unloading_id = self._find_optimal_unloading_point(vehicle_id)
            
            # 创建任务模板
            if self.create_mission_with_specific_points(template_id, best_loading_id, best_unloading_id):
                return self.assign_mission(vehicle_id, template_id)
        else:
            # 寻找最佳卸载点
            best_unloading_id = self._find_optimal_unloading_point(vehicle_id, vehicle_pos)
            best_loading_id = self._find_optimal_loading_point(vehicle_id)
            
            # 创建任务模板
            if self.create_mission_with_specific_points(template_id, best_loading_id, best_unloading_id):
                return self.assign_mission(vehicle_id, template_id)
        
        return False
    
    def _find_optimal_loading_point(self, vehicle_id, position=None):
        """寻找最优装载点"""
        if not self.env.loading_points:
            return 0
        
        # 使用车辆位置
        if position is None and vehicle_id in self.vehicle_statuses:
            position = self.vehicle_statuses[vehicle_id]['position']
        
        # 如果车辆有首选装载点，优先使用
        if (vehicle_id in self.vehicle_statuses and 
            self.vehicle_statuses[vehicle_id]['preferred_loading_point'] is not None):
            return self.vehicle_statuses[vehicle_id]['preferred_loading_point']
        
        best_point_id = 0
        best_score = float('-inf')
        
        for i, point in enumerate(self.env.loading_points):
            # 计算距离分数 - 距离越近分数越高
            distance = self._calculate_distance(position, point)
            distance_score = 1000 / (distance + 1)  # 避免除零
            
            # 计算使用情况分数 - 使用车辆越少分数越高
            usage_count = len(self.loading_point_usage.get(i, []))
            usage_score = 1000 / (usage_count + 1)
            
            # 总分 - 可以调整权重
            score = distance_score * 0.7 + usage_score * 0.3
            
            if score > best_score:
                best_score = score
                best_point_id = i
        
        return best_point_id
    
    def _find_optimal_unloading_point(self, vehicle_id, position=None):
        """寻找最优卸载点"""
        if not self.env.unloading_points:
            return 0
        
        # 使用车辆位置
        if position is None and vehicle_id in self.vehicle_statuses:
            position = self.vehicle_statuses[vehicle_id]['position']
        
        # 如果车辆有首选卸载点，优先使用
        if (vehicle_id in self.vehicle_statuses and 
            self.vehicle_statuses[vehicle_id]['preferred_unloading_point'] is not None):
            return self.vehicle_statuses[vehicle_id]['preferred_unloading_point']
        
        best_point_id = 0
        best_score = float('-inf')
        
        for i, point in enumerate(self.env.unloading_points):
            # 计算距离分数
            distance = self._calculate_distance(position, point)
            distance_score = 1000 / (distance + 1)
            
            # 计算使用情况分数
            usage_count = len(self.unloading_point_usage.get(i, []))
            usage_score = 1000 / (usage_count + 1)
            
            # 总分
            score = distance_score * 0.7 + usage_score * 0.3
            
            if score > best_score:
                best_score = score
                best_point_id = i
        
        return best_point_id
    
    def _start_next_task(self, vehicle_id):
        """开始执行车辆的下一个任务"""
        if vehicle_id not in self.task_queues or not self.task_queues[vehicle_id]:
            # 没有更多任务
            return False
        
        # 获取下一个任务
        task_id = self.task_queues[vehicle_id][0]
        task = self.tasks[task_id]
        
        # 更新任务状态
        task.status = 'in_progress'
        task.assigned_vehicle = vehicle_id
        task.start_time = self.env.current_time if hasattr(self.env, 'current_time') else 0
        
        # 更新车辆状态
        self.vehicle_statuses[vehicle_id]['status'] = 'moving'
        self.vehicle_statuses[vehicle_id]['current_task'] = task_id
        
        # 规划路径 - 增强版，集成骨干网络
        if self.path_planner:
            task.path = self._plan_path_with_backbone(
                vehicle_id,
                self.vehicle_statuses[vehicle_id]['position'],
                task.goal
            )
            
            # 更新车辆路径
            if vehicle_id in self.env.vehicles and task.path:
                self.env.vehicles[vehicle_id]['path'] = task.path
                self.env.vehicles[vehicle_id]['path_index'] = 0
                self.env.vehicles[vehicle_id]['progress'] = 0.0
        
        return True
    
    def _plan_path_with_backbone(self, vehicle_id, start, goal):
        """使用骨干网络规划路径"""
        if not self.path_planner:
            return None
        
        # 如果有骨干网络且规划器支持，使用增强规划
        if self.backbone_network and hasattr(self.path_planner, 'set_backbone_network'):
            # 确保规划器已设置骨干网络
            self.path_planner.set_backbone_network(self.backbone_network)
            
            # 规划路径，启用骨干网络
            path = self.path_planner.plan_path(vehicle_id, start, goal, use_backbone=True)
            
            # 如果使用骨干网络规划成功，分析并保存路径结构
            if path:
                path_structure = self._analyze_path_structure(path)
                
                # 获取当前任务
                if vehicle_id in self.vehicle_statuses:
                    task_id = self.vehicle_statuses[vehicle_id]['current_task']
                    if task_id in self.tasks:
                        self.tasks[task_id].path_structure = path_structure
                
                return path
        
        # 回退到普通路径规划
        return self.path_planner.plan_path(vehicle_id, start, goal)
    
    def _analyze_path_structure(self, path):
        """分析路径，识别骨干网络部分"""
        if not self.backbone_network or not path or len(path) < 2:
            return {
                'entry_point': None,
                'exit_point': None,
                'backbone_segment': None,
                'to_backbone_path': path,
                'backbone_path': None,
                'from_backbone_path': None
            }
        
        # 找出路径进入和离开骨干网络的点
        entry_point = None
        entry_index = -1
        exit_point = None
        exit_index = -1
        backbone_segment = None
        
        # 遍历路径点，寻找最接近骨干网络的点
        for i, point in enumerate(path):
            # 查找最近的骨干连接点
            nearest_conn = self.backbone_network.find_nearest_connection(point, max_distance=5.0)
            
            if nearest_conn:
                if entry_point is None:
                    # 找到进入点
                    entry_point = nearest_conn
                    entry_index = i
                    backbone_segment = nearest_conn.get('path_id')
                else:
                    # 更新离开点
                    exit_point = nearest_conn
                    exit_index = i
        
        # 如果找到了入口和出口，划分路径
        if entry_index >= 0 and exit_index > entry_index:
            to_backbone = path[:entry_index+1]
            backbone = path[entry_index:exit_index+1]
            from_backbone = path[exit_index:]
            
            return {
                'entry_point': entry_point,
                'exit_point': exit_point,
                'backbone_segment': backbone_segment,
                'to_backbone_path': to_backbone,
                'backbone_path': backbone,
                'from_backbone_path': from_backbone
            }
        elif entry_index >= 0:
            # 只找到入口
            to_backbone = path[:entry_index+1]
            backbone = path[entry_index:]
            
            return {
                'entry_point': entry_point,
                'exit_point': None,
                'backbone_segment': backbone_segment,
                'to_backbone_path': to_backbone,
                'backbone_path': backbone,
                'from_backbone_path': None
            }
        else:
            # 没找到骨干网络连接
            return {
                'entry_point': None,
                'exit_point': None,
                'backbone_segment': None,
                'to_backbone_path': path,
                'backbone_path': None,
                'from_backbone_path': None
            }
    
    def update(self, time_delta):
        """更新所有车辆状态和任务进度"""
        for vehicle_id, status in self.vehicle_statuses.items():
            self._update_vehicle(vehicle_id, time_delta)
    
    def _update_vehicle(self, vehicle_id, time_delta):
        """更新单个车辆状态"""
        if vehicle_id not in self.vehicle_statuses:
            return
        
        status = self.vehicle_statuses[vehicle_id]
        
        # 更新时间统计
        if status['status'] == 'idle':
            status['idle_time'] += time_delta
        else:
            status['active_time'] += time_delta
        
        # 计算利用率
        total_time = status['active_time'] + status['idle_time']
        if total_time > 0:
            status['utilization_rate'] = status['active_time'] / total_time
        
        # 更新车辆位置和状态
        vehicle = self.env.vehicles.get(vehicle_id)
        if not vehicle:
            return
        
        # 获取当前任务
        current_task_id = status['current_task']
        if not current_task_id or current_task_id not in self.tasks:
            # 如果没有当前任务，但有待执行任务，开始下一个任务
            if self.task_queues.get(vehicle_id) and status['status'] == 'idle':
                self._start_next_task(vehicle_id)
            return
        
        current_task = self.tasks[current_task_id]
        
        # 根据状态更新
        if status['status'] == 'moving':
            # 正在移动，更新位置
            if 'path' in vehicle and vehicle['path']:
                # 使用路径更新位置
                path_index = vehicle.get('path_index', 0)
                path = vehicle['path']
                
                if path_index >= len(path):
                    # 到达目标
                    self._handle_task_completion(vehicle_id, current_task_id)
                else:
                    # 更新路径进度
                    progress = vehicle.get('progress', 0.0)
                    
                    # 模拟车辆移动
                    speed = 0.05  # 单位时间移动距离
                    progress += speed * time_delta
                    
                    if progress >= 1.0:
                        # 移动到下一个路径点
                        path_index += 1
                        progress = 0.0
                        
                        if path_index >= len(path):
                            # 到达目标
                            status['position'] = current_task.goal
                            vehicle['position'] = current_task.goal
                            self._handle_task_completion(vehicle_id, current_task_id)
                        else:
                            # 更新位置到新的路径点
                            status['position'] = path[path_index]
                            vehicle['position'] = path[path_index]
                            vehicle['path_index'] = path_index
                            vehicle['progress'] = progress
                            
                            # 新增: 检查是否进入或离开骨干网络
                            self._check_backbone_transition(vehicle_id, current_task, path_index)
                    else:
                        # 在当前路径段内插值计算位置
                        current_point = path[path_index]
                        next_point = path[path_index + 1] if path_index + 1 < len(path) else current_task.goal
                        
                        # 线性插值
                        x = current_point[0] + (next_point[0] - current_point[0]) * progress
                        y = current_point[1] + (next_point[1] - current_point[1]) * progress
                        theta = math.atan2(next_point[1] - current_point[1], next_point[0] - current_point[0])
                        
                        status['position'] = (x, y, theta)
                        vehicle['position'] = status['position']
                        vehicle['progress'] = progress
                    
                    # 更新任务进度
                    path_length = len(path)
                    current_task.progress = min(1.0, (path_index + progress) / path_length)
        
        elif status['status'] == 'loading':
            # 正在装载，等待完成
            loading_time = 50  # 装载时间
            vehicle['loading_progress'] = vehicle.get('loading_progress', 0) + time_delta
            
            if vehicle['loading_progress'] >= loading_time:
                # 装载完成
                status['load'] = status['max_load']
                vehicle['load'] = status['max_load']
                
                # 完成当前任务
                self._complete_task(vehicle_id, current_task_id)
                
                # 开始下一个任务（前往卸载点）
                self._start_next_task(vehicle_id)
        
        elif status['status'] == 'unloading':
            # 正在卸载，等待完成
            unloading_time = 30  # 卸载时间
            vehicle['unloading_progress'] = vehicle.get('unloading_progress', 0) + time_delta
            
            if vehicle['unloading_progress'] >= unloading_time:
                # 卸载完成
                status['load'] = 0
                vehicle['load'] = 0
                
                # 完成当前任务
                self._complete_task(vehicle_id, current_task_id)
                
                # 开始下一个任务（返回起点）
                self._start_next_task(vehicle_id)
    
    def _check_backbone_transition(self, vehicle_id, task, path_index):
        """检查车辆是否进入或离开骨干网络，并进行相应处理"""
        if not task.path_structure:
            return
        
        # 提取路径结构信息
        entry_index = -1
        exit_index = -1
        
        if task.path_structure['to_backbone_path']:
            entry_index = len(task.path_structure['to_backbone_path']) - 1
        
        if task.path_structure['backbone_path'] and task.path_structure['from_backbone_path']:
            exit_index = len(task.path_structure['to_backbone_path']) + len(task.path_structure['backbone_path']) - 1
        
        # 检查是否进入骨干网络
        if entry_index >= 0 and path_index == entry_index:
            # 已进入骨干网络
            if self.backbone_network and task.path_structure['backbone_segment']:
                # 更新骨干网络流量
                path_id = task.path_structure['backbone_segment']
                self.backbone_network.update_traffic_flow(path_id, 1)  # 增加流量
        
        # 检查是否离开骨干网络
        if exit_index >= 0 and path_index == exit_index:
            # 已离开骨干网络
            if self.backbone_network and task.path_structure['backbone_segment']:
                # 更新骨干网络流量
                path_id = task.path_structure['backbone_segment']
                self.backbone_network.update_traffic_flow(path_id, -1)  # 减少流量
    
    def _handle_task_completion(self, vehicle_id, task_id):
        """处理任务完成"""
        if task_id not in self.tasks or vehicle_id not in self.vehicle_statuses:
            return
        
        task = self.tasks[task_id]
        status = self.vehicle_statuses[vehicle_id]
        
        # 根据任务类型执行不同操作
        if task.task_type == 'to_loading':
            # 到达装载点，开始装载
            status['status'] = 'loading'
            self.env.vehicles[vehicle_id]['status'] = 'loading'
            self.env.vehicles[vehicle_id]['loading_progress'] = 0
            
            # 新增: 记录使用的装载点
            if task.loading_point_id is not None:
                status['preferred_loading_point'] = task.loading_point_id
        
        elif task.task_type == 'to_unloading':
            # 到达卸载点，开始卸载
            status['status'] = 'unloading'
            self.env.vehicles[vehicle_id]['status'] = 'unloading'
            self.env.vehicles[vehicle_id]['unloading_progress'] = 0
            
            # 新增: 记录使用的卸载点
            if task.unloading_point_id is not None:
                status['preferred_unloading_point'] = task.unloading_point_id
        
        elif task.task_type == 'to_initial':
            # 返回起点，完成循环
            self._complete_task(vehicle_id, task_id)
            
            # 更新完成循环计数
            self.env.vehicles[vehicle_id]['completed_cycles'] = self.env.vehicles[vehicle_id].get('completed_cycles', 0) + 1
            
            # 检查是否有更多任务
            if self.task_queues[vehicle_id]:
                # 移除已完成的任务
                self.task_queues[vehicle_id].pop(0)
                
                # 如果还有任务模板，添加新循环
                if self.mission_templates:
                    template_id = list(self.mission_templates.keys())[0]
                    self.assign_mission(vehicle_id, template_id)
                
                # 开始下一个任务
                self._start_next_task(vehicle_id)
            else:
                # 没有更多任务，车辆空闲
                status['status'] = 'idle'
                status['current_task'] = None
                self.env.vehicles[vehicle_id]['status'] = 'idle'
    
    def _complete_task(self, vehicle_id, task_id):
        """完成任务并更新统计信息"""
        if task_id not in self.tasks:
            return
        
        task = self.tasks[task_id]
        
        # 更新任务状态
        task.status = 'completed'
        task.progress = 1.0
        task.completion_time = self.env.current_time if hasattr(self.env, 'current_time') else 0
        
        # 更新统计信息
        self.stats['completed_tasks'] += 1
        
        # 计算路径长度
        if task.path:
            path_length = 0
            for i in range(len(task.path) - 1):
                point1 = task.path[i]
                point2 = task.path[i + 1]
                dist = ((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2) ** 0.5
                path_length += dist
            
            # 更新总距离
            self.stats['total_distance'] += path_length
            
            # 更新车辆总距离
            self.vehicle_statuses[vehicle_id]['total_distance'] += path_length
        
        # 更新车辆完成任务计数
        self.vehicle_statuses[vehicle_id]['completed_tasks'] += 1
        
        # 如果是任务队列的第一个任务，从队列中移除
        if self.task_queues[vehicle_id] and self.task_queues[vehicle_id][0] == task_id:
            self.task_queues[vehicle_id].pop(0)
        
        # 新增: 清理骨干网络上的流量
        if self.backbone_network and task.path_structure and task.path_structure['backbone_segment']:
            # 确保释放骨干网络上的流量
            path_id = task.path_structure['backbone_segment']
            self.backbone_network.update_traffic_flow(path_id, -1)  # 减少流量
    
    def get_vehicle_info(self, vehicle_id):
        """获取车辆详细信息"""
        if vehicle_id not in self.vehicle_statuses:
            return None
        
        info = self.vehicle_statuses[vehicle_id].copy()
        
        # 添加当前任务信息
        if info['current_task'] and info['current_task'] in self.tasks:
            task = self.tasks[info['current_task']]
            task_info = task.to_dict()
            
            # 添加路径结构信息
            if task.path_structure:
                task_info['path_structure'] = {}
                for key, value in task.path_structure.items():
                    if key in ['entry_point', 'exit_point', 'backbone_segment']:
                        task_info['path_structure'][key] = value
                    elif key in ['to_backbone_path', 'backbone_path', 'from_backbone_path'] and value:
                        task_info['path_structure'][key] = len(value)
            
            info['current_task_info'] = task_info
        
        # 添加任务队列信息
        if vehicle_id in self.task_queues:
            info['task_queue'] = []
            for task_id in self.task_queues[vehicle_id]:
                if task_id in self.tasks:
                    info['task_queue'].append(self.tasks[task_id].to_dict())
        
        return info
    
    def get_stats(self):
        """获取调度统计信息"""
        # 更新最新统计数据
        for vehicle_id, status in self.vehicle_statuses.items():
            self.stats['vehicle_utilization'][vehicle_id] = status['utilization_rate']
        
        # 计算全局利用率
        if self.vehicle_statuses:
            total_utilization = sum(self.stats['vehicle_utilization'].values())
            self.stats['average_utilization'] = total_utilization / len(self.vehicle_statuses)
        
        # 收集骨干路径使用情况
        if self.backbone_network and hasattr(self.backbone_network, 'paths'):
            backbone_usage = {}
            for path_id, path_data in self.backbone_network.paths.items():
                backbone_usage[path_id] = {
                    'traffic_flow': path_data.get('traffic_flow', 0),
                    'capacity': path_data.get('capacity', 1)
                }
            self.stats['backbone_usage'] = backbone_usage
        
        return self.stats
    
    def _calculate_distance(self, pos1, pos2):
        """计算两点之间的欧几里得距离"""
        # 确保坐标至少有x,y两个值
        x1 = pos1[0] if len(pos1) > 0 else 0
        y1 = pos1[1] if len(pos1) > 1 else 0
        x2 = pos2[0] if len(pos2) > 0 else 0
        y2 = pos2[1] if len(pos2) > 1 else 0
        
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    
    def is_point_accessible(self, position, point):
        """检查从当前位置是否可以到达指定点"""
        if not self.path_planner:
            # 简单的直线检查
            return True
        
        # 使用路径规划器检查可达性
        path = self.path_planner.plan_path(0, position, point, use_backbone=False)
        return path is not None and len(path) > 1

import math
import heapq
from typing import List, Dict, Tuple, Set, Optional, Any
from vehicle_scheduler import VehicleTask, VehicleScheduler

class ECBSVehicleScheduler(VehicleScheduler):
    """ECBS增强版车辆调度器，结合了ECBS原理进行任务协调和冲突解决 - 增强版"""
    
    def __init__(self, env, path_planner=None, traffic_manager=None, backbone_network=None):
        """
        初始化ECBS车辆调度器
        
        Args:
            env: 环境对象
            path_planner: 路径规划器
            traffic_manager: 交通管理器
            backbone_network: 骨干路径网络
        """
        super().__init__(env, path_planner, backbone_network)
        self.traffic_manager = traffic_manager
        
        # ECBS特有属性
        self.vehicle_priorities = {}  # 车辆优先级 {vehicle_id: priority}
        self.task_priorities = {}     # 任务优先级 {task_id: priority}
        self.conflict_counts = {}     # 车辆冲突计数 {vehicle_id: count}
        
        # 冲突解决策略
        self.conflict_resolution_strategy = 'ecbs'  # 'ecbs', 'priority', 'time_window'
        
        # 并行执行控制
        self.parallel_execution = True
        self.max_parallel_vehicles = 5  # 最大并行车辆数
        
        # 批量任务规划参数
        self.batch_planning = True
        self.max_batch_size = 10
        self.planning_horizon = 100.0  # 规划时间范围
        
        # 骨干路径使用情况
        self.backbone_path_usage = {}  # {path_id: {start_time: [vehicle_ids]}}
    
    def set_traffic_manager(self, traffic_manager):
        """设置交通管理器"""
        self.traffic_manager = traffic_manager
    
    def initialize_vehicles(self):
        """初始化车辆状态，并设置初始优先级"""
        super().initialize_vehicles()
        
        # 根据车辆ID设置初始优先级（简单策略：ID越小优先级越高）
        for i, vehicle_id in enumerate(self.vehicle_statuses.keys()):
            self.vehicle_priorities[vehicle_id] = len(self.vehicle_statuses) - i
            self.conflict_counts[vehicle_id] = 0
    
    def assign_tasks_batch(self, tasks_batch):
        """
        使用ECBS原理批量分配任务
        
        Args:
            tasks_batch: 任务列表
            
        Returns:
            dict: 分配结果 {vehicle_id: task_id}
        """
        if not tasks_batch:
            return {}
            
        # 按优先级排序任务
        sorted_tasks = sorted(
            tasks_batch, 
            key=lambda t: self.task_priorities.get(t.task_id, 1),
            reverse=True
        )
        
        # 初始化分配结果
        assignments = {}
        
        # 为每个任务找到最佳车辆
        for task in sorted_tasks:
            # 寻找最合适的车辆
            best_vehicle = self._find_best_vehicle_for_task(task)
            
            if best_vehicle:
                assignments[best_vehicle] = task.task_id
                self.assign_task(best_vehicle, task.task_id)
        
        # 如果设置了批量规划，为所有分配的车辆规划协调路径
        if self.batch_planning and assignments:
            vehicle_ids = list(assignments.keys())
            self._plan_coordinated_paths(vehicle_ids)
        
        return assignments
    
    def assign_task(self, vehicle_id, task_id):
        """
        Assign a single task to a vehicle
        
        Args:
            vehicle_id: ID of the vehicle to assign the task to
            task_id: ID of the task to assign
            
        Returns:
            bool: True if successfully assigned, False otherwise
        """
        if vehicle_id not in self.vehicle_statuses:
            return False
            
        if task_id not in self.tasks:
            return False
        
        # Mark the task as assigned
        task = self.tasks[task_id]
        task.status = 'assigned'
        task.assigned_vehicle = vehicle_id
        
        # Add task to vehicle's queue
        if vehicle_id not in self.task_queues:
            self.task_queues[vehicle_id] = []
        
        self.task_queues[vehicle_id].append(task_id)
        
        # If vehicle is idle, start the task immediately
        if self.vehicle_statuses[vehicle_id]['status'] == 'idle':
            self._start_next_task(vehicle_id)
        
        return True
    
    def _find_best_vehicle_for_task(self, task):
        """
        为任务找到最佳车辆，考虑多种因素
        
        Args:
            task: 任务对象
            
        Returns:
            int or None: 最佳车辆ID，如果没找到则返回None
        """
        best_vehicle = None
        best_score = float('-inf')
        
        for vehicle_id, status in self.vehicle_statuses.items():
            # 如果车辆有太多任务，跳过
            if vehicle_id in self.task_queues and len(self.task_queues[vehicle_id]) > 3:
                continue
                
            # 计算距离因素
            distance = self._calculate_distance(status['position'], task.start)
            distance_score = 1000 / (distance + 10)  # 避免除零
            
            # 计算负载因素
            load_factor = 1.0
            if task.task_type == 'to_unloading' and status['load'] < status['max_load'] * 0.5:
                load_factor = 0.5  # 降低空车去卸载点的分数
            elif task.task_type == 'to_loading' and status['load'] > status['max_load'] * 0.5:
                load_factor = 0.5  # 降低满载车去装载点的分数
            
            # 计算优先级因素
            priority_factor = self.vehicle_priorities.get(vehicle_id, 1) / 10.0
            
            # 计算冲突历史因素 - 冲突越多，分数越低
            conflict_factor = 1.0 / (1.0 + 0.1 * self.conflict_counts.get(vehicle_id, 0))
            
            # 检查任务点位是否与车辆首选点位匹配
            point_preference_factor = 1.0
            if (task.task_type == 'to_loading' and 
                status['preferred_loading_point'] is not None and
                task.loading_point_id == status['preferred_loading_point']):
                point_preference_factor = 1.5  # 提高匹配首选装载点的分数
            
            if (task.task_type == 'to_unloading' and 
                status['preferred_unloading_point'] is not None and
                task.unloading_point_id == status['preferred_unloading_point']):
                point_preference_factor = 1.5  # 提高匹配首选卸载点的分数
            
            # 计算总分
            score = distance_score * load_factor * priority_factor * conflict_factor * point_preference_factor
            
            # 如果是当前空闲的车辆，额外加分
            if status['status'] == 'idle':
                score *= 1.5
            
            # 更新最佳车辆
            if score > best_score:
                best_score = score
                best_vehicle = vehicle_id
        
        return best_vehicle
    
    def _plan_coordinated_paths(self, vehicle_ids):
        """
        为多个车辆规划协调的无冲突路径，保留骨干网络结构
        
        Args:
            vehicle_ids: 车辆ID列表
            
        Returns:
            bool: 规划是否成功
        """
        if not self.traffic_manager or not vehicle_ids:
            return False
            
        # 收集每个车辆的当前任务和路径需求
        paths = {}
        vehicle_tasks = {}
        path_structures = {}  # 保存路径结构信息
        
        for vehicle_id in vehicle_ids:
            # 获取当前任务
            task_id = self.vehicle_statuses[vehicle_id]['current_task']
            if not task_id or task_id not in self.tasks:
                continue
                
            task = self.tasks[task_id]
            vehicle_tasks[vehicle_id] = task
            
            # 规划初始路径 - 使用骨干网络
            path, structure = self._plan_structured_path(vehicle_id, task.start, task.goal)
            
            if path:
                paths[vehicle_id] = path
                path_structures[vehicle_id] = structure
        
        # 如果我们有多个车辆的路径，使用ECBS解决冲突
        if len(paths) > 1:
            # 使用交通管理器解决冲突，传递骨干网络和路径结构
            conflict_free_paths = self.traffic_manager.resolve_conflicts(
                paths, 
                backbone_network=self.backbone_network,
                path_structures=path_structures
            )
            
            if conflict_free_paths:
                # 更新任务和车辆的路径
                for vehicle_id, path in conflict_free_paths.items():
                    if vehicle_id in vehicle_tasks:
                        task = vehicle_tasks[vehicle_id]
                        task.path = path
                        
                        # 分析并保存新路径的结构
                        task.path_structure = self._analyze_path_structure(path)
                        
                        # 向交通管理器注册路径
                        if self.traffic_manager:
                            self.traffic_manager.register_vehicle_path(
                                vehicle_id,
                                path,
                                self.env.current_time if hasattr(self.env, 'current_time') else 0
                            )
                        
                        # 更新骨干路径使用情况
                        self._update_backbone_usage(vehicle_id, task)
                        
                        # 更新车辆路径
                        if vehicle_id in self.env.vehicles:
                            self.env.vehicles[vehicle_id]['path'] = path
                            self.env.vehicles[vehicle_id]['path_index'] = 0
                            self.env.vehicles[vehicle_id]['progress'] = 0.0
                
                return True
        elif len(paths) == 1:
            # 只有一个车辆，直接使用路径
            vehicle_id = list(paths.keys())[0]
            task = vehicle_tasks[vehicle_id]
            task.path = paths[vehicle_id]
            
            # 分析并保存路径结构
            task.path_structure = path_structures[vehicle_id]
            
            # 向交通管理器注册路径
            if self.traffic_manager:
                self.traffic_manager.register_vehicle_path(
                    vehicle_id,
                    task.path,
                    self.env.current_time if hasattr(self.env, 'current_time') else 0
                )
            
            # 更新骨干路径使用情况
            self._update_backbone_usage(vehicle_id, task)
            
            # 更新车辆路径
            if vehicle_id in self.env.vehicles:
                self.env.vehicles[vehicle_id]['path'] = task.path
                self.env.vehicles[vehicle_id]['path_index'] = 0
                self.env.vehicles[vehicle_id]['progress'] = 0.0
            
            return True
        
        return False
    
    def _plan_structured_path(self, vehicle_id, start, goal):
        """
        规划具有结构的路径，明确识别骨干网络部分
        
        Args:
            vehicle_id: 车辆ID
            start: 起点
            goal: 终点
            
        Returns:
            tuple: (路径点列表, 路径结构)
        """
        if not self.path_planner or not self.backbone_network:
            # 没有骨干网络，使用普通规划
            path = self.path_planner.plan_path(vehicle_id, start, goal) if self.path_planner else None
            return path, {'to_backbone_path': path}
        
        # 第1步: 寻找起点和终点附近的骨干网络接入点
        start_conn = self.backbone_network.find_nearest_connection(start, max_distance=20.0)
        end_conn = self.backbone_network.find_nearest_connection(goal, max_distance=20.0)
        
        if not start_conn or not end_conn:
            # 没有找到合适的接入点，使用普通规划
            path = self.path_planner.plan_path(vehicle_id, start, goal)
            return path, {'to_backbone_path': path}
        
        # 第2步: 规划从起点到骨干网络接入点的路径
        to_backbone_path = self.path_planner.plan_path(
            vehicle_id, 
            start, 
            start_conn['position'],
            use_backbone=False  # 不使用骨干网络
        )
        
        if not to_backbone_path:
            # 无法到达接入点，使用普通规划
            path = self.path_planner.plan_path(vehicle_id, start, goal)
            return path, {'to_backbone_path': path}
        
        # 第3步: 在骨干网络中规划路径
        # 查找连接start_conn和end_conn的骨干路径
        backbone_path_id = None
        backbone_path = None
        
        if start_conn.get('path_id') == end_conn.get('path_id'):
            # 两个连接点在同一条骨干路径上
            backbone_path_id = start_conn['path_id']
            start_index = start_conn.get('path_index', 0)
            end_index = end_conn.get('path_index', 0)
            
            # 获取骨干路径段
            if backbone_path_id in self.backbone_network.paths:
                path_data = self.backbone_network.paths[backbone_path_id]
                
                if start_index <= end_index:
                    backbone_path = path_data['path'][start_index:end_index+1]
                else:
                    # 反向走
                    backbone_path = list(reversed(path_data['path'][end_index:start_index+1]))
        else:
            # 两个连接点在不同骨干路径上，需要在骨干网络中查找路径
            backbone_segments = self.backbone_network.find_path(
                start_conn.get('path_id'), 
                end_conn.get('path_id')
            )
            
            if backbone_segments:
                # 拼接所有骨干路径段
                backbone_path = []
                
                for segment_id in backbone_segments:
                    if segment_id in self.backbone_network.paths:
                        segment_path = self.backbone_network.paths[segment_id]['path']
                        
                        # 跳过第一个点以避免重复
                        if backbone_path:
                            segment_path = segment_path[1:]
                            
                        backbone_path.extend(segment_path)
                        
                backbone_path_id = "+".join(backbone_segments)  # 多段路径ID
        
        if not backbone_path:
            # 骨干网络中无法找到路径，使用直接路径
            # 尝试保留已规划的 to_backbone_path
            direct_path = self.path_planner.plan_path(vehicle_id, start_conn['position'], goal)
            
            if direct_path:
                # 合并路径
                combined_path = to_backbone_path[:-1] + direct_path
                return combined_path, {
                    'to_backbone_path': to_backbone_path,
                    'from_backbone_path': direct_path
                }
            else:
                # 完全回退到普通规划
                path = self.path_planner.plan_path(vehicle_id, start, goal)
                return path, {'to_backbone_path': path}
        
        # 第4步: 规划从骨干网络出口点到终点的路径
        from_backbone_path = self.path_planner.plan_path(
            vehicle_id, 
            end_conn['position'], 
            goal,
            use_backbone=False  # 不使用骨干网络
        )
        
        if not from_backbone_path:
            # 无法从出口点到达终点，使用普通规划
            path = self.path_planner.plan_path(vehicle_id, start, goal)
            return path, {'to_backbone_path': path}
        
        # 第5步: 合并三段路径
        # 注意：避免重复连接点
        if to_backbone_path[-1] == backbone_path[0]:
            complete_path = to_backbone_path[:-1] + backbone_path
        else:
            complete_path = to_backbone_path + backbone_path
            
        if backbone_path[-1] == from_backbone_path[0]:
            complete_path = complete_path[:-1] + from_backbone_path
        else:
            complete_path = complete_path + from_backbone_path
        
        # 构建路径结构
        path_structure = {
            'entry_point': start_conn,
            'exit_point': end_conn,
            'backbone_segment': backbone_path_id,
            'to_backbone_path': to_backbone_path,
            'backbone_path': backbone_path,
            'from_backbone_path': from_backbone_path
        }
        
        return complete_path, path_structure
    
    def _update_backbone_usage(self, vehicle_id, task):
        """更新骨干路径使用情况"""
        if not task.path_structure or not task.path_structure.get('backbone_segment'):
            return
        
        backbone_segment = task.path_structure['backbone_segment']
        current_time = self.env.current_time if hasattr(self.env, 'current_time') else 0
        
        # 添加到使用记录
        if backbone_segment not in self.backbone_path_usage:
            self.backbone_path_usage[backbone_segment] = {}
        
        if current_time not in self.backbone_path_usage[backbone_segment]:
            self.backbone_path_usage[backbone_segment][current_time] = []
        
        if vehicle_id not in self.backbone_path_usage[backbone_segment][current_time]:
            self.backbone_path_usage[backbone_segment][current_time].append(vehicle_id)
        
        # 更新骨干网络流量
        if self.backbone_network and backbone_segment in self.backbone_network.paths:
            self.backbone_network.update_traffic_flow(backbone_segment, 1)  # 增加流量
    
    def _plan_initial_path(self, vehicle_id, start, goal):
        """
        规划初始路径 - 增强版本，确保使用骨干网络
        
        Args:
            vehicle_id: 车辆ID
            start: 起点
            goal: 终点
            
        Returns:
            list or None: 路径点列表
        """
        if self.backbone_network and self.path_planner:
            # 使用_plan_structured_path替代简单的plan_path
            path, _ = self._plan_structured_path(vehicle_id, start, goal)
            return path
        elif self.path_planner:
            return self.path_planner.plan_path(
                vehicle_id,
                start,
                goal,
                use_backbone=True,  # 优先使用主干网络
                check_conflicts=False  # 冲突检查将在ECBS中统一处理
            )
        return None
    
    def update(self, time_delta):
        """
        更新所有车辆状态和任务进度，包括冲突检测和解决
        
        Args:
            time_delta: 时间步长
            
        Returns:
            bool: 更新是否成功
        """
        # 首先更新车辆和任务状态
        for vehicle_id in list(self.vehicle_statuses.keys()):
            self._update_vehicle(vehicle_id, time_delta)
        
        # 检查当前执行中的路径是否有冲突
        if self.traffic_manager:
            self._check_and_resolve_execution_conflicts()
        
        # 选择下一批要执行的任务
        self._select_next_batch_tasks()
        
        return True
    
    def _check_and_resolve_execution_conflicts(self):
        """检查并解决执行中的路径冲突，保留骨干网络结构"""
        # 收集当前活动的车辆和路径
        active_vehicles = []
        active_paths = {}
        path_structures = {}  # 保存路径结构信息
        
        for vehicle_id, status in self.vehicle_statuses.items():
            if status['status'] == 'moving' and status['current_task']:
                active_vehicles.append(vehicle_id)
                
                # 获取当前路径
                task_id = status['current_task']
                if task_id in self.tasks and self.tasks[task_id].path:
                    # 获取剩余路径 - 从当前位置到终点
                    if vehicle_id in self.env.vehicles:
                        vehicle = self.env.vehicles[vehicle_id]
                        path_index = vehicle.get('path_index', 0)
                        path = self.tasks[task_id].path[path_index:]
                        
                        active_paths[vehicle_id] = path
                        
                        # 保存路径结构信息
                        task = self.tasks[task_id]
                        if task.path_structure:
                            path_structures[vehicle_id] = task.path_structure
        
        # 如果有多个活动车辆，检查冲突
        if len(active_vehicles) > 1:
            conflicts = self.traffic_manager.detect_conflicts(active_paths)
            
            # 如果发现冲突，记录并解决
            if conflicts:
                # 记录冲突车辆
                for conflict in conflicts:
                    self.conflict_counts[conflict.agent1] = self.conflict_counts.get(conflict.agent1, 0) + 1
                    self.conflict_counts[conflict.agent2] = self.conflict_counts.get(conflict.agent2, 0) + 1
                
                # 根据冲突解决策略处理
                if self.conflict_resolution_strategy == 'ecbs':
                    # 使用ECBS解决冲突，传递路径结构信息
                    new_paths = self.traffic_manager.resolve_conflicts(
                        active_paths,
                        backbone_network=self.backbone_network,
                        path_structures=path_structures
                    )
                    
                    # 更新路径
                    if new_paths:
                        for vehicle_id, path in new_paths.items():
                            task_id = self.vehicle_statuses[vehicle_id]['current_task']
                            if task_id in self.tasks:
                                # 保存原路径的执行状态
                                orig_path_index = 0
                                if vehicle_id in self.env.vehicles:
                                    orig_path_index = self.env.vehicles[vehicle_id].get('path_index', 0)
                                
                                # 更新任务路径
                                self.tasks[task_id].path = path
                                
                                # 分析并保存新路径的结构
                                self.tasks[task_id].path_structure = self._analyze_path_structure(path)
                                
                                # 更新车辆路径
                                self.env.vehicles[vehicle_id]['path'] = path
                                self.env.vehicles[vehicle_id]['path_index'] = 0  # 从头开始
                                self.env.vehicles[vehicle_id]['progress'] = 0.0
                                
                                # 重新向交通管理器注册路径
                                self.traffic_manager.release_vehicle_path(vehicle_id)
                                self.traffic_manager.register_vehicle_path(
                                    vehicle_id,
                                    path,
                                    self.env.current_time if hasattr(self.env, 'current_time') else 0
                                )
                
                elif self.conflict_resolution_strategy == 'priority':
                    # 基于优先级解决冲突 - 高优先级车辆保持路径，低优先级车辆重新规划
                    for conflict in conflicts:
                        # 比较优先级
                        prio1 = self.vehicle_priorities.get(conflict.agent1, 1)
                        prio2 = self.vehicle_priorities.get(conflict.agent2, 1)
                        
                        # 低优先级车辆重新规划
                        if prio1 > prio2:
                            lower_prio_agent = conflict.agent2
                        else:
                            lower_prio_agent = conflict.agent1
                        
                        # 为低优先级车辆重新规划路径，保留骨干网络结构
                        self._replan_structured_path(lower_prio_agent)
                        
                elif self.conflict_resolution_strategy == 'time_window':
                    # 时间窗口策略 - 调整车辆速度以避免冲突
                    for conflict in conflicts:
                        # 为两个车辆都尝试调整速度
                        speed_factor1 = self.traffic_manager.suggest_speed_adjustment(conflict.agent1)
                        speed_factor2 = self.traffic_manager.suggest_speed_adjustment(conflict.agent2)
                        
                        # 应用速度调整
                        if speed_factor1 and conflict.agent1 in self.env.vehicles:
                            self.env.vehicles[conflict.agent1]['speed'] = self.env.vehicles[conflict.agent1].get('speed', 1.0) * speed_factor1
                        
                        if speed_factor2 and conflict.agent2 in self.env.vehicles:
                            self.env.vehicles[conflict.agent2]['speed'] = self.env.vehicles[conflict.agent2].get('speed', 1.0) * speed_factor2
    
    def _replan_structured_path(self, vehicle_id):
        """
        为车辆重新规划路径，保留骨干网络结构
        
        Args:
            vehicle_id: 车辆ID
            
        Returns:
            bool: 规划是否成功
        """
        if vehicle_id not in self.vehicle_statuses:
            return False
            
        status = self.vehicle_statuses[vehicle_id]
        task_id = status['current_task']
        
        if not task_id or task_id not in self.tasks:
            return False
            
        task = self.tasks[task_id]
        
        # 获取当前位置和目标
        current_pos = status['position']
        goal = task.goal
        
        # 规划新路径，保留骨干网络结构
        new_path, structure = self._plan_structured_path(vehicle_id, current_pos, goal)
        
        if new_path:
            # 更新任务路径
            task.path = new_path
            task.path_structure = structure
            
            # 更新车辆路径
            if vehicle_id in self.env.vehicles:
                self.env.vehicles[vehicle_id]['path'] = new_path
                self.env.vehicles[vehicle_id]['path_index'] = 0
                self.env.vehicles[vehicle_id]['progress'] = 0.0
            
            # 更新交通管理器
            if self.traffic_manager:
                self.traffic_manager.release_vehicle_path(vehicle_id)
                self.traffic_manager.register_vehicle_path(
                    vehicle_id,
                    new_path,
                    self.env.current_time if hasattr(self.env, 'current_time') else 0
                )
            
            return True
        
        # 如果无法规划结构化路径，回退到交通管理器的建议路径
        if self.traffic_manager:
            fallback_path = self.traffic_manager.suggest_path_adjustment(vehicle_id, current_pos, goal)
            
            if fallback_path:
                # 更新任务路径
                task.path = fallback_path
                task.path_structure = self._analyze_path_structure(fallback_path)
                
                # 更新车辆路径
                if vehicle_id in self.env.vehicles:
                    self.env.vehicles[vehicle_id]['path'] = fallback_path
                    self.env.vehicles[vehicle_id]['path_index'] = 0
                    self.env.vehicles[vehicle_id]['progress'] = 0.0
                
                # 更新交通管理器
                self.traffic_manager.release_vehicle_path(vehicle_id)
                self.traffic_manager.register_vehicle_path(
                    vehicle_id,
                    fallback_path,
                    self.env.current_time if hasattr(self.env, 'current_time') else 0
                )
                
                return True
        
        return False
    
    def _select_next_batch_tasks(self):
        """选择下一批要执行的任务"""
        # 只选择空闲车辆和已经有任务队列的车辆
        available_vehicles = []
        
        for vehicle_id, status in self.vehicle_statuses.items():
            if status['status'] == 'idle':
                # 检查是否有任务队列
                if vehicle_id in self.task_queues and self.task_queues[vehicle_id]:
                    available_vehicles.append(vehicle_id)
        
        # 如果没有可用车辆，直接返回
        if not available_vehicles:
            return
        
        # 按优先级排序车辆
        sorted_vehicles = sorted(
            available_vehicles,
            key=lambda v: self.vehicle_priorities.get(v, 1),
            reverse=True
        )
        
        # 限制同时执行的车辆数量
        batch_vehicles = sorted_vehicles[:self.max_parallel_vehicles]
        
        # 为每个选定的车辆启动任务
        if len(batch_vehicles) > 1 and self.batch_planning:
            # 批量规划路径
            self._plan_coordinated_paths(batch_vehicles)
        else:
            # 单独启动任务
            for vehicle_id in batch_vehicles:
                self._start_next_task(vehicle_id)
    
    def adjust_vehicle_priority(self, vehicle_id, adjustment):
        """根据执行情况调整车辆优先级"""
        if vehicle_id in self.vehicle_priorities:
            self.vehicle_priorities[vehicle_id] += adjustment
            # 确保优先级不为负
            self.vehicle_priorities[vehicle_id] = max(1, self.vehicle_priorities[vehicle_id])
    
    def create_ecbs_mission_template(self, template_id, loading_point=None, unloading_point=None):
        """创建带有ECBS特性的任务模板"""
        # 首先创建基本任务模板
        if not super().create_mission_template(template_id, loading_point, unloading_point):
            return False
        
        # 增强任务优先级设置
        template = self.mission_templates[template_id]
        
        # 调整任务优先级 - 采用更细致的优先级策略
        for i, task in enumerate(template):
            if task['task_type'] == 'to_loading':
                task['priority'] = 2  # 前往装载点优先级中等
            elif task['task_type'] == 'to_unloading':
                task['priority'] = 3  # 前往卸载点优先级高(满载)
            elif task['task_type'] == 'to_initial':
                task['priority'] = 1  # 返回起点优先级低
        
        return True
    
    def assign_mission(self, vehicle_id, template_id):
        """以ECBS感知方式分配任务"""
        # 首先尝试常规分配
        if not super().assign_mission(vehicle_id, template_id):
            return False
        
        # 如果成功分配，更新任务优先级
        current_prio = self.vehicle_priorities.get(vehicle_id, 1)
        
        # 获取车辆的任务队列
        if vehicle_id in self.task_queues:
            for task_id in self.task_queues[vehicle_id]:
                if task_id in self.tasks:
                    # 将车辆优先级融入任务优先级
                    self.tasks[task_id].priority *= current_prio / 10.0
                    # 保存到任务优先级字典
                    self.task_priorities[task_id] = self.tasks[task_id].priority
        
        # 如果当前车辆空闲，尝试规划初始路径
        if self.vehicle_statuses[vehicle_id]['status'] == 'idle':
            # 已经通过super().assign_mission中的_start_next_task启动了任务
            # 现在需要确保路径规划适当考虑了ECBS
            task_id = self.vehicle_statuses[vehicle_id]['current_task']
            
            if task_id in self.tasks and self.tasks[task_id].path:
                # 向交通管理器注册路径
                if self.traffic_manager:
                    self.traffic_manager.register_vehicle_path(
                        vehicle_id,
                        self.tasks[task_id].path,
                        self.env.current_time if hasattr(self.env, 'current_time') else 0
                    )
        
        return True
    
    def _complete_task(self, vehicle_id, task_id):
        """完成任务并更新统计信息，添加ECBS相关处理"""
        # 首先调用原始方法
        super()._complete_task(vehicle_id, task_id)
        
        # 释放交通管理器中的路径
        if self.traffic_manager:
            self.traffic_manager.release_vehicle_path(vehicle_id)
        
        # 根据任务完成情况调整车辆优先级
        if vehicle_id in self.conflict_counts:
            conflicts = self.conflict_counts[vehicle_id]
            
            # 如果没有冲突，小幅提高优先级
            if conflicts == 0:
                self.adjust_vehicle_priority(vehicle_id, 0.5)
            # 如果冲突很多，降低优先级
            elif conflicts > 5:
                self.adjust_vehicle_priority(vehicle_id, -1.0)
            
            # 重置冲突计数
            self.conflict_counts[vehicle_id] = 0
    
    def _handle_task_completion(self, vehicle_id, task_id):
        """处理任务完成"""
        # 在更新状态前释放路径
        if self.traffic_manager:
            self.traffic_manager.release_vehicle_path(vehicle_id)
        
        # 释放骨干网络资源
        task = self.tasks.get(task_id)
        if task and task.path_structure and task.path_structure.get('backbone_segment'):
            backbone_segment = task.path_structure['backbone_segment']
            if self.backbone_network and backbone_segment in self.backbone_network.paths:
                self.backbone_network.update_traffic_flow(backbone_segment, -1)  # 减少流量
        
        # 调用原始处理方法
        super()._handle_task_completion(vehicle_id, task_id)
    
    def get_vehicle_info(self, vehicle_id):
        """获取车辆详细信息，包括ECBS特有信息"""
        # 获取基本信息
        info = super().get_vehicle_info(vehicle_id)
        
        if not info:
            return None
        
        # 添加ECBS特有信息
        info['priority'] = self.vehicle_priorities.get(vehicle_id, 1)
        info['conflict_count'] = self.conflict_counts.get(vehicle_id, 0)
        
        # 获取潜在的冲突信息
        if self.traffic_manager and vehicle_id in self.vehicle_statuses:
            status = self.vehicle_statuses[vehicle_id]
            if status['current_task'] and status['current_task'] in self.tasks:
                task = self.tasks[status['current_task']]
                if task.path:
                    # 简化的冲突检测
                    conflicts = []
                    for other_id in self.vehicle_statuses:
                        if other_id != vehicle_id:
                            other_status = self.vehicle_statuses[other_id]
                            if other_status['current_task'] and other_status['current_task'] in self.tasks:
                                other_task = self.tasks[other_status['current_task']]
                                if other_task.path:
                                    # 检查两个路径是否有潜在冲突
                                    conflict = self.traffic_manager.conflict_detector.check_path_conflict(
                                        vehicle_id, task.path,
                                        other_id, other_task.path
                                    )
                                    if conflict:
                                        conflicts.append({
                                            'with_vehicle': other_id,
                                            'location': conflict.location,
                                            'time': conflict.time_step
                                        })
                    
                    info['potential_conflicts'] = conflicts
        
        return info
    
    def get_stats(self):
        """获取调度统计信息，包括ECBS特有信息"""
        # 获取基本统计信息
        stats = super().get_stats()
        
        # 添加ECBS特有信息
        stats['conflict_resolution'] = {
            'strategy': self.conflict_resolution_strategy,
            'total_conflicts': sum(self.conflict_counts.values()),
            'conflicts_by_vehicle': self.conflict_counts.copy()
        }
        
        # 添加骨干网络使用情况
        stats['backbone_usage_details'] = {}
        for path_id, time_data in self.backbone_path_usage.items():
            stats['backbone_usage_details'][path_id] = {
                'vehicle_count': sum(len(vehicles) for vehicles in time_data.values()),
                'time_slots': len(time_data)
            }
        
        return stats