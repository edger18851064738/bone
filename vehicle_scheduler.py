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


class VehicleScheduler:
    """车辆调度器，管理任务分配和调度"""
    
    def __init__(self, env, path_planner=None):
        self.env = env
        self.path_planner = path_planner
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
    
    def set_path_planner(self, path_planner):
        """设置路径规划器"""
        self.path_planner = path_planner
    
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
                'idle_time': 0
            }
            
            # 初始化任务队列
            self.task_queues[vehicle_id] = []
    
    def create_mission_template(self, template_id, loading_point=None, unloading_point=None):
        """创建任务模板（装载-卸载-返回循环）"""
        if loading_point is None and self.env.loading_points:
            loading_point = self.env.loading_points[0]
        
        if unloading_point is None and self.env.unloading_points:
            unloading_point = self.env.unloading_points[0]
        
        if not loading_point or not unloading_point:
            return False
        
        # 创建三段任务模板
        template = [
            {
                'task_type': 'to_loading',
                'goal': (loading_point[0], loading_point[1], 0),
                'priority': 1
            },
            {
                'task_type': 'to_unloading',
                'goal': (unloading_point[0], unloading_point[1], 0),
                'priority': 2
            },
            {
                'task_type': 'to_initial',
                'goal': None,  # 将在分配时设置为车辆初始位置
                'priority': 1
            }
        ]
        
        self.mission_templates[template_id] = template
        return True
    
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
            
            # 创建任务
            task = VehicleTask(
                task_id,
                task_template['task_type'],
                current_position,  # 起点设为前一任务的终点
                task_template['goal'] if task_template['goal'] else initial_position,
                task_template['priority']
            )
            
            # 更新下一任务的起点
            current_position = task.goal
            
            # 添加到任务字典
            self.tasks[task_id] = task
            
            # 添加到车辆任务队列
            self.task_queues[vehicle_id].append(task_id)
        
        # 如果车辆空闲，立即开始执行任务
        if self.vehicle_statuses[vehicle_id]['status'] == 'idle':
            self._start_next_task(vehicle_id)
        
        return True
    
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
        
        # 规划路径
        if self.path_planner:
            task.path = self.path_planner.plan_path(
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
        
        elif task.task_type == 'to_unloading':
            # 到达卸载点，开始卸载
            status['status'] = 'unloading'
            self.env.vehicles[vehicle_id]['status'] = 'unloading'
            self.env.vehicles[vehicle_id]['unloading_progress'] = 0
        
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
    
    def get_vehicle_info(self, vehicle_id):
        """获取车辆详细信息"""
        if vehicle_id not in self.vehicle_statuses:
            return None
        
        info = self.vehicle_statuses[vehicle_id].copy()
        
        # 添加当前任务信息
        if info['current_task'] and info['current_task'] in self.tasks:
            info['current_task_info'] = self.tasks[info['current_task']].to_dict()
        
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
        
        return self.stats