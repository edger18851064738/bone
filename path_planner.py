class PathPlanner:
    """路径规划器，连接车辆与主干路径网络"""
    
    def __init__(self, env, backbone_network=None, local_planner=None):
        self.env = env
        self.backbone_network = backbone_network
        self.local_planner = local_planner  # 本地路径规划器，用于连接主干网络
        self.route_cache = {}  # 路由缓存
    
    def set_backbone_network(self, backbone_network):
        """设置主干路径网络"""
        self.backbone_network = backbone_network
    
    def set_local_planner(self, local_planner):
        """设置本地路径规划器"""
        self.local_planner = local_planner
    
    def plan_path(self, vehicle_id, start, goal):
        """规划从起点到终点的完整路径"""
        if not self.backbone_network:
            # 无主干网络，使用本地规划器直接规划
            return self._plan_direct_path(start, goal)
        
        # 查找最近的主干网络连接点
        start_conn = self.backbone_network.find_nearest_connection(start)
        goal_conn = self.backbone_network.find_nearest_connection(goal)
        
        if not start_conn or not goal_conn:
            # 无法连接到主干网络，使用直接规划
            return self._plan_direct_path(start, goal)
        
        # 构建完整路径：起点 -> 主干入口 -> 主干网络 -> 主干出口 -> 终点
        
        # 1. 从起点到主干网络入口的路径
        path_to_backbone = self._plan_local_path(start, start_conn['position'])
        
        if not path_to_backbone:
            # 无法连接到主干网络，使用直接规划
            return self._plan_direct_path(start, goal)
        
        # 2. 在主干网络中的路径
        backbone_path = self._plan_backbone_path(start_conn, goal_conn)
        
        if not backbone_path:
            # 主干网络中无法找到路径，使用直接规划
            return self._plan_direct_path(start, goal)
        
        # 3. 从主干网络出口到终点的路径
        path_from_backbone = self._plan_local_path(goal_conn['position'], goal)
        
        if not path_from_backbone:
            # 无法从主干网络连接到终点，使用直接规划
            return self._plan_direct_path(start, goal)
        
        # 合并三段路径
        complete_path = path_to_backbone[:-1] + backbone_path + path_from_backbone
        
        # 更新主干网络流量
        self._update_backbone_traffic(start_conn, goal_conn, vehicle_id, 1)
        
        return complete_path
    
    def _plan_direct_path(self, start, goal):
        """使用本地规划器直接规划路径"""
        if self.local_planner:
            return self.local_planner.plan_path(start, goal)
        else:
            # 简单直线路径（实际应使用更复杂的避障算法）
            return [start, goal]
    
    def _plan_local_path(self, start, goal):
        """规划本地路径（非主干网络部分）"""
        return self._plan_direct_path(start, goal)
    
    def _plan_backbone_path(self, start_conn, goal_conn):
        """在主干网络中规划路径"""
        # 如果在同一条路径上
        if start_conn['path_id'] == goal_conn['path_id']:
            # 直接获取该路径段
            return self.backbone_network.get_path_segment(
                start_conn['path_id'],
                start_conn['path_index'],
                goal_conn['path_index']
            )
        
        # 不在同一条路径上，需要路由
        # 获取连接点所在节点的ID
        start_node_id = self.backbone_network.paths[start_conn['path_id']]['start']['id']
        if start_conn['path_index'] > len(self.backbone_network.paths[start_conn['path_id']]['path']) // 2:
            # 如果连接点更接近终点，使用终点作为起始节点
            start_node_id = self.backbone_network.paths[start_conn['path_id']]['end']['id']
        
        goal_node_id = self.backbone_network.paths[goal_conn['path_id']]['start']['id']
        if goal_conn['path_index'] > len(self.backbone_network.paths[goal_conn['path_id']]['path']) // 2:
            # 如果连接点更接近终点，使用终点作为目标节点
            goal_node_id = self.backbone_network.paths[goal_conn['path_id']]['end']['id']
        
        # 在路径图中查找最优路径
        path_ids = self.backbone_network.find_path(start_node_id, goal_node_id)
        
        if not path_ids:
            return None
        
        # 构建完整路径
        complete_path = []
        
        # 添加从起始连接点到第一条路径起点的段
        first_path_id = path_ids[0]
        first_start_id = self.backbone_network.paths[first_path_id]['start']['id']
        
        if start_node_id == first_start_id:
            # 连接点在路径起点附近
            segment = self.backbone_network.get_path_segment(
                start_conn['path_id'],
                start_conn['path_index'],
                0
            )
        else:
            # 连接点在路径终点附近
            segment = self.backbone_network.get_path_segment(
                start_conn['path_id'],
                start_conn['path_index'],
                len(self.backbone_network.paths[start_conn['path_id']]['path']) - 1
            )
        
        if segment:
            complete_path.extend(segment[:-1])  # 不包括末尾点，避免重复
        
        # 添加中间路径段
        for i in range(len(path_ids)):
            path_id = path_ids[i]
            path = self.backbone_network.paths[path_id]['path']
            
            if i == 0 and i == len(path_ids) - 1:
                # 只有一条路径，需要考虑起止连接点
                continue  # 在后面处理
            elif i == 0:
                # 第一条路径，从起点到终点
                complete_path.extend(path)
            elif i == len(path_ids) - 1:
                # 最后一条路径，从起点到连接点
                complete_path.extend(path[:-1])  # 不包括末尾点，避免重复
            else:
                # 中间路径，全部包括
                complete_path.extend(path[:-1])  # 不包括末尾点，避免重复
        
        # 添加从最后一条路径终点到目标连接点的段
        last_path_id = path_ids[-1]
        last_end_id = self.backbone_network.paths[last_path_id]['end']['id']
        
        if goal_node_id == last_end_id:
            # 连接点在路径终点附近
            segment = self.backbone_network.get_path_segment(
                goal_conn['path_id'],
                len(self.backbone_network.paths[goal_conn['path_id']]['path']) - 1,
                goal_conn['path_index']
            )
        else:
            # 连接点在路径起点附近
            segment = self.backbone_network.get_path_segment(
                goal_conn['path_id'],
                0,
                goal_conn['path_index']
            )
        
        if segment:
            complete_path.extend(segment)
        
        return complete_path
    
    def _update_backbone_traffic(self, start_conn, goal_conn, vehicle_id, delta=1):
        """更新主干网络的交通流量"""
        # 记录使用情况，用于后续释放
        if delta > 0:  # 增加流量
            if vehicle_id not in self.route_cache:
                self.route_cache[vehicle_id] = []
            
            # 清除之前的路由缓存
            if self.route_cache[vehicle_id]:
                # 释放之前的路径流量
                for path_id in self.route_cache[vehicle_id]:
                    self.backbone_network.update_traffic_flow(path_id, -1)
                
                self.route_cache[vehicle_id] = []
            
            # 如果是同一条路径
            if start_conn['path_id'] == goal_conn['path_id']:
                self.backbone_network.update_traffic_flow(start_conn['path_id'], delta)
                self.route_cache[vehicle_id].append(start_conn['path_id'])
            else:
                # 获取完整路径
                start_node_id = self.backbone_network.paths[start_conn['path_id']]['start']['id']
                if start_conn['path_index'] > len(self.backbone_network.paths[start_conn['path_id']]['path']) // 2:
                    start_node_id = self.backbone_network.paths[start_conn['path_id']]['end']['id']
                
                goal_node_id = self.backbone_network.paths[goal_conn['path_id']]['start']['id']
                if goal_conn['path_index'] > len(self.backbone_network.paths[goal_conn['path_id']]['path']) // 2:
                    goal_node_id = self.backbone_network.paths[goal_conn['path_id']]['end']['id']
                
                path_ids = self.backbone_network.find_path(start_node_id, goal_node_id)
                
                if path_ids:
                    # 更新所有路径的流量
                    for path_id in path_ids:
                        self.backbone_network.update_traffic_flow(path_id, delta)
                        self.route_cache[vehicle_id].append(path_id)
        
        elif delta < 0:  # 减少流量
            # 释放路径流量
            if vehicle_id in self.route_cache:
                for path_id in self.route_cache[vehicle_id]:
                    self.backbone_network.update_traffic_flow(path_id, delta)
                
                self.route_cache[vehicle_id] = []
    
    def release_path(self, vehicle_id):
        """释放车辆占用的路径流量"""
        self._update_backbone_traffic(None, None, vehicle_id, -1)