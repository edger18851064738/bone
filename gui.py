class BackbonePathVisualizer(QGraphicsItemGroup):
    """主干路径可视化组件"""
    
    def __init__(self, backbone_network, parent=None):
        super().__init__(parent)
        self.backbone_network = backbone_network
        self.path_items = {}
        self.connection_items = {}
        self.node_items = {}
        self.setZValue(1)  # 确保显示在合适的层
        self.update_visualization()
    
    def update_visualization(self):
        """更新可视化"""
        # 清除现有项
        for item in self.path_items.values():
            self.removeFromGroup(item)
        
        for item in self.connection_items.values():
            self.removeFromGroup(item)
        
        for item in self.node_items.values():
            self.removeFromGroup(item)
        
        self.path_items = {}
        self.connection_items = {}
        self.node_items = {}
        
        if not self.backbone_network:
            return
        
        # 绘制主干路径
        for path_id, path_data in self.backbone_network.paths.items():
            path = path_data['path']
            
            if not path or len(path) < 2:
                continue
            
            # 创建路径项
            painter_path = QPainterPath()
            painter_path.moveTo(path[0][0], path[0][1])
            
            for point in path[1:]:
                painter_path.lineTo(point[0], point[1])
            
            path_item = QGraphicsPathItem(painter_path)
            
            # 设置样式，使用渐变色
            gradient = QLinearGradient(path[0][0], path[0][1], path[-1][0], path[-1][1])
            gradient.setColorAt(0, QColor(40, 120, 180, 180))  # 起点颜色
            gradient.setColorAt(1, QColor(120, 40, 180, 180))  # 终点颜色
            
            pen = QPen(gradient, 2.0)  # 加粗的路径线
            path_item.setPen(pen)
            
            # 添加到组
            self.addToGroup(path_item)
            self.path_items[path_id] = path_item
        
        # 绘制连接点
        for conn_id, conn_data in self.backbone_network.connections.items():
            position = conn_data['position']
            conn_type = conn_data.get('type', 'midpath')
            
            # 根据类型选择不同样式
            if conn_type == 'endpoint':
                # 端点连接点，较大
                radius = 2.5
                color = QColor(220, 120, 40)  # 橙色
            else:
                # 中间连接点，较小
                radius = 1.5
                color = QColor(120, 220, 40)  # 绿色
            
            conn_item = QGraphicsEllipseItem(
                position[0] - radius, position[1] - radius,
                radius * 2, radius * 2
            )
            
            conn_item.setBrush(QBrush(color))
            conn_item.setPen(QPen(Qt.black, 0.5))
            
            # 添加到组
            self.addToGroup(conn_item)
            self.connection_items[conn_id] = conn_item
        
        # 绘制节点（如装载点、卸载点等）
        # 遍历所有路径的起点和终点
        nodes = {}
        
        for path_id, path_data in self.backbone_network.paths.items():
            start = path_data['start']
            end = path_data['end']
            
            nodes[start['id']] = {
                'position': start['position'],
                'type': start['type']
            }
            
            nodes[end['id']] = {
                'position': end['position'],
                'type': end['type']
            }
        
        for node_id, node_data in nodes.items():
            position = node_data['position']
            node_type = node_data['type']
            
            # 根据类型选择不同样式
            if node_type == 'loading_point':
                # 装载点，较大
                radius = 4.0
                color = QColor(0, 180, 0)  # 绿色
            elif node_type == 'unloading_point':
                # 卸载点，较大
                radius = 4.0
                color = QColor(180, 0, 0)  # 红色
            else:
                # 其他点，中等大小
                radius = 3.0
                color = QColor(100, 100, 100)  # 灰色
            
            node_item = QGraphicsEllipseItem(
                position[0] - radius, position[1] - radius,
                radius * 2, radius * 2
            )
            
            node_item.setBrush(QBrush(color))
            node_item.setPen(QPen(Qt.black, 0.8))
            
            # 添加标签
            label_item = QGraphicsTextItem(node_id)
            label_item.setPos(position[0] + radius + 1, position[1] - radius - 2)
            label_item.setFont(QFont("Arial", 3))
            
            # 添加到组
            self.addToGroup(node_item)
            self.addToGroup(label_item)
            self.node_items[node_id] = node_item
    
    def update_traffic_flow(self):
        """更新交通流量可视化"""
        if not self.backbone_network:
            return
        
        # 根据流量更新路径样式
        for path_id, path_item in self.path_items.items():
            if path_id in self.backbone_network.paths:
                path_data = self.backbone_network.paths[path_id]
                traffic_flow = path_data.get('traffic_flow', 0)
                capacity = path_data.get('capacity', 1)
                
                # 计算流量比例
                ratio = min(1.0, traffic_flow / max(1, capacity))
                
                # 根据流量比例调整线宽和颜色
                width = 1.0 + ratio * 3.0  # 1-4之间的线宽
                
                # 颜色从绿色到黄色再到红色
                if ratio < 0.5:
                    # 绿色到黄色
                    r = int(255 * ratio * 2)
                    g = 255
                    b = 0
                else:
                    # 黄色到红色
                    r = 255
                    g = int(255 * (2 - ratio * 2))
                    b = 0
                
                color = QColor(r, g, b, 180)
                
                path_item.setPen(QPen(color, width))


class MineGUI(QMainWindow):
    """露天矿多车调度系统GUI"""
    
    def __init__(self):
        super().__init__()
        
        # 系统组件
        self.env = None
        self.backbone_network = None
        self.path_planner = None
        self.vehicle_scheduler = None
        self.traffic_manager = None
        
        # 初始化UI
        self.init_ui()
        
        # 创建定时器用于更新
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(50)  # 20 FPS
    
    def init_ui(self):
        """初始化用户界面"""
        # 设置主窗口
        self.setWindowTitle("露天矿多车调度系统 (基于主干路径)")
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建中心部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 主布局
        self.main_layout = QHBoxLayout(self.central_widget)
        
        # 创建左侧控制面板
        self.create_control_panel()
        
        # 创建右侧显示区域
        self.create_display_area()
        
        # 创建菜单栏
        self.create_menu_bar()
        
        # 创建状态栏
        self.statusBar().showMessage("系统就绪")
        
        # 创建工具栏
        self.create_tool_bar()
    
    def create_control_panel(self):
        """创建左侧控制面板"""
        # 控制面板主容器
        self.control_panel = QFrame()
        self.control_panel.setFrameShape(QFrame.StyledPanel)
        self.control_panel.setMinimumWidth(280)
        self.control_panel.setMaximumWidth(350)
        
        # 控制面板布局
        control_layout = QVBoxLayout(self.control_panel)
        
        # 创建选项卡控件
        self.tab_widget = QTabWidget()
        control_layout.addWidget(self.tab_widget)
        
        # 环境选项卡
        self.env_tab = QWidget()
        self.env_layout = QVBoxLayout(self.env_tab)
        self.tab_widget.addTab(self.env_tab, "环境")
        
        # 环境选项卡内容
        self.create_env_tab()
        
        # 路径选项卡
        self.path_tab = QWidget()
        self.path_layout = QVBoxLayout(self.path_tab)
        self.tab_widget.addTab(self.path_tab, "路径")
        
        # 路径选项卡内容
        self.create_path_tab()
        
        # 车辆选项卡
        self.vehicle_tab = QWidget()
        self.vehicle_layout = QVBoxLayout(self.vehicle_tab)
        self.tab_widget.addTab(self.vehicle_tab, "车辆")
        
        # 车辆选项卡内容
        self.create_vehicle_tab()
        
        # 任务选项卡
        self.task_tab = QWidget()
        self.task_layout = QVBoxLayout(self.task_tab)
        self.tab_widget.addTab(self.task_tab, "任务")
        
        # 任务选项卡内容
        self.create_task_tab()
        
        # 日志区域
        self.create_log_area(control_layout)
        
        # 添加到主布局
        self.main_layout.addWidget(self.control_panel, 1)
    
    def create_env_tab(self):
        """创建环境选项卡内容"""
        # 文件加载组
        file_group = QGroupBox("环境加载")
        file_layout = QGridLayout()
        
        # 地图文件
        file_layout.addWidget(QLabel("地图文件:"), 0, 0)
        self.map_path = QLabel("未选择")
        self.map_path.setStyleSheet("background-color: #f8f9fa; padding: 5px; border-radius: 3px;")
        file_layout.addWidget(self.map_path, 0, 1)
        
        # 浏览按钮
        self.browse_button = QPushButton("浏览...")
        self.browse_button.clicked.connect(self.open_map_file)
        file_layout.addWidget(self.browse_button, 0, 2)
        
        # 加载按钮
        self.load_button = QPushButton("加载环境")
        self.load_button.clicked.connect(self.load_environment)
        file_layout.addWidget(self.load_button, 1, 0, 1, 3)
        
        file_group.setLayout(file_layout)
        self.env_layout.addWidget(file_group)
        
        # 环境信息组
        info_group = QGroupBox("环境信息")
        info_layout = QGridLayout()
        
        labels = ["宽度:", "高度:", "装载点:", "卸载点:", "车辆数:"]
        self.env_info_values = {}
        
        for i, label in enumerate(labels):
            info_layout.addWidget(QLabel(label), i, 0)
            self.env_info_values[label] = QLabel("--")
            self.env_info_values[label].setStyleSheet("background-color: #f8f9fa; padding: 3px; border-radius: 3px;")
            info_layout.addWidget(self.env_info_values[label], i, 1)
        
        info_group.setLayout(info_layout)
        self.env_layout.addWidget(info_group)
        
        # 环境控制组
        control_group = QGroupBox("环境控制")
        control_layout = QVBoxLayout()
        
        # 模拟速度
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("模拟速度:"))
        
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(100)
        self.speed_slider.setValue(50)
        self.speed_slider.valueChanged.connect(self.update_simulation_speed)
        speed_layout.addWidget(self.speed_slider)
        
        self.speed_value = QLabel("1.0x")
        self.speed_value.setAlignment(Qt.AlignCenter)
        self.speed_value.setMinimumWidth(40)
        self.speed_value.setStyleSheet("background-color: #f8f9fa; padding: 3px; border-radius: 3px;")
        speed_layout.addWidget(self.speed_value)
        
        control_layout.addLayout(speed_layout)
        
        # 控制按钮
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("开始")
        self.start_button.clicked.connect(self.start_simulation)
        button_layout.addWidget(self.start_button)
        
        self.pause_button = QPushButton("暂停")
        self.pause_button.clicked.connect(self.pause_simulation)
        self.pause_button.setEnabled(False)
        button_layout.addWidget(self.pause_button)
        
        self.reset_button = QPushButton("重置")
        self.reset_button.clicked.connect(self.reset_simulation)
        button_layout.addWidget(self.reset_button)
        
        control_layout.addLayout(button_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        control_layout.addWidget(self.progress_bar)
        
        control_group.setLayout(control_layout)
        self.env_layout.addWidget(control_group)
        
        # 添加空白区域
        self.env_layout.addStretch()
    
    def create_path_tab(self):
        """创建路径选项卡内容"""
        # 路径生成组
        generate_group = QGroupBox("生成主干路径")
        generate_layout = QVBoxLayout()
        
        # 参数设置
        param_layout = QGridLayout()
        
        param_layout.addWidget(QLabel("连接点间距:"), 0, 0)
        self.conn_spacing = QSpinBox()
        self.conn_spacing.setRange(5, 50)
        self.conn_spacing.setValue(10)
        param_layout.addWidget(self.conn_spacing, 0, 1)
        
        param_layout.addWidget(QLabel("路径平滑度:"), 1, 0)
        self.path_smoothness = QSpinBox()
        self.path_smoothness.setRange(1, 10)
        self.path_smoothness.setValue(5)
        param_layout.addWidget(self.path_smoothness, 1, 1)
        
        generate_layout.addLayout(param_layout)
        
        # 生成按钮
        self.generate_paths_button = QPushButton("生成主干路径网络")
        self.generate_paths_button.clicked.connect(self.generate_backbone_network)
        generate_layout.addWidget(self.generate_paths_button)
        
        generate_group.setLayout(generate_layout)
        self.path_layout.addWidget(generate_group)
        
        # 路径信息组
        info_group = QGroupBox("路径网络信息")
        info_layout = QGridLayout()
        
        labels = ["路径总数:", "连接点总数:", "总长度:", "平均长度:"]
        self.path_info_values = {}
        
        for i, label in enumerate(labels):
            info_layout.addWidget(QLabel(label), i, 0)
            self.path_info_values[label] = QLabel("--")
            self.path_info_values[label].setStyleSheet("background-color: #f8f9fa; padding: 3px; border-radius: 3px;")
            info_layout.addWidget(self.path_info_values[label], i, 1)
        
        info_group.setLayout(info_layout)
        self.path_layout.addWidget(info_group)
        
        # 路径显示选项
        display_group = QGroupBox("显示选项")
        display_layout = QVBoxLayout()
        
        self.show_paths_cb = QCheckBox("显示主干路径")
        self.show_paths_cb.setChecked(True)
        self.show_paths_cb.stateChanged.connect(self.update_path_display)
        display_layout.addWidget(self.show_paths_cb)
        
        self.show_connections_cb = QCheckBox("显示连接点")
        self.show_connections_cb.setChecked(True)
        self.show_connections_cb.stateChanged.connect(self.update_path_display)
        display_layout.addWidget(self.show_connections_cb)
        
        self.show_traffic_cb = QCheckBox("显示交通流量")
        self.show_traffic_cb.setChecked(True)
        self.show_traffic_cb.stateChanged.connect(self.update_path_display)
        display_layout.addWidget(self.show_traffic_cb)
        
        display_group.setLayout(display_layout)
        self.path_layout.addWidget(display_group)
        
        # 编辑工具
        tools_group = QGroupBox("编辑工具")
        tools_layout = QVBoxLayout()
        
        self.edit_mode_cb = QCheckBox("编辑模式")
        self.edit_mode_cb.setChecked(False)
        self.edit_mode_cb.stateChanged.connect(self.toggle_edit_mode)
        tools_layout.addWidget(self.edit_mode_cb)
        
        tool_button_layout = QHBoxLayout()
        
        self.add_path_button = QPushButton("添加路径")
        self.add_path_button.setEnabled(False)
        self.add_path_button.clicked.connect(self.start_add_path)
        tool_button_layout.addWidget(self.add_path_button)
        
        self.delete_path_button = QPushButton("删除路径")
        self.delete_path_button.setEnabled(False)
        self.delete_path_button.clicked.connect(self.start_delete_path)
        tool_button_layout.addWidget(self.delete_path_button)
        
        tools_layout.addLayout(tool_button_layout)
        
        tools_group.setLayout(tools_layout)
        self.path_layout.addWidget(tools_group)
        
        # 添加空白区域
        self.path_layout.addStretch()
    
    def create_vehicle_tab(self):
        """创建车辆选项卡内容"""
        # 车辆选择组
        select_group = QGroupBox("选择车辆")
        select_layout = QVBoxLayout()
        
        self.vehicle_combo = QComboBox()
        self.vehicle_combo.currentIndexChanged.connect(self.update_vehicle_info)
        select_layout.addWidget(self.vehicle_combo)
        
        select_group.setLayout(select_layout)
        self.vehicle_layout.addWidget(select_group)
        
        # 车辆信息组
        info_group = QGroupBox("车辆信息")
        info_layout = QGridLayout()
        
        labels = ["ID:", "位置:", "状态:", "载重:", "当前任务:", "完成任务:"]
        self.vehicle_info_values = {}
        
        for i, label in enumerate(labels):
            info_layout.addWidget(QLabel(label), i, 0)
            self.vehicle_info_values[label] = QLabel("--")
            self.vehicle_info_values[label].setStyleSheet("background-color: #f8f9fa; padding: 3px; border-radius: 3px;")
            info_layout.addWidget(self.vehicle_info_values[label], i, 1)
        
        info_group.setLayout(info_layout)
        self.vehicle_layout.addWidget(info_group)
        
        # 车辆控制组
        control_group = QGroupBox("车辆控制")
        control_layout = QVBoxLayout()
        
        # 任务控制
        task_layout = QHBoxLayout()
        
        self.assign_task_button = QPushButton("分配任务")
        self.assign_task_button.clicked.connect(self.assign_vehicle_task)
        task_layout.addWidget(self.assign_task_button)
        
        self.cancel_task_button = QPushButton("取消任务")
        self.cancel_task_button.clicked.connect(self.cancel_vehicle_task)
        task_layout.addWidget(self.cancel_task_button)
        
        control_layout.addLayout(task_layout)
        
        # 位置控制
        position_layout = QHBoxLayout()
        
        self.goto_loading_button = QPushButton("前往装载点")
        self.goto_loading_button.clicked.connect(self.goto_loading_point)
        position_layout.addWidget(self.goto_loading_button)
        
        self.goto_unloading_button = QPushButton("前往卸载点")
        self.goto_unloading_button.clicked.connect(self.goto_unloading_point)
        position_layout.addWidget(self.goto_unloading_button)
        
        control_layout.addLayout(position_layout)
        
        # 添加返回起点按钮
        self.return_button = QPushButton("返回起点")
        self.return_button.clicked.connect(self.return_to_start)
        control_layout.addWidget(self.return_button)
        
        control_group.setLayout(control_layout)
        self.vehicle_layout.addWidget(control_group)
        
        # 车辆显示选项
        display_group = QGroupBox("显示选项")
        display_layout = QVBoxLayout()
        
        self.show_vehicles_cb = QCheckBox("显示所有车辆")
        self.show_vehicles_cb.setChecked(True)
        self.show_vehicles_cb.stateChanged.connect(self.update_vehicle_display)
        display_layout.addWidget(self.show_vehicles_cb)
        
        self.show_vehicle_paths_cb = QCheckBox("显示车辆路径")
        self.show_vehicle_paths_cb.setChecked(True)
        self.show_vehicle_paths_cb.stateChanged.connect(self.update_vehicle_display)
        display_layout.addWidget(self.show_vehicle_paths_cb)
        
        self.show_vehicle_labels_cb = QCheckBox("显示车辆标签")
        self.show_vehicle_labels_cb.setChecked(True)
        self.show_vehicle_labels_cb.stateChanged.connect(self.update_vehicle_display)
        display_layout.addWidget(self.show_vehicle_labels_cb)
        
        display_group.setLayout(display_layout)
        self.vehicle_layout.addWidget(display_group)
        
        # 添加空白区域
        self.vehicle_layout.addStretch()
    
    def create_task_tab(self):
        """创建任务选项卡内容"""
        # 任务列表组
        list_group = QGroupBox("任务列表")
        list_layout = QVBoxLayout()
        
        self.task_list = QListWidget()
        self.task_list.itemClicked.connect(self.update_task_info)
        list_layout.addWidget(self.task_list)
        
        list_group.setLayout(list_layout)
        self.task_layout.addWidget(list_group)
        
        # 任务信息组
        info_group = QGroupBox("任务信息")
        info_layout = QGridLayout()
        
        labels = ["ID:", "类型:", "状态:", "车辆:", "进度:", "开始时间:", "完成时间:"]
        self.task_info_values = {}
        
        for i, label in enumerate(labels):
            info_layout.addWidget(QLabel(label), i, 0)
            self.task_info_values[label] = QLabel("--")
            self.task_info_values[label].setStyleSheet("background-color: #f8f9fa; padding: 3px; border-radius: 3px;")
            info_layout.addWidget(self.task_info_values[label], i, 1)
        
        info_group.setLayout(info_layout)
        self.task_layout.addWidget(info_group)
        
        # 任务操作组
        action_group = QGroupBox("任务操作")
        action_layout = QVBoxLayout()
        
        # 模板创建按钮
        self.create_template_button = QPushButton("创建任务模板")
        self.create_template_button.clicked.connect(self.create_task_template)
        action_layout.addWidget(self.create_template_button)
        
        # 自动分配按钮
        self.auto_assign_button = QPushButton("自动分配任务")
        self.auto_assign_button.clicked.connect(self.auto_assign_tasks)
        action_layout.addWidget(self.auto_assign_button)
        
        action_group.setLayout(action_layout)
        self.task_layout.addWidget(action_group)
        
        # 统计信息组
        stats_group = QGroupBox("统计信息")
        stats_layout = QGridLayout()
        
        labels = ["总任务数:", "完成任务:", "失败任务:", "平均利用率:", "总运行时间:"]
        self.task_stats_values = {}
        
        for i, label in enumerate(labels):
            stats_layout.addWidget(QLabel(label), i, 0)
            self.task_stats_values[label] = QLabel("--")
            self.task_stats_values[label].setStyleSheet("background-color: #f8f9fa; padding: 3px; border-radius: 3px;")
            stats_layout.addWidget(self.task_stats_values[label], i, 1)
        
        stats_group.setLayout(stats_layout)
        self.task_layout.addWidget(stats_group)
        
        # 添加空白区域
        self.task_layout.addStretch()
    
    def create_log_area(self, parent_layout):
        """创建日志区域"""
        log_group = QGroupBox("系统日志")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        log_layout.addWidget(self.log_text)
        
        # 清除按钮
        self.clear_log_button = QPushButton("清除日志")
        self.clear_log_button.clicked.connect(self.clear_log)
        log_layout.addWidget(self.clear_log_button)
        
        log_group.setLayout(log_layout)
        parent_layout.addWidget(log_group)
    
    def create_display_area(self):
        """创建右侧显示区域"""
        # 创建显示区域容器
        self.display_area = QFrame()
        self.display_area.setFrameShape(QFrame.StyledPanel)
        
        # 显示区域布局
        display_layout = QVBoxLayout(self.display_area)
        
        # 创建视图控制工具栏
        view_toolbar = QToolBar()
        view_toolbar.setIconSize(QSize(16, 16))
        view_toolbar.setMovable(False)
        
        # 添加视图控制按钮
        zoom_in_action = QAction("放大", self)
        zoom_in_action.triggered.connect(self.zoom_in)
        view_toolbar.addAction(zoom_in_action)
        
        zoom_out_action = QAction("缩小", self)
        zoom_out_action.triggered.connect(self.zoom_out)
        view_toolbar.addAction(zoom_out_action)
        
        fit_view_action = QAction("适应视图", self)
        fit_view_action.triggered.connect(self.fit_view)
        view_toolbar.addAction(fit_view_action)
        
        display_layout.addWidget(view_toolbar)
        
        # 创建图形视图
        self.graphics_view = QGraphicsView()
        self.graphics_view.setRenderHint(QPainter.Antialiasing)
        self.graphics_view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        
        # 设置场景
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)
        
        display_layout.addWidget(self.graphics_view)
        
        # 添加到主布局
        self.main_layout.addWidget(self.display_area, 3)
    
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件")
        
        open_action = QAction("打开地图", self)
        open_action.triggered.connect(self.open_map_file)
        file_menu.addAction(open_action)
        
        save_action = QAction("保存结果", self)
        save_action.triggered.connect(self.save_results)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 视图菜单
        view_menu = menubar.addMenu("视图")
        
        self.show_backbone_action = QAction("显示主干路径", self, checkable=True)
        self.show_backbone_action.setChecked(True)
        self.show_backbone_action.triggered.connect(self.toggle_backbone_display)
        view_menu.addAction(self.show_backbone_action)
        
        self.show_vehicles_action = QAction("显示车辆", self, checkable=True)
        self.show_vehicles_action.setChecked(True)
        self.show_vehicles_action.triggered.connect(self.toggle_vehicles_display)
        view_menu.addAction(self.show_vehicles_action)
        
        # 工具菜单
        tools_menu = menubar.addMenu("工具")
        
        generate_network_action = QAction("生成主干网络", self)
        generate_network_action.triggered.connect(self.generate_backbone_network)
        tools_menu.addAction(generate_network_action)
        
        optimize_paths_action = QAction("优化路径", self)
        optimize_paths_action.triggered.connect(self.optimize_paths)
        tools_menu.addAction(optimize_paths_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu("帮助")
        
        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_tool_bar(self):
        """创建工具栏"""
        toolbar = QToolBar("主工具栏")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)
        
        # 打开地图
        open_action = QAction("打开地图", self)
        open_action.triggered.connect(self.open_map_file)
        toolbar.addAction(open_action)
        
        toolbar.addSeparator()
        
        # 模拟控制
        self.start_sim_action = QAction("开始", self)
        self.start_sim_action.triggered.connect(self.start_simulation)
        toolbar.addAction(self.start_sim_action)
        
        self.pause_sim_action = QAction("暂停", self)
        self.pause_sim_action.triggered.connect(self.pause_simulation)
        self.pause_sim_action.setEnabled(False)
        toolbar.addAction(self.pause_sim_action)
        
        self.reset_sim_action = QAction("重置", self)
        self.reset_sim_action.triggered.connect(self.reset_simulation)
        toolbar.addAction(self.reset_sim_action)
        
        toolbar.addSeparator()
        
        # 主干网络生成
        self.generate_network_action = QAction("生成主干网络", self)
        self.generate_network_action.triggered.connect(self.generate_backbone_network)
        toolbar.addAction(self.generate_network_action)
    
    # 事件处理方法
    def open_map_file(self):
        """打开地图文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开地图文件", "", 
            "矿山地图文件 (*.json);;所有文件 (*)"
        )
        
        if file_path:
            self.map_path.setText(os.path.basename(file_path))
            self.map_file_path = file_path
            self.log("已选择地图文件: " + os.path.basename(file_path))
    
    def load_environment(self):
        """加载环境"""
        if not hasattr(self, 'map_file_path') or not self.map_file_path:
            self.log("请先选择地图文件!", "error")
            return
        
        try:
            self.log("正在加载环境...")
            
            # 加载环境（实际实现时替换为实际的环境加载代码）
            # 示例：使用环境加载器
            from mine_loader import MineEnvironmentLoader
            loader = MineEnvironmentLoader()
            self.env = loader.load_environment(self.map_file_path)
            
            # 更新环境信息显示
            self.update_env_info()
            
            # 更新车辆下拉框
            self.update_vehicle_combo()
            
            # 创建场景
            self.create_scene()
            
            self.log("环境加载成功: " + os.path.basename(self.map_file_path))
            
            # 创建其他系统组件
            self.create_system_components()
            
            # 启用相关按钮
            self.enable_controls(True)
            
        except Exception as e:
            self.log(f"加载环境失败: {str(e)}", "error")
    
    def update_env_info(self):
        """更新环境信息显示"""
        if not self.env:
            return
        
        # 更新标签值
        self.env_info_values["宽度:"].setText(str(self.env.width))
        self.env_info_values["高度:"].setText(str(self.env.height))
        self.env_info_values["装载点:"].setText(str(len(self.env.loading_points)))
        self.env_info_values["卸载点:"].setText(str(len(self.env.unloading_points)))
        self.env_info_values["车辆数:"].setText(str(len(self.env.vehicles)))
    
    def update_vehicle_combo(self):
        """更新车辆下拉框"""
        self.vehicle_combo.clear()
        
        if not self.env:
            return
        
        for v_id in sorted(self.env.vehicles.keys()):
            self.vehicle_combo.addItem(f"车辆 {v_id}", v_id)
    
    def create_scene(self):
        """创建场景"""
        # 清除现有场景
        self.graphics_scene.clear()
        
        if not self.env:
            return
        
        # 设置场景大小
        self.graphics_scene.setSceneRect(0, 0, self.env.width, self.env.height)
        
        # 绘制背景网格
        self.draw_grid()
        
        # 绘制障碍物
        self.draw_obstacles()
        
        # 绘制关键点（装载点、卸载点等）
        self.draw_key_points()
        
        # 绘制车辆
        self.draw_vehicles()
        
        # 调整视图以显示整个场景
        self.fit_view()
    
    def draw_grid(self):
        """绘制背景网格"""
        if not self.env:
            return
        
        # 设置网格大小
        grid_size = 10
        
        # 网格线颜色
        grid_pen = QPen(QColor(220, 220, 220))
        grid_pen.setWidth(0)
        
        # 主网格线颜色
        major_grid_pen = QPen(QColor(200, 200, 200))
        major_grid_pen.setWidth(0)
        
        # 绘制垂直线
        for x in range(0, self.env.width + 1, grid_size):
            if x % (grid_size * 5) == 0:
                # 主网格线
                line = self.graphics_scene.addLine(x, 0, x, self.env.height, major_grid_pen)
            else:
                # 次网格线
                line = self.graphics_scene.addLine(x, 0, x, self.env.height, grid_pen)
            
            line.setZValue(-100)  # 确保网格在最底层
        
        # 绘制水平线
        for y in range(0, self.env.height + 1, grid_size):
            if y % (grid_size * 5) == 0:
                # 主网格线
                line = self.graphics_scene.addLine(0, y, self.env.width, y, major_grid_pen)
            else:
                # 次网格线
                line = self.graphics_scene.addLine(0, y, self.env.width, y, grid_pen)
            
            line.setZValue(-100)  # 确保网格在最底层
        
        # 添加坐标标签
        label_interval = 50  # 每50个单位显示一个标签
        
        for x in range(0, self.env.width + 1, label_interval):
            label = self.graphics_scene.addText(str(x))
            label.setPos(x, 5)
            label.setDefaultTextColor(QColor(100, 100, 100))
            label.setZValue(-90)
        
        for y in range(0, self.env.height + 1, label_interval):
            label = self.graphics_scene.addText(str(y))
            label.setPos(5, y)
            label.setDefaultTextColor(QColor(100, 100, 100))
            label.setZValue(-90)
    
    def draw_obstacles(self):
        """绘制障碍物"""
        if not self.env:
            return
        
        # 障碍物颜色
        obstacle_brush = QBrush(QColor(80, 80, 90))
        obstacle_pen = QPen(QColor(60, 60, 70), 0.5)
        
        # 遍历环境网格，标记障碍物
        for x in range(self.env.width):
            for y in range(self.env.height):
                if self.env.grid[x, y] == 1:  # 障碍物
                    rect = self.graphics_scene.addRect(x, y, 1, 1, obstacle_pen, obstacle_brush)
                    rect.setZValue(-50)  # 确保在网格之上
    
    def draw_key_points(self):
        """绘制关键点（装载点、卸载点等）"""
        if not self.env:
            return
        
        # 装载点
        for i, point in enumerate(self.env.loading_points):
            x, y = point
            
            # 创建装载点图形
            radius = 4.0
            loading_item = self.graphics_scene.addEllipse(
                x - radius, y - radius, radius * 2, radius * 2,
                QPen(QColor(0, 120, 0), 1.0),
                QBrush(QColor(0, 180, 0, 180))
            )
            loading_item.setZValue(5)
            
            # 添加标签
            text = self.graphics_scene.addText(f"装载点{i+1}")
            text.setPos(x - 20, y - 15)
            text.setDefaultTextColor(QColor(0, 120, 0))
            text.setZValue(5)
        
        # 卸载点
        for i, point in enumerate(self.env.unloading_points):
            x, y = point
            
            # 创建卸载点图形
            radius = 4.0
            unloading_item = self.graphics_scene.addEllipse(
                x - radius, y - radius, radius * 2, radius * 2,
                QPen(QColor(120, 0, 0), 1.0),
                QBrush(QColor(180, 0, 0, 180))
            )
            unloading_item.setZValue(5)
            
            # 添加标签
            text = self.graphics_scene.addText(f"卸载点{i+1}")
            text.setPos(x - 20, y - 15)
            text.setDefaultTextColor(QColor(120, 0, 0))
            text.setZValue(5)
    
    def draw_vehicles(self):
        """绘制车辆"""
        if not self.env:
            return
        
        # 清除现有车辆图形
        for item in getattr(self, 'vehicle_items', {}).values():
            self.graphics_scene.removeItem(item)
        
        self.vehicle_items = {}
        
        # 车辆颜色
        vehicle_colors = {
            'idle': QColor(128, 128, 128),      # 灰色
            'moving': QColor(0, 123, 255),      # 蓝色
            'loading': QColor(40, 167, 69),     # 绿色
            'unloading': QColor(220, 53, 69)    # 红色
        }
        
        # 添加新车辆图形
        for vehicle_id, vehicle_data in self.env.vehicles.items():
            # 获取车辆位置
            position = vehicle_data.get('position', (0, 0, 0))
            x, y, theta = position
            
            # 创建车辆图形
            vehicle_length = 6.0
            vehicle_width = 3.0
            
            # 构建车辆多边形
            polygon = QPolygonF()
            
            # 计算车辆四个角点的相对坐标（默认朝向为Y轴正方向）
            half_length = vehicle_length / 2
            half_width = vehicle_width / 2
            
            corners = [
                QPointF(half_width, half_length),     # 右前
                QPointF(-half_width, half_length),    # 左前
                QPointF(-half_width, -half_length),   # 左后
                QPointF(half_width, -half_length)     # 右后
            ]
            
            # 创建旋转和平移变换
            transform = QTransform()
            transform.rotate(theta * 180 / math.pi)  # 旋转（角度制）
            transform.translate(x, y)  # 平移
            
            # 应用变换并添加到多边形
            for corner in corners:
                polygon.append(transform.map(corner))
            
            # 创建车辆图形项
            status = vehicle_data.get('status', 'idle')
            color = vehicle_colors.get(status, vehicle_colors['idle'])
            
            vehicle_item = self.graphics_scene.addPolygon(
                polygon,
                QPen(Qt.black, 0.5),
                QBrush(color)
            )
            vehicle_item.setZValue(10)  # 确保车辆在最上层
            
            # 添加车辆ID标签
            label = self.graphics_scene.addText(str(vehicle_id))
            label.setPos(x - 5, y - 5)
            label.setDefaultTextColor(Qt.white)
            label.setZValue(11)
            
            # 存储车辆图形项（包括多边形和标签）
            self.vehicle_items[vehicle_id] = {
                'polygon': vehicle_item,
                'label': label
            }
    
    def create_system_components(self):
        """创建系统其他组件"""
        if not self.env:
            return
        
        # 创建主干路径网络
        self.backbone_network = BackbonePathNetwork(self.env)
        
        # 创建路径规划器
        self.path_planner = PathPlanner(self.env)
        
        # 创建车辆调度器
        self.vehicle_scheduler = VehicleScheduler(self.env, self.path_planner)
        
        # 创建交通管理器
        self.traffic_manager = TrafficManager(self.env)
        
        # 初始化车辆状态
        self.vehicle_scheduler.initialize_vehicles()
        
        # 创建任务模板
        if self.env.loading_points and self.env.unloading_points:
            self.vehicle_scheduler.create_mission_template("default")
        
        self.log("系统组件初始化完成")
    
    def generate_backbone_network(self):
        """生成主干路径网络"""
        if not self.env:
            self.log("请先加载环境!", "error")
            return
        
        try:
            self.log("正在生成主干路径网络...")
            
            # 生成网络
            self.backbone_network.generate_network()
            
            # 更新路径信息显示
            self.update_path_info()
            
            # 设置规划器和交通管理器
            self.path_planner.set_backbone_network(self.backbone_network)
            self.traffic_manager.set_backbone_network(self.backbone_network)
            
            # 绘制主干路径网络
            self.draw_backbone_network()
            
            self.log(f"主干路径网络生成成功 - {len(self.backbone_network.paths)} 条路径")
            
        except Exception as e:
            self.log(f"生成主干路径网络失败: {str(e)}", "error")
    
    def update_path_info(self):
        """更新路径信息显示"""
        if not self.backbone_network:
            return
        
        # 计算统计信息
        num_paths = len(self.backbone_network.paths)
        num_connections = len(self.backbone_network.connections)
        
        total_length = 0
        for path_data in self.backbone_network.paths.values():
            total_length += path_data['length']
        
        avg_length = total_length / num_paths if num_paths > 0 else 0
        
        # 更新显示
        self.path_info_values["路径总数:"].setText(str(num_paths))
        self.path_info_values["连接点总数:"].setText(str(num_connections))
        self.path_info_values["总长度:"].setText(f"{total_length:.1f}")
        self.path_info_values["平均长度:"].setText(f"{avg_length:.1f}")
    
    def draw_backbone_network(self):
        """绘制主干路径网络"""
        # 清除现有主干网络图形
        if hasattr(self, 'backbone_visualizer'):
            self.graphics_scene.removeItem(self.backbone_visualizer)
        
        # 创建新的可视化器
        self.backbone_visualizer = BackbonePathVisualizer(self.backbone_network)
        self.graphics_scene.addItem(self.backbone_visualizer)
    
    def update_display(self):
        """更新显示（由定时器触发）"""
        if not self.env:
            return
        
        # 更新车辆位置
        self.update_vehicles()
        
        # 更新选中车辆信息
        index = self.vehicle_combo.currentIndex()
        if index >= 0:
            self.update_vehicle_info(index)
        
        # 更新任务信息
        self.update_task_list()
        
        # 更新主干网络流量可视化
        if hasattr(self, 'backbone_visualizer') and self.backbone_visualizer:
            self.backbone_visualizer.update_traffic_flow()
    
    def update_vehicles(self):
        """更新车辆位置和状态"""
        if not self.env or not hasattr(self, 'vehicle_items'):
            return
        
        # 车辆颜色
        vehicle_colors = {
            'idle': QColor(128, 128, 128),      # 灰色
            'moving': QColor(0, 123, 255),      # 蓝色
            'loading': QColor(40, 167, 69),     # 绿色
            'unloading': QColor(220, 53, 69)    # 红色
        }
        
        # 更新所有车辆
        for vehicle_id, vehicle_data in self.env.vehicles.items():
            # 获取车辆位置
            position = vehicle_data.get('position', (0, 0, 0))
            x, y, theta = position
            
            # 如果车辆已经有图形项
            if vehicle_id in self.vehicle_items:
                items = self.vehicle_items[vehicle_id]
                
                # 获取状态
                status = vehicle_data.get('status', 'idle')
                color = vehicle_colors.get(status, vehicle_colors['idle'])
                
                # 更新多边形
                vehicle_length = 6.0
                vehicle_width = 3.0
                
                # 构建车辆多边形
                polygon = QPolygonF()
                
                # 计算车辆四个角点的相对坐标
                half_length = vehicle_length / 2
                half_width = vehicle_width / 2
                
                corners = [
                    QPointF(half_width, half_length),     # 右前
                    QPointF(-half_width, half_length),    # 左前
                    QPointF(-half_width, -half_length),   # 左后
                    QPointF(half_width, -half_length)     # 右后
                ]
                
                # 创建旋转和平移变换
                transform = QTransform()
                transform.rotate(theta * 180 / math.pi)  # 旋转（角度制）
                transform.translate(x, y)  # 平移
                
                # 应用变换并添加到多边形
                for corner in corners:
                    polygon.append(transform.map(corner))
                
                # 更新多边形
                items['polygon'].setPolygon(polygon)
                items['polygon'].setBrush(QBrush(color))
                
                # 更新标签位置
                items['label'].setPos(x - 5, y - 5)
    
    def update_vehicle_info(self, index):
        """更新车辆信息显示"""
        if not self.env or index < 0:
            return
        
        # 获取选中的车辆ID
        vehicle_id = self.vehicle_combo.itemData(index)
        
        if vehicle_id not in self.env.vehicles:
            return
        
        # 获取车辆信息
        vehicle_data = self.env.vehicles[vehicle_id]
        
        # 如果有调度器，获取更详细的信息
        if self.vehicle_scheduler:
            vehicle_info = self.vehicle_scheduler.get_vehicle_info(vehicle_id)
        else:
            vehicle_info = vehicle_data
        
        # 更新显示
        position = vehicle_data.get('position', (0, 0, 0))
        self.vehicle_info_values["ID:"].setText(str(vehicle_id))
        self.vehicle_info_values["位置:"].setText(f"({position[0]:.1f}, {position[1]:.1f})")
        
        status = vehicle_data.get('status', 'idle')
        status_map = {
            'idle': '空闲',
            'moving': '移动中',
            'loading': '装载中',
            'unloading': '卸载中'
        }
        self.vehicle_info_values["状态:"].setText(status_map.get(status, status))
        
        load = vehicle_data.get('load', 0)
        max_load = vehicle_data.get('max_load', 100)
        self.vehicle_info_values["载重:"].setText(f"{load}/{max_load}")
        
        # 显示当前任务
        if vehicle_info and 'current_task' in vehicle_info and vehicle_info['current_task']:
            task_id = vehicle_info['current_task']
            if task_id in self.vehicle_scheduler.tasks:
                task = self.vehicle_scheduler.tasks[task_id]
                task_text = f"{task.task_type} ({task.progress:.0%})"
                self.vehicle_info_values["当前任务:"].setText(task_text)
            else:
                self.vehicle_info_values["当前任务:"].setText(str(task_id))
        else:
            self.vehicle_info_values["当前任务:"].setText("无")
        
        # 显示已完成任务数
        completed = vehicle_info.get('completed_tasks', 0) if vehicle_info else 0
        self.vehicle_info_values["完成任务:"].setText(str(completed))
    
    def update_task_list(self):
        """更新任务列表"""
        if not self.vehicle_scheduler:
            return
        
        # 保存当前选择
        current_item = self.task_list.currentItem()
        current_task_id = current_item.data(Qt.UserRole) if current_item else None
        
        # 清除列表
        self.task_list.clear()
        
        # 添加所有任务
        for task_id, task in self.vehicle_scheduler.tasks.items():
            item = QListWidgetItem(f"{task_id} - {task.task_type} ({task.status})")
            item.setData(Qt.UserRole, task_id)
            
            # 根据状态设置颜色
            if task.status == 'completed':
                item.setForeground(QBrush(QColor(40, 167, 69)))  # 绿色
            elif task.status == 'failed':
                item.setForeground(QBrush(QColor(220, 53, 69)))  # 红色
            elif task.status == 'in_progress':
                item.setForeground(QBrush(QColor(0, 123, 255)))  # 蓝色
            
            self.task_list.addItem(item)
            
            # 如果是之前选中的任务，重新选中
            if task_id == current_task_id:
                self.task_list.setCurrentItem(item)
        
        # 更新统计信息
        stats = self.vehicle_scheduler.get_stats()
        
        self.task_stats_values["总任务数:"].setText(str(len(self.vehicle_scheduler.tasks)))
        self.task_stats_values["完成任务:"].setText(str(stats['completed_tasks']))
        self.task_stats_values["失败任务:"].setText(str(stats['failed_tasks']))
        
        avg_util = stats.get('average_utilization', 0)
        self.task_stats_values["平均利用率:"].setText(f"{avg_util:.1%}")
        
        # 如果环境有当前时间，显示运行时间
        if hasattr(self.env, 'current_time'):
            self.task_stats_values["总运行时间:"].setText(f"{self.env.current_time:.1f}")
    
    def update_task_info(self, item):
        """更新任务信息显示"""
        if not item or not self.vehicle_scheduler:
            return
        
        # 获取任务ID
        task_id = item.data(Qt.UserRole)
        
        if task_id not in self.vehicle_scheduler.tasks:
            return
        
        # 获取任务信息
        task = self.vehicle_scheduler.tasks[task_id]
        
        # 更新显示
        self.task_info_values["ID:"].setText(task.task_id)
        self.task_info_values["类型:"].setText(task.task_type)
        self.task_info_values["状态:"].setText(task.status)
        self.task_info_values["车辆:"].setText(str(task.assigned_vehicle) if task.assigned_vehicle else "未分配")
        self.task_info_values["进度:"].setText(f"{task.progress:.0%}")
        self.task_info_values["开始时间:"].setText(f"{task.start_time:.1f}" if task.start_time else "--")
        self.task_info_values["完成时间:"].setText(f"{task.completion_time:.1f}" if task.completion_time else "--")
    
    def enable_controls(self, enabled):
        """启用或禁用控制按钮"""
        # 环境选项卡
        self.start_button.setEnabled(enabled)
        self.reset_button.setEnabled(enabled)
        
        # 工具栏
        self.start_sim_action.setEnabled(enabled)
        self.reset_sim_action.setEnabled(enabled)
        
        # 路径选项卡
        self.generate_paths_button.setEnabled(enabled)
        
        # 车辆选项卡按钮
        self.assign_task_button.setEnabled(enabled)
        self.cancel_task_button.setEnabled(enabled)
        self.goto_loading_button.setEnabled(enabled)
        self.goto_unloading_button.setEnabled(enabled)
        self.return_button.setEnabled(enabled)
        
        # 任务选项卡按钮
        self.create_template_button.setEnabled(enabled)
        self.auto_assign_button.setEnabled(enabled)
    
    def log(self, message, level="info"):
        """添加日志消息"""
        # 获取当前时间
        current_time = time.strftime("%H:%M:%S")
        
        # 根据日志级别设置颜色
        if level == "error":
            color = "red"
        elif level == "warning":
            color = "orange"
        elif level == "success":
            color = "green"
        else:
            color = "black"
        
        # 格式化消息
        formatted_message = f'<span style="color: {color};">[{current_time}] {message}</span>'
        
        # 添加到日志文本框
        self.log_text.append(formatted_message)
        
        # 滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        # 同时更新状态栏
        self.statusBar().showMessage(message)
    
    def clear_log(self):
        """清除日志"""
        self.log_text.clear()
    
    # 视图控制方法
    def zoom_in(self):
        """放大视图"""
        self.graphics_view.scale(1.2, 1.2)
    
    def zoom_out(self):
        """缩小视图"""
        self.graphics_view.scale(1/1.2, 1/1.2)
    
    def fit_view(self):
        """调整视图以显示整个场景"""
        self.graphics_view.fitInView(self.graphics_scene.sceneRect(), Qt.KeepAspectRatio)
    
    # 模拟控制方法
    def update_simulation_speed(self):
        """更新模拟速度"""
        value = self.speed_slider.value()
        speed = value / 50.0  # 转换为速度倍率 (0.02-2.0)
        
        self.speed_value.setText(f"{speed:.1f}x")
        
        # 如果环境有时间步长属性，更新时间步长
        if hasattr(self.env, 'time_step'):
            self.env.time_step = 0.5 * speed  # 基础时间步长 * 速度倍率
        
        self.log(f"模拟速度已设置为 {speed:.1f}x")
    
    def start_simulation(self):
        """开始模拟"""
        if not self.env:
            self.log("请先加载环境!", "error")
            return
        
        # 更新按钮状态
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        
        self.start_sim_action.setEnabled(False)
        self.pause_sim_action.setEnabled(True)
        
        # 开始模拟逻辑
        self.is_simulating = True
        
        # 创建模拟定时器
        if not hasattr(self, 'sim_timer'):
            self.sim_timer = QTimer(self)
            self.sim_timer.timeout.connect(self.simulation_step)
        
        # 启动定时器
        self.sim_timer.start(100)  # 100ms一次更新
        
        self.log("模拟开始运行")
    
    def pause_simulation(self):
        """暂停模拟"""
        # 更新按钮状态
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        
        self.start_sim_action.setEnabled(True)
        self.pause_sim_action.setEnabled(False)
        
        # 暂停模拟逻辑
        self.is_simulating = False
        
        # 停止定时器
        if hasattr(self, 'sim_timer') and self.sim_timer.isActive():
            self.sim_timer.stop()
        
        self.log("模拟已暂停")
    
    def reset_simulation(self):
        """重置模拟"""
        # 询问确认
        reply = QMessageBox.question(
            self, '确认重置', 
            '确定要重置模拟吗？这将清除所有当前数据。',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # 停止模拟
        if hasattr(self, 'is_simulating') and self.is_simulating:
            self.pause_simulation()
        
        # 重置环境
        if self.env:
            self.env.reset()
        
        # 重置调度器
        if self.vehicle_scheduler:
            self.vehicle_scheduler.initialize_vehicles()
        
        # 重置交通管理器
        if self.traffic_manager:
            # 交通管理器没有显式的重置方法，可能需要重新创建
            self.traffic_manager = TrafficManager(self.env)
            self.traffic_manager.set_backbone_network(self.backbone_network)
        
        # 更新显示
        self.create_scene()
        
        # 如果已经生成了主干网络，重新绘制
        if self.backbone_network and self.backbone_network.paths:
            self.draw_backbone_network()
        
        # 重置进度条
        self.progress_bar.setValue(0)
        
        # 更新车辆和任务信息
        self.update_vehicle_combo()
        if self.vehicle_combo.count() > 0:
            self.update_vehicle_info(0)
        
        self.update_task_list()
        
        self.log("模拟已重置")
    
    def simulation_step(self):
        """模拟单步执行"""
        if not self.is_simulating or not self.env:
            return
        
        # 获取时间步长
        time_step = getattr(self.env, 'time_step', 0.5)
        
        # 更新环境时间
        if not hasattr(self.env, 'current_time'):
            self.env.current_time = 0
        
        self.env.current_time += time_step
        
        # 更新车辆调度器
        if self.vehicle_scheduler:
            self.vehicle_scheduler.update(time_step)
        
        # 更新进度条
        max_time = 3600  # 最长模拟时间1小时
        progress = min(100, int(self.env.current_time * 100 / max_time))
        self.progress_bar.setValue(progress)
        
        # 如果进度达到100%，自动停止
        if progress >= 100:
            self.pause_simulation()
            self.log("模拟已完成", "success")
    
    # 车辆操作方法
    def assign_vehicle_task(self):
        """为当前选中的车辆分配任务"""
        if not self.vehicle_scheduler:
            self.log("车辆调度器未初始化", "error")
            return
        
        # 获取选中的车辆
        index = self.vehicle_combo.currentIndex()
        if index < 0:
            self.log("请先选择车辆", "warning")
            return
        
        vehicle_id = self.vehicle_combo.itemData(index)
        
        # 如果已经有任务模板，使用默认模板
        if "default" in self.vehicle_scheduler.mission_templates:
            if self.vehicle_scheduler.assign_mission(vehicle_id, "default"):
                self.log(f"已为车辆 {vehicle_id} 分配默认任务", "success")
            else:
                self.log(f"为车辆 {vehicle_id} 分配任务失败", "error")
        else:
            self.log("没有可用的任务模板", "error")
    
    def cancel_vehicle_task(self):
        """取消当前选中车辆的任务"""
        if not self.vehicle_scheduler:
            return
        
        # 获取选中的车辆
        index = self.vehicle_combo.currentIndex()
        if index < 0:
            self.log("请先选择车辆", "warning")
            return
        
        vehicle_id = self.vehicle_combo.itemData(index)
        
        # 清空任务队列
        if vehicle_id in self.vehicle_scheduler.task_queues:
            self.vehicle_scheduler.task_queues[vehicle_id] = []
            
            # 如果有当前任务，标记为完成
            status = self.vehicle_scheduler.vehicle_statuses[vehicle_id]
            if status['current_task']:
                task_id = status['current_task']
                if task_id in self.vehicle_scheduler.tasks:
                    task = self.vehicle_scheduler.tasks[task_id]
                    task.status = 'completed'
                    task.progress = 1.0
                    task.completion_time = self.env.current_time if hasattr(self.env, 'current_time') else 0
                
                # 重置车辆状态
                status['status'] = 'idle'
                status['current_task'] = None
                self.env.vehicles[vehicle_id]['status'] = 'idle'
                
                # 释放交通管理器中的路径
                if self.traffic_manager:
                    self.traffic_manager.release_vehicle_path(vehicle_id)
                
                self.log(f"已取消车辆 {vehicle_id} 的所有任务", "success")
            else:
                self.log(f"车辆 {vehicle_id} 没有活动任务", "warning")
    
    def goto_loading_point(self):
        """命令当前选中车辆前往装载点"""
        if not self.vehicle_scheduler or not self.env.loading_points:
            return
        
        # 获取选中的车辆
        index = self.vehicle_combo.currentIndex()
        if index < 0:
            self.log("请先选择车辆", "warning")
            return
        
        vehicle_id = self.vehicle_combo.itemData(index)
        
        # 选择一个装载点
        loading_point = self.env.loading_points[0]
        
        # 创建任务
        task_id = f"task_{self.vehicle_scheduler.task_counter}"
        self.vehicle_scheduler.task_counter += 1
        
        vehicle_position = self.env.vehicles[vehicle_id]['position']
        
        task = VehicleTask(
            task_id,
            'to_loading',
            vehicle_position,
            (loading_point[0], loading_point[1], 0),
            1
        )
        
        # 添加到任务字典
        self.vehicle_scheduler.tasks[task_id] = task
        
        # 清空并添加到车辆任务队列
        self.vehicle_scheduler.task_queues[vehicle_id] = [task_id]
        
        # 如果车辆空闲，立即开始执行任务
        status = self.vehicle_scheduler.vehicle_statuses[vehicle_id]
        if status['status'] == 'idle':
            self.vehicle_scheduler._start_next_task(vehicle_id)
        
        self.log(f"已命令车辆 {vehicle_id} 前往装载点", "success")
    
    def goto_unloading_point(self):
        """命令当前选中车辆前往卸载点"""
        if not self.vehicle_scheduler or not self.env.unloading_points:
            return
        
        # 获取选中的车辆
        index = self.vehicle_combo.currentIndex()
        if index < 0:
            self.log("请先选择车辆", "warning")
            return
        
        vehicle_id = self.vehicle_combo.itemData(index)
        
        # 选择一个卸载点
        unloading_point = self.env.unloading_points[0]
        
        # 创建任务
        task_id = f"task_{self.vehicle_scheduler.task_counter}"
        self.vehicle_scheduler.task_counter += 1
        
        vehicle_position = self.env.vehicles[vehicle_id]['position']
        
        task = VehicleTask(
            task_id,
            'to_unloading',
            vehicle_position,
            (unloading_point[0], unloading_point[1], 0),
            1
        )
        
        # 添加到任务字典
        self.vehicle_scheduler.tasks[task_id] = task
        
        # 清空并添加到车辆任务队列
        self.vehicle_scheduler.task_queues[vehicle_id] = [task_id]
        
        # 如果车辆空闲，立即开始执行任务
        status = self.vehicle_scheduler.vehicle_statuses[vehicle_id]
        if status['status'] == 'idle':
            self.vehicle_scheduler._start_next_task(vehicle_id)
        
        self.log(f"已命令车辆 {vehicle_id} 前往卸载点", "success")
    
    def return_to_start(self):
        """命令当前选中车辆返回起点"""
        if not self.vehicle_scheduler:
            return
        
        # 获取选中的车辆
        index = self.vehicle_combo.currentIndex()
        if index < 0:
            self.log("请先选择车辆", "warning")
            return
        
        vehicle_id = self.vehicle_combo.itemData(index)
        
        # 获取车辆初始位置
        if vehicle_id not in self.env.vehicles:
            return
        
        vehicle = self.env.vehicles[vehicle_id]
        initial_position = vehicle.get('initial_position')
        
        if not initial_position:
            self.log(f"车辆 {vehicle_id} 没有初始位置", "error")
            return
        
        # 创建任务
        task_id = f"task_{self.vehicle_scheduler.task_counter}"
        self.vehicle_scheduler.task_counter += 1
        
        vehicle_position = vehicle['position']
        
        task = VehicleTask(
            task_id,
            'to_initial',
            vehicle_position,
            initial_position,
            1
        )
        
        # 添加到任务字典
        self.vehicle_scheduler.tasks[task_id] = task
        
        # 清空并添加到车辆任务队列
        self.vehicle_scheduler.task_queues[vehicle_id] = [task_id]
        
        # 如果车辆空闲，立即开始执行任务
        status = self.vehicle_scheduler.vehicle_statuses[vehicle_id]
        if status['status'] == 'idle':
            self.vehicle_scheduler._start_next_task(vehicle_id)
        
        self.log(f"已命令车辆 {vehicle_id} 返回起点", "success")
    
    # 任务操作方法
    def create_task_template(self):
        """创建任务模板"""
        if not self.vehicle_scheduler or not self.env.loading_points or not self.env.unloading_points:
            self.log("环境未初始化或缺少装载点/卸载点", "error")
            return
        
        # 创建默认模板
        if self.vehicle_scheduler.create_mission_template("default"):
            self.log("已创建默认任务模板", "success")
        else:
            self.log("创建任务模板失败", "error")
    
    def auto_assign_tasks(self):
        """自动为所有车辆分配任务"""
        if not self.vehicle_scheduler:
            self.log("车辆调度器未初始化", "error")
            return
        
        # 确保有任务模板
        if "default" not in self.vehicle_scheduler.mission_templates:
            self.create_task_template()
        
        # 为所有车辆分配任务
        success_count = 0
        for vehicle_id in self.env.vehicles.keys():
            if self.vehicle_scheduler.assign_mission(vehicle_id, "default"):
                success_count += 1
        
        self.log(f"已为 {success_count}/{len(self.env.vehicles)} 辆车分配任务", "success")
    
    # 显示选项控制
    def update_path_display(self):
        """更新路径显示选项"""
        if not hasattr(self, 'backbone_visualizer') or not self.backbone_visualizer:
            return
        
        # 显示/隐藏主干路径
        self.backbone_visualizer.setVisible(self.show_paths_cb.isChecked())
        
        # TODO: 实现连接点和交通流量的显示控制
    
    def update_vehicle_display(self):
        """更新车辆显示选项"""
        if not hasattr(self, 'vehicle_items'):
            return
        
        # 显示/隐藏所有车辆
        show_vehicles = self.show_vehicles_cb.isChecked()
        
        for vehicle_id, items in self.vehicle_items.items():
            items['polygon'].setVisible(show_vehicles)
            
            # 根据标签选项显示/隐藏标签
            items['label'].setVisible(show_vehicles and self.show_vehicle_labels_cb.isChecked())
        
        # TODO: 实现车辆路径的显示控制
    
    def toggle_edit_mode(self, checked):
        """切换编辑模式"""
        self.add_path_button.setEnabled(checked)
        self.delete_path_button.setEnabled(checked)
        
        # TODO: 实现编辑模式的功能
    
    def start_add_path(self):
        """开始添加路径"""
        # TODO: 实现添加路径功能
        self.log("添加路径功能尚未实现", "warning")
    
    def start_delete_path(self):
        """开始删除路径"""
        # TODO: 实现删除路径功能
        self.log("删除路径功能尚未实现", "warning")
    
    def toggle_backbone_display(self, checked):
        """切换主干路径显示"""
        if hasattr(self, 'backbone_visualizer') and self.backbone_visualizer:
            self.backbone_visualizer.setVisible(checked)
            
            # 同步复选框状态
            self.show_paths_cb.setChecked(checked)
    
    def toggle_vehicles_display(self, checked):
        """切换车辆显示"""
        if hasattr(self, 'vehicle_items'):
            for vehicle_id, items in self.vehicle_items.items():
                items['polygon'].setVisible(checked)
                items['label'].setVisible(checked and self.show_vehicle_labels_cb.isChecked())
            
            # 同步复选框状态
            self.show_vehicles_cb.setChecked(checked)
    
    def optimize_paths(self):
        """优化路径"""
        if not self.backbone_network or not self.backbone_network.paths:
            self.log("请先生成主干路径网络", "warning")
            return
        
        try:
            self.log("正在优化路径...")
            
            # 重新优化所有路径
            self.backbone_network._optimize_all_paths()
            
            # 更新路径信息
            self.update_path_info()
            
            # 重新绘制主干网络
            self.draw_backbone_network()
            
            self.log("路径优化完成", "success")
            
        except Exception as e:
            self.log(f"路径优化失败: {str(e)}", "error")
    
    def save_results(self):
        """保存结果"""
        if not self.env:
            self.log("没有可保存的结果", "warning")
            return
        
        # 打开保存对话框
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "保存结果",
            f"simulation_result_{time.strftime('%Y%m%d_%H%M%S')}",
            "PNG图像 (*.png);;JSON数据 (*.json);;所有文件 (*)"
        )
        
        if not file_path:
            return
        
        try:
            # 根据文件类型保存不同格式
            if file_path.endswith('.png'):
                # 保存场景截图
                pixmap = QPixmap(self.graphics_view.viewport().size())
                pixmap.fill(Qt.white)
                
                painter = QPainter(pixmap)
                self.graphics_view.render(painter)
                painter.end()
                
                if pixmap.save(file_path):
                    self.log(f"截图已保存到: {file_path}", "success")
                else:
                    self.log("保存截图失败", "error")
                
            elif file_path.endswith('.json'):
                # 保存模拟数据
                data = {
                    'time': self.env.current_time if hasattr(self.env, 'current_time') else 0,
                    'vehicles': {},
                    'tasks': {}
                }
                
                # 添加车辆数据
                for vehicle_id, vehicle in self.env.vehicles.items():
                    data['vehicles'][vehicle_id] = {
                        'position': vehicle.get('position'),
                        'status': vehicle.get('status'),
                        'load': vehicle.get('load'),
                        'completed_cycles': vehicle.get('completed_cycles', 0)
                    }
                
                # 添加任务数据
                if self.vehicle_scheduler:
                    for task_id, task in self.vehicle_scheduler.tasks.items():
                        data['tasks'][task_id] = task.to_dict()
                
                # 保存为JSON
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                self.log(f"模拟数据已保存到: {file_path}", "success")
            
            else:
                self.log("未知的文件格式", "error")
        
        except Exception as e:
            self.log(f"保存结果失败: {str(e)}", "error")
    
    def show_about(self):
        """显示关于对话框"""
        about_text = """
        <div style="text-align:center;">
            <h2>露天矿多车协同调度系统</h2>
            <p>基于主干路径的多车辆协同规划与调度</p>
            <p>版本: 1.0</p>
            <hr>
            <p>采用主干路径网络优化露天矿车辆调度效率</p>
            <p>支持多车辆协同、冲突避免和交通管理</p>
            <hr>
            <p>开发者: XXX</p>
            <p>Copyright © 2023</p>
        </div>
        """
        
        QMessageBox.about(self, "关于", about_text)