"""
main.py - 露天矿多车协同调度系统 - 极简版GUI
保留核心功能，删除冗余部分，专注于开发和测试
"""

import sys
import os
import math
import time
import json
from typing import Dict, List, Tuple, Optional

from PyQt5.QtCore import (Qt, QTimer, pyqtSignal, QObject, QThread, QRectF, 
                         QPointF, QLineF, QSizeF, QPropertyAnimation, QVariantAnimation, 
                         QSize, QRect, QEasingCurve, pyqtProperty)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QComboBox, QCheckBox, QFileDialog, QSlider,
                            QGroupBox, QTabWidget, QSpinBox, QDoubleSpinBox, QProgressBar,
                            QMessageBox, QTextEdit, QSplitter, QAction, QStatusBar, QToolBar,
                            QMenu, QDockWidget, QGraphicsScene, QGraphicsView, QGraphicsItem,
                            QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsPolygonItem, 
                            QGraphicsPathItem, QGraphicsLineItem, QGraphicsTextItem, QGraphicsItemGroup,
                            QGridLayout, QFrame, QStyleFactory, QScrollArea, QTableWidget, 
                            QTableWidgetItem, QListWidget, QListWidgetItem, QTreeWidget, 
                            QTreeWidgetItem, QHeaderView, QLCDNumber, QDial,QPlainTextEdit,QGraphicsObject)
from PyQt5.QtGui import (QIcon, QFont, QPixmap, QPen, QBrush, QColor, QPainter, QPainterPath,
                        QTransform, QPolygonF, QLinearGradient, QRadialGradient, QPalette,
                        QFontMetrics, QConicalGradient, QCursor)
from PyQt5.QtChart import QChart, QChartView, QLineSeries, QPieSeries, QValueAxis, QBarSeries, QBarSet

# 系统组件导入
from environment import OpenPitMineEnv
from backbone_network import SimplifiedBackbonePathNetwork
from traffic_manager import OptimizedTrafficManager
from vehicle_scheduler import SimplifiedVehicleScheduler, SimplifiedECBSVehicleScheduler

# 规划器导入
try:
    from hybridastar import HybridAStarPlanner
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False

try:
    from RRT import OptimizedRRTPlanner
except ImportError:
    from path_planner import SimplifiedPathPlanner as OptimizedRRTPlanner

class SimpleVehicleItem(QGraphicsEllipseItem):
    """简单车辆显示项"""
    def __init__(self, vehicle_id, vehicle_data):
        super().__init__(-3, -3, 6, 6)
        self.vehicle_id = vehicle_id
        self.vehicle_data = vehicle_data
        
        # 根据状态设置颜色
        status = vehicle_data.get('status', 'idle')
        colors = {
            'idle': QColor(128, 128, 128),
            'moving': QColor(0, 150, 255),
            'loading': QColor(0, 255, 0),
            'unloading': QColor(255, 100, 100)
        }
        
        color = colors.get(status, colors['idle'])
        self.setBrush(QBrush(color))
        self.setPen(QPen(Qt.black, 1))
        
        # 设置位置
        pos = vehicle_data.get('position', (0, 0, 0))
        self.setPos(pos[0], pos[1])
        
        # 设置提示信息
        load = vehicle_data.get('load', 0)
        self.setToolTip(f"车辆 {vehicle_id}\n状态: {status}\n载重: {load}")

class SimpleMineScene(QGraphicsScene):
    """简化的矿场显示场景"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.env = None
        self.backbone_network = None
        
        # 显示项缓存
        self.vehicle_items = {}
        self.path_items = {}
        
    def set_environment(self, env):
        """设置环境"""
        self.env = env
        if env:
            self.setSceneRect(0, 0, env.width, env.height)
        
        self.clear()
        self.vehicle_items.clear()
        self.path_items.clear()
        
        if env:
            self.draw_environment()
    
    def set_backbone_network(self, backbone_network):
        """设置骨干网络"""
        self.backbone_network = backbone_network
        self.update_backbone_display()
    
    def draw_environment(self):
        """绘制环境"""
        if not self.env:
            return
        
        # 背景
        bg = QGraphicsRectItem(0, 0, self.env.width, self.env.height)
        bg.setBrush(QBrush(QColor(240, 240, 240)))
        bg.setPen(QPen(Qt.NoPen))
        bg.setZValue(-100)
        self.addItem(bg)
        
        # 障碍物
        for x, y in self.env.obstacle_points:
            obstacle = QGraphicsRectItem(x, y, 1, 1)
            obstacle.setBrush(QBrush(QColor(80, 80, 80)))
            obstacle.setPen(QPen(Qt.NoPen))
            self.addItem(obstacle)
        
        # 装载点
        for i, point in enumerate(self.env.loading_points):
            item = QGraphicsEllipseItem(point[0]-8, point[1]-8, 16, 16)
            item.setBrush(QBrush(QColor(0, 200, 0, 150)))
            item.setPen(QPen(QColor(0, 150, 0), 2))
            item.setToolTip(f"装载点 {i+1}")
            self.addItem(item)
        
        # 卸载点
        for i, point in enumerate(self.env.unloading_points):
            item = QGraphicsRectItem(point[0]-8, point[1]-8, 16, 16)
            item.setBrush(QBrush(QColor(200, 0, 0, 150)))
            item.setPen(QPen(QColor(150, 0, 0), 2))
            item.setToolTip(f"卸载点 {i+1}")
            self.addItem(item)
        
        # 车辆
        self.update_vehicles()
    
    def update_backbone_display(self):
        """更新骨干网络显示"""
        if not self.backbone_network:
            return
        
        # 清除旧的骨干路径
        for item in self.items():
            if hasattr(item, 'is_backbone_path'):
                self.removeItem(item)
        
        # 绘制骨干路径
        for path_id, path_data in self.backbone_network.backbone_paths.items():
            path = path_data.get('path', [])
            if len(path) < 2:
                continue
            
            # 创建路径多边形
            polygon = QPolygonF()
            for point in path:
                polygon.append(QPointF(point[0], point[1]))
            
            path_item = QGraphicsPathItem()
            painter_path = path_item.path()
            painter_path.addPolygon(polygon)
            path_item.setPath(painter_path)
            
            # 设置样式
            quality = path_data.get('quality', 0.5)
            if quality >= 0.8:
                color = QColor(0, 255, 0, 100)
            elif quality >= 0.6:
                color = QColor(255, 255, 0, 100)
            else:
                color = QColor(255, 0, 0, 100)
            
            path_item.setPen(QPen(color, 2))
            path_item.is_backbone_path = True
            path_item.setZValue(1)
            self.addItem(path_item)
    
    def update_vehicles(self):
        """更新车辆显示"""
        if not self.env:
            return
        
        # 移除不存在的车辆
        for vehicle_id in list(self.vehicle_items.keys()):
            if vehicle_id not in self.env.vehicles:
                self.removeItem(self.vehicle_items[vehicle_id])
                del self.vehicle_items[vehicle_id]
        
        # 更新或添加车辆
        for vehicle_id, vehicle_data in self.env.vehicles.items():
            if vehicle_id in self.vehicle_items:
                # 更新现有车辆
                item = self.vehicle_items[vehicle_id]
                pos = vehicle_data.get('position', (0, 0, 0))
                item.setPos(pos[0], pos[1])
                
                # 更新颜色
                status = vehicle_data.get('status', 'idle')
                colors = {
                    'idle': QColor(128, 128, 128),
                    'moving': QColor(0, 150, 255),
                    'loading': QColor(0, 255, 0),
                    'unloading': QColor(255, 100, 100)
                }
                item.setBrush(QBrush(colors.get(status, colors['idle'])))
            else:
                # 添加新车辆
                item = SimpleVehicleItem(vehicle_id, vehicle_data)
                self.addItem(item)
                self.vehicle_items[vehicle_id] = item
        
        # 更新路径显示
        self.update_paths()
    
    def update_paths(self):
        """更新路径显示"""
        # 清除旧路径
        for item in self.path_items.values():
            self.removeItem(item)
        self.path_items.clear()
        
        # 绘制车辆路径
        for vehicle_id, vehicle_data in self.env.vehicles.items():
            path = vehicle_data.get('path')
            if path and len(path) > 1:
                polygon = QPolygonF()
                for point in path:
                    polygon.append(point[0], point[1])
                
                path_item = QGraphicsPathItem()
                painter_path = path_item.path()
                painter_path.addPolygon(polygon)
                path_item.setPath(painter_path)
                
                path_item.setPen(QPen(QColor(0, 100, 255, 150), 1))
                path_item.setZValue(2)
                self.addItem(path_item)
                self.path_items[vehicle_id] = path_item

class SimpleMineView(QGraphicsView):
    """简化的矿场视图"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        
        # 创建场景
        self.mine_scene = SimpleMineScene(self)
        self.setScene(self.mine_scene)
        
        # 状态显示
        self.coord_label = QLabel("坐标: (0, 0)", self)
        self.coord_label.setStyleSheet("background: rgba(255,255,255,200); padding: 5px;")
        self.coord_label.move(10, 10)
    
    def mouseMoveEvent(self, event):
        """鼠标移动显示坐标"""
        super().mouseMoveEvent(event)
        scene_pos = self.mapToScene(event.pos())
        self.coord_label.setText(f"坐标: ({scene_pos.x():.1f}, {scene_pos.y():.1f})")
    
    def wheelEvent(self, event):
        """滚轮缩放"""
        factor = 1.2 if event.angleDelta().y() > 0 else 1/1.2
        self.scale(factor, factor)

class ControlPanel(QWidget):
    """控制面板"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # 文件操作
        file_group = QGroupBox("文件操作")
        file_layout = QVBoxLayout()
        
        self.file_label = QLabel("未选择文件")
        file_layout.addWidget(self.file_label)
        
        file_btn_layout = QHBoxLayout()
        self.browse_btn = QPushButton("浏览")
        self.load_btn = QPushButton("加载")
        file_btn_layout.addWidget(self.browse_btn)
        file_btn_layout.addWidget(self.load_btn)
        file_layout.addLayout(file_btn_layout)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # 规划器选择
        planner_group = QGroupBox("路径规划器")
        planner_layout = QVBoxLayout()
        
        self.planner_combo = QComboBox()
        if HYBRID_AVAILABLE:
            self.planner_combo.addItems(["混合A*", "RRT"])
        else:
            self.planner_combo.addItems(["RRT"])
        planner_layout.addWidget(self.planner_combo)
        
        self.planner_info = QLabel("混合A*: 确定性，高质量")
        self.planner_info.setStyleSheet("color: green; font-size: 10px;")
        planner_layout.addWidget(self.planner_info)
        
        planner_group.setLayout(planner_layout)
        layout.addWidget(planner_group)
        
        # 骨干网络
        backbone_group = QGroupBox("骨干网络")
        backbone_layout = QVBoxLayout()
        
        self.generate_btn = QPushButton("生成骨干网络")
        backbone_layout.addWidget(self.generate_btn)
        
        self.backbone_stats = QLabel("路径: 0, 接口: 0")
        backbone_layout.addWidget(self.backbone_stats)
        
        backbone_group.setLayout(backbone_layout)
        layout.addWidget(backbone_group)
        
        # 调度控制
        schedule_group = QGroupBox("车辆调度")
        schedule_layout = QVBoxLayout()
        
        # 调度器类型
        self.scheduler_combo = QComboBox()
        self.scheduler_combo.addItems(["标准调度", "ECBS调度"])
        schedule_layout.addWidget(self.scheduler_combo)
        
        # 任务分配
        self.assign_all_btn = QPushButton("分配所有任务")
        self.cancel_all_btn = QPushButton("取消所有任务")
        schedule_layout.addWidget(self.assign_all_btn)
        schedule_layout.addWidget(self.cancel_all_btn)
        
        schedule_group.setLayout(schedule_layout)
        layout.addWidget(schedule_group)
        
        # 仿真控制
        sim_group = QGroupBox("仿真控制")
        sim_layout = QVBoxLayout()
        
        sim_btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始")
        self.pause_btn = QPushButton("暂停")
        self.reset_btn = QPushButton("重置")
        
        sim_btn_layout.addWidget(self.start_btn)
        sim_btn_layout.addWidget(self.pause_btn)
        sim_btn_layout.addWidget(self.reset_btn)
        sim_layout.addLayout(sim_btn_layout)
        
        # 仿真速度
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("速度:"))
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.1, 5.0)
        self.speed_spin.setValue(1.0)
        self.speed_spin.setSingleStep(0.1)
        speed_layout.addWidget(self.speed_spin)
        sim_layout.addLayout(speed_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        sim_layout.addWidget(self.progress_bar)
        
        sim_group.setLayout(sim_layout)
        layout.addWidget(sim_group)
        
        # 统计信息
        stats_group = QGroupBox("统计信息")
        stats_layout = QVBoxLayout()
        
        self.vehicle_stats = QLabel("活跃车辆: 0/0")
        self.task_stats = QLabel("完成任务: 0")
        self.time_stats = QLabel("仿真时间: 00:00")
        
        stats_layout.addWidget(self.vehicle_stats)
        stats_layout.addWidget(self.task_stats)
        stats_layout.addWidget(self.time_stats)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # 显示选项
        display_group = QGroupBox("显示选项")
        display_layout = QVBoxLayout()
        
        self.show_backbone_cb = QCheckBox("显示骨干路径")
        self.show_backbone_cb.setChecked(True)
        
        self.show_paths_cb = QCheckBox("显示车辆路径")
        self.show_paths_cb.setChecked(True)
        
        display_layout.addWidget(self.show_backbone_cb)
        display_layout.addWidget(self.show_paths_cb)
        
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
        layout.addStretch()
    
    def set_main_window(self, main_window):
        """设置主窗口引用"""
        self.main_window = main_window
        
        # 连接信号
        self.browse_btn.clicked.connect(main_window.browse_file)
        self.load_btn.clicked.connect(main_window.load_environment)
        self.planner_combo.currentIndexChanged.connect(main_window.change_planner)
        self.generate_btn.clicked.connect(main_window.generate_backbone)
        self.scheduler_combo.currentIndexChanged.connect(main_window.change_scheduler)
        self.assign_all_btn.clicked.connect(main_window.assign_all_tasks)
        self.cancel_all_btn.clicked.connect(main_window.cancel_all_tasks)
        self.start_btn.clicked.connect(main_window.start_simulation)
        self.pause_btn.clicked.connect(main_window.pause_simulation)
        self.reset_btn.clicked.connect(main_window.reset_simulation)
        self.speed_spin.valueChanged.connect(main_window.update_speed)
        
        # 显示选项
        self.show_backbone_cb.stateChanged.connect(
            lambda: main_window.toggle_backbone_display(self.show_backbone_cb.isChecked())
        )
        self.show_paths_cb.stateChanged.connect(
            lambda: main_window.toggle_path_display(self.show_paths_cb.isChecked())
        )
    
    def update_backbone_stats(self, path_count, interface_count):
        """更新骨干网络统计"""
        self.backbone_stats.setText(f"路径: {path_count}, 接口: {interface_count}")
    
    def update_vehicle_stats(self, active, total):
        """更新车辆统计"""
        self.vehicle_stats.setText(f"活跃车辆: {active}/{total}")
    
    def update_task_stats(self, completed):
        """更新任务统计"""
        self.task_stats.setText(f"完成任务: {completed}")
    
    def update_time_stats(self, sim_time):
        """更新时间统计"""
        hours = int(sim_time // 3600)
        minutes = int((sim_time % 3600) // 60)
        seconds = int(sim_time % 60)
        self.time_stats.setText(f"仿真时间: {hours:02d}:{minutes:02d}:{seconds:02d}")

class SimpleMineGUI(QMainWindow):
    """简化版露天矿GUI主窗口"""
    
    def __init__(self):
        super().__init__()
        
        # 系统组件
        self.env = None
        self.backbone_network = None
        self.path_planner = None
        self.vehicle_scheduler = None
        self.traffic_manager = None
        
        # 状态变量
        self.map_file_path = None
        self.is_simulating = False
        self.simulation_time = 0
        self.simulation_speed = 1.0
        
        self.init_ui()
        
        # 定时器
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(100)  # 10 FPS
        
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.simulation_step)
    
    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle("露天矿多车协同调度系统 - 简化版")
        self.setGeometry(100, 100, 1200, 800)
        
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 控制面板
        self.control_panel = ControlPanel()
        self.control_panel.set_main_window(self)
        self.control_panel.setFixedWidth(280)
        main_layout.addWidget(self.control_panel)
        
        # 显示区域
        self.graphics_view = SimpleMineView()
        main_layout.addWidget(self.graphics_view, 1)
        
        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("系统就绪")
        
        # 日志区域（可折叠）
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(120)
        self.log_text.setPlaceholderText("系统日志...")
        
        # 使用分割器
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.graphics_view)
        splitter.addWidget(self.log_text)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        
        # 重新布局
        main_layout.removeWidget(self.graphics_view)
        main_layout.addWidget(splitter, 1)
    
    def log(self, message, level="info"):
        """记录日志"""
        timestamp = time.strftime("%H:%M:%S")
        colors = {
            "info": "black",
            "success": "green", 
            "warning": "orange",
            "error": "red"
        }
        color = colors.get(level, "black")
        
        formatted_msg = f'<span style="color: {color};">[{timestamp}] {message}</span>'
        self.log_text.append(formatted_msg)
        
        # 滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        # 状态栏显示
        self.status_bar.showMessage(message, 3000)
    
    # ===== 文件操作 =====
    
    def browse_file(self):
        """浏览文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择地图文件", "", "JSON文件 (*.json)"
        )
        
        if file_path:
            self.map_file_path = file_path
            filename = os.path.basename(file_path)
            self.control_panel.file_label.setText(filename)
            self.log(f"已选择文件: {filename}")
    
    def load_environment(self):
        """加载环境"""
        if not self.map_file_path:
            QMessageBox.warning(self, "警告", "请先选择地图文件")
            return
        
        try:
            self.log("正在加载环境...")
            
            # 创建环境
            self.env = OpenPitMineEnv()
            if not self.env.load_from_file(self.map_file_path):
                raise Exception("环境加载失败")
            
            # 设置到视图
            self.graphics_view.mine_scene.set_environment(self.env)
            self.graphics_view.fitInView(self.graphics_view.mine_scene.sceneRect(), Qt.KeepAspectRatio)
            
            # 创建系统组件
            self.create_system_components()
            
            self.log("环境加载成功", "success")
            
        except Exception as e:
            self.log(f"加载失败: {str(e)}", "error")
            QMessageBox.critical(self, "错误", f"加载环境失败:\n{str(e)}")
    
    def create_system_components(self):
        """创建系统组件"""
        try:
            # 骨干网络
            self.backbone_network = SimplifiedBackbonePathNetwork(self.env)
            
            # 路径规划器
            self.path_planner = self._create_planner()
            
            # 交通管理器
            self.traffic_manager = OptimizedTrafficManager(self.env, self.backbone_network)
            
            # 车辆调度器
            scheduler_index = self.control_panel.scheduler_combo.currentIndex()
            if scheduler_index == 1:  # ECBS
                self.vehicle_scheduler = SimplifiedECBSVehicleScheduler(
                    self.env, self.path_planner, self.traffic_manager, self.backbone_network
                )
            else:  # 标准
                self.vehicle_scheduler = SimplifiedVehicleScheduler(
                    self.env, self.path_planner, self.backbone_network, self.traffic_manager
                )
            
            # 初始化
            self.vehicle_scheduler.initialize_vehicles()
            
            # 创建默认任务模板
            if self.env.loading_points and self.env.unloading_points:
                self.vehicle_scheduler.create_enhanced_mission_template("default")
            
            self.log("系统组件初始化完成", "success")
            
        except Exception as e:
            self.log(f"组件初始化失败: {str(e)}", "error")
    
    def _create_planner(self):
        """创建路径规划器"""
        planner_index = self.control_panel.planner_combo.currentIndex()
        
        if planner_index == 0 and HYBRID_AVAILABLE:
            # 混合A*
            planner = HybridAStarPlanner(
                self.env,
                vehicle_length=6.0,
                vehicle_width=3.0,
                turning_radius=8.0,
                step_size=1.0
            )
            planner.set_backbone_network(self.backbone_network)
            self.log("已创建混合A*规划器", "success")
            
        else:
            # RRT
            planner = OptimizedRRTPlanner(
                self.env,
                vehicle_length=6.0,
                vehicle_width=3.0,
                turning_radius=8.0,
                step_size=0.8
            )
            if hasattr(planner, 'set_backbone_network'):
                planner.set_backbone_network(self.backbone_network)
            self.log("已创建RRT规划器", "success")
        
        return planner
    
    # ===== 规划器控制 =====
    
    def change_planner(self):
        """切换规划器"""
        if not self.env:
            return
        
        planner_index = self.control_panel.planner_combo.currentIndex()
        planner_name = "混合A*" if planner_index == 0 else "RRT"
        
        self.path_planner = self._create_planner()
        
        if self.vehicle_scheduler:
            self.vehicle_scheduler.path_planner = self.path_planner
        
        # 更新信息显示
        if planner_index == 0:
            self.control_panel.planner_info.setText("混合A*: 确定性，高质量")
            self.control_panel.planner_info.setStyleSheet("color: green; font-size: 10px;")
        else:
            self.control_panel.planner_info.setText("RRT: 随机采样，探索性强")
            self.control_panel.planner_info.setStyleSheet("color: blue; font-size: 10px;")
        
        self.log(f"已切换到{planner_name}规划器", "success")
    
    # ===== 骨干网络控制 =====
    
    def generate_backbone(self):
        """生成骨干网络"""
        if not self.backbone_network:
            QMessageBox.warning(self, "警告", "请先加载环境")
            return
        
        try:
            self.log("正在生成骨干网络...")
            
            success = self.backbone_network.generate_backbone_network(
                quality_threshold=0.5,
                interface_spacing=20
            )
            
            if success:
                # 更新显示
                self.graphics_view.mine_scene.set_backbone_network(self.backbone_network)
                
                # 更新统计
                path_count = len(self.backbone_network.backbone_paths)
                interface_count = len(self.backbone_network.backbone_interfaces)
                self.control_panel.update_backbone_stats(path_count, interface_count)
                
                # 更新规划器引用
                if self.path_planner and hasattr(self.path_planner, 'set_backbone_network'):
                    self.path_planner.set_backbone_network(self.backbone_network)
                
                self.log(f"骨干网络生成成功: {path_count}条路径, {interface_count}个接口", "success")
                
            else:
                self.log("骨干网络生成失败", "error")
                
        except Exception as e:
            self.log(f"生成骨干网络失败: {str(e)}", "error")
    
    # ===== 调度控制 =====
    
    def change_scheduler(self):
        """切换调度器"""
        if not self.env:
            return
        
        scheduler_index = self.control_panel.scheduler_combo.currentIndex()
        
        if scheduler_index == 1:  # ECBS
            self.vehicle_scheduler = SimplifiedECBSVehicleScheduler(
                self.env, self.path_planner, self.traffic_manager, self.backbone_network
            )
            self.log("已切换到ECBS调度器", "success")
        else:  # 标准
            self.vehicle_scheduler = SimplifiedVehicleScheduler(
                self.env, self.path_planner, self.backbone_network, self.traffic_manager
            )
            self.log("已切换到标准调度器", "success")
        
        self.vehicle_scheduler.initialize_vehicles()
    
    def assign_all_tasks(self):
        """分配所有任务"""
        if not self.vehicle_scheduler:
            QMessageBox.warning(self, "警告", "请先加载环境")
            return
        
        count = 0
        for vehicle_id in self.env.vehicles.keys():
            if self.vehicle_scheduler.assign_mission_intelligently(vehicle_id):
                count += 1
        
        self.log(f"已为 {count} 个车辆分配任务", "success")
    
    def cancel_all_tasks(self):
        """取消所有任务"""
        if not self.vehicle_scheduler:
            return
        
        # 清理调度器
        for vehicle_id in self.env.vehicles.keys():
            if vehicle_id in self.vehicle_scheduler.active_assignments:
                self.vehicle_scheduler.active_assignments[vehicle_id] = []
            
            if vehicle_id in self.vehicle_scheduler.vehicle_states:
                vehicle_state = self.vehicle_scheduler.vehicle_states[vehicle_id]
                vehicle_state.status = vehicle_state.status.__class__.IDLE
                vehicle_state.current_task = None
            
            # 重置环境中的车辆状态
            if vehicle_id in self.env.vehicles:
                self.env.vehicles[vehicle_id]['status'] = 'idle'
                self.env.vehicles[vehicle_id]['path'] = None
        
        # 释放交通管理器路径
        if self.traffic_manager:
            self.traffic_manager.clear_all_paths()
        
        self.log("已取消所有任务", "success")
    
    # ===== 仿真控制 =====
    
    def start_simulation(self):
        """开始仿真"""
        if not self.env:
            QMessageBox.warning(self, "警告", "请先加载环境")
            return
        
        self.is_simulating = True
        self.control_panel.start_btn.setEnabled(False)
        self.control_panel.pause_btn.setEnabled(True)
        
        interval = max(50, int(100 / self.simulation_speed))
        self.sim_timer.start(interval)
        
        self.log("仿真已开始")
    
    def pause_simulation(self):
        """暂停仿真"""
        self.is_simulating = False
        self.control_panel.start_btn.setEnabled(True)
        self.control_panel.pause_btn.setEnabled(False)
        
        self.sim_timer.stop()
        
        self.log("仿真已暂停")
    
    def reset_simulation(self):
        """重置仿真"""
        if self.is_simulating:
            self.pause_simulation()
        
        self.simulation_time = 0
        
        if self.env:
            self.env.reset()
        
        if self.vehicle_scheduler:
            self.vehicle_scheduler.initialize_vehicles()
        
        self.control_panel.progress_bar.setValue(0)
        self.log("仿真已重置")
    
    def update_speed(self, value):
        """更新仿真速度"""
        self.simulation_speed = value
        
        if self.is_simulating:
            interval = max(50, int(100 / self.simulation_speed))
            self.sim_timer.start(interval)
    
    def simulation_step(self):
        """仿真步骤"""
        if not self.is_simulating:
            return
        
        time_step = 0.5 * self.simulation_speed
        self.simulation_time += time_step
        
        # 更新调度器
        if self.vehicle_scheduler:
            self.vehicle_scheduler.update(time_step)
        
        # 更新进度条（假设1小时为100%）
        progress = min(100, int(self.simulation_time * 100 / 3600))
        self.control_panel.progress_bar.setValue(progress)
        
        if progress >= 100:
            self.pause_simulation()
            self.log("仿真完成", "success")
    
    # ===== 显示控制 =====
    
    def toggle_backbone_display(self, show):
        """切换骨干路径显示"""
        if show:
            self.graphics_view.mine_scene.update_backbone_display()
        else:
            # 隐藏骨干路径
            for item in self.graphics_view.mine_scene.items():
                if hasattr(item, 'is_backbone_path'):
                    item.setVisible(False)
    
    def toggle_path_display(self, show):
        """切换路径显示"""
        for item in self.graphics_view.mine_scene.path_items.values():
            item.setVisible(show)
    
    def update_display(self):
        """更新显示"""
        if not self.env:
            return
        
        # 更新车辆显示
        self.graphics_view.mine_scene.update_vehicles()
        
        # 更新统计信息
        if self.vehicle_scheduler:
            stats = self.vehicle_scheduler.get_comprehensive_stats()
            real_time = stats.get('real_time', {})
            
            active = real_time.get('active_vehicles', 0)
            total = len(self.env.vehicles)
            completed = stats.get('completed_tasks', 0)
            
            self.control_panel.update_vehicle_stats(active, total)
            self.control_panel.update_task_stats(completed)
            self.control_panel.update_time_stats(self.simulation_time)

def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用属性
    app.setApplicationName("露天矿调度系统")
    app.setApplicationVersion("1.0")
    
    # 创建主窗口
    window = SimpleMineGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()