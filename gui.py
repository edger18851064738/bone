import sys
import os
import math
import time
import json
from collections import defaultdict
from PyQt5.QtCore import (Qt, QTimer, pyqtSignal, QObject, QThread, QRectF, 
                         QPointF, QLineF, QSizeF, QPropertyAnimation, QVariantAnimation, QSize)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QComboBox, QCheckBox, QFileDialog, QSlider,
                            QGroupBox, QTabWidget, QSpinBox, QDoubleSpinBox, QProgressBar,
                            QMessageBox, QTextEdit, QSplitter, QAction, QStatusBar, QToolBar,
                            QMenu, QDockWidget, QGraphicsScene, QGraphicsView, QGraphicsItem,
                            QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsPolygonItem, 
                            QGraphicsPathItem, QGraphicsLineItem, QGraphicsTextItem, QGraphicsItemGroup,
                            QGridLayout, QFrame, QStyleFactory, QScrollArea, QTableWidget, QTableWidgetItem,
                            QListWidget, QListWidgetItem)
from PyQt5.QtGui import (QIcon, QFont, QPixmap, QPen, QBrush, QColor, QPainter, QPainterPath,
                        QTransform, QPolygonF, QLinearGradient, QRadialGradient, QPalette)

# 导入系统组件
from backbone_network import SimplifiedBackbonePathNetwork
from path_planner import SimplifiedPathPlanner
from traffic_manager import OptimizedTrafficManager
from vehicle_scheduler import SimplifiedVehicleScheduler, SimplifiedECBSVehicleScheduler, VehicleTask
from environment import OpenPitMineEnv

# 全局样式
GLOBAL_STYLE = """
QMainWindow { background-color: #f8f9fa; }
QTabWidget::pane { border: 1px solid #d3d3d3; background-color: white; border-radius: 4px; }
QTabBar::tab { background-color: #f1f1f1; border: 1px solid #d3d3d3; border-bottom: none; 
               border-top-left-radius: 4px; border-top-right-radius: 4px; padding: 8px 16px; 
               font-weight: bold; min-width: 100px; }
QTabBar::tab:selected { background-color: white; border-bottom: 1px solid white; }
QGroupBox { font-weight: bold; border: 1px solid #d3d3d3; border-radius: 4px; margin-top: 12px; 
            padding-top: 15px; background-color: #ffffff; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 5px 8px; 
                   background-color: #f8f9fa; border-radius: 2px; }
QPushButton { background-color: #007bff; color: white; border: none; padding: 8px 16px; 
              border-radius: 4px; font-weight: bold; }
QPushButton:hover { background-color: #0069d9; }
QPushButton:pressed { background-color: #005cbf; }
QPushButton:disabled { background-color: #6c757d; color: #f8f9fa; }
QComboBox { border: 1px solid #d3d3d3; border-radius: 4px; padding: 5px; min-width: 120px; 
            background-color: white; }
QSpinBox, QDoubleSpinBox { border: 1px solid #d3d3d3; border-radius: 4px; padding: 5px; min-width: 80px; }
QTextEdit { border: 1px solid #d3d3d3; border-radius: 4px; background-color: white; }
QProgressBar { border: 1px solid #d3d3d3; border-radius: 4px; background-color: #f8f9fa; 
               text-align: center; font-weight: bold; color: black; height: 18px; }
QProgressBar::chunk { background-color: #28a745; border-radius: 3px; }
"""

# 车辆状态颜色
VEHICLE_COLORS = {
    'idle': QColor(128, 128, 128),
    'loading': QColor(40, 167, 69),
    'unloading': QColor(220, 53, 69),
    'moving': QColor(0, 123, 255)
}

class EnhancedVehicleGraphicsItem(QGraphicsItemGroup):
    """增强的车辆图形项"""
    def __init__(self, vehicle_id, vehicle_data, parent=None):
        super().__init__(parent)
        self.vehicle_id = vehicle_id
        self.vehicle_data = vehicle_data
        self.position = vehicle_data['position']
        
        # 创建车辆组件
        self.vehicle_body = QGraphicsPolygonItem(self)
        self.vehicle_label = QGraphicsTextItem(str(vehicle_id), self)
        self.status_indicator = QGraphicsEllipseItem(self)
        
        self.setZValue(10)
        self.update_appearance()
        self.update_position()
    
    def update_appearance(self):
        """更新车辆外观"""
        status = self.vehicle_data.get('status', 'idle')
        color = VEHICLE_COLORS.get(status, VEHICLE_COLORS['idle'])
        
        # 车辆主体
        self.vehicle_body.setBrush(QBrush(color))
        self.vehicle_body.setPen(QPen(Qt.black, 0.5))
        
        # 标签
        self.vehicle_label.setDefaultTextColor(Qt.white)
        self.vehicle_label.setFont(QFont("Arial", 3, QFont.Bold))
        
        # 状态指示器
        indicator_color = color.lighter(150)
        self.status_indicator.setBrush(QBrush(indicator_color))
        self.status_indicator.setPen(QPen(Qt.black, 0.3))
    
    def update_position(self):
        """更新车辆位置"""
        if not self.position or len(self.position) < 3:
            return
        
        x, y, theta = self.position
        
        # 创建车辆多边形（简化的矩形）
        length, width = 6.0, 3.0
        half_length, half_width = length/2, width/2
        
        corners = [
            QPointF(half_length, half_width),
            QPointF(half_length, -half_width),
            QPointF(-half_length, -half_width),
            QPointF(-half_length, half_width)
        ]
        
        # 应用旋转
        transform = QTransform()
        transform.rotate(theta * 180 / math.pi)
        
        polygon = QPolygonF()
        for corner in corners:
            rotated = transform.map(corner)
            polygon.append(QPointF(x + rotated.x(), y + rotated.y()))
        
        self.vehicle_body.setPolygon(polygon)
        
        # 更新标签和指示器位置
        self.vehicle_label.setPos(x - 1, y - 1)
        self.status_indicator.setRect(x - 1, y - width/2 - 2, 2, 1)
    
    def update_data(self, vehicle_data):
        """更新车辆数据"""
        self.vehicle_data = vehicle_data
        self.position = vehicle_data['position']
        self.update_appearance()
        self.update_position()

class PathGraphicsItem(QGraphicsItemGroup):
    """路径图形项"""
    def __init__(self, path, path_structure=None, parent=None):
        super().__init__(parent)
        self.path_data = path
        self.path_structure = path_structure
        self.setZValue(5)
        
        if path_structure and path_structure.get('type') == 'backbone_assisted':
            self.create_backbone_assisted_path()
        else:
            self.create_direct_path()
    
    def create_backbone_assisted_path(self):
        """创建骨干辅助路径"""
        if not self.path_data:
            return
        
        # 使用不同颜色区分路径段
        painter_path = QPainterPath()
        painter_path.moveTo(self.path_data[0][0], self.path_data[0][1])
        
        for point in self.path_data[1:]:
            painter_path.lineTo(point[0], point[1])
        
        path_item = QGraphicsPathItem(painter_path)
        
        # 骨干辅助路径使用绿色实线
        pen = QPen(QColor(50, 180, 50, 200), 1.2)
        path_item.setPen(pen)
        
        self.addToGroup(path_item)
    
    def create_direct_path(self):
        """创建直接路径"""
        if not self.path_data:
            return
        
        painter_path = QPainterPath()
        painter_path.moveTo(self.path_data[0][0], self.path_data[0][1])
        
        for point in self.path_data[1:]:
            painter_path.lineTo(point[0], point[1])
        
        path_item = QGraphicsPathItem(painter_path)
        
        # 直接路径使用蓝色虚线
        pen = QPen(QColor(0, 123, 255, 180), 1.0)
        pen.setStyle(Qt.DashLine)
        pen.setDashPattern([5, 3])
        path_item.setPen(pen)
        
        self.addToGroup(path_item)

class BackbonePathVisualizer(QGraphicsItemGroup):
    """骨干路径可视化器"""
    def __init__(self, backbone_network, parent=None):
        super().__init__(parent)
        self.backbone_network = backbone_network
        self.setZValue(1)
        self.update_visualization()
    
    def update_visualization(self):
        """更新骨干路径可视化"""
        # 清除现有项
        for item in self.childItems():
            self.removeFromGroup(item)
        
        if not self.backbone_network or not self.backbone_network.paths:
            return
        
        # 绘制骨干路径
        for path_id, path_data in self.backbone_network.paths.items():
            path = path_data.get('path', [])
            if len(path) < 2:
                continue
            
            painter_path = QPainterPath()
            painter_path.moveTo(path[0][0], path[0][1])
            
            for point in path[1:]:
                painter_path.lineTo(point[0], point[1])
            
            path_item = QGraphicsPathItem(painter_path)
            
            # 根据路径质量设置颜色
            quality = path_data.get('quality', 0.5)
            if quality >= 0.8:
                color = QColor(40, 180, 40, 150)
            elif quality >= 0.6:
                color = QColor(180, 180, 40, 150)
            else:
                color = QColor(180, 40, 40, 150)
            
            pen = QPen(color, 1.5)
            path_item.setPen(pen)
            
            self.addToGroup(path_item)
        
        # 绘制特殊点
        self._draw_special_points()
    
    def _draw_special_points(self):
        """绘制特殊点"""
        if not self.backbone_network or not hasattr(self.backbone_network, 'special_points'):
            return
        
        special_points = self.backbone_network.special_points
        
        # 装载点（绿色圆形）
        for point in special_points.get('loading', []):
            pos = point['position']
            item = QGraphicsEllipseItem(pos[0]-3, pos[1]-3, 6, 6)
            item.setBrush(QBrush(QColor(0, 200, 0)))
            item.setPen(QPen(Qt.black, 0.5))
            self.addToGroup(item)
        
        # 卸载点（红色方形）
        for point in special_points.get('unloading', []):
            pos = point['position']
            item = QGraphicsRectItem(pos[0]-3, pos[1]-3, 6, 6)
            item.setBrush(QBrush(QColor(200, 0, 0)))
            item.setPen(QPen(Qt.black, 0.5))
            self.addToGroup(item)
        
        # 停车点（蓝色菱形）
        for point in special_points.get('parking', []):
            pos = point['position']
            diamond = QPolygonF([
                QPointF(pos[0], pos[1]-3),
                QPointF(pos[0]+3, pos[1]),
                QPointF(pos[0], pos[1]+3),
                QPointF(pos[0]-3, pos[1])
            ])
            item = QGraphicsPolygonItem(diamond)
            item.setBrush(QBrush(QColor(0, 0, 200)))
            item.setPen(QPen(Qt.black, 0.5))
            self.addToGroup(item)

class MineGraphicsScene(QGraphicsScene):
    """矿场图形场景"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.env = None
        self.backbone_network = None
        self.show_trajectories = True
        self.show_backbone = True
        self.show_interfaces = True  # 新增接口显示开关
        
        # 图形项容器
        self.obstacle_items = []
        self.vehicle_items = {}
        self.path_items = {}
        self.backbone_visualizer = None
        self.interface_items = {}  # 接口图形项
        
        self.setSceneRect(0, 0, 500, 500)
    
    def set_environment(self, env):
        """设置环境"""
        self.env = env
        if env:
            self.setSceneRect(0, 0, env.width, env.height)
        
        self.clear()
        self._reset_items()
        
        if env:
            self.draw_environment()
    
    def _reset_items(self):
        """重置图形项"""
        self.obstacle_items = []
        self.vehicle_items = {}
        self.path_items = {}
        self.backbone_visualizer = None
    
    def draw_environment(self):
        """绘制环境"""
        if not self.env:
            return
        
        # 背景
        background = QGraphicsRectItem(0, 0, self.env.width, self.env.height)
        background.setBrush(QBrush(QColor(245, 245, 250)))
        background.setPen(QPen(Qt.NoPen))
        background.setZValue(-100)
        self.addItem(background)
        
        # 网格线
        self._draw_grid()
        
        # 障碍物
        self._draw_obstacles()
        
        # 特殊点
        self._draw_loading_points()
        self._draw_unloading_points()
        
        # 车辆
        self.draw_vehicles()
    
    def _draw_grid(self):
        """绘制网格"""
        grid_size = 20
        pen = QPen(QColor(230, 230, 240))
        
        # 垂直线
        for x in range(0, self.env.width + 1, grid_size):
            line = QGraphicsLineItem(x, 0, x, self.env.height)
            line.setPen(pen)
            line.setZValue(-90)
            self.addItem(line)
        
        # 水平线
        for y in range(0, self.env.height + 1, grid_size):
            line = QGraphicsLineItem(0, y, self.env.width, y)
            line.setPen(pen)
            line.setZValue(-90)
            self.addItem(line)
    
    def _draw_obstacles(self):
        """绘制障碍物"""
        for x, y in self.env.obstacle_points:
            rect = QGraphicsRectItem(x, y, 1, 1)
            rect.setBrush(QBrush(QColor(80, 80, 90)))
            rect.setPen(QPen(QColor(60, 60, 70), 0.2))
            rect.setZValue(-50)
            self.addItem(rect)
            self.obstacle_items.append(rect)
    
    def _draw_loading_points(self):
        """绘制装载点"""
        for i, point in enumerate(self.env.loading_points):
            x, y = point[0], point[1]
            
            # 装载区域
            area = QGraphicsEllipseItem(x-8, y-8, 16, 16)
            area.setBrush(QBrush(QColor(200, 255, 200, 120)))
            area.setPen(QPen(QColor(0, 150, 0), 1))
            area.setZValue(-20)
            self.addItem(area)
            
            # 中心点
            center = QGraphicsEllipseItem(x-2, y-2, 4, 4)
            center.setBrush(QBrush(QColor(0, 200, 0)))
            center.setPen(QPen(Qt.black, 0.5))
            center.setZValue(-10)
            self.addItem(center)
            
            # 标签
            text = QGraphicsTextItem(f"L{i+1}")
            text.setPos(x-5, y-12)
            text.setDefaultTextColor(QColor(0, 100, 0))
            text.setFont(QFont("Arial", 3, QFont.Bold))
            text.setZValue(-10)
            self.addItem(text)
    
    def _draw_unloading_points(self):
        """绘制卸载点"""
        for i, point in enumerate(self.env.unloading_points):
            x, y = point[0], point[1]
            
            # 卸载区域
            area = QGraphicsEllipseItem(x-8, y-8, 16, 16)
            area.setBrush(QBrush(QColor(255, 200, 200, 120)))
            area.setPen(QPen(QColor(150, 0, 0), 1))
            area.setZValue(-20)
            self.addItem(area)
            
            # 中心点
            center = QGraphicsRectItem(x-2, y-2, 4, 4)
            center.setBrush(QBrush(QColor(200, 0, 0)))
            center.setPen(QPen(Qt.black, 0.5))
            center.setZValue(-10)
            self.addItem(center)
            
            # 标签
            text = QGraphicsTextItem(f"U{i+1}")
            text.setPos(x-5, y-12)
            text.setDefaultTextColor(QColor(150, 0, 0))
            text.setFont(QFont("Arial", 3, QFont.Bold))
            text.setZValue(-10)
            self.addItem(text)
    
    def draw_vehicles(self):
        """绘制车辆"""
        if not self.env:
            return
        
        # 清除现有车辆
        for item in self.vehicle_items.values():
            self.removeItem(item)
        self.vehicle_items.clear()
        
        for item in self.path_items.values():
            self.removeItem(item)
        self.path_items.clear()
        
        # 添加车辆
        for vehicle_id, vehicle_data in self.env.vehicles.items():
            vehicle_item = EnhancedVehicleGraphicsItem(vehicle_id, vehicle_data)
            self.addItem(vehicle_item)
            self.vehicle_items[vehicle_id] = vehicle_item
            
            # 绘制路径
            if (self.show_trajectories and 'path' in vehicle_data and 
                vehicle_data['path']):
                path_structure = vehicle_data.get('path_structure')
                path_item = PathGraphicsItem(
                    vehicle_data['path'], 
                    path_structure
                )
                self.addItem(path_item)
                self.path_items[vehicle_id] = path_item
    
    def update_vehicles(self):
        """更新车辆显示"""
        if not self.env:
            return
        
        for vehicle_id, vehicle_data in self.env.vehicles.items():
            if vehicle_id in self.vehicle_items:
                self.vehicle_items[vehicle_id].update_data(vehicle_data)
            else:
                # 添加新车辆
                vehicle_item = EnhancedVehicleGraphicsItem(vehicle_id, vehicle_data)
                self.addItem(vehicle_item)
                self.vehicle_items[vehicle_id] = vehicle_item
            
            # 更新路径
            if (self.show_trajectories and 'path' in vehicle_data and 
                vehicle_data['path']):
                
                if vehicle_id in self.path_items:
                    self.removeItem(self.path_items[vehicle_id])
                
                path_structure = vehicle_data.get('path_structure')
                path_item = PathGraphicsItem(
                    vehicle_data['path'],
                    path_structure
                )
                self.addItem(path_item)
                self.path_items[vehicle_id] = path_item
    
    def set_backbone_network(self, backbone_network):
        """设置骨干路径网络并显示接口"""
        self.backbone_network = backbone_network
        
        if self.backbone_visualizer:
            self.removeItem(self.backbone_visualizer)
        
        if backbone_network and self.show_backbone:
            self.backbone_visualizer = BackbonePathVisualizer(backbone_network)
            self.addItem(self.backbone_visualizer)
        
        # 显示接口点
        self.update_interface_display()
    def update_interface_display(self):
        """更新接口显示"""
        # 清除现有接口项
        for item in self.interface_items.values():
            self.removeItem(item)
        self.interface_items.clear()
        
        if (not self.show_interfaces or not self.backbone_network or 
            not hasattr(self.backbone_network, 'backbone_interfaces')):
            return
        
        # 添加接口项
        for interface_id, interface in self.backbone_network.backbone_interfaces.items():
            interface_item = InterfaceGraphicsItem(interface)
            self.addItem(interface_item)
            self.interface_items[interface_id] = interface_item    
    def set_show_interfaces(self, show):
        """设置是否显示接口"""
        self.show_interfaces = show
        self.update_interface_display()

    def set_show_trajectories(self, show):
        """设置是否显示轨迹"""
        self.show_trajectories = show
        self.update_vehicles()
    
    def set_show_backbone(self, show):
        """设置是否显示骨干路径"""
        self.show_backbone = show
        if self.backbone_visualizer:
            self.backbone_visualizer.setVisible(show)

class MineGraphicsView(QGraphicsView):
    """矿场图形视图"""
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        
        # 创建场景
        self.mine_scene = MineGraphicsScene(self)
        self.setScene(self.mine_scene)
        
        # 坐标显示
        self.coord_label = QLabel(self)
        self.coord_label.setStyleSheet(
            "background-color: rgba(255, 255, 255, 180); "
            "padding: 5px; border-radius: 3px;"
        )
        self.coord_label.setAlignment(Qt.AlignCenter)
        self.coord_label.setFixedSize(120, 25)
        self.coord_label.move(10, 10)
        self.coord_label.show()
    
    def wheelEvent(self, event):
        """鼠标滚轮缩放"""
        factor = 1.2 if event.angleDelta().y() > 0 else 1/1.2
        self.scale(factor, factor)
    
    def mouseMoveEvent(self, event):
        """鼠标移动显示坐标"""
        super().mouseMoveEvent(event)
        scene_pos = self.mapToScene(event.pos())
        x, y = scene_pos.x(), scene_pos.y()
        self.coord_label.setText(f"({x:.1f}, {y:.1f})")
    
    def set_environment(self, env):
        """设置环境"""
        self.mine_scene.set_environment(env)
        if env:
            self.fitInView(self.mine_scene.sceneRect(), Qt.KeepAspectRatio)
    
    def update_vehicles(self):
        """更新车辆"""
        self.mine_scene.update_vehicles()
    
    def set_backbone_network(self, backbone_network):
        """设置骨干路径网络"""
        self.mine_scene.set_backbone_network(backbone_network)
    
    def set_show_trajectories(self, show):
        """设置显示轨迹"""
        self.mine_scene.set_show_trajectories(show)
    
    def set_show_backbone(self, show):
        """设置显示骨干路径"""
        self.mine_scene.set_show_backbone(show)

class VehicleInfoPanel(QWidget):
    """车辆信息面板"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # 车辆选择
        self.vehicle_combo = QComboBox()
        self.vehicle_combo.currentIndexChanged.connect(self.update_vehicle_info)
        self.layout.addWidget(QLabel("选择车辆:"))
        self.layout.addWidget(self.vehicle_combo)
        
        # 基本信息
        self.info_group = QGroupBox("基本信息")
        info_layout = QGridLayout()
        
        self.info_labels = {}
        labels = ["位置:", "状态:", "载重:", "当前任务:", "完成循环:"]
        
        for i, label in enumerate(labels):
            info_layout.addWidget(QLabel(label), i, 0)
            value_label = QLabel("--")
            value_label.setStyleSheet("background-color: #f8f9fa; padding: 3px; border-radius: 3px;")
            info_layout.addWidget(value_label, i, 1)
            self.info_labels[label] = value_label
        
        self.info_group.setLayout(info_layout)
        self.layout.addWidget(self.info_group)
        
        # 路径信息
        self.path_group = QGroupBox("路径信息")
        path_layout = QGridLayout()
        
        self.path_labels = {}
        path_fields = ["路径类型:", "骨干利用率:", "路径质量:", "路径长度:"]
        
        for i, label in enumerate(path_fields):
            path_layout.addWidget(QLabel(label), i, 0)
            value_label = QLabel("--")
            value_label.setStyleSheet("background-color: #f8f9fa; padding: 3px; border-radius: 3px;")
            path_layout.addWidget(value_label, i, 1)
            self.path_labels[label] = value_label
        
        self.path_group.setLayout(path_layout)
        self.layout.addWidget(self.path_group)
        
        self.layout.addStretch()
    
    def set_environment(self, env, scheduler=None):
        """设置环境和调度器"""
        self.env = env
        self.scheduler = scheduler
        
        self.vehicle_combo.clear()
        if env and env.vehicles:
            for v_id in sorted(env.vehicles.keys()):
                self.vehicle_combo.addItem(f"车辆 {v_id}", v_id)
        
        if self.vehicle_combo.count() > 0:
            self.update_vehicle_info(0)
    
    def update_vehicle_info(self, index=None):
        """更新车辆信息显示 - 包含接口信息"""
        if not hasattr(self, 'env') or not self.env:
            return
        
        if index is None or index < 0 or index >= self.vehicle_combo.count():
            return
        
        v_id = self.vehicle_combo.itemData(index)
        if v_id not in self.env.vehicles:
            return
        
        vehicle = self.env.vehicles[v_id]
        
        # 更新基本信息
        pos = vehicle.get('position', (0, 0, 0))
        self.info_labels["位置:"].setText(f"({pos[0]:.1f}, {pos[1]:.1f}, {math.degrees(pos[2]):.1f}°)")
        
        status_map = {
            'idle': '空闲', 'moving': '移动中',
            'loading': '装载中', 'unloading': '卸载中'
        }
        status = vehicle.get('status', 'idle')
        self.info_labels["状态:"].setText(status_map.get(status, status))
        
        load = vehicle.get('load', 0)
        max_load = vehicle.get('max_load', 100)
        self.info_labels["载重:"].setText(f"{load}/{max_load}")
        
        # 当前任务
        if self.scheduler and hasattr(self.scheduler, 'vehicle_statuses'):
            vehicle_status = self.scheduler.vehicle_statuses.get(v_id, {})
            current_task = vehicle_status.get('current_task', '无')
            self.info_labels["当前任务:"].setText(str(current_task))
        else:
            self.info_labels["当前任务:"].setText("--")
        
        cycles = vehicle.get('completed_cycles', 0)
        self.info_labels["完成循环:"].setText(str(cycles))
        
        # 路径信息
        path_structure = vehicle.get('path_structure', {})
        
        path_type = path_structure.get('type', 'unknown')
        type_map = {
            'interface_assisted': '接口辅助',
            'backbone_only': '骨干直接',
            'direct': '直接路径',
            'conflict_resolved': '冲突解决'
        }
        self.path_labels["路径类型:"].setText(type_map.get(path_type, path_type))
        
        backbone_util = path_structure.get('backbone_utilization', 0)
        self.path_labels["骨干利用率:"].setText(f"{backbone_util:.1%}")
        
        # 动态添加接口信息标签（如果还没有的话）
        if "使用接口:" not in self.path_labels:
            # 创建新的标签
            interface_label = QLabel("--")
            interface_label.setStyleSheet("background-color: #f8f9fa; padding: 3px; border-radius: 3px;")
            self.path_labels["使用接口:"] = interface_label
            
            # 添加到布局
            layout = self.path_group.layout()
            row_count = layout.rowCount()
            layout.addWidget(QLabel("使用接口:"), row_count, 0)
            layout.addWidget(interface_label, row_count, 1)
        
        # 显示接口信息
        interface_id = path_structure.get('interface_id', 'N/A')
        if interface_id and interface_id != 'N/A':
            # 如果接口ID太长，只显示后面部分
            if len(str(interface_id)) > 15:
                display_interface = str(interface_id).split('_')[-1]
            else:
                display_interface = str(interface_id)
        else:
            display_interface = "N/A"
        
        self.path_labels["使用接口:"].setText(display_interface)
        
        # 如果使用了接口，还可以显示更多信息
        if self.scheduler and v_id in self.scheduler.vehicle_statuses:
            task_id = self.scheduler.vehicle_statuses[v_id].get('current_task')
            if task_id and task_id in self.scheduler.tasks:
                task = self.scheduler.tasks[task_id]
                quality = getattr(task, 'quality_score', 0)
                self.path_labels["路径质量:"].setText(f"{quality:.2f}")
                
                path_length = path_structure.get('total_length', 0)
                if isinstance(path_length, int) and path_length > 0:
                    self.path_labels["路径长度:"].setText(f"{path_length} 点")
                else:
                    self.path_labels["路径长度:"].setText("--")
            else:
                self.path_labels["路径质量:"].setText("--")
                self.path_labels["路径长度:"].setText("--")
        else:
            self.path_labels["路径质量:"].setText("--")
            self.path_labels["路径长度:"].setText("--")

class ControlPanel(QWidget):
    """控制面板"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = None
        
        layout = QVBoxLayout(self)
        
        # 环境控制
        env_group = QGroupBox("环境控制")
        env_layout = QVBoxLayout()
        
        # 文件选择
        file_layout = QHBoxLayout()
        self.file_label = QLabel("未选择文件")
        self.file_label.setStyleSheet("background-color: #f8f9fa; padding: 5px; border-radius: 3px;")
        self.browse_btn = QPushButton("浏览...")
        self.load_btn = QPushButton("加载环境")
        
        file_layout.addWidget(self.file_label, 1)
        file_layout.addWidget(self.browse_btn)
        file_layout.addWidget(self.load_btn)
        env_layout.addLayout(file_layout)
        
        env_group.setLayout(env_layout)
        layout.addWidget(env_group)
        
        # 骨干路径控制
        backbone_group = QGroupBox("骨干路径")
        backbone_layout = QVBoxLayout()
        
        param_layout = QGridLayout()
        param_layout.addWidget(QLabel("质量阈值:"), 0, 0)
        self.quality_spin = QDoubleSpinBox()
        self.quality_spin.setRange(0.1, 1.0)
        self.quality_spin.setSingleStep(0.1)
        self.quality_spin.setValue(0.6)
        param_layout.addWidget(self.quality_spin, 0, 1)
        
        backbone_layout.addLayout(param_layout)
        
        self.generate_btn = QPushButton("生成骨干路径")
        backbone_layout.addWidget(self.generate_btn)
        
        backbone_group.setLayout(backbone_layout)
        layout.addWidget(backbone_group)
        
        # 仿真控制
        sim_group = QGroupBox("仿真控制")
        sim_layout = QVBoxLayout()
        
        # 速度控制
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("速度:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 100)
        self.speed_slider.setValue(50)
        self.speed_label = QLabel("1.0x")
        speed_layout.addWidget(self.speed_slider)
        speed_layout.addWidget(self.speed_label)
        sim_layout.addLayout(speed_layout)
        
        # 控制按钮
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始")
        self.pause_btn = QPushButton("暂停")
        self.reset_btn = QPushButton("重置")
        
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.pause_btn)
        btn_layout.addWidget(self.reset_btn)
        sim_layout.addLayout(btn_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        sim_layout.addWidget(self.progress_bar)
        
        sim_group.setLayout(sim_layout)
        layout.addWidget(sim_group)
        
        # 车辆控制
        vehicle_group = QGroupBox("车辆控制")
        vehicle_layout = QVBoxLayout()
        
        # 点位选择
        point_layout = QGridLayout()
        
        point_layout.addWidget(QLabel("装载点:"), 0, 0)
        self.loading_combo = QComboBox()
        point_layout.addWidget(self.loading_combo, 0, 1)
        
        point_layout.addWidget(QLabel("卸载点:"), 1, 0)
        self.unloading_combo = QComboBox()
        point_layout.addWidget(self.unloading_combo, 1, 1)
        
        vehicle_layout.addLayout(point_layout)
        
        # 任务按钮
        task_layout = QHBoxLayout()
        self.assign_task_btn = QPushButton("分配任务")
        self.cancel_task_btn = QPushButton("取消任务")
        
        task_layout.addWidget(self.assign_task_btn)
        task_layout.addWidget(self.cancel_task_btn)
        vehicle_layout.addLayout(task_layout)
        
        vehicle_group.setLayout(vehicle_layout)
        layout.addWidget(vehicle_group)
        
        # 显示选项
        display_group = QGroupBox("显示选项")
        display_layout = QVBoxLayout()
        
        self.show_backbone_cb = QCheckBox("显示骨干路径")
        self.show_backbone_cb.setChecked(True)
        self.show_paths_cb = QCheckBox("显示车辆路径")
        self.show_paths_cb.setChecked(True)
        self.show_interfaces_cb = QCheckBox("显示骨干接口")
        self.show_interfaces_cb.setChecked(True)
        display_layout.addWidget(self.show_interfaces_cb)        
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
        self.generate_btn.clicked.connect(main_window.generate_backbone_network)
        self.start_btn.clicked.connect(main_window.start_simulation)
        self.pause_btn.clicked.connect(main_window.pause_simulation)
        self.reset_btn.clicked.connect(main_window.reset_simulation)
        self.assign_task_btn.clicked.connect(main_window.assign_vehicle_task)
        self.cancel_task_btn.clicked.connect(main_window.cancel_vehicle_task)
        self.speed_slider.valueChanged.connect(main_window.update_simulation_speed)
        self.show_backbone_cb.stateChanged.connect(main_window.toggle_backbone_display)
        self.show_paths_cb.stateChanged.connect(main_window.toggle_path_display)
        self.show_interfaces_cb.stateChanged.connect(main_window.toggle_interface_display)
class OptimizedMineGUI(QMainWindow):
    """优化的露天矿多车协同调度系统GUI"""
    
    def __init__(self):
        super().__init__()
        
        # 系统组件
        self.env = None
        self.backbone_network = None
        self.path_planner = None
        self.vehicle_scheduler = None
        self.traffic_manager = None
        
        # GUI状态
        self.is_simulating = False
        self.simulation_speed = 1.0
        self.map_file_path = None
        
        # 设置样式
        self.setStyleSheet(GLOBAL_STYLE)
        
        # 初始化UI
        self.init_ui()
        
        # 定时器
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(50)  # 20 FPS
        
        self.sim_timer = QTimer(self)
        self.sim_timer.timeout.connect(self.simulation_step)
    
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("露天矿多车协同调度系统 - 骨干路径版")
        self.setGeometry(100, 100, 1200, 800)
        
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧控制面板
        self.control_panel = ControlPanel()
        self.control_panel.set_main_window(self)
        self.control_panel.setMinimumWidth(280)
        self.control_panel.setMaximumWidth(320)
        main_layout.addWidget(self.control_panel, 0)
        
        # 右侧显示区域
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # 图形视图
        self.graphics_view = MineGraphicsView()
        right_layout.addWidget(self.graphics_view, 1)
        
        # 底部信息面板
        bottom_splitter = QSplitter(Qt.Horizontal)
        
        # 车辆信息面板
        self.vehicle_info_panel = VehicleInfoPanel()
        bottom_splitter.addWidget(self.vehicle_info_panel)
        
        # 日志面板
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.addWidget(QLabel("系统日志"))
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(120)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        clear_log_btn = QPushButton("清除日志")
        clear_log_btn.clicked.connect(self.clear_log)
        log_layout.addWidget(clear_log_btn)
        
        bottom_splitter.addWidget(log_widget)
        bottom_splitter.setStretchFactor(0, 1)
        bottom_splitter.setStretchFactor(1, 1)
        
        right_layout.addWidget(bottom_splitter, 0)
        
        main_layout.addWidget(right_widget, 1)
        
        # 创建菜单和工具栏
        self.create_menu_bar()
        self.create_tool_bar()
        
        # 状态栏
        self.statusBar().showMessage("系统就绪")
    
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件")
        
        open_action = QAction("打开地图", self)
        open_action.triggered.connect(self.browse_file)
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
        
        zoom_in_action = QAction("放大", self)
        zoom_in_action.triggered.connect(self.zoom_in)
        view_menu.addAction(zoom_in_action)
        
        zoom_out_action = QAction("缩小", self)
        zoom_out_action.triggered.connect(self.zoom_out)
        view_menu.addAction(zoom_out_action)
        
        fit_view_action = QAction("适应视图", self)
        fit_view_action.triggered.connect(self.fit_view)
        view_menu.addAction(fit_view_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu("帮助")
        
        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_tool_bar(self):
        """创建工具栏"""
        toolbar = QToolBar("主工具栏")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # 文件操作
        open_action = QAction("打开", self)
        open_action.triggered.connect(self.browse_file)
        toolbar.addAction(open_action)
        
        toolbar.addSeparator()
        
        # 仿真控制
        start_action = QAction("开始", self)
        start_action.triggered.connect(self.start_simulation)
        toolbar.addAction(start_action)
        
        pause_action = QAction("暂停", self)
        pause_action.triggered.connect(self.pause_simulation)
        toolbar.addAction(pause_action)
        
        reset_action = QAction("重置", self)
        reset_action.triggered.connect(self.reset_simulation)
        toolbar.addAction(reset_action)
        
        toolbar.addSeparator()
        
        # 骨干路径生成
        generate_action = QAction("生成骨干路径", self)
        generate_action.triggered.connect(self.generate_backbone_network)
        toolbar.addAction(generate_action)
    
    # 主要功能方法
    def browse_file(self):
        """浏览文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开地图文件", "", "地图文件 (*.json);;所有文件 (*)"
        )
        
        if file_path:
            self.map_file_path = file_path
            self.control_panel.file_label.setText(os.path.basename(file_path))
            self.log("已选择地图文件: " + os.path.basename(file_path))
    
    def load_environment(self):
        """加载环境"""
        if not self.map_file_path:
            self.log("请先选择地图文件", "error")
            return
        
        try:
            self.log("正在加载环境...")
            
            # 使用环境加载器 - 恢复正确的加载方式
            from mine_loader import MineEnvironmentLoader
            loader = MineEnvironmentLoader()
            self.env = loader.load_environment(self.map_file_path)
            
            # 设置到图形视图
            self.graphics_view.set_environment(self.env)
            
            # 更新控制面板的组合框
            self.update_point_combos()
            
            # 创建系统组件
            self.create_system_components()
            
            # 更新车辆信息面板
            self.vehicle_info_panel.set_environment(self.env, self.vehicle_scheduler)
            
            self.log("环境加载成功", "success")
            
            # 启用控件
            self.enable_controls(True)
                
        except Exception as e:
            self.log(f"加载环境失败: {str(e)}", "error")
    
    def create_system_components(self):
        """创建系统组件"""
        if not self.env:
            return
        
        try:
            # 创建骨干路径网络
            self.backbone_network = SimplifiedBackbonePathNetwork(self.env)
            
            # 创建路径规划器
            self.path_planner = SimplifiedPathPlanner(self.env, self.backbone_network)
            
            # 创建交通管理器
            self.traffic_manager = OptimizedTrafficManager(self.env, self.backbone_network)
            
            # 创建车辆调度器
            try:
                self.vehicle_scheduler = SimplifiedECBSVehicleScheduler(
                    self.env, self.path_planner, self.traffic_manager, self.backbone_network
                )
                self.log("使用ECBS增强调度器", "success")
            except Exception as e:
                self.vehicle_scheduler = SimplifiedVehicleScheduler(
                    self.env, self.path_planner, self.backbone_network, self.traffic_manager
                )
                self.log("使用标准调度器", "warning")
            
            # 初始化车辆状态
            self.vehicle_scheduler.initialize_vehicles()
            
            # 创建默认任务模板
            if self.env.loading_points and self.env.unloading_points:
                self.vehicle_scheduler.create_mission_template("default")
            
            self.log("系统组件初始化完成", "success")
            
        except Exception as e:
            self.log(f"系统组件初始化失败: {str(e)}", "error")
    
    def update_point_combos(self):
        """更新点位组合框"""
        if not self.env:
            return
        
        # 更新装载点
        self.control_panel.loading_combo.clear()
        for i, point in enumerate(self.env.loading_points):
            self.control_panel.loading_combo.addItem(
                f"装载点 {i+1} ({point[0]:.0f}, {point[1]:.0f})", i
            )
        
        # 更新卸载点
        self.control_panel.unloading_combo.clear()
        for i, point in enumerate(self.env.unloading_points):
            self.control_panel.unloading_combo.addItem(
                f"卸载点 {i+1} ({point[0]:.0f}, {point[1]:.0f})", i
            )
    
    def generate_backbone_network(self):
        """生成骨干路径网络 - 更新版本"""
        if not self.env or not self.backbone_network:
            self.log("请先加载环境", "error")
            return
        
        try:
            self.log("正在生成骨干路径网络...")
            
            # 使用新的接口系统生成
            quality_threshold = self.control_panel.quality_spin.value()
            interface_spacing = 8  # 使用更小的接口间距
            
            start_time = time.time()
            success = self.backbone_network.generate_backbone_network(
                quality_threshold=quality_threshold,
                interface_spacing=interface_spacing
            )
            generation_time = time.time() - start_time
            
            if success:
                # 调试新接口系统
                self.backbone_network.debug_interface_system()
                
                # 更新到其他组件
                self.path_planner.set_backbone_network(self.backbone_network)
                self.traffic_manager.set_backbone_network(self.backbone_network)
                self.vehicle_scheduler.set_backbone_network(self.backbone_network)
                

                # 在图形视图中显示
                self.graphics_view.set_backbone_network(self.backbone_network)
                
                path_count = len(self.backbone_network.backbone_paths)
                interface_count = len(self.backbone_network.backbone_interfaces)
                
                self.log(f"骨干路径网络生成成功 - {path_count} 条路径，"
                        f"{interface_count} 个接口，耗时 {generation_time:.2f}s", "success")
            else:
                self.log("骨干路径网络生成失败", "error")
                
        except Exception as e:
            self.log(f"生成骨干路径网络失败: {str(e)}", "error")
    
    def start_simulation(self):
        """开始仿真"""
        if not self.env:
            self.log("请先加载环境", "error")
            return
        
        self.is_simulating = True
        self.control_panel.start_btn.setEnabled(False)
        self.control_panel.pause_btn.setEnabled(True)
        
        # 启动定时器
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
        
        if self.env:
            self.env.reset()
        
        if self.vehicle_scheduler:
            self.vehicle_scheduler.initialize_vehicles()
        
        self.graphics_view.set_environment(self.env)
        
        if self.backbone_network:
            self.graphics_view.set_backbone_network(self.backbone_network)
        
        self.control_panel.progress_bar.setValue(0)
        self.vehicle_info_panel.set_environment(self.env, self.vehicle_scheduler)
        
        self.log("仿真已重置")
    
    def simulation_step(self):
        """仿真步骤"""
        if not self.is_simulating or not self.env:
            return
        
        time_step = 0.5 * self.simulation_speed
        
        if not hasattr(self.env, 'current_time'):
            self.env.current_time = 0
        
        self.env.current_time += time_step
        
        # 更新调度器
        if self.vehicle_scheduler:
            try:
                self.vehicle_scheduler.update(time_step)
            except Exception as e:
                self.log(f"调度器更新错误: {e}", "error")
        
        # 更新进度条
        max_time = 3600  # 1小时
        progress = min(100, int(self.env.current_time * 100 / max_time))
        self.control_panel.progress_bar.setValue(progress)
        
        if progress >= 100:
            self.pause_simulation()
            self.log("仿真完成", "success")
    
    def update_display(self):
        """更新显示"""
        if not self.env:
            return
        
        try:
            # 更新车辆显示
            self.graphics_view.update_vehicles()
            
            # 更新车辆信息面板
            current_index = self.vehicle_info_panel.vehicle_combo.currentIndex()
            if current_index >= 0:
                self.vehicle_info_panel.update_vehicle_info(current_index)
            
            # 更新状态栏
            if self.vehicle_scheduler:
                stats = self.vehicle_scheduler.get_stats()
                completed = stats.get('completed_tasks', 0)
                total_vehicles = len(self.env.vehicles)
                
                status_text = f"车辆: {total_vehicles}, 已完成任务: {completed}"
                
                if hasattr(self.vehicle_scheduler, 'conflict_counts'):
                    total_conflicts = sum(self.vehicle_scheduler.conflict_counts.values())
                    if total_conflicts > 0:
                        status_text += f", ECBS解决冲突: {total_conflicts}"
                
                self.statusBar().showMessage(status_text)
                
        except Exception as e:
            self.log(f"显示更新错误: {e}", "error")
    def toggle_interface_display(self, state):
        """切换接口显示"""
        show = state == Qt.Checked
        self.graphics_view.mine_scene.set_show_interfaces(show)    
    def assign_vehicle_task(self):
        """分配车辆任务"""
        if not self.vehicle_scheduler:
            self.log("调度器未初始化", "error")
            return
        
        # 获取选中的车辆
        vehicle_id = self.vehicle_info_panel.vehicle_combo.itemData(
            self.vehicle_info_panel.vehicle_combo.currentIndex()
        )
        
        if vehicle_id is None:
            self.log("请选择一个车辆", "warning")
            return
        
        # 获取选中的点位
        loading_id = self.control_panel.loading_combo.currentData()
        unloading_id = self.control_panel.unloading_combo.currentData()
        
        if loading_id is None or unloading_id is None:
            # 使用默认任务模板
            if "default" in self.vehicle_scheduler.mission_templates:
                if self.vehicle_scheduler.assign_mission(vehicle_id, "default"):
                    self.log(f"已为车辆 {vehicle_id} 分配默认任务", "success")
                else:
                    self.log(f"车辆 {vehicle_id} 任务分配失败", "error")
            else:
                self.log("无可用任务模板", "error")
        else:
            # 创建特定任务
            template_id = f"specific_{vehicle_id}_{loading_id}_{unloading_id}"
            
            if self.vehicle_scheduler.create_mission_with_specific_points(
                template_id, loading_id, unloading_id
            ):
                if self.vehicle_scheduler.assign_mission(vehicle_id, template_id):
                    self.log(f"已为车辆 {vehicle_id} 分配特定任务: L{loading_id+1}→U{unloading_id+1}", "success")
                else:
                    self.log(f"车辆 {vehicle_id} 特定任务分配失败", "error")
            else:
                self.log("特定任务模板创建失败", "error")
    
    def cancel_vehicle_task(self):
        """取消车辆任务"""
        if not self.vehicle_scheduler:
            return
        
        vehicle_id = self.vehicle_info_panel.vehicle_combo.itemData(
            self.vehicle_info_panel.vehicle_combo.currentIndex()
        )
        
        if vehicle_id is None:
            self.log("请选择一个车辆", "warning")
            return
        
        # 清除任务队列
        if vehicle_id in self.vehicle_scheduler.task_queues:
            self.vehicle_scheduler.task_queues[vehicle_id] = []
        
        # 重置车辆状态
        if vehicle_id in self.vehicle_scheduler.vehicle_statuses:
            status = self.vehicle_scheduler.vehicle_statuses[vehicle_id]
            status['status'] = 'idle'
            status['current_task'] = None
            
        if vehicle_id in self.env.vehicles:
            self.env.vehicles[vehicle_id]['status'] = 'idle'
        
        # 释放交通管理器中的路径
        if self.traffic_manager:
            self.traffic_manager.release_vehicle_path(vehicle_id)
        
        self.log(f"已取消车辆 {vehicle_id} 的任务", "success")
    
    def update_simulation_speed(self):
        """更新仿真速度"""
        value = self.control_panel.speed_slider.value()
        self.simulation_speed = value / 50.0
        self.control_panel.speed_label.setText(f"{self.simulation_speed:.1f}x")
        
        # 更新定时器间隔
        if self.is_simulating:
            interval = max(50, int(100 / self.simulation_speed))
            self.sim_timer.start(interval)
    
    def toggle_backbone_display(self, state):
        """切换骨干路径显示"""
        show = state == Qt.Checked
        self.graphics_view.set_show_backbone(show)
    
    def toggle_path_display(self, state):
        """切换路径显示"""
        show = state == Qt.Checked
        self.graphics_view.set_show_trajectories(show)
    
    def enable_controls(self, enabled):
        """启用/禁用控件"""
        self.control_panel.start_btn.setEnabled(enabled)
        self.control_panel.reset_btn.setEnabled(enabled)
        self.control_panel.generate_btn.setEnabled(enabled)
        self.control_panel.assign_task_btn.setEnabled(enabled)
        self.control_panel.cancel_task_btn.setEnabled(enabled)
    
    def zoom_in(self):
        """放大"""
        self.graphics_view.scale(1.2, 1.2)
    
    def zoom_out(self):
        """缩小"""
        self.graphics_view.scale(1/1.2, 1/1.2)
    
    def fit_view(self):
        """适应视图"""
        self.graphics_view.fitInView(
            self.graphics_view.mine_scene.sceneRect(), Qt.KeepAspectRatio
        )
    
    def save_results(self):
        """保存结果"""
        if not self.env:
            self.log("没有结果可保存", "warning")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", f"simulation_result_{time.strftime('%Y%m%d_%H%M%S')}.json",
            "JSON文件 (*.json);;所有文件 (*)"
        )
        
        if file_path:
            try:
                data = {
                    'time': getattr(self.env, 'current_time', 0),
                    'vehicles': {},
                    'stats': {}
                }
                
                for vehicle_id, vehicle in self.env.vehicles.items():
                    data['vehicles'][vehicle_id] = {
                        'position': vehicle.get('position'),
                        'status': vehicle.get('status'),
                        'completed_cycles': vehicle.get('completed_cycles', 0)
                    }
                
                if self.vehicle_scheduler:
                    data['stats'] = self.vehicle_scheduler.get_stats()
                
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                self.log(f"结果已保存到: {file_path}", "success")
                
            except Exception as e:
                self.log(f"保存失败: {str(e)}", "error")
    
    def show_about(self):
        """显示关于对话框"""
        about_text = """
        <h2>露天矿多车协同调度系统</h2>
        <p>基于骨干路径的智能调度系统</p>
        <p>版本: 2.1 优化版</p>
        <hr>
        <h3>主要特性:</h3>
        <ul>
            <li>简化的骨干路径网络</li>
            <li>智能路径规划与拼接</li>
            <li>ECBS冲突检测与解决</li>
            <li>实时车辆状态监控</li>
            <li>直观的可视化界面</li>
        </ul>
        """
        
        QMessageBox.about(self, "关于", about_text)
    
    def log(self, message, level="info"):
        """添加日志"""
        current_time = time.strftime("%H:%M:%S")
        
        color_map = {
            "error": "red",
            "warning": "orange", 
            "success": "green",
            "info": "black"
        }
        
        color = color_map.get(level, "black")
        formatted_message = f'<span style="color: {color};">[{current_time}] {message}</span>'
        
        self.log_text.append(formatted_message)
        
        # 滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def clear_log(self):
        """清除日志"""
        self.log_text.clear()
class InterfaceGraphicsItem(QGraphicsItemGroup):
    """接口点图形项"""
    def __init__(self, interface, parent=None):
        super().__init__(parent)
        self.interface = interface
        self.setZValue(8)
        
        self.create_interface_visual()
    
    def create_interface_visual(self):
        """创建接口可视化"""
        x, y = self.interface.position[0], self.interface.position[1]
        
        # 接口圆圈
        radius = 3
        if self.interface.is_occupied:
            color = QColor(255, 100, 100, 180)  # 红色表示占用
        else:
            color = QColor(100, 255, 100, 180)  # 绿色表示可用
        
        circle = QGraphicsEllipseItem(x - radius, y - radius, radius * 2, radius * 2)
        circle.setBrush(QBrush(color))
        circle.setPen(QPen(Qt.black, 0.5))
        self.addToGroup(circle)
        
        # 方向指示箭头
        arrow_length = 5
        direction = self.interface.direction
        end_x = x + arrow_length * math.cos(direction)
        end_y = y + arrow_length * math.sin(direction)
        
        arrow = QGraphicsLineItem(x, y, end_x, end_y)
        arrow.setPen(QPen(Qt.blue, 1.0))
        self.addToGroup(arrow)
        
        # 接口ID标签（小字体）
        if hasattr(self.interface, 'interface_id'):
            label = QGraphicsTextItem(self.interface.interface_id.split('_')[-1])
            label.setPos(x + 4, y - 2)
            label.setDefaultTextColor(Qt.black)
            label.setFont(QFont("Arial", 2))
            self.addToGroup(label)
def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = OptimizedMineGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()