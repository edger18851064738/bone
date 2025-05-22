import sys
import os
import math
import time
import json
import numpy as np
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
from backbone_network import OptimizedBackbonePathNetwork
from path_planner import OptimizedPathPlanner
from traffic_manager import OptimizedTrafficManager
from vehicle_scheduler import VehicleScheduler, ECBSVehicleScheduler, VehicleTask
from environment import OpenPitMineEnv

# 全局样式表定义
GLOBAL_STYLESHEET = """
QMainWindow {
    background-color: #f8f9fa;
}

QTabWidget::pane {
    border: 1px solid #d3d3d3;
    background-color: white;
    border-radius: 4px;
}

QTabBar::tab {
    background-color: #f1f1f1;
    border: 1px solid #d3d3d3;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    padding: 8px 16px;
    font-weight: bold;
    min-width: 120px;
}

QTabBar::tab:selected {
    background-color: white;
    border-bottom: 1px solid white;
}

QGroupBox {
    font-weight: bold;
    border: 1px solid #d3d3d3;
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 20px;
    background-color: #ffffff;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 5px 8px;
    background-color: #f8f9fa;
    border-radius: 2px;
}

QPushButton {
    background-color: #007bff;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 4px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #0069d9;
}

QPushButton:pressed {
    background-color: #005cbf;
}

QPushButton:disabled {
    background-color: #6c757d;
    color: #f8f9fa;
}

QLabel {
    font-size: 12px;
}

QComboBox {
    border: 1px solid #d3d3d3;
    border-radius: 4px;
    padding: 5px;
    min-width: 150px;
    background-color: white;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox QAbstractItemView {
    border: 1px solid #d3d3d3;
    border-radius: 4px;
    selection-background-color: #007bff;
    selection-color: white;
}

QSpinBox, QDoubleSpinBox {
    border: 1px solid #d3d3d3;
    border-radius: 4px;
    padding: 5px;
    min-width: 100px;
}

QTextEdit {
    border: 1px solid #d3d3d3;
    border-radius: 4px;
    background-color: white;
}

QProgressBar {
    border: 1px solid #d3d3d3;
    border-radius: 4px;
    background-color: #f8f9fa;
    text-align: center;
    font-weight: bold;
    color: black;
    height: 20px;
}

QProgressBar::chunk {
    background-color: #28a745;
    border-radius: 3px;
}

QSlider::groove:horizontal {
    border: 1px solid #d3d3d3;
    height: 8px;
    background: #f8f9fa;
    border-radius: 4px;
}

QSlider::handle:horizontal {
    background: #007bff;
    border: 1px solid #007bff;
    width: 18px;
    height: 18px;
    border-radius: 9px;
    margin: -5px 0;
}

QSlider::handle:horizontal:hover {
    background: #0069d9;
}

QStatusBar {
    background-color: #f8f9fa;
    border-top: 1px solid #d3d3d3;
    color: #495057;
}

QToolBar {
    background-color: #f8f9fa;
    border-bottom: 1px solid #d3d3d3;
    spacing: 6px;
}

QMenuBar {
    background-color: #f8f9fa;
}

QMenuBar::item {
    padding: 6px 16px;
}

QMenuBar::item:selected {
    background-color: #007bff;
    color: white;
}

QMenu {
    background-color: white;
    border: 1px solid #d3d3d3;
}

QMenu::item {
    padding: 6px 24px 6px 24px;
}

QMenu::item:selected {
    background-color: #007bff;
    color: white;
}

QCheckBox {
    padding: 5px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
}

QFrame[frameShape="4"] { /* QFrame::HLine */
    color: #d3d3d3;
    height: 1px;
}

QTableWidget {
    border: 1px solid #d3d3d3;
    border-radius: 4px;
}

QTableWidget::item {
    padding: 5px;
}

QHeaderView::section {
    background-color: #f8f9fa;
    padding: 5px;
    border: 1px solid #d3d3d3;
    font-weight: bold;
}

QListWidget {
    border: 1px solid #d3d3d3;
    border-radius: 4px;
}

QListWidget::item {
    padding: 5px;
    border-bottom: 1px solid #f1f1f1;
}

QListWidget::item:selected {
    background-color: #007bff;
    color: white;
}
"""

# 车辆显示颜色配置
VEHICLE_COLORS = {
    'idle': QColor(128, 128, 128),      # 灰色
    'loading': QColor(40, 167, 69),     # 绿色
    'unloading': QColor(220, 53, 69),   # 红色
    'moving': QColor(0, 123, 255)       # 蓝色
}

class EnhancedVehicleGraphicsItem(QGraphicsItemGroup):
    """增强的车辆图形项 - 支持路径结构可视化"""
    def __init__(self, vehicle_id, vehicle_data, parent=None):
        super().__init__(parent)
        self.vehicle_id = vehicle_id
        self.vehicle_data = vehicle_data
        self.position = vehicle_data['position']
        self.vehicle_length = 6.0
        self.vehicle_width = 3.0
        
        # 创建车辆图形组件
        self.vehicle_body = QGraphicsPolygonItem(self)
        self.vehicle_label = QGraphicsTextItem(str(vehicle_id), self)
        self.status_label = QGraphicsTextItem("", self)
        self.load_indicator = QGraphicsRectItem(self)
        
        # 设置车辆颜色和样式
        self._update_vehicle_appearance()
        
        # 设置Z值
        self.setZValue(10)
        
        # 更新位置
        self.update_position()
    
    def _update_vehicle_appearance(self):
        """更新车辆外观"""
        status = self.vehicle_data.get('status', 'idle')
        color = VEHICLE_COLORS.get(status, VEHICLE_COLORS['idle'])
        
        # 调试：确保颜色映射正确
        if status not in VEHICLE_COLORS:
            print(f"警告: 未知的车辆状态 '{status}', 使用默认颜色")
            color = VEHICLE_COLORS['idle']
        
        # 车辆主体
        self.vehicle_body.setBrush(QBrush(color))
        self.vehicle_body.setPen(QPen(Qt.black, 0.5))
        
        # 车辆标签
        self.vehicle_label.setDefaultTextColor(Qt.white)
        font = QFont("Arial", 3, QFont.Bold)
        self.vehicle_label.setFont(font)
        
        # 状态标签
        status_text = {
            'idle': '空闲',
            'loading': '装载中',
            'unloading': '卸载中',
            'moving': '移动中'
        }.get(status, '')
        
        self.status_label.setPlainText(status_text)
        self.status_label.setDefaultTextColor(Qt.black)
        self.status_label.setFont(QFont("Arial", 2))
        
        # 载重指示器
        load = self.vehicle_data.get('load', 0)
        max_load = self.vehicle_data.get('max_load', 100)
        load_ratio = load / max_load if max_load > 0 else 0
        
        # 根据载重比例设置颜色
        if load_ratio < 0.3:
            load_color = QColor(100, 200, 100)  # 绿色
        elif load_ratio < 0.7:
            load_color = QColor(255, 255, 100)  # 黄色
        else:
            load_color = QColor(255, 100, 100)  # 红色
        
        self.load_indicator.setBrush(QBrush(load_color))
        self.load_indicator.setPen(QPen(Qt.black, 0.3))
    
    def update_position(self):
        """更新车辆位置和朝向"""
        if not self.position or len(self.position) < 3:
            return
            
        x, y, theta = self.position
        
        # 创建车辆多边形
        polygon = QPolygonF()
        half_length = self.vehicle_length / 2
        half_width = self.vehicle_width / 2
        corners_relative = [
            QPointF(half_length, half_width),
            QPointF(half_length, -half_width),
            QPointF(-half_length, -half_width),
            QPointF(-half_length, half_width)
        ]
        
        # 应用旋转
        transform = QTransform()
        transform.rotate(theta * 180 / math.pi)
        
        for corner in corners_relative:
            rotated_corner = transform.map(corner)
            polygon.append(QPointF(x + rotated_corner.x(), y + rotated_corner.y()))
        
        self.vehicle_body.setPolygon(polygon)
        
        # 更新标签位置
        self.vehicle_label.setPos(x - 1.5, y - 1.5)
        self.status_label.setPos(x - 5, y + 3.5)
        
        # 更新载重指示器
        indicator_width = 2.0
        indicator_height = 0.5
        self.load_indicator.setRect(
            x - indicator_width/2, 
            y - self.vehicle_width/2 - 1,
            indicator_width, 
            indicator_height
        )
    
    def update_data(self, vehicle_data):
        """更新车辆数据"""
        self.vehicle_data = vehicle_data
        self.position = vehicle_data['position']
        
        # 调试输出 - 可以在需要时启用
        # print(f"车辆 {self.vehicle_id} 状态更新: {vehicle_data.get('status', 'unknown')}")
        
        # 更新外观
        self._update_vehicle_appearance()
        
        # 更新位置
        self.update_position()


class EnhancedPathGraphicsItem(QGraphicsItemGroup):
    """增强的路径图形项 - 区分不同路径段"""
    def __init__(self, path, parent=None, vehicle_id=None, path_structure=None):
        super().__init__(parent)
        self.path_data = path
        self.vehicle_id = vehicle_id
        self.path_structure = path_structure
        self.path_segments = []
        
        # 根据路径结构创建可视化
        if path_structure and path_structure.get('type') in ['three_segment', 'hybrid']:
            self.create_structured_path()
        else:
            self.create_simple_path()
        
        self.setZValue(5)
    
    def create_structured_path(self):
        """创建结构化路径 - 不同部分使用不同样式"""
        if not self.path_structure:
            return
        
        # 提取路径各部分
        to_backbone = self.path_structure.get('to_backbone_path')
        backbone = self.path_structure.get('backbone_path')
        from_backbone = self.path_structure.get('from_backbone_path')
        
        # 绘制从起点到骨干网络的路径（虚线，亮蓝色）
        if to_backbone and len(to_backbone) > 1:
            segment_item = self._create_path_segment(
                to_backbone,
                QColor(0, 200, 255, 220),
                0.8,
                Qt.DashLine,
                [5, 3]
            )
            if segment_item:
                self.addToGroup(segment_item)
                self.path_segments.append(segment_item)
        
        # 绘制骨干网络路径（实线，粗，绿色）
        if backbone and len(backbone) > 1:
            segment_item = self._create_path_segment(
                backbone,
                QColor(50, 180, 50, 230),
                1.5,
                Qt.SolidLine
            )
            if segment_item:
                self.addToGroup(segment_item)
                self.path_segments.append(segment_item)
        
        # 绘制从骨干网络到终点的路径（虚线，红色）
        if from_backbone and len(from_backbone) > 1:
            segment_item = self._create_path_segment(
                from_backbone,
                QColor(255, 100, 100, 220),
                0.8,
                Qt.DashLine,
                [5, 3]
            )
            if segment_item:
                self.addToGroup(segment_item)
                self.path_segments.append(segment_item)
    
    def create_simple_path(self):
        """创建简单路径"""
        if not self.path_data or len(self.path_data) < 2:
            return
        
        segment_item = self._create_path_segment(
            self.path_data,
            QColor(0, 190, 255, 200),
            1.0,
            Qt.DashLine,
            [7, 4]
        )
        
        if segment_item:
            self.addToGroup(segment_item)
            self.path_segments.append(segment_item)
    
    def _create_path_segment(self, path_points, color, width, style, dash_pattern=None):
        """创建路径段图形项"""
        if not path_points or len(path_points) < 2:
            return None
        
        painter_path = QPainterPath()
        painter_path.moveTo(path_points[0][0], path_points[0][1])
        
        for point in path_points[1:]:
            painter_path.lineTo(point[0], point[1])
        
        path_item = QGraphicsPathItem(painter_path)
        
        pen = QPen(color, width)
        pen.setStyle(style)
        
        if dash_pattern:
            pen.setDashPattern(dash_pattern)
        
        path_item.setPen(pen)
        
        return path_item
    
    def update_structure(self, new_structure):
        """更新路径结构"""
        for item in self.path_segments:
            self.removeFromGroup(item)
            if item.scene():
                item.scene().removeItem(item)
        
        self.path_segments = []
        self.path_structure = new_structure
        
        if new_structure and new_structure.get('type') in ['three_segment', 'hybrid']:
            self.create_structured_path()
        else:
            self.create_simple_path()


class BackboneNetworkVisualizer(QGraphicsItemGroup):
    """骨干路径网络可视化组件 - 增强版"""
    
    def __init__(self, backbone_network, parent=None):
        super().__init__(parent)
        self.backbone_network = backbone_network
        self.path_items = {}
        self.connection_items = {}
        self.node_items = {}
        self.traffic_flow_items = {}
        self.setZValue(1)
        self.update_visualization()
    
    def update_visualization(self):
        """更新骨干网络可视化"""
        self._clear_items()
        
        if not self.backbone_network:
            return
        
        self._draw_paths()
        self._draw_connections()
        self._draw_nodes()
        self._draw_traffic_flow()
    
    def _clear_items(self):
        """清除现有图形项"""
        for items_dict in [self.path_items, self.connection_items, 
                          self.node_items, self.traffic_flow_items]:
            for item in items_dict.values():
                self.removeFromGroup(item)
            items_dict.clear()
    
    def _draw_paths(self):
        """绘制骨干路径"""
        for path_id, path_data in self.backbone_network.paths.items():
            path = path_data['path']
            
            if not path or len(path) < 2:
                continue
            
            painter_path = QPainterPath()
            painter_path.moveTo(path[0][0], path[0][1])
            
            for point in path[1:]:
                painter_path.lineTo(point[0], point[1])
            
            path_item = QGraphicsPathItem(painter_path)
            
            # 根据路径质量设置颜色
            quality = path_data.get('quality_score', 0.5)
            if quality >= 0.8:
                color = QColor(40, 180, 40, 180)  # 高质量-绿色
            elif quality >= 0.6:
                color = QColor(180, 180, 40, 180)  # 中等质量-黄色
            else:
                color = QColor(180, 40, 40, 180)  # 低质量-红色
            
            pen = QPen(color, 1.2)
            path_item.setPen(pen)
            
            self.addToGroup(path_item)
            self.path_items[path_id] = path_item
    
    def _draw_connections(self):
        """绘制连接点"""
        for conn_id, conn_data in self.backbone_network.connections.items():
            position = conn_data['position']
            conn_type = conn_data.get('type', 'intermediate')
            
            if conn_type == 'endpoint':
                radius = 2.0
                color = QColor(220, 120, 40)
            else:
                radius = 1.5
                color = QColor(120, 220, 40)
            
            conn_item = QGraphicsEllipseItem(
                position[0] - radius, position[1] - radius,
                radius * 2, radius * 2
            )
            
            conn_item.setBrush(QBrush(color))
            conn_item.setPen(QPen(Qt.black, 0.5))
            
            self.addToGroup(conn_item)
            self.connection_items[conn_id] = conn_item
    
    def _draw_nodes(self):
        """绘制关键节点"""
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
            
            if node_type == 'loading_point':
                radius = 3.5
                color = QColor(0, 180, 0)
                symbol = "L"
            elif node_type == 'unloading_point':
                radius = 3.5
                color = QColor(180, 0, 0)
                symbol = "U"
            else:
                radius = 3.0
                color = QColor(100, 100, 100)
                symbol = "P"
            
            node_item = QGraphicsEllipseItem(
                position[0] - radius, position[1] - radius,
                radius * 2, radius * 2
            )
            
            node_item.setBrush(QBrush(color))
            node_item.setPen(QPen(Qt.black, 0.8))
            
            # 添加标签
            label_item = QGraphicsTextItem(symbol)
            label_item.setPos(position[0] - 1, position[1] - 1)
            label_item.setFont(QFont("Arial", 4, QFont.Bold))
            label_item.setDefaultTextColor(Qt.white)
            
            self.addToGroup(node_item)
            self.addToGroup(label_item)
            self.node_items[node_id] = node_item
    
    def _draw_traffic_flow(self):
        """绘制交通流指示"""
        for path_id, path_data in self.backbone_network.paths.items():
            traffic_flow = path_data.get('traffic_flow', 0)
            capacity = path_data.get('capacity', 1)
            
            if traffic_flow > 0:
                ratio = min(1.0, traffic_flow / max(1, capacity))
                
                # 在路径旁边绘制流量指示器
                path = path_data['path']
                if len(path) >= 2:
                    mid_point = path[len(path)//2]
                    
                    # 流量指示器颜色
                    if ratio < 0.5:
                        flow_color = QColor(0, 255, 0, 150)  # 绿色
                    elif ratio < 0.8:
                        flow_color = QColor(255, 255, 0, 150)  # 黄色
                    else:
                        flow_color = QColor(255, 0, 0, 150)  # 红色
                    
                    # 绘制流量圆圈
                    flow_radius = 2 + ratio * 3
                    flow_item = QGraphicsEllipseItem(
                        mid_point[0] - flow_radius, mid_point[1] - flow_radius,
                        flow_radius * 2, flow_radius * 2
                    )
                    
                    flow_item.setBrush(QBrush(flow_color))
                    flow_item.setPen(QPen(Qt.NoPen))
                    
                    self.addToGroup(flow_item)
                    self.traffic_flow_items[path_id] = flow_item
    
    def update_traffic_flow(self):
        """更新交通流可视化"""
        # 清除旧的流量指示器
        for item in self.traffic_flow_items.values():
            self.removeFromGroup(item)
        self.traffic_flow_items.clear()
        
        # 重新绘制流量指示器
        self._draw_traffic_flow()


class MineGraphicsScene(QGraphicsScene):
    """矿场图形场景 - 增强版"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.env = None
        self.planner = None
        self.show_trajectories = True
        self.show_backbone = True
        self.show_traffic_flow = True
        self.grid_size = 10.0
        
        # 存储图形项
        self.grid_items = []
        self.obstacle_items = []
        self.loading_point_items = []
        self.unloading_point_items = []
        self.vehicle_items = {}
        self.path_items = {}
        self.backbone_visualizer = None
        
        self.setSceneRect(0, 0, 500, 500)
        self.background_color = QColor(245, 245, 250)
    
    def set_environment(self, env, planner=None):
        """设置环境"""
        self.env = env
        self.planner = planner
        
        if env:
            self.setSceneRect(0, 0, env.width, env.height)
        
        self.clear()
        self._reset_items()
        
        if env:
            self.draw_grid()
            self.draw_obstacles()
            self.draw_loading_points()
            self.draw_unloading_points()
            self.draw_vehicles()
    
    def _reset_items(self):
        """重置所有图形项列表"""
        self.grid_items = []
        self.obstacle_items = []
        self.loading_point_items = []
        self.unloading_point_items = []
        self.vehicle_items = {}
        self.path_items = {}
        self.backbone_visualizer = None
    
    def draw_grid(self):
        """绘制网格"""
        if not self.env:
            return
            
        width, height = self.env.width, self.env.height
        
        # 背景
        background = QGraphicsRectItem(0, 0, width, height)
        background.setBrush(QBrush(self.background_color))
        background.setPen(QPen(Qt.NoPen))
        background.setZValue(-100)
        self.addItem(background)
        self.grid_items.append(background)
        
        # 网格线
        major_pen = QPen(QColor(210, 210, 220))
        minor_pen = QPen(QColor(230, 230, 240))
        
        # 垂直线
        for x in range(0, width + 1, int(self.grid_size)):
            pen = major_pen if x % (int(self.grid_size * 5)) == 0 else minor_pen
            line = QGraphicsLineItem(x, 0, x, height)
            line.setPen(pen)
            line.setZValue(-90)
            self.addItem(line)
            self.grid_items.append(line)
        
        # 水平线
        for y in range(0, height + 1, int(self.grid_size)):
            pen = major_pen if y % (int(self.grid_size * 5)) == 0 else minor_pen
            line = QGraphicsLineItem(0, y, width, y)
            line.setPen(pen)
            line.setZValue(-90)
            self.addItem(line)
            self.grid_items.append(line)
    
    def draw_obstacles(self):
        """绘制障碍物"""
        if not self.env:
            return
        
        gradient = QLinearGradient(0, 0, 10, 10)
        gradient.setColorAt(0, QColor(80, 80, 90))
        gradient.setColorAt(1, QColor(60, 60, 70))
        
        brush = QBrush(gradient)
        pen = QPen(QColor(40, 40, 50), 0.2)
        
        for x, y in self.env.obstacle_points:
            rect = QGraphicsRectItem(x, y, 1, 1)
            rect.setBrush(brush)
            rect.setPen(pen)
            rect.setZValue(-50)
            self.addItem(rect)
            self.obstacle_items.append(rect)
    
    def draw_loading_points(self):
        """绘制装载点"""
        if not self.env:
            return
            
        for i, point in enumerate(self.env.loading_points):
            x, y = point[0], point[1]
            
            # 外发光
            glow_radius = 12
            glow = QGraphicsEllipseItem(x - glow_radius/2, y - glow_radius/2, 
                                       glow_radius, glow_radius)
            gradient = QRadialGradient(x, y, glow_radius/2)
            gradient.setColorAt(0, QColor(0, 200, 0, 100))
            gradient.setColorAt(1, QColor(0, 150, 0, 0))
            glow.setBrush(QBrush(gradient))
            glow.setPen(QPen(Qt.NoPen))
            glow.setZValue(-25)
            self.addItem(glow)
            self.loading_point_items.append(glow)
            
            # 装载区域
            area_radius = 8
            area = QGraphicsEllipseItem(x - area_radius/2, y - area_radius/2,
                                       area_radius, area_radius)
            area.setBrush(QBrush(QColor(200, 255, 200, 120)))
            area.setPen(QPen(QColor(0, 120, 0), 0.5))
            area.setZValue(-20)
            self.addItem(area)
            self.loading_point_items.append(area)
            
            # 中心标记
            center_radius = 4
            center = QGraphicsEllipseItem(x - center_radius/2, y - center_radius/2,
                                         center_radius, center_radius)
            center_gradient = QRadialGradient(x, y, center_radius/2)
            center_gradient.setColorAt(0, QColor(100, 200, 100))
            center_gradient.setColorAt(1, QColor(0, 150, 0))
            center.setBrush(QBrush(center_gradient))
            center.setPen(QPen(Qt.black, 0.5))
            center.setZValue(-10)
            self.addItem(center)
            self.loading_point_items.append(center)
            
            # 标签
            text = QGraphicsTextItem(f"装载点{i+1}")
            text.setPos(x - 12, y - 15)
            text.setDefaultTextColor(QColor(0, 100, 0))
            text.setFont(QFont("SimHei", 3, QFont.Bold))
            text.setZValue(-10)
            self.addItem(text)
            self.loading_point_items.append(text)
    
    def draw_unloading_points(self):
        """绘制卸载点"""
        if not self.env:
            return
            
        for i, point in enumerate(self.env.unloading_points):
            x, y = point[0], point[1]
            
            # 外发光
            glow_radius = 12
            glow = QGraphicsEllipseItem(x - glow_radius/2, y - glow_radius/2,
                                       glow_radius, glow_radius)
            gradient = QRadialGradient(x, y, glow_radius/2)
            gradient.setColorAt(0, QColor(200, 0, 0, 100))
            gradient.setColorAt(1, QColor(150, 0, 0, 0))
            glow.setBrush(QBrush(gradient))
            glow.setPen(QPen(Qt.NoPen))
            glow.setZValue(-25)
            self.addItem(glow)
            self.unloading_point_items.append(glow)
            
            # 卸载区域
            area_radius = 8
            area = QGraphicsEllipseItem(x - area_radius/2, y - area_radius/2,
                                       area_radius, area_radius)
            area.setBrush(QBrush(QColor(255, 200, 200, 120)))
            area.setPen(QPen(QColor(120, 0, 0), 0.5))
            area.setZValue(-20)
            self.addItem(area)
            self.unloading_point_items.append(area)
            
            # 中心标记
            center_size = 4
            center = QGraphicsRectItem(x - center_size/2, y - center_size/2,
                                      center_size, center_size)
            center.setBrush(QBrush(QColor(200, 50, 50)))
            center.setPen(QPen(Qt.black, 0.5))
            center.setZValue(-10)
            self.addItem(center)
            self.unloading_point_items.append(center)
            
            # 标签
            text = QGraphicsTextItem(f"卸载点{i+1}")
            text.setPos(x - 12, y - 15)
            text.setDefaultTextColor(QColor(120, 0, 0))
            text.setFont(QFont("SimHei", 3, QFont.Bold))
            text.setZValue(-10)
            self.addItem(text)
            self.unloading_point_items.append(text)
    
    def draw_vehicles(self):
        """绘制车辆"""
        if not self.env:
            return
            
        # 清除现有车辆
        for item in self.vehicle_items.values():
            self.removeItem(item)
        self.vehicle_items.clear()
        
        # 清除路径
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
                path_item = EnhancedPathGraphicsItem(
                    vehicle_data['path'], 
                    vehicle_id=vehicle_id,
                    path_structure=path_structure
                )
                self.addItem(path_item)
                self.path_items[vehicle_id] = path_item
    
    def update_vehicles(self):
        """更新车辆"""
        if not self.env:
            return
            
        try:
            for vehicle_id, vehicle_data in self.env.vehicles.items():
                if vehicle_id in self.vehicle_items:
                    # 更新现有车辆
                    self.vehicle_items[vehicle_id].update_data(vehicle_data)
                else:
                    # 添加新车辆
                    vehicle_item = EnhancedVehicleGraphicsItem(vehicle_id, vehicle_data)
                    self.addItem(vehicle_item)
                    self.vehicle_items[vehicle_id] = vehicle_item
                
                # 更新路径 - 修复路径更新逻辑
                if (self.show_trajectories and 'path' in vehicle_data and 
                    vehicle_data['path']):
                    
                    # 移除旧路径
                    if vehicle_id in self.path_items:
                        self.removeItem(self.path_items[vehicle_id])
                    
                    # 添加新路径
                    path_structure = vehicle_data.get('path_structure')
                    path_item = EnhancedPathGraphicsItem(
                        vehicle_data['path'],
                        vehicle_id=vehicle_id,
                        path_structure=path_structure
                    )
                    self.addItem(path_item)
                    self.path_items[vehicle_id] = path_item
                elif vehicle_id in self.path_items and not self.show_trajectories:
                    # 隐藏路径
                    self.removeItem(self.path_items[vehicle_id])
                    del self.path_items[vehicle_id]
            
            # 移除已删除的车辆
            vehicle_ids_to_remove = []
            for vehicle_id in self.vehicle_items:
                if vehicle_id not in self.env.vehicles:
                    vehicle_ids_to_remove.append(vehicle_id)
            
            for vehicle_id in vehicle_ids_to_remove:
                if vehicle_id in self.vehicle_items:
                    self.removeItem(self.vehicle_items[vehicle_id])
                    del self.vehicle_items[vehicle_id]
                if vehicle_id in self.path_items:
                    self.removeItem(self.path_items[vehicle_id])
                    del self.path_items[vehicle_id]
        
        except Exception as e:
            print(f"车辆更新错误: {e}")
    
    def set_backbone_network(self, backbone_network):
        """设置骨干网络"""
        if self.backbone_visualizer:
            self.removeItem(self.backbone_visualizer)
        
        if backbone_network and self.show_backbone:
            self.backbone_visualizer = BackboneNetworkVisualizer(backbone_network)
            self.addItem(self.backbone_visualizer)
    
    def set_show_trajectories(self, show):
        """设置是否显示轨迹"""
        self.show_trajectories = show
        self.update_vehicles()
    
    def set_show_backbone(self, show):
        """设置是否显示骨干网络"""
        self.show_backbone = show
        if self.backbone_visualizer:
            self.backbone_visualizer.setVisible(show)
    
    def update_traffic_flow(self):
        """更新交通流"""
        if self.backbone_visualizer and self.show_traffic_flow:
            self.backbone_visualizer.update_traffic_flow()


class MineGraphicsView(QGraphicsView):
    """矿场图形视图 - 增强版"""
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 渲染优化
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setRenderHint(QPainter.TextAntialiasing, True)
        
        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        
        # 性能优化
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setOptimizationFlags(QGraphicsView.DontAdjustForAntialiasing)
        
        # 创建场景
        self.mine_scene = MineGraphicsScene(self)
        self.setScene(self.mine_scene)
        
        self.setBackgroundBrush(QBrush(QColor(245, 245, 250)))
        
        # 坐标显示
        self.coord_label = QLabel(self)
        self.coord_label.setStyleSheet(
            "background-color: rgba(255, 255, 255, 180); "
            "padding: 5px; border-radius: 3px;"
        )
        self.coord_label.setAlignment(Qt.AlignCenter)
        self.coord_label.setFixedSize(150, 25)
        self.coord_label.move(10, 10)
        self.coord_label.show()
    
    def wheelEvent(self, event):
        """鼠标滚轮缩放"""
        factor = 1.2
        if event.angleDelta().y() < 0:
            factor = 1.0 / factor
        self.scale(factor, factor)
    
    def mouseMoveEvent(self, event):
        """鼠标移动显示坐标"""
        super().mouseMoveEvent(event)
        scene_pos = self.mapToScene(event.pos())
        x, y = scene_pos.x(), scene_pos.y()
        self.coord_label.setText(f"X: {x:.1f}, Y: {y:.1f}")
    
    def set_environment(self, env, planner=None):
        """设置环境"""
        self.mine_scene.set_environment(env, planner)
        if env:
            self.fitInView(self.mine_scene.sceneRect(), Qt.KeepAspectRatio)
    
    def update_vehicles(self):
        """更新车辆"""
        self.mine_scene.update_vehicles()
        self.viewport().update()
    
    def set_backbone_network(self, backbone_network):
        """设置骨干网络"""
        self.mine_scene.set_backbone_network(backbone_network)
    
    def set_show_trajectories(self, show):
        """设置显示轨迹"""
        self.mine_scene.set_show_trajectories(show)
        self.viewport().update()
    
    def set_show_backbone(self, show):
        """设置显示骨干网络"""
        self.mine_scene.set_show_backbone(show)
        self.viewport().update()


class EnhancedVehicleInfoPanel(QWidget):
    """增强的车辆信息面板 - 显示路径结构和点位信息"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(10)
        
        # 标题
        self.title_label = QLabel("车辆详细信息")
        self.title_label.setFont(QFont("SimHei", 12, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("color: #2C3E50; margin-bottom: 10px;")
        self.layout.addWidget(self.title_label)
        
        # 分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        self.layout.addWidget(separator)
        
        # 车辆选择
        self.vehicle_group = QGroupBox("选择车辆")
        self.vehicle_layout = QVBoxLayout()
        
        self.vehicle_combo = QComboBox()
        self.vehicle_combo.setMinimumHeight(30)
        self.vehicle_combo.currentIndexChanged.connect(self.update_vehicle_info)
        self.vehicle_layout.addWidget(self.vehicle_combo)
        
        self.vehicle_group.setLayout(self.vehicle_layout)
        self.layout.addWidget(self.vehicle_group)
        
        # 基本信息
        self.basic_info_group = QGroupBox("基本信息")
        self.basic_info_layout = QGridLayout()
        self.basic_info_layout.setColumnStretch(1, 1)
        self.basic_info_layout.setVerticalSpacing(8)
        
        basic_fields = [
            ("id", "车辆ID:"),
            ("position", "当前位置:"),
            ("status", "状态:"),
            ("load", "载重:"),
            ("completed", "已完成循环:")
        ]
        
        self.basic_labels = {}
        self.basic_values = {}
        self.status_indicators = {}
        
        for i, (field, label) in enumerate(basic_fields):
            self.basic_labels[field] = QLabel(label)
            self.basic_labels[field].setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.basic_labels[field].setStyleSheet("font-weight: bold;")
            
            self.basic_values[field] = QLabel("-")
            self.basic_values[field].setStyleSheet(
                "background-color: #f8f9fa; padding: 3px 5px; border-radius: 3px;"
            )
            
            self.basic_info_layout.addWidget(self.basic_labels[field], i, 0)
            self.basic_info_layout.addWidget(self.basic_values[field], i, 1)
            
            if field == "status":
                indicator = QFrame()
                indicator.setFixedSize(16, 16)
                indicator.setFrameShape(QFrame.Box)
                indicator.setStyleSheet("background-color: #6c757d; border-radius: 8px;")
                self.status_indicators[field] = indicator
                self.basic_info_layout.addWidget(indicator, i, 2)
        
        self.basic_info_group.setLayout(self.basic_info_layout)
        self.layout.addWidget(self.basic_info_group)
        
        # 点位偏好信息
        self.preference_group = QGroupBox("点位偏好")
        self.preference_layout = QGridLayout()
        self.preference_layout.setColumnStretch(1, 1)
        
        self.preference_layout.addWidget(QLabel("首选装载点:"), 0, 0)
        self.preferred_loading = QLabel("-")
        self.preferred_loading.setStyleSheet(
            "background-color: #e6f7ff; padding: 3px 5px; border-radius: 3px; color: #1890ff;"
        )
        self.preference_layout.addWidget(self.preferred_loading, 0, 1)
        
        self.preference_layout.addWidget(QLabel("首选卸载点:"), 1, 0)
        self.preferred_unloading = QLabel("-")
        self.preferred_unloading.setStyleSheet(
            "background-color: #fff2e8; padding: 3px 5px; border-radius: 3px; color: #fa8c16;"
        )
        self.preference_layout.addWidget(self.preferred_unloading, 1, 1)
        
        self.preference_group.setLayout(self.preference_layout)
        self.layout.addWidget(self.preference_group)
        
        # 当前任务信息
        self.task_info_group = QGroupBox("当前任务")
        self.task_info_layout = QGridLayout()
        self.task_info_layout.setColumnStretch(1, 1)
        
        task_fields = [
            ("task_id", "任务ID:"),
            ("task_type", "任务类型:"),
            ("progress", "任务进度:"),
            ("quality_score", "路径质量:"),
            ("backbone_usage", "骨干利用率:")
        ]
        
        self.task_labels = {}
        self.task_values = {}
        
        for i, (field, label) in enumerate(task_fields):
            self.task_labels[field] = QLabel(label)
            self.task_labels[field].setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.task_labels[field].setStyleSheet("font-weight: bold;")
            
            self.task_values[field] = QLabel("-")
            self.task_values[field].setStyleSheet(
                "background-color: #f8f9fa; padding: 3px 5px; border-radius: 3px;"
            )
            
            self.task_info_layout.addWidget(self.task_labels[field], i, 0)
            self.task_info_layout.addWidget(self.task_values[field], i, 1)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumHeight(20)
        self.task_info_layout.addWidget(QLabel("视觉进度:"), len(task_fields), 0)
        self.task_info_layout.addWidget(self.progress_bar, len(task_fields), 1)
        
        self.task_info_group.setLayout(self.task_info_layout)
        self.layout.addWidget(self.task_info_group)
        
        # 路径结构信息
        self.path_structure_group = QGroupBox("路径结构")
        self.path_structure_layout = QVBoxLayout()
        
        # 路径类型显示
        self.path_type_layout = QHBoxLayout()
        self.path_type_layout.addWidget(QLabel("路径类型:"))
        self.path_type_value = QLabel("-")
        self.path_type_value.setStyleSheet(
            "background-color: #f6ffed; padding: 3px 8px; border-radius: 3px; "
            "color: #52c41a; font-weight: bold;"
        )
        self.path_type_layout.addWidget(self.path_type_value)
        self.path_type_layout.addStretch()
        self.path_structure_layout.addLayout(self.path_type_layout)
        
        # 路径段信息表格
        self.path_segments_table = QTableWidget(0, 3)
        self.path_segments_table.setHorizontalHeaderLabels(["路径段", "点数", "描述"])
        self.path_segments_table.setMaximumHeight(120)
        self.path_segments_table.horizontalHeader().setStretchLastSection(True)
        self.path_structure_layout.addWidget(self.path_segments_table)
        
        # 骨干路径信息
        self.backbone_info_layout = QHBoxLayout()
        self.backbone_info_layout.addWidget(QLabel("骨干路径:"))
        self.backbone_path_value = QLabel("-")
        self.backbone_path_value.setStyleSheet(
            "background-color: #e6fffb; padding: 3px 8px; border-radius: 3px; color: #13c2c2;"
        )
        self.backbone_info_layout.addWidget(self.backbone_path_value)
        self.backbone_info_layout.addStretch()
        self.path_structure_layout.addLayout(self.backbone_info_layout)
        
        self.path_structure_group.setLayout(self.path_structure_layout)
        self.layout.addWidget(self.path_structure_group)
        
        # 任务队列信息
        self.queue_group = QGroupBox("任务队列")
        self.queue_layout = QVBoxLayout()
        
        self.queue_table = QTableWidget(0, 4)
        self.queue_table.setHorizontalHeaderLabels(["序号", "类型", "起点", "终点"])
        self.queue_table.setMaximumHeight(100)
        self.queue_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.queue_table.horizontalHeader().setStretchLastSection(True)
        self.queue_layout.addWidget(self.queue_table)
        
        self.queue_group.setLayout(self.queue_layout)
        self.layout.addWidget(self.queue_group)
        
        # ECBS信息
        self.ecbs_group = QGroupBox("ECBS状态")
        self.ecbs_layout = QGridLayout()
        self.ecbs_layout.setColumnStretch(1, 1)
        
        self.ecbs_layout.addWidget(QLabel("车辆优先级:"), 0, 0)
        self.priority_value = QLabel("-")
        self.priority_value.setStyleSheet(
            "background-color: #fff0f6; padding: 3px 5px; border-radius: 3px; color: #eb2f96;"
        )
        self.ecbs_layout.addWidget(self.priority_value, 0, 1)
        
        self.ecbs_layout.addWidget(QLabel("历史冲突:"), 1, 0)
        self.conflict_count_value = QLabel("-")
        self.conflict_count_value.setStyleSheet(
            "background-color: #fff1f0; padding: 3px 5px; border-radius: 3px; color: #ff4d4f;"
        )
        self.ecbs_layout.addWidget(self.conflict_count_value, 1, 1)
        
        self.ecbs_group.setLayout(self.ecbs_layout)
        self.layout.addWidget(self.ecbs_group)
        
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
        """更新车辆信息显示"""
        if not hasattr(self, 'env') or not self.env:
            return
            
        if index is None or index < 0 or index >= self.vehicle_combo.count():
            return
            
        v_id = self.vehicle_combo.itemData(index)
        
        if v_id not in self.env.vehicles:
            return
        
        vehicle = self.env.vehicles[v_id]
        
        # 更新基本信息
        self._update_basic_info(v_id, vehicle)
        
        # 更新点位偏好信息
        self._update_preference_info(v_id)
        
        # 更新当前任务信息
        self._update_task_info(v_id)
        
        # 更新路径结构信息
        self._update_path_structure_info(v_id)
        
        # 更新任务队列
        self._update_task_queue_info(v_id)
        
        # 更新ECBS信息
        self._update_ecbs_info(v_id)
    
    def _update_basic_info(self, v_id, vehicle):
        """更新基本信息"""
        self.basic_values["id"].setText(str(v_id))
        
        if 'position' in vehicle:
            pos = vehicle['position']
            if len(pos) >= 3:
                angle_deg = pos[2] * 180 / math.pi
                self.basic_values["position"].setText(
                    f"({pos[0]:.1f}, {pos[1]:.1f}, {angle_deg:.1f}°)"
                )
            else:
                self.basic_values["position"].setText(f"({pos[0]:.1f}, {pos[1]:.1f})")
        
        if 'status' in vehicle:
            status = vehicle['status']
            status_map = {
                'idle': '空闲',
                'moving': '移动中',
                'loading': '装载中',
                'unloading': '卸载中'
            }
            status_text = status_map.get(status, status)
            self.basic_values["status"].setText(status_text)
            
            status_colors = {
                'idle': '#6c757d',
                'moving': '#007bff',
                'loading': '#28a745',
                'unloading': '#dc3545'
            }
            color = status_colors.get(status, '#6c757d')
            self.status_indicators["status"].setStyleSheet(
                f"background-color: {color}; border-radius: 8px;"
            )
        
        if 'load' in vehicle:
            max_load = vehicle.get('max_load', 100)
            load_percent = int(vehicle['load'] / max_load * 100)
            self.basic_values["load"].setText(f"{vehicle['load']}/{max_load} ({load_percent}%)")
        
        completed_cycles = vehicle.get('completed_cycles', 0)
        self.basic_values["completed"].setText(str(completed_cycles))
    
    def _update_preference_info(self, v_id):
        """更新点位偏好信息"""
        if not self.scheduler or not hasattr(self.scheduler, 'vehicle_statuses'):
            self.preferred_loading.setText("-")
            self.preferred_unloading.setText("-")
            return
        
        if v_id not in self.scheduler.vehicle_statuses:
            return
        
        status = self.scheduler.vehicle_statuses[v_id]
        
        if status.get('preferred_loading_point') is not None:
            loading_id = status['preferred_loading_point']
            self.preferred_loading.setText(f"装载点 {loading_id + 1}")
        else:
            self.preferred_loading.setText("无")
        
        if status.get('preferred_unloading_point') is not None:
            unloading_id = status['preferred_unloading_point']
            self.preferred_unloading.setText(f"卸载点 {unloading_id + 1}")
        else:
            self.preferred_unloading.setText("无")
    
    def _update_task_info(self, v_id):
        """更新当前任务信息"""
        if not self.scheduler or not hasattr(self.scheduler, 'vehicle_statuses'):
            self._clear_task_info()
            return
        
        if v_id not in self.scheduler.vehicle_statuses:
            self._clear_task_info()
            return
        
        status = self.scheduler.vehicle_statuses[v_id]
        current_task_id = status.get('current_task')
        
        if not current_task_id or current_task_id not in self.scheduler.tasks:
            self._clear_task_info()
            return
        
        task = self.scheduler.tasks[current_task_id]
        
        self.task_values["task_id"].setText(task.task_id)
        
        task_type_map = {
            'to_loading': '前往装载点',
            'to_unloading': '前往卸载点',
            'to_initial': '返回起点'
        }
        self.task_values["task_type"].setText(
            task_type_map.get(task.task_type, task.task_type)
        )
        
        progress_percent = int(task.progress * 100)
        self.task_values["progress"].setText(f"{progress_percent}%")
        self.progress_bar.setValue(progress_percent)
        
        if hasattr(task, 'quality_score'):
            quality_text = f"{task.quality_score:.2f}"
            if task.quality_score >= 0.8:
                quality_color = "#52c41a"
            elif task.quality_score >= 0.6:
                quality_color = "#faad14"
            else:
                quality_color = "#ff4d4f"
            
            self.task_values["quality_score"].setText(quality_text)
            self.task_values["quality_score"].setStyleSheet(
                f"background-color: #f8f9fa; padding: 3px 5px; border-radius: 3px; "
                f"color: {quality_color}; font-weight: bold;"
            )
        else:
            self.task_values["quality_score"].setText("-")
        
        if hasattr(task, 'backbone_utilization'):
            backbone_percent = int(task.backbone_utilization * 100)
            self.task_values["backbone_usage"].setText(f"{backbone_percent}%")
        else:
            self.task_values["backbone_usage"].setText("-")
    
    def _update_path_structure_info(self, v_id):
        """更新路径结构信息"""
        if not self.scheduler or not hasattr(self.scheduler, 'tasks'):
            self._clear_path_structure_info()
            return
        
        if v_id not in self.scheduler.vehicle_statuses:
            self._clear_path_structure_info()
            return
        
        status = self.scheduler.vehicle_statuses[v_id]
        current_task_id = status.get('current_task')
        
        if not current_task_id or current_task_id not in self.scheduler.tasks:
            self._clear_path_structure_info()
            return
        
        task = self.scheduler.tasks[current_task_id]
        
        if not hasattr(task, 'path_structure') or not task.path_structure:
            self._clear_path_structure_info()
            return
        
        structure = task.path_structure
        
        path_type = structure.get('type', 'unknown')
        type_map = {
            'three_segment': '三段式路径',
            'direct': '直接路径',
            'hybrid': '混合路径',
            'unknown': '未知类型'
        }
        self.path_type_value.setText(type_map.get(path_type, path_type))
        
        # 路径段信息表格
        self.path_segments_table.setRowCount(0)
        
        segments = [
            ('to_backbone_path', '起点→骨干', '从起点到骨干网络入口'),
            ('backbone_path', '骨干路径', '在骨干网络中的路径'),
            ('from_backbone_path', '骨干→终点', '从骨干网络出口到终点')
        ]
        
        row = 0
        for segment_key, segment_name, segment_desc in segments:
            segment_data = structure.get(segment_key)
            if segment_data:
                self.path_segments_table.insertRow(row)
                self.path_segments_table.setItem(row, 0, QTableWidgetItem(segment_name))
                self.path_segments_table.setItem(row, 1, QTableWidgetItem(str(len(segment_data))))
                self.path_segments_table.setItem(row, 2, QTableWidgetItem(segment_desc))
                row += 1
        
        backbone_segment = structure.get('backbone_segment')
        if backbone_segment:
            self.backbone_path_value.setText(str(backbone_segment))
        else:
            self.backbone_path_value.setText("未使用")
    
    def _update_task_queue_info(self, v_id):
        """更新任务队列信息"""
        if not self.scheduler or not hasattr(self.scheduler, 'vehicle_statuses'):
            self.queue_table.setRowCount(0)
            return
        
        if v_id not in self.scheduler.vehicle_statuses:
            self.queue_table.setRowCount(0)
            return
        
        status = self.scheduler.vehicle_statuses[v_id]
        task_queue = status.get('task_queue', [])
        
        self.queue_table.setRowCount(len(task_queue))
        
        for i, task_info in enumerate(task_queue):
            self.queue_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            
            task_type = task_info.get('task_type', '')
            task_type_map = {
                'to_loading': '→装载点',
                'to_unloading': '→卸载点',
                'to_initial': '→起点'
            }
            type_text = task_type_map.get(task_type, task_type)
            self.queue_table.setItem(i, 1, QTableWidgetItem(type_text))
            
            start = task_info.get('start', (0, 0))
            start_text = f"({start[0]:.0f}, {start[1]:.0f})"
            self.queue_table.setItem(i, 2, QTableWidgetItem(start_text))
            
            goal = task_info.get('goal', (0, 0))
            goal_text = f"({goal[0]:.0f}, {goal[1]:.0f})"
            self.queue_table.setItem(i, 3, QTableWidgetItem(goal_text))
    
    def _update_ecbs_info(self, v_id):
        """更新ECBS信息"""
        if not self.scheduler:
            self.priority_value.setText("-")
            self.conflict_count_value.setText("-")
            return
        
        if hasattr(self.scheduler, 'vehicle_priorities'):
            priority = self.scheduler.vehicle_priorities.get(v_id, 1)
            self.priority_value.setText(str(priority))
            
            conflict_count = self.scheduler.conflict_counts.get(v_id, 0)
            self.conflict_count_value.setText(str(conflict_count))
        else:
            self.priority_value.setText("不适用")
            self.conflict_count_value.setText("不适用")
    
    def _clear_task_info(self):
        """清空任务信息"""
        for value in self.task_values.values():
            value.setText("-")
        self.progress_bar.setValue(0)
    
    def _clear_path_structure_info(self):
        """清空路径结构信息"""
        self.path_type_value.setText("-")
        self.path_segments_table.setRowCount(0)
        self.backbone_path_value.setText("-")


class TaskControlPanel(QWidget):
    """任务控制面板 - 新增特定点位选择功能"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(10)
        
        # 标题
        self.title_label = QLabel("任务控制")
        self.title_label.setFont(QFont("SimHei", 12, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("color: #2C3E50; margin-bottom: 10px;")
        self.layout.addWidget(self.title_label)
        
        # 装载点选择
        loading_group = QGroupBox("装载点选择")
        loading_layout = QVBoxLayout()
        
        self.loading_points_combo = QComboBox()
        self.loading_points_combo.setMinimumHeight(25)
        loading_layout.addWidget(self.loading_points_combo)
        
        self.loading_status_label = QLabel("状态：未选择")
        self.loading_status_label.setStyleSheet("color: #666; font-size: 11px;")
        loading_layout.addWidget(self.loading_status_label)
        
        loading_group.setLayout(loading_layout)
        self.layout.addWidget(loading_group)
        
        # 卸载点选择
        unloading_group = QGroupBox("卸载点选择")
        unloading_layout = QVBoxLayout()
        
        self.unloading_points_combo = QComboBox()
        self.unloading_points_combo.setMinimumHeight(25)
        unloading_layout.addWidget(self.unloading_points_combo)
        
        self.unloading_status_label = QLabel("状态：未选择")
        self.unloading_status_label.setStyleSheet("color: #666; font-size: 11px;")
        unloading_layout.addWidget(self.unloading_status_label)
        
        unloading_group.setLayout(unloading_layout)
        self.layout.addWidget(unloading_group)
        
        # 任务分配按钮组
        button_group = QGroupBox("任务分配")
        button_layout = QVBoxLayout()
        
        self.assign_specific_button = QPushButton("分配特定路径任务")
        self.assign_specific_button.setStyleSheet("""
            QPushButton {
                background-color: #52c41a;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #389e0d;
            }
            QPushButton:pressed {
                background-color: #237804;
            }
        """)
        self.assign_specific_button.clicked.connect(self.assign_specific_task)
        button_layout.addWidget(self.assign_specific_button)
        
        self.assign_optimal_button = QPushButton("分配最优任务")
        self.assign_optimal_button.setStyleSheet("""
            QPushButton {
                background-color: #1890ff;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #096dd9;
            }
            QPushButton:pressed {
                background-color: #0050b3;
            }
        """)
        self.assign_optimal_button.clicked.connect(self.assign_optimal_task)
        button_layout.addWidget(self.assign_optimal_button)
        
        self.batch_assign_button = QPushButton("批量ECBS分配")
        self.batch_assign_button.setStyleSheet("""
            QPushButton {
                background-color: #722ed1;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #531dab;
            }
            QPushButton:pressed {
                background-color: #391085;
            }
        """)
        self.batch_assign_button.clicked.connect(self.batch_assign_tasks)
        button_layout.addWidget(self.batch_assign_button)
        
        button_group.setLayout(button_layout)
        self.layout.addWidget(button_group)
        
        # 点位使用统计
        stats_group = QGroupBox("点位使用统计")
        stats_layout = QVBoxLayout()
        
        self.loading_stats_table = QTableWidget(0, 3)
        self.loading_stats_table.setHorizontalHeaderLabels(["装载点", "使用次数", "当前车辆"])
        self.loading_stats_table.setMaximumHeight(100)
        stats_layout.addWidget(QLabel("装载点使用情况:"))
        stats_layout.addWidget(self.loading_stats_table)
        
        self.unloading_stats_table = QTableWidget(0, 3)
        self.unloading_stats_table.setHorizontalHeaderLabels(["卸载点", "使用次数", "当前车辆"])
        self.unloading_stats_table.setMaximumHeight(100)
        stats_layout.addWidget(QLabel("卸载点使用情况:"))
        stats_layout.addWidget(self.unloading_stats_table)
        
        stats_group.setLayout(stats_layout)
        self.layout.addWidget(stats_group)
        
        self.layout.addStretch()
        
        # 连接事件
        self.loading_points_combo.currentIndexChanged.connect(self.update_loading_status)
        self.unloading_points_combo.currentIndexChanged.connect(self.update_unloading_status)
    
    def set_environment(self, env, scheduler, main_window=None):
        """设置环境、调度器和主窗口引用"""
        self.env = env
        self.scheduler = scheduler
        self.main_window = main_window
        
        self.loading_points_combo.clear()
        if env and env.loading_points:
            for i, point in enumerate(env.loading_points):
                self.loading_points_combo.addItem(
                    f"装载点 {i+1} ({point[0]:.1f}, {point[1]:.1f})", i
                )
        
        self.unloading_points_combo.clear()
        if env and env.unloading_points:
            for i, point in enumerate(env.unloading_points):
                self.unloading_points_combo.addItem(
                    f"卸载点 {i+1} ({point[0]:.1f}, {point[1]:.1f})", i
                )
        
        self.update_loading_status()
        self.update_unloading_status()
        self.update_point_usage_stats()
    
    def update_loading_status(self):
        """更新装载点状态"""
        if self.loading_points_combo.count() > 0:
            current_idx = self.loading_points_combo.currentIndex()
            if current_idx >= 0:
                self.loading_status_label.setText(f"状态：已选择装载点 {current_idx + 1}")
                self.loading_status_label.setStyleSheet("color: #52c41a; font-size: 11px;")
        else:
            self.loading_status_label.setText("状态：无可用装载点")
            self.loading_status_label.setStyleSheet("color: #ff4d4f; font-size: 11px;")
    
    def update_unloading_status(self):
        """更新卸载点状态"""
        if self.unloading_points_combo.count() > 0:
            current_idx = self.unloading_points_combo.currentIndex()
            if current_idx >= 0:
                self.unloading_status_label.setText(f"状态：已选择卸载点 {current_idx + 1}")
                self.unloading_status_label.setStyleSheet("color: #fa8c16; font-size: 11px;")
        else:
            self.unloading_status_label.setText("状态：无可用卸载点")
            self.unloading_status_label.setStyleSheet("color: #ff4d4f; font-size: 11px;")
    
    def assign_specific_task(self):
        """分配使用特定装载点和卸载点的任务"""
        if not hasattr(self, 'env') or not hasattr(self, 'scheduler') or not self.main_window:
            return
        
        vehicle_id = self.get_selected_vehicle()
        if not vehicle_id:
            self.show_message("请先选择一个车辆", "warning")
            return
        
        loading_point_id = self.loading_points_combo.currentData()
        unloading_point_id = self.unloading_points_combo.currentData()
        
        if loading_point_id is None or unloading_point_id is None:
            self.show_message("请选择装载点和卸载点", "warning")
            return
        
        template_id = f"specific_mission_{vehicle_id}_{loading_point_id}_{unloading_point_id}"
        
        try:
            if self.scheduler.create_mission_with_specific_points(
                template_id, loading_point_id, unloading_point_id
            ):
                if self.scheduler.assign_mission(vehicle_id, template_id):
                    self.show_message(
                        f"已为车辆 {vehicle_id} 分配特定任务：\n"
                        f"装载点 {loading_point_id + 1} → 卸载点 {unloading_point_id + 1}",
                        "success"
                    )
                    self.update_point_usage_stats()
                else:
                    self.show_message("任务分配失败", "error")
            else:
                self.show_message("任务模板创建失败", "error")
        except Exception as e:
            self.show_message(f"分配任务时出错: {str(e)}", "error")
    
    def assign_optimal_task(self):
        """分配最优任务"""
        vehicle_id = self.get_selected_vehicle()
        if not vehicle_id:
            self.show_message("请先选择一个车辆", "warning")
            return
        
        try:
            if self.scheduler.assign_optimal_mission(vehicle_id):
                self.show_message(f"已为车辆 {vehicle_id} 分配最优任务", "success")
                self.update_point_usage_stats()
            else:
                self.show_message("最优任务分配失败", "error")
        except Exception as e:
            self.show_message(f"分配最优任务时出错: {str(e)}", "error")
    
    def batch_assign_tasks(self):
        """批量ECBS分配任务"""
        if not hasattr(self.scheduler, 'assign_tasks_batch'):
            self.show_message("当前调度器不支持批量ECBS分配", "warning")
            return
        
        try:
            tasks = []
            for vehicle_id, vehicle in self.env.vehicles.items():
                if vehicle.get('status') == 'idle':
                    vehicle_pos = vehicle['position']
                    loading_point = self.env.loading_points[0] if self.env.loading_points else None
                    unloading_point = self.env.unloading_points[0] if self.env.unloading_points else None
                    
                    if loading_point and unloading_point:
                        task = VehicleTask(
                            f"batch_task_{len(tasks)}",
                            'to_loading',
                            vehicle_pos,
                            (loading_point[0], loading_point[1], 0),
                            priority=2,
                            loading_point_id=0,
                            unloading_point_id=0
                        )
                        tasks.append(task)
                        self.scheduler.tasks[task.task_id] = task
            
            if tasks:
                assignments = self.scheduler.assign_tasks_batch(tasks)
                self.show_message(
                    f"批量ECBS分配完成：\n已分配 {len(assignments)} 个任务",
                    "success"
                )
                self.update_point_usage_stats()
            else:
                self.show_message("没有空闲车辆可以分配任务", "info")
                
        except Exception as e:
            self.show_message(f"批量分配任务时出错: {str(e)}", "error")
    
    def update_point_usage_stats(self):
        """更新点位使用统计"""
        if not self.scheduler:
            return
        
        # 更新装载点统计
        self.loading_stats_table.setRowCount(len(self.env.loading_points))
        for i, point in enumerate(self.env.loading_points):
            self.loading_stats_table.setItem(i, 0, QTableWidgetItem(f"装载点 {i+1}"))
            
            usage_count = 0
            if hasattr(self.scheduler, 'loading_point_usage'):
                usage_count = self.scheduler.loading_point_usage.get(i, 0)
            self.loading_stats_table.setItem(i, 1, QTableWidgetItem(str(usage_count)))
            
            current_vehicles = []
            if hasattr(self.scheduler, 'vehicle_statuses'):
                for v_id, status in self.scheduler.vehicle_statuses.items():
                    if status.get('preferred_loading_point') == i:
                        current_vehicles.append(v_id)
            
            vehicles_text = ", ".join(current_vehicles) if current_vehicles else "无"
            self.loading_stats_table.setItem(i, 2, QTableWidgetItem(vehicles_text))
        
        # 更新卸载点统计
        self.unloading_stats_table.setRowCount(len(self.env.unloading_points))
        for i, point in enumerate(self.env.unloading_points):
            self.unloading_stats_table.setItem(i, 0, QTableWidgetItem(f"卸载点 {i+1}"))
            
            usage_count = 0
            if hasattr(self.scheduler, 'unloading_point_usage'):
                usage_count = self.scheduler.unloading_point_usage.get(i, 0)
            self.unloading_stats_table.setItem(i, 1, QTableWidgetItem(str(usage_count)))
            
            current_vehicles = []
            if hasattr(self.scheduler, 'vehicle_statuses'):
                for v_id, status in self.scheduler.vehicle_statuses.items():
                    if status.get('preferred_unloading_point') == i:
                        current_vehicles.append(v_id)
            
            vehicles_text = ", ".join(current_vehicles) if current_vehicles else "无"
            self.unloading_stats_table.setItem(i, 2, QTableWidgetItem(vehicles_text))
    
    def get_selected_vehicle(self):
        """获取当前选中的车辆"""
        if self.main_window and hasattr(self.main_window, 'vehicle_info_panel'):
            current_index = self.main_window.vehicle_info_panel.vehicle_combo.currentIndex()
            if current_index >= 0:
                return self.main_window.vehicle_info_panel.vehicle_combo.itemData(current_index)
        return None
    
    def show_message(self, message, msg_type="info"):
        """显示消息"""
        if self.main_window and hasattr(self.main_window, 'log'):
            self.main_window.log(message, msg_type)


class ECBSConfigPanel(QWidget):
    """ECBS配置面板"""
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(10)
        
        # 标题
        self.title_label = QLabel("ECBS配置")
        self.title_label.setFont(QFont("SimHei", 12, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("color: #2C3E50; margin-bottom: 10px;")
        self.layout.addWidget(self.title_label)
        
        # 分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        self.layout.addWidget(separator)
        
        # ECBS参数组
        self.params_group = QGroupBox("算法参数")
        self.params_layout = QGridLayout()
        
        # 子最优界
        self.params_layout.addWidget(QLabel("子最优界:"), 0, 0)
        self.subopt_slider = QSlider(Qt.Horizontal)
        self.subopt_slider.setMinimum(10)
        self.subopt_slider.setMaximum(30)
        self.subopt_slider.setValue(15)
        self.subopt_slider.valueChanged.connect(self.update_subopt_display)
        self.params_layout.addWidget(self.subopt_slider, 0, 1)
        
        self.subopt_value = QLabel("1.5")
        self.params_layout.addWidget(self.subopt_value, 0, 2)
        
        # 冲突解决策略
        self.params_layout.addWidget(QLabel("解决策略:"), 1, 0)
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["ECBS", "优先级", "时间窗口"])
        self.params_layout.addWidget(self.strategy_combo, 1, 1, 1, 2)
        
        # 冲突检测范围
        self.params_layout.addWidget(QLabel("检测范围:"), 2, 0)
        self.detection_range = QDoubleSpinBox()
        self.detection_range.setRange(1.0, 20.0)
        self.detection_range.setSingleStep(0.5)
        self.detection_range.setValue(5.0)
        self.params_layout.addWidget(self.detection_range, 2, 1, 1, 2)
        
        self.params_group.setLayout(self.params_layout)
        self.layout.addWidget(self.params_group)
        
        # 高级选项组
        self.advanced_group = QGroupBox("高级选项")
        self.advanced_layout = QVBoxLayout()
        
        self.focal_enabled = QCheckBox("启用焦点列表优化")
        self.focal_enabled.setChecked(True)
        self.advanced_layout.addWidget(self.focal_enabled)
        
        self.pathreuse_enabled = QCheckBox("启用路径重用")
        self.pathreuse_enabled.setChecked(True)
        self.advanced_layout.addWidget(self.pathreuse_enabled)
        
        self.priority_enabled = QCheckBox("启用优先级调度")
        self.priority_enabled.setChecked(True)
        self.advanced_layout.addWidget(self.priority_enabled)
        
        self.advanced_group.setLayout(self.advanced_layout)
        self.layout.addWidget(self.advanced_group)
        
        # 冲突状态组
        self.conflict_group = QGroupBox("冲突状态")
        self.conflict_layout = QVBoxLayout()
        
        self.conflict_stats = QLabel("已解决冲突: 0")
        self.conflict_stats.setStyleSheet("font-weight: bold;")
        self.conflict_layout.addWidget(self.conflict_stats)
        
        self.conflict_table = QTableWidget(0, 3)
        self.conflict_table.setHorizontalHeaderLabels(["车辆1", "车辆2", "类型"])
        self.conflict_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.conflict_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.conflict_layout.addWidget(self.conflict_table)
        
        self.conflict_group.setLayout(self.conflict_layout)
        self.layout.addWidget(self.conflict_group)
        
        # 应用按钮
        self.apply_button = QPushButton("应用ECBS设置")
        self.layout.addWidget(self.apply_button)
        
        self.layout.addStretch()
    
    def update_subopt_display(self):
        """更新子最优界显示"""
        value = self.subopt_slider.value() / 10.0
        self.subopt_value.setText(f"{value:.1f}")
    
    def get_settings(self):
        """获取当前设置"""
        return {
            "suboptimality": self.subopt_slider.value() / 10.0,
            "strategy": self.strategy_combo.currentText(),
            "detection_range": self.detection_range.value(),
            "focal_enabled": self.focal_enabled.isChecked(),
            "pathreuse_enabled": self.pathreuse_enabled.isChecked(),
            "priority_enabled": self.priority_enabled.isChecked()
        }
    
    def update_conflict_stats(self, resolved_count, conflicts):
        """更新冲突统计信息"""
        self.conflict_stats.setText(f"已解决冲突: {resolved_count}")
        
        self.conflict_table.setRowCount(len(conflicts))
        
        for i, conflict in enumerate(conflicts):
            self.conflict_table.setItem(i, 0, QTableWidgetItem(str(conflict.agent1)))
            self.conflict_table.setItem(i, 1, QTableWidgetItem(str(conflict.agent2)))
            self.conflict_table.setItem(i, 2, QTableWidgetItem(conflict.conflict_type))


class PerformanceMonitorPanel(QWidget):
    """性能监控面板 - 新增"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(10)
        
        # 标题
        self.title_label = QLabel("性能监控")
        self.title_label.setFont(QFont("SimHei", 12, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("color: #2C3E50; margin-bottom: 10px;")
        self.layout.addWidget(self.title_label)
        
        # 系统性能组
        self.system_group = QGroupBox("系统性能")
        self.system_layout = QGridLayout()
        
        # 性能指标
        metrics = [
            ("fps", "帧率:", "FPS"),
            ("cpu_usage", "CPU使用率:", "%"),
            ("memory_usage", "内存使用:", "MB"),
            ("planning_time", "路径规划时间:", "ms"),
            ("conflict_resolution", "冲突解决率:", "%")
        ]
        
        self.metric_labels = {}
        self.metric_values = {}
        self.components = {}
        for i, (key, label, unit) in enumerate(metrics):
            self.metric_labels[key] = QLabel(label)
            self.metric_labels[key].setStyleSheet("font-weight: bold;")
            self.system_layout.addWidget(self.metric_labels[key], i, 0)
            
            self.metric_values[key] = QLabel("--")
            self.metric_values[key].setStyleSheet(
                "background-color: #f8f9fa; padding: 3px 5px; border-radius: 3px;"
            )
            self.system_layout.addWidget(self.metric_values[key], i, 1)
            
            unit_label = QLabel(unit)
            unit_label.setStyleSheet("color: #666;")
            self.system_layout.addWidget(unit_label, i, 2)
        
        self.system_group.setLayout(self.system_layout)
        self.layout.addWidget(self.system_group)
        
        # 算法性能组
        self.algo_group = QGroupBox("算法性能")
        self.algo_layout = QGridLayout()
        
        algo_metrics = [
            ("backbone_efficiency", "骨干网络效率:", "%"),
            ("cache_hit_rate", "缓存命中率:", "%"),
            ("path_quality", "平均路径质量:", "分"),
            ("ecbs_convergence", "ECBS收敛时间:", "ms")
        ]
        
        self.algo_labels = {}
        self.algo_values = {}
        
        for i, (key, label, unit) in enumerate(algo_metrics):
            self.algo_labels[key] = QLabel(label)
            self.algo_labels[key].setStyleSheet("font-weight: bold;")
            self.algo_layout.addWidget(self.algo_labels[key], i, 0)
            
            self.algo_values[key] = QLabel("--")
            self.algo_values[key].setStyleSheet(
                "background-color: #f8f9fa; padding: 3px 5px; border-radius: 3px;"
            )
            self.algo_layout.addWidget(self.algo_values[key], i, 1)
            
            unit_label = QLabel(unit)
            unit_label.setStyleSheet("color: #666;")
            self.algo_layout.addWidget(unit_label, i, 2)
        
        self.algo_group.setLayout(self.algo_layout)
        self.layout.addWidget(self.algo_group)
        
        # 实时统计图表区域
        self.chart_group = QGroupBox("实时性能图表")
        self.chart_layout = QVBoxLayout()
        
        # 简化的性能趋势显示
        self.performance_text = QTextEdit()
        self.performance_text.setMaximumHeight(120)
        self.performance_text.setReadOnly(True)
        self.chart_layout.addWidget(self.performance_text)
        
        self.chart_group.setLayout(self.chart_layout)
        self.layout.addWidget(self.chart_group)
        
        self.layout.addStretch()
        
        # 更新定时器
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_performance_data)
        self.update_timer.start(1000)  # 每秒更新
        
        # 性能数据缓存
        self.performance_history = []
        self.max_history = 60  # 保存60秒历史
    
    def set_system_components(self, **components):
        """设置系统组件用于性能监控"""
        self.components = components
    
    def update_performance_data(self):
        """更新性能数据"""
        current_time = time.time()
        
        # 系统性能
        self.metric_values["fps"].setText("60")  # 模拟帧率
        self.metric_values["cpu_usage"].setText("15.2")
        self.metric_values["memory_usage"].setText("256")
        
        # 路径规划性能
        if 'path_planner' in self.components:
            planner = self.components['path_planner']
            if hasattr(planner, 'get_performance_stats'):
                stats = planner.get_performance_stats()
                avg_time = stats.get('avg_planning_time', 0) * 1000
                self.metric_values["planning_time"].setText(f"{avg_time:.1f}")
        
        # 算法性能
        if 'backbone_network' in self.components:
            backbone = self.components['backbone_network']
            if hasattr(backbone, 'get_performance_stats'):
                stats = backbone.get_performance_stats()
                # 模拟骨干网络效率
                self.algo_values["backbone_efficiency"].setText("78.5")
        
        if 'traffic_manager' in self.components:
            traffic = self.components['traffic_manager']
            if hasattr(traffic, 'get_performance_stats'):
                stats = traffic.get_performance_stats()
                cache_rate = stats.get('cache_hit_rate', 0) * 100
                self.algo_values["cache_hit_rate"].setText(f"{cache_rate:.1f}")
        
        # 更新历史记录
        self.performance_history.append({
            'time': current_time,
            'cpu': 15.2,
            'memory': 256,
            'planning_time': 45.2
        })
        
        if len(self.performance_history) > self.max_history:
            self.performance_history.pop(0)
        
        # 更新图表文本
        self.update_chart_text()
    
    def update_chart_text(self):
        """更新图表文本显示"""
        if len(self.performance_history) < 2:
            return
        
        recent_data = self.performance_history[-10:]  # 最近10秒
        
        html = "<style>table {width:100%;} th {background:#f0f0f0;}</style>"
        html += "<table border='1' cellspacing='0' cellpadding='3'>"
        html += "<tr><th>时间</th><th>CPU%</th><th>内存MB</th><th>规划ms</th></tr>"
        
        for data in recent_data[-5:]:  # 显示最近5条
            time_str = time.strftime("%H:%M:%S", time.localtime(data['time']))
            html += f"<tr>"
            html += f"<td>{time_str}</td>"
            html += f"<td>{data['cpu']:.1f}</td>"
            html += f"<td>{data['memory']}</td>"
            html += f"<td>{data['planning_time']:.1f}</td>"
            html += f"</tr>"
        
        html += "</table>"
        self.performance_text.setHtml(html)
class BackboneNetworkDebugger:
    """骨干网络调试工具"""
    
    def __init__(self, main_window):
        self.main_window = main_window
    
    def debug_full_system(self):
        """完整的系统调试"""
        print("=" * 60)
        print("🔍 开始骨干网络系统调试")
        print("=" * 60)
        
        self.debug_backbone_basic_info()
        self.debug_path_planner_config()
        self.debug_access_point_finding()
        self.debug_path_planning_process()
        
        print("=" * 60)
        print("🏁 骨干网络系统调试完成")
        print("=" * 60)
    
    def debug_backbone_basic_info(self):
        """检查骨干网络基本信息"""
        print("\n📊 1. 骨干网络基本信息检查")
        print("-" * 40)
        
        backbone = self.main_window.backbone_network
        if not backbone:
            print("❌ 骨干网络对象为None!")
            self.main_window.log("❌ 骨干网络对象为None!", "error")
            return False
        
        print(f"✅ 骨干网络对象存在")
        
        # 检查路径数量
        path_count = len(backbone.paths) if hasattr(backbone, 'paths') else 0
        print(f"📈 路径数量: {path_count}")
        self.main_window.log(f"📈 路径数量: {path_count}", "info")
        
        if path_count == 0:
            print("❌ 骨干网络中没有路径!")
            self.main_window.log("❌ 骨干网络中没有路径! 请先点击'生成骨干路径网络'", "error")
            return False
        
        # 检查连接点数量
        conn_count = len(backbone.connections) if hasattr(backbone, 'connections') else 0
        print(f"🔗 连接点数量: {conn_count}")
        self.main_window.log(f"🔗 连接点数量: {conn_count}", "info")
        
        return True
    
    def debug_path_planner_config(self):
        """检查路径规划器配置"""
        print("\n🛠️ 2. 路径规划器配置检查")
        print("-" * 40)
        
        planner = self.main_window.path_planner
        if not planner:
            print("❌ 路径规划器对象为None!")
            self.main_window.log("❌ 路径规划器对象为None!", "error")
            return False
        
        print("✅ 路径规划器存在")
        self.main_window.log("✅ 路径规划器存在", "success")
        
        # 检查骨干网络绑定
        if hasattr(planner, 'backbone_network') and planner.backbone_network:
            print("✅ 路径规划器已绑定骨干网络")
            self.main_window.log("✅ 路径规划器已绑定骨干网络", "success")
        else:
            print("❌ 路径规划器未绑定骨干网络!")
            self.main_window.log("❌ 路径规划器未绑定骨干网络!", "error")
            return False
        
        # 检查RRT规划器
        if hasattr(planner, 'rrt_planner') and planner.rrt_planner:
            print("✅ RRT规划器存在")
            self.main_window.log("✅ RRT规划器存在", "success")
        else:
            print("❌ RRT规划器为None!")
            self.main_window.log("❌ RRT规划器为None!", "error")
        
        return True
    
    def debug_access_point_finding(self):
        """测试接入点查找"""
        print("\n🎯 3. 接入点查找测试")
        print("-" * 40)
        
        backbone = self.main_window.backbone_network
        env = self.main_window.env
        
        if not backbone or not env:
            self.main_window.log("❌ 缺少必要组件", "error")
            return False
        
        # 测试装载点
        if env.loading_points:
            test_pos = env.loading_points[0]
            print(f"🧪 测试位置: ({test_pos[0]:.1f}, {test_pos[1]:.1f})")
            
            try:
                nearest_conn = backbone.find_nearest_connection_optimized(test_pos, max_distance=20.0)
                if nearest_conn:
                    distance = nearest_conn.get('distance', 0)
                    print(f"✅ 找到连接点，距离: {distance:.2f}")
                    self.main_window.log(f"✅ 找到连接点，距离: {distance:.2f}", "success")
                else:
                    print("❌ 未找到连接点")
                    self.main_window.log("❌ 未找到连接点", "error")
            except Exception as e:
                print(f"❌ 查找出错: {e}")
                self.main_window.log(f"❌ 查找出错: {e}", "error")
        
        return True
    
    def debug_path_planning_process(self):
        """测试路径规划流程 - 修复版"""
        print("\n🗺️ 4. 路径规划流程测试")
        print("-" * 40)
        
        env = self.main_window.env
        planner = self.main_window.path_planner
        
        if not env or not planner or not env.loading_points or not env.unloading_points:
            self.main_window.log("❌ 缺少必要组件", "error")
            return False
        
        start = env.loading_points[0]
        goal = env.unloading_points[0]
        
        print(f"🚀 测试路径规划: 从({start[0]:.1f},{start[1]:.1f}) 到({goal[0]:.1f},{goal[1]:.1f})")
        self.main_window.log(f"🚀 测试路径规划", "info")
        
        try:
            # 启用调试并限制尝试次数
            original_debug = getattr(planner, 'debug', False)
            planner.debug = True
            
            # 限制测试规模，避免无限输出
            original_max_attempts = 3
            
            result = planner.plan_path(
                vehicle_id="debug_test",
                start=start,
                goal=goal,
                use_backbone=True,
                strategy='backbone_first',
                max_attempts=2  # 限制尝试次数
            )
            
            if result:
                if isinstance(result, tuple) and len(result) == 2:
                    path, structure = result
                    path_type = structure.get('type', 'unknown') if structure else 'unknown'
                else:
                    path = result
                    path_type = 'direct'
                
                print(f"✅ 规划成功，类型: {path_type}")
                self.main_window.log(f"✅ 规划成功，类型: {path_type}", "success")
                
                if path_type == 'three_segment':
                    print("🎉 成功使用骨干网络!")
                    self.main_window.log("🎉 成功使用骨干网络!", "success")
                elif path_type == 'direct':
                    print("⚠️ 使用了直接路径，未使用骨干网络")
                    self.main_window.log("⚠️ 使用了直接路径，未使用骨干网络", "warning")
                else:
                    print(f"ℹ️ 使用了 {path_type} 路径")
                    self.main_window.log(f"ℹ️ 使用了 {path_type} 路径", "info")
            else:
                print("❌ 路径规划失败")
                self.main_window.log("❌ 路径规划失败", "error")
            
            # 恢复调试设置
            planner.debug = original_debug
            
            return True
        
        except Exception as e:
            print(f"❌ 规划出错: {e}")
            self.main_window.log(f"❌ 规划出错: {e}", "error")
            
            # 打印详细错误信息用于调试
            import traceback
            print(f"详细错误:\n{traceback.format_exc()}")
            
            return False

class OptimizedMineGUI(QMainWindow):
    """优化后的露天矿多车协同调度系统GUI - 完整版（修复装载点选择问题）"""
    
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
        
        # 应用样式表
        self.setStyleSheet(GLOBAL_STYLESHEET)
        
        # 初始化UI
        self.init_ui()
        
        # 创建更新定时器
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(50)  # 20 FPS
        
        # 模拟定时器
        self.sim_timer = QTimer(self)
        self.sim_timer.timeout.connect(self.simulation_step)
    
    def init_ui(self):
        """初始化用户界面"""
        # 设置主窗口
        self.setWindowTitle("露天矿多车协同调度系统 (基于骨干网络) - 优化版")
        self.setGeometry(100, 100, 1400, 900)
        
        # 创建中央窗口部件
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
        
        # 创建工具栏
        self.create_tool_bar()
        
        # 创建状态栏
        self.statusBar().showMessage("系统就绪 - 等待加载环境")
    
    def create_control_panel(self):
        """创建左侧控制面板"""
        # 控制面板容器
        self.control_panel = QFrame()
        self.control_panel.setFrameShape(QFrame.StyledPanel)
        self.control_panel.setMinimumWidth(350)
        self.control_panel.setMaximumWidth(400)
        
        # 控制面板布局
        control_layout = QVBoxLayout(self.control_panel)
        
        # 创建选项卡
        self.tab_widget = QTabWidget()
        control_layout.addWidget(self.tab_widget)
        
        # 环境选项卡
        self.create_env_tab()
        
        # 路径选项卡
        self.create_path_tab()
        
        # 车辆选项卡
        self.create_vehicle_tab()
        
        # 任务选项卡
        self.create_task_tab()
        
        # 性能选项卡
        self.create_performance_tab()
        
        # 日志区域
        self.create_log_area(control_layout)
        
        # 添加到主布局
        self.main_layout.addWidget(self.control_panel, 1)
    
    def create_env_tab(self):
        """创建环境选项卡"""
        self.env_tab = QWidget()
        self.env_layout = QVBoxLayout(self.env_tab)
        self.tab_widget.addTab(self.env_tab, "环境")
        
        # 文件加载组
        file_group = QGroupBox("环境加载")
        file_layout = QGridLayout()
        
        file_layout.addWidget(QLabel("地图文件:"), 0, 0)
        self.map_path = QLabel("未选择")
        self.map_path.setStyleSheet("background-color: #f8f9fa; padding: 5px; border-radius: 3px;")
        file_layout.addWidget(self.map_path, 0, 1)
        
        self.browse_button = QPushButton("浏览...")
        self.browse_button.clicked.connect(self.open_map_file)
        file_layout.addWidget(self.browse_button, 0, 2)
        
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
            self.env_info_values[label].setStyleSheet(
                "background-color: #f8f9fa; padding: 3px; border-radius: 3px;"
            )
            info_layout.addWidget(self.env_info_values[label], i, 1)
        
        info_group.setLayout(info_layout)
        self.env_layout.addWidget(info_group)
        
        # 环境控制组
        control_group = QGroupBox("模拟控制")
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
        self.speed_value.setStyleSheet(
            "background-color: #f8f9fa; padding: 3px; border-radius: 3px;"
        )
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
        
        self.env_layout.addStretch()
    
    def create_path_tab(self):
        """创建路径选项卡"""
        self.path_tab = QWidget()
        self.path_layout = QVBoxLayout(self.path_tab)
        self.tab_widget.addTab(self.path_tab, "路径")
        
        # 路径生成组
        generate_group = QGroupBox("生成骨干路径")
        generate_layout = QVBoxLayout()
        
        # 参数设置
        param_layout = QGridLayout()
        
        param_layout.addWidget(QLabel("连接点间距:"), 0, 0)
        self.conn_spacing = QSpinBox()
        self.conn_spacing.setRange(5, 50)
        self.conn_spacing.setValue(10)
        param_layout.addWidget(self.conn_spacing, 0, 1)
        
        param_layout.addWidget(QLabel("质量阈值:"), 1, 0)
        self.quality_threshold = QDoubleSpinBox()
        self.quality_threshold.setRange(0.1, 1.0)
        self.quality_threshold.setSingleStep(0.1)
        self.quality_threshold.setValue(0.6)
        param_layout.addWidget(self.quality_threshold, 1, 1)
        
        generate_layout.addLayout(param_layout)
        
        # 生成按钮
        self.generate_paths_button = QPushButton("生成骨干路径网络")
        self.generate_paths_button.clicked.connect(self.generate_backbone_network)
        generate_layout.addWidget(self.generate_paths_button)
        
        generate_group.setLayout(generate_layout)
        self.path_layout.addWidget(generate_group)
        
        # 路径信息组
        info_group = QGroupBox("路径网络信息")
        info_layout = QGridLayout()
        
        labels = ["总路径数:", "连接点数:", "总长度:", "平均质量:", "生成时间:"]
        self.path_info_values = {}
        
        for i, label in enumerate(labels):
            info_layout.addWidget(QLabel(label), i, 0)
            self.path_info_values[label] = QLabel("--")
            self.path_info_values[label].setStyleSheet(
                "background-color: #f8f9fa; padding: 3px; border-radius: 3px;"
            )
            info_layout.addWidget(self.path_info_values[label], i, 1)
        
        info_group.setLayout(info_layout)
        self.path_layout.addWidget(info_group)
        
        # 路径显示选项
        display_group = QGroupBox("显示选项")
        display_layout = QVBoxLayout()
        
        self.show_paths_cb = QCheckBox("显示骨干路径")
        self.show_paths_cb.setChecked(True)
        self.show_paths_cb.stateChanged.connect(self.update_path_display)
        display_layout.addWidget(self.show_paths_cb)
        
        self.show_connections_cb = QCheckBox("显示连接点")
        self.show_connections_cb.setChecked(True)
        self.show_connections_cb.stateChanged.connect(self.update_path_display)
        display_layout.addWidget(self.show_connections_cb)
        
        self.show_traffic_cb = QCheckBox("显示交通流")
        self.show_traffic_cb.setChecked(True)
        self.show_traffic_cb.stateChanged.connect(self.update_path_display)
        display_layout.addWidget(self.show_traffic_cb)
        
        self.show_quality_cb = QCheckBox("显示路径质量")
        self.show_quality_cb.setChecked(True)
        self.show_quality_cb.stateChanged.connect(self.update_path_display)
        display_layout.addWidget(self.show_quality_cb)
        
        display_group.setLayout(display_layout)
        self.path_layout.addWidget(display_group)
        
        
        # ECBS配置面板
        self.ecbs_panel = ECBSConfigPanel()
        self.ecbs_panel.apply_button.clicked.connect(self.apply_ecbs_settings)
        self.path_layout.addWidget(self.ecbs_panel)
# 调试功能组 - 新增
        debug_group = QGroupBox("🔍 系统调试")
        debug_layout = QVBoxLayout()
        
        # 系统诊断按钮
        self.debug_system_button = QPushButton("🔍 骨干网络系统诊断")
        self.debug_system_button.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #138496; }
        """)
        self.debug_system_button.clicked.connect(self.debug_backbone_system)
        debug_layout.addWidget(self.debug_system_button)
        
        # 测试路径规划按钮  
        self.test_planning_button = QPushButton("🧪 测试路径规划")
        self.test_planning_button.setStyleSheet("""
            QPushButton {
                background-color: #6f42c1;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #5a32a3; }
        """)
        self.test_planning_button.clicked.connect(self.test_single_path_planning)
        debug_layout.addWidget(self.test_planning_button)
        
        debug_group.setLayout(debug_layout)
        self.path_layout.addWidget(debug_group)        
        self.path_layout.addStretch()
    
    def create_vehicle_tab(self):
        """创建车辆选项卡 - 修复版"""
        self.vehicle_tab = QWidget()
        self.vehicle_layout = QVBoxLayout(self.vehicle_tab)
        self.tab_widget.addTab(self.vehicle_tab, "车辆")
        
        # 增强的车辆信息面板
        self.vehicle_info_panel = EnhancedVehicleInfoPanel()
        self.vehicle_layout.addWidget(self.vehicle_info_panel)
        
        # 车辆控制组 - 修复版：添加装载点和卸载点选择
        control_group = QGroupBox("车辆控制")
        control_layout = QVBoxLayout()
        
        # ✅ 新增: 装载点选择
        loading_selection_layout = QHBoxLayout()
        loading_selection_layout.addWidget(QLabel("选择装载点:"))
        self.vehicle_loading_combo = QComboBox()
        self.vehicle_loading_combo.setMinimumHeight(25)
        self.vehicle_loading_combo.setToolTip("选择车辆要前往的装载点")
        loading_selection_layout.addWidget(self.vehicle_loading_combo)
        control_layout.addLayout(loading_selection_layout)
        
        # ✅ 新增: 卸载点选择  
        unloading_selection_layout = QHBoxLayout()
        unloading_selection_layout.addWidget(QLabel("选择卸载点:"))
        self.vehicle_unloading_combo = QComboBox()
        self.vehicle_unloading_combo.setMinimumHeight(25)
        self.vehicle_unloading_combo.setToolTip("选择车辆要前往的卸载点")
        unloading_selection_layout.addWidget(self.vehicle_unloading_combo)
        control_layout.addLayout(unloading_selection_layout)
        
        # 任务控制
        task_layout = QHBoxLayout()
        
        self.assign_task_button = QPushButton("分配任务")
        self.assign_task_button.clicked.connect(self.assign_vehicle_task)
        task_layout.addWidget(self.assign_task_button)
        
        self.cancel_task_button = QPushButton("取消任务")
        self.cancel_task_button.clicked.connect(self.cancel_vehicle_task)
        task_layout.addWidget(self.cancel_task_button)
        
        control_layout.addLayout(task_layout)
        
        # ✅ 修改后的位置控制按钮
        position_layout = QHBoxLayout()
        
        self.goto_loading_button = QPushButton("前往选定装载点")
        self.goto_loading_button.clicked.connect(self.goto_selected_loading_point)
        self.goto_loading_button.setToolTip("将车辆派往当前选中的装载点")
        position_layout.addWidget(self.goto_loading_button)
        
        self.goto_unloading_button = QPushButton("前往选定卸载点")
        self.goto_unloading_button.clicked.connect(self.goto_selected_unloading_point)
        self.goto_unloading_button.setToolTip("将车辆派往当前选中的卸载点")
        position_layout.addWidget(self.goto_unloading_button)
        
        control_layout.addLayout(position_layout)
        
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
        
        self.show_path_structure_cb = QCheckBox("显示路径结构")
        self.show_path_structure_cb.setChecked(True)
        self.show_path_structure_cb.stateChanged.connect(self.update_vehicle_display)
        display_layout.addWidget(self.show_path_structure_cb)
        
        display_group.setLayout(display_layout)
        self.vehicle_layout.addWidget(display_group)
        
        self.vehicle_layout.addStretch()
    
    def create_task_tab(self):
        """创建任务选项卡"""
        self.task_tab = QWidget()
        self.task_layout = QVBoxLayout(self.task_tab)
        self.tab_widget.addTab(self.task_tab, "任务")
        
        # 任务控制面板
        self.task_control_panel = TaskControlPanel()
        self.task_layout.addWidget(self.task_control_panel)
        
        # 任务列表组
        list_group = QGroupBox("任务列表")
        list_layout = QVBoxLayout()
        
        self.task_list = QListWidget()
        self.task_list.itemClicked.connect(self.update_task_info)
        list_layout.addWidget(self.task_list)
        
        list_group.setLayout(list_layout)
        self.task_layout.addWidget(list_group)
        
        # 任务信息组
        info_group = QGroupBox("任务详细信息")
        info_layout = QGridLayout()
        
        labels = ["ID:", "类型:", "状态:", "车辆:", "进度:", "质量评分:", "骨干利用率:"]
        self.task_info_values = {}
        
        for i, label in enumerate(labels):
            info_layout.addWidget(QLabel(label), i, 0)
            self.task_info_values[label] = QLabel("--")
            self.task_info_values[label].setStyleSheet(
                "background-color: #f8f9fa; padding: 3px; border-radius: 3px;"
            )
            info_layout.addWidget(self.task_info_values[label], i, 1)
        
        info_group.setLayout(info_layout)
        self.task_layout.addWidget(info_group)
        
        # 统计组
        stats_group = QGroupBox("系统统计")
        stats_layout = QGridLayout()
        
        labels = [
            "总任务数:", "已完成任务:", "失败任务:", 
            "平均利用率:", "骨干使用效率:", "冲突解决次数:"
        ]
        self.task_stats_values = {}
        
        for i, label in enumerate(labels):
            stats_layout.addWidget(QLabel(label), i, 0)
            self.task_stats_values[label] = QLabel("--")
            self.task_stats_values[label].setStyleSheet(
                "background-color: #f8f9fa; padding: 3px; border-radius: 3px;"
            )
            stats_layout.addWidget(self.task_stats_values[label], i, 1)
        
        stats_group.setLayout(stats_layout)
        self.task_layout.addWidget(stats_group)
        
        self.task_layout.addStretch()
    
    def create_performance_tab(self):
        """创建性能选项卡"""
        self.performance_tab = QWidget()
        self.performance_layout = QVBoxLayout(self.performance_tab)
        self.tab_widget.addTab(self.performance_tab, "性能")
        
        # 性能监控面板
        self.performance_panel = PerformanceMonitorPanel()
        self.performance_layout.addWidget(self.performance_panel)
    
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
        # 显示区域容器
        self.display_area = QFrame()
        self.display_area.setFrameShape(QFrame.StyledPanel)
        
        # 显示区域布局
        display_layout = QVBoxLayout(self.display_area)
        
        # 视图控制工具栏
        view_toolbar = QToolBar()
        view_toolbar.setIconSize(QSize(16, 16))
        view_toolbar.setMovable(False)
        
        # 视图控制按钮
        zoom_in_action = QAction("放大", self)
        zoom_in_action.triggered.connect(self.zoom_in)
        view_toolbar.addAction(zoom_in_action)
        
        zoom_out_action = QAction("缩小", self)
        zoom_out_action.triggered.connect(self.zoom_out)
        view_toolbar.addAction(zoom_out_action)
        
        fit_view_action = QAction("适应视图", self)
        fit_view_action.triggered.connect(self.fit_view)
        view_toolbar.addAction(fit_view_action)
        
        view_toolbar.addSeparator()
        
        # 显示选项
        self.show_backbone_action = QAction("骨干网络", self)
        self.show_backbone_action.setCheckable(True)
        self.show_backbone_action.setChecked(True)
        self.show_backbone_action.triggered.connect(self.toggle_backbone_display)
        view_toolbar.addAction(self.show_backbone_action)
        
        self.show_vehicles_action = QAction("车辆", self)
        self.show_vehicles_action.setCheckable(True)
        self.show_vehicles_action.setChecked(True)
        self.show_vehicles_action.triggered.connect(self.toggle_vehicles_display)
        view_toolbar.addAction(self.show_vehicles_action)
        
        display_layout.addWidget(view_toolbar)
        
        # 图形视图
        self.graphics_view = MineGraphicsView()
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
        
        view_menu.addAction(self.show_backbone_action)
        view_menu.addAction(self.show_vehicles_action)
        
        # 工具菜单
        tools_menu = menubar.addMenu("工具")
        
        generate_network_action = QAction("生成骨干网络", self)
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
        
        # 文件操作
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
        
        # 网络生成
        self.generate_network_action = QAction("生成骨干网络", self)
        self.generate_network_action.triggered.connect(self.generate_backbone_network)
        toolbar.addAction(self.generate_network_action)
    
    # 核心功能方法
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
            
            # 使用环境加载器
            from mine_loader import MineEnvironmentLoader
            loader = MineEnvironmentLoader()
            self.env = loader.load_environment(self.map_file_path)
            
            # 更新环境信息
            self.update_env_info()
            
            # 更新车辆信息面板
            self.update_vehicle_combo()
            
            # 设置环境到图形视图
            self.graphics_view.set_environment(self.env)
            
            self.log("环境已加载: " + os.path.basename(self.map_file_path))
            
            # 创建系统组件
            self.create_system_components()
            
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
            self.backbone_network = OptimizedBackbonePathNetwork(self.env)
            
            # 创建路径规划器
            self.path_planner = OptimizedPathPlanner(self.env)
            self.path_planner.set_backbone_network(self.backbone_network)
            
            # 创建交通管理器
            self.traffic_manager = OptimizedTrafficManager(self.env, self.backbone_network)
            
            # 创建车辆调度器（优先使用ECBS版本）
            try:
                self.vehicle_scheduler = ECBSVehicleScheduler(
                    self.env, 
                    self.path_planner, 
                    self.traffic_manager,
                    self.backbone_network
                )
                self.log("使用ECBS增强型车辆调度器", "success")
            except Exception as e:
                self.vehicle_scheduler = VehicleScheduler(
                    self.env, 
                    self.path_planner, 
                    self.backbone_network,
                    self.traffic_manager
                )
                self.log("使用标准车辆调度器", "warning")
            
            # 初始化车辆状态
            self.vehicle_scheduler.initialize_vehicles()
            
            # 设置系统组件到性能监控
            self.performance_panel.set_system_components(
                env=self.env,
                backbone_network=self.backbone_network,
                path_planner=self.path_planner,
                traffic_manager=self.traffic_manager,
                vehicle_scheduler=self.vehicle_scheduler
            )
            
            # 创建任务模板
            if self.env.loading_points and self.env.unloading_points:
                if hasattr(self.vehicle_scheduler, 'create_ecbs_mission_template'):
                    self.vehicle_scheduler.create_ecbs_mission_template("default")
                else:
                    self.vehicle_scheduler.create_mission_template("default")
            
            self.log("系统组件已初始化", "success")
            
        except Exception as e:
            self.log(f"系统组件初始化失败: {str(e)}", "error")
    
    def generate_backbone_network(self):
        """生成骨干路径网络"""
        if not self.env:
            self.log("请先加载环境!", "error")
            return
        
        if not self.backbone_network:
            self.log("骨干网络未初始化!", "error")
            return
        
        try:
            self.log("正在生成骨干路径网络...")
            
            # 获取参数
            spacing = self.conn_spacing.value()
            quality_threshold = self.quality_threshold.value()
            
            # 生成网络
            start_time = time.time()
            self.backbone_network.generate_network(spacing, quality_threshold)
            generation_time = time.time() - start_time
            
            # 更新路径信息
            self.update_path_info()
            
            # 更新生成时间
            self.path_info_values["生成时间:"].setText(f"{generation_time:.2f}s")
            
            # 设置骨干网络到其他组件
            self.path_planner.set_backbone_network(self.backbone_network)
            self.traffic_manager.set_backbone_network(self.backbone_network)
            self.vehicle_scheduler.set_backbone_network(self.backbone_network)
            
            # 在图形视图中显示
            self.graphics_view.set_backbone_network(self.backbone_network)
            
            self.log(f"骨干路径网络已生成 - {len(self.backbone_network.paths)} 条路径", "success")
            
        except Exception as e:
            self.log(f"生成骨干网络失败: {str(e)}", "error")
    
    def update_env_info(self):
        """更新环境信息"""
        if not self.env:
            return
        
        self.env_info_values["宽度:"].setText(str(self.env.width))
        self.env_info_values["高度:"].setText(str(self.env.height))
        self.env_info_values["装载点:"].setText(str(len(self.env.loading_points)))
        self.env_info_values["卸载点:"].setText(str(len(self.env.unloading_points)))
        self.env_info_values["车辆数:"].setText(str(len(self.env.vehicles)))
    
    def update_vehicle_combo(self):
        """更新车辆下拉框 - 修复版"""
        self.vehicle_info_panel.set_environment(self.env, self.vehicle_scheduler)
        self.task_control_panel.set_environment(self.env, self.vehicle_scheduler, self)
        
        # ✅ 新增: 更新车辆选项卡中的装载点和卸载点选择框
        if hasattr(self, 'vehicle_loading_combo'):
            self.vehicle_loading_combo.clear()
            if self.env and self.env.loading_points:
                for i, point in enumerate(self.env.loading_points):
                    self.vehicle_loading_combo.addItem(
                        f"装载点 {i+1} ({point[0]:.1f}, {point[1]:.1f})", i
                    )
        
        if hasattr(self, 'vehicle_unloading_combo'):
            self.vehicle_unloading_combo.clear()
            if self.env and self.env.unloading_points:
                for i, point in enumerate(self.env.unloading_points):
                    self.vehicle_unloading_combo.addItem(
                        f"卸载点 {i+1} ({point[0]:.1f}, {point[1]:.1f})", i
                    )
    
    def update_path_info(self):
        """更新路径信息"""
        if not self.backbone_network:
            return
        
        num_paths = len(self.backbone_network.paths)
        num_connections = len(self.backbone_network.connections)
        
        total_length = 0
        total_quality = 0
        quality_count = 0
        
        for path_data in self.backbone_network.paths.values():
            total_length += path_data.get('length', 0)
            quality = path_data.get('quality_score', 0)
            if quality > 0:
                total_quality += quality
                quality_count += 1
        
        avg_quality = total_quality / quality_count if quality_count > 0 else 0
        
        self.path_info_values["总路径数:"].setText(str(num_paths))
        self.path_info_values["连接点数:"].setText(str(num_connections))
        self.path_info_values["总长度:"].setText(f"{total_length:.1f}")
        self.path_info_values["平均质量:"].setText(f"{avg_quality:.2f}")
    
    # 模拟控制方法
    def update_simulation_speed(self):
        """更新模拟速度"""
        value = self.speed_slider.value()
        speed = value / 50.0
        self.simulation_speed = speed
        
        self.speed_value.setText(f"{speed:.1f}x")
        
        if hasattr(self.env, 'time_step'):
            self.env.time_step = 0.5 * speed
        
        self.log(f"模拟速度设置为 {speed:.1f}x")
    
    def start_simulation(self):
        """开始模拟"""
        if not self.env:
            self.log("请先加载环境!", "error")
            return
        
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.start_sim_action.setEnabled(False)
        self.pause_sim_action.setEnabled(True)
        
        self.is_simulating = True
        
        # 启动模拟定时器
        interval = max(50, int(100 / self.simulation_speed))
        self.sim_timer.start(interval)
        
        self.log("模拟已开始")
    
    def pause_simulation(self):
        """暂停模拟"""
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.start_sim_action.setEnabled(True)
        self.pause_sim_action.setEnabled(False)
        
        self.is_simulating = False
        self.sim_timer.stop()
        
        self.log("模拟已暂停")
    
    def reset_simulation(self):
        """重置模拟"""
        reply = QMessageBox.question(
            self, '确认重置', 
            '确定要重置模拟吗？这将清除所有当前数据。',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        if self.is_simulating:
            self.pause_simulation()
        
        if self.env:
            self.env.reset()
        
        if self.vehicle_scheduler:
            self.vehicle_scheduler.initialize_vehicles()
        
        # 重新设置环境
        self.graphics_view.set_environment(self.env)
        
        if self.backbone_network and hasattr(self.backbone_network, 'paths') and self.backbone_network.paths:
            self.graphics_view.set_backbone_network(self.backbone_network)
        
        self.progress_bar.setValue(0)
        self.update_vehicle_combo()
        
        self.log("模拟已重置")
    
    def simulation_step(self):
        """模拟步骤"""
        if not self.is_simulating or not self.env:
            return
        
        time_step = getattr(self.env, 'time_step', 0.5)
        
        if not hasattr(self.env, 'current_time'):
            self.env.current_time = 0
        
        self.env.current_time += time_step
        
        # 更新车辆调度器
        if self.vehicle_scheduler:
            try:
                self.vehicle_scheduler.update(time_step)
                
                # 确保车辆状态同步 - 添加调试信息
                if hasattr(self, 'debug') and self.debug:
                    for vehicle_id, vehicle in self.env.vehicles.items():
                        scheduler_status = self.vehicle_scheduler.vehicle_statuses.get(vehicle_id, {})
                        print(f"车辆 {vehicle_id}: 环境状态={vehicle.get('status', 'unknown')}, "
                            f"调度器状态={scheduler_status.get('status', 'unknown')}")
            
            except Exception as e:
                self.log(f"车辆调度器更新错误: {e}", "error")
        
        # 更新进度条
        max_time = 3600  # 1小时
        progress = min(100, int(self.env.current_time * 100 / max_time))
        self.progress_bar.setValue(progress)
        
        # 检查是否完成
        if progress >= 100:
            self.pause_simulation()
            self.log("模拟完成", "success")
    
    def update_display(self):
        """更新显示"""
        if not self.env:
            return
        
        try:
            # 更新车辆位置和状态
            self.graphics_view.update_vehicles()
            
            # 更新车辆信息面板
            current_index = self.vehicle_info_panel.vehicle_combo.currentIndex()
            if current_index >= 0:
                self.vehicle_info_panel.update_vehicle_info(current_index)
            
            # 更新任务列表
            self.update_task_list()
            
            # 更新交通流
            if hasattr(self.graphics_view, 'mine_scene'):
                self.graphics_view.mine_scene.update_traffic_flow()
            
            # 显示系统状态 - 增强调试信息
            if self.vehicle_scheduler and hasattr(self.vehicle_scheduler, 'conflict_counts'):
                total_conflicts = sum(self.vehicle_scheduler.conflict_counts.values())
                
                # 统计车辆状态
                status_counts = {'idle': 0, 'moving': 0, 'loading': 0, 'unloading': 0}
                for vehicle in self.env.vehicles.values():
                    status = vehicle.get('status', 'idle')
                    status_counts[status] = status_counts.get(status, 0) + 1
                
                status_text = f"车辆状态: 空闲{status_counts['idle']}, 移动{status_counts['moving']}, " \
                            f"装载{status_counts['loading']}, 卸载{status_counts['unloading']}"
                
                if total_conflicts > 0:
                    status_text += f" | ECBS已解决 {total_conflicts} 个路径冲突"
                
                self.statusBar().showMessage(status_text)
        
        except Exception as e:
            self.log(f"显示更新错误: {e}", "error")
    
    def update_task_list(self):
        """更新任务列表"""
        if not self.vehicle_scheduler:
            return
        
        current_item = self.task_list.currentItem()
        current_task_id = current_item.data(Qt.UserRole) if current_item else None
        
        self.task_list.clear()
        
        for task_id, task in self.vehicle_scheduler.tasks.items():
            item = QListWidgetItem(f"{task_id} - {task.task_type} ({task.status})")
            item.setData(Qt.UserRole, task_id)
            
            # 状态颜色
            if task.status == 'completed':
                item.setForeground(QBrush(QColor(40, 167, 69)))
            elif task.status == 'failed':
                item.setForeground(QBrush(QColor(220, 53, 69)))
            elif task.status == 'in_progress':
                item.setForeground(QBrush(QColor(0, 123, 255)))
            
            self.task_list.addItem(item)
            
            if task_id == current_task_id:
                self.task_list.setCurrentItem(item)
        
        # 更新统计
        stats = self.vehicle_scheduler.get_stats()
        
        self.task_stats_values["总任务数:"].setText(str(len(self.vehicle_scheduler.tasks)))
        self.task_stats_values["已完成任务:"].setText(str(stats.get('completed_tasks', 0)))
        self.task_stats_values["失败任务:"].setText(str(stats.get('failed_tasks', 0)))
        
        avg_util = stats.get('average_utilization', 0)
        self.task_stats_values["平均利用率:"].setText(f"{avg_util:.1%}")
        
        backbone_eff = stats.get('backbone_usage_efficiency', 0)
        self.task_stats_values["骨干使用效率:"].setText(f"{backbone_eff:.1%}")
        
        conflict_count = stats.get('conflict_resolution_count', 0)
        self.task_stats_values["冲突解决次数:"].setText(str(conflict_count))
    
    def update_task_info(self, item):
        """更新任务信息"""
        if not item or not self.vehicle_scheduler:
            return
        
        task_id = item.data(Qt.UserRole)
        
        if task_id not in self.vehicle_scheduler.tasks:
            return
        
        task = self.vehicle_scheduler.tasks[task_id]
        
        self.task_info_values["ID:"].setText(task.task_id)
        self.task_info_values["类型:"].setText(task.task_type)
        self.task_info_values["状态:"].setText(task.status)
        self.task_info_values["车辆:"].setText(str(task.assigned_vehicle) if task.assigned_vehicle else "未分配")
        self.task_info_values["进度:"].setText(f"{task.progress:.0%}")
        self.task_info_values["质量评分:"].setText(f"{task.quality_score:.2f}" if hasattr(task, 'quality_score') else "--")
        self.task_info_values["骨干利用率:"].setText(f"{task.backbone_utilization:.1%}" if hasattr(task, 'backbone_utilization') else "--")
    
    # ✅ 修复后的车辆操作方法
    def goto_selected_loading_point(self):
        """前往选定的装载点 - 修复版"""
        if not self.vehicle_scheduler or not self.env.loading_points:
            self.log("没有可用的装载点", "warning")
            return
        
        # 获取选中的车辆
        vehicle_id = self.vehicle_info_panel.vehicle_combo.itemData(
            self.vehicle_info_panel.vehicle_combo.currentIndex()
        )
        if vehicle_id is None:
            self.log("请先选择一个车辆", "warning")
            return
        
        # 获取选中的装载点ID - 修复关键点！
        if not hasattr(self, 'vehicle_loading_combo'):
            self.log("装载点选择器未初始化", "error")
            return
            
        loading_point_id = self.vehicle_loading_combo.currentData()
        if loading_point_id is None:
            self.log("请选择一个装载点", "warning")
            return
        
        # 验证装载点ID的有效性
        if loading_point_id < 0 or loading_point_id >= len(self.env.loading_points):
            self.log(f"无效的装载点ID: {loading_point_id}", "error")
            return
        
        loading_point = self.env.loading_points[loading_point_id]
        task_id = f"task_{self.vehicle_scheduler.task_counter}"
        self.vehicle_scheduler.task_counter += 1
        
        vehicle_position = self.env.vehicles[vehicle_id]['position']
        
        # 创建任务时正确设置loading_point_id - 修复关键点！
        task = VehicleTask(
            task_id,
            'to_loading',
            vehicle_position,
            (loading_point[0], loading_point[1], 0),
            2,  # 装载任务优先级较高
            loading_point_id=loading_point_id,  # ✅ 关键修复
            unloading_point_id=None
        )
        
        self.vehicle_scheduler.tasks[task_id] = task
        self.vehicle_scheduler.task_queues[vehicle_id] = [task_id]
        
        # 更新车辆状态中的任务队列信息
        self.vehicle_scheduler.vehicle_statuses[vehicle_id]['task_queue'].append({
            'task_id': task_id,
            'task_type': task.task_type,
            'start': task.start,
            'goal': task.goal,
            'priority': task.priority,
            'loading_point_id': loading_point_id  # 添加装载点信息
        })
        
        status = self.vehicle_scheduler.vehicle_statuses[vehicle_id]
        if status['status'] == 'idle':
            self.vehicle_scheduler._start_next_task(vehicle_id)
        
        self.log(f"已命令车辆 {vehicle_id} 前往装载点 {loading_point_id + 1}", "success")
        
        # ✅ 可选: 添加调试信息
        if hasattr(self, 'debug_enabled') and self.debug_enabled:
            self.debug_task_assignment(vehicle_id, task)
    
    def goto_selected_unloading_point(self):
        """前往选定的卸载点 - 修复版"""
        if not self.vehicle_scheduler or not self.env.unloading_points:
            self.log("没有可用的卸载点", "warning")
            return
        
        vehicle_id = self.vehicle_info_panel.vehicle_combo.itemData(
            self.vehicle_info_panel.vehicle_combo.currentIndex()
        )
        if vehicle_id is None:
            self.log("请先选择一个车辆", "warning")
            return
        
        # 获取选中的卸载点ID - 修复关键点！
        if not hasattr(self, 'vehicle_unloading_combo'):
            self.log("卸载点选择器未初始化", "error")
            return
            
        unloading_point_id = self.vehicle_unloading_combo.currentData()
        if unloading_point_id is None:
            self.log("请选择一个卸载点", "warning")
            return
        
        if unloading_point_id < 0 or unloading_point_id >= len(self.env.unloading_points):
            self.log(f"无效的卸载点ID: {unloading_point_id}", "error")
            return
        
        unloading_point = self.env.unloading_points[unloading_point_id]
        task_id = f"task_{self.vehicle_scheduler.task_counter}"
        self.vehicle_scheduler.task_counter += 1
        
        vehicle_position = self.env.vehicles[vehicle_id]['position']
        
        # 创建任务时正确设置unloading_point_id - 修复关键点！
        task = VehicleTask(
            task_id,
            'to_unloading',
            vehicle_position,
            (unloading_point[0], unloading_point[1], 0),
            3,  # 卸载任务优先级最高
            loading_point_id=None,
            unloading_point_id=unloading_point_id  # ✅ 关键修复
        )
        
        self.vehicle_scheduler.tasks[task_id] = task
        self.vehicle_scheduler.task_queues[vehicle_id] = [task_id]
        
        # 更新车辆状态中的任务队列信息
        self.vehicle_scheduler.vehicle_statuses[vehicle_id]['task_queue'].append({
            'task_id': task_id,
            'task_type': task.task_type,
            'start': task.start,
            'goal': task.goal,
            'priority': task.priority,
            'unloading_point_id': unloading_point_id  # 添加卸载点信息
        })
        
        status = self.vehicle_scheduler.vehicle_statuses[vehicle_id]
        if status['status'] == 'idle':
            self.vehicle_scheduler._start_next_task(vehicle_id)
        
        self.log(f"已命令车辆 {vehicle_id} 前往卸载点 {unloading_point_id + 1}", "success")
        
        # ✅ 可选: 添加调试信息
        if hasattr(self, 'debug_enabled') and self.debug_enabled:
            self.debug_task_assignment(vehicle_id, task)
    
    def assign_vehicle_task(self):
        """分配任务给当前车辆"""
        if not self.vehicle_scheduler:
            self.log("车辆调度器未初始化", "error")
            return
        
        vehicle_id = self.vehicle_info_panel.vehicle_combo.itemData(
            self.vehicle_info_panel.vehicle_combo.currentIndex()
        )
        if vehicle_id is None:
            self.log("请先选择一个车辆", "warning")
            return
        
        if "default" in self.vehicle_scheduler.mission_templates:
            if self.vehicle_scheduler.assign_mission(vehicle_id, "default"):
                self.log(f"已将默认任务分配给车辆 {vehicle_id}", "success")
            else:
                self.log(f"无法分配任务给车辆 {vehicle_id}", "error")
        else:
            self.log("无可用的任务模板", "error")
    
    def cancel_vehicle_task(self):
        """取消车辆任务"""
        if not self.vehicle_scheduler:
            return
        
        vehicle_id = self.vehicle_info_panel.vehicle_combo.itemData(
            self.vehicle_info_panel.vehicle_combo.currentIndex()
        )
        if vehicle_id is None:
            self.log("请先选择一个车辆", "warning")
            return
        
        # 清除任务队列
        if vehicle_id in self.vehicle_scheduler.task_queues:
            self.vehicle_scheduler.task_queues[vehicle_id] = []
            
            status = self.vehicle_scheduler.vehicle_statuses[vehicle_id]
            if status.get('current_task'):
                task_id = status['current_task']
                if task_id in self.vehicle_scheduler.tasks:
                    task = self.vehicle_scheduler.tasks[task_id]
                    task.status = 'completed'
                    task.progress = 1.0
                    task.completion_time = self.env.current_time if hasattr(self.env, 'current_time') else 0
                
                status['status'] = 'idle'
                status['current_task'] = None
                self.env.vehicles[vehicle_id]['status'] = 'idle'
                
                if self.traffic_manager:
                    self.traffic_manager.release_vehicle_path(vehicle_id)
                
                self.log(f"已取消车辆 {vehicle_id} 的所有任务", "success")
            else:
                self.log(f"车辆 {vehicle_id} 没有活动任务", "warning")
    
    def return_to_start(self):
        """返回起点"""
        if not self.vehicle_scheduler:
            return
        
        vehicle_id = self.vehicle_info_panel.vehicle_combo.itemData(
            self.vehicle_info_panel.vehicle_combo.currentIndex()
        )
        if vehicle_id is None:
            self.log("请先选择一个车辆", "warning")
            return
        
        if vehicle_id not in self.env.vehicles:
            return
        
        vehicle = self.env.vehicles[vehicle_id]
        initial_position = vehicle.get('initial_position')
        
        if not initial_position:
            self.log(f"车辆 {vehicle_id} 没有初始位置", "error")
            return
        
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
        
        self.vehicle_scheduler.tasks[task_id] = task
        self.vehicle_scheduler.task_queues[vehicle_id] = [task_id]
        
        status = self.vehicle_scheduler.vehicle_statuses[vehicle_id]
        if status['status'] == 'idle':
            self.vehicle_scheduler._start_next_task(vehicle_id)
        
        self.log(f"已命令车辆 {vehicle_id} 返回起点", "success")
    
    # 显示控制方法
    def update_path_display(self):
        """更新路径显示"""
        show_backbone = self.show_paths_cb.isChecked()
        self.graphics_view.set_show_backbone(show_backbone)
        self.show_backbone_action.setChecked(show_backbone)
    
    def update_vehicle_display(self):
        """更新车辆显示"""
        show_trajectories = self.show_vehicle_paths_cb.isChecked()
        self.graphics_view.set_show_trajectories(show_trajectories)
    
    def toggle_backbone_display(self, checked):
        """切换骨干网络显示"""
        self.graphics_view.set_show_backbone(checked)
        self.show_paths_cb.setChecked(checked)
    
    def toggle_vehicles_display(self, checked):
        """切换车辆显示"""
        self.show_vehicles_cb.setChecked(checked)
        self.update_vehicle_display()
    
    def apply_ecbs_settings(self):
        """应用ECBS设置"""
        if not self.traffic_manager:
            self.log("交通管理器不可用", "warning")
            return
        
        settings = self.ecbs_panel.get_settings()
        subopt = settings["suboptimality"]
        strategy = settings["strategy"]
        
        if hasattr(self.traffic_manager, 'ecbs_solver'):
            self.traffic_manager.ecbs_solver.suboptimality_bound = subopt
        
        if isinstance(self.vehicle_scheduler, ECBSVehicleScheduler):
            if hasattr(self.vehicle_scheduler, 'conflict_resolution_strategy'):
                self.vehicle_scheduler.conflict_resolution_strategy = strategy.lower()
        
        self.log(f"已应用ECBS设置: 界限={subopt}, 策略={strategy}", "success")
    
    def optimize_paths(self):
        """优化路径"""
        if not self.backbone_network or not self.backbone_network.paths:
            self.log("请先生成骨干路径网络", "warning")
            return
        
        try:
            self.log("正在优化路径...")
            
            # 重新优化所有路径
            self.backbone_network._optimize_all_paths_advanced()
            
            # 更新路径信息
            self.update_path_info()
            
            # 重新绘制
            self.graphics_view.set_backbone_network(self.backbone_network)
            
            self.log("路径优化完成", "success")
            
        except Exception as e:
            self.log(f"路径优化失败: {str(e)}", "error")
    
    # 视图控制方法
    def zoom_in(self):
        """放大"""
        self.graphics_view.scale(1.2, 1.2)
    
    def zoom_out(self):
        """缩小"""
        self.graphics_view.scale(1/1.2, 1/1.2)
    
    def fit_view(self):
        """适应视图"""
        self.graphics_view.fitInView(
            self.graphics_view.mine_scene.sceneRect(), 
            Qt.KeepAspectRatio
        )
    
    def enable_controls(self, enabled):
        """启用/禁用控件"""
        self.start_button.setEnabled(enabled)
        self.reset_button.setEnabled(enabled)
        self.start_sim_action.setEnabled(enabled)
        self.reset_sim_action.setEnabled(enabled)
        self.generate_paths_button.setEnabled(enabled)
        self.assign_task_button.setEnabled(enabled)
        self.cancel_task_button.setEnabled(enabled)
        self.goto_loading_button.setEnabled(enabled)
        self.goto_unloading_button.setEnabled(enabled)
        self.return_button.setEnabled(enabled)
    
    def save_results(self):
        """保存结果"""
        if not self.env:
            self.log("没有结果可保存", "warning")
            return
        
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "保存结果",
            f"simulation_result_{time.strftime('%Y%m%d_%H%M%S')}",
            "PNG图像 (*.png);;JSON数据 (*.json);;所有文件 (*)"
        )
        
        if not file_path:
            return
        
        try:
            if file_path.endswith('.png'):
                # 保存截图
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
                # 保存数据
                data = {
                    'time': self.env.current_time if hasattr(self.env, 'current_time') else 0,
                    'vehicles': {},
                    'tasks': {},
                    'stats': {}
                }
                
                for vehicle_id, vehicle in self.env.vehicles.items():
                    data['vehicles'][vehicle_id] = {
                        'position': vehicle.get('position'),
                        'status': vehicle.get('status'),
                        'load': vehicle.get('load'),
                        'completed_cycles': vehicle.get('completed_cycles', 0)
                    }
                
                if self.vehicle_scheduler:
                    for task_id, task in self.vehicle_scheduler.tasks.items():
                        data['tasks'][task_id] = task.to_dict()
                    
                    data['stats'] = self.vehicle_scheduler.get_stats()
                
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
            <p>基于骨干路径网络的多车辆规划与调度 - 优化版</p>
            <p>版本: 2.0</p>
            <hr>
            <h3>主要特性:</h3>
            <ul style="text-align:left;">
                <li>优化的骨干路径网络生成与管理</li>
                <li>智能路径规划与质量评估</li>
                <li>ECBS冲突检测与解决</li>
                <li>增强的车辆调度与任务管理</li>
                <li>实时性能监控与可视化</li>
                <li>多层次的用户交互界面</li>
            </ul>
            <hr>
            <p>开发者: 矿山智能调度团队</p>
            <p>版权所有 © 2024</p>
        </div>
        """
        
        QMessageBox.about(self, "关于", about_text)
    
    # ✅ 新增: 调试方法
    def debug_task_assignment(self, vehicle_id, task):
        """调试任务分配"""
        print(f"=== 任务分配调试信息 ===")
        print(f"车辆ID: {vehicle_id}")
        print(f"任务ID: {task.task_id}")
        print(f"任务类型: {task.task_type}")
        print(f"装载点ID: {task.loading_point_id}")
        print(f"卸载点ID: {task.unloading_point_id}")
        print(f"目标位置: {task.goal}")
        print(f"========================")
    
    def log(self, message, level="info"):
        """添加日志消息"""
        current_time = time.strftime("%H:%M:%S")
        
        if level == "error":
            color = "red"
        elif level == "warning":
            color = "orange"
        elif level == "success":
            color = "green"
        else:
            color = "black"
        
        formatted_message = f'<span style="color: {color};">[{current_time}] {message}</span>'
        
        self.log_text.append(formatted_message)
        
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        self.statusBar().showMessage(message)
    
    def clear_log(self):
        """清除日志"""
        self.log_text.clear()

    def debug_backbone_system(self):
        """调试骨干网络系统 - 增强版"""
        self.log("🔍 开始骨干网络系统诊断...", "info")
        
        try:
            # 临时启用调试模式
            if self.backbone_network:
                self.backbone_network.debug = True
            
            if self.path_planner:
                self.path_planner.debug = True
            
            debugger = BackboneNetworkDebugger(self)
            debugger.debug_full_system()
            
            self.log("🏁 系统诊断完成，请查看控制台输出", "success")
            
        except Exception as e:
            self.log(f"❌ 系统诊断出错: {e}", "error")
            import traceback
            print(f"诊断错误详情:\n{traceback.format_exc()}")
        
        finally:
            # 关闭调试模式
            if self.backbone_network:
                self.backbone_network.debug = False
            
            if self.path_planner:
                self.path_planner.debug = False

    
    def test_single_path_planning(self):
        """测试单次路径规划 - 增强调试版"""
        if not self.env or not self.path_planner:
            self.log("❌ 系统未初始化", "error")
            return
        
        if not self.env.vehicles:
            self.log("❌ 没有车辆", "error")
            return
        
        vehicle_id = list(self.env.vehicles.keys())[0]
        vehicle = self.env.vehicles[vehicle_id]
        start = vehicle['position']
        
        if not self.env.loading_points:
            self.log("❌ 没有装载点", "error")
            return
        
        goal = self.env.loading_points[0]
        
        self.log(f"🧪 测试车辆{vehicle_id}路径规划", "info")
        self.log(f"  起点: {start}", "info")
        self.log(f"  终点: {goal}", "info")
        
        # 启用详细调试
        original_debug = getattr(self.path_planner, 'debug', False)
        original_verbose = getattr(self.path_planner, 'verbose_logging', False)
        
        try:
            # 启用调试模式
            self.path_planner.debug = True
            self.path_planner.verbose_logging = True
            
            # 如果骨干网络存在，也启用其调试
            if self.backbone_network:
                self.backbone_network.debug = True
            
            # 限制测试次数，避免无限循环
            max_test_attempts = 3
            
            for attempt in range(max_test_attempts):
                self.log(f"🔄 第 {attempt + 1} 次尝试", "info")
                
                try:
                    result = self.path_planner.plan_path(
                        vehicle_id, start, goal, 
                        use_backbone=True, 
                        max_attempts=1  # 限制每次内部尝试次数
                    )
                    
                    if result:
                        path, structure = result if isinstance(result, tuple) else (result, {})
                        path_type = structure.get('type', 'unknown') if structure else 'unknown'
                        
                        self.log(f"✅ 第{attempt + 1}次规划成功，类型: {path_type}", "success")
                        
                        if path_type == 'three_segment':
                            self.log("🎉 成功使用骨干网络!", "success")
                            # 测试成功，退出循环
                            break
                        elif path_type == 'direct':
                            self.log("⚠️ 使用直接路径，未使用骨干网络", "warning")
                            if attempt < max_test_attempts - 1:
                                continue  # 尝试下一次
                            else:
                                break
                        else:
                            self.log(f"ℹ️ 使用了 {path_type} 路径", "info")
                            break
                    else:
                        self.log(f"❌ 第{attempt + 1}次规划失败", "error")
                        if attempt < max_test_attempts - 1:
                            self.log("🔄 准备重试...", "info")
                        else:
                            self.log("❌ 所有尝试都失败了", "error")
                
                except Exception as e:
                    self.log(f"❌ 第{attempt + 1}次规划出错: {e}", "error")
                    if attempt < max_test_attempts - 1:
                        self.log("🔄 准备重试...", "info")
                    else:
                        self.log("❌ 所有尝试都出错了", "error")
                    
                    # 打印详细错误信息
                    import traceback
                    error_details = traceback.format_exc()
                    print(f"详细错误信息:\n{error_details}")
        
        finally:
            # 恢复原始设置
            self.path_planner.debug = original_debug
            self.path_planner.verbose_logging = original_verbose
            
            if self.backbone_network:
                self.backbone_network.debug = False
# 保持向后兼容性
MineGUI = OptimizedMineGUI



def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle("Fusion")
    
    # 创建并显示主窗口
    window = OptimizedMineGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()