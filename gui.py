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

# 导入其他项目组件
from backbone_network import BackbonePathNetwork
from path_planner import PathPlanner
from traffic_manager import TrafficManager, Conflict
from vehicle_scheduler import VehicleScheduler, VehicleTask, ECBSVehicleScheduler
from environment import OpenPitMineEnv
from path_utils import improved_visualize_environment

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

class VehicleGraphicsItem(QGraphicsItemGroup):
    """车辆图形项 - 高级版"""
    def __init__(self, vehicle_id, vehicle_data, parent=None):
        super().__init__(parent)
        self.vehicle_id = vehicle_id
        self.vehicle_data = vehicle_data
        self.position = vehicle_data['position']
        self.vehicle_length = 6.0  # 车辆长度
        self.vehicle_width = 3.0   # 车辆宽度
        
        # 创建车辆图形
        self.vehicle_body = QGraphicsPolygonItem(self)
        self.vehicle_label = QGraphicsTextItem(str(vehicle_id), self)
        self.status_label = QGraphicsTextItem("", self)
        
        # 设置车辆颜色
        status = vehicle_data.get('status', 'idle')
        color = VEHICLE_COLORS.get(status, VEHICLE_COLORS['idle'])
        
        # 设置样式
        self.vehicle_body.setBrush(QBrush(color))
        self.vehicle_body.setPen(QPen(Qt.black, 0.5))
        
        # 改进标签样式
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
        
        # 设置Z值，确保车辆显示在上层
        self.setZValue(10)
        
        # 更新车辆位置和朝向
        self.update_position()
    
    def update_position(self):
        """更新车辆位置和朝向 - 修复版"""
        # 确保position是一个有效的元组，包含三个值：x, y, theta
        if not self.position or len(self.position) < 3:
            return
            
        x, y, theta = self.position
        
        # 创建车辆多边形
        polygon = QPolygonF()
        
        # 计算车辆四个角点的相对坐标 - 修改为默认沿Y轴方向
        half_length = self.vehicle_length / 2
        half_width = self.vehicle_width / 2
        corners_relative = [
            QPointF(half_length, half_width),     # 前右
            QPointF(half_length, -half_width),    # 前左
            QPointF(-half_length, -half_width),   # 后左
            QPointF(-half_length, half_width)     # 后右
        ]
        
        # 创建变换矩阵
        transform = QTransform()
        transform.rotate(theta * 180 / math.pi)  # 旋转(角度制)
        
        # 应用旋转，然后添加到多边形
        for corner in corners_relative:
            rotated_corner = transform.map(corner)
            polygon.append(QPointF(x + rotated_corner.x(), y + rotated_corner.y()))
        
        # 设置多边形
        self.vehicle_body.setPolygon(polygon)
        
        # 更新标签位置 - 固定偏移，确保位置一致
        self.vehicle_label.setPos(x - 1.5, y - 1.5)
        
        # 状态标签显示在车辆下方
        self.status_label.setPos(x - 5, y + 3.5)

    def update_data(self, vehicle_data):
        """更新车辆数据"""
        self.vehicle_data = vehicle_data
        
        # 保存旧位置用于调试
        old_position = self.position
        
        # 更新位置
        self.position = vehicle_data['position']
        
        # 更新状态标签
        status = vehicle_data.get('status', 'idle')
        status_text = {
            'idle': '空闲',
            'loading': '装载中',
            'unloading': '卸载中',
            'moving': '移动中'
        }.get(status, '')
        
        # 根据车辆状态更新颜色
        color = VEHICLE_COLORS.get(status, VEHICLE_COLORS['idle'])
        
        self.vehicle_body.setBrush(QBrush(color))
        self.status_label.setPlainText(status_text)
        
        # 更新车辆位置
        self.update_position()


class BackbonePathVisualizer(QGraphicsItemGroup):
    """骨干路径网络可视化组件 - 美化版"""
    
    def __init__(self, backbone_network, parent=None):
        """初始化骨干网络可视化器
        
        参数:
            backbone_network: 骨干网络对象
            parent: 父级QGraphicsItem
        """
        super().__init__(parent)
        self.backbone_network = backbone_network
        self.path_items = {}
        self.connection_items = {}
        self.node_items = {}
        self.setZValue(1)  # 确保显示在适当层
        self.update_visualization()
    
    def update_visualization(self):
        """更新骨干网络可视化"""
        # 清除现有项目
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
        
        # 绘制骨干路径
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
            
            # 设置渐变颜色风格
            gradient = QLinearGradient(path[0][0], path[0][1], path[-1][0], path[-1][1])
            gradient.setColorAt(0, QColor(40, 120, 180, 180))  # 起点颜色
            gradient.setColorAt(1, QColor(120, 40, 180, 180))  # 终点颜色
            
            pen = QPen(gradient, 0.5)  # 粗线路径线
            path_item.setPen(pen)
            
            # 添加到组
            self.addToGroup(path_item)
            self.path_items[path_id] = path_item
        
        # 绘制连接点
        for conn_id, conn_data in self.backbone_network.connections.items():
            position = conn_data['position']
            conn_type = conn_data.get('type', 'midpath')
            
            # 不同类型不同样式
            if conn_type == 'endpoint':
                # 终点连接，较大
                radius = 1.5
                color = QColor(220, 120, 40)  # 橙色
            else:
                # 中间路径连接，较小
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
        
        # 绘制节点（装载点、卸载点等）
        # 获取所有路径的起点和终点
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
            
            # 不同类型不同样式
            if node_type == 'loading_point':
                # 装载点
                radius = 3.0
                color = QColor(0, 180, 0)  # 绿色
            elif node_type == 'unloading_point':
                # 卸载点
                radius = 3.0
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
            node_item.setPen(QPen(Qt.black, 0.5))
            
            # 添加标签
            label_item = QGraphicsTextItem(node_id)
            label_item.setPos(position[0] + radius + 1, position[1] - radius - 2)
            label_item.setFont(QFont("Arial", 3))
            
            # 添加到组
            self.addToGroup(node_item)
            self.addToGroup(label_item)
            self.node_items[node_id] = node_item
    
    def update_traffic_flow(self):
        """更新交通流可视化 - 显示ECBS路径使用情况"""
        if not self.backbone_network:
            return
        
        # 根据交通流更新路径
        for path_id, path_item in self.path_items.items():
            if path_id in self.backbone_network.paths:
                path_data = self.backbone_network.paths[path_id]
                traffic_flow = path_data.get('traffic_flow', 0)
                capacity = path_data.get('capacity', 1)
                
                # 计算流量比例
                ratio = min(1.0, traffic_flow / max(1, capacity))
                
                # 根据交通流调整线宽和颜色
                width = 1.0 + ratio * 3.0  # 1-4范围
                
                # 颜色从绿到黄到红
                if ratio < 0.5:
                    # 绿到黄
                    r = int(255 * ratio * 2)
                    g = 255
                    b = 0
                else:
                    # 黄到红
                    r = 255
                    g = int(255 * (2 - ratio * 2))
                    b = 0
                
                color = QColor(r, g, b, 180)
                
                path_item.setPen(QPen(color, width))


class PathGraphicsItem(QGraphicsPathItem):
    """路径图形项 - 优化版，区分不同路径部分"""
    def __init__(self, path, parent=None, vehicle_id=None, path_structure=None):
        super().__init__(parent)
        self.path_data = path
        self.vehicle_id = vehicle_id
        self.path_structure = path_structure  # 新增: 路径结构信息
        self.path_segments = []  # 存储路径段图形项
        self.update_path()
        
        # 根据路径结构使用不同的样式
        if path_structure and all(k in path_structure for k in ['to_backbone_path', 'backbone_path', 'from_backbone_path']):
            # 创建结构化路径，显示不同段
            self.create_structured_path()
        else:
            # 使用默认样式
            gradient = QLinearGradient(0, 0, 100, 100)
            
            # 使用与车辆颜色匹配的渐变
            if vehicle_id is not None:
                gradient.setColorAt(0, QColor(0, 190, 255, 220))  # 起点为亮蓝色
                gradient.setColorAt(1, QColor(255, 100, 100, 220))  # 终点为红色
            else:
                gradient.setColorAt(0, QColor(0, 200, 100, 180))  # 起点为绿色
                gradient.setColorAt(1, QColor(200, 0, 100, 180))  # 终点为紫色
            
            # 改进线条样式
            pen = QPen(gradient, 0.5)
            pen.setStyle(Qt.DashLine)
            pen.setDashPattern([5, 3])
            self.setPen(pen)
        
        # 设置Z值，确保路径显示在车辆下方
        self.setZValue(5)
        
        # 添加路径点标记以便于调试
        self.path_points = []
        self.add_path_point_markers()
    
    def create_structured_path(self):
        """创建结构化路径 - 不同部分使用不同样式"""
        if not self.path_structure:
            return
        
        # 清除现有路径段
        for item in self.path_segments:
            if item.scene():
                item.scene().removeItem(item)
        self.path_segments = []
            
        # 提取路径各部分
        to_backbone = self.path_structure.get('to_backbone_path')
        backbone = self.path_structure.get('backbone_path')
        from_backbone = self.path_structure.get('from_backbone_path')
        
        # 绘制从起点到骨干网络的路径（虚线，亮蓝色）
        if to_backbone and len(to_backbone) > 1:
            # 创建路径
            painter_path = QPainterPath()
            painter_path.moveTo(to_backbone[0][0], to_backbone[0][1])
            
            for point in to_backbone[1:]:
                painter_path.lineTo(point[0], point[1])
                
            # 设置样式
            pen_to_backbone = QPen(QColor(0, 200, 255, 220), 0.5)
            pen_to_backbone.setStyle(Qt.DashLine)
            pen_to_backbone.setDashPattern([5, 3])
            
            # 创建路径项
            to_backbone_item = QGraphicsPathItem(painter_path)
            to_backbone_item.setPen(pen_to_backbone)
            to_backbone_item.setZValue(5)
            
            if self.scene():
                self.scene().addItem(to_backbone_item)
                self.path_segments.append(to_backbone_item)
        
        # 绘制骨干网络路径（实线，粗，绿色）
        if backbone and len(backbone) > 1:
            # 创建路径
            painter_path = QPainterPath()
            painter_path.moveTo(backbone[0][0], backbone[0][1])
            
            for point in backbone[1:]:
                painter_path.lineTo(point[0], point[1])
                
            # 设置样式
            pen_backbone = QPen(QColor(50, 180, 50, 220), 1.0)  # 粗一点，更明显
            
            # 创建路径项
            backbone_item = QGraphicsPathItem(painter_path)
            backbone_item.setPen(pen_backbone)
            backbone_item.setZValue(5)
            
            if self.scene():
                self.scene().addItem(backbone_item)
                self.path_segments.append(backbone_item)
        
        # 绘制从骨干网络到终点的路径（虚线，红色）
        if from_backbone and len(from_backbone) > 1:
            # 创建路径
            painter_path = QPainterPath()
            painter_path.moveTo(from_backbone[0][0], from_backbone[0][1])
            
            for point in from_backbone[1:]:
                painter_path.lineTo(point[0], point[1])
                
            # 设置样式
            pen_from_backbone = QPen(QColor(255, 100, 100, 220), 0.5)
            pen_from_backbone.setStyle(Qt.DashLine)
            pen_from_backbone.setDashPattern([5, 3])
            
            # 创建路径项
            from_backbone_item = QGraphicsPathItem(painter_path)
            from_backbone_item.setPen(pen_from_backbone)
            from_backbone_item.setZValue(5)
            
            if self.scene():
                self.scene().addItem(from_backbone_item)
                self.path_segments.append(from_backbone_item)

class MineGraphicsScene(QGraphicsScene):
    """矿场图形场景 - 优化版"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.env = None
        self.planner = None
        self.show_trajectories = True
        self.grid_size = 10.0
        
        # 存储所有图形项
        self.grid_items = []
        self.obstacle_items = []
        self.loading_point_items = []
        self.unloading_point_items = []
        self.vehicle_items = {}
        self.path_items = {}
        
        # 设置场景大小
        self.setSceneRect(0, 0, 500, 500)
        
        # 场景样式
        self.background_color = QColor(245, 245, 250)  # 更柔和的背景色
    
    def set_environment(self, env, planner=None):
        """设置环境"""
        self.env = env
        self.planner = planner
        
        # 更新场景大小
        width, height = env.width, env.height
        self.setSceneRect(0, 0, width, height)
        
        # 清除所有图形项
        self.clear()
        self.grid_items = []
        self.obstacle_items = []
        self.loading_point_items = []
        self.unloading_point_items = []
        self.vehicle_items = {}
        self.path_items = {}
        
        # 绘制背景网格
        self.draw_grid()
        
        # 绘制障碍物
        self.draw_obstacles()
        
        # 绘制装载点和卸载点
        self.draw_loading_points()
        self.draw_unloading_points()
        
        # 绘制车辆
        self.draw_vehicles()
    
    def draw_grid(self):
        """绘制背景网格 - 美化版"""
        if not self.env:
            return
            
        width, height = self.env.width, self.env.height
        grid_size = self.grid_size
        
        # 创建背景矩形
        background = QGraphicsRectItem(0, 0, width, height)
        background.setBrush(QBrush(self.background_color))
        background.setPen(QPen(Qt.NoPen))
        background.setZValue(-100)  # 确保在最底层
        self.addItem(background)
        self.grid_items.append(background)
        
        # 绘制网格线 - 使用更柔和的颜色
        major_pen = QPen(QColor(210, 210, 220))
        major_pen.setWidth(0)
        
        minor_pen = QPen(QColor(230, 230, 240))
        minor_pen.setWidth(0)
        
        # 垂直线 - 主网格线和次网格线
        for x in range(0, width + 1, int(grid_size)):
            if x % (int(grid_size * 5)) == 0:
                # 主网格线
                line = QGraphicsLineItem(x, 0, x, height)
                line.setPen(major_pen)
                line.setZValue(-90)
                self.addItem(line)
                self.grid_items.append(line)
            else:
                # 次网格线
                line = QGraphicsLineItem(x, 0, x, height)
                line.setPen(minor_pen)
                line.setZValue(-95)
                self.addItem(line)
                self.grid_items.append(line)
        
        # 水平线 - 主网格线和次网格线
        for y in range(0, height + 1, int(grid_size)):
            if y % (int(grid_size * 5)) == 0:
                # 主网格线
                line = QGraphicsLineItem(0, y, width, y)
                line.setPen(major_pen)
                line.setZValue(-90)
                self.addItem(line)
                self.grid_items.append(line)
            else:
                # 次网格线
                line = QGraphicsLineItem(0, y, width, y)
                line.setPen(minor_pen)
                line.setZValue(-95)
                self.addItem(line)
                self.grid_items.append(line)
        
        # 添加坐标轴标签 - 每隔一定距离显示一个标签
        label_interval = 50
        label_color = QColor(100, 100, 120)
        
        # X轴标签
        for x in range(0, width + 1, label_interval):
            label = QGraphicsTextItem(str(x))
            label.setPos(x, 0)
            label.setDefaultTextColor(label_color)
            label.setFont(QFont("Arial", 6))
            label.setZValue(-80)
            self.addItem(label)
            self.grid_items.append(label)
        
        # Y轴标签
        for y in range(0, height + 1, label_interval):
            label = QGraphicsTextItem(str(y))
            label.setPos(0, y)
            label.setDefaultTextColor(label_color)
            label.setFont(QFont("Arial", 6))
            label.setZValue(-80)
            self.addItem(label)
            self.grid_items.append(label)
    
    def draw_obstacles(self):
        """绘制障碍物 - 美化版"""
        if not self.env:
            return
        
        # 障碍物颜色 - 使用渐变填充
        obstacle_gradient = QLinearGradient(0, 0, 10, 10)
        obstacle_gradient.setColorAt(0, QColor(80, 80, 90))
        obstacle_gradient.setColorAt(1, QColor(60, 60, 70))
        
        obstacle_brush = QBrush(obstacle_gradient)
        obstacle_pen = QPen(QColor(40, 40, 50), 0.2)
        
        # 获取障碍物列表
        obstacles = self.env._get_obstacle_list()
        
        for obstacle in obstacles:
            x, y = obstacle['x'], obstacle['y']
            width, height = obstacle['width'], obstacle['height']
            
            # 创建矩形
            rect = QGraphicsRectItem(x, y, width, height)
            rect.setBrush(obstacle_brush)
            rect.setPen(obstacle_pen)
            rect.setZValue(-50)  # 确保在网格之上，车辆之下
            self.addItem(rect)
            self.obstacle_items.append(rect)
    
    def draw_loading_points(self):
        """绘制装载点 - 美化版"""
        if not self.env:
            return
            
        for i, point in enumerate(self.env.loading_points):
            x, y = point[0], point[1]
            
            # 创建更美观的装载点图形
            
            # 1. 外发光效果
            glow_radius = 12
            glow = QGraphicsEllipseItem(x - glow_radius/2, y - glow_radius/2, glow_radius, glow_radius)
            gradient = QRadialGradient(x, y, glow_radius/2)
            gradient.setColorAt(0, QColor(0, 200, 0, 100))
            gradient.setColorAt(1, QColor(0, 150, 0, 0))
            glow.setBrush(QBrush(gradient))
            glow.setPen(QPen(Qt.NoPen))
            glow.setZValue(-25)
            self.addItem(glow)
            self.loading_point_items.append(glow)
            
            # 2. 装载区域
            area_radius = 8
            area = QGraphicsEllipseItem(x - area_radius/2, y - area_radius/2, area_radius, area_radius)
            area.setBrush(QBrush(QColor(200, 255, 200, 120)))  # 浅绿色，半透明
            area.setPen(QPen(QColor(0, 120, 0), 0.5))
            area.setZValue(-20)
            self.addItem(area)
            self.loading_point_items.append(area)
            
            # 3. 中心标记
            center_radius = 4
            center = QGraphicsEllipseItem(x - center_radius/2, y - center_radius/2, center_radius, center_radius)
            center_gradient = QRadialGradient(x, y, center_radius/2)
            center_gradient.setColorAt(0, QColor(100, 200, 100))
            center_gradient.setColorAt(1, QColor(0, 150, 0))
            center.setBrush(QBrush(center_gradient))
            center.setPen(QPen(Qt.black, 0.5))
            center.setZValue(-10)
            self.addItem(center)
            self.loading_point_items.append(center)
            
            # 4. 添加标签
            text = QGraphicsTextItem(f"装载点{i+1}")
            text.setPos(x - 12, y - 15)
            text.setDefaultTextColor(QColor(0, 100, 0))
            text.setFont(QFont("SimHei", 3, QFont.Bold))
            text.setZValue(-10)
            self.addItem(text)
            self.loading_point_items.append(text)
    
    def draw_unloading_points(self):
        """绘制卸载点 - 美化版"""
        if not self.env:
            return
            
        for i, point in enumerate(self.env.unloading_points):
            x, y = point[0], point[1]
            
            # 创建更美观的卸载点图形
            
            # 1. 外发光效果
            glow_radius = 12
            glow = QGraphicsEllipseItem(x - glow_radius/2, y - glow_radius/2, glow_radius, glow_radius)
            gradient = QRadialGradient(x, y, glow_radius/2)
            gradient.setColorAt(0, QColor(200, 0, 0, 100))
            gradient.setColorAt(1, QColor(150, 0, 0, 0))
            glow.setBrush(QBrush(gradient))
            glow.setPen(QPen(Qt.NoPen))
            glow.setZValue(-25)
            self.addItem(glow)
            self.unloading_point_items.append(glow)
            
            # 2. 卸载区域
            area_radius = 8
            area = QGraphicsEllipseItem(x - area_radius/2, y - area_radius/2, area_radius, area_radius)
            area.setBrush(QBrush(QColor(255, 200, 200, 120)))  # 浅红色，半透明
            area.setPen(QPen(QColor(120, 0, 0), 0.5))
            area.setZValue(-20)
            self.addItem(area)
            self.unloading_point_items.append(area)
            
            # 3. 中心标记 - 使用方形区分
            center_size = 4
            center = QGraphicsRectItem(x - center_size/2, y - center_size/2, center_size, center_size)
            center.setBrush(QBrush(QColor(200, 50, 50)))
            center.setPen(QPen(Qt.black, 0.5))
            center.setZValue(-10)
            self.addItem(center)
            self.unloading_point_items.append(center)
            
            # 4. 添加标签
            text = QGraphicsTextItem(f"卸载点{i+1}")
            text.setPos(x - 12, y - 15)
            text.setDefaultTextColor(QColor(120, 0, 0))
            text.setFont(QFont("SimHei", 3, QFont.Bold))
            text.setZValue(-10)
            self.addItem(text)
            self.unloading_point_items.append(text)
    
    def draw_vehicles(self):
        """绘制车辆 - 优化版"""
        if not self.env:
            return
            
        # 清除现有车辆图形
        for item in self.vehicle_items.values():
            self.removeItem(item)
        self.vehicle_items.clear()
        
        # 清除路径图形
        for item in self.path_items.values():
            self.removeItem(item)
        self.path_items.clear()
        
        # 添加新车辆图形
        for vehicle_id, vehicle_data in self.env.vehicles.items():
            # 创建车辆图形项
            vehicle_item = VehicleGraphicsItem(vehicle_id, vehicle_data)
            self.addItem(vehicle_item)
            self.vehicle_items[vehicle_id] = vehicle_item
            
            # 如果需要显示轨迹，并且有路径
            if self.show_trajectories and 'path' in vehicle_data and vehicle_data['path']:
                path = vehicle_data['path']
                path_item = PathGraphicsItem(path, vehicle_id=vehicle_id)
                self.addItem(path_item)
                self.path_items[vehicle_id] = path_item
    
    def update_vehicles(self):
        """更新车辆位置和状态 - 优化版"""
        if not self.env:
            return
            
        for vehicle_id, vehicle_data in self.env.vehicles.items():
            # 更新车辆图形
            if vehicle_id in self.vehicle_items:
                # 更新现有车辆
                self.vehicle_items[vehicle_id].update_data(vehicle_data)
            else:
                # 如果不存在，创建新的
                vehicle_item = VehicleGraphicsItem(vehicle_id, vehicle_data)
                self.addItem(vehicle_item)
                self.vehicle_items[vehicle_id] = vehicle_item
            
            # 更新路径
            if self.show_trajectories and 'path' in vehicle_data and vehicle_data['path']:
                path = vehicle_data['path']
                
                if vehicle_id in self.path_items:
                    # 移除旧路径
                    self.removeItem(self.path_items[vehicle_id])
                
                # 创建新路径
                path_item = PathGraphicsItem(path, vehicle_id=vehicle_id)
                self.addItem(path_item)
                self.path_items[vehicle_id] = path_item
            elif vehicle_id in self.path_items and not self.show_trajectories:
                # 如果不显示轨迹，移除路径
                self.removeItem(self.path_items[vehicle_id])
                del self.path_items[vehicle_id]
    
    def set_show_trajectories(self, show):
        """设置是否显示轨迹"""
        if self.show_trajectories != show:
            self.show_trajectories = show
            
            # 更新轨迹显示
            if show:
                # 显示所有轨迹
                for vehicle_id, vehicle_data in self.env.vehicles.items():
                    if 'path' in vehicle_data and vehicle_data['path']:
                        path = vehicle_data['path']
                        path_item = PathGraphicsItem(path, vehicle_id=vehicle_id)
                        self.addItem(path_item)
                        self.path_items[vehicle_id] = path_item
            else:
                # 隐藏所有轨迹
                for item in self.path_items.values():
                    self.removeItem(item)
                self.path_items.clear()


class MineGraphicsView(QGraphicsView):
    """矿场图形视图 - 美化版"""
    def __init__(self, parent=None):
        super().__init__(parent)
        # 启用抗锯齿
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setRenderHint(QPainter.TextAntialiasing, True)
        
        # 启用鼠标追踪
        self.setMouseTracking(True)
        
        # 启用拖拽和缩放
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        
        # 设置视图更新模式
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        
        # 关闭优化标志以确保精确绘制
        self.setOptimizationFlags(QGraphicsView.DontAdjustForAntialiasing)
        
        # 创建场景
        self.mine_scene = MineGraphicsScene(self)
        self.setScene(self.mine_scene)
        
        # 设置背景色
        self.setBackgroundBrush(QBrush(QColor(245, 245, 250)))
        
        # 添加坐标显示标签
        self.coord_label = QLabel(self)
        self.coord_label.setStyleSheet("background-color: rgba(255, 255, 255, 180); padding: 5px; border-radius: 3px;")
        self.coord_label.setAlignment(Qt.AlignCenter)
        self.coord_label.setFixedSize(150, 25)
        self.coord_label.move(10, 10)
        self.coord_label.show()
    
    def wheelEvent(self, event):
        """鼠标滚轮事件 - 用于缩放"""
        factor = 1.2
        
        if event.angleDelta().y() < 0:
            # 缩小
            factor = 1.0 / factor
            
        self.scale(factor, factor)
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件 - 显示坐标"""
        super().mouseMoveEvent(event)
        
        # 获取场景坐标
        scene_pos = self.mapToScene(event.pos())
        x, y = scene_pos.x(), scene_pos.y()
        
        # 更新坐标显示
        self.coord_label.setText(f"X: {x:.1f}, Y: {y:.1f}")
    
    def set_environment(self, env, planner=None):
        """设置环境"""
        self.mine_scene.set_environment(env, planner)
        
        # 调整视图以显示整个场景
        self.fitInView(self.mine_scene.sceneRect(), Qt.KeepAspectRatio)
    
    def update_vehicles(self):
        """更新车辆位置和状态"""
        self.mine_scene.update_vehicles()
        
        # 确保所有更改都被绘制
        self.viewport().update()
    
    def set_show_trajectories(self, show):
        """设置是否显示轨迹"""
        self.mine_scene.set_show_trajectories(show)
        
        # 更新视图
        self.viewport().update()
    
    def resizeEvent(self, event):
        """窗口大小改变事件"""
        super().resizeEvent(event)
        
        # 调整视图，确保场景正确显示
        if self.mine_scene and not self.mine_scene.sceneRect().isEmpty():
            self.fitInView(self.mine_scene.sceneRect(), Qt.KeepAspectRatio)
        
        # 调整坐标标签位置
        self.coord_label.move(10, 10)


class VehicleInfoPanel(QWidget):
    """车辆信息面板 - 美化版"""
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 创建布局
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(10)
        
        # 标题
        self.title_label = QLabel("车辆信息")
        self.title_label.setFont(QFont("SimHei", 12, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("color: #2C3E50; margin-bottom: 10px;")
        self.layout.addWidget(self.title_label)
        
        # 添加分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        self.layout.addWidget(separator)
        
        # 车辆选择区域
        self.vehicle_group = QGroupBox("选择车辆")
        self.vehicle_layout = QVBoxLayout()
        
        self.vehicle_combo = QComboBox()
        self.vehicle_combo.setMinimumHeight(30)
        self.vehicle_combo.currentIndexChanged.connect(self.update_vehicle_info)
        self.vehicle_layout.addWidget(self.vehicle_combo)
        
        self.vehicle_group.setLayout(self.vehicle_layout)
        self.layout.addWidget(self.vehicle_group)
        
        # 车辆详情
        self.info_group = QGroupBox("详细信息")
        self.info_layout = QGridLayout()
        self.info_layout.setColumnStretch(1, 1)  # 让值列有更多空间
        self.info_layout.setVerticalSpacing(8)
        
        # 添加信息标签
        self.labels = {}
        self.values = {}
        
        info_fields = [
            ("id", "车辆ID:"),
            ("position", "当前位置:"),
            ("status", "状态:"),
            ("load", "载重:"),
            ("task", "当前任务:"),
            ("goal", "目标点:"),
            ("completed", "已完成循环:")
        ]
        
        # 添加状态指示器
        self.status_indicators = {}
        status_colors = {
            'idle': '#6c757d',      # 灰色
            'moving': '#007bff',    # 蓝色
            'loading': '#28a745',   # 绿色
            'unloading': '#dc3545'  # 红色
        }
        
        for i, (field, label) in enumerate(info_fields):
            self.labels[field] = QLabel(label)
            self.labels[field].setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.labels[field].setStyleSheet("font-weight: bold;")
            
            self.values[field] = QLabel("-")
            self.values[field].setStyleSheet("background-color: #f8f9fa; padding: 3px 5px; border-radius: 3px;")
            
            self.info_layout.addWidget(self.labels[field], i, 0)
            self.info_layout.addWidget(self.values[field], i, 1)
            
            # 为状态字段添加颜色指示器
            if field == "status":
                indicator = QFrame()
                indicator.setFixedSize(16, 16)
                indicator.setFrameShape(QFrame.Box)
                indicator.setStyleSheet(f"background-color: {status_colors['idle']}; border-radius: 8px;")
                self.status_indicators[field] = indicator
                self.info_layout.addWidget(indicator, i, 2)
        
        self.info_group.setLayout(self.info_layout)
        self.layout.addWidget(self.info_group)
        
        # 任务信息
        self.task_group = QGroupBox("任务队列")
        self.task_layout = QVBoxLayout()
        
        self.task_text = QTextEdit()
        self.task_text.setReadOnly(True)
        self.task_text.setMaximumHeight(100)
        self.task_layout.addWidget(self.task_text)
        
        self.task_group.setLayout(self.task_layout)
        self.layout.addWidget(self.task_group)
        
        # 添加一个空白区域
        self.layout.addStretch()
    
    def set_environment(self, env):
        """设置环境并更新车辆列表"""
        if not env:
            return
            
        self.env = env
        
        # 更新车辆列表
        self.vehicle_combo.clear()
        
        for v_id in sorted(env.vehicles.keys()):
            self.vehicle_combo.addItem(f"车辆 {v_id}", v_id)
        
        # 更新信息显示
        if self.vehicle_combo.count() > 0:
            self.update_vehicle_info(0)
    
    def update_vehicle_info(self, index=None):
        """更新车辆信息显示 - 美化版"""
        if not hasattr(self, 'env') or not self.env:
            return
            
        if index is None or index < 0 or index >= self.vehicle_combo.count():
            return
            
        # 获取选中的车辆ID
        v_id = self.vehicle_combo.itemData(index)
        
        if v_id not in self.env.vehicles:
            return
            
        # 获取车辆信息
        vehicle = self.env.vehicles[v_id]
        
        # 更新显示
        self.values["id"].setText(str(v_id))
        
        if 'position' in vehicle:
            pos = vehicle['position']
            if len(pos) >= 3:
                angle_deg = pos[2] * 180 / math.pi
                self.values["position"].setText(f"({pos[0]:.1f}, {pos[1]:.1f}, {angle_deg:.1f}°)")
            else:
                self.values["position"].setText(f"({pos[0]:.1f}, {pos[1]:.1f})")
        
        # 更新状态和状态指示器
        if 'status' in vehicle:
            status = vehicle['status']
            status_map = {
                'idle': '空闲',
                'moving': '移动中',
                'loading': '装载中',
                'unloading': '卸载中'
            }
            status_text = status_map.get(status, status)
            self.values["status"].setText(status_text)
            
            # 更新状态指示器颜色
            status_colors = {
                'idle': '#6c757d',      # 灰色
                'moving': '#007bff',    # 蓝色
                'loading': '#28a745',   # 绿色
                'unloading': '#dc3545'  # 红色
            }
            color = status_colors.get(status, '#6c757d')
            self.status_indicators["status"].setStyleSheet(f"background-color: {color}; border-radius: 8px;")
        
        if 'load' in vehicle:
            max_load = vehicle.get('max_load', 100)
            load_percent = int(vehicle['load'] / max_load * 100)
            self.values["load"].setText(f"{vehicle['load']}/{max_load} ({load_percent}%)")
        
        # 检查是否有任务队列
        if 'task_queue' in vehicle and len(vehicle['task_queue']) > 0:
            task_idx = vehicle.get('current_task_index', 0)
            if task_idx < len(vehicle['task_queue']):
                task = vehicle['task_queue'][task_idx]
                task_type = task.get('task_type', '')
                task_map = {
                    'to_loading': '前往装载点',
                    'to_unloading': '前往卸载点',
                    'to_initial': '返回起点'
                }
                self.values["task"].setText(task_map.get(task_type, task_type))
                
                # 更新任务队列文本区域
                self.update_task_queue_text(vehicle)
        # 检查当前任务
        elif vehicle.get('current_task') and isinstance(vehicle.get('current_task'), str):
            task_id = vehicle.get('current_task')
            self.values["task"].setText(task_id)
        
        if 'goal' in vehicle:
            goal = vehicle['goal']
            if isinstance(goal, tuple) and len(goal) >= 2:
                if len(goal) >= 3:
                    angle_deg = goal[2] * 180 / math.pi
                    self.values["goal"].setText(f"({goal[0]:.1f}, {goal[1]:.1f}, {angle_deg:.1f}°)")
                else:
                    self.values["goal"].setText(f"({goal[0]:.1f}, {goal[1]:.1f})")
        
        # 显示已完成循环数
        completed_cycles = vehicle.get('completed_cycles', 0)
        self.values["completed"].setText(str(completed_cycles))
    
    def update_task_queue_text(self, vehicle):
        """更新任务队列显示"""
        if not vehicle or 'task_queue' not in vehicle:
            self.task_text.setText("无任务队列")
            return
            
        task_queue = vehicle['task_queue']
        current_idx = vehicle.get('current_task_index', 0)
        
        html = "<style>table {width:100%;} th {text-align:left; background:#f0f0f0;} .current {background:#e6f7ff; font-weight:bold;}</style>"
        html += "<table border='0' cellspacing='0' cellpadding='3'>"
        html += "<tr><th>序号</th><th>任务类型</th><th>起点</th><th>终点</th></tr>"
        
        for i, task in enumerate(task_queue):
            task_type = task.get('task_type', '')
            task_map = {
                'to_loading': '前往装载点',
                'to_unloading': '前往卸载点',
                'to_initial': '返回起点'
            }
            task_name = task_map.get(task_type, task_type)
            
            start = task.get('start', (-1, -1))
            goal = task.get('goal', (-1, -1))
            
            row_class = "current" if i == current_idx else ""
            html += f"<tr class='{row_class}'>"
            html += f"<td>{i+1}</td><td>{task_name}</td>"
            html += f"<td>({start[0]:.1f}, {start[1]:.1f})</td>"
            html += f"<td>({goal[0]:.1f}, {goal[1]:.1f})</td>"
            html += "</tr>"
            
        html += "</table>"
        self.task_text.setHtml(html)


class ECBSConfigPanel(QWidget):
    """ECBS配置面板"""
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 创建布局
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(10)
        
        # 标题
        self.title_label = QLabel("ECBS配置")
        self.title_label.setFont(QFont("SimHei", 12, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("color: #2C3E50; margin-bottom: 10px;")
        self.layout.addWidget(self.title_label)
        
        # 添加分隔线
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
        self.subopt_slider.setValue(15)  # 默认1.5
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
        
        # 高级选项开关
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
        
        # 冲突表格
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
        
        # 添加一个空白区域
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
        
        # 更新冲突表格
        self.conflict_table.setRowCount(len(conflicts))
        
        for i, conflict in enumerate(conflicts):
            self.conflict_table.setItem(i, 0, QTableWidgetItem(str(conflict.agent1)))
            self.conflict_table.setItem(i, 1, QTableWidgetItem(str(conflict.agent2)))
            self.conflict_table.setItem(i, 2, QTableWidgetItem(conflict.conflict_type))


class MineGUI(QMainWindow):
    """露天矿多车协调系统GUI - 美化版"""
    
    def __init__(self):
        """初始化主GUI窗口"""
        super().__init__()
        
        # 系统组件
        self.env = None
        self.backbone_network = None
        self.path_planner = None
        self.vehicle_scheduler = None
        self.traffic_manager = None
        
        # 应用全局样式表
        self.setStyleSheet(GLOBAL_STYLESHEET)
        
        # 初始化UI
        self.init_ui()
        
        # 创建更新定时器
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(50)  # 20 FPS
    
    def init_ui(self):
        """初始化用户界面"""
        # 设置主窗口
        self.setWindowTitle("露天矿多车协同调度系统 (基于骨干网络)")
        self.setGeometry(100, 100, 1200, 800)
        
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
        
        # 创建状态栏
        self.statusBar().showMessage("系统就绪")
        
        # 创建工具栏
        self.create_tool_bar()
    
    def create_control_panel(self):
        """创建左侧控制面板"""
        # 控制面板主容器
        self.control_panel = QFrame()
        self.control_panel.setFrameShape(QFrame.StyledPanel)
        self.control_panel.setMinimumWidth(300)
        self.control_panel.setMaximumWidth(350)
        
        # 控制面板布局
        control_layout = QVBoxLayout(self.control_panel)
        
        # 创建选项卡窗口部件
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
        
        # 添加空间
        self.env_layout.addStretch()
    
    def create_path_tab(self):
        """创建路径选项卡内容"""
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
        
        param_layout.addWidget(QLabel("路径平滑度:"), 1, 0)
        self.path_smoothness = QSpinBox()
        self.path_smoothness.setRange(1, 10)
        self.path_smoothness.setValue(5)
        param_layout.addWidget(self.path_smoothness, 1, 1)
        
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
        
        labels = ["总路径数:", "连接点数:", "总长度:", "平均长度:"]
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
        
        display_group.setLayout(display_layout)
        self.path_layout.addWidget(display_group)
        
        # ECBS配置面板
        self.ecbs_panel = ECBSConfigPanel()
        self.ecbs_panel.apply_button.clicked.connect(self.apply_ecbs_settings)
        self.path_layout.addWidget(self.ecbs_panel)
        
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
        
        # 添加空间
        self.path_layout.addStretch()
    
    def create_vehicle_tab(self):
        """创建车辆选项卡内容"""
        # 自定义车辆信息面板
        self.vehicle_info_panel = VehicleInfoPanel()
        self.vehicle_layout.addWidget(self.vehicle_info_panel)
        
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
        
        # 添加返回按钮
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
        
        # 添加空间
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
        
        # 统计组
        stats_group = QGroupBox("统计")
        stats_layout = QGridLayout()
        
        labels = ["总任务数:", "已完成任务:", "失败任务:", "平均利用率:", "总运行时间:"]
        self.task_stats_values = {}
        
        for i, label in enumerate(labels):
            stats_layout.addWidget(QLabel(label), i, 0)
            self.task_stats_values[label] = QLabel("--")
            self.task_stats_values[label].setStyleSheet("background-color: #f8f9fa; padding: 3px; border-radius: 3px;")
            stats_layout.addWidget(self.task_stats_values[label], i, 1)
        
        stats_group.setLayout(stats_layout)
        self.task_layout.addWidget(stats_group)
        
        # 添加空间
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
        
        self.show_backbone_action = QAction("显示骨干路径", self, checkable=True)
        self.show_backbone_action.setChecked(True)
        self.show_backbone_action.triggered.connect(self.toggle_backbone_display)
        view_menu.addAction(self.show_backbone_action)
        
        self.show_vehicles_action = QAction("显示车辆", self, checkable=True)
        self.show_vehicles_action.setChecked(True)
        self.show_vehicles_action.triggered.connect(self.toggle_vehicles_display)
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
        
        # 骨干网络生成
        self.generate_network_action = QAction("生成骨干网络", self)
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
            
            # 使用环境加载器加载环境
            from mine_loader import MineEnvironmentLoader
            loader = MineEnvironmentLoader()
            self.env = loader.load_environment(self.map_file_path)
            
            # 更新环境信息显示
            self.update_env_info()
            
            # 更新车辆下拉框
            self.update_vehicle_combo()
            
            # 创建场景
            self.create_scene()
            
            self.log("环境已加载: " + os.path.basename(self.map_file_path))
            
            # 创建其他系统组件
            self.create_system_components()
            
            # 启用相关按钮
            self.enable_controls(True)
            print(f"Environment loaded: {self.env is not None}")
            if self.env:
                print(f"Obstacles: {len(self.env._get_obstacle_list())}")
                print(f"Loading points: {len(self.env.loading_points)}")
                print(f"Unloading points: {len(self.env.unloading_points)}")
                print(f"Parking areas: {len(self.env.parking_areas)}")            
        except Exception as e:
            self.log(f"加载环境失败: {str(e)}", "error")
    
    def update_env_info(self):
        """更新环境信息显示"""
        if not self.env:
            return
        
        # 更新标签
        self.env_info_values["宽度:"].setText(str(self.env.width))
        self.env_info_values["高度:"].setText(str(self.env.height))
        self.env_info_values["装载点:"].setText(str(len(self.env.loading_points)))
        self.env_info_values["卸载点:"].setText(str(len(self.env.unloading_points)))
        self.env_info_values["车辆数:"].setText(str(len(self.env.vehicles)))
    
    def update_vehicle_combo(self):
        """更新车辆下拉框"""
        self.vehicle_info_panel.set_environment(self.env)
    
    def create_scene(self):
        """创建场景"""
        # 设置环境到图形视图
        self.graphics_view.set_environment(self.env)
    
    def create_system_components(self):
        """创建系统组件（骨干网络、路径规划器等）"""
        if not self.env:
            return
        
        # 创建骨干路径网络
        self.backbone_network = BackbonePathNetwork(self.env)
        
        # 创建路径规划器
        self.path_planner = PathPlanner(self.env)
        
        # 设置骨干网络到路径规划器
        self.path_planner.set_backbone_network(self.backbone_network)
        
        # 创建ECBS增强型交通管理器
        self.traffic_manager = TrafficManager(self.env, self.backbone_network)
        
        # 创建车辆调度器（如果有ECBS增强版则使用）
        try:
            # 尝试使用ECBS增强型调度器
            self.vehicle_scheduler = ECBSVehicleScheduler(
                self.env, 
                self.path_planner, 
                self.traffic_manager
            )
            self.log("使用ECBS增强型车辆调度器")
        except:
            # 回退到常规调度器
            self.vehicle_scheduler = VehicleScheduler(self.env, self.path_planner)
            self.log("使用标准车辆调度器")
        
        # 初始化车辆状态
        self.vehicle_scheduler.initialize_vehicles()
        
        # 创建任务模板
        if self.env.loading_points and self.env.unloading_points:
            if isinstance(self.vehicle_scheduler, ECBSVehicleScheduler):
                self.vehicle_scheduler.create_ecbs_mission_template("default")
            else:
                self.vehicle_scheduler.create_mission_template("default")
        
        self.log("系统组件已初始化")
    
    def generate_backbone_network(self):
        """生成骨干路径网络"""
        if not self.env:
            self.log("请先加载环境!", "error")
            return
        
        try:
            self.log("正在生成骨干路径网络...")
            
            # 生成网络
            self.backbone_network.generate_network()
            
            # 更新路径信息显示
            self.update_path_info()
            
            # 设置骨干网络到规划器和交通管理器
            self.path_planner.set_backbone_network(self.backbone_network)
            self.traffic_manager.set_backbone_network(self.backbone_network)
            
            # 绘制骨干网络
            self.draw_backbone_network()
            
            self.log(f"骨干路径网络已生成 - {len(self.backbone_network.paths)} 条路径")
            
        except Exception as e:
            self.log(f"生成骨干网络失败: {str(e)}", "error")
    
    def update_path_info(self):
        """更新路径信息显示"""
        if not self.backbone_network:
            return
        
        # 计算统计信息
        num_paths = len(self.backbone_network.paths)
        num_connections = len(self.backbone_network.connections)
        
        total_length = 0
        for path_data in self.backbone_network.paths.values():
            total_length += path_data.get('length', 0)
        
        avg_length = total_length / num_paths if num_paths > 0 else 0
        
        # 更新显示
        self.path_info_values["总路径数:"].setText(str(num_paths))
        self.path_info_values["连接点数:"].setText(str(num_connections))
        self.path_info_values["总长度:"].setText(f"{total_length:.1f}")
        self.path_info_values["平均长度:"].setText(f"{avg_length:.1f}")
    
    def draw_backbone_network(self):
        """绘制骨干路径网络"""
        if hasattr(self, 'backbone_visualizer'):
            self.graphics_view.mine_scene.removeItem(self.backbone_visualizer)
        
        # 创建新的可视化器
        self.backbone_visualizer = BackbonePathVisualizer(self.backbone_network)
        self.graphics_view.mine_scene.addItem(self.backbone_visualizer)
    
    def update_display(self):
        """更新显示（由定时器触发）"""
        if not self.env:
            return
        
        # 更新车辆位置
        self.graphics_view.update_vehicles()
        
        # 更新选中的车辆信息
        self.vehicle_info_panel.update_vehicle_info(self.vehicle_info_panel.vehicle_combo.currentIndex())
        
        # 更新任务信息
        self.update_task_list()
        
        # 更新骨干网络交通流可视化
        if hasattr(self, 'backbone_visualizer') and self.backbone_visualizer:
            self.backbone_visualizer.update_traffic_flow()
        
        # 显示ECBS冲突解决状态（如果可用）
        if self.traffic_manager and self.vehicle_scheduler:
            # 检查是否有活跃的冲突
            conflicts_resolved = 0
            if hasattr(self.vehicle_scheduler, 'conflict_counts'):
                conflicts_resolved = sum(self.vehicle_scheduler.conflict_counts.values())
            
            if conflicts_resolved > 0:
                self.statusBar().showMessage(f"ECBS已解决 {conflicts_resolved} 个路径冲突")
                
                # 更新任务选项卡中的冲突信息
                self.ecbs_panel.update_conflict_stats(conflicts_resolved, [])
    
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
            
            # 重新选择之前选中的任务
            if task_id == current_task_id:
                self.task_list.setCurrentItem(item)
        
        # 更新统计信息
        stats = self.vehicle_scheduler.get_stats()
        
        self.task_stats_values["总任务数:"].setText(str(len(self.vehicle_scheduler.tasks)))
        self.task_stats_values["已完成任务:"].setText(str(stats['completed_tasks']))
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
        """放大"""
        self.graphics_view.scale(1.2, 1.2)
    
    def zoom_out(self):
        """缩小"""
        self.graphics_view.scale(1/1.2, 1/1.2)
    
    def fit_view(self):
        """调整视图以显示整个场景"""
        self.graphics_view.fitInView(self.graphics_view.mine_scene.sceneRect(), Qt.KeepAspectRatio)
    
    # 模拟控制方法
    def update_simulation_speed(self):
        """更新模拟速度"""
        value = self.speed_slider.value()
        speed = value / 50.0  # 转换为速度倍增器(0.02-2.0)
        
        self.speed_value.setText(f"{speed:.1f}x")
        
        # 如果环境有time_step属性，更新它
        if hasattr(self.env, 'time_step'):
            self.env.time_step = 0.5 * speed  # 基础时间步长 * 速度倍增器
        
        self.log(f"模拟速度设置为 {speed:.1f}x")
    
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
        self.sim_timer.start(100)  # 每100毫秒更新一次
        
        self.log("模拟已开始")
    
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
        
        # 如果骨干网络已生成，重新绘制
        if self.backbone_network and hasattr(self.backbone_network, 'paths') and self.backbone_network.paths:
            self.draw_backbone_network()
        
        # 重置进度条
        self.progress_bar.setValue(0)
        
        # 更新车辆和任务信息
        self.update_vehicle_combo()
        self.update_task_list()
        
        self.log("模拟已重置")
    
    def simulation_step(self):
        """单个模拟步骤与ECBS冲突解决"""
        if not self.is_simulating or not self.env:
            return
        
        # 获取时间步长
        time_step = getattr(self.env, 'time_step', 0.5)
        
        # 更新环境时间
        if not hasattr(self.env, 'current_time'):
            self.env.current_time = 0
        
        self.env.current_time += time_step
        
        # 使用ECBS更新车辆调度器
        if self.vehicle_scheduler:
            # 对于ECBSVehicleScheduler，update方法包括冲突检测和解决
            self.vehicle_scheduler.update(time_step)
            
            # 同步车辆状态
            self._synchronize_vehicle_statuses()
            
            # 检查是否有任何活跃的冲突
            has_conflicts = False
            if hasattr(self.traffic_manager, 'has_conflict'):
                has_conflicts = self.traffic_manager.has_conflict
            
            # 如果有冲突，记录日志
            if has_conflicts:
                self.log("ECBS正在解决路径冲突...", "warning")
        
        # 更新进度条
        max_time = 3600  # 最大模拟时间：1小时
        progress = min(100, int(self.env.current_time * 100 / max_time))
        self.progress_bar.setValue(progress)
        
        # 如果进度达到100%，自动停止
        if progress >= 100:
            self.pause_simulation()
            self.log("模拟完成", "success")
    def _synchronize_vehicle_statuses(self):
        """确保车辆状态根据其行为正确更新"""
        if not self.env:
            return
            
        for vehicle_id, vehicle_data in self.env.vehicles.items():
            # 检查车辆是否有路径和目标但状态不是moving
            if 'path' in vehicle_data and vehicle_data['path'] and len(vehicle_data['path']) > 1:
                path_index = vehicle_data.get('path_index', 0)
                
                # 如果还没到达终点
                if path_index < len(vehicle_data['path']) - 1:
                    # 如果车辆状态不是moving，将其更新为moving
                    if vehicle_data.get('status') != 'moving':
                        vehicle_data['status'] = 'moving'
                        print(f"Vehicle {vehicle_id} status updated to 'moving'")
                        
                    # 确保有正确的朝向角度
                    current_point = vehicle_data['path'][path_index]
                    next_point = vehicle_data['path'][path_index + 1]
                    
                    # 获取方向向量
                    dx = next_point[0] - current_point[0]
                    dy = next_point[1] - current_point[1]
                    
                    # 只有在有明显位移时才更新朝向
                    if abs(dx) > 0.01 or abs(dy) > 0.01:
                        # 计算车辆朝向角度
                        theta = math.atan2(dy, dx)
                        
                        # 更新位置中的朝向角度
                        x, y, _ = vehicle_data['position']
                        vehicle_data['position'] = (x, y, theta)    
    # 车辆操作方法
    def assign_vehicle_task(self):
        """使用ECBS调度分配任务给当前选中的车辆"""
        if not self.vehicle_scheduler:
            self.log("车辆调度器未初始化", "error")
            return
        
        # 获取选中的车辆
        vehicle_id = self.vehicle_info_panel.vehicle_combo.itemData(
            self.vehicle_info_panel.vehicle_combo.currentIndex()
        )
        if vehicle_id is None:
            self.log("请先选择一个车辆", "warning")
            return
        
        # 如果有任务模板，使用ECBS感知分配
        if "default" in self.vehicle_scheduler.mission_templates:
            # 使用调度器的assign_mission方法
            if self.vehicle_scheduler.assign_mission(vehicle_id, "default"):
                self.log(f"已将默认任务分配给车辆 {vehicle_id}", "success")
            else:
                self.log(f"无法分配任务给车辆 {vehicle_id}", "error")
        else:
            self.log("无可用的任务模板", "error")
    
    def cancel_vehicle_task(self):
        """取消当前车辆的任务"""
        if not self.vehicle_scheduler:
            return
        
        # 获取选中的车辆
        vehicle_id = self.vehicle_info_panel.vehicle_combo.itemData(
            self.vehicle_info_panel.vehicle_combo.currentIndex()
        )
        if vehicle_id is None:
            self.log("请先选择一个车辆", "warning")
            return
        
        # 清除任务队列
        if vehicle_id in self.vehicle_scheduler.task_queues:
            self.vehicle_scheduler.task_queues[vehicle_id] = []
            
            # 如果有当前任务，标记为已完成
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
                
                # 在交通管理器中释放路径
                if self.traffic_manager:
                    self.traffic_manager.release_vehicle_path(vehicle_id)
                
                self.log(f"已取消车辆 {vehicle_id} 的所有任务", "success")
            else:
                self.log(f"车辆 {vehicle_id} 没有活动任务", "warning")
    
    def goto_loading_point(self):
        """命令当前车辆前往装载点"""
        if not self.vehicle_scheduler or not self.env.loading_points:
            return
        
        # 获取选中的车辆
        vehicle_id = self.vehicle_info_panel.vehicle_combo.itemData(
            self.vehicle_info_panel.vehicle_combo.currentIndex()
        )
        if vehicle_id is None:
            self.log("请先选择一个车辆", "warning")
            return
        
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
        
        # 清除并添加到车辆任务队列
        self.vehicle_scheduler.task_queues[vehicle_id] = [task_id]
        
        # 如果车辆空闲，立即开始任务
        status = self.vehicle_scheduler.vehicle_statuses[vehicle_id]
        if status['status'] == 'idle':
            self.vehicle_scheduler._start_next_task(vehicle_id)
        
        self.log(f"已命令车辆 {vehicle_id} 前往装载点", "success")
    
    def goto_unloading_point(self):
        """命令当前车辆前往卸载点"""
        if not self.vehicle_scheduler or not self.env.unloading_points:
            return
        
        # 获取选中的车辆
        vehicle_id = self.vehicle_info_panel.vehicle_combo.itemData(
            self.vehicle_info_panel.vehicle_combo.currentIndex()
        )
        if vehicle_id is None:
            self.log("请先选择一个车辆", "warning")
            return
        
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
        
        # 清除并添加到车辆任务队列
        self.vehicle_scheduler.task_queues[vehicle_id] = [task_id]
        
        # 如果车辆空闲，立即开始任务
        status = self.vehicle_scheduler.vehicle_statuses[vehicle_id]
        if status['status'] == 'idle':
            self.vehicle_scheduler._start_next_task(vehicle_id)
        
        self.log(f"已命令车辆 {vehicle_id} 前往卸载点", "success")
    
    def return_to_start(self):
        """命令当前车辆返回起点"""
        if not self.vehicle_scheduler:
            return
        
        # 获取选中的车辆
        vehicle_id = self.vehicle_info_panel.vehicle_combo.itemData(
            self.vehicle_info_panel.vehicle_combo.currentIndex()
        )
        if vehicle_id is None:
            self.log("请先选择一个车辆", "warning")
            return
        
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
        
        # 清除并添加到车辆任务队列
        self.vehicle_scheduler.task_queues[vehicle_id] = [task_id]
        
        # 如果车辆空闲，立即开始任务
        status = self.vehicle_scheduler.vehicle_statuses[vehicle_id]
        if status['status'] == 'idle':
            self.vehicle_scheduler._start_next_task(vehicle_id)
        
        self.log(f"已命令车辆 {vehicle_id} 返回起点", "success")
    
    # 任务操作方法
    def create_task_template(self):
        """创建任务模板"""
        if not self.vehicle_scheduler or not self.env.loading_points or not self.env.unloading_points:
            self.log("环境未初始化或缺少装载/卸载点", "error")
            return
        
        # 创建默认模板
        if isinstance(self.vehicle_scheduler, ECBSVehicleScheduler):
            if self.vehicle_scheduler.create_ecbs_mission_template("default"):
                self.log("已创建支持ECBS的默认任务模板", "success")
            else:
                self.log("创建任务模板失败", "error")
        else:
            if self.vehicle_scheduler.create_mission_template("default"):
                self.log("已创建默认任务模板", "success")
            else:
                self.log("创建任务模板失败", "error")
    
    def auto_assign_tasks(self):
        """使用ECBS批量规划自动分配任务给所有车辆"""
        if not self.vehicle_scheduler:
            self.log("车辆调度器未初始化", "error")
            return
        
        # 确保我们有任务模板
        if "default" not in self.vehicle_scheduler.mission_templates:
            self.create_task_template()
        
        # 对于ECBSVehicleScheduler，使用批量分配以获得更好的协调
        if isinstance(self.vehicle_scheduler, ECBSVehicleScheduler):
            # 准备用于ECBS协调的车辆批次
            vehicle_ids = list(self.env.vehicles.keys())
            
            self.log(f"正在使用ECBS为 {len(vehicle_ids)} 辆车辆规划协调路径...")
            
            # 收集所有车辆的任务
            tasks = []
            for vehicle_id in vehicle_ids:
                # 从模板创建任务
                template = self.vehicle_scheduler.mission_templates["default"]
                vehicle_pos = self.env.vehicles[vehicle_id]['position']
                
                for task_template in template:
                    task = VehicleTask(
                        f"task_{self.vehicle_scheduler.task_counter}",
                        task_template['task_type'],
                        vehicle_pos,
                        task_template['goal'] if task_template['goal'] else vehicle_pos,
                        task_template['priority']
                    )
                    self.vehicle_scheduler.task_counter += 1
                    tasks.append(task)
                    vehicle_pos = task.goal  # 更新下一个任务的起点
            
            # 使用ECBS批量分配
            assignments = self.vehicle_scheduler.assign_tasks_batch(tasks)
            self.log(f"ECBS批量分配完成，共 {len(assignments)} 个分配")
        else:
            # 回退到简单分配
            success_count = 0
            for vehicle_id in self.env.vehicles.keys():
                if self.vehicle_scheduler.assign_mission(vehicle_id, "default"):
                    success_count += 1
            
            self.log(f"已分配任务给 {success_count}/{len(self.env.vehicles)} 辆车辆", "success")
    
    # 显示更新方法
    def update_path_display(self):
        """更新路径显示选项"""
        if hasattr(self, 'backbone_visualizer') and self.backbone_visualizer:
            # 显示/隐藏骨干路径
            show_paths = self.show_paths_cb.isChecked()
            self.backbone_visualizer.setVisible(show_paths)
            
            # 更新连接点和交通流显示
            for path_id, path_item in self.backbone_visualizer.path_items.items():
                path_item.setVisible(show_paths and self.show_paths_cb.isChecked())
            
            for conn_id, conn_item in self.backbone_visualizer.connection_items.items():
                conn_item.setVisible(show_paths and self.show_connections_cb.isChecked())
            
            if self.show_traffic_cb.isChecked():
                self.backbone_visualizer.update_traffic_flow()
    
    def update_vehicle_display(self):
        """更新车辆显示选项"""
        # 设置车辆视图参数
        self.graphics_view.set_show_trajectories(self.show_vehicle_paths_cb.isChecked())
        
        # 更新车辆显示
        for vehicle_id, vehicle_data in self.env.vehicles.items():
            if vehicle_id in self.graphics_view.mine_scene.vehicle_items:
                vehicle_item = self.graphics_view.mine_scene.vehicle_items[vehicle_id]
                vehicle_item.setVisible(self.show_vehicles_cb.isChecked())
                
                # 显示/隐藏标签
                if hasattr(vehicle_item, 'vehicle_label'):
                    vehicle_item.vehicle_label.setVisible(self.show_vehicles_cb.isChecked() and self.show_vehicle_labels_cb.isChecked())
    
    def toggle_edit_mode(self, checked):
        """切换编辑模式"""
        self.add_path_button.setEnabled(checked)
        self.delete_path_button.setEnabled(checked)
        
        # TODO: 实现编辑模式功能
        self.log("编辑模式 " + ("已启用" if checked else "已禁用"))
    
    def start_add_path(self):
        """开始添加路径"""
        # TODO: 实现添加路径功能
        self.log("添加路径功能尚未实现", "warning")
    
    def start_delete_path(self):
        """开始删除路径"""
        # TODO: 实现删除路径功能
        self.log("删除路径功能尚未实现", "warning")
    
    def toggle_backbone_display(self, checked):
        """切换骨干路径显示"""
        if hasattr(self, 'backbone_visualizer') and self.backbone_visualizer:
            self.backbone_visualizer.setVisible(checked)
            
            # 同步复选框状态
            self.show_paths_cb.setChecked(checked)
    
    def toggle_vehicles_display(self, checked):
        """切换车辆显示"""
        # 更新复选框状态
        self.show_vehicles_cb.setChecked(checked)
        
        # 更新车辆显示
        self.update_vehicle_display()
    
    def optimize_paths(self):
        """优化路径"""
        if not self.backbone_network or not self.backbone_network.paths:
            self.log("请先生成骨干路径网络", "warning")
            return
        
        try:
            self.log("正在优化路径...")
            
            # 重新优化所有路径
            self.backbone_network._optimize_all_paths()
            
            # 更新路径信息
            self.update_path_info()
            
            # 重新绘制骨干网络
            self.draw_backbone_network()
            
            self.log("路径优化完成", "success")
            
        except Exception as e:
            self.log(f"路径优化失败: {str(e)}", "error")
    
    def apply_ecbs_settings(self):
        """应用ECBS设置到交通管理器"""
        if not self.traffic_manager:
            self.log("交通管理器不可用", "warning")
            return
        
        # 获取子最优性边界
        settings = self.ecbs_panel.get_settings()
        subopt = settings["suboptimality"]
        strategy = settings["strategy"]
        
        # 应用设置
        if hasattr(self.traffic_manager, 'suboptimality_bound'):
            self.traffic_manager.suboptimality_bound = subopt
        
        # 应用到车辆调度器如果它是ECBSVehicleScheduler
        if isinstance(self.vehicle_scheduler, ECBSVehicleScheduler):
            if hasattr(self.vehicle_scheduler, 'conflict_resolution_strategy'):
                self.vehicle_scheduler.conflict_resolution_strategy = strategy.lower()
            
            # 应用高级设置
            if hasattr(self.vehicle_scheduler, 'use_focal_list'):
                self.vehicle_scheduler.use_focal_list = settings["focal_enabled"]
            
            if hasattr(self.vehicle_scheduler, 'reuse_paths'):
                self.vehicle_scheduler.reuse_paths = settings["pathreuse_enabled"]
        
        self.log(f"已应用ECBS设置: 界限={subopt}, 策略={strategy}", "success")
    
    def save_results(self):
        """保存结果"""
        if not self.env:
            self.log("没有结果可保存", "warning")
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
            <p>基于骨干路径网络的多车辆规划与调度</p>
            <p>版本: 1.0</p>
            <hr>
            <p>使用骨干路径网络优化矿山车辆调度效率</p>
            <p>支持多车辆协调、冲突避免和交通管理</p>
            <hr>
            <p>开发者: XXX</p>
            <p>版权所有 © 2023</p>
        </div>
        """
        
        QMessageBox.about(self, "关于", about_text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle("Fusion")
    
    # 创建并显示主窗口
    window = MineGUI()
    window.show()
    
    sys.exit(app.exec_())