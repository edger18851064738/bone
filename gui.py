import sys
import os
import math
import time
import json
from collections import defaultdict, deque
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

# 导入系统组件
from backbone_network import SimplifiedBackbonePathNetwork
from path_planner import SimplifiedPathPlanner
from traffic_manager import OptimizedTrafficManager
from vehicle_scheduler import SimplifiedVehicleScheduler, SimplifiedECBSVehicleScheduler
from environment import OpenPitMineEnv

# 增强的全局样式
# ENHANCED_STYLE = """
# QMainWindow { 
#     background-color: #1e1e2e; 
# }
# QWidget {
#     background-color: #2b2b3c;
#     color: #cdd6f4;
# }
# QTabWidget::pane { 
#     border: 1px solid #45475a; 
#     background-color: #313244; 
#     border-radius: 6px; 
# }
# QTabBar::tab { 
#     background-color: #45475a; 
#     border: 1px solid #585b70; 
#     border-bottom: none; 
#     border-top-left-radius: 6px; 
#     border-top-right-radius: 6px; 
#     padding: 10px 20px; 
#     font-weight: bold; 
#     min-width: 120px; 
# }
# QTabBar::tab:selected { 
#     background-color: #313244; 
#     border-bottom: 2px solid #89b4fa; 
# }
# QTabBar::tab:hover {
#     background-color: #585b70;
# }
# QGroupBox { 
#     font-weight: bold; 
#     border: 2px solid #45475a; 
#     border-radius: 8px; 
#     margin-top: 14px; 
#     padding-top: 20px; 
#     background-color: #313244; 
# }
# QGroupBox::title { 
#     subcontrol-origin: margin; 
#     subcontrol-position: top left; 
#     padding: 8px 12px; 
#     background-color: #45475a; 
#     border-radius: 4px; 
#     color: #f5e0dc;
# }
# QPushButton { 
#     background-color: #89b4fa; 
#     color: #1e1e2e; 
#     border: none; 
#     padding: 10px 20px; 
#     border-radius: 6px; 
#     font-weight: bold; 
#     min-width: 100px;
# }
# QPushButton:hover { 
#     background-color: #89dceb; 
# }
# QPushButton:pressed { 
#     background-color: #74c7ec; 
# }
# QPushButton:disabled { 
#     background-color: #6c7086; 
#     color: #45475a; 
# }
# QComboBox { 
#     border: 2px solid #45475a; 
#     border-radius: 6px; 
#     padding: 8px; 
#     min-width: 150px; 
#     background-color: #313244; 
#     color: #cdd6f4;
# }
# QComboBox:hover {
#     border-color: #89b4fa;
# }
# QComboBox::drop-down {
#     border: none;
# }
# QComboBox::down-arrow {
#     image: none;
#     border-left: 5px solid transparent;
#     border-right: 5px solid transparent;
#     border-top: 5px solid #cdd6f4;
#     margin-right: 5px;
# }
# QSpinBox, QDoubleSpinBox { 
#     border: 2px solid #45475a; 
#     border-radius: 6px; 
#     padding: 8px; 
#     min-width: 100px; 
#     background-color: #313244;
#     color: #cdd6f4;
# }
# QSpinBox:hover, QDoubleSpinBox:hover {
#     border-color: #89b4fa;
# }
# QTextEdit { 
#     border: 2px solid #45475a; 
#     border-radius: 6px; 
#     background-color: #1e1e2e; 
#     color: #cdd6f4;
#     padding: 8px;
# }
# QProgressBar { 
#     border: 2px solid #45475a; 
#     border-radius: 6px; 
#     background-color: #313244; 
#     text-align: center; 
#     font-weight: bold; 
#     color: #cdd6f4; 
#     height: 24px; 
# }
# QProgressBar::chunk { 
#     background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
#                                stop:0 #f38ba8, 
#                                stop:0.5 #eba0ac, 
#                                stop:1 #f38ba8);
#     border-radius: 4px; 
# }
# QSlider::groove:horizontal {
#     height: 8px;
#     background: #45475a;
#     border-radius: 4px;
# }
# QSlider::handle:horizontal {
#     background: #89b4fa;
#     width: 20px;
#     height: 20px;
#     margin: -6px 0;
#     border-radius: 10px;
# }
# QSlider::handle:horizontal:hover {
#     background: #89dceb;
# }
# QCheckBox {
#     spacing: 8px;
# }
# QCheckBox::indicator {
#     width: 20px;
#     height: 20px;
#     border: 2px solid #45475a;
#     border-radius: 4px;
#     background-color: #313244;
# }
# QCheckBox::indicator:checked {
#     background-color: #89b4fa;
#     image: url(checkmark.png);
# }
# QTableWidget {
#     border: 2px solid #45475a;
#     border-radius: 6px;
#     background-color: #1e1e2e;
#     gridline-color: #45475a;
# }
# QTableWidget::item {
#     padding: 5px;
# }
# QTableWidget::item:selected {
#     background-color: #45475a;
# }
# QHeaderView::section {
#     background-color: #313244;
#     border: 1px solid #45475a;
#     padding: 5px;
#     font-weight: bold;
# }
# QTreeWidget {
#     border: 2px solid #45475a;
#     border-radius: 6px;
#     background-color: #1e1e2e;
# }
# QTreeWidget::item {
#     padding: 5px;
# }
# QTreeWidget::item:selected {
#     background-color: #45475a;
# }
# QTreeWidget::item:hover {
#     background-color: #313244;
# }
# QLCDNumber {
#     border: 2px solid #45475a;
#     border-radius: 6px;
#     background-color: #1e1e2e;
#     color: #a6e3a1;
# }
# QDockWidget {
#     color: #cdd6f4;
#     font-weight: bold;
# }
# QDockWidget::title {
#     background-color: #313244;
#     padding: 10px;
#     border-bottom: 2px solid #45475a;
# }
# QStatusBar {
#     background-color: #1e1e2e;
#     border-top: 1px solid #45475a;
#     color: #cdd6f4;
# }
# QToolBar {
#     background-color: #313244;
#     border: none;
#     spacing: 5px;
#     padding: 5px;
# }
# QToolBar::separator {
#     background-color: #45475a;
#     width: 1px;
#     margin: 5px;
# }
# """

# 增强的车辆状态颜色
VEHICLE_STATUS_COLORS = {
    'idle': QColor(108, 112, 134),      # 灰色
    'loading': QColor(166, 227, 161),   # 绿色
    'unloading': QColor(243, 139, 168), # 粉色
    'moving': QColor(137, 180, 250),    # 蓝色
    'waiting': QColor(249, 226, 175),   # 黄色
    'maintenance': QColor(235, 160, 172) # 红色
}

class AnimatedVehicleItem(QGraphicsObject):
    """动画车辆图形项"""
    def __init__(self, vehicle_id, vehicle_data, parent=None):
        super().__init__(parent)
        self.vehicle_id = vehicle_id
        self.vehicle_data = vehicle_data
        self.position = vehicle_data['position']
        self.target_position = self.position
        
        # 创建车辆组件
        self.vehicle_body = QGraphicsPolygonItem(self)
        self.vehicle_label = QGraphicsTextItem(str(vehicle_id), self)
        self.status_indicator = QGraphicsEllipseItem(self)
        self.load_indicator = QGraphicsRectItem(self)
        
        # 动画属性
        self.position_animation = QPropertyAnimation(self, b"animated_position")
        self.position_animation.setDuration(500)
        self.position_animation.setEasingCurve(QEasingCurve.InOutQuad)
        
        # 特效
        self.glow_effect = None
        self.create_glow_effect()
        
        self.setZValue(15)
        self.update_appearance()
        self.update_position()
    
    def create_glow_effect(self):
        """创建发光效果"""
        # 这里可以添加更复杂的特效
        pass
    
    @pyqtProperty(QPointF)
    def animated_position(self):
        return QPointF(self.position[0], self.position[1])
    
    @animated_position.setter
    def animated_position(self, value):
        self.position = (value.x(), value.y(), self.position[2])
        self.update_position()
    
    def update_appearance(self):
        """更新车辆外观"""
        status = self.vehicle_data.get('status', 'idle')
        color = VEHICLE_STATUS_COLORS.get(status, VEHICLE_STATUS_COLORS['idle'])
        
        # 车辆主体
        gradient = QLinearGradient(0, 0, 10, 10)
        gradient.setColorAt(0, color.lighter(120))
        gradient.setColorAt(1, color)
        self.vehicle_body.setBrush(QBrush(gradient))
        self.vehicle_body.setPen(QPen(color.darker(150), 1))
        
        # 标签
        self.vehicle_label.setDefaultTextColor(Qt.white)
        font = QFont("Arial", 3, QFont.Bold)
        self.vehicle_label.setFont(font)
        
        # 状态指示器动画
        if status == 'moving':
            # 移动状态 - 脉冲效果
            self.status_indicator.setBrush(QBrush(color.lighter(150)))
            self.status_indicator.setPen(QPen(color, 0.5))
        else:
            self.status_indicator.setBrush(QBrush(color))
            self.status_indicator.setPen(QPen(Qt.black, 0.3))
        
        # 负载指示器
        load_ratio = self.vehicle_data.get('load', 0) / self.vehicle_data.get('max_load', 100)
        load_color = QColor(255, int(255 * (1 - load_ratio)), 0)
        self.load_indicator.setBrush(QBrush(load_color))
        self.load_indicator.setPen(QPen(Qt.black, 0.2))
    
    def update_position(self):
        """更新车辆位置"""
        if not self.position or len(self.position) < 3:
            return
        
        x, y, theta = self.position
        
        # 创建车辆多边形
        length, width = 6.0, 3.0
        half_length, half_width = length/2, width/2
        
        # 车辆形状 - 更真实的卡车形状
        truck_shape = [
            QPointF(half_length * 0.8, half_width * 0.9),      # 右前
            QPointF(half_length, half_width * 0.7),            # 右前尖
            QPointF(half_length, -half_width * 0.7),           # 左前尖
            QPointF(half_length * 0.8, -half_width * 0.9),     # 左前
            QPointF(-half_length * 0.6, -half_width),          # 左后
            QPointF(-half_length, -half_width * 0.8),          # 左后尖
            QPointF(-half_length, half_width * 0.8),           # 右后尖
            QPointF(-half_length * 0.6, half_width)            # 右后
        ]
        
        # 应用旋转
        transform = QTransform()
        transform.rotate(theta * 180 / math.pi)
        
        polygon = QPolygonF()
        for point in truck_shape:
            rotated = transform.map(point)
            polygon.append(QPointF(x + rotated.x(), y + rotated.y()))
        
        self.vehicle_body.setPolygon(polygon)
        
        # 更新标签位置
        self.vehicle_label.setPos(x - 2, y - 2)
        
        # 更新状态指示器
        self.status_indicator.setRect(x + half_length - 2, y - 1, 2, 2)
        
        # 更新负载指示器
        load_ratio = self.vehicle_data.get('load', 0) / self.vehicle_data.get('max_load', 100)
        self.load_indicator.setRect(x - half_length, y - width/2 - 3, length * load_ratio, 1)
    
    def animate_to_position(self, new_position):
        """动画移动到新位置"""
        self.target_position = new_position
        self.position_animation.setStartValue(QPointF(self.position[0], self.position[1]))
        self.position_animation.setEndValue(QPointF(new_position[0], new_position[1]))
        self.position_animation.start()
        self.position = new_position
    
    def update_data(self, vehicle_data):
        """更新车辆数据"""
        self.vehicle_data = vehicle_data
        new_position = vehicle_data['position']
        
        # 如果位置变化较大，使用动画
        if (abs(new_position[0] - self.position[0]) > 0.5 or 
            abs(new_position[1] - self.position[1]) > 0.5):
            self.animate_to_position(new_position)
        else:
            self.position = new_position
            self.update_position()
        
        self.update_appearance()

class ConflictVisualization(QGraphicsItemGroup):
    """冲突可视化"""
    def __init__(self, conflict, parent=None):
        super().__init__(parent)
        self.conflict = conflict
        self.setZValue(20)
        
        # 创建冲突标记
        self.create_conflict_marker()
        
        # 动画
        self.pulse_animation = QVariantAnimation()
        self.pulse_animation.setDuration(1000)
        self.pulse_animation.setStartValue(0.5)
        self.pulse_animation.setEndValue(1.5)
        self.pulse_animation.setLoopCount(-1)
        self.pulse_animation.valueChanged.connect(self.update_pulse)
        self.pulse_animation.start()
    
    def create_conflict_marker(self):
        """创建冲突标记"""
        x, y = self.conflict.location
        
        # 冲突区域
        conflict_area = QGraphicsEllipseItem(x-10, y-10, 20, 20)
        
        # 根据冲突类型设置颜色
        colors = {
            'vertex': QColor(255, 100, 100, 100),
            'edge': QColor(255, 150, 100, 100),
            'interface': QColor(255, 200, 100, 150),
            'backbone': QColor(255, 100, 200, 100),
            'predictive': QColor(100, 100, 255, 80)
        }
        
        color = colors.get(self.conflict.conflict_type, QColor(255, 100, 100, 100))
        conflict_area.setBrush(QBrush(color))
        conflict_area.setPen(QPen(color.darker(150), 2))
        
        self.addToGroup(conflict_area)
        self.conflict_area = conflict_area
        
        # 冲突类型标签
        label = QGraphicsTextItem(self.conflict.conflict_type)
        label.setPos(x - 20, y - 25)
        label.setDefaultTextColor(Qt.white)
        label.setFont(QFont("Arial", 3))
        self.addToGroup(label)
    
    def update_pulse(self, value):
        """更新脉冲动画"""
        scale_transform = QTransform()
        scale_transform.scale(value, value)
        self.conflict_area.setTransform(scale_transform)
        
        # 更新透明度
        opacity = 1.0 - (value - 0.5) / 1.0
        self.setOpacity(opacity)

class EnhancedInterfaceItem(QGraphicsItemGroup):
    """增强的接口图形项"""
    def __init__(self, interface, parent=None):
        super().__init__(parent)
        self.interface = interface
        self.setZValue(12)
        
        self.create_interface_visual()
        
        # 使用动画
        self.usage_animation = QVariantAnimation()
        self.usage_animation.setDuration(500)
        self.usage_animation.valueChanged.connect(self.update_usage_effect)
    
    def create_interface_visual(self):
        """创建接口可视化"""
        x, y = self.interface.position[0], self.interface.position[1]
        
        # 外圈 - 影响范围
        influence_circle = QGraphicsEllipseItem(x - 8, y - 8, 16, 16)
        influence_circle.setBrush(QBrush(QColor(100, 150, 255, 30)))
        influence_circle.setPen(QPen(QColor(100, 150, 255, 50), 1, Qt.DashLine))
        self.addToGroup(influence_circle)
        
        # 接口主体
        radius = 4
        if self.interface.is_occupied:
            color = QColor(255, 100, 100, 200)
        else:
            # 根据可达性评分设置颜色
            accessibility = getattr(self.interface, 'accessibility_score', 0.5)
            green = int(100 + 155 * accessibility)
            color = QColor(255 - green, green, 100, 200)
        
        self.interface_circle = QGraphicsEllipseItem(x - radius, y - radius, radius * 2, radius * 2)
        self.interface_circle.setBrush(QBrush(color))
        self.interface_circle.setPen(QPen(Qt.black, 1))
        self.addToGroup(self.interface_circle)
        
        # 方向指示器
        arrow_length = 6
        direction = self.interface.direction
        
        arrow_path = QPainterPath()
        arrow_path.moveTo(x, y)
        end_x = x + arrow_length * math.cos(direction)
        end_y = y + arrow_length * math.sin(direction)
        arrow_path.lineTo(end_x, end_y)
        
        # 箭头头部
        arrow_angle = 0.5
        arrow_head_length = 2
        arrow_path.lineTo(
            end_x - arrow_head_length * math.cos(direction - arrow_angle),
            end_y - arrow_head_length * math.sin(direction - arrow_angle)
        )
        arrow_path.moveTo(end_x, end_y)
        arrow_path.lineTo(
            end_x - arrow_head_length * math.cos(direction + arrow_angle),
            end_y - arrow_head_length * math.sin(direction + arrow_angle)
        )
        
        arrow_item = QGraphicsPathItem(arrow_path)
        arrow_item.setPen(QPen(QColor(50, 100, 200), 1.5))
        self.addToGroup(arrow_item)
        
        # 接口ID和统计信息
        info_text = f"{self.interface.interface_id.split('_')[-1]}"
        if hasattr(self.interface, 'usage_count'):
            info_text += f"\n{self.interface.usage_count}"
        
        label = QGraphicsTextItem(info_text)
        label.setPos(x + 5, y - 5)
        label.setDefaultTextColor(Qt.black)
        label.setFont(QFont("Arial", 2))
        self.addToGroup(label)
        
        # 工具提示
        self.setToolTip(f"接口 {self.interface.interface_id}\n"
                       f"可达性: {getattr(self.interface, 'accessibility_score', 0.5):.2f}\n"
                       f"使用次数: {self.interface.usage_count}\n"
                       f"状态: {'占用' if self.interface.is_occupied else '空闲'}")
    
    def update_usage_effect(self, value):
        """更新使用效果"""
        # 使用时的闪烁效果
        self.interface_circle.setOpacity(value)
    
    def trigger_usage_animation(self):
        """触发使用动画"""
        self.usage_animation.setStartValue(1.0)
        self.usage_animation.setEndValue(0.3)
        self.usage_animation.start()

class BackbonePathVisualization(QGraphicsItemGroup):
    """增强的骨干路径可视化"""
    def __init__(self, backbone_network, parent=None):
        super().__init__(parent)
        self.backbone_network = backbone_network
        self.setZValue(3)
        self.path_items = {}
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
            
            # 创建渐变路径
            painter_path = QPainterPath()
            painter_path.moveTo(path[0][0], path[0][1])
            
            for i in range(1, len(path)):
                painter_path.lineTo(path[i][0], path[i][1])
            
            path_item = QGraphicsPathItem(painter_path)
            
            # 根据路径属性设置样式
            quality = path_data.get('quality', 0.5)
            usage = path_data.get('usage_count', 0)
            
            # 颜色基于质量
            if quality >= 0.8:
                base_color = QColor(100, 255, 100, 180)
            elif quality >= 0.6:
                base_color = QColor(255, 255, 100, 180)
            else:
                base_color = QColor(255, 100, 100, 180)
            
            # 线宽基于使用频率
            width = 2 + min(usage / 10, 3)
            
            pen = QPen(base_color, width)
            pen.setStyle(Qt.SolidLine)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            
            path_item.setPen(pen)
            self.addToGroup(path_item)
            self.path_items[path_id] = path_item
            
            # 路径起终点标记
            self._add_path_endpoints(path_data)
    
    def _add_path_endpoints(self, path_data):
        """添加路径起终点标记"""
        path = path_data.get('path', [])
        if len(path) < 2:
            return
        
        # 起点标记
        start_point = path_data.get('start_point', {})
        start_pos = path[0]
        start_marker = QGraphicsEllipseItem(
            start_pos[0] - 3, start_pos[1] - 3, 6, 6
        )
        start_marker.setBrush(QBrush(QColor(100, 255, 100)))
        start_marker.setPen(QPen(Qt.black, 1))
        self.addToGroup(start_marker)
        
        # 终点标记
        end_point = path_data.get('end_point', {})
        end_pos = path[-1]
        end_marker = QGraphicsRectItem(
            end_pos[0] - 3, end_pos[1] - 3, 6, 6
        )
        end_marker.setBrush(QBrush(QColor(255, 100, 100)))
        end_marker.setPen(QPen(Qt.black, 1))
        self.addToGroup(end_marker)

class OptimizedMineScene(QGraphicsScene):
    """优化的矿场图形场景"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.env = None
        self.backbone_network = None
        self.traffic_manager = None
        
        # 显示选项
        self.show_options = {
            'trajectories': True,
            'backbone': True,
            'interfaces': True,
            'conflicts': True,
            'heatmap': False,
            'grid': True,
            'labels': True
        }
        
        # 图形项容器
        self.obstacle_items = []
        self.vehicle_items = {}
        self.path_items = {}
        self.interface_items = {}
        self.conflict_items = {}
        self.backbone_visualizer = None
        self.heatmap_item = None
        
        # 性能优化
        self.update_counter = 0
        self.full_update_interval = 10
        
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
    
    def set_backbone_network(self, backbone_network):
        """设置骨干路径网络"""
        self.backbone_network = backbone_network
        self.update_backbone_visualization()
        self.update_interface_display()
    
    def set_traffic_manager(self, traffic_manager):
        """设置交通管理器"""
        self.traffic_manager = traffic_manager
    
    def _reset_items(self):
        """重置图形项"""
        self.obstacle_items = []
        self.vehicle_items = {}
        self.path_items = {}
        self.interface_items = {}
        self.conflict_items = {}
        self.backbone_visualizer = None
        self.heatmap_item = None
    
    def draw_environment(self):
        """绘制环境"""
        if not self.env:
            return
        
        # 背景
        gradient = QLinearGradient(0, 0, self.env.width, self.env.height)
        gradient.setColorAt(0, QColor(30, 30, 46))
        gradient.setColorAt(1, QColor(45, 45, 60))
        
        background = QGraphicsRectItem(0, 0, self.env.width, self.env.height)
        background.setBrush(QBrush(gradient))
        background.setPen(QPen(Qt.NoPen))
        background.setZValue(-100)
        self.addItem(background)
        
        # 网格线
        if self.show_options['grid']:
            self._draw_grid()
        
        # 障碍物
        self._draw_obstacles()
        
        # 特殊点
        self._draw_special_points()
        
        # 车辆
        self.draw_vehicles()
    
    def _draw_grid(self):
        """绘制网格"""
        grid_size = 20
        pen = QPen(QColor(69, 71, 90, 100))
        pen.setStyle(Qt.DotLine)
        
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
        # 创建障碍物图层
        obstacle_layer = QGraphicsItemGroup()
        
        for x, y in self.env.obstacle_points:
            rect = QGraphicsRectItem(x, y, 1, 1)
            rect.setBrush(QBrush(QColor(88, 91, 112)))
            rect.setPen(QPen(QColor(69, 71, 90), 0.1))
            obstacle_layer.addToGroup(rect)
            self.obstacle_items.append(rect)
        
        obstacle_layer.setZValue(-50)
        self.addItem(obstacle_layer)
    
    def _draw_special_points(self):
        """绘制特殊点（装载点、卸载点等）"""
        # 装载点
        for i, point in enumerate(self.env.loading_points):
            self._draw_loading_point(i, point)
        
        # 卸载点
        for i, point in enumerate(self.env.unloading_points):
            self._draw_unloading_point(i, point)
        
        # 停车区
        for i, point in enumerate(getattr(self.env, 'parking_areas', [])):
            self._draw_parking_area(i, point)
    
    def _draw_loading_point(self, index, point):
        """绘制装载点"""
        x, y = point[0], point[1]
        
        # 装载区域 - 渐变效果
        gradient = QRadialGradient(x, y, 10)
        gradient.setColorAt(0, QColor(166, 227, 161, 200))
        gradient.setColorAt(1, QColor(166, 227, 161, 50))
        
        area = QGraphicsEllipseItem(x-10, y-10, 20, 20)
        area.setBrush(QBrush(gradient))
        area.setPen(QPen(QColor(166, 227, 161), 2))
        area.setZValue(-20)
        self.addItem(area)
        
        # 中心标记
        center = QGraphicsEllipseItem(x-3, y-3, 6, 6)
        center.setBrush(QBrush(QColor(134, 239, 128)))
        center.setPen(QPen(Qt.black, 1))
        center.setZValue(-10)
        self.addItem(center)
        
        # 标签
        if self.show_options['labels']:
            text = QGraphicsTextItem(f"L{index+1}")
            text.setPos(x-8, y-15)
            text.setDefaultTextColor(QColor(166, 227, 161))
            text.setFont(QFont("Arial", 4, QFont.Bold))
            text.setZValue(-10)
            self.addItem(text)
    
    def _draw_unloading_point(self, index, point):
        """绘制卸载点"""
        x, y = point[0], point[1]
        
        # 卸载区域 - 渐变效果
        gradient = QRadialGradient(x, y, 10)
        gradient.setColorAt(0, QColor(243, 139, 168, 200))
        gradient.setColorAt(1, QColor(243, 139, 168, 50))
        
        area = QGraphicsEllipseItem(x-10, y-10, 20, 20)
        area.setBrush(QBrush(gradient))
        area.setPen(QPen(QColor(243, 139, 168), 2))
        area.setZValue(-20)
        self.addItem(area)
        
        # 中心标记
        center = QGraphicsRectItem(x-3, y-3, 6, 6)
        center.setBrush(QBrush(QColor(245, 194, 231)))
        center.setPen(QPen(Qt.black, 1))
        center.setZValue(-10)
        self.addItem(center)
        
        # 标签
        if self.show_options['labels']:
            text = QGraphicsTextItem(f"U{index+1}")
            text.setPos(x-8, y-15)
            text.setDefaultTextColor(QColor(243, 139, 168))
            text.setFont(QFont("Arial", 4, QFont.Bold))
            text.setZValue(-10)
            self.addItem(text)
    
    def _draw_parking_area(self, index, point):
        """绘制停车区"""
        x, y = point[0], point[1]
        
        # 停车区域
        gradient = QRadialGradient(x, y, 12)
        gradient.setColorAt(0, QColor(137, 180, 250, 200))
        gradient.setColorAt(1, QColor(137, 180, 250, 50))
        
        area = QGraphicsEllipseItem(x-12, y-12, 24, 24)
        area.setBrush(QBrush(gradient))
        area.setPen(QPen(QColor(137, 180, 250), 2, Qt.DashLine))
        area.setZValue(-20)
        self.addItem(area)
        
        # 中心标记 - 菱形
        diamond = QPolygonF([
            QPointF(x, y-4),
            QPointF(x+4, y),
            QPointF(x, y+4),
            QPointF(x-4, y)
        ])
        center = QGraphicsPolygonItem(diamond)
        center.setBrush(QBrush(QColor(147, 153, 274)))
        center.setPen(QPen(Qt.black, 1))
        center.setZValue(-10)
        self.addItem(center)
        
        # 标签
        if self.show_options['labels']:
            text = QGraphicsTextItem(f"P{index+1}")
            text.setPos(x-8, y-20)
            text.setDefaultTextColor(QColor(137, 180, 250))
            text.setFont(QFont("Arial", 4, QFont.Bold))
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
            vehicle_item = AnimatedVehicleItem(vehicle_id, vehicle_data)
            self.addItem(vehicle_item)
            self.vehicle_items[vehicle_id] = vehicle_item
            
            # 绘制路径
            if (self.show_options['trajectories'] and 
                'path' in vehicle_data and vehicle_data['path']):
                self._draw_vehicle_path(vehicle_id, vehicle_data)
    
    def _draw_vehicle_path(self, vehicle_id, vehicle_data):
        """绘制车辆路径"""
        path = vehicle_data['path']
        path_structure = vehicle_data.get('path_structure', {})
        
        if not path or len(path) < 2:
            return
        
        painter_path = QPainterPath()
        painter_path.moveTo(path[0][0], path[0][1])
        
        for point in path[1:]:
            painter_path.lineTo(point[0], point[1])
        
        path_item = QGraphicsPathItem(painter_path)
        
        # 根据路径类型设置样式
        path_type = path_structure.get('type', 'direct')
        if path_type in ['interface_assisted', 'backbone_only']:
            # 骨干辅助路径 - 实线
            pen = QPen(QColor(166, 227, 161, 200), 2)
            pen.setCapStyle(Qt.RoundCap)
        else:
            # 直接路径 - 虚线
            pen = QPen(QColor(137, 180, 250, 180), 1.5)
            pen.setStyle(Qt.DashLine)
            pen.setDashPattern([5, 3])
        
        path_item.setPen(pen)
        path_item.setZValue(5)
        self.addItem(path_item)
        self.path_items[vehicle_id] = path_item
    
    def update_vehicles(self):
        """更新车辆显示"""
        if not self.env:
            return
        
        self.update_counter += 1
        
        for vehicle_id, vehicle_data in self.env.vehicles.items():
            if vehicle_id in self.vehicle_items:
                self.vehicle_items[vehicle_id].update_data(vehicle_data)
            else:
                # 添加新车辆
                vehicle_item = AnimatedVehicleItem(vehicle_id, vehicle_data)
                self.addItem(vehicle_item)
                self.vehicle_items[vehicle_id] = vehicle_item
            
            # 更新路径
            if self.show_options['trajectories']:
                if vehicle_id in self.path_items:
                    self.removeItem(self.path_items[vehicle_id])
                
                if 'path' in vehicle_data and vehicle_data['path']:
                    self._draw_vehicle_path(vehicle_id, vehicle_data)
        
        # 定期全面更新
        if self.update_counter % self.full_update_interval == 0:
            self.update_conflicts()
            self.update_interface_display()
    
    def update_backbone_visualization(self):
        """更新骨干路径可视化"""
        if self.backbone_visualizer:
            self.removeItem(self.backbone_visualizer)
        
        if self.backbone_network and self.show_options['backbone']:
            self.backbone_visualizer = BackbonePathVisualization(self.backbone_network)
            self.addItem(self.backbone_visualizer)
    
    def update_interface_display(self):
        """更新接口显示"""
        # 清除现有接口项
        for item in self.interface_items.values():
            self.removeItem(item)
        self.interface_items.clear()
        
        if (not self.show_options['interfaces'] or not self.backbone_network or 
            not hasattr(self.backbone_network, 'backbone_interfaces')):
            return
        
        # 添加接口项
        for interface_id, interface in self.backbone_network.backbone_interfaces.items():
            interface_item = EnhancedInterfaceItem(interface)
            self.addItem(interface_item)
            self.interface_items[interface_id] = interface_item
    
    def update_conflicts(self):
        """更新冲突显示"""
        # 清除旧冲突
        for item in self.conflict_items.values():
            self.removeItem(item)
        self.conflict_items.clear()
        
        if (not self.show_options['conflicts'] or not self.traffic_manager):
            return
        
        # 获取当前冲突
        conflicts = self.traffic_manager.detect_all_conflicts()
        
        for i, conflict in enumerate(conflicts[:10]):  # 最多显示10个冲突
            conflict_item = ConflictVisualization(conflict)
            self.addItem(conflict_item)
            self.conflict_items[i] = conflict_item
    
    def toggle_show_option(self, option, show):
        """切换显示选项"""
        if option in self.show_options:
            self.show_options[option] = show
            
            # 触发相应更新
            if option == 'backbone':
                self.update_backbone_visualization()
            elif option == 'interfaces':
                self.update_interface_display()
            elif option == 'conflicts':
                self.update_conflicts()
            elif option == 'grid':
                self.draw_environment()
            elif option == 'trajectories':
                self.draw_vehicles()

class OptimizedMineView(QGraphicsView):
    """优化的矿场图形视图"""
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.TextAntialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        
        # 创建场景
        self.mine_scene = OptimizedMineScene(self)
        self.setScene(self.mine_scene)
        
        # 缩放限制
        self.min_scale = 0.1
        self.max_scale = 5.0
        self.current_scale = 1.0
        
        # 坐标显示
        self.coord_label = QLabel(self)
        self.coord_label.setStyleSheet(
            "background-color: rgba(49, 50, 68, 0.9); "
            "color: #cdd6f4; "
            "padding: 8px; "
            "border-radius: 4px; "
            "font-weight: bold;"
        )
        self.coord_label.setAlignment(Qt.AlignCenter)
        self.coord_label.setFixedSize(150, 30)
        self.coord_label.move(10, 10)
        self.coord_label.show()
        
        # 缩放指示器
        self.zoom_label = QLabel(self)
        self.zoom_label.setStyleSheet(
            "background-color: rgba(49, 50, 68, 0.9); "
            "color: #cdd6f4; "
            "padding: 8px; "
            "border-radius: 4px; "
            "font-weight: bold;"
        )
        self.zoom_label.setAlignment(Qt.AlignCenter)
        self.zoom_label.setFixedSize(100, 30)
        self.zoom_label.move(170, 10)
        self.zoom_label.setText(f"缩放: {int(self.current_scale * 100)}%")
        self.zoom_label.show()
    
    def wheelEvent(self, event):
        """鼠标滚轮缩放"""
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1/1.15
        
        new_scale = self.current_scale * factor
        if self.min_scale <= new_scale <= self.max_scale:
            self.scale(factor, factor)
            self.current_scale = new_scale
            self.zoom_label.setText(f"缩放: {int(self.current_scale * 100)}%")
    
    def mouseMoveEvent(self, event):
        """鼠标移动显示坐标"""
        super().mouseMoveEvent(event)
        scene_pos = self.mapToScene(event.pos())
        x, y = scene_pos.x(), scene_pos.y()
        self.coord_label.setText(f"坐标: ({x:.1f}, {y:.1f})")
    
    def set_environment(self, env):
        """设置环境"""
        if env:
            # 先设置场景环境（更新场景尺寸）
            self.mine_scene.set_environment(env)
            # 再进行视图适配
            self.fitInView(self.mine_scene.sceneRect(), Qt.KeepAspectRatio)
            self.current_scale = self.transform().m11()
            self.zoom_label.setText(f"缩放: {int(self.current_scale * 100)}%")

    
    def set_backbone_network(self, backbone_network):
        """设置骨干网络 - 避免重绘"""
        # 只设置引用，不立即更新可视化
        self.mine_scene.backbone_network = backbone_network

    def set_traffic_manager(self, traffic_manager):
        """设置交通管理器 - 避免重绘"""
        # 只设置引用
        self.mine_scene.traffic_manager = traffic_manager
    
    def update_vehicles(self):
        """更新车辆"""
        self.mine_scene.update_vehicles()
    
    def toggle_show_option(self, option, show):
        """切换显示选项"""
        self.mine_scene.toggle_show_option(option, show)

class RealTimeStatsWidget(QWidget):
    """实时统计显示部件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scheduler = None
        self.traffic_manager = None
        
        layout = QVBoxLayout(self)
        
        # 标题
        title = QLabel("实时统计")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #f5e0dc;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # 创建统计网格
        stats_grid = QGridLayout()
        
        # 车辆统计
        self.active_vehicles_lcd = self._create_lcd("活跃车辆")
        stats_grid.addWidget(self.active_vehicles_lcd[0], 0, 0)
        stats_grid.addWidget(self.active_vehicles_lcd[1], 1, 0)
        
        self.idle_vehicles_lcd = self._create_lcd("空闲车辆")
        stats_grid.addWidget(self.idle_vehicles_lcd[0], 0, 1)
        stats_grid.addWidget(self.idle_vehicles_lcd[1], 1, 1)
        
        # 任务统计
        self.active_tasks_lcd = self._create_lcd("进行中任务")
        stats_grid.addWidget(self.active_tasks_lcd[0], 2, 0)
        stats_grid.addWidget(self.active_tasks_lcd[1], 3, 0)
        
        self.completed_tasks_lcd = self._create_lcd("完成任务")
        stats_grid.addWidget(self.completed_tasks_lcd[0], 2, 1)
        stats_grid.addWidget(self.completed_tasks_lcd[1], 3, 1)
        
        # 效率指标
        self.utilization_bar = self._create_progress_bar("平均利用率")
        stats_grid.addWidget(self.utilization_bar[0], 4, 0, 1, 2)
        stats_grid.addWidget(self.utilization_bar[1], 5, 0, 1, 2)
        
        self.backbone_usage_bar = self._create_progress_bar("骨干使用率")
        stats_grid.addWidget(self.backbone_usage_bar[0], 6, 0, 1, 2)
        stats_grid.addWidget(self.backbone_usage_bar[1], 7, 0, 1, 2)
        
        layout.addLayout(stats_grid)
        
        # 冲突统计
        conflict_group = QGroupBox("冲突统计")
        conflict_layout = QVBoxLayout()
        
        self.conflicts_label = QLabel("当前冲突: 0")
        self.resolved_label = QLabel("已解决: 0")
        self.ecbs_calls_label = QLabel("ECBS调用: 0")
        
        conflict_layout.addWidget(self.conflicts_label)
        conflict_layout.addWidget(self.resolved_label)
        conflict_layout.addWidget(self.ecbs_calls_label)
        
        conflict_group.setLayout(conflict_layout)
        layout.addWidget(conflict_group)
        
        layout.addStretch()
    
    def _create_lcd(self, label_text):
        """创建LCD显示器"""
        label = QLabel(label_text)
        label.setAlignment(Qt.AlignCenter)
        
        lcd = QLCDNumber()
        lcd.setSegmentStyle(QLCDNumber.Flat)
        lcd.setDigitCount(3)
        lcd.setMinimumHeight(40)
        
        return (label, lcd)
    
    def _create_progress_bar(self, label_text):
        """创建进度条"""
        label = QLabel(label_text)
        label.setAlignment(Qt.AlignCenter)
        
        progress = QProgressBar()
        progress.setMinimumHeight(25)
        progress.setTextVisible(True)
        
        return (label, progress)
    
    def set_scheduler(self, scheduler):
        """设置调度器"""
        self.scheduler = scheduler
    
    def set_traffic_manager(self, traffic_manager):
        """设置交通管理器"""
        self.traffic_manager = traffic_manager
    
    def update_stats(self):
        """更新统计显示"""
        if not self.scheduler:
            return
        
        stats = self.scheduler.get_comprehensive_stats()
        
        # 更新车辆统计
        real_time = stats.get('real_time', {})
        self.active_vehicles_lcd[1].display(real_time.get('active_vehicles', 0))
        self.idle_vehicles_lcd[1].display(real_time.get('idle_vehicles', 0))
        
        # 更新任务统计
        self.active_tasks_lcd[1].display(real_time.get('active_tasks', 0))
        self.completed_tasks_lcd[1].display(stats.get('completed_tasks', 0))
        
        # 更新效率指标
        avg_utilization = 0
        if 'vehicle_utilization' in stats:
            utilizations = list(stats['vehicle_utilization'].values())
            if utilizations:
                avg_utilization = sum(utilizations) / len(utilizations) * 100
        
        self.utilization_bar[1].setValue(int(avg_utilization))
        
        backbone_rate = stats.get('backbone_utilization_rate', 0) * 100
        self.backbone_usage_bar[1].setValue(int(backbone_rate))
        
        # 更新冲突统计
        if self.traffic_manager:
            traffic_stats = self.traffic_manager.get_comprehensive_stats()
            self.conflicts_label.setText(f"当前冲突: {traffic_stats.get('conflicts_detected', 0)}")
            self.resolved_label.setText(f"已解决: {traffic_stats.get('conflicts_resolved', 0)}")
            self.ecbs_calls_label.setText(f"ECBS调用: {traffic_stats.get('ecbs_calls', 0)}")

class PerformanceChartWidget(QWidget):
    """性能图表部件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scheduler = None
        self.history_length = 60  # 显示最近60个数据点
        
        layout = QVBoxLayout(self)
        
        # 创建图表
        self.chart = QChart()
        self.chart.setTitle("系统性能趋势")
        self.chart.setAnimationOptions(QChart.SeriesAnimations)
        self.chart.setBackgroundBrush(QBrush(QColor(49, 50, 68)))
        self.chart.setTitleBrush(QBrush(QColor(245, 224, 220)))
        
        # 创建系列
        self.utilization_series = QLineSeries()
        self.utilization_series.setName("车辆利用率")
        self.utilization_series.setPen(QPen(QColor(166, 227, 161), 2))
        
        self.completion_series = QLineSeries()
        self.completion_series.setName("任务完成率")
        self.completion_series.setPen(QPen(QColor(137, 180, 250), 2))
        
        self.chart.addSeries(self.utilization_series)
        self.chart.addSeries(self.completion_series)
        
        # 创建坐标轴
        self.axis_x = QValueAxis()
        self.axis_x.setLabelFormat("%d")
        self.axis_x.setTitleText("时间 (秒)")
        self.axis_x.setRange(0, self.history_length)
        self.axis_x.setGridLineVisible(True)
        self.axis_x.setLabelsColor(QColor(205, 214, 244))
        self.axis_x.setTitleBrush(QBrush(QColor(205, 214, 244)))
        
        self.axis_y = QValueAxis()
        self.axis_y.setLabelFormat("%.0f%%")
        self.axis_y.setTitleText("百分比")
        self.axis_y.setRange(0, 100)
        self.axis_y.setGridLineVisible(True)
        self.axis_y.setLabelsColor(QColor(205, 214, 244))
        self.axis_y.setTitleBrush(QBrush(QColor(205, 214, 244)))
        
        self.chart.addAxis(self.axis_x, Qt.AlignBottom)
        self.chart.addAxis(self.axis_y, Qt.AlignLeft)
        
        self.utilization_series.attachAxis(self.axis_x)
        self.utilization_series.attachAxis(self.axis_y)
        self.completion_series.attachAxis(self.axis_x)
        self.completion_series.attachAxis(self.axis_y)
        
        # 创建图表视图
        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.Antialiasing)
        
        layout.addWidget(self.chart_view)
        
        # 数据缓存
        self.time_data = []
        self.utilization_data = []
        self.completion_data = []
        self.current_time = 0
    
    def set_scheduler(self, scheduler):
        """设置调度器"""
        self.scheduler = scheduler
    
    def update_chart(self):
        """更新图表"""
        if not self.scheduler:
            return
        
        stats = self.scheduler.get_comprehensive_stats()
        
        # 计算当前值
        avg_utilization = 0
        if 'vehicle_utilization' in stats:
            utilizations = list(stats['vehicle_utilization'].values())
            if utilizations:
                avg_utilization = sum(utilizations) / len(utilizations) * 100
        
        completion_rate = 0
        if stats.get('total_tasks', 0) > 0:
            completion_rate = stats.get('completed_tasks', 0) / stats['total_tasks'] * 100
        
        # 添加新数据
        self.current_time += 1
        self.time_data.append(self.current_time)
        self.utilization_data.append(avg_utilization)
        self.completion_data.append(completion_rate)
        
        # 限制数据长度
        if len(self.time_data) > self.history_length:
            self.time_data.pop(0)
            self.utilization_data.pop(0)
            self.completion_data.pop(0)
        
        # 更新系列
        self.utilization_series.clear()
        self.completion_series.clear()
        
        for i in range(len(self.time_data)):
            self.utilization_series.append(i, self.utilization_data[i])
            self.completion_series.append(i, self.completion_data[i])
        
        # 调整X轴范围
        if self.current_time > self.history_length:
            self.axis_x.setRange(0, self.history_length)

class VehicleDetailPanel(QWidget):
    """车辆详细信息面板"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.env = None
        self.scheduler = None
        
        layout = QVBoxLayout(self)
        
        # 车辆选择
        select_layout = QHBoxLayout()
        select_layout.addWidget(QLabel("选择车辆:"))
        
        self.vehicle_combo = QComboBox()
        self.vehicle_combo.currentIndexChanged.connect(self.update_vehicle_info)
        select_layout.addWidget(self.vehicle_combo, 1)
        
        layout.addLayout(select_layout)
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        
        # 基本信息标签页
        self.basic_tab = self._create_basic_info_tab()
        self.tab_widget.addTab(self.basic_tab, "基本信息")
        
        # 任务信息标签页
        self.task_tab = self._create_task_info_tab()
        self.tab_widget.addTab(self.task_tab, "任务信息")
        
        # 性能统计标签页
        self.stats_tab = self._create_stats_tab()
        self.tab_widget.addTab(self.stats_tab, "性能统计")
        
        layout.addWidget(self.tab_widget)
    
    def _create_basic_info_tab(self):
        """创建基本信息标签页"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        self.info_labels = {}
        info_items = [
            ("位置:", "position"),
            ("状态:", "status"),
            ("载重:", "load"),
            ("速度:", "speed"),
            ("朝向:", "heading"),
            ("总里程:", "total_distance")
        ]
        
        for i, (label_text, key) in enumerate(info_items):
            label = QLabel(label_text)
            label.setStyleSheet("font-weight: bold;")
            layout.addWidget(label, i, 0, Qt.AlignRight)
            
            value_label = QLabel("--")
            value_label.setStyleSheet(
                "background-color: #1e1e2e; "
                "padding: 5px; "
                "border-radius: 4px; "
                "color: #a6e3a1;"
            )
            layout.addWidget(value_label, i, 1)
            self.info_labels[key] = value_label
        
        layout.setColumnStretch(1, 1)
        return widget
    
    def _create_task_info_tab(self):
        """创建任务信息标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 当前任务
        current_group = QGroupBox("当前任务")
        current_layout = QVBoxLayout()
        
        self.current_task_tree = QTreeWidget()
        self.current_task_tree.setHeaderLabels(["属性", "值"])
        self.current_task_tree.setAlternatingRowColors(True)
        current_layout.addWidget(self.current_task_tree)
        
        current_group.setLayout(current_layout)
        layout.addWidget(current_group)
        
        # 任务队列
        queue_group = QGroupBox("任务队列")
        queue_layout = QVBoxLayout()
        
        self.task_queue_list = QListWidget()
        queue_layout.addWidget(self.task_queue_list)
        
        queue_group.setLayout(queue_layout)
        layout.addWidget(queue_group)
        
        return widget
    
    def _create_stats_tab(self):
        """创建性能统计标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 效率指标
        efficiency_group = QGroupBox("效率指标")
        efficiency_layout = QGridLayout()
        
        self.stats_labels = {}
        stats_items = [
            ("利用率:", "utilization"),
            ("完成任务:", "completed_tasks"),
            ("骨干使用:", "backbone_usage"),
            ("接口效率:", "interface_efficiency")
        ]
        
        for i, (label_text, key) in enumerate(stats_items):
            label = QLabel(label_text)
            label.setStyleSheet("font-weight: bold;")
            efficiency_layout.addWidget(label, i // 2, (i % 2) * 2, Qt.AlignRight)
            
            value_label = QLabel("--")
            value_label.setStyleSheet(
                "background-color: #1e1e2e; "
                "padding: 5px; "
                "border-radius: 4px; "
                "color: #89b4fa;"
            )
            efficiency_layout.addWidget(value_label, i // 2, (i % 2) * 2 + 1)
            self.stats_labels[key] = value_label
        
        efficiency_group.setLayout(efficiency_layout)
        layout.addWidget(efficiency_group)
        
        # 历史记录
        history_group = QGroupBox("性能历史")
        history_layout = QVBoxLayout()
        
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(4)
        self.history_table.setHorizontalHeaderLabels(["时间", "任务", "质量", "耗时"])
        self.history_table.horizontalHeader().setStretchLastSection(True)
        history_layout.addWidget(self.history_table)
        
        history_group.setLayout(history_layout)
        layout.addWidget(history_group)
        
        return widget
    
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
        
        # 更新基本信息 - 适配VehicleInfo对象
        pos = vehicle.position if hasattr(vehicle, 'position') else (0, 0, 0)
        self.info_labels["position"].setText(f"({pos[0]:.1f}, {pos[1]:.1f})")
        
        status_map = {
            'idle': '空闲', 'moving': '移动中',
            'loading': '装载中', 'unloading': '卸载中',
            'waiting': '等待中', 'maintenance': '维护中'
        }
        status = vehicle.status if hasattr(vehicle, 'status') else 'idle'
        self.info_labels["status"].setText(status_map.get(status, status))
        
        current_load = vehicle.current_load if hasattr(vehicle, 'current_load') else 0
        max_load = vehicle.max_load if hasattr(vehicle, 'max_load') else 100
        self.info_labels["load"].setText(f"{current_load}/{max_load} ({current_load/max_load*100:.0f}%)")
        
        # 从调度器获取详细信息
        if self.scheduler:
            vehicle_info = self.scheduler.get_vehicle_info(v_id)
            if vehicle_info:
                self._update_detailed_info(vehicle_info)
            else:
                # 直接使用VehicleInfo对象的属性
                self.info_labels["speed"].setText(f"{getattr(vehicle, 'speed', 1.0):.1f}")
                
                heading = math.degrees(pos[2]) if len(pos) > 2 else 0
                self.info_labels["heading"].setText(f"{heading:.0f}°")
                
                total_distance = getattr(vehicle, 'total_distance', 0)
                self.info_labels["total_distance"].setText(f"{total_distance:.1f}")
    
    def _update_detailed_info(self, vehicle_info):
        """更新详细信息"""
        # 基本信息
        self.info_labels["speed"].setText(f"{vehicle_info.get('speed', 1.0):.1f}")
        
        pos = vehicle_info.get('position', (0, 0, 0))
        heading = math.degrees(pos[2]) if len(pos) > 2 else 0
        self.info_labels["heading"].setText(f"{heading:.0f}°")
        
        self.info_labels["total_distance"].setText(f"{vehicle_info.get('total_distance', 0):.1f}")
        
        # 当前任务
        self.current_task_tree.clear()
        current_task = vehicle_info.get('current_task')
        if current_task:
            task_item = QTreeWidgetItem(["任务ID", current_task['task_id']])
            self.current_task_tree.addTopLevelItem(task_item)
            
            QTreeWidgetItem(task_item, ["类型", current_task['task_type']])
            QTreeWidgetItem(task_item, ["进度", f"{current_task['progress']*100:.0f}%"])
            QTreeWidgetItem(task_item, ["质量", f"{current_task['quality_score']:.2f}"])
            
            # 路径结构
            path_structure = current_task.get('path_structure', {})
            if path_structure:
                structure_item = QTreeWidgetItem(task_item, ["路径结构", ""])
                QTreeWidgetItem(structure_item, ["类型", path_structure.get('type', 'unknown')])
                
                if path_structure.get('interface_id'):
                    QTreeWidgetItem(structure_item, ["接口", path_structure['interface_id']])
            
            task_item.setExpanded(True)
        
        # 任务队列
        self.task_queue_list.clear()
        task_queue = vehicle_info.get('task_queue', [])
        for task in task_queue:
            item_text = f"{task['task_id']} - {task['task_type']} (优先级: {task['priority']})"
            self.task_queue_list.addItem(item_text)
        
        # 效率指标
        self.stats_labels["utilization"].setText(f"{vehicle_info.get('utilization_rate', 0)*100:.0f}%")
        self.stats_labels["completed_tasks"].setText(str(vehicle_info.get('completed_tasks', 0)))
        
        efficiency = vehicle_info.get('efficiency_metrics', {})
        backbone_rate = efficiency.get('backbone_usage_rate', 0)
        self.stats_labels["backbone_usage"].setText(f"{backbone_rate*100:.0f}%")
        
        interface_eff = efficiency.get('interface_efficiency', 0)
        self.stats_labels["interface_efficiency"].setText(f"{interface_eff:.2f}")
        
        # 性能历史
        self.history_table.setRowCount(0)
        history = vehicle_info.get('performance_history', [])
        for record in history[-10:]:  # 最近10条
            row = self.history_table.rowCount()
            self.history_table.insertRow(row)
            
            # 时间
            timestamp = record.get('timestamp', 0)
            time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
            self.history_table.setItem(row, 0, QTableWidgetItem(time_str))
            
            # 任务指标
            metrics = record.get('task_metrics', {})
            self.history_table.setItem(row, 1, QTableWidgetItem("完成"))
            self.history_table.setItem(row, 2, QTableWidgetItem(f"{metrics.get('quality_score', 0):.2f}"))
            self.history_table.setItem(row, 3, QTableWidgetItem(f"{metrics.get('execution_time', 0):.0f}s"))

class AdvancedControlPanel(QWidget):
    """高级控制面板"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = None
        
        layout = QVBoxLayout(self)
        
        # 创建折叠式部件容器
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # 环境控制
        self.env_group = self._create_environment_group()
        scroll_layout.addWidget(self.env_group)
        
        # 骨干网络控制
        self.backbone_group = self._create_backbone_group()
        scroll_layout.addWidget(self.backbone_group)
        
        # 调度策略
        self.scheduling_group = self._create_scheduling_group()
        scroll_layout.addWidget(self.scheduling_group)
        
        # 仿真控制
        self.simulation_group = self._create_simulation_group()
        scroll_layout.addWidget(self.simulation_group)
        
        # 可视化选项
        self.display_group = self._create_display_group()
        scroll_layout.addWidget(self.display_group)
        
        # 高级选项
        self.advanced_group = self._create_advanced_group()
        scroll_layout.addWidget(self.advanced_group)
        
        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
    
    def _create_environment_group(self):
        """创建环境控制组"""
        group = QGroupBox("环境控制")
        layout = QVBoxLayout()
        
        # 文件选择
        file_layout = QHBoxLayout()
        self.file_label = QLabel("未选择文件")
        self.file_label.setStyleSheet(
            "background-color: #1e1e2e; "
            "padding: 8px; "
            "border-radius: 4px;"
        )
        self.browse_btn = QPushButton("浏览...")
        self.load_btn = QPushButton("加载")
        self.save_btn = QPushButton("保存")
        
        file_layout.addWidget(self.file_label, 1)
        file_layout.addWidget(self.browse_btn)
        layout.addLayout(file_layout)
        
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.save_btn)
        layout.addLayout(btn_layout)
        
        group.setLayout(layout)
        return group
    
    def _create_backbone_group(self):
        """创建骨干网络控制组"""
        group = QGroupBox("骨干网络")
        layout = QVBoxLayout()
        
        # 参数设置
        param_layout = QGridLayout()
        
        param_layout.addWidget(QLabel("质量阈值:"), 0, 0)
        self.quality_spin = QDoubleSpinBox()
        self.quality_spin.setRange(0.1, 1.0)
        self.quality_spin.setSingleStep(0.1)
        self.quality_spin.setValue(0.6)
        param_layout.addWidget(self.quality_spin, 0, 1)
        
        param_layout.addWidget(QLabel("接口间距:"), 1, 0)
        self.interface_spacing_spin = QSpinBox()
        self.interface_spacing_spin.setRange(5, 20)
        self.interface_spacing_spin.setValue(8)
        param_layout.addWidget(self.interface_spacing_spin, 1, 1)
        
        layout.addLayout(param_layout)
        
        # 生成按钮
        self.generate_btn = QPushButton("生成骨干路径")
        layout.addWidget(self.generate_btn)
        
        # 统计信息
        self.backbone_stats_label = QLabel("路径: 0 条, 接口: 0 个")
        self.backbone_stats_label.setStyleSheet("color: #a6e3a1;")
        layout.addWidget(self.backbone_stats_label)
        
        group.setLayout(layout)
        return group
    
    def _create_scheduling_group(self):
        """创建调度策略组"""
        group = QGroupBox("调度策略")
        layout = QVBoxLayout()
        
        # 调度器选择
        scheduler_layout = QHBoxLayout()
        scheduler_layout.addWidget(QLabel("调度器:"))
        self.scheduler_combo = QComboBox()
        self.scheduler_combo.addItems(["标准调度", "ECBS增强", "预测性调度"])
        scheduler_layout.addWidget(self.scheduler_combo, 1)
        layout.addLayout(scheduler_layout)
        
        # 任务分配
        task_layout = QGridLayout()
        
        task_layout.addWidget(QLabel("装载点:"), 0, 0)
        self.loading_combo = QComboBox()
        task_layout.addWidget(self.loading_combo, 0, 1)
        
        task_layout.addWidget(QLabel("卸载点:"), 1, 0)
        self.unloading_combo = QComboBox()
        task_layout.addWidget(self.unloading_combo, 1, 1)
        
        layout.addLayout(task_layout)
        
        # 任务控制按钮
        task_btn_layout = QHBoxLayout()
        self.assign_task_btn = QPushButton("分配任务")
        self.assign_all_btn = QPushButton("批量分配")
        self.cancel_task_btn = QPushButton("取消任务")
        
        task_btn_layout.addWidget(self.assign_task_btn)
        task_btn_layout.addWidget(self.assign_all_btn)
        task_btn_layout.addWidget(self.cancel_task_btn)
        layout.addLayout(task_btn_layout)
        
        group.setLayout(layout)
        return group
    
    def _create_simulation_group(self):
        """创建仿真控制组"""
        group = QGroupBox("仿真控制")
        layout = QVBoxLayout()
        
        # 速度控制
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("仿真速度:"))
        
        self.speed_dial = QDial()
        self.speed_dial.setRange(1, 200)
        self.speed_dial.setValue(100)
        self.speed_dial.setNotchesVisible(True)
        self.speed_dial.setMaximumSize(60, 60)
        speed_layout.addWidget(self.speed_dial)
        
        self.speed_label = QLabel("1.0x")
        self.speed_label.setMinimumWidth(50)
        self.speed_label.setAlignment(Qt.AlignCenter)
        speed_layout.addWidget(self.speed_label)
        
        layout.addLayout(speed_layout)
        
        # 控制按钮
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始")
        self.pause_btn = QPushButton("暂停")
        self.reset_btn = QPushButton("重置")
        
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.pause_btn)
        btn_layout.addWidget(self.reset_btn)
        layout.addLayout(btn_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)
        
        # 时间显示
        self.time_label = QLabel("仿真时间: 00:00:00")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("font-size: 14px; color: #f5e0dc;")
        layout.addWidget(self.time_label)
        
        group.setLayout(layout)
        return group
    
    def _create_display_group(self):
        """创建显示选项组"""
        group = QGroupBox("显示选项")
        layout = QVBoxLayout()
        
        self.show_backbone_cb = QCheckBox("显示骨干路径")
        self.show_backbone_cb.setChecked(True)
        
        self.show_interfaces_cb = QCheckBox("显示接口点")
        self.show_interfaces_cb.setChecked(True)
        
        self.show_paths_cb = QCheckBox("显示车辆路径")
        self.show_paths_cb.setChecked(True)
        
        self.show_conflicts_cb = QCheckBox("显示冲突")
        self.show_conflicts_cb.setChecked(True)
        
        self.show_grid_cb = QCheckBox("显示网格")
        self.show_grid_cb.setChecked(True)
        
        self.show_labels_cb = QCheckBox("显示标签")
        self.show_labels_cb.setChecked(True)
        
        layout.addWidget(self.show_backbone_cb)
        layout.addWidget(self.show_interfaces_cb)
        layout.addWidget(self.show_paths_cb)
        layout.addWidget(self.show_conflicts_cb)
        layout.addWidget(self.show_grid_cb)
        layout.addWidget(self.show_labels_cb)
        
        group.setLayout(layout)
        return group
    
    def _create_advanced_group(self):
        """创建高级选项组"""
        group = QGroupBox("高级选项")
        layout = QVBoxLayout()
        
        # ECBS参数
        ecbs_layout = QGridLayout()
        ecbs_layout.addWidget(QLabel("冲突检测间隔:"), 0, 0)
        self.conflict_interval_spin = QSpinBox()
        self.conflict_interval_spin.setRange(1, 60)
        self.conflict_interval_spin.setValue(10)
        self.conflict_interval_spin.setSuffix(" 秒")
        ecbs_layout.addWidget(self.conflict_interval_spin, 0, 1)
        
        ecbs_layout.addWidget(QLabel("安全距离:"), 1, 0)
        self.safety_distance_spin = QDoubleSpinBox()
        self.safety_distance_spin.setRange(1.0, 20.0)
        self.safety_distance_spin.setValue(6.0)
        self.safety_distance_spin.setSuffix(" m")
        ecbs_layout.addWidget(self.safety_distance_spin, 1, 1)
        
        layout.addLayout(ecbs_layout)
        
        # 性能选项
        self.batch_planning_cb = QCheckBox("启用批量规划")
        self.batch_planning_cb.setChecked(True)
        
        self.predictive_scheduling_cb = QCheckBox("启用预测调度")
        self.predictive_scheduling_cb.setChecked(True)
        
        self.cache_optimization_cb = QCheckBox("启用缓存优化")
        self.cache_optimization_cb.setChecked(True)
        
        layout.addWidget(self.batch_planning_cb)
        layout.addWidget(self.predictive_scheduling_cb)
        layout.addWidget(self.cache_optimization_cb)
        
        # 调试选项
        debug_btn = QPushButton("调试信息")
        debug_btn.clicked.connect(self._show_debug_info)
        layout.addWidget(debug_btn)
        
        group.setLayout(layout)
        return group
    
    def _show_debug_info(self):
        """显示调试信息"""
        if self.main_window:
            self.main_window.show_debug_dialog()
    
    def set_main_window(self, main_window):
        """设置主窗口引用"""
        self.main_window = main_window
        
        # 连接信号
        self.browse_btn.clicked.connect(main_window.browse_file)
        self.load_btn.clicked.connect(main_window.load_environment)
        self.save_btn.clicked.connect(main_window.save_environment)
        self.generate_btn.clicked.connect(main_window.generate_backbone_network)
        
        self.scheduler_combo.currentIndexChanged.connect(main_window.change_scheduler)
        self.assign_task_btn.clicked.connect(main_window.assign_vehicle_task)
        self.assign_all_btn.clicked.connect(main_window.assign_all_vehicles)
        self.cancel_task_btn.clicked.connect(main_window.cancel_vehicle_task)
        
        self.start_btn.clicked.connect(main_window.start_simulation)
        self.pause_btn.clicked.connect(main_window.pause_simulation)
        self.reset_btn.clicked.connect(main_window.reset_simulation)
        
        self.speed_dial.valueChanged.connect(main_window.update_simulation_speed)
        
        # 显示选项
        self.show_backbone_cb.stateChanged.connect(
            lambda s: main_window.toggle_display_option('backbone', s == Qt.Checked)
        )
        self.show_interfaces_cb.stateChanged.connect(
            lambda s: main_window.toggle_display_option('interfaces', s == Qt.Checked)
        )
        self.show_paths_cb.stateChanged.connect(
            lambda s: main_window.toggle_display_option('trajectories', s == Qt.Checked)
        )
        self.show_conflicts_cb.stateChanged.connect(
            lambda s: main_window.toggle_display_option('conflicts', s == Qt.Checked)
        )
        self.show_grid_cb.stateChanged.connect(
            lambda s: main_window.toggle_display_option('grid', s == Qt.Checked)
        )
        self.show_labels_cb.stateChanged.connect(
            lambda s: main_window.toggle_display_option('labels', s == Qt.Checked)
        )
        
        # 高级选项
        self.conflict_interval_spin.valueChanged.connect(main_window.update_conflict_interval)
        self.safety_distance_spin.valueChanged.connect(main_window.update_safety_distance)
        self.batch_planning_cb.stateChanged.connect(main_window.toggle_batch_planning)
        self.predictive_scheduling_cb.stateChanged.connect(main_window.toggle_predictive_scheduling)
        self.cache_optimization_cb.stateChanged.connect(main_window.toggle_cache_optimization)
    
    def update_backbone_stats(self, path_count, interface_count):
        """更新骨干网络统计"""
        self.backbone_stats_label.setText(f"路径: {path_count} 条, 接口: {interface_count} 个")

class EnhancedMineGUI(QMainWindow):
    """增强的露天矿多车协同调度系统GUI"""
    
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
        self.simulation_time = 0
        self.map_file_path = None
        
        # 设置样式
        #self.setStyleSheet(ENHANCED_STYLE)
        
        # 初始化UI
        self.init_ui()
        
        # 定时器
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(50)  # 20 FPS
        
        self.sim_timer = QTimer(self)
        self.sim_timer.timeout.connect(self.simulation_step)
        
        self.stats_timer = QTimer(self)
        self.stats_timer.timeout.connect(self.update_statistics)
        self.stats_timer.start(1000)  # 1秒更新一次统计
    
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("露天矿多车协同调度系统 - 优化版")
        self.setGeometry(50, 50, 1600, 900)
        
        # 设置窗口图标（如果有的话）
        # self.setWindowIcon(QIcon("icon.png"))
        
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建左侧停靠窗口 - 控制面板
        self.control_dock = QDockWidget("控制面板", self)
        self.control_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.control_panel = AdvancedControlPanel()
        self.control_panel.set_main_window(self)
        self.control_dock.setWidget(self.control_panel)
        self.control_dock.setMinimumWidth(300)
        self.control_dock.setMaximumWidth(400)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.control_dock)
        
        # 创建右侧停靠窗口 - 实时统计
        self.stats_dock = QDockWidget("实时统计", self)
        self.stats_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.stats_widget = RealTimeStatsWidget()
        self.stats_dock.setWidget(self.stats_widget)
        self.stats_dock.setMinimumWidth(250)
        self.stats_dock.setMaximumWidth(350)
        self.addDockWidget(Qt.RightDockWidgetArea, self.stats_dock)
        
        # 创建底部停靠窗口 - 车辆详情
        self.vehicle_dock = QDockWidget("车辆详情", self)
        self.vehicle_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.vehicle_panel = VehicleDetailPanel()
        self.vehicle_dock.setWidget(self.vehicle_panel)
        self.vehicle_dock.setMinimumHeight(200)
        self.vehicle_dock.setMaximumHeight(400)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.vehicle_dock)
        
        # 中央区域 - 主视图和图表
        central_splitter = QSplitter(Qt.Vertical)
        
        # 图形视图
        self.graphics_view = OptimizedMineView()
        central_splitter.addWidget(self.graphics_view)
        
        # 性能图表
        self.chart_widget = PerformanceChartWidget()
        central_splitter.addWidget(self.chart_widget)
        
        central_splitter.setStretchFactor(0, 3)
        central_splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(central_splitter)
        
        # 创建菜单和工具栏
        self.create_menu_bar()
        self.create_tool_bar()
        
        # 状态栏
        self.create_status_bar()
        
        # 创建系统日志窗口（初始隐藏）
        self.create_log_window()
    
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #313244;
                color: #cdd6f4;
                padding: 5px;
            }
            QMenuBar::item:selected {
                background-color: #45475a;
            }
            QMenu {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
            }
            QMenu::item:selected {
                background-color: #45475a;
            }
        """)
        
        # 文件菜单
        file_menu = menubar.addMenu("文件(&F)")
        
        new_action = QAction("新建环境(&N)", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_environment)
        file_menu.addAction(new_action)
        
        open_action = QAction("打开地图(&O)...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.browse_file)
        file_menu.addAction(open_action)
        
        save_action = QAction("保存环境(&S)", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_environment)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        export_action = QAction("导出结果(&E)...", self)
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("退出(&X)", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 视图菜单
        view_menu = menubar.addMenu("视图(&V)")
        
        view_menu.addAction(self.control_dock.toggleViewAction())
        view_menu.addAction(self.stats_dock.toggleViewAction())
        view_menu.addAction(self.vehicle_dock.toggleViewAction())
        
        view_menu.addSeparator()
        
        zoom_in_action = QAction("放大(&I)", self)
        zoom_in_action.setShortcut("Ctrl++")
        zoom_in_action.triggered.connect(self.zoom_in)
        view_menu.addAction(zoom_in_action)
        
        zoom_out_action = QAction("缩小(&O)", self)
        zoom_out_action.setShortcut("Ctrl+-")
        zoom_out_action.triggered.connect(self.zoom_out)
        view_menu.addAction(zoom_out_action)
        
        fit_view_action = QAction("适应视图(&F)", self)
        fit_view_action.setShortcut("Ctrl+0")
        fit_view_action.triggered.connect(self.fit_view)
        view_menu.addAction(fit_view_action)
        
        view_menu.addSeparator()
        
        fullscreen_action = QAction("全屏(&U)", self)
        fullscreen_action.setShortcut("F11")
        fullscreen_action.setCheckable(True)
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)
        
        # 仿真菜单
        sim_menu = menubar.addMenu("仿真(&S)")
        
        start_action = QAction("开始(&S)", self)
        start_action.setShortcut("F5")
        start_action.triggered.connect(self.start_simulation)
        sim_menu.addAction(start_action)
        
        pause_action = QAction("暂停(&P)", self)
        pause_action.setShortcut("F6")
        pause_action.triggered.connect(self.pause_simulation)
        sim_menu.addAction(pause_action)
        
        reset_action = QAction("重置(&R)", self)
        reset_action.setShortcut("F7")
        reset_action.triggered.connect(self.reset_simulation)
        sim_menu.addAction(reset_action)
        
        sim_menu.addSeparator()
        
        step_action = QAction("单步执行(&T)", self)
        step_action.setShortcut("F8")
        step_action.triggered.connect(self.step_simulation)
        sim_menu.addAction(step_action)
        
        # 工具菜单
        tools_menu = menubar.addMenu("工具(&T)")
        
        backbone_action = QAction("生成骨干网络(&B)", self)
        backbone_action.triggered.connect(self.generate_backbone_network)
        tools_menu.addAction(backbone_action)
        
        tools_menu.addSeparator()
        
        log_action = QAction("显示日志(&L)", self)
        log_action.setShortcut("Ctrl+L")
        log_action.triggered.connect(self.show_log_window)
        tools_menu.addAction(log_action)
        
        debug_action = QAction("调试信息(&D)", self)
        debug_action.setShortcut("Ctrl+D")
        debug_action.triggered.connect(self.show_debug_dialog)
        tools_menu.addAction(debug_action)
        
        tools_menu.addSeparator()
        
        settings_action = QAction("设置(&S)...", self)
        settings_action.triggered.connect(self.show_settings_dialog)
        tools_menu.addAction(settings_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu("帮助(&H)")
        
        manual_action = QAction("用户手册(&M)", self)
        manual_action.setShortcut("F1")
        manual_action.triggered.connect(self.show_manual)
        help_menu.addAction(manual_action)
        
        help_menu.addSeparator()
        
        about_action = QAction("关于(&A)...", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_tool_bar(self):
        """创建工具栏"""
        toolbar = QToolBar("主工具栏")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)
        
        # 文件操作
        new_action = toolbar.addAction("新建")
        new_action.triggered.connect(self.new_environment)
        
        open_action = toolbar.addAction("打开")
        open_action.triggered.connect(self.browse_file)
        
        save_action = toolbar.addAction("保存")
        save_action.triggered.connect(self.save_environment)
        
        toolbar.addSeparator()
        
        # 仿真控制
        self.start_action = toolbar.addAction("▶ 开始")
        self.start_action.triggered.connect(self.start_simulation)
        
        self.pause_action = toolbar.addAction("⏸ 暂停")
        self.pause_action.triggered.connect(self.pause_simulation)
        self.pause_action.setEnabled(False)
        
        reset_action = toolbar.addAction("⏹ 重置")
        reset_action.triggered.connect(self.reset_simulation)
        
        toolbar.addSeparator()
        
        # 骨干路径
        backbone_action = toolbar.addAction("🛤 骨干网络")
        backbone_action.triggered.connect(self.generate_backbone_network)
        
        toolbar.addSeparator()
        
        # 视图控制
        zoom_in_action = toolbar.addAction("🔍+")
        zoom_in_action.triggered.connect(self.zoom_in)
        
        zoom_out_action = toolbar.addAction("🔍-")
        zoom_out_action.triggered.connect(self.zoom_out)
        
        fit_action = toolbar.addAction("🔍⊡")
        fit_action.triggered.connect(self.fit_view)
    
    def create_status_bar(self):
        """创建状态栏"""
        self.status_bar = self.statusBar()
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #1e1e2e;
                border-top: 1px solid #45475a;
                color: #cdd6f4;
                font-size: 12px;
            }
            QStatusBar::item {
                border: none;
            }
        """)
        
        # 添加状态栏部件
        self.status_label = QLabel("系统就绪")
        self.status_bar.addWidget(self.status_label)
        
        # 分隔符
        self.status_bar.addPermanentWidget(QLabel(" | "))
        
        # 车辆统计
        self.vehicle_status_label = QLabel("车辆: 0/0")
        self.status_bar.addPermanentWidget(self.vehicle_status_label)
        
        self.status_bar.addPermanentWidget(QLabel(" | "))
        
        # 任务统计
        self.task_status_label = QLabel("任务: 0/0")
        self.status_bar.addPermanentWidget(self.task_status_label)
        
        self.status_bar.addPermanentWidget(QLabel(" | "))
        
        # 性能指标
        self.performance_label = QLabel("性能: --")
        self.status_bar.addPermanentWidget(self.performance_label)
        
        self.status_bar.addPermanentWidget(QLabel(" | "))
        
        # 仿真时间
        self.sim_time_label = QLabel("00:00:00")
        self.status_bar.addPermanentWidget(self.sim_time_label)
    
    def create_log_window(self):
        """创建日志窗口"""
        self.log_dock = QDockWidget("系统日志", self)
        self.log_dock.setFeatures(QDockWidget.DockWidgetClosable | 
                                 QDockWidget.DockWidgetMovable | 
                                 QDockWidget.DockWidgetFloatable)
        
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        
        # 日志文本
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)

        log_layout.addWidget(self.log_text)
        
        # 日志控制
        log_control = QHBoxLayout()
        
        clear_btn = QPushButton("清除")
        clear_btn.clicked.connect(self.clear_log)
        log_control.addWidget(clear_btn)
        
        save_log_btn = QPushButton("保存日志")
        save_log_btn.clicked.connect(self.save_log)
        log_control.addWidget(save_log_btn)
        
        log_control.addStretch()
        
        # 日志级别
        log_control.addWidget(QLabel("级别:"))
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["全部", "信息", "警告", "错误"])
        log_control.addWidget(self.log_level_combo)
        
        log_layout.addLayout(log_control)
        
        self.log_dock.setWidget(log_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)
        self.log_dock.hide()
    
    # 主要功能方法
    def browse_file(self):
        """浏览文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开地图文件", "", "地图文件 (*.json);;所有文件 (*)"
        )
        
        if file_path:
            self.map_file_path = file_path
            self.control_panel.file_label.setText(os.path.basename(file_path))
            self.log(f"已选择地图文件: {os.path.basename(file_path)}")
    
    def load_environment(self):
        """加载环境 - 分离复杂操作"""
        if not self.map_file_path:
            self.log("请先选择地图文件", "error")
            QMessageBox.warning(self, "警告", "请先选择地图文件")
            return

        try:
            self.log("正在加载环境...")
            
            # 创建并加载环境
            self.env = OpenPitMineEnv()
            if not self.env.load_from_file(self.map_file_path):
                raise Exception("环境加载失败")
            
            # ✅ 先设置到图形视图并居中 - 像简化版本一样简单
            self.graphics_view.set_environment(self.env)
            
            # 调试信息：输出实际加载的尺寸
            self.log(f"环境尺寸: {self.env.width} x {self.env.height}")
            self.log(f"场景矩形: {self.graphics_view.mine_scene.sceneRect()}")
            
            # ✅ 然后处理其他UI更新（不影响视图）
            self.update_point_combos()
            self.vehicle_panel.set_environment(self.env, None)  # 先不传调度器
            
            # ✅ 最后创建复杂组件（分离操作）
            self.create_system_components()
            
            # ✅ 组件创建完成后，更新相关面板
            self.vehicle_panel.scheduler = self.vehicle_scheduler  # 设置调度器
            self.stats_widget.set_scheduler(self.vehicle_scheduler)
            self.stats_widget.set_traffic_manager(self.traffic_manager)
            self.chart_widget.set_scheduler(self.vehicle_scheduler)
            
            # ✅ 只在需要时设置组件引用（不触发重绘）
            self.graphics_view.mine_scene.backbone_network = self.backbone_network
            self.graphics_view.mine_scene.traffic_manager = self.traffic_manager
            
            self.log("环境加载成功", "success")
            self.status_label.setText("环境已加载")
            self.enable_controls(True)
            
        except Exception as e:
            self.log(f"加载环境失败: {str(e)}", "error")
            QMessageBox.critical(self, "错误", f"加载环境失败:\n{str(e)}")

    def _center_view(self):
        """居中显示视图"""
        if self.env:
            # 确保场景矩形正确
            self.graphics_view.mine_scene.setSceneRect(0, 0, self.env.width, self.env.height)
            # 强制更新场景
            self.graphics_view.mine_scene.update()
            # 居中显示
            self.graphics_view.fitInView(
                self.graphics_view.mine_scene.sceneRect(), Qt.KeepAspectRatio
            )
            # 更新缩放标签
            self.graphics_view.current_scale = self.graphics_view.transform().m11()
            self.graphics_view.zoom_label.setText(f"缩放: {int(self.graphics_view.current_scale * 100)}%")
            self.log("视图已居中显示")
    def save_environment(self):
        """保存环境"""
        if not self.env:
            QMessageBox.warning(self, "警告", "没有加载的环境")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存环境", 
            f"mine_env_{time.strftime('%Y%m%d_%H%M%S')}.json",
            "JSON文件 (*.json);;所有文件 (*)"
        )
        
        if file_path:
            try:
                # 设置环境引用
                self.env.set_backbone_network(self.backbone_network)
                self.env.set_vehicle_scheduler(self.vehicle_scheduler)
                self.env.set_traffic_manager(self.traffic_manager)
                
                # 保存
                if self.env.save_to_file(file_path):
                    self.log(f"环境已保存到: {file_path}", "success")
                    QMessageBox.information(self, "成功", "环境保存成功")
                else:
                    raise Exception("保存失败")
                    
            except Exception as e:
                self.log(f"保存环境失败: {str(e)}", "error")
                QMessageBox.critical(self, "错误", f"保存环境失败:\n{str(e)}")
    
    def new_environment(self):
        """创建新环境"""
        # 这里可以添加创建新环境的对话框
        self.log("创建新环境功能尚未实现", "warning")
    
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
            
            # 创建车辆调度器 - 根据选择
            scheduler_type = self.control_panel.scheduler_combo.currentIndex()
            if scheduler_type == 1:  # ECBS增强
                self.vehicle_scheduler = SimplifiedECBSVehicleScheduler(
                    self.env, self.path_planner, self.traffic_manager, self.backbone_network
                )
                self.log("使用ECBS增强调度器", "success")
            else:  # 标准调度
                self.vehicle_scheduler = SimplifiedVehicleScheduler(
                    self.env, self.path_planner, self.backbone_network, self.traffic_manager
                )
                self.log("使用标准调度器", "success")
            
            # 初始化车辆状态
            self.vehicle_scheduler.initialize_vehicles()
            
            # 创建默认任务模板
            if self.env.loading_points and self.env.unloading_points:
                self.vehicle_scheduler.create_enhanced_mission_template("default")
            
            self.log("系统组件初始化完成", "success")
            
        except Exception as e:
            self.log(f"系统组件初始化失败: {str(e)}", "error")
            import traceback
            traceback.print_exc()
    
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
        """生成骨干路径网络"""
        if not self.env or not self.backbone_network:
            self.log("请先加载环境", "error")
            QMessageBox.warning(self, "警告", "请先加载环境")
            return
        
        try:
            self.log("正在生成骨干路径网络...")
            self.status_label.setText("生成骨干网络中...")
            
            quality_threshold = self.control_panel.quality_spin.value()
            interface_spacing = self.control_panel.interface_spacing_spin.value()
            
            start_time = time.time()
            success = self.backbone_network.generate_backbone_network(
                quality_threshold=quality_threshold,
                interface_spacing=interface_spacing
            )
            generation_time = time.time() - start_time
            
            if success:
                # 更新组件引用
                self.path_planner.set_backbone_network(self.backbone_network)
                self.traffic_manager.set_backbone_network(self.backbone_network)
                self.vehicle_scheduler.set_backbone_network(self.backbone_network)
                
                # ✅ 只有在这里才更新可视化显示
                self.graphics_view.mine_scene.set_backbone_network(self.backbone_network)
                
                path_count = len(self.backbone_network.backbone_paths)
                interface_count = len(self.backbone_network.backbone_interfaces)
                
                self.control_panel.update_backbone_stats(path_count, interface_count)
                
                self.log(f"骨干路径网络生成成功 - {path_count} 条路径，"
                        f"{interface_count} 个接口，耗时 {generation_time:.2f}s", "success")
                
                self.status_label.setText("骨干网络已生成")
                
                QMessageBox.information(
                    self, "成功", 
                    f"骨干路径网络生成成功！\n"
                    f"路径数量: {path_count}\n"
                    f"接口数量: {interface_count}\n"
                    f"生成耗时: {generation_time:.2f} 秒"
                )
            else:
                self.log("骨干路径网络生成失败", "error")
                QMessageBox.critical(self, "错误", "骨干路径网络生成失败")
                
        except Exception as e:
            self.log(f"生成骨干路径网络失败: {str(e)}", "error")
            QMessageBox.critical(self, "错误", f"生成骨干路径网络失败:\n{str(e)}")
    
    def start_simulation(self):
        """开始仿真"""
        if not self.env:
            self.log("请先加载环境", "error")
            return
        
        self.is_simulating = True
        self.control_panel.start_btn.setEnabled(False)
        self.control_panel.pause_btn.setEnabled(True)
        self.start_action.setEnabled(False)
        self.pause_action.setEnabled(True)
        
        # 启动定时器
        interval = max(50, int(100 / self.simulation_speed))
        self.sim_timer.start(interval)
        
        self.log("仿真已开始")
        self.status_label.setText("仿真运行中...")
    
    def pause_simulation(self):
        """暂停仿真"""
        self.is_simulating = False
        self.control_panel.start_btn.setEnabled(True)
        self.control_panel.pause_btn.setEnabled(False)
        self.start_action.setEnabled(True)
        self.pause_action.setEnabled(False)
        
        self.sim_timer.stop()
        
        self.log("仿真已暂停")
        self.status_label.setText("仿真已暂停")
    
    def reset_simulation(self):
        """重置仿真 - 保持简单"""
        if self.is_simulating:
            self.pause_simulation()
        
        if self.env:
            self.env.reset()
        
        if self.vehicle_scheduler:
            self.vehicle_scheduler.initialize_vehicles()
        
        # ✅ 重置时也保持简单的设置
        self.graphics_view.set_environment(self.env)
        
        self.control_panel.progress_bar.setValue(0)
        self.vehicle_panel.set_environment(self.env, self.vehicle_scheduler)
        
        # 重置图表
        if hasattr(self.chart_widget, 'current_time'):
            self.chart_widget.current_time = 0
            self.chart_widget.time_data.clear()
            self.chart_widget.utilization_data.clear()
            self.chart_widget.completion_data.clear()
        
        self.log("仿真已重置")
        self.status_label.setText("仿真已重置")
    
    def step_simulation(self):
        """单步执行仿真"""
        if not self.env or self.is_simulating:
            return
        
        self.simulation_step()
        self.log("执行单步仿真")
    
    def simulation_step(self):
        """仿真步骤"""
        if not self.is_simulating or not self.env:
            return
        
        time_step = 0.5 * self.simulation_speed
        self.simulation_time += time_step
        
        # 更新环境
        self.env.current_time = self.simulation_time
        
        # 更新调度器
        if self.vehicle_scheduler:
            try:
                self.vehicle_scheduler.update(time_step)
            except Exception as e:
                self.log(f"调度器更新错误: {e}", "error")
        
        # 更新进度条
        max_time = 3600  # 1小时
        progress = min(100, int(self.simulation_time * 100 / max_time))
        self.control_panel.progress_bar.setValue(progress)
        
        # 更新时间显示
        hours = int(self.simulation_time // 3600)
        minutes = int((self.simulation_time % 3600) // 60)
        seconds = int(self.simulation_time % 60)
        self.control_panel.time_label.setText(f"仿真时间: {hours:02d}:{minutes:02d}:{seconds:02d}")
        self.sim_time_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        
        if progress >= 100:
            self.pause_simulation()
            self.log("仿真完成", "success")
            QMessageBox.information(self, "完成", "仿真已完成！")
    
    def update_display(self):
        """更新显示"""
        if not self.env:
            return
        
        try:
            # 更新车辆显示
            self.graphics_view.update_vehicles()
            
            # 更新车辆信息面板
            current_index = self.vehicle_panel.vehicle_combo.currentIndex()
            if current_index >= 0:
                self.vehicle_panel.update_vehicle_info(current_index)
            
        except Exception as e:
            self.log(f"显示更新错误: {e}", "error")
    
    def update_statistics(self):
        """更新统计信息"""
        if not self.vehicle_scheduler:
            return
        
        try:
            # 更新实时统计
            self.stats_widget.update_stats()
            
            # 更新性能图表
            if self.is_simulating:
                self.chart_widget.update_chart()
            
            # 更新状态栏
            stats = self.vehicle_scheduler.get_comprehensive_stats()
            real_time = stats.get('real_time', {})
            
            # 车辆状态
            active = real_time.get('active_vehicles', 0)
            total = len(self.env.vehicles) if self.env else 0
            self.vehicle_status_label.setText(f"车辆: {active}/{total}")
            
            # 任务状态
            completed = stats.get('completed_tasks', 0)
            total_tasks = stats.get('total_tasks', 0)
            self.task_status_label.setText(f"任务: {completed}/{total_tasks}")
            
            # 性能指标
            if stats.get('backbone_utilization_rate', 0) > 0:
                perf_text = f"骨干: {stats['backbone_utilization_rate']*100:.0f}%"
            else:
                perf_text = "性能: --"
            self.performance_label.setText(perf_text)
            
        except Exception as e:
            self.log(f"统计更新错误: {e}", "error")
    
    def assign_vehicle_task(self):
        """分配车辆任务"""
        if not self.vehicle_scheduler:
            self.log("调度器未初始化", "error")
            return
        
        # 获取选中的车辆
        vehicle_id = self.vehicle_panel.vehicle_combo.itemData(
            self.vehicle_panel.vehicle_combo.currentIndex()
        )
        
        if vehicle_id is None:
            self.log("请选择一个车辆", "warning")
            QMessageBox.warning(self, "警告", "请选择一个车辆")
            return
        
        # 获取选中的点位
        loading_id = self.control_panel.loading_combo.currentData()
        unloading_id = self.control_panel.unloading_combo.currentData()
        
        try:
            if loading_id is None or unloading_id is None:
                # 使用默认任务
                if self.vehicle_scheduler.assign_mission_intelligently(vehicle_id):
                    self.log(f"已为车辆 {vehicle_id} 分配默认任务", "success")
                else:
                    self.log(f"车辆 {vehicle_id} 任务分配失败", "error")
            else:
                # 创建特定任务
                template_id = f"specific_{vehicle_id}_{loading_id}_{unloading_id}"
                
                if self.vehicle_scheduler.create_enhanced_mission_template(
                    template_id, loading_id, unloading_id
                ):
                    if self.vehicle_scheduler.assign_mission_intelligently(vehicle_id, template_id):
                        self.log(f"已为车辆 {vehicle_id} 分配特定任务: L{loading_id+1}→U{unloading_id+1}", "success")
                    else:
                        self.log(f"车辆 {vehicle_id} 特定任务分配失败", "error")
                else:
                    self.log("特定任务模板创建失败", "error")
                    
        except Exception as e:
            self.log(f"任务分配错误: {e}", "error")
    
    def assign_all_vehicles(self):
        """批量分配任务"""
        if not self.vehicle_scheduler or not self.env:
            return
        
        assigned_count = 0
        for vehicle_id in self.env.vehicles.keys():
            if self.vehicle_scheduler.assign_mission_intelligently(vehicle_id):
                assigned_count += 1
        
        self.log(f"已为 {assigned_count} 个车辆分配任务", "success")
        QMessageBox.information(self, "成功", f"已为 {assigned_count} 个车辆分配任务")
    
    def cancel_vehicle_task(self):
        """取消车辆任务"""
        if not self.vehicle_scheduler:
            return
        
        vehicle_id = self.vehicle_panel.vehicle_combo.itemData(
            self.vehicle_panel.vehicle_combo.currentIndex()
        )
        
        if vehicle_id is None:
            self.log("请选择一个车辆", "warning")
            return
        
        # 清除任务队列
        if vehicle_id in self.vehicle_scheduler.active_assignments:
            self.vehicle_scheduler.active_assignments[vehicle_id] = []
        
        # 重置车辆状态
        if vehicle_id in self.vehicle_scheduler.vehicle_states:
            vehicle_state = self.vehicle_scheduler.vehicle_states[vehicle_id]
            vehicle_state.status = vehicle_state.status.__class__.IDLE
            vehicle_state.current_task = None
            
        if vehicle_id in self.env.vehicles:
            self.env.vehicles[vehicle_id]['status'] = 'idle'
        
        # 释放交通管理器中的路径
        if self.traffic_manager:
            self.traffic_manager.release_vehicle_path(vehicle_id)
        
        self.log(f"已取消车辆 {vehicle_id} 的任务", "success")
    
    def update_simulation_speed(self):
        """更新仿真速度"""
        value = self.control_panel.speed_dial.value()
        self.simulation_speed = value / 100.0
        self.control_panel.speed_label.setText(f"{self.simulation_speed:.1f}x")
        
        # 更新定时器间隔
        if self.is_simulating:
            interval = max(50, int(100 / self.simulation_speed))
            self.sim_timer.start(interval)
    
    def toggle_display_option(self, option, show):
        """切换显示选项"""
        self.graphics_view.toggle_show_option(option, show)
    
    def change_scheduler(self):
        """更改调度器类型"""
        if not self.env:
            return
        
        scheduler_type = self.control_panel.scheduler_combo.currentIndex()
        
        try:
            if scheduler_type == 1:  # ECBS增强
                self.vehicle_scheduler = SimplifiedECBSVehicleScheduler(
                    self.env, self.path_planner, self.traffic_manager, self.backbone_network
                )
                self.log("切换到ECBS增强调度器", "success")
            elif scheduler_type == 2:  # 预测性调度
                # 使用带预测功能的调度器
                self.vehicle_scheduler = SimplifiedECBSVehicleScheduler(
                    self.env, self.path_planner, self.traffic_manager, self.backbone_network
                )
                self.vehicle_scheduler.scheduling_strategy = 'predictive'
                self.log("切换到预测性调度器", "success")
            else:  # 标准调度
                self.vehicle_scheduler = SimplifiedVehicleScheduler(
                    self.env, self.path_planner, self.backbone_network, self.traffic_manager
                )
                self.log("切换到标准调度器", "success")
            
            # 初始化车辆状态
            self.vehicle_scheduler.initialize_vehicles()
            
            # 更新各个面板
            self.vehicle_panel.set_environment(self.env, self.vehicle_scheduler)
            self.stats_widget.set_scheduler(self.vehicle_scheduler)
            self.chart_widget.set_scheduler(self.vehicle_scheduler)
            
        except Exception as e:
            self.log(f"切换调度器失败: {e}", "error")
    
    def update_conflict_interval(self, value):
        """更新冲突检测间隔"""
        if hasattr(self.vehicle_scheduler, 'conflict_detection_interval'):
            self.vehicle_scheduler.conflict_detection_interval = value
            self.log(f"冲突检测间隔更新为 {value} 秒")
    
    def update_safety_distance(self, value):
        """更新安全距离"""
        if self.traffic_manager:
            self.traffic_manager.safety_distance = value
            self.log(f"安全距离更新为 {value} 米")
    
    def toggle_batch_planning(self, state):
        """切换批量规划"""
        if self.vehicle_scheduler:
            self.vehicle_scheduler.batch_planning = (state == Qt.Checked)
            self.log(f"批量规划: {'启用' if state == Qt.Checked else '禁用'}")
    
    def toggle_predictive_scheduling(self, state):
        """切换预测调度"""
        if self.vehicle_scheduler:
            if state == Qt.Checked:
                self.vehicle_scheduler.scheduling_strategy = 'predictive'
            else:
                self.vehicle_scheduler.scheduling_strategy = 'greedy'
            self.log(f"预测调度: {'启用' if state == Qt.Checked else '禁用'}")
    
    def toggle_cache_optimization(self, state):
        """切换缓存优化"""
        # 这里可以控制各个组件的缓存
        enabled = (state == Qt.Checked)
        
        if self.path_planner:
            # 路径规划器缓存控制
            pass
        
        self.log(f"缓存优化: {'启用' if enabled else '禁用'}")
    
    def enable_controls(self, enabled):
        """启用/禁用控件"""
        self.control_panel.start_btn.setEnabled(enabled)
        self.control_panel.reset_btn.setEnabled(enabled)
        self.control_panel.generate_btn.setEnabled(enabled)
        self.control_panel.assign_task_btn.setEnabled(enabled)
        self.control_panel.assign_all_btn.setEnabled(enabled)
        self.control_panel.cancel_task_btn.setEnabled(enabled)
        self.control_panel.save_btn.setEnabled(enabled)
    
    def zoom_in(self):
        """放大"""
        self.graphics_view.scale(1.2, 1.2)
        self.graphics_view.current_scale *= 1.2
        self.graphics_view.zoom_label.setText(f"缩放: {int(self.graphics_view.current_scale * 100)}%")
    
    def zoom_out(self):
        """缩小"""
        self.graphics_view.scale(1/1.2, 1/1.2)
        self.graphics_view.current_scale /= 1.2
        self.graphics_view.zoom_label.setText(f"缩放: {int(self.graphics_view.current_scale * 100)}%")
    
    def fit_view(self):
        """适应视图"""
        self.graphics_view.fitInView(
            self.graphics_view.mine_scene.sceneRect(), Qt.KeepAspectRatio
        )
        self.graphics_view.current_scale = self.graphics_view.transform().m11()
        self.graphics_view.zoom_label.setText(f"缩放: {int(self.graphics_view.current_scale * 100)}%")
    
    def toggle_fullscreen(self, checked):
        """切换全屏"""
        if checked:
            self.showFullScreen()
        else:
            self.showNormal()
    
    def export_results(self):
        """导出结果"""
        if not self.env or not self.vehicle_scheduler:
            QMessageBox.warning(self, "警告", "没有可导出的结果")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出结果", 
            f"simulation_results_{time.strftime('%Y%m%d_%H%M%S')}.json",
            "JSON文件 (*.json);;CSV文件 (*.csv);;所有文件 (*)"
        )
        
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    self._export_to_csv(file_path)
                else:
                    self._export_to_json(file_path)
                
                self.log(f"结果已导出到: {file_path}", "success")
                QMessageBox.information(self, "成功", "结果导出成功")
                
            except Exception as e:
                self.log(f"导出失败: {str(e)}", "error")
                QMessageBox.critical(self, "错误", f"导出失败:\n{str(e)}")
    
    def _export_to_json(self, file_path):
        """导出为JSON格式"""
        data = {
            'simulation_time': self.simulation_time,
            'vehicles': {},
            'scheduler_stats': self.vehicle_scheduler.get_comprehensive_stats() if self.vehicle_scheduler else {},
            'traffic_stats': self.traffic_manager.get_comprehensive_stats() if self.traffic_manager else {},
            'backbone_stats': {}
        }
        
        # 车辆信息
        for vehicle_id, vehicle in self.env.vehicles.items():
            data['vehicles'][vehicle_id] = {
                'position': vehicle.get('position'),
                'status': vehicle.get('status'),
                'completed_cycles': vehicle.get('completed_cycles', 0),
                'total_distance': 0
            }
            
            # 从调度器获取详细信息
            if self.vehicle_scheduler:
                vehicle_info = self.vehicle_scheduler.get_vehicle_info(vehicle_id)
                if vehicle_info:
                    data['vehicles'][vehicle_id].update({
                        'total_distance': vehicle_info.get('total_distance', 0),
                        'utilization_rate': vehicle_info.get('utilization_rate', 0),
                        'completed_tasks': vehicle_info.get('completed_tasks', 0)
                    })
        
        # 骨干网络统计
        if self.backbone_network:
            data['backbone_stats'] = {
                'total_paths': len(self.backbone_network.backbone_paths),
                'total_interfaces': len(self.backbone_network.backbone_interfaces),
                'interface_usage': dict(self.backbone_network.stats['interface_usage'])
            }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _export_to_csv(self, file_path):
        """导出为CSV格式"""
        import csv
        
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # 写入标题
            writer.writerow(['车辆ID', '状态', '位置X', '位置Y', '完成任务数', 
                           '总里程', '利用率', '完成循环数'])
            
            # 写入车辆数据
            for vehicle_id, vehicle in self.env.vehicles.items():
                pos = vehicle.get('position', (0, 0, 0))
                row = [
                    vehicle_id,
                    vehicle.get('status', 'unknown'),
                    f"{pos[0]:.1f}",
                    f"{pos[1]:.1f}"
                ]
                
                # 添加调度器信息
                if self.vehicle_scheduler:
                    vehicle_info = self.vehicle_scheduler.get_vehicle_info(vehicle_id)
                    if vehicle_info:
                        row.extend([
                            vehicle_info.get('completed_tasks', 0),
                            f"{vehicle_info.get('total_distance', 0):.1f}",
                            f"{vehicle_info.get('utilization_rate', 0):.2%}",
                            vehicle.get('completed_cycles', 0)
                        ])
                    else:
                        row.extend([0, 0, 0, 0])
                else:
                    row.extend([0, 0, 0, 0])
                
                writer.writerow(row)
    
    def show_log_window(self):
        """显示日志窗口"""
        self.log_dock.show()
    
    def save_log(self):
        """保存日志"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存日志",
            f"system_log_{time.strftime('%Y%m%d_%H%M%S')}.txt",
            "文本文件 (*.txt);;所有文件 (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.toPlainText())
                self.log(f"日志已保存到: {file_path}", "success")
            except Exception as e:
                self.log(f"保存日志失败: {e}", "error")
    
    def show_debug_dialog(self):
        """显示调试信息对话框"""
        debug_text = "=== 系统调试信息 ===\n\n"
        
        # 环境信息
        if self.env:
            debug_text += f"环境尺寸: {self.env.width} x {self.env.height}\n"
            debug_text += f"车辆数量: {len(self.env.vehicles)}\n"
            debug_text += f"装载点: {len(self.env.loading_points)}\n"
            debug_text += f"卸载点: {len(self.env.unloading_points)}\n\n"
        
        # 骨干网络信息
        if self.backbone_network:
            debug_text += f"骨干路径: {len(self.backbone_network.backbone_paths)}\n"
            debug_text += f"接口数量: {len(self.backbone_network.backbone_interfaces)}\n"
            
            # 接口使用统计
            if hasattr(self.backbone_network, 'stats'):
                debug_text += f"总使用次数: {self.backbone_network.stats['total_usage']}\n"
                
                # 最常用接口
                if self.backbone_network.stats['interface_usage']:
                    most_used = max(self.backbone_network.stats['interface_usage'].items(),
                                  key=lambda x: x[1])
                    debug_text += f"最常用接口: {most_used[0]} ({most_used[1]}次)\n"
            debug_text += "\n"
        
        # 调度器信息
        if self.vehicle_scheduler:
            stats = self.vehicle_scheduler.get_comprehensive_stats()
            debug_text += f"总任务数: {stats.get('total_tasks', 0)}\n"
            debug_text += f"完成任务: {stats.get('completed_tasks', 0)}\n"
            debug_text += f"失败任务: {stats.get('failed_tasks', 0)}\n"
            debug_text += f"平均完成时间: {stats.get('average_completion_time', 0):.1f}s\n"
            debug_text += f"骨干利用率: {stats.get('backbone_utilization_rate', 0):.1%}\n\n"
        
        # 交通管理器信息
        if self.traffic_manager:
            traffic_stats = self.traffic_manager.get_comprehensive_stats()
            debug_text += f"检测冲突: {traffic_stats.get('conflicts_detected', 0)}\n"
            debug_text += f"解决冲突: {traffic_stats.get('conflicts_resolved', 0)}\n"
            debug_text += f"ECBS调用: {traffic_stats.get('ecbs_calls', 0)}\n"
            
            if traffic_stats.get('conflicts_detected', 0) > 0:
                efficiency = traffic_stats.get('resolution_efficiency', 0)
                debug_text += f"解决效率: {efficiency:.1%}\n"
        
        # 显示对话框
        msg = QMessageBox(self)
        msg.setWindowTitle("调试信息")
        msg.setText(debug_text)
        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setDetailedText(json.dumps(
            self.vehicle_scheduler.get_comprehensive_stats() if self.vehicle_scheduler else {},
            indent=2
        ))
        msg.exec_()
    
    def show_settings_dialog(self):
        """显示设置对话框"""
        # 这里可以实现更详细的设置对话框
        self.log("设置对话框功能尚未实现", "warning")
    
    def show_manual(self):
        """显示用户手册"""
        manual_text = """
        <h2>露天矿多车协同调度系统 - 用户手册</h2>
        
        <h3>快速开始</h3>
        <ol>
            <li>点击"浏览"选择地图文件</li>
            <li>点击"加载"加载环境</li>
            <li>点击"生成骨干网络"创建骨干路径</li>
            <li>选择车辆并分配任务</li>
            <li>点击"开始"运行仿真</li>
        </ol>
        
        <h3>主要功能</h3>
        <ul>
            <li><b>骨干路径网络</b>: 预先规划的高质量路径，提供快速路径规划</li>
            <li><b>接口系统</b>: 智能的路径接入点，支持灵活的路径拼接</li>
            <li><b>ECBS冲突解决</b>: 实时检测和解决车辆冲突</li>
            <li><b>预测性调度</b>: 基于历史数据的智能任务分配</li>
        </ul>
        
        <h3>快捷键</h3>
        <ul>
            <li>Ctrl+O - 打开文件</li>
            <li>Ctrl+S - 保存环境</li>
            <li>F5 - 开始仿真</li>
            <li>F6 - 暂停仿真</li>
            <li>F7 - 重置仿真</li>
            <li>Ctrl++ - 放大视图</li>
            <li>Ctrl+- - 缩小视图</li>
            <li>F11 - 全屏模式</li>
        </ul>
        """
        
        msg = QMessageBox(self)
        msg.setWindowTitle("用户手册")
        msg.setTextFormat(Qt.RichText)
        msg.setText(manual_text)
        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
    
    def show_about(self):
        """显示关于对话框"""
        about_text = """
        <h2>露天矿多车协同调度系统</h2>
        <p>版本: 3.0 优化版</p>
        <p>基于骨干路径和接口系统的智能调度平台</p>
        <hr>
        <h3>核心特性:</h3>
        <ul>
            <li>优化的双向RRT路径规划</li>
            <li>骨干路径网络与接口系统</li>
            <li>增强的ECBS冲突解决</li>
            <li>智能任务分配与预测调度</li>
            <li>实时性能监控与优化</li>
        </ul>
        <hr>
        <p>© 2024 智能矿山调度系统</p>
        """
        
        QMessageBox.about(self, "关于", about_text)
    
    def log(self, message, level="info"):
        """添加日志"""
        current_time = time.strftime("%H:%M:%S")
        
        color_map = {
            "error": "#f38ba8",
            "warning": "#f9e2af",
            "success": "#a6e3a1",
            "info": "#cdd6f4"
        }
        
        color = color_map.get(level, "#cdd6f4")
        formatted_message = f'<span style="color: {color};">[{current_time}] {message}</span>'
        
        self.log_text.append(formatted_message)
        
        # 滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        # 根据级别显示状态栏消息
        if level == "error":
            self.status_bar.showMessage(f"错误: {message}", 5000)
        elif level == "warning":
            self.status_bar.showMessage(f"警告: {message}", 3000)
    
    def clear_log(self):
        """清除日志"""
        self.log_text.clear()
        self.log("日志已清除")
    
    def closeEvent(self, event):
        """关闭事件"""
        if self.is_simulating:
            reply = QMessageBox.question(
                self, '确认退出',
                '仿真正在运行，确定要退出吗？',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                event.ignore()
                return
        
        # 停止所有定时器
        self.update_timer.stop()
        self.sim_timer.stop()
        self.stats_timer.stop()
        
        event.accept()


def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # 设置应用图标（如果有的话）
    # app.setWindowIcon(QIcon("icon.png"))
    
    window = EnhancedMineGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()