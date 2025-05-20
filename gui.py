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

# Import other project components
from backbone_network import BackbonePathNetwork
from path_planner import PathPlanner
from traffic_manager import TrafficManager, Conflict
from vehicle_scheduler import VehicleScheduler, VehicleTask, ECBSVehicleScheduler
from environment import OpenPitMineEnv
from path_utils import improved_visualize_environment

class BackbonePathVisualizer(QGraphicsItemGroup):
    """Backbone path network visualization component"""
    
    def __init__(self, backbone_network, parent=None):
        """Initialize the backbone network visualizer
        
        Args:
            backbone_network: The backbone network object
            parent: Parent QGraphicsItem
        """
        super().__init__(parent)
        self.backbone_network = backbone_network
        self.path_items = {}
        self.connection_items = {}
        self.node_items = {}
        self.setZValue(1)  # Ensure display in appropriate layer
        self.update_visualization()
    
    def update_visualization(self):
        """Update the visualization of the backbone network"""
        # Clear existing items
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
        
        # Draw backbone paths
        for path_id, path_data in self.backbone_network.paths.items():
            path = path_data['path']
            
            if not path or len(path) < 2:
                continue
            
            # Create path item
            painter_path = QPainterPath()
            painter_path.moveTo(path[0][0], path[0][1])
            
            for point in path[1:]:
                painter_path.lineTo(point[0], point[1])
            
            path_item = QGraphicsPathItem(painter_path)
            
            # Set style with gradient color
            gradient = QLinearGradient(path[0][0], path[0][1], path[-1][0], path[-1][1])
            gradient.setColorAt(0, QColor(40, 120, 180, 180))  # Start color
            gradient.setColorAt(1, QColor(120, 40, 180, 180))  # End color
            
            pen = QPen(gradient, 2.0)  # Bold path line
            path_item.setPen(pen)
            
            # Add to group
            self.addToGroup(path_item)
            self.path_items[path_id] = path_item
        
        # Draw connection points
        for conn_id, conn_data in self.backbone_network.connections.items():
            position = conn_data['position']
            conn_type = conn_data.get('type', 'midpath')
            
            # Different styles based on type
            if conn_type == 'endpoint':
                # Endpoint connections, larger
                radius = 2.5
                color = QColor(220, 120, 40)  # Orange
            else:
                # Mid-path connections, smaller
                radius = 1.5
                color = QColor(120, 220, 40)  # Green
            
            conn_item = QGraphicsEllipseItem(
                position[0] - radius, position[1] - radius,
                radius * 2, radius * 2
            )
            
            conn_item.setBrush(QBrush(color))
            conn_item.setPen(QPen(Qt.black, 0.5))
            
            # Add to group
            self.addToGroup(conn_item)
            self.connection_items[conn_id] = conn_item
        
        # Draw nodes (loading points, unloading points, etc.)
        # Get all paths' start and end points
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
            
            # Different styles based on type
            if node_type == 'loading_point':
                # Loading point, larger
                radius = 4.0
                color = QColor(0, 180, 0)  # Green
            elif node_type == 'unloading_point':
                # Unloading point, larger
                radius = 4.0
                color = QColor(180, 0, 0)  # Red
            else:
                # Other point, medium size
                radius = 3.0
                color = QColor(100, 100, 100)  # Gray
            
            node_item = QGraphicsEllipseItem(
                position[0] - radius, position[1] - radius,
                radius * 2, radius * 2
            )
            
            node_item.setBrush(QBrush(color))
            node_item.setPen(QPen(Qt.black, 0.8))
            
            # Add label
            label_item = QGraphicsTextItem(node_id)
            label_item.setPos(position[0] + radius + 1, position[1] - radius - 2)
            label_item.setFont(QFont("Arial", 3))
            
            # Add to group
            self.addToGroup(node_item)
            self.addToGroup(label_item)
            self.node_items[node_id] = node_item
    
    def update_traffic_flow(self):
        """Update traffic flow visualization - shows ECBS path usage"""
        if not self.backbone_network:
            return
        
        # Update paths based on traffic flow
        for path_id, path_item in self.path_items.items():
            if path_id in self.backbone_network.paths:
                path_data = self.backbone_network.paths[path_id]
                traffic_flow = path_data.get('traffic_flow', 0)
                capacity = path_data.get('capacity', 1)
                
                # Calculate flow ratio
                ratio = min(1.0, traffic_flow / max(1, capacity))
                
                # Adjust line width and color based on traffic flow
                width = 1.0 + ratio * 3.0  # 1-4 range
                
                # Color from green to yellow to red
                if ratio < 0.5:
                    # Green to yellow
                    r = int(255 * ratio * 2)
                    g = 255
                    b = 0
                else:
                    # Yellow to red
                    r = 255
                    g = int(255 * (2 - ratio * 2))
                    b = 0
                
                color = QColor(r, g, b, 180)
                
                path_item.setPen(QPen(color, width))


class MineGUI(QMainWindow):
    """Open Pit Mine Multi-Vehicle Coordination System GUI"""
    
    def __init__(self):
        """Initialize the main GUI window"""
        super().__init__()
        
        # System components
        self.env = None
        self.backbone_network = None
        self.path_planner = None
        self.vehicle_scheduler = None
        self.traffic_manager = None
        
        # Initialize UI
        self.init_ui()
        
        # Create update timer
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(50)  # 20 FPS
    
    def init_ui(self):
        """Initialize the user interface"""
        # Set main window
        self.setWindowTitle("Open Pit Mine Multi-Vehicle Coordination System (Backbone Network-based)")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main layout
        self.main_layout = QHBoxLayout(self.central_widget)
        
        # Create left control panel
        self.create_control_panel()
        
        # Create right display area
        self.create_display_area()
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create status bar
        self.statusBar().showMessage("System Ready")
        
        # Create toolbar
        self.create_tool_bar()
    
    def create_control_panel(self):
        """Create left control panel"""
        # Control panel main container
        self.control_panel = QFrame()
        self.control_panel.setFrameShape(QFrame.StyledPanel)
        self.control_panel.setMinimumWidth(280)
        self.control_panel.setMaximumWidth(350)
        
        # Control panel layout
        control_layout = QVBoxLayout(self.control_panel)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        control_layout.addWidget(self.tab_widget)
        
        # Environment tab
        self.env_tab = QWidget()
        self.env_layout = QVBoxLayout(self.env_tab)
        self.tab_widget.addTab(self.env_tab, "Environment")
        
        # Environment tab content
        self.create_env_tab()
        
        # Path tab
        self.path_tab = QWidget()
        self.path_layout = QVBoxLayout(self.path_tab)
        self.tab_widget.addTab(self.path_tab, "Paths")
        
        # Path tab content
        self.create_path_tab()
        
        # Vehicle tab
        self.vehicle_tab = QWidget()
        self.vehicle_layout = QVBoxLayout(self.vehicle_tab)
        self.tab_widget.addTab(self.vehicle_tab, "Vehicles")
        
        # Vehicle tab content
        self.create_vehicle_tab()
        
        # Task tab
        self.task_tab = QWidget()
        self.task_layout = QVBoxLayout(self.task_tab)
        self.tab_widget.addTab(self.task_tab, "Tasks")
        
        # Task tab content
        self.create_task_tab()
        
        # Log area
        self.create_log_area(control_layout)
        
        # Add to main layout
        self.main_layout.addWidget(self.control_panel, 1)
    
    def create_env_tab(self):
        """Create environment tab content"""
        # File loading group
        file_group = QGroupBox("Environment Loading")
        file_layout = QGridLayout()
        
        # Map file
        file_layout.addWidget(QLabel("Map File:"), 0, 0)
        self.map_path = QLabel("Not selected")
        self.map_path.setStyleSheet("background-color: #f8f9fa; padding: 5px; border-radius: 3px;")
        file_layout.addWidget(self.map_path, 0, 1)
        
        # Browse button
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.open_map_file)
        file_layout.addWidget(self.browse_button, 0, 2)
        
        # Load button
        self.load_button = QPushButton("Load Environment")
        self.load_button.clicked.connect(self.load_environment)
        file_layout.addWidget(self.load_button, 1, 0, 1, 3)
        
        file_group.setLayout(file_layout)
        self.env_layout.addWidget(file_group)
        
        # Environment info group
        info_group = QGroupBox("Environment Information")
        info_layout = QGridLayout()
        
        labels = ["Width:", "Height:", "Loading Points:", "Unloading Points:", "Vehicles:"]
        self.env_info_values = {}
        
        for i, label in enumerate(labels):
            info_layout.addWidget(QLabel(label), i, 0)
            self.env_info_values[label] = QLabel("--")
            self.env_info_values[label].setStyleSheet("background-color: #f8f9fa; padding: 3px; border-radius: 3px;")
            info_layout.addWidget(self.env_info_values[label], i, 1)
        
        info_group.setLayout(info_layout)
        self.env_layout.addWidget(info_group)
        
        # Environment control group
        control_group = QGroupBox("Environment Control")
        control_layout = QVBoxLayout()
        
        # Simulation speed
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Simulation Speed:"))
        
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
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_simulation)
        button_layout.addWidget(self.start_button)
        
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_simulation)
        self.pause_button.setEnabled(False)
        button_layout.addWidget(self.pause_button)
        
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_simulation)
        button_layout.addWidget(self.reset_button)
        
        control_layout.addLayout(button_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        control_layout.addWidget(self.progress_bar)
        
        control_group.setLayout(control_layout)
        self.env_layout.addWidget(control_group)
        
        # Add spacing
        self.env_layout.addStretch()
    
    def create_path_tab(self):
        """Create path tab content with ECBS configuration"""
        # Path generation group
        generate_group = QGroupBox("Generate Backbone Path")
        generate_layout = QVBoxLayout()
        
        # Parameter settings
        param_layout = QGridLayout()
        
        param_layout.addWidget(QLabel("Connection Spacing:"), 0, 0)
        self.conn_spacing = QSpinBox()
        self.conn_spacing.setRange(5, 50)
        self.conn_spacing.setValue(10)
        param_layout.addWidget(self.conn_spacing, 0, 1)
        
        param_layout.addWidget(QLabel("Path Smoothness:"), 1, 0)
        self.path_smoothness = QSpinBox()
        self.path_smoothness.setRange(1, 10)
        self.path_smoothness.setValue(5)
        param_layout.addWidget(self.path_smoothness, 1, 1)
        
        generate_layout.addLayout(param_layout)
        
        # Generate button
        self.generate_paths_button = QPushButton("Generate Backbone Path Network")
        self.generate_paths_button.clicked.connect(self.generate_backbone_network)
        generate_layout.addWidget(self.generate_paths_button)
        
        generate_group.setLayout(generate_layout)
        self.path_layout.addWidget(generate_group)
        
        # Path info group
        info_group = QGroupBox("Path Network Information")
        info_layout = QGridLayout()
        
        labels = ["Total Paths:", "Connection Points:", "Total Length:", "Average Length:"]
        self.path_info_values = {}
        
        for i, label in enumerate(labels):
            info_layout.addWidget(QLabel(label), i, 0)
            self.path_info_values[label] = QLabel("--")
            self.path_info_values[label].setStyleSheet("background-color: #f8f9fa; padding: 3px; border-radius: 3px;")
            info_layout.addWidget(self.path_info_values[label], i, 1)
        
        info_group.setLayout(info_layout)
        self.path_layout.addWidget(info_group)
        
        # Path display options
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout()
        
        self.show_paths_cb = QCheckBox("Show Backbone Paths")
        self.show_paths_cb.setChecked(True)
        self.show_paths_cb.stateChanged.connect(self.update_path_display)
        display_layout.addWidget(self.show_paths_cb)
        
        self.show_connections_cb = QCheckBox("Show Connection Points")
        self.show_connections_cb.setChecked(True)
        self.show_connections_cb.stateChanged.connect(self.update_path_display)
        display_layout.addWidget(self.show_connections_cb)
        
        self.show_traffic_cb = QCheckBox("Show Traffic Flow")
        self.show_traffic_cb.setChecked(True)
        self.show_traffic_cb.stateChanged.connect(self.update_path_display)
        display_layout.addWidget(self.show_traffic_cb)
        
        display_group.setLayout(display_layout)
        self.path_layout.addWidget(display_group)
        
        # ECBS configuration group
        ecbs_group = QGroupBox("ECBS Configuration")
        ecbs_layout = QGridLayout()
        
        # Suboptimality bound
        ecbs_layout.addWidget(QLabel("Suboptimality Bound:"), 0, 0)
        self.subopt_slider = QSlider(Qt.Horizontal)
        self.subopt_slider.setMinimum(10)
        self.subopt_slider.setMaximum(30)
        self.subopt_slider.setValue(15)  # Default 1.5
        self.subopt_slider.valueChanged.connect(self.update_ecbs_params)
        ecbs_layout.addWidget(self.subopt_slider, 0, 1)
        
        self.subopt_value = QLabel("1.5")
        self.subopt_value.setAlignment(Qt.AlignCenter)
        ecbs_layout.addWidget(self.subopt_value, 0, 2)
        
        # Conflict resolution strategy
        ecbs_layout.addWidget(QLabel("Resolution Strategy:"), 1, 0)
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["ECBS", "Priority", "Time Window"])
        self.strategy_combo.currentIndexChanged.connect(self.update_ecbs_params)
        ecbs_layout.addWidget(self.strategy_combo, 1, 1, 1, 2)
        
        # Apply button
        self.apply_ecbs_button = QPushButton("Apply ECBS Settings")
        self.apply_ecbs_button.clicked.connect(self.apply_ecbs_settings)
        ecbs_layout.addWidget(self.apply_ecbs_button, 2, 0, 1, 3)
        
        ecbs_group.setLayout(ecbs_layout)
        self.path_layout.addWidget(ecbs_group)
        
        # Edit tools
        tools_group = QGroupBox("Edit Tools")
        tools_layout = QVBoxLayout()
        
        self.edit_mode_cb = QCheckBox("Edit Mode")
        self.edit_mode_cb.setChecked(False)
        self.edit_mode_cb.stateChanged.connect(self.toggle_edit_mode)
        tools_layout.addWidget(self.edit_mode_cb)
        
        tool_button_layout = QHBoxLayout()
        
        self.add_path_button = QPushButton("Add Path")
        self.add_path_button.setEnabled(False)
        self.add_path_button.clicked.connect(self.start_add_path)
        tool_button_layout.addWidget(self.add_path_button)
        
        self.delete_path_button = QPushButton("Delete Path")
        self.delete_path_button.setEnabled(False)
        self.delete_path_button.clicked.connect(self.start_delete_path)
        tool_button_layout.addWidget(self.delete_path_button)
        
        tools_layout.addLayout(tool_button_layout)
        
        tools_group.setLayout(tools_layout)
        self.path_layout.addWidget(tools_group)
        
        # Add spacing
        self.path_layout.addStretch()
    
    def create_vehicle_tab(self):
        """Create vehicle tab content"""
        # Vehicle selection group
        select_group = QGroupBox("Select Vehicle")
        select_layout = QVBoxLayout()
        
        self.vehicle_combo = QComboBox()
        self.vehicle_combo.currentIndexChanged.connect(self.update_vehicle_info)
        select_layout.addWidget(self.vehicle_combo)
        
        select_group.setLayout(select_layout)
        self.vehicle_layout.addWidget(select_group)
        
        # Vehicle info group
        info_group = QGroupBox("Vehicle Information")
        info_layout = QGridLayout()
        
        labels = ["ID:", "Position:", "Status:", "Load:", "Current Task:", "Completed Tasks:"]
        self.vehicle_info_values = {}
        
        for i, label in enumerate(labels):
            info_layout.addWidget(QLabel(label), i, 0)
            self.vehicle_info_values[label] = QLabel("--")
            self.vehicle_info_values[label].setStyleSheet("background-color: #f8f9fa; padding: 3px; border-radius: 3px;")
            info_layout.addWidget(self.vehicle_info_values[label], i, 1)
        
        info_group.setLayout(info_layout)
        self.vehicle_layout.addWidget(info_group)
        
        # Vehicle control group
        control_group = QGroupBox("Vehicle Control")
        control_layout = QVBoxLayout()
        
        # Task control
        task_layout = QHBoxLayout()
        
        self.assign_task_button = QPushButton("Assign Task")
        self.assign_task_button.clicked.connect(self.assign_vehicle_task)
        task_layout.addWidget(self.assign_task_button)
        
        self.cancel_task_button = QPushButton("Cancel Task")
        self.cancel_task_button.clicked.connect(self.cancel_vehicle_task)
        task_layout.addWidget(self.cancel_task_button)
        
        control_layout.addLayout(task_layout)
        
        # Position control
        position_layout = QHBoxLayout()
        
        self.goto_loading_button = QPushButton("Go to Loading Point")
        self.goto_loading_button.clicked.connect(self.goto_loading_point)
        position_layout.addWidget(self.goto_loading_button)
        
        self.goto_unloading_button = QPushButton("Go to Unloading Point")
        self.goto_unloading_button.clicked.connect(self.goto_unloading_point)
        position_layout.addWidget(self.goto_unloading_button)
        
        control_layout.addLayout(position_layout)
        
        # Add return button
        self.return_button = QPushButton("Return to Start")
        self.return_button.clicked.connect(self.return_to_start)
        control_layout.addWidget(self.return_button)
        
        control_group.setLayout(control_layout)
        self.vehicle_layout.addWidget(control_group)
        
        # Vehicle display options
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout()
        
        self.show_vehicles_cb = QCheckBox("Show All Vehicles")
        self.show_vehicles_cb.setChecked(True)
        self.show_vehicles_cb.stateChanged.connect(self.update_vehicle_display)
        display_layout.addWidget(self.show_vehicles_cb)
        
        self.show_vehicle_paths_cb = QCheckBox("Show Vehicle Paths")
        self.show_vehicle_paths_cb.setChecked(True)
        self.show_vehicle_paths_cb.stateChanged.connect(self.update_vehicle_display)
        display_layout.addWidget(self.show_vehicle_paths_cb)
        
        self.show_vehicle_labels_cb = QCheckBox("Show Vehicle Labels")
        self.show_vehicle_labels_cb.setChecked(True)
        self.show_vehicle_labels_cb.stateChanged.connect(self.update_vehicle_display)
        display_layout.addWidget(self.show_vehicle_labels_cb)
        
        display_group.setLayout(display_layout)
        self.vehicle_layout.addWidget(display_group)
        
        # Add spacing
        self.vehicle_layout.addStretch()
    
    def create_task_tab(self):
        """Create task tab content with ECBS status display"""
        # Task list group
        list_group = QGroupBox("Task List")
        list_layout = QVBoxLayout()
        
        self.task_list = QListWidget()
        self.task_list.itemClicked.connect(self.update_task_info)
        list_layout.addWidget(self.task_list)
        
        list_group.setLayout(list_layout)
        self.task_layout.addWidget(list_group)
        
        # Task info group
        info_group = QGroupBox("Task Information")
        info_layout = QGridLayout()
        
        labels = ["ID:", "Type:", "Status:", "Vehicle:", "Progress:", "Start Time:", "Completion Time:"]
        self.task_info_values = {}
        
        for i, label in enumerate(labels):
            info_layout.addWidget(QLabel(label), i, 0)
            self.task_info_values[label] = QLabel("--")
            self.task_info_values[label].setStyleSheet("background-color: #f8f9fa; padding: 3px; border-radius: 3px;")
            info_layout.addWidget(self.task_info_values[label], i, 1)
        
        info_group.setLayout(info_layout)
        self.task_layout.addWidget(info_group)
        
        # Task action group
        action_group = QGroupBox("Task Actions")
        action_layout = QVBoxLayout()
        
        # Template creation button
        self.create_template_button = QPushButton("Create Task Template")
        self.create_template_button.clicked.connect(self.create_task_template)
        action_layout.addWidget(self.create_template_button)
        
        # Auto-assign button
        self.auto_assign_button = QPushButton("Auto-Assign Tasks")
        self.auto_assign_button.clicked.connect(self.auto_assign_tasks)
        action_layout.addWidget(self.auto_assign_button)
        
        action_group.setLayout(action_layout)
        self.task_layout.addWidget(action_group)
        
        # ECBS conflict information section
        ecbs_group = QGroupBox("ECBS Conflict Status")
        ecbs_layout = QVBoxLayout()
        
        self.conflict_info_label = QLabel("Resolved conflicts: 0")
        self.conflict_info_label.setStyleSheet("font-weight: bold;")
        ecbs_layout.addWidget(self.conflict_info_label)
        
        # Add conflict history table
        self.conflict_table = QTableWidget(0, 3)
        self.conflict_table.setHorizontalHeaderLabels(["Vehicle 1", "Vehicle 2", "Type"])
        self.conflict_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.conflict_table.setEditTriggers(QTableWidget.NoEditTriggers)
        ecbs_layout.addWidget(self.conflict_table)
        
        # Add refresh button for conflict display
        refresh_button = QPushButton("Refresh Conflict Data")
        refresh_button.clicked.connect(self.refresh_conflict_data)
        ecbs_layout.addWidget(refresh_button)
        
        ecbs_group.setLayout(ecbs_layout)
        self.task_layout.addWidget(ecbs_group)
        
        # Stats group
        stats_group = QGroupBox("Statistics")
        stats_layout = QGridLayout()
        
        labels = ["Total Tasks:", "Completed Tasks:", "Failed Tasks:", "Average Utilization:", "Total Runtime:"]
        self.task_stats_values = {}
        
        for i, label in enumerate(labels):
            stats_layout.addWidget(QLabel(label), i, 0)
            self.task_stats_values[label] = QLabel("--")
            self.task_stats_values[label].setStyleSheet("background-color: #f8f9fa; padding: 3px; border-radius: 3px;")
            stats_layout.addWidget(self.task_stats_values[label], i, 1)
        
        stats_group.setLayout(stats_layout)
        self.task_layout.addWidget(stats_group)
        
        # Add spacing
        self.task_layout.addStretch()
    
    def create_log_area(self, parent_layout):
        """Create log area"""
        log_group = QGroupBox("System Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        log_layout.addWidget(self.log_text)
        
        # Clear button
        self.clear_log_button = QPushButton("Clear Log")
        self.clear_log_button.clicked.connect(self.clear_log)
        log_layout.addWidget(self.clear_log_button)
        
        log_group.setLayout(log_layout)
        parent_layout.addWidget(log_group)
    
    def create_display_area(self):
        """Create right display area"""
        # Create display area container
        self.display_area = QFrame()
        self.display_area.setFrameShape(QFrame.StyledPanel)
        
        # Display area layout
        display_layout = QVBoxLayout(self.display_area)
        
        # Create view control toolbar
        view_toolbar = QToolBar()
        view_toolbar.setIconSize(QSize(16, 16))
        view_toolbar.setMovable(False)
        
        # Add view control buttons
        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.triggered.connect(self.zoom_in)
        view_toolbar.addAction(zoom_in_action)
        
        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.triggered.connect(self.zoom_out)
        view_toolbar.addAction(zoom_out_action)
        
        fit_view_action = QAction("Fit View", self)
        fit_view_action.triggered.connect(self.fit_view)
        view_toolbar.addAction(fit_view_action)
        
        display_layout.addWidget(view_toolbar)
        
        # Create graphics view
        self.graphics_view = QGraphicsView()
        self.graphics_view.setRenderHint(QPainter.Antialiasing)
        self.graphics_view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        
        # Set scene
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)
        
        display_layout.addWidget(self.graphics_view)
        
        # Add to main layout
        self.main_layout.addWidget(self.display_area, 3)
    
    def create_menu_bar(self):
        """Create menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        open_action = QAction("Open Map", self)
        open_action.triggered.connect(self.open_map_file)
        file_menu.addAction(open_action)
        
        save_action = QAction("Save Results", self)
        save_action.triggered.connect(self.save_results)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        self.show_backbone_action = QAction("Show Backbone Paths", self, checkable=True)
        self.show_backbone_action.setChecked(True)
        self.show_backbone_action.triggered.connect(self.toggle_backbone_display)
        view_menu.addAction(self.show_backbone_action)
        
        self.show_vehicles_action = QAction("Show Vehicles", self, checkable=True)
        self.show_vehicles_action.setChecked(True)
        self.show_vehicles_action.triggered.connect(self.toggle_vehicles_display)
        view_menu.addAction(self.show_vehicles_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        
        generate_network_action = QAction("Generate Backbone Network", self)
        generate_network_action.triggered.connect(self.generate_backbone_network)
        tools_menu.addAction(generate_network_action)
        
        optimize_paths_action = QAction("Optimize Paths", self)
        optimize_paths_action.triggered.connect(self.optimize_paths)
        tools_menu.addAction(optimize_paths_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_tool_bar(self):
        """Create toolbar"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)
        
        # Open map
        open_action = QAction("Open Map", self)
        open_action.triggered.connect(self.open_map_file)
        toolbar.addAction(open_action)
        
        toolbar.addSeparator()
        
        # Simulation control
        self.start_sim_action = QAction("Start", self)
        self.start_sim_action.triggered.connect(self.start_simulation)
        toolbar.addAction(self.start_sim_action)
        
        self.pause_sim_action = QAction("Pause", self)
        self.pause_sim_action.triggered.connect(self.pause_simulation)
        self.pause_sim_action.setEnabled(False)
        toolbar.addAction(self.pause_sim_action)
        
        self.reset_sim_action = QAction("Reset", self)
        self.reset_sim_action.triggered.connect(self.reset_simulation)
        toolbar.addAction(self.reset_sim_action)
        
        toolbar.addSeparator()
        
        # Backbone network generation
        self.generate_network_action = QAction("Generate Backbone Network", self)
        self.generate_network_action.triggered.connect(self.generate_backbone_network)
        toolbar.addAction(self.generate_network_action)
    
    # Event handling methods
    def open_map_file(self):
        """Open map file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Map File", "", 
            "Mine Map Files (*.json);;All Files (*)"
        )
        
        if file_path:
            self.map_path.setText(os.path.basename(file_path))
            self.map_file_path = file_path
            self.log("Map file selected: " + os.path.basename(file_path))
    
    def load_environment(self):
        """Load environment"""
        if not hasattr(self, 'map_file_path') or not self.map_file_path:
            self.log("Please select a map file first!", "error")
            return
        
        try:
            self.log("Loading environment...")
            
            # Load environment using the environment loader
            from mine_loader import MineEnvironmentLoader
            loader = MineEnvironmentLoader()
            self.env = loader.load_environment(self.map_file_path)
            
            # Update environment info display
            self.update_env_info()
            
            # Update vehicle dropdown
            self.update_vehicle_combo()
            
            # Create scene
            self.create_scene()
            
            self.log("Environment loaded: " + os.path.basename(self.map_file_path))
            
            # Create other system components
            self.create_system_components()
            
            # Enable relevant buttons
            self.enable_controls(True)
            
        except Exception as e:
            self.log(f"Failed to load environment: {str(e)}", "error")
    
    def update_env_info(self):
        """Update environment information display"""
        if not self.env:
            return
        
        # Update labels
        self.env_info_values["Width:"].setText(str(self.env.width))
        self.env_info_values["Height:"].setText(str(self.env.height))
        self.env_info_values["Loading Points:"].setText(str(len(self.env.loading_points)))
        self.env_info_values["Unloading Points:"].setText(str(len(self.env.unloading_points)))
        self.env_info_values["Vehicles:"].setText(str(len(self.env.vehicles)))
    
    def update_vehicle_combo(self):
        """Update vehicle dropdown"""
        self.vehicle_combo.clear()
        
        if not self.env:
            return
        
        for v_id in sorted(self.env.vehicles.keys()):
            self.vehicle_combo.addItem(f"Vehicle {v_id}", v_id)
    
    def create_scene(self):
        """Create scene"""
        # Clear existing scene
        self.graphics_scene.clear()
        
        if not self.env:
            return
        
        # Set scene size
        self.graphics_scene.setSceneRect(0, 0, self.env.width, self.env.height)
        
        # Draw background grid
        self.draw_grid()
        
        # Draw obstacles
        self.draw_obstacles()
        
        # Draw key points (loading points, unloading points, etc.)
        self.draw_key_points()
        
        # Draw vehicles
        self.draw_vehicles()
        
        # Adjust view to show entire scene
        self.fit_view()
    
    def draw_grid(self):
        """Draw background grid"""
        if not self.env:
            return
        
        # Set grid size
        grid_size = 10
        
        # Grid line color
        grid_pen = QPen(QColor(220, 220, 220))
        grid_pen.setWidth(0)
        
        # Major grid line color
        major_grid_pen = QPen(QColor(200, 200, 200))
        major_grid_pen.setWidth(0)
        
        # Draw vertical lines
        for x in range(0, self.env.width + 1, grid_size):
            if x % (grid_size * 5) == 0:
                # Major grid line
                line = self.graphics_scene.addLine(x, 0, x, self.env.height, major_grid_pen)
            else:
                # Minor grid line
                line = self.graphics_scene.addLine(x, 0, x, self.env.height, grid_pen)
            
            line.setZValue(-100)  # Ensure grid is in bottom layer
        
        # Draw horizontal lines
        for y in range(0, self.env.height + 1, grid_size):
            if y % (grid_size * 5) == 0:
                # Major grid line
                line = self.graphics_scene.addLine(0, y, self.env.width, y, major_grid_pen)
            else:
                # Minor grid line
                line = self.graphics_scene.addLine(0, y, self.env.width, y, grid_pen)
            
            line.setZValue(-100)  # Ensure grid is in bottom layer
        
        # Add coordinate labels
        label_interval = 50  # Show label every 50 units
        
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
        """Draw obstacles"""
        if not self.env:
            return
        
        # Obstacle color
        obstacle_brush = QBrush(QColor(80, 80, 90))
        obstacle_pen = QPen(QColor(60, 60, 70), 0.5)
        
        # Find contiguous obstacle regions (for efficiency)
        obstacles = self.env._get_obstacle_list()
        
        for obstacle in obstacles:
            x, y = obstacle['x'], obstacle['y']
            width, height = obstacle['width'], obstacle['height']
            
            # Create rectangle
            rect = self.graphics_scene.addRect(x, y, width, height, obstacle_pen, obstacle_brush)
            rect.setZValue(-50)  # Ensure above grid
    
    def draw_key_points(self):
        """Draw key points (loading points, unloading points)"""
        if not self.env:
            return
        
        # Loading points
        for i, point in enumerate(self.env.loading_points):
            x, y = point[0], point[1]
            
            # Create loading point graphic
            radius = 4.0
            loading_item = self.graphics_scene.addEllipse(
                x - radius, y - radius, radius * 2, radius * 2,
                QPen(QColor(0, 120, 0), 1.0),
                QBrush(QColor(0, 180, 0, 180))
            )
            loading_item.setZValue(5)
            
            # Add label
            text = self.graphics_scene.addText(f"Loading Point {i+1}")
            text.setPos(x - 20, y - 15)
            text.setDefaultTextColor(QColor(0, 120, 0))
            text.setZValue(5)
        
        # Unloading points
        for i, point in enumerate(self.env.unloading_points):
            x, y = point[0], point[1]
            
            # Create unloading point graphic
            radius = 4.0
            unloading_item = self.graphics_scene.addEllipse(
                x - radius, y - radius, radius * 2, radius * 2,
                QPen(QColor(120, 0, 0), 1.0),
                QBrush(QColor(180, 0, 0, 180))
            )
            unloading_item.setZValue(5)
            
            # Add label
            text = self.graphics_scene.addText(f"Unloading Point {i+1}")
            text.setPos(x - 20, y - 15)
            text.setDefaultTextColor(QColor(120, 0, 0))
            text.setZValue(5)
    
    def draw_vehicles(self):
        """Draw vehicles"""
        if not self.env:
            return
        
        # Clear existing vehicle graphics
        self.vehicle_items = {}
        
        # Vehicle colors
        vehicle_colors = {
            'idle': QColor(128, 128, 128),      # Gray
            'moving': QColor(0, 123, 255),      # Blue
            'loading': QColor(40, 167, 69),     # Green
            'unloading': QColor(220, 53, 69)    # Red
        }
        
        # Add new vehicle graphics
        for vehicle_id, vehicle_data in self.env.vehicles.items():
            # Get vehicle position
            position = vehicle_data.get('position', (0, 0, 0))
            x, y, theta = position
            
            # Create vehicle graphic
            vehicle_length = 6.0
            vehicle_width = 3.0
            
            # Build vehicle polygon
            polygon = QPolygonF()
            
            # Calculate vehicle corners (default facing Y-axis direction)
            half_length = vehicle_length / 2
            half_width = vehicle_width / 2
            
            corners = [
                QPointF(half_width, half_length),     # Front right
                QPointF(-half_width, half_length),    # Front left
                QPointF(-half_width, -half_length),   # Rear left
                QPointF(half_width, -half_length)     # Rear right
            ]
            
            # Create rotation and translation transform
            transform = QTransform()
            transform.rotate(theta * 180 / math.pi)  # Rotation (in degrees)
            transform.translate(x, y)  # Translation
            
            # Apply transform and add to polygon
            for corner in corners:
                polygon.append(transform.map(corner))
            
            # Create vehicle graphic item
            status = vehicle_data.get('status', 'idle')
            color = vehicle_colors.get(status, vehicle_colors['idle'])
            
            vehicle_item = self.graphics_scene.addPolygon(
                polygon,
                QPen(Qt.black, 0.5),
                QBrush(color)
            )
            vehicle_item.setZValue(10)  # Ensure vehicles are on top
            
            # Add vehicle ID label
            label = self.graphics_scene.addText(str(vehicle_id))
            label.setPos(x - 5, y - 5)
            label.setDefaultTextColor(Qt.white)
            label.setZValue(11)
            
            # Store vehicle graphic item (polygon and label)
            self.vehicle_items[vehicle_id] = {
                'polygon': vehicle_item,
                'label': label
            }
    
    def create_system_components(self):
        """Create system components (backbone network, path planner, etc.)"""
        if not self.env:
            return
        
        # Create backbone path network
        self.backbone_network = BackbonePathNetwork(self.env)
        
        # Create path planner
        self.path_planner = PathPlanner(self.env)
        
        # Set backbone network to path planner
        self.path_planner.set_backbone_network(self.backbone_network)
        
        # Create ECBS-enhanced traffic manager
        self.traffic_manager = TrafficManager(self.env, self.backbone_network)
        
        # Create vehicle scheduler (using ECBS-enhanced version if available)
        try:
            # Try to use ECBS-enhanced scheduler
            self.vehicle_scheduler = ECBSVehicleScheduler(
                self.env, 
                self.path_planner, 
                self.traffic_manager
            )
            self.log("Using ECBS-enhanced vehicle scheduler")
        except:
            # Fall back to regular scheduler
            self.vehicle_scheduler = VehicleScheduler(self.env, self.path_planner)
            self.log("Using standard vehicle scheduler")
        
        # Initialize vehicle statuses
        self.vehicle_scheduler.initialize_vehicles()
        
        # Create task template
        if self.env.loading_points and self.env.unloading_points:
            if isinstance(self.vehicle_scheduler, ECBSVehicleScheduler):
                self.vehicle_scheduler.create_ecbs_mission_template("default")
            else:
                self.vehicle_scheduler.create_mission_template("default")
        
        self.log("System components initialized")
    
    def generate_backbone_network(self):
        """Generate backbone path network"""
        if not self.env:
            self.log("Please load environment first!", "error")
            return
        
        try:
            self.log("Generating backbone path network...")
            
            # Generate network
            self.backbone_network.generate_network()
            
            # Update path info display
            self.update_path_info()
            
            # Set backbone network to planner and traffic manager
            self.path_planner.set_backbone_network(self.backbone_network)
            self.traffic_manager.set_backbone_network(self.backbone_network)
            
            # Draw backbone network
            self.draw_backbone_network()
            
            self.log(f"Backbone path network generated - {len(self.backbone_network.paths)} paths")
            
        except Exception as e:
            self.log(f"Failed to generate backbone network: {str(e)}", "error")
    
    def update_path_info(self):
        """Update path information display"""
        if not self.backbone_network:
            return
        
        # Calculate statistics
        num_paths = len(self.backbone_network.paths)
        num_connections = len(self.backbone_network.connections)
        
        total_length = 0
        for path_data in self.backbone_network.paths.values():
            total_length += path_data.get('length', 0)
        
        avg_length = total_length / num_paths if num_paths > 0 else 0
        
        # Update display
        self.path_info_values["Total Paths:"].setText(str(num_paths))
        self.path_info_values["Connection Points:"].setText(str(num_connections))
        self.path_info_values["Total Length:"].setText(f"{total_length:.1f}")
        self.path_info_values["Average Length:"].setText(f"{avg_length:.1f}")
    
    def draw_backbone_network(self):
        """Draw backbone path network"""
        # Clear existing backbone network graphic
        if hasattr(self, 'backbone_visualizer'):
            self.graphics_scene.removeItem(self.backbone_visualizer)
        
        # Create new visualizer
        self.backbone_visualizer = BackbonePathVisualizer(self.backbone_network)
        self.graphics_scene.addItem(self.backbone_visualizer)
    
    def update_display(self):
        """Update display (triggered by timer)"""
        if not self.env:
            return
        
        # Update vehicle positions
        self.update_vehicles()
        
        # Update selected vehicle info
        index = self.vehicle_combo.currentIndex()
        if index >= 0:
            self.update_vehicle_info(index)
        
        # Update task info
        self.update_task_list()
        
        # Update backbone network traffic flow visualization
        if hasattr(self, 'backbone_visualizer') and self.backbone_visualizer:
            self.backbone_visualizer.update_traffic_flow()
        
        # Display ECBS conflict resolution status if available
        if self.traffic_manager and self.vehicle_scheduler:
            # Check for any active conflicts
            conflicts_resolved = 0
            if hasattr(self.vehicle_scheduler, 'conflict_counts'):
                conflicts_resolved = sum(self.vehicle_scheduler.conflict_counts.values())
            
            if conflicts_resolved > 0:
                self.statusBar().showMessage(f"ECBS resolved {conflicts_resolved} path conflicts")
                
                # Update the conflict information in the task tab
                if hasattr(self, 'conflict_info_label'):
                    self.conflict_info_label.setText(f"Resolved conflicts: {conflicts_resolved}")
    
    def update_vehicles(self):
        """Update vehicle positions and status"""
        if not self.env or not hasattr(self, 'vehicle_items'):
            return
        
        # Vehicle colors
        vehicle_colors = {
            'idle': QColor(128, 128, 128),      # Gray
            'moving': QColor(0, 123, 255),      # Blue
            'loading': QColor(40, 167, 69),     # Green
            'unloading': QColor(220, 53, 69)    # Red
        }
        
        # Update all vehicles
        for vehicle_id, vehicle_data in self.env.vehicles.items():
            # Get vehicle position
            position = vehicle_data.get('position', (0, 0, 0))
            x, y, theta = position
            
            # If vehicle already has graphic item
            if vehicle_id in self.vehicle_items:
                items = self.vehicle_items[vehicle_id]
                
                # Get status
                status = vehicle_data.get('status', 'idle')
                color = vehicle_colors.get(status, vehicle_colors['idle'])
                
                # Update polygon
                vehicle_length = 6.0
                vehicle_width = 3.0
                
                # Build vehicle polygon
                polygon = QPolygonF()
                
                # Calculate vehicle corners
                half_length = vehicle_length / 2
                half_width = vehicle_width / 2
                
                corners = [
                    QPointF(half_width, half_length),     # Front right
                    QPointF(-half_width, half_length),    # Front left
                    QPointF(-half_width, -half_length),   # Rear left
                    QPointF(half_width, -half_length)     # Rear right
                ]
                
                # Create rotation and translation transform
                transform = QTransform()
                transform.rotate(theta * 180 / math.pi)  # Rotation (in degrees)
                transform.translate(x, y)  # Translation
                
                # Apply transform and add to polygon
                for corner in corners:
                    polygon.append(transform.map(corner))
                
                # Update polygon
                items['polygon'].setPolygon(polygon)
                items['polygon'].setBrush(QBrush(color))
                
                # Update label position
                items['label'].setPos(x - 5, y - 5)
    
    def update_vehicle_info(self, index):
        """Update vehicle information display"""
        if not self.env or index < 0:
            return
        
        # Get selected vehicle ID
        vehicle_id = self.vehicle_combo.itemData(index)
        
        if vehicle_id not in self.env.vehicles:
            return
        
        # Get vehicle information
        vehicle_data = self.env.vehicles[vehicle_id]
        
        # Get more detailed info from scheduler if available
        if self.vehicle_scheduler:
            vehicle_info = self.vehicle_scheduler.get_vehicle_info(vehicle_id)
        else:
            vehicle_info = vehicle_data
        
        # Update display
        position = vehicle_data.get('position', (0, 0, 0))
        self.vehicle_info_values["ID:"].setText(str(vehicle_id))
        self.vehicle_info_values["Position:"].setText(f"({position[0]:.1f}, {position[1]:.1f})")
        
        status = vehicle_data.get('status', 'idle')
        status_map = {
            'idle': 'Idle',
            'moving': 'Moving',
            'loading': 'Loading',
            'unloading': 'Unloading'
        }
        self.vehicle_info_values["Status:"].setText(status_map.get(status, status))
        
        load = vehicle_data.get('load', 0)
        max_load = vehicle_data.get('max_load', 100)
        self.vehicle_info_values["Load:"].setText(f"{load}/{max_load}")
        
        # Show current task
        if vehicle_info and 'current_task' in vehicle_info and vehicle_info['current_task']:
            task_id = vehicle_info['current_task']
            if task_id in self.vehicle_scheduler.tasks:
                task = self.vehicle_scheduler.tasks[task_id]
                task_text = f"{task.task_type} ({task.progress:.0%})"
                self.vehicle_info_values["Current Task:"].setText(task_text)
            else:
                self.vehicle_info_values["Current Task:"].setText(str(task_id))
        else:
            self.vehicle_info_values["Current Task:"].setText("None")
        
        # Show completed tasks
        completed = vehicle_info.get('completed_tasks', 0) if vehicle_info else 0
        self.vehicle_info_values["Completed Tasks:"].setText(str(completed))
    
    def update_task_list(self):
        """Update task list"""
        if not self.vehicle_scheduler:
            return
        
        # Save current selection
        current_item = self.task_list.currentItem()
        current_task_id = current_item.data(Qt.UserRole) if current_item else None
        
        # Clear list
        self.task_list.clear()
        
        # Add all tasks
        for task_id, task in self.vehicle_scheduler.tasks.items():
            item = QListWidgetItem(f"{task_id} - {task.task_type} ({task.status})")
            item.setData(Qt.UserRole, task_id)
            
            # Set color based on status
            if task.status == 'completed':
                item.setForeground(QBrush(QColor(40, 167, 69)))  # Green
            elif task.status == 'failed':
                item.setForeground(QBrush(QColor(220, 53, 69)))  # Red
            elif task.status == 'in_progress':
                item.setForeground(QBrush(QColor(0, 123, 255)))  # Blue
            
            self.task_list.addItem(item)
            
            # Re-select previously selected task
            if task_id == current_task_id:
                self.task_list.setCurrentItem(item)
        
        # Update statistics
        stats = self.vehicle_scheduler.get_stats()
        
        self.task_stats_values["Total Tasks:"].setText(str(len(self.vehicle_scheduler.tasks)))
        self.task_stats_values["Completed Tasks:"].setText(str(stats['completed_tasks']))
        self.task_stats_values["Failed Tasks:"].setText(str(stats['failed_tasks']))
        
        avg_util = stats.get('average_utilization', 0)
        self.task_stats_values["Average Utilization:"].setText(f"{avg_util:.1%}")
        
        # If environment has current time, show runtime
        if hasattr(self.env, 'current_time'):
            self.task_stats_values["Total Runtime:"].setText(f"{self.env.current_time:.1f}")
    
    def update_task_info(self, item):
        """Update task information display"""
        if not item or not self.vehicle_scheduler:
            return
        
        # Get task ID
        task_id = item.data(Qt.UserRole)
        
        if task_id not in self.vehicle_scheduler.tasks:
            return
        
        # Get task information
        task = self.vehicle_scheduler.tasks[task_id]
        
        # Update display
        self.task_info_values["ID:"].setText(task.task_id)
        self.task_info_values["Type:"].setText(task.task_type)
        self.task_info_values["Status:"].setText(task.status)
        self.task_info_values["Vehicle:"].setText(str(task.assigned_vehicle) if task.assigned_vehicle else "Unassigned")
        self.task_info_values["Progress:"].setText(f"{task.progress:.0%}")
        self.task_info_values["Start Time:"].setText(f"{task.start_time:.1f}" if task.start_time else "--")
        self.task_info_values["Completion Time:"].setText(f"{task.completion_time:.1f}" if task.completion_time else "--")
    
    def enable_controls(self, enabled):
        """Enable or disable control buttons"""
        # Environment tab
        self.start_button.setEnabled(enabled)
        self.reset_button.setEnabled(enabled)
        
        # Toolbar
        self.start_sim_action.setEnabled(enabled)
        self.reset_sim_action.setEnabled(enabled)
        
        # Path tab
        self.generate_paths_button.setEnabled(enabled)
        
        # Vehicle tab buttons
        self.assign_task_button.setEnabled(enabled)
        self.cancel_task_button.setEnabled(enabled)
        self.goto_loading_button.setEnabled(enabled)
        self.goto_unloading_button.setEnabled(enabled)
        self.return_button.setEnabled(enabled)
        
        # Task tab buttons
        self.create_template_button.setEnabled(enabled)
        self.auto_assign_button.setEnabled(enabled)
    
    def log(self, message, level="info"):
        """Add log message"""
        # Get current time
        current_time = time.strftime("%H:%M:%S")
        
        # Set color based on log level
        if level == "error":
            color = "red"
        elif level == "warning":
            color = "orange"
        elif level == "success":
            color = "green"
        else:
            color = "black"
        
        # Format message
        formatted_message = f'<span style="color: {color};">[{current_time}] {message}</span>'
        
        # Add to log text box
        self.log_text.append(formatted_message)
        
        # Scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        # Also update status bar
        self.statusBar().showMessage(message)
    
    def clear_log(self):
        """Clear log"""
        self.log_text.clear()
    
    # View control methods
    def zoom_in(self):
        """Zoom in"""
        self.graphics_view.scale(1.2, 1.2)
    
    def zoom_out(self):
        """Zoom out"""
        self.graphics_view.scale(1/1.2, 1/1.2)
    
    def fit_view(self):
        """Fit view to show entire scene"""
        self.graphics_view.fitInView(self.graphics_scene.sceneRect(), Qt.KeepAspectRatio)
    
    # Simulation control methods
    def update_simulation_speed(self):
        """Update simulation speed"""
        value = self.speed_slider.value()
        speed = value / 50.0  # Convert to speed multiplier (0.02-2.0)
        
        self.speed_value.setText(f"{speed:.1f}x")
        
        # If environment has time step property, update it
        if hasattr(self.env, 'time_step'):
            self.env.time_step = 0.5 * speed  # Base time step * speed multiplier
        
        self.log(f"Simulation speed set to {speed:.1f}x")
    
    def start_simulation(self):
        """Start simulation"""
        if not self.env:
            self.log("Please load environment first!", "error")
            return
        
        # Update button states
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        
        self.start_sim_action.setEnabled(False)
        self.pause_sim_action.setEnabled(True)
        
        # Start simulation logic
        self.is_simulating = True
        
        # Create simulation timer
        if not hasattr(self, 'sim_timer'):
            self.sim_timer = QTimer(self)
            self.sim_timer.timeout.connect(self.simulation_step)
        
        # Start timer
        self.sim_timer.start(100)  # Update every 100ms
        
        self.log("Simulation started")
    
    def pause_simulation(self):
        """Pause simulation"""
        # Update button states
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        
        self.start_sim_action.setEnabled(True)
        self.pause_sim_action.setEnabled(False)
        
        # Pause simulation logic
        self.is_simulating = False
        
        # Stop timer
        if hasattr(self, 'sim_timer') and self.sim_timer.isActive():
            self.sim_timer.stop()
        
        self.log("Simulation paused")
    
    def reset_simulation(self):
        """Reset simulation"""
        # Ask for confirmation
        reply = QMessageBox.question(
            self, 'Confirm Reset', 
            'Are you sure you want to reset the simulation? This will clear all current data.',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Stop simulation
        if hasattr(self, 'is_simulating') and self.is_simulating:
            self.pause_simulation()
        
        # Reset environment
        if self.env:
            self.env.reset()
        
        # Reset scheduler
        if self.vehicle_scheduler:
            self.vehicle_scheduler.initialize_vehicles()
        
        # Reset traffic manager
        if self.traffic_manager:
            # Traffic manager has no explicit reset method, may need to recreate
            self.traffic_manager = TrafficManager(self.env)
            self.traffic_manager.set_backbone_network(self.backbone_network)
        
        # Update display
        self.create_scene()
        
        # If backbone network was generated, redraw it
        if self.backbone_network and hasattr(self.backbone_network, 'paths') and self.backbone_network.paths:
            self.draw_backbone_network()
        
        # Reset progress bar
        self.progress_bar.setValue(0)
        
        # Update vehicle and task info
        self.update_vehicle_combo()
        if self.vehicle_combo.count() > 0:
            self.update_vehicle_info(0)
        
        self.update_task_list()
        
        self.log("Simulation reset")
    
    def simulation_step(self):
        """Single simulation step with ECBS conflict resolution"""
        if not self.is_simulating or not self.env:
            return
        
        # Get time step
        time_step = getattr(self.env, 'time_step', 0.5)
        
        # Update environment time
        if not hasattr(self.env, 'current_time'):
            self.env.current_time = 0
        
        self.env.current_time += time_step
        
        # Update vehicle scheduler with ECBS
        if self.vehicle_scheduler:
            # For ECBSVehicleScheduler, the update method includes conflict detection and resolution
            self.vehicle_scheduler.update(time_step)
            
            # Check if there are any active conflicts
            has_conflicts = False
            if hasattr(self.traffic_manager, 'has_conflict'):
                has_conflicts = self.traffic_manager.has_conflict
            
            # Log conflicts if they occur
            if has_conflicts:
                self.log("ECBS resolving path conflicts...", "warning")
        
        # Update progress bar
        max_time = 3600  # Max simulation time: 1 hour
        progress = min(100, int(self.env.current_time * 100 / max_time))
        self.progress_bar.setValue(progress)
        
        # If progress reaches 100%, auto-stop
        if progress >= 100:
            self.pause_simulation()
            self.log("Simulation completed", "success")
    
    # Vehicle operation methods
    def assign_vehicle_task(self):
        """Assign task to currently selected vehicle using ECBS scheduling"""
        if not self.vehicle_scheduler:
            self.log("Vehicle scheduler not initialized", "error")
            return
        
        # Get selected vehicle
        index = self.vehicle_combo.currentIndex()
        if index < 0:
            self.log("Please select a vehicle first", "warning")
            return
        
        vehicle_id = self.vehicle_combo.itemData(index)
        
        # If there's a task template, use it with ECBS-aware assignment
        if "default" in self.vehicle_scheduler.mission_templates:
            # Use the scheduler's assign_mission method
            if self.vehicle_scheduler.assign_mission(vehicle_id, "default"):
                self.log(f"Assigned default mission to vehicle {vehicle_id}", "success")
            else:
                self.log(f"Failed to assign mission to vehicle {vehicle_id}", "error")
        else:
            self.log("No available task template", "error")
    
    def cancel_vehicle_task(self):
        """Cancel current vehicle's task"""
        if not self.vehicle_scheduler:
            return
        
        # Get selected vehicle
        index = self.vehicle_combo.currentIndex()
        if index < 0:
            self.log("Please select a vehicle first", "warning")
            return
        
        vehicle_id = self.vehicle_combo.itemData(index)
        
        # Clear task queue
        if vehicle_id in self.vehicle_scheduler.task_queues:
            self.vehicle_scheduler.task_queues[vehicle_id] = []
            
            # If there's a current task, mark it as completed
            status = self.vehicle_scheduler.vehicle_statuses[vehicle_id]
            if status['current_task']:
                task_id = status['current_task']
                if task_id in self.vehicle_scheduler.tasks:
                    task = self.vehicle_scheduler.tasks[task_id]
                    task.status = 'completed'
                    task.progress = 1.0
                    task.completion_time = self.env.current_time if hasattr(self.env, 'current_time') else 0
                
                # Reset vehicle status
                status['status'] = 'idle'
                status['current_task'] = None
                self.env.vehicles[vehicle_id]['status'] = 'idle'
                
                # Release path in traffic manager
                if self.traffic_manager:
                    self.traffic_manager.release_vehicle_path(vehicle_id)
                
                self.log(f"Cancelled all tasks for vehicle {vehicle_id}", "success")
            else:
                self.log(f"Vehicle {vehicle_id} has no active tasks", "warning")
    
    def goto_loading_point(self):
        """Command current vehicle to go to loading point"""
        if not self.vehicle_scheduler or not self.env.loading_points:
            return
        
        # Get selected vehicle
        index = self.vehicle_combo.currentIndex()
        if index < 0:
            self.log("Please select a vehicle first", "warning")
            return
        
        vehicle_id = self.vehicle_combo.itemData(index)
        
        # Select a loading point
        loading_point = self.env.loading_points[0]
        
        # Create task
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
        
        # Add to task dictionary
        self.vehicle_scheduler.tasks[task_id] = task
        
        # Clear and add to vehicle task queue
        self.vehicle_scheduler.task_queues[vehicle_id] = [task_id]
        
        # If vehicle is idle, start task immediately
        status = self.vehicle_scheduler.vehicle_statuses[vehicle_id]
        if status['status'] == 'idle':
            self.vehicle_scheduler._start_next_task(vehicle_id)
        
        self.log(f"Commanded vehicle {vehicle_id} to go to loading point", "success")
    
    def goto_unloading_point(self):
        """Command current vehicle to go to unloading point"""
        if not self.vehicle_scheduler or not self.env.unloading_points:
            return
        
        # Get selected vehicle
        index = self.vehicle_combo.currentIndex()
        if index < 0:
            self.log("Please select a vehicle first", "warning")
            return
        
        vehicle_id = self.vehicle_combo.itemData(index)
        
        # Select an unloading point
        unloading_point = self.env.unloading_points[0]
        
        # Create task
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
        
        # Add to task dictionary
        self.vehicle_scheduler.tasks[task_id] = task
        
        # Clear and add to vehicle task queue
        self.vehicle_scheduler.task_queues[vehicle_id] = [task_id]
        
        # If vehicle is idle, start task immediately
        status = self.vehicle_scheduler.vehicle_statuses[vehicle_id]
        if status['status'] == 'idle':
            self.vehicle_scheduler._start_next_task(vehicle_id)
        
        self.log(f"Commanded vehicle {vehicle_id} to go to unloading point", "success")
    
    def return_to_start(self):
        """Command current vehicle to return to start point"""
        if not self.vehicle_scheduler:
            return
        
        # Get selected vehicle
        index = self.vehicle_combo.currentIndex()
        if index < 0:
            self.log("Please select a vehicle first", "warning")
            return
        
        vehicle_id = self.vehicle_combo.itemData(index)
        
        # Get vehicle initial position
        if vehicle_id not in self.env.vehicles:
            return
        
        vehicle = self.env.vehicles[vehicle_id]
        initial_position = vehicle.get('initial_position')
        
        if not initial_position:
            self.log(f"Vehicle {vehicle_id} has no initial position", "error")
            return
        
        # Create task
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
        
        # Add to task dictionary
        self.vehicle_scheduler.tasks[task_id] = task
        
        # Clear and add to vehicle task queue
        self.vehicle_scheduler.task_queues[vehicle_id] = [task_id]
        
        # If vehicle is idle, start task immediately
        status = self.vehicle_scheduler.vehicle_statuses[vehicle_id]
        if status['status'] == 'idle':
            self.vehicle_scheduler._start_next_task(vehicle_id)
        
        self.log(f"Commanded vehicle {vehicle_id} to return to start point", "success")
    
    # Task operation methods
    def create_task_template(self):
        """Create task template"""
        if not self.vehicle_scheduler or not self.env.loading_points or not self.env.unloading_points:
            self.log("Environment not initialized or missing loading/unloading points", "error")
            return
        
        # Create default template
        if isinstance(self.vehicle_scheduler, ECBSVehicleScheduler):
            if self.vehicle_scheduler.create_ecbs_mission_template("default"):
                self.log("Created default task template with ECBS support", "success")
            else:
                self.log("Failed to create task template", "error")
        else:
            if self.vehicle_scheduler.create_mission_template("default"):
                self.log("Created default task template", "success")
            else:
                self.log("Failed to create task template", "error")
    
    def auto_assign_tasks(self):
        """Auto-assign tasks to all vehicles using ECBS batch planning"""
        if not self.vehicle_scheduler:
            self.log("Vehicle scheduler not initialized", "error")
            return
        
        # Ensure we have a task template
        if "default" not in self.vehicle_scheduler.mission_templates:
            self.create_task_template()
        
        # For ECBSVehicleScheduler, use batch assignment for better coordination
        if isinstance(self.vehicle_scheduler, ECBSVehicleScheduler):
            # Prepare batch of vehicles for ECBS coordination
            vehicle_ids = list(self.env.vehicles.keys())
            
            self.log(f"Planning coordinated paths for {len(vehicle_ids)} vehicles using ECBS...")
            
            # Collect tasks for all vehicles
            tasks = []
            for vehicle_id in vehicle_ids:
                # Create tasks from template
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
                    vehicle_pos = task.goal  # Update for next task
            
            # Use ECBS batch assignment
            assignments = self.vehicle_scheduler.assign_tasks_batch(tasks)
            self.log(f"ECBS batch assignment completed with {len(assignments)} assignments")
        else:
            # Fallback to simple assignment
            success_count = 0
            for vehicle_id in self.env.vehicles.keys():
                if self.vehicle_scheduler.assign_mission(vehicle_id, "default"):
                    success_count += 1
            
            self.log(f"Assigned tasks to {success_count}/{len(self.env.vehicles)} vehicles", "success")
    
    def refresh_conflict_data(self):
        """Refresh ECBS conflict data display"""
        if not self.traffic_manager or not hasattr(self.traffic_manager, 'detect_conflicts'):
            self.log("Traffic manager not available", "warning")
            return
        
        # Collect active paths
        active_paths = {}
        for vehicle_id, vehicle in self.env.vehicles.items():
            if 'path' in vehicle and vehicle['path']:
                active_paths[vehicle_id] = vehicle['path']
        
        # Detect conflicts
        if active_paths and len(active_paths) > 1:
            conflicts = self.traffic_manager.detect_conflicts(active_paths)
            
            # Update conflict table
            self.conflict_table.setRowCount(len(conflicts))
            
            for i, conflict in enumerate(conflicts):
                self.conflict_table.setItem(i, 0, QTableWidgetItem(str(conflict.agent1)))
                self.conflict_table.setItem(i, 1, QTableWidgetItem(str(conflict.agent2)))
                self.conflict_table.setItem(i, 2, QTableWidgetItem(conflict.conflict_type))
            
            self.log(f"Found {len(conflicts)} potential conflicts", "warning")
        else:
            self.conflict_table.setRowCount(0)
            self.log("No active paths to check for conflicts", "info")
    
    # Display update methods
    def update_path_display(self):
        """Update path display options"""
        if not hasattr(self, 'backbone_visualizer') or not self.backbone_visualizer:
            return
        
        # Show/hide backbone paths
        self.backbone_visualizer.setVisible(self.show_paths_cb.isChecked())
        
        # TODO: Implement connection point and traffic flow display controls
    
    def update_vehicle_display(self):
        """Update vehicle display options"""
        if not hasattr(self, 'vehicle_items'):
            return
        
        # Show/hide all vehicles
        show_vehicles = self.show_vehicles_cb.isChecked()
        
        for vehicle_id, items in self.vehicle_items.items():
            items['polygon'].setVisible(show_vehicles)
            
            # Show/hide labels based on option
            items['label'].setVisible(show_vehicles and self.show_vehicle_labels_cb.isChecked())
        
        # TODO: Implement vehicle path display control
    
    def toggle_edit_mode(self, checked):
        """Toggle edit mode"""
        self.add_path_button.setEnabled(checked)
        self.delete_path_button.setEnabled(checked)
        
        # TODO: Implement edit mode functionality
        self.log("Edit mode " + ("enabled" if checked else "disabled"))
    
    def start_add_path(self):
        """Start adding path"""
        # TODO: Implement add path functionality
        self.log("Add path functionality not yet implemented", "warning")
    
    def start_delete_path(self):
        """Start deleting path"""
        # TODO: Implement delete path functionality
        self.log("Delete path functionality not yet implemented", "warning")
    
    def toggle_backbone_display(self, checked):
        """Toggle backbone path display"""
        if hasattr(self, 'backbone_visualizer') and self.backbone_visualizer:
            self.backbone_visualizer.setVisible(checked)
            
            # Sync checkbox state
            self.show_paths_cb.setChecked(checked)
    
    def toggle_vehicles_display(self, checked):
        """Toggle vehicle display"""
        if hasattr(self, 'vehicle_items'):
            for vehicle_id, items in self.vehicle_items.items():
                items['polygon'].setVisible(checked)
                items['label'].setVisible(checked and self.show_vehicle_labels_cb.isChecked())
            
            # Sync checkbox state
            self.show_vehicles_cb.setChecked(checked)
    
    def optimize_paths(self):
        """Optimize paths"""
        if not self.backbone_network or not self.backbone_network.paths:
            self.log("Please generate backbone path network first", "warning")
            return
        
        try:
            self.log("Optimizing paths...")
            
            # Re-optimize all paths
            self.backbone_network._optimize_all_paths()
            
            # Update path information
            self.update_path_info()
            
            # Redraw backbone network
            self.draw_backbone_network()
            
            self.log("Path optimization complete", "success")
            
        except Exception as e:
            self.log(f"Path optimization failed: {str(e)}", "error")
    
    def save_results(self):
        """Save results"""
        if not self.env:
            self.log("No results to save", "warning")
            return
        
        # Open save dialog
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Results",
            f"simulation_result_{time.strftime('%Y%m%d_%H%M%S')}",
            "PNG Image (*.png);;JSON Data (*.json);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # Save different formats based on file type
            if file_path.endswith('.png'):
                # Save scene screenshot
                pixmap = QPixmap(self.graphics_view.viewport().size())
                pixmap.fill(Qt.white)
                
                painter = QPainter(pixmap)
                self.graphics_view.render(painter)
                painter.end()
                
                if pixmap.save(file_path):
                    self.log(f"Screenshot saved to: {file_path}", "success")
                else:
                    self.log("Failed to save screenshot", "error")
                
            elif file_path.endswith('.json'):
                # Save simulation data
                data = {
                    'time': self.env.current_time if hasattr(self.env, 'current_time') else 0,
                    'vehicles': {},
                    'tasks': {}
                }
                
                # Add vehicle data
                for vehicle_id, vehicle in self.env.vehicles.items():
                    data['vehicles'][vehicle_id] = {
                        'position': vehicle.get('position'),
                        'status': vehicle.get('status'),
                        'load': vehicle.get('load'),
                        'completed_cycles': vehicle.get('completed_cycles', 0)
                    }
                
                # Add task data
                if self.vehicle_scheduler:
                    for task_id, task in self.vehicle_scheduler.tasks.items():
                        data['tasks'][task_id] = task.to_dict()
                
                # Save as JSON
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                self.log(f"Simulation data saved to: {file_path}", "success")
            
            else:
                self.log("Unknown file format", "error")
        
        except Exception as e:
            self.log(f"Failed to save results: {str(e)}", "error")
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
        <div style="text-align:center;">
            <h2>Open Pit Mine Multi-Vehicle Coordination System</h2>
            <p>Backbone Path Network-based Multi-Vehicle Planning and Scheduling</p>
            <p>Version: 1.0</p>
            <hr>
            <p>Uses backbone path network to optimize mine vehicle scheduling efficiency</p>
            <p>Supports multi-vehicle coordination, conflict avoidance, and traffic management</p>
            <hr>
            <p>Developer: XXX</p>
            <p>Copyright © 2023</p>
        </div>
        """
        
        QMessageBox.about(self, "About", about_text)
    
    def update_ecbs_params(self):
        """Update ECBS parameter display"""
        # Update suboptimality bound display
        subopt = self.subopt_slider.value() / 10.0
        self.subopt_value.setText(f"{subopt:.1f}")
    
    def apply_ecbs_settings(self):
        """Apply ECBS settings to the traffic manager"""
        if not self.traffic_manager:
            self.log("Traffic manager not available", "warning")
            return
        
        # Get suboptimality bound
        subopt = self.subopt_slider.value() / 10.0
        
        # Get conflict resolution strategy
        strategy = self.strategy_combo.currentText().lower()
        
        # Apply settings
        if hasattr(self.traffic_manager, 'suboptimality_bound'):
            self.traffic_manager.suboptimality_bound = subopt
        
        # Apply to vehicle scheduler if it's an ECBSVehicleScheduler
        if isinstance(self.vehicle_scheduler, ECBSVehicleScheduler):
            if hasattr(self.vehicle_scheduler, 'conflict_resolution_strategy'):
                self.vehicle_scheduler.conflict_resolution_strategy = strategy
        
        self.log(f"Applied ECBS settings: bound={subopt}, strategy={strategy}", "success")
    
    def set_up_integrated_system(self):
        """Set up the complete integrated system with ECBS"""
        # Create the environment
        env = OpenPitMineEnv(width=500, height=500)
        
        # Create the path planner with the environment
        path_planner = PathPlanner(env)
        
        # Create the backbone network
        backbone_network = BackbonePathNetwork(env)
        backbone_network.generate_network()
        
        # Set the backbone network to the path planner
        path_planner.set_backbone_network(backbone_network)
        
        # Create the ECBS traffic manager
        traffic_manager = TrafficManager(env, backbone_network)
        
        # Create the ECBS-enhanced vehicle scheduler
        scheduler = ECBSVehicleScheduler(env, path_planner, traffic_manager)
        
        # Initialize all components
        scheduler.initialize_vehicles()
        
        # Create mission templates
        scheduler.create_ecbs_mission_template("standard_cycle")
        
        # Assign missions to vehicles
        for vehicle_id in env.vehicles.keys():
            scheduler.assign_mission(vehicle_id, "standard_cycle")
        
        # Save references
        self.env = env
        self.path_planner = path_planner
        self.backbone_network = backbone_network
        self.traffic_manager = traffic_manager
        self.vehicle_scheduler = scheduler
        
        # Start the simulation
        self.start_simulation()
        
        self.log("Full integrated system with ECBS initialized and running", "success")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and show main window
    window = MineGUI()
    window.show()
    
    sys.exit(app.exec_())