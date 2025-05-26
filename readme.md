一个基于骨干路径网络和增强ECBS算法的智能露天矿车辆调度系统，具有实时冲突检测、预测性规划和现代化图形界面。
基于骨干路径规划设计的露天矿多车协同调度系统.png
🚀 主要特性
核心算法

混合双向RRT路径规划 - 高效的随机树路径搜索算法
骨干路径网络 - 预计算的高质量路径骨干，支持快速路径拼接
增强ECBS冲突解决 - 实时多车冲突检测与解决
接口系统 - 智能的路径接入点管理

智能调度

预测性调度 - 基于历史数据的任务分配优化
动态负载平衡 - 实时调整车辆任务分配
多级优先级 - 支持紧急任务插队
自适应参数调整 - 根据系统负载动态优化

现代化界面

实时可视化 - 车辆路径、冲突、骨干网络可视化
性能监控 - 实时统计图表和性能指标
交互式控制 - 直观的参数调整和任务管理
停靠窗口 - 灵活的界面布局

📋 系统要求
软件依赖

Python 3.8+
PyQt5 5.15+
NumPy 1.19+
Matplotlib 3.3+ (可选，用于高级图表)

硬件建议

RAM: 8GB+ (推荐16GB)
CPU: 4核心+ (支持并行处理)
显存: 2GB+ (用于图形渲染)

🛠 安装与配置
1. 克隆仓库
bashgit clone https://github.com/your-repo/mine-scheduling-system.git
cd mine-scheduling-system
2. 安装依赖
bashpip install -r requirements.txt
3. 运行系统
bashpython gui.py
📖 快速开始
基本工作流程

加载地图环境
文件 → 打开地图 → 选择JSON格式地图文件

生成骨干网络
点击"生成骨干网络" → 调整质量阈值和接口间距 → 等待生成完成

分配车辆任务
选择车辆 → 选择装载/卸载点 → 点击"分配任务"

启动仿真
点击"开始"按钮 → 调整仿真速度 → 观察实时执行


地图文件格式
系统支持JSON格式的地图文件：
json{
  "width": 500,
  "height": 500,
  "obstacles": [
    {"x": 100, "y": 100, "width": 50, "height": 30}
  ],
  "loading_points": [
    {"x": 50, "y": 50, "theta": 0}
  ],
  "unloading_points": [
    {"x": 450, "y": 450, "theta": 0}
  ],
  "vehicles_info": [
    {"id": "v1", "x": 25, "y": 25, "type": "dump_truck", "max_load": 100}
  ]
}
🏗 系统架构
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   GUI层         │    │   控制层        │    │   算法层        │
│                 │    │                 │    │                 │
│ • 图形界面      │◄──►│ • 车辆调度器    │◄──►│ • RRT规划器     │
│ • 实时监控      │    │ • 交通管理器    │    │ • ECBS求解器    │
│ • 用户交互      │    │ • 任务管理      │    │ • 路径优化      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   数据层        │
                    │                 │
                    │ • 环境模型      │
                    │ • 骨干网络      │
                    │ • 车辆状态      │
                    └─────────────────┘
📂 项目结构
mine-scheduling-system/
├── README.md                 # 项目说明文档
├── requirements.txt          # 依赖包列表
├── gui.py                   # 主GUI界面
├── environment.py           # 环境管理系统
├── backbone_network.py      # 骨干路径网络
├── path_planner.py         # 路径规划器
├── traffic_manager.py      # 交通管理器
├── vehicle_scheduler.py    # 车辆调度器
├── RRT.py                  # 混合双向RRT算法
├── examples/               # 示例地图和配置
│   ├── simple_mine.json   # 简单矿场地图
│   ├── complex_mine.json  # 复杂矿场地图
│   └── test_scenarios/    # 测试场景
├── docs/                   # 详细文档
│   ├── api.md             # API文档
│   ├── algorithms.md      # 算法说明
│   └── configuration.md   # 配置指南
└── tests/                  # 单元测试
    ├── test_environment.py
    ├── test_planner.py
    └── test_scheduler.py
🔧 核心组件详解
1. 骨干路径网络 (Backbone Network)
python# 生成骨干网络
backbone = SimplifiedBackbonePathNetwork(env)
success = backbone.generate_backbone_network(
    quality_threshold=0.6,
    interface_spacing=8
)
特性：

预计算高质量路径
智能接口点分布
支持路径快速拼接
RRT集成优化

2. 路径规划器 (Path Planner)
python# 规划路径
planner = SimplifiedPathPlanner(env, backbone)
path, structure = planner.plan_path(
    vehicle_id, start, goal,
    use_backbone=True
)
算法流程：

检查目标是否为特殊点
查找可用骨干路径
选择最优接口点
规划接入路径
拼接完整路径
回退到直接RRT（如需要）

3. 交通管理器 (Traffic Manager)
python# 冲突检测和解决
traffic_mgr = OptimizedTrafficManager(env, backbone)
conflicts = traffic_mgr.detect_all_conflicts()
resolved = traffic_mgr.resolve_conflicts_enhanced(conflicts)
冲突类型：

顶点冲突 (Vertex Conflicts)
边冲突 (Edge Conflicts)
接口冲突 (Interface Conflicts)
预测性冲突 (Predictive Conflicts)

4. 车辆调度器 (Vehicle Scheduler)
python# 智能任务分配
scheduler = SimplifiedVehicleScheduler(env, planner, backbone, traffic_mgr)
scheduler.create_enhanced_mission_template("default")
scheduler.assign_mission_intelligently(vehicle_id)
调度策略：

贪心分配
预测性调度
负载平衡
优先级管理

📊 性能优化
缓存系统

路径缓存 - LRU缓存常用路径
冲突缓存 - 缓存冲突检测结果
预测缓存 - 预测性路径缓存

并行处理

多线程冲突检测 - 并行检测车辆对冲突
批量路径规划 - 批量处理多个规划请求
异步更新 - 非阻塞状态更新

自适应参数
python# 根据系统负载调整参数
if system_load > 0.8:
    max_iterations = base_iterations * 0.6  # 降低迭代次数
    quality_threshold *= 0.9                # 降低质量要求
🔍 API 参考
环境管理
python# 创建环境
env = OpenPitMineEnv()
env.load_from_file("map.json")

# 添加车辆
env.add_vehicle("v1", position=(10, 10, 0), vehicle_type="dump_truck")

# 添加障碍物
env.add_obstacle(x=100, y=100, width=50, height=30)
路径规划
python# 基础路径规划
path, structure = planner.plan_path(
    vehicle_id="v1",
    start=(0, 0, 0),
    goal=(100, 100, 0),
    use_backbone=True,
    check_conflicts=True
)

# 高级规划请求
request = PlanningRequest(
    vehicle_id="v1",
    start=(0, 0, 0),
    goal=(100, 100, 0),
    priority=2,
    quality_requirement=0.8
)
result = advanced_planner.plan_path_advanced(request)
任务管理
python# 创建任务模板
scheduler.create_enhanced_mission_template(
    template_id="mining_cycle",
    loading_point_id=0,
    unloading_point_id=1
)

# 分配任务
success = scheduler.assign_mission_intelligently(
    vehicle_id="v1",
    template_id="mining_cycle"
)
📈 监控与调试
实时统计
python# 获取系统统计
stats = scheduler.get_comprehensive_stats()
print(f"完成任务: {stats['completed_tasks']}")
print(f"骨干利用率: {stats['backbone_utilization_rate']:.1%}")

# 获取性能指标
perf_stats = planner.get_performance_stats()
print(f"平均规划时间: {perf_stats['avg_planning_time']:.3f}s")
print(f"缓存命中率: {perf_stats['cache_hit_rate']:.1%}")
调试模式
python# 启用详细日志
planner.set_debug(True)
traffic_manager.verbose_logging = True

# 显示调试信息
backbone.debug_interface_system()
⚙️ 配置参数
核心参数
python# 骨干网络配置
BACKBONE_CONFIG = {
    'quality_threshold': 0.6,      # 路径质量阈值
    'interface_spacing': 8,        # 接口间距
    'max_attempts': 3              # 最大尝试次数
}

# ECBS配置
ECBS_CONFIG = {
    'suboptimality_bound': 1.3,    # 次优界限
    'max_search_time': 20.0,       # 最大搜索时间
    'safety_distance': 6.0         # 安全距离
}

# RRT配置
RRT_CONFIG = {
    'max_iterations': 4000,        # 最大迭代次数
    'step_size': 0.8,             # 步长
    'goal_bias': 0.1              # 目标偏置
}
性能调优
python# 启用并行处理
traffic_manager.parallel_detection = True
traffic_manager.max_detection_threads = 4

# 调整缓存大小
planner.cache_config['max_size'] = 500
planner.cache_config['ttl'] = 300

# 设置预测参数
scheduler.conflict_detection_interval = 15.0
traffic_manager.prediction_horizon = 50.0
🧪 测试与验证
运行测试套件
bash# 运行所有测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_planner.py -v

# 性能测试
python tests/performance_test.py
验证场景
python# 测试基本功能
python examples/basic_test.py

# 压力测试
python examples/stress_test.py

# 算法对比
python examples/algorithm_comparison.py
🚧 开发指南
代码风格

遵循PEP 8规范
使用类型提示
详细的文档字符串
单元测试覆盖

扩展开发
python# 自定义调度策略
class CustomScheduler(SimplifiedVehicleScheduler):
    def custom_assignment_strategy(self, vehicle_id):
        # 实现自定义逻辑
        pass

# 自定义冲突解决
class CustomECBS(EnhancedECBSSolver):
    def custom_conflict_resolution(self, conflict):
        # 实现自定义解决策略
        pass
🐛 常见问题
Q: 路径规划失败怎么办？
A: 检查起终点是否有效，调整RRT参数，确保环境配置正确。
Q: 系统运行缓慢？
A: 启用并行处理，调整缓存配置，降低图形更新频率。
Q: 车辆卡住不动？
A: 检查任务分配，验证路径有效性，查看冲突解决日志。
Q: 内存占用过高？
A: 限制缓存大小，清理历史数据，优化数据结构。
📝 更新日志
v3.0 (当前版本)

✅ 全新的骨干路径网络系统
✅ 增强的ECBS冲突解决
✅ 现代化GUI界面
✅ 智能接口系统
✅ 性能监控与优化

v2.1

✅ 基础RRT路径规划
✅ 简单任务调度
✅ 基本GUI界面

🤝 贡献指南

Fork 项目
创建特性分支 (git checkout -b feature/AmazingFeature)
提交更改 (git commit -m 'Add some AmazingFeature')
推送到分支 (git push origin feature/AmazingFeature)
开启Pull Request

📄 许可证
本项目基于 MIT 许可证开源 - 查看 LICENSE 文件了解详情。
👥 作者与贡献者

主要开发者 - 系统架构与算法实现
算法团队 - RRT与ECBS算法优化
界面团队 - GUI设计与用户体验

🙏 致谢

OpenAI - AI辅助开发支持
PyQt5团队 - 优秀的GUI框架
NumPy项目 - 高性能数值计算
开源社区 - 算法参考与灵感

📧 联系我们

项目主页: GitHub Repository
问题反馈: Issues
技术讨论: Discussions