import numpy as np
import json
import os
import matplotlib.pyplot as plt
from environment import OpenPitMineEnv
from hybrid_astar import HybridAStarPlanner
from path_utils import improved_visualize_environment

class MineEnvironmentLoader:
    """用于加载各种格式地图文件到露天矿环境的工具类"""
    
    def __init__(self):
        """初始化加载器"""
        pass
    
    def load_environment(self, map_file, scen_file=None):
        """根据文件扩展名自动选择加载方法"""
        print(f"加载地图文件: {map_file}")
        
        # 创建环境对象
        env = OpenPitMineEnv()
        
        # 根据文件扩展名选择加载方法
        if map_file.endswith("_mine.json") or map_file.endswith(".json"):
            self.load_mine_json(env, map_file)
        elif map_file.endswith(".map"):
            self.load_mapf_file(env, map_file, scen_file)
        else:
            raise ValueError(f"不支持的地图文件格式: {map_file}")
            
        print(f"地图加载完成，大小: {env.width}x{env.height}")
        return env
    
    def load_mine_json(self, env, map_file):
        """加载自定义矿山JSON格式文件到环境"""
        with open(map_file, 'r') as f:
            mine_data = json.load(f)
        
        # 设置维度
        rows = mine_data["dimensions"]["rows"]
        cols = mine_data["dimensions"]["cols"]
        env.width, env.height = cols, rows
        env.map_size = (cols, rows)
        
        # 加载网格 - 转置坐标系(行列 -> xy)
        grid = np.array(mine_data["grid"], dtype=np.int8)
        
        # 确保网格大小正确
        if grid.shape[0] != rows or grid.shape[1] != cols:
            print(f"警告: 网格维度与声明不符，调整为: {rows}x{cols}")
            temp_grid = np.zeros((rows, cols), dtype=np.int8)
            r, c = min(grid.shape[0], rows), min(grid.shape[1], cols)
            temp_grid[:r, :c] = grid[:r, :c]
            grid = temp_grid
        
        # 创建环境网格 - 需要转置，因为环境使用(x,y)而不是(row,col)
        env.grid = np.zeros((cols, rows), dtype=np.int8)
        for row in range(rows):
            for col in range(cols):
                env.grid[col, row] = grid[row, col]
        
        # 加载特殊点 - 转换行列坐标到xy坐标
        env.loading_points = []
        for point in mine_data.get("loading_points", []):
            row, col = point
            env.add_loading_point((col, row))  # 转换为(x,y)
        
        env.unloading_points = []
        for point in mine_data.get("unloading_points", []):
            row, col = point
            env.add_unloading_point((col, row))  # 转换为(x,y)
        
        # 清除现有车辆
        env.vehicles = {}
        
        # 添加车辆 - 转换行列坐标到xy坐标，添加朝向角度
        for i, point in enumerate(mine_data.get("vehicle_positions", [])):
            row, col = point
            # 默认朝向0度(向右)
            env.add_vehicle(i+1, (col, row, 0))  # 转换为(x,y,theta)
        
        # 加载场景数据(如果文件存在)
        scen_file = map_file.replace(".json", ".scen")
        if os.path.exists(scen_file):
            print(f"发现匹配的场景文件，尝试加载: {scen_file}")
            self.load_mine_scenario(env, scen_file)
        
        return env
    
    def load_mapf_file(self, env, map_file, scen_file=None):
        """加载标准MAPF格式文件到环境"""
        # 读取.map文件
        with open(map_file, 'r') as f:
            lines = f.readlines()
        
        # 解析头部信息
        width = 0
        height = 0
        map_section_start = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("height"):
                height = int(line.split()[1])
            elif line.startswith("width"):
                width = int(line.split()[1])
            elif line == "map":
                map_section_start = i + 1
                break
        
        # 初始化网格
        env.width = width
        env.height = height
        env.map_size = (width, height)
        env.grid = np.zeros((width, height), dtype=np.int8)
        
        # 加载网格数据
        for y, line in enumerate(lines[map_section_start:map_section_start+height]):
            if y >= height:
                break
                
            for x, char in enumerate(line.strip()):
                if x >= width:
                    break
                    
                # 将 . 转换为 0(可通行)，@ 转换为 1(障碍物)
                env.grid[x, y] = 0 if char == '.' else 1
        
        # 清除现有点位
        env.loading_points = []
        env.unloading_points = []
        env.vehicles = {}
        
        # 如果提供了场景文件，设置车辆和任务
        if scen_file and os.path.exists(scen_file):
            self.load_mapf_scenario(env, scen_file)
        
        return env
    
    def load_mine_scenario(self, env, scen_file):
        """加载矿山场景文件 - 修复版本"""
        try:
            with open(scen_file, 'r') as f:
                scen_data = json.load(f)
            
            print(f"加载场景文件: {scen_file}")
            
            scenarios = scen_data.get("scenarios", [])
            print(f"找到 {len(scenarios)} 个场景任务")
            
            # 处理每个场景中的车辆和任务链
            for scenario in scenarios:
                # 获取车辆初始位置
                if "initial_position" in scenario:
                    vehicle_pos = scenario["initial_position"]
                    start_x, start_y = vehicle_pos[1], vehicle_pos[0]  # 转换坐标
                    
                    # 添加车辆
                    vehicle_id = scenario.get("id", len(env.vehicles) + 1)
                    env.add_vehicle(vehicle_id, (start_x, start_y, 0))
                    env.vehicles[vehicle_id]["initial_position"] = (start_x, start_y, 0)
                    
                    # 加载任务队列
                    if "tasks" in scenario and len(scenario["tasks"]) > 0:
                        env.vehicles[vehicle_id]["task_queue"] = []
                        
                        # 打印任务数量
                        print(f"  车辆 {vehicle_id}: 发现 {len(scenario['tasks'])} 个任务")
                        
                        for task in scenario["tasks"]:
                            # 确保任务有必需的字段
                            if "start" not in task or "goal" not in task:
                                print(f"  警告: 任务缺少起点或终点，跳过")
                                continue
                            
                            # 转换坐标
                            start_row, start_col = task["start"]["row"], task["start"]["col"]
                            goal_row, goal_col = task["goal"]["row"], task["goal"]["col"]
                            
                            task_entry = {
                                "start": (start_col, start_row, 0),
                                "goal": (goal_col, goal_row, 0),
                                "task_type": task.get("task_type", "generic"),
                                "vehicle_type": task.get("vehicle_type", "generic_vehicle")
                            }
                            
                            # 添加任务到队列
                            env.vehicles[vehicle_id]["task_queue"].append(task_entry)
                            print(f"    添加任务: {task_entry['task_type']} - 从 {task_entry['start']} 到 {task_entry['goal']}")
                        
                        # 设置第一个任务为当前目标
                        if env.vehicles[vehicle_id]["task_queue"]:
                            first_task = env.vehicles[vehicle_id]["task_queue"][0]
                            env.vehicles[vehicle_id]["goal"] = first_task["goal"]
                            env.vehicles[vehicle_id]["status"] = "idle"
                            env.vehicles[vehicle_id]["current_task_index"] = 0
                            print(f"  设置车辆 {vehicle_id} 初始目标: {first_task['goal']}")
                        else:
                            # 如果任务队列为空，删除键以便自动创建任务
                            print(f"  警告: 车辆 {vehicle_id} 任务队列为空，将被移除")
                            if "task_queue" in env.vehicles[vehicle_id]:
                                del env.vehicles[vehicle_id]["task_queue"]
                    else:
                        print(f"  车辆 {vehicle_id}: 无任务")
            
            # 确保所有车辆的任务队列非空
            for v_id, vehicle in env.vehicles.items():
                if "task_queue" in vehicle and len(vehicle["task_queue"]) == 0:
                    print(f"警告: 车辆 {v_id} 任务队列为空，将被移除")
                    del vehicle["task_queue"]
                elif "task_queue" in vehicle:
                    print(f"车辆 {v_id} 有 {len(vehicle['task_queue'])} 个任务")
                    
            return True
        except Exception as e:
            print(f"加载场景文件出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_mapf_scenario(self, env, scen_file):
        """加载标准MAPF场景文件"""
        with open(scen_file, 'r') as f:
            lines = f.readlines()
        
        # 跳过头部行
        if not lines:
            return []
            
        scenarios = []
        for i, line in enumerate(lines[1:], 1):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) >= 9:
                # 解析任务
                task_id = int(parts[0])
                start_x = int(parts[4])
                start_y = int(parts[5])
                goal_x = int(parts[6])
                goal_y = int(parts[7])
                
                # 计算基础朝向角度
                dx = goal_x - start_x
                dy = goal_y - start_y
                theta = np.arctan2(dy, dx) if abs(dx) > 0.001 or abs(dy) > 0.001 else 0
                
                # 添加车辆
                env.add_vehicle(task_id+1, (start_x, start_y, theta), (goal_x, goal_y, theta))
                
                # 随机分配颜色
                env.vehicles[task_id+1]["color"] = np.random.rand(3)
                
                # 为简单起见，将目标点标记为装载点或卸载点(交替模式)
                if task_id % 2 == 0:
                    if (goal_x, goal_y) not in [p[:2] for p in env.loading_points]:
                        env.add_loading_point((goal_x, goal_y))
                else:
                    if (goal_x, goal_y) not in [p[:2] for p in env.unloading_points]:
                        env.add_unloading_point((goal_x, goal_y))
                
                scenarios.append({
                    "id": task_id+1,
                    "start": {"x": start_x, "y": start_y},
                    "goal": {"x": goal_x, "y": goal_y}
                })
        
        print(f"已加载 {len(scenarios)} 个场景任务")
        return scenarios

def test_mine_path_planning_from_file(map_file, scen_file=None):
    """从文件测试路径规划"""
    print(f"从文件加载环境: {map_file}")
    
    # 创建加载器并加载环境
    loader = MineEnvironmentLoader()
    env = loader.load_environment(map_file, scen_file)
    
    # 创建路径规划器
    planner = HybridAStarPlanner(env, 
                               vehicle_length=6.0, 
                               vehicle_width=3.0, 
                               turning_radius=8.0,
                               step_size=0.8,
                               grid_resolution=0.3)
    
    # 启用路径平滑和RS曲线
    planner.path_smoothing = True
    planner.smoothing_factor = 0.5
    planner.smoothing_iterations = 10
    planner.rs_step_size = 0.2
    planner.use_rs_heuristic = True
    planner.analytic_expansion_step = 5
    
    # 可视化初始环境
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 12))
    improved_visualize_environment(env, planner, ax)
    ax.set_title("从文件加载的环境")
    fig.canvas.draw()
    plt.pause(2)
    
    # 为每个车辆规划路径
    for v_id, vehicle in env.vehicles.items():
        if vehicle['goal'] is not None:
            print(f"规划车辆 {v_id} 的路径...")
            path = planner.plan_path(
                vehicle['position'],
                vehicle['goal']
            )
            
            if path:
                vehicle['path'] = path
                vehicle['status'] = 'moving'
                print(f"  路径长度: {len(path)}")
            else:
                print(f"  无法为车辆 {v_id} 找到路径")
    
    # 可视化所有路径
    improved_visualize_environment(env, planner, ax)
    ax.set_title("规划的路径")
    fig.canvas.draw()
    plt.pause(2)
    
    # 模拟移动
    all_paths = [v['path'] for v in env.vehicles.values() if 'path' in v and v['path']]
    if all_paths:
        max_steps = max(len(path) for path in all_paths)
        step_interval = max(1, max_steps // 50)
        
        for i in range(0, max_steps, step_interval):
            # 更新车辆位置
            for v_id, vehicle in env.vehicles.items():
                if 'path' in vehicle and vehicle['path'] and i < len(vehicle['path']):
                    vehicle['position'] = vehicle['path'][i]
            
            # 可视化当前状态
            improved_visualize_environment(env, planner, ax)
            ax.set_title(f'模拟步骤: {i}/{max_steps}')
            fig.canvas.draw()
            plt.pause(0.05)
    
    plt.ioff()
    plt.show()
    
    return env, planner

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python mine_loader.py <map_file> [scen_file]")
        sys.exit(1)
    
    map_file = sys.argv[1]
    scen_file = sys.argv[2] if len(sys.argv) >= 3 else None
    
    test_mine_path_planning_from_file(map_file, scen_file)