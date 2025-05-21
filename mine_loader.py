import numpy as np
import json
import os
import matplotlib.pyplot as plt
from environment import OpenPitMineEnv

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
        rows = mine_data.get("dimensions", {}).get("rows", mine_data.get("height", 500))
        cols = mine_data.get("dimensions", {}).get("cols", mine_data.get("width", 500))
        env.width, env.height = cols, rows
        env.map_size = (cols, rows)
        
        # 重置网格
        env.grid = np.zeros((cols, rows), dtype=np.uint8)
        env.obstacle_points = []
        
        # 加载网格 - 转置坐标系(行列 -> xy)
        if "grid" in mine_data:
            grid = np.array(mine_data["grid"], dtype=np.int8)
            
            # 确保网格大小正确
            if grid.shape[0] != rows or grid.shape[1] != cols:
                print(f"警告: 网格维度与声明不符，调整为: {rows}x{cols}")
                temp_grid = np.zeros((rows, cols), dtype=np.int8)
                r, c = min(grid.shape[0], rows), min(grid.shape[1], cols)
                temp_grid[:r, :c] = grid[:r, :c]
                grid = temp_grid
            
            # 创建环境网格 - 需要转置，因为环境使用(x,y)而不是(row,col)
            for row in range(rows):
                for col in range(cols):
                    if grid[row, col] == 1:
                        # 添加障碍点
                        env.add_obstacle_point(col, row)
        elif "obstacles" in mine_data:
            # 从障碍物列表加载
            for obstacle in mine_data.get("obstacles", []):
                if isinstance(obstacle, dict):
                    x = obstacle.get("x", 0)
                    y = obstacle.get("y", 0)
                    width = obstacle.get("width", 1)
                    height = obstacle.get("height", 1)
                    
                    for i in range(width):
                        for j in range(height):
                            nx, ny = x + i, y + j
                            if 0 <= nx < cols and 0 <= ny < rows:
                                env.add_obstacle_point(nx, ny)
        
        # 加载装载点
        env.loading_points = []
        for point in mine_data.get("loading_points", []):
            if isinstance(point, list):
                if len(point) >= 2:
                    row, col = point[0], point[1]
                    theta = 0.0 if len(point) <= 2 else point[2]
                    env.add_loading_point((col, row, theta))
            elif isinstance(point, dict):
                x = point.get("x", 0)
                y = point.get("y", 0)
                theta = point.get("theta", 0.0)
                env.add_loading_point((x, y, theta))
        
        # 加载卸载点
        env.unloading_points = []
        for point in mine_data.get("unloading_points", []):
            if isinstance(point, list):
                if len(point) >= 2:
                    row, col = point[0], point[1]
                    theta = 0.0 if len(point) <= 2 else point[2]
                    env.add_unloading_point((col, row, theta))
            elif isinstance(point, dict):
                x = point.get("x", 0)
                y = point.get("y", 0)
                theta = point.get("theta", 0.0)
                env.add_unloading_point((x, y, theta))
        
        # 加载停车区
        env.parking_areas = []
        for point in mine_data.get("parking_areas", []):
            if isinstance(point, list):
                if len(point) >= 2:
                    row, col = point[0], point[1]
                    theta = 0.0 if len(point) <= 2 else point[2]
                    capacity = 5
                    if len(point) >= 4:
                        capacity = point[3]
                    env.add_parking_area((col, row, theta), capacity)
            elif isinstance(point, dict):
                x = point.get("x", 0)
                y = point.get("y", 0)
                theta = point.get("theta", 0.0)
                capacity = point.get("capacity", 5)
                env.add_parking_area((x, y, theta), capacity)
        
        # 清除现有车辆
        env.vehicles = {}
        
        # 添加车辆 - 转换行列坐标到xy坐标，添加朝向角度
        for i, point in enumerate(mine_data.get("vehicle_positions", [])):
            if isinstance(point, list):
                if len(point) >= 2:
                    row, col = point[0], point[1]
                    theta = 0.0 if len(point) <= 2 else point[2]
                    env.add_vehicle(f"vehicle_{i+1}", (col, row, theta))
            elif isinstance(point, dict):
                x = point.get("x", 0)
                y = point.get("y", 0)
                theta = point.get("theta", 0.0)
                env.add_vehicle(f"vehicle_{i+1}", (x, y, theta))
        
        # 加载车辆详细信息
        for vehicle in mine_data.get("vehicles_info", []):
            if isinstance(vehicle, dict):
                vehicle_id = str(vehicle.get("id", f"v_{len(env.vehicles) + 1}"))
                x = vehicle.get("x", 0)
                y = vehicle.get("y", 0)
                theta = vehicle.get("theta", 0.0)
                v_type = vehicle.get("type", "dump_truck")
                max_load = vehicle.get("max_load", 100)
                
                # 如果车辆已存在，则更新信息；否则添加新车辆
                if vehicle_id in env.vehicles:
                    env.vehicles[vehicle_id]["type"] = v_type
                    env.vehicles[vehicle_id]["max_load"] = max_load
                else:
                    env.add_vehicle(vehicle_id, (x, y, theta), None, v_type, max_load)
        
        # 加载场景数据(如果文件存在)
        scen_file = map_file.replace("_mine.json", ".scen").replace(".json", ".scen")
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
        env.grid = np.zeros((width, height), dtype=np.uint8)
        env.obstacle_points = []
        
        # 加载网格数据
        for y, line in enumerate(lines[map_section_start:map_section_start+height]):
            if y >= height:
                break
                
            for x, char in enumerate(line.strip()):
                if x >= width:
                    break
                    
                # 将 . 转换为 0(可通行)，@ 或 T 或 O 等转换为 1(障碍物)
                if char != '.':
                    env.add_obstacle_point(x, y)
        
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
                    if len(vehicle_pos) >= 2:
                        # 确保格式正确并转换坐标
                        start_x, start_y = vehicle_pos[1], vehicle_pos[0]  # 转换坐标
                        theta = 0.0 if len(vehicle_pos) <= 2 else vehicle_pos[2]
                        
                        # 添加车辆
                        vehicle_id = str(scenario.get("id", f"v_{len(env.vehicles) + 1}"))
                        if env.add_vehicle(vehicle_id, (start_x, start_y, theta)):
                            # 设置初始位置
                            env.vehicles[vehicle_id]["initial_position"] = (start_x, start_y, theta)
                            
                            # 设置目标位置（如果有）
                            if "goal" in scenario:
                                goal_pos = scenario["goal"]
                                if len(goal_pos) >= 2:
                                    goal_x, goal_y = goal_pos[1], goal_pos[0]  # 转换坐标
                                    goal_theta = 0.0 if len(goal_pos) <= 2 else goal_pos[2]
                                    env.vehicles[vehicle_id]["goal"] = (goal_x, goal_y, goal_theta)
                            
                            print(f"  添加车辆 {vehicle_id} 在位置 {(start_x, start_y, theta)}")
                        else:
                            print(f"  添加车辆 {vehicle_id} 失败，位置可能无效: {(start_x, start_y, theta)}")
            
            return True
        except Exception as e:
            print(f"加载场景文件出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_mapf_scenario(self, env, scen_file):
        """加载标准MAPF场景文件"""
        try:
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
                    vehicle_id = f"vehicle_{task_id+1}"
                    if env.add_vehicle(vehicle_id, (start_x, start_y, theta), (goal_x, goal_y, theta)):
                        # 为简单起见，将目标点标记为装载点或卸载点(交替模式)
                        if task_id % 2 == 0:
                            if (goal_x, goal_y) not in [(p[0], p[1]) for p in env.loading_points]:
                                env.add_loading_point((goal_x, goal_y, theta))
                        else:
                            if (goal_x, goal_y) not in [(p[0], p[1]) for p in env.unloading_points]:
                                env.add_unloading_point((goal_x, goal_y, theta))
                        
                        scenarios.append({
                            "id": vehicle_id,
                            "start": {"x": start_x, "y": start_y, "theta": theta},
                            "goal": {"x": goal_x, "y": goal_y, "theta": theta}
                        })
                        
                        print(f"  添加车辆 {vehicle_id} 在位置 {(start_x, start_y, theta)} 目标 {(goal_x, goal_y, theta)}")
                    else:
                        print(f"  添加车辆 {vehicle_id} 失败，位置可能无效")
            
            print(f"已加载 {len(scenarios)} 个场景任务")
            return scenarios
        except Exception as e:
            print(f"加载MAPF场景文件出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    def load_mine_map(self, data):
        """加载标准矿山格式的数据"""
        # 获取尺寸
        self.rows = data.get("dimensions", {}).get("rows", self.rows)
        self.cols = data.get("dimensions", {}).get("cols", self.cols)
        
        # 更新UI输入框
        self.col_entry.delete(0, tk.END)
        self.col_entry.insert(0, str(self.cols))
        self.row_entry.delete(0, tk.END)
        self.row_entry.insert(0, str(self.rows))
        
        # 创建新网格
        if "grid" in data:
            # 直接加载网格数据
            self.grid = np.array(data["grid"], dtype=np.int8)
        else:
            # 网格不存在，重置并从障碍物填充
            self.grid = np.zeros((self.rows, self.cols), dtype=np.int8)
            
            # 从矩形障碍物设置网格
            for obstacle in data.get("obstacles", []):
                x = int(obstacle["x"])
                y = int(obstacle["y"])
                width = int(obstacle.get("width", 1))
                height = int(obstacle.get("height", 1))
                
                # 填充网格障碍物
                for r in range(max(0, y), min(self.rows, y + height)):
                    for c in range(max(0, x), min(self.cols, x + width)):
                        self.grid[r, c] = 1
        
        # 清除现有特殊点
        self.loading_points = []
        self.unloading_points = []
        self.parking_areas = []
        self.vehicles = []
        
        # 加载装载点 - 处理两种可能的格式
        for point in data.get("loading_points", []):
            if isinstance(point, list):
                # 格式是 [row, col, theta]
                row, col = point[0], point[1]
                theta = point[2] if len(point) > 2 else 0.0
                self.loading_points.append({
                    "x": col,  # 注意x对应col
                    "y": row,  # y对应row
                    "theta": theta
                })
            elif isinstance(point, dict):
                # 直接字典格式
                self.loading_points.append(point.copy())
        
        # 加载卸载点 - 处理两种可能的格式
        for point in data.get("unloading_points", []):
            if isinstance(point, list):
                # 格式是 [row, col, theta]
                row, col = point[0], point[1]
                theta = point[2] if len(point) > 2 else 0.0
                self.unloading_points.append({
                    "x": col,
                    "y": row,
                    "theta": theta
                })
            elif isinstance(point, dict):
                # 直接字典格式
                self.unloading_points.append(point.copy())
        
        # 加载停车区
        for point in data.get("parking_areas", []):
            if isinstance(point, list):
                # 格式是 [row, col, theta, capacity]
                row, col = point[0], point[1]
                theta = point[2] if len(point) > 2 else 0.0
                capacity = point[3] if len(point) > 3 else 5
                self.parking_areas.append({
                    "x": col,
                    "y": row,
                    "theta": theta,
                    "capacity": capacity
                })
            elif isinstance(point, dict):
                # 直接字典格式
                self.parking_areas.append(point.copy())
        
        # 加载车辆
        # 首先尝试vehicle_positions字段
        for i, point in enumerate(data.get("vehicle_positions", [])):
            if isinstance(point, list):
                # 格式是 [row, col, theta]
                row, col = point[0], point[1]
                theta = point[2] if len(point) > 2 else 0.0
                self.vehicles.append({
                    "id": i+1,
                    "x": col,
                    "y": row,
                    "theta": theta,
                    "type": "dump_truck",
                    "max_load": 100.0
                })
        
        # 然后尝试vehicles_info字段
        for vehicle in data.get("vehicles_info", []):
            if isinstance(vehicle, dict):
                self.vehicles.append(vehicle.copy())
def visualize_environment(env):
    """简单可视化环境"""
    # 创建图像
    plt.figure(figsize=(10, 10))
    
    # 绘制障碍物
    obstacle_x = []
    obstacle_y = []
    for x in range(env.width):
        for y in range(env.height):
            if env.grid[x, y] == 1:
                obstacle_x.append(x)
                obstacle_y.append(y)
    
    plt.scatter(obstacle_x, obstacle_y, c='black', marker='s', s=5)
    
    # 绘制装载点
    loading_x = [p[0] for p in env.loading_points]
    loading_y = [p[1] for p in env.loading_points]
    plt.scatter(loading_x, loading_y, c='green', marker='o', s=50)
    
    # 绘制卸载点
    unloading_x = [p[0] for p in env.unloading_points]
    unloading_y = [p[1] for p in env.unloading_points]
    plt.scatter(unloading_x, unloading_y, c='red', marker='o', s=50)
    
    # 绘制车辆
    for v_id, vehicle in env.vehicles.items():
        x, y, theta = vehicle['position']
        plt.scatter(x, y, c='blue', marker='*', s=80)
        
        # 绘制车辆朝向
        dx = np.cos(theta) * 5
        dy = np.sin(theta) * 5
        plt.arrow(x, y, dx, dy, head_width=2, head_length=2, fc='blue', ec='blue')
        
        # 绘制车辆目标（如果有）
        if vehicle['goal'] is not None:
            gx, gy, _ = vehicle['goal']
            plt.scatter(gx, gy, c='blue', marker='x', s=50)
    
    plt.xlim(0, env.width)
    plt.ylim(0, env.height)
    plt.title('矿山环境可视化')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def test_mine_loader(map_file, scen_file=None):
    """测试地图加载器"""
    # 创建加载器并加载环境
    loader = MineEnvironmentLoader()
    env = loader.load_environment(map_file, scen_file)
    
    # 打印环境信息
    print(f"环境大小: {env.width}x{env.height}")
    print(f"障碍点数量: {len(env.obstacle_points)}")
    print(f"装载点数量: {len(env.loading_points)}")
    print(f"卸载点数量: {len(env.unloading_points)}")
    print(f"车辆数量: {len(env.vehicles)}")
    
    # 可视化环境
    visualize_environment(env)
    
    return env

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python mine_loader.py <map_file> [scen_file]")
        sys.exit(1)
    
    map_file = sys.argv[1]
    scen_file = sys.argv[2] if len(sys.argv) >= 3 else None
    
    test_mine_loader(map_file, scen_file)