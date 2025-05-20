import numpy as np
import json
import os
import random

class MineFormatConverter:
    """转换不同格式的矿山地图和场景文件"""
    
    def __init__(self):
        """初始化转换器"""
        pass
    
    def convert_to_mine_json(self, input_file, output_file=None):
        """将其他格式转换为矿山JSON格式"""
        if input_file.endswith('.map'):
            return self.convert_mapf_to_mine_json(input_file, output_file)
        else:
            raise ValueError(f"不支持的输入文件格式: {input_file}")
    
    def convert_mapf_to_mine_json(self, map_file, output_file=None, scen_file=None):
        """将MAPF格式转换为矿山JSON格式"""
        # 设置默认输出文件名
        if output_file is None:
            base_name = os.path.splitext(map_file)[0]
            output_file = f"{base_name}_mine.json"
        
        print(f"转换MAPF文件 {map_file} 到矿山JSON格式 {output_file}")
        
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
        
        if width == 0 or height == 0 or map_section_start == 0:
            raise ValueError("无效的MAPF文件格式")
        
        # 创建网格
        grid = np.zeros((height, width), dtype=np.int8)
        
        # 解析地图数据
        for row, line in enumerate(lines[map_section_start:map_section_start+height]):
            if row >= height:
                break
                
            for col, char in enumerate(line.strip()):
                if col >= width:
                    break
                    
                # 将 . 转换为 0(可通行)，@ 转换为 1(障碍物)
                grid[row, col] = 0 if char == '.' else 1
        
        # 初始化特殊点位
        loading_points = []
        unloading_points = []
        vehicle_positions = []
        
        # 如果有场景文件，解析场景数据
        if scen_file and os.path.exists(scen_file):
            loading_points, unloading_points, vehicle_positions = self.extract_points_from_scen(scen_file, width, height)
        else:
            # 尝试查找自动生成的场景文件
            base_name = os.path.splitext(map_file)[0]
            auto_scen_file = f"{base_name}.scen"
            if os.path.exists(auto_scen_file):
                loading_points, unloading_points, vehicle_positions = self.extract_points_from_scen(auto_scen_file, width, height)
        
        # 如果没有特殊点，创建一些示例点
        if not (loading_points or unloading_points or vehicle_positions):
            loading_points, unloading_points, vehicle_positions = self.generate_example_points(grid)
        
        # 创建矿山JSON结构
        mine_data = {
            "grid": grid.tolist(),
            "dimensions": {"rows": height, "cols": width},
            "loading_points": loading_points,
            "unloading_points": unloading_points,
            "vehicle_positions": vehicle_positions
        }
        
        # 保存到JSON文件
        with open(output_file, 'w') as f:
            json.dump(mine_data, f, indent=2)
        
        print(f"转换完成，已保存到 {output_file}")
        print(f"  装载点: {len(loading_points)}")
        print(f"  卸载点: {len(unloading_points)}")
        print(f"  车辆位置: {len(vehicle_positions)}")
        
        # 如果没有对应的场景文件，创建一个
        mine_scen_file = output_file.replace(".json", ".scen")
        self.create_mine_scenarios(mine_data, mine_scen_file)
        
        return output_file
    
    def extract_points_from_scen(self, scen_file, width, height):
        """从场景文件中提取特殊点位"""
        loading_points = []
        unloading_points = []
        vehicle_positions = []
        
        if scen_file.endswith('.scen'):
            # 尝试读取标准MAPF场景文件
            with open(scen_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) <= 1:
                return [], [], []
                
            # 处理每个任务
            for i, line in enumerate(lines[1:], 1):
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split()
                if len(parts) >= 9:
                    # 解析任务
                    start_x = int(parts[4])
                    start_y = int(parts[5])
                    goal_x = int(parts[6])
                    goal_y = int(parts[7])
                    
                    # 确保坐标在范围内
                    if 0 <= start_y < height and 0 <= start_x < width:
                        vehicle_positions.append((start_y, start_x))  # 行列格式
                    
                    # 基于任务ID分配装载点或卸载点
                    if i % 2 == 0:
                        if 0 <= goal_y < height and 0 <= goal_x < width:
                            loading_points.append((goal_y, goal_x))  # 行列格式
                    else:
                        if 0 <= goal_y < height and 0 <= goal_x < width:
                            unloading_points.append((goal_y, goal_x))  # 行列格式
        
        # 去除重复点
        loading_points = list(set(loading_points))
        unloading_points = list(set(unloading_points))
        vehicle_positions = list(set(vehicle_positions))
        
        return loading_points, unloading_points, vehicle_positions
    
    def generate_example_points(self, grid):
        """生成示例特殊点位"""
        height, width = grid.shape
        loading_points = []
        unloading_points = []
        vehicle_positions = []
        
        # 找出所有可通行单元格
        traversable = []
        for row in range(height):
            for col in range(width):
                if grid[row, col] == 0:  # 可通行
                    traversable.append((row, col))
        
        if len(traversable) < 3:
            return [], [], []
        
        # 生成几个装载点(左侧)
        left_half = [(r, c) for r, c in traversable if c < width // 2]
        if left_half:
            num_loading = min(3, len(left_half))
            loading_points = random.sample(left_half, num_loading)
        
        # 生成几个卸载点(右侧)
        right_half = [(r, c) for r, c in traversable if c >= width // 2]
        if right_half:
            num_unloading = min(3, len(right_half))
            unloading_points = random.sample(right_half, num_unloading)
        
        # 生成几个车辆位置(靠近装载点)
        if loading_points:
            for lp in loading_points:
                nearby = [(r, c) for r, c in traversable 
                         if abs(r - lp[0]) + abs(c - lp[1]) < 5 and (r, c) != lp]
                if nearby:
                    vehicle_positions.append(random.choice(nearby))
        
        # 如果没有车辆位置，随机选择
        if not vehicle_positions and traversable:
            num_vehicles = min(2, len(traversable))
            vehicle_candidates = [p for p in traversable 
                                if p not in loading_points and p not in unloading_points]
            if vehicle_candidates:
                vehicle_positions = random.sample(vehicle_candidates, num_vehicles)
        
        return loading_points, unloading_points, vehicle_positions
    
    def create_mine_scenarios(self, mine_data, output_file, num_scenarios=5):
        grid = np.array(mine_data["grid"])
        loading_points = mine_data["loading_points"]
        unloading_points = mine_data["unloading_points"]
        vehicle_positions = mine_data["vehicle_positions"]
        
        # 创建场景结构
        scenarios = []
        
        # 为每个车辆创建完整三段任务
        for i, vehicle_pos in enumerate(vehicle_positions):
            if i >= num_scenarios:
                break
                
            # 选择最近的装载点
            min_dist = float('inf')
            closest_loading = None
            for loading_pos in loading_points:
                dist = abs(vehicle_pos[0] - loading_pos[0]) + abs(vehicle_pos[1] - loading_pos[1])
                if dist < min_dist:
                    min_dist = dist
                    closest_loading = loading_pos
            
            # 选择最近的卸载点
            min_dist = float('inf')
            closest_unloading = None
            for unloading_pos in unloading_points:
                dist = abs(closest_loading[0] - unloading_pos[0]) + abs(closest_loading[1] - unloading_pos[1])
                if dist < min_dist:
                    min_dist = dist
                    closest_unloading = unloading_pos
            
            if closest_loading and closest_unloading:
                # 创建三段完整任务链
                vehicle_scenario = {
                    "id": i + 1,
                    "initial_position": vehicle_pos,
                    "tasks": [
                        # 1. 起点到装载点
                        {
                            "start": {"row": vehicle_pos[0], "col": vehicle_pos[1]},
                            "goal": {"row": closest_loading[0], "col": closest_loading[1]},
                            "vehicle_type": "empty_truck",
                            "task_type": "to_loading"
                        },
                        # 2. 装载点到卸载点
                        {
                            "start": {"row": closest_loading[0], "col": closest_loading[1]},
                            "goal": {"row": closest_unloading[0], "col": closest_unloading[1]},
                            "vehicle_type": "mining_truck", 
                            "task_type": "to_unloading"
                        },
                        # 3. 卸载点回起点
                        {
                            "start": {"row": closest_unloading[0], "col": closest_unloading[1]},
                            "goal": {"row": vehicle_pos[0], "col": vehicle_pos[1]},
                            "vehicle_type": "empty_truck",
                            "task_type": "to_initial"
                        }
                    ]
                }
                scenarios.append(vehicle_scenario)
    
    def convert_mine_json_to_mapf(self, mine_json_file, output_map=None, output_scen=None):
        """将矿山JSON格式转换为MAPF格式"""
        # 设置默认输出文件名
        if output_map is None:
            base_name = os.path.splitext(mine_json_file)[0].replace('_mine', '')
            output_map = f"{base_name}.map"
        
        if output_scen is None:
            base_name = os.path.splitext(output_map)[0]
            output_scen = f"{base_name}.scen"
        
        print(f"转换矿山JSON文件 {mine_json_file} 到MAPF格式")
        
        # 读取JSON文件
        with open(mine_json_file, 'r') as f:
            mine_data = json.load(f)
        
        # 提取网格数据
        grid = np.array(mine_data["grid"])
        height, width = grid.shape
        
        # 生成.map文件
        with open(output_map, 'w') as f:
            f.write(f"type octile\n")
            f.write(f"height {height}\n")
            f.write(f"width {width}\n")
            f.write("map\n")
            
            for row in range(height):
                line = ""
                for col in range(width):
                    if grid[row, col] == 0:  # 可通行
                        line += "."
                    else:  # 障碍物
                        line += "@"
                f.write(line + "\n")
        
        print(f"已生成MAPF地图文件: {output_map}")
        
        # 生成场景文件
        self.generate_mapf_scenario(mine_data, output_map, output_scen)
        
        return output_map, output_scen
    
    def generate_mapf_scenario(self, mine_data, map_file, output_scen):
        """生成MAPF场景文件"""
        # 提取特殊点
        loading_points = mine_data.get("loading_points", [])
        unloading_points = mine_data.get("unloading_points", [])
        vehicle_positions = mine_data.get("vehicle_positions", [])
        
        # 获取地图尺寸
        if "dimensions" in mine_data:
            height = mine_data["dimensions"]["rows"]
            width = mine_data["dimensions"]["cols"]
        else:
            grid = np.array(mine_data["grid"])
            height, width = grid.shape
        
        # 打开场景文件
        with open(output_scen, 'w') as f:
            f.write("version 1\n")
            
            task_id = 0
            
            # 生成从车辆位置到装载点的任务
            for vehicle_pos in vehicle_positions:
                # 查找最近的装载点
                if loading_points:
                    min_dist = float('inf')
                    closest_loading = None
                    
                    for loading_pos in loading_points:
                        dist = abs(vehicle_pos[0] - loading_pos[0]) + abs(vehicle_pos[1] - loading_pos[1])
                        if dist < min_dist:
                            min_dist = dist
                            closest_loading = loading_pos
                    
                    if closest_loading:
                        # 转换行列坐标为xy坐标
                        start_row, start_col = vehicle_pos
                        goal_row, goal_col = closest_loading
                        
                        # 计算曼哈顿距离
                        manhattan_dist = abs(goal_row - start_row) + abs(goal_col - start_col)
                        optimal_length = manhattan_dist * random.uniform(1.0, 1.5)
                        
                        # 写入任务
                        f.write(f"{task_id} {os.path.basename(map_file)} {width} {height} {start_col} {start_row} {goal_col} {goal_row} {optimal_length:.6f}\n")
                        task_id += 1
            
            # 生成从装载点到卸载点的任务
            for i, loading_pos in enumerate(loading_points):
                if i < len(unloading_points):
                    unloading_pos = unloading_points[i]
                    
                    # 转换坐标
                    start_row, start_col = loading_pos
                    goal_row, goal_col = unloading_pos
                    
                    # 计算曼哈顿距离
                    manhattan_dist = abs(goal_row - start_row) + abs(goal_col - start_col)
                    optimal_length = manhattan_dist * random.uniform(1.0, 1.5)
                    
                    # 写入任务
                    f.write(f"{task_id} {os.path.basename(map_file)} {width} {height} {start_col} {start_row} {goal_col} {goal_row} {optimal_length:.6f}\n")
                    task_id += 1
        
        print(f"已生成MAPF场景文件: {output_scen} 包含 {task_id} 个任务")
        return output_scen

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  python mine_format_converter.py convert_to_mine input.map [output.json]")
        print("  python mine_format_converter.py convert_to_mapf input_mine.json [output.map] [output.scen]")
        sys.exit(1)
    
    converter = MineFormatConverter()
    
    command = sys.argv[1]
    
    if command == "convert_to_mine":
        if len(sys.argv) < 3:
            print("请提供输入文件名")
            sys.exit(1)
            
        input_file = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) >= 4 else None
        
        converter.convert_to_mine_json(input_file, output_file)
        
    elif command == "convert_to_mapf":
        if len(sys.argv) < 3:
            print("请提供输入文件名")
            sys.exit(1)
            
        input_file = sys.argv[2]
        output_map = sys.argv[3] if len(sys.argv) >= 4 else None
        output_scen = sys.argv[4] if len(sys.argv) >= 5 else None
        
        converter.convert_mine_json_to_mapf(input_file, output_map, output_scen)
        
    else:
        print(f"未知命令: {command}")
        sys.exit(1)