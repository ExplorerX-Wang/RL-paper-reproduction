import carla
import math
import numpy as np
from typing import List, Tuple, Optional
from enum import Enum


class RoadOption(Enum):
    """道路选项枚举"""
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


class GlobalPathPlanner:
    """
    Carla全局路径规划器
    考虑车道方向和交通规则的路径规划算法
    """

    def __init__(self, world_map: carla.Map, sampling_resolution: float = 2.0):
        """
        初始化路径规划器

        Args:
            world_map: Carla地图对象
            sampling_resolution: 路径点采样分辨率(米)
        """
        self._map = world_map
        self._sampling_resolution = sampling_resolution
        print("正在初始化路径规划器...")

    def plan_route(self, start_location: carla.Location,
                   end_location: carla.Location,
                   vehicle_transform: carla.Transform = None) -> List[Tuple[carla.Waypoint, RoadOption]]:
        """
        规划从起点到终点的路径

        Args:
            start_location: 起始位置
            end_location: 目标位置
            vehicle_transform: 车辆当前变换(包含方向信息)

        Returns:
            路径点列表，每个元素为(waypoint, road_option)元组
        """
        print(
            f"开始规划路径: 起点({start_location.x:.2f}, {start_location.y:.2f}) -> 终点({end_location.x:.2f}, {end_location.y:.2f})")

        # 获取起点和终点最近的waypoint
        start_waypoint = self._map.get_waypoint(start_location, project_to_road=True)
        end_waypoint = self._map.get_waypoint(end_location, project_to_road=True)

        if not start_waypoint:
            print("错误: 无法找到起点对应的waypoint")
            return []
        if not end_waypoint:
            print("错误: 无法找到终点对应的waypoint")
            return []

        print(f"找到起点waypoint: 车道ID={start_waypoint.lane_id}, 道路ID={start_waypoint.road_id}")
        print(f"找到终点waypoint: 车道ID={end_waypoint.lane_id}, 道路ID={end_waypoint.road_id}")

        # 确保起点waypoint与车辆前进方向一致
        if vehicle_transform:
            start_waypoint = self._get_forward_waypoint(start_waypoint, vehicle_transform)
            print(f"调整后起点waypoint: 车道ID={start_waypoint.lane_id}, 道路ID={start_waypoint.road_id}")

        # 使用简化的贪婪搜索算法
        route = self._greedy_search(start_waypoint, end_waypoint)

        if not route:
            print("警告: 贪婪搜索失败，尝试直接连接...")
            # 如果贪婪搜索失败，尝试简单的直接路径
            route = self._simple_direct_path(start_waypoint, end_waypoint)

        if not route:
            print("错误: 所有路径规划方法都失败了")
            return []

        print(f"路径规划成功! 找到 {len(route)} 个waypoint")

        # 生成详细的waypoint路径
        detailed_route = self._generate_detailed_route(route)

        return detailed_route

    def _get_forward_waypoint(self, waypoint: carla.Waypoint,
                              vehicle_transform: carla.Transform) -> carla.Waypoint:
        """
        获取与车辆前进方向一致的waypoint
        """
        vehicle_forward = vehicle_transform.get_forward_vector()
        wp_forward = waypoint.transform.get_forward_vector()

        # 计算方向一致性
        dot_product = (vehicle_forward.x * wp_forward.x +
                       vehicle_forward.y * wp_forward.y)

        print(f"车辆前进方向一致性检查: dot_product = {dot_product:.3f}")

        # 如果方向基本一致，直接返回
        if dot_product > 0.3:
            return waypoint

        # 如果方向不一致，尝试寻找相邻车道
        print("车辆方向与waypoint不一致，尝试寻找合适的车道...")

        # 检查左车道
        left_lane = waypoint.get_left_lane()
        if left_lane and left_lane.lane_type == carla.LaneType.Driving:
            left_forward = left_lane.transform.get_forward_vector()
            left_dot = (vehicle_forward.x * left_forward.x +
                        vehicle_forward.y * left_forward.y)
            if left_dot > 0.3:
                print("使用左车道")
                return left_lane

        # 检查右车道
        right_lane = waypoint.get_right_lane()
        if right_lane and right_lane.lane_type == carla.LaneType.Driving:
            right_forward = right_lane.transform.get_forward_vector()
            right_dot = (vehicle_forward.x * right_forward.x +
                         vehicle_forward.y * right_forward.y)
            if right_dot > 0.3:
                print("使用右车道")
                return right_lane

        # 如果都不合适，返回原waypoint
        print("警告: 未找到方向一致的车道，使用原waypoint")
        return waypoint

    def _greedy_search(self, start_wp: carla.Waypoint,
                       end_wp: carla.Waypoint) -> List[carla.Waypoint]:
        """
        使用贪婪搜索算法规划路径
        """
        print("使用贪婪搜索算法...")

        route = [start_wp]
        current_wp = start_wp
        visited = set()
        max_iterations = 1000
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # 检查是否到达目标
            distance_to_goal = self._calculate_distance(current_wp, end_wp)
            print(f"迭代 {iteration}: 距离目标 {distance_to_goal:.2f}m")

            if distance_to_goal < 10.0:  # 增加到达阈值
                print(f"到达目标! 最终距离: {distance_to_goal:.2f}m")
                break

            # 避免循环
            wp_key = (current_wp.road_id, current_wp.lane_id,
                      int(current_wp.s))  # 使用s坐标作为位置标识
            if wp_key in visited:
                print(f"检测到循环，跳出搜索。访问过的waypoint数量: {len(visited)}")
                break
            visited.add(wp_key)

            # 获取下一个可行的waypoint
            next_waypoints = current_wp.next(self._sampling_resolution)

            if not next_waypoints:
                print("没有找到下一个waypoint，搜索结束")
                break

            # 选择最接近目标的waypoint
            best_wp = None
            best_distance = float('inf')

            for next_wp in next_waypoints:
                # 检查waypoint是否有效
                if (next_wp.lane_type != carla.LaneType.Driving or
                        not self._is_valid_forward_direction(current_wp, next_wp)):
                    continue

                # 计算到目标的距离
                distance = self._calculate_distance(next_wp, end_wp)

                # 添加一些启发式因子
                # 惩罚过度转弯
                turn_penalty = abs(self._get_angle_difference(
                    current_wp.transform.rotation.yaw,
                    next_wp.transform.rotation.yaw)) * 0.1

                total_cost = distance + turn_penalty

                if total_cost < best_distance:
                    best_distance = total_cost
                    best_wp = next_wp

            if best_wp is None:
                print("没有找到有效的下一个waypoint")
                break

            route.append(best_wp)
            current_wp = best_wp

        if iteration >= max_iterations:
            print("警告: 达到最大迭代次数")

        return route

    def _simple_direct_path(self, start_wp: carla.Waypoint,
                            end_wp: carla.Waypoint) -> List[carla.Waypoint]:
        """
        创建简单的直接路径（备用方案）
        """
        print("尝试创建简单直接路径...")

        route = [start_wp]
        current_wp = start_wp

        # 最多尝试50步
        for step in range(50):
            distance_to_goal = self._calculate_distance(current_wp, end_wp)

            if distance_to_goal < 15.0:
                print(f"直接路径成功! 步数: {step}, 最终距离: {distance_to_goal:.2f}m")
                break

            # 获取下一个waypoint
            next_waypoints = current_wp.next(self._sampling_resolution)

            if not next_waypoints:
                break

            # 选择第一个有效的waypoint
            found_next = False
            for next_wp in next_waypoints:
                if next_wp.lane_type == carla.LaneType.Driving:
                    route.append(next_wp)
                    current_wp = next_wp
                    found_next = True
                    break

            if not found_next:
                break

        return route if len(route) > 1 else []

    def _calculate_distance(self, wp1: carla.Waypoint, wp2: carla.Waypoint) -> float:
        """计算两个waypoint之间的欧几里得距离"""
        loc1 = wp1.transform.location
        loc2 = wp2.transform.location
        return math.sqrt((loc1.x - loc2.x) ** 2 + (loc1.y - loc2.y) ** 2 + (loc1.z - loc2.z) ** 2)

    def _is_valid_forward_direction(self, current_wp: carla.Waypoint,
                                    next_wp: carla.Waypoint) -> bool:
        """
        检查下一个waypoint是否在有效的前进方向上
        """
        # 获取当前waypoint的前进方向
        current_forward = current_wp.transform.get_forward_vector()

        # 计算从当前位置到下一个位置的方向向量
        current_loc = current_wp.transform.location
        next_loc = next_wp.transform.location

        direction_x = next_loc.x - current_loc.x
        direction_y = next_loc.y - current_loc.y

        # 归一化方向向量
        direction_length = math.sqrt(direction_x ** 2 + direction_y ** 2)
        if direction_length < 0.1:  # 距离太近
            return True

        direction_x /= direction_length
        direction_y /= direction_length

        # 计算点积
        dot_product = current_forward.x * direction_x + current_forward.y * direction_y

        # 如果点积大于0.2，认为是有效的前进方向
        return dot_product > 0.2

    def _get_angle_difference(self, angle1: float, angle2: float) -> float:
        """计算两个角度之间的差值"""
        diff = angle2 - angle1
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        return abs(diff)

    def _generate_detailed_route(self, route: List[carla.Waypoint]) -> List[Tuple[carla.Waypoint, RoadOption]]:
        """
        生成详细的路径信息，包含道路选项
        """
        if not route:
            return []

        if len(route) == 1:
            return [(route[0], RoadOption.LANEFOLLOW)]

        detailed_route = []

        for i in range(len(route)):
            if i == 0:
                # 起始点
                road_option = RoadOption.LANEFOLLOW
            elif i == len(route) - 1:
                # 终点
                road_option = RoadOption.LANEFOLLOW
            else:
                # 中间点，根据前后waypoint判断行为
                road_option = self._determine_road_option(route[i - 1], route[i], route[i + 1])

            detailed_route.append((route[i], road_option))

        return detailed_route

    def _determine_road_option(self, prev_wp: carla.Waypoint,
                               current_wp: carla.Waypoint,
                               next_wp: carla.Waypoint) -> RoadOption:
        """
        根据waypoint序列确定道路选项
        """
        # 检查是否变道
        if prev_wp.lane_id != current_wp.lane_id:
            if prev_wp.lane_id > current_wp.lane_id:
                return RoadOption.CHANGELANELEFT
            else:
                return RoadOption.CHANGELANERIGHT

        # 检查转向角度
        prev_yaw = prev_wp.transform.rotation.yaw
        next_yaw = next_wp.transform.rotation.yaw

        angle_diff = self._get_angle_difference(prev_yaw, next_yaw)

        if angle_diff > 20:  # 降低转弯检测阈值
            # 判断左转还是右转
            yaw_diff = next_yaw - prev_yaw
            while yaw_diff > 180:
                yaw_diff -= 360
            while yaw_diff < -180:
                yaw_diff += 360

            if yaw_diff > 0:
                return RoadOption.LEFT
            else:
                return RoadOption.RIGHT
        elif angle_diff > 5:
            return RoadOption.STRAIGHT
        else:
            return RoadOption.LANEFOLLOW


# 使用示例
def example_usage():
    """使用示例"""
    try:
        # 连接到Carla服务器
        print("连接到Carla服务器...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
        world = client.get_world()
        world_map = world.get_map()

        print(f"成功连接到地图: {world_map.name}")

        # 创建路径规划器
        planner = GlobalPathPlanner(world_map, sampling_resolution=3.0)

        # 获取一些spawn点作为测试位置
        spawn_points = world_map.get_spawn_points()
        if len(spawn_points) < 2:
            print("错误: 地图中spawn点不足")
            return

        # 使用spawn点作为起点和终点
        start_transform = spawn_points[0]
        end_transform = spawn_points[min(5, len(spawn_points) - 1)]  # 选择距离适中的终点

        start_location = start_transform.location
        end_location = end_transform.location

        print(f"使用spawn点作为测试:")
        print(f"起点: ({start_location.x:.2f}, {start_location.y:.2f}, {start_location.z:.2f})")
        print(f"终点: ({end_location.x:.2f}, {end_location.y:.2f}, {end_location.z:.2f})")
        print(
            f"直线距离: {math.sqrt((end_location.x - start_location.x) ** 2 + (end_location.y - start_location.y) ** 2):.2f}m")

        # 规划路径
        route = planner.plan_route(start_location, end_location, start_transform)

        if route:
            print(f"\n路径规划成功！路径包含 {len(route)} 个waypoint")
            print("路径详情:")
            for i, (waypoint, road_option) in enumerate(route[:10]):  # 只显示前10个点
                loc = waypoint.transform.location
                print(f"  {i:2d}: ({loc.x:7.2f}, {loc.y:7.2f}) - 车道{waypoint.lane_id:2d} - {road_option.name}")

            if len(route) > 10:
                print(f"  ... 还有 {len(route) - 10} 个waypoint")

            # 显示最后几个点
            if len(route) > 10:
                print("最后几个waypoint:")
                for i, (waypoint, road_option) in enumerate(route[-3:], len(route) - 3):
                    loc = waypoint.transform.location
                    print(f"  {i:2d}: ({loc.x:7.2f}, {loc.y:7.2f}) - 车道{waypoint.lane_id:2d} - {road_option.name}")

        else:
            print("路径规划失败!")

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    example_usage()