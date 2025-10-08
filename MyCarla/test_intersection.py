# import carla
#
# client = carla.Client('localhost', 2000)
# client.set_timeout(10.0)
#
# world = client.get_world()
# map = world.get_map()
#
# # 生成全地图 waypoint
# #waypoints = map.generate_waypoints(distance=2.0)  # 每隔 2 米生成 waypoint
#
# spawn_points = map.get_spawn_points()
# junction_dict = {}
# # 找到所有属于 junction 的 waypoint
# junction_wps = [wp for wp in spawn_points if wp.is_junction]
# for junction_wp in junction_wps:
#     #print(junction_wp.junction_id)
#     if junction_wp.junction_id == 189:
#         world.debug.draw_point(junction_wp.transform.location,size=0.5, color=carla.Color(r=255, g=0, b=0), life_time=60.0)
#         world.debug.draw_string(
#             junction_wp.transform.location + carla.Location(z=0.5),  # 稍微抬高，避免文字和地面重叠
#             f"({junction_wp.transform.location.x}, {junction_wp.transform.location.y})",
#             draw_shadow=True,
#             color=carla.Color(r=0, g=255, b=0),
#             life_time=60.0
#         )
# if junction_wp.road_id not in junction_dict:
#    junction_dict[junction_wp.road_id] = junction_wp.transform.location
# world.debug.draw_point(junction_wp.transform.location,size=0.5, color=carla.Color(r=255, g=0, b=0), life_time=60.0)


# 收集所有 unique road_id + junction_waypoint
# junction_dict = {}
# for wp in junction_wps:
#     if wp.road_id not in junction_dict:
#         junction_dict[wp.road_id] = wp.transform.location
#
# # 打印十字路口的 ID 和位置
# for road_id, loc in junction_dict.items():
#     print(f"Junction road_id: {road_id}, Location: {loc}")
#
# for loc in junction_dict.values():
#     world.debug.draw_point(loc, size=0.5, color=carla.Color(r=255, g=0, b=0), life_time=60.0)
#     world.debug.draw_string(
#         loc + carla.Location(z=0.5),  # 稍微抬高，避免文字和地面重叠
#         f"({loc.x}, {loc.y})",
#         draw_shadow=True,
#         color=carla.Color(r=0, g=255, b=0),
#         life_time=60.0
# )
# nearest_wp = map.get_waypoint(
#     carla.Location(x=-45.811, y=-1.704, z=0),
#     project_to_road=True,
#     lane_type=carla.LaneType.Driving
# )
# world.debug.draw_point(
#     nearest_wp.transform.location,
#     size=0.3,
#     color=carla.Color(r=0, g=255, b=0),
#     life_time=20.0
# )
#
# world.debug.draw_string(
#     nearest_wp.transform.location + carla.Location(z=0.5),
#     f"({nearest_wp.transform.location.x}, {nearest_wp.transform.location.y})",
#     color=carla.Color(r=255, g=255, b=0),
#     life_time=20.0
# )
import carla
import math

def distance(loc1, loc2):
    dx = loc1.x - loc2.x
    dy = loc1.y - loc2.y
    dz = loc1.z - loc2.z
    return math.sqrt(dx*dx + dy*dy + dz*dz)

# 连接 CARLA
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()
map = world.get_map()

# 获取所有 spawn point
spawn_points = map.get_spawn_points()

# 假设你知道路口中心位置 (替换成你实际的坐标)
junction_location = carla.Location(x=-45.248, y=20.426, z=0)

# 筛选距离 < 30 米 的 spawn point
nearby_spawn_points = [sp for sp in spawn_points
                       if distance(sp.location, junction_location) < 30.0]

print(f"找到 {len(nearby_spawn_points)} 个 spawn point 在路口附近")

# 可视化
for sp in nearby_spawn_points:
    world.debug.draw_point(
        sp.location,
        size=0.3,
        color=carla.Color(r=0, g=255, b=0),
        life_time=30.0
    )
    world.debug.draw_string(
        sp.location + carla.Location(z=0.5),
        f"{sp.location.x}, {sp.location.y}",
        color=carla.Color(r=255, g=255, b=0),
        life_time=30.0
    )
