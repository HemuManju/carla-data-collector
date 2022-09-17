def process_waypoints(waypoints, current_position):
    processed_waypoints = []
    for waypoint in waypoints:
        try:
            processed_waypoints.append(
                [
                    waypoint.transform.location.x,
                    waypoint.transform.location.y,
                    waypoint.transform.rotation.yaw,
                ]
            )
        except AttributeError:
            processed_waypoints.append(
                [
                    current_position.location.x,
                    current_position.location.y,
                    current_position.rotation.yaw,
                ]
            )

    return processed_waypoints

