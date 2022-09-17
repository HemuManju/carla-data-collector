class PreProcessData:
    def __init__(self, config):
        self.cfg = config
        self.n_collision = 0

    def preprocess_sensor_data(self, sensor_data):
        # Check if there is a collision
        if 'collision' in sensor_data.keys():
            self.n_collision += 1

    def process(
        self,
        sensor_data,
        waypoint_data,
        traffic_data,
        collision_data,
        vehicle_data=-1,
        other_data={'other_data': -1},
    ):
        """Function to do all the pre processing of observations (sensor data).

        :param sensor_data: dictionary {sensor_name: sensor_data}

        Should return a tuple or list with two items, the processed observations,
        as well as a variable with additional information about such observation.
        The information variable can be empty
        """
        # Check if collision or obstacle has happend/detected
        for event_sensor in ['obstacle', 'collision']:
            if event_sensor in sensor_data.keys():
                sensor_data[event_sensor] = True
            else:
                sensor_data[event_sensor] = False

        # Preprocess sensor data
        data = {
            **sensor_data,
            **waypoint_data,
            **traffic_data,
            **vehicle_data,
            **collision_data,
            **other_data,
        }
        return data
