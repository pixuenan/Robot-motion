import numpy as np
# global dictionaries for robot movement and sensing
dir_sensors = {'u': ['l', 'u', 'r'], 'r': ['u', 'r', 'd'],
               'd': ['r', 'd', 'l'], 'l': ['d', 'l', 'u'],
               'up': ['l', 'u', 'r'], 'right': ['u', 'r', 'd'],
               'down': ['r', 'd', 'l'], 'left': ['d', 'l', 'u']}
dir_move = {'u': [0, 1], 'r': [1, 0], 'd': [0, -1], 'l': [-1, 0],
            'up': [0, 1], 'right': [1, 0], 'down': [0, -1], 'left': [-1, 0]}
dir_reverse = {'u': 'd', 'r': 'l', 'd': 'u', 'l': 'r',
               'up': 'd', 'right': 'l', 'down': 'u', 'left': 'r'}
dir_rotation_heading = {90: [('l', 'u'), ('r', 'd'), ('u', 'r'), ('d', 'l'),
                             ('left', 'u'), ('right', 'd'), ('up', 'r'), ('down', 'l')],
                        -90: [('l', 'd'), ('r', 'u'), ('u', 'l'), ('d', 'r'),
                              ('left', 'd'), ('right', 'u'), ('up', 'l'), ('down', 'r')]}

class Robot(object):
    def __init__(self, maze_dim):
        '''
        Use the initialization function to set up attributes that your robot
        will use to learn and navigate the maze. Some initial attributes are
        provided based on common information, including the size of the maze
        the robot is placed in.
        '''
        self.location = [0, 0]
        self.heading = 'up'
        self.maze_dim = maze_dim
        self.goal_loc = [[self.maze_dim/2-1, self.maze_dim/2-1],
                         [self.maze_dim/2, self.maze_dim/2-1],
                         [self.maze_dim/2-1, self.maze_dim/2],
                         [self.maze_dim/2, self.maze_dim/2]]

        self.possible_action = dict()
        self.Q_dict = dict()
        self.move = 0
        self.build_Q_dict()
        self.epsilon = 0.1

    def build_Q_dict(self):
        '''
        Build the dictionary of the score for every location in the maze in terms of next movement and heading
        :return: {[row, column]: {[next_heading, next_movement]: score}}
        '''
        for row in range(self.maze_dim):
            for column in range(self.maze_dim):
                default_score = dict()
                for heading in ['u', 'l', 'r', 'd']:
                    for move in [1, 2, 3]:
                        next_loc = self.update_location(dir_sensors[heading][0], [row, column], move, heading)[1]
                        goal_dist = abs(next_loc[0] - ((self.maze_dim - 1) / 2)) + abs(
                            next_loc[1] - ((self.maze_dim - 1) / 2))
                        dist_reward = self.maze_dim - goal_dist
                        default_score[(heading, move)] = dist_reward
                self.Q_dict[(row, column)] = default_score.copy()

    @staticmethod
    def update_location(cur_heading, cur_location, movement, heading):
        '''
        Update robot location based on the movement and heading chosen
        '''
        # perform movement
        # keep heading when chose to step back, otherwise change heading
        if dir_reverse[cur_heading] != heading:
            cur_heading = heading
        if abs(movement) > 3:
            print "Movement limited to three squares in a turn."
        movement = max(min(int(movement), 3), -3)  # fix to range [-3, 3]
        while movement:
            if movement > 0:
                cur_location[0] += dir_move[cur_heading][0]
                cur_location[1] += dir_move[cur_heading][1]
                movement -= 1
            else:
                rev_heading = dir_reverse[cur_heading]
                cur_location[0] += dir_move[rev_heading][0]
                cur_location[1] += dir_move[rev_heading][1]
                movement += 1

        return cur_heading, cur_location

    def choose_action(self):
        if np.random.binomial(1, self.epsilon) == 1:
            heading = np.random.choice(self.possible_action.keys())
            movement = np.random.choice(range(self.possible_action[heading]))
        else:
            action_dict = self.Q_dict[tuple(self.location)].copy()
            for key, value in action_dict.items():
                if key[0] not in self.possible_action.keys() or key[1] > self.possible_action[key[0]]:
                    del action_dict[key]
            max_Q_action = []
            max_Q_value = max(action_dict.values())
            for key, value in action_dict.items():
                if value == max_Q_value:
                    max_Q_action += [key]
            index = np.random.choice(range(len(max_Q_action)))
            heading, movement = max_Q_action[index]
        return heading, movement

    def next_pos_move(self, sensors):
        '''
        Use this function to determine the possible next move
        :return:
        '''
        self.possible_action = dict()
        for idx, possible_heading in enumerate(dir_sensors[self.heading]):
            wall_distance = sensors[idx]
            if wall_distance != 0:
                self.possible_action[possible_heading] = min(wall_distance, 3)

    def decide_move_n_rotation(self, heading, movement):
        '''
        Decide movement and rotation
        :return:
        '''
        if dir_reverse[self.heading] == heading:
            movement = -movement
        if (self.heading, heading) in dir_rotation_heading[90]:
            rotation = 90
        elif (self.heading, heading) in dir_rotation_heading[-90]:
            rotation = -90
        else:
            rotation = 0

        return movement, rotation

    def next_move(self, sensors):
        '''
        Use this function to determine the next move the robot should make,
        based on the input from the sensors after its previous move. Sensor
        inputs are a list of three distances from the robot's left, front, and
        right-facing sensors, in that order.

        Outputs should be a tuple of two values. The first value indicates
        robot rotation (if any), as a number: 0 for no rotation, +90 for a
        90-degree rotation clockwise, and -90 for a 90-degree rotation
        counterclockwise. Other values will result in no rotation. The second
        value indicates robot movement, and the robot will attempt to move the
        number of indicated squares: a positive number indicates forwards
        movement, while a negative number indicates backwards movement. The
        robot may move a maximum of three units per turn. Any excess movement
        is ignored.

        If the robot wants to end a run (e.g. during the first training run in
        the maze) then returing the tuple ('Reset', 'Reset') will indicate to
        the tester to end the run and return the robot to the start.
        '''
        print self.location, self.heading, sensors
        self.next_pos_move(sensors)
        if not self.location in self.goal_loc:
            if sensors.count(0) == 3:
                heading, movement = dir_reverse[self.heading], 1
            else:
                heading, movement = self.choose_action()
            movement, rotation = self.decide_move_n_rotation(heading, movement)
            self.heading, self.location = self.update_location(self.heading, self.location, movement, heading)
            print rotation, movement
            return rotation, movement
        else:
            return "Reset", "Reset"
