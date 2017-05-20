import numpy as np
import random
import logging

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
        self.pre_location = self.location
        self.pre_heading = self.heading
        self.maze_dim = maze_dim

        self.max_time = 1000
        self.Q = dict()
        self.trace_list = []
        self.trace_back = False
        self.trace_back_step = 0
        self.move = 0
        self.train_deadline = 5 * self.maze_dim / 2
        logging.basicConfig(filename='test.log', level=logging.DEBUG)  ###

    def build_Q_dict(self):
        '''
        Build the dictionary of the score for every location in the maze in terms of next movement and heading
        :return: {[row, column]: {[next_heading, next_movement]: score}}
        '''
        for row in range(self.maze_dim):
            for column in range(self.maze_dim):
                default_score = dict()
                for direction in ['u', 'l', 'r', 'd']:
                    for move in [-3, -2, -1, 1, 2, 3]:
                        default_score[[direction, move]] = 0
                self.Q_dict[[row, column]] = default_score

    def update_location(self, movement):
        '''
        Update robot location based on the movement and rotation
        '''

        # perform movement
        self.pre_location = self.location
        self.pre_heading = self.heading
        if abs(movement) > 3:
            print "Movement limited to three squares in a turn."
        movement = max(min(int(movement), 3), -3) # fix to range [-3, 3]
        while movement:
            if movement > 0:
                self.location[0] += dir_move[self.heading][0]
                self.location[1] += dir_move[self.heading][1]
                movement -= 1
            else:
                rev_heading = dir_reverse[self.heading]
                self.location[0] += dir_move[rev_heading][0]
                self.location[1] += dir_move[rev_heading][1]
                movement += 1

    def traceback_move(self):
        '''
        Traceback to the initial location after enter the goal zone.
        :return:
        '''
        if self.trace_back_step == 0:
            rotation = 0
        else:
            rotation = self.trace_list[-1 - self.trace_back_step][0]
        movement = 0 - self.trace_list[-1][1]
        self.trace_back_step += 1
        # perform rotation
        if rotation == -90:
            heading = dir_sensors[self.heading][0]
        elif rotation == 90:
            heading = dir_sensors[self.heading][2]
        else:
            heading = self.heading
        return movement, heading

    def next_pos_move(self, sensors):
        '''
        Use this function to determine the possible next move
        :return:
        '''
        dir_possible = dict()
        if sum(sensors) != 0:
            for idx, possible_heading in enumerate(dir_sensors[self.heading]):
                wall_distance = sensors[idx]
                logging.info("+++pos h:" + possible_heading + ":wal d:" + str(wall_distance))
                if wall_distance != 0:
                    dir_possible[possible_heading] = wall_distance
        # dead end, move back 1 step
        else:
            dir_possible[dir_reverse[self.heading]] = 1
        return dir_possible

    @staticmethod
    def random_move(dir_possible):
        # random select heading and movement
        if len(dir_possible.keys()) > 1:
            heading = random.choice(dir_possible.keys())
            movement = random.choice(range(1, min(dir_possible[heading], 3) + 1))
        else:
            heading = dir_possible.keys()[0]
            movement = min(dir_possible[heading], 3)
        return heading, movement

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
        logging.info("cur h:" + self.heading + " nex h:" + heading + " rot:" + str(rotation) + " nex m:" + str(movement))  ###
        return movement, rotation

    def reset_traceback(self):
        '''
        Reset variables after traceback
        :return:
        '''
        self.heading = 'up'
        self.trace_back = False
        self.trace_back_step = 0
        self.trace_list = []
        self.move = 0

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
        # select random direction in available directions

        ####################################
        # check if the robot is in a special location
        ####################################
        goal_bounds = [self.maze_dim/2 - 1, self.maze_dim/2]
        # check for goal entered
        if self.location[0] in goal_bounds and self.location[1] in goal_bounds or self.move > self.train_deadline:
            self.trace_back = True

        # create a list to record all of the movements to trace back to the start position
        # empty this list when restart
        # reset traceback after the robot back to the initial location
        if self.location == [0, 0] and self.trace_back:
            self.reset_traceback()

        if not self.trace_back:
            # get distance to every direction wall
            # collect possible heading and movement
            dir_possible = self.next_pos_move(sensors)
            # random select heading and movement
            random_m = True
            if random_m:
                heading, movement = self.random_move(dir_possible)

            self.heading = heading
            # set movement and rotation
            movement, rotation = self.decide_move_n_rotation(heading, movement)
            self.trace_list += [(rotation, movement)]
            self.move += 1

        # traceback
        else:
            movement, heading = self.traceback_move()

        # print self.heading, heading, rotation, movement
        # update location and heading
        if dir_reverse[self.heading] != heading:
            self.heading = heading
        self.update_location(movement)
        logging.info(self.location)  ###

        return rotation, movement
