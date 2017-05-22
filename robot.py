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
        self.dead_end = False
        self.Q = dict()
        self.trace_list = []
        self.trace_back = False
        self.rotate = False
        self.trace_back_step = 1
        self.move = 0
        self.train_deadline = 5 * self.maze_dim / 2
        self.initial_location_pos_move = dict()
        logging.basicConfig(filename='test.log', filemode='w', level=logging.DEBUG)  ###

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

    def update_location(self, movement, heading):
        '''
        Update robot location based on the movement and heading chosen
        '''

        # perform movement
        self.pre_location = self.location
        self.pre_heading = self.heading
        # keep heading when chose to step back, otherwise change heading
        if dir_reverse[self.heading] != heading:
            self.heading = heading
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
        if self.trace_back_step == 1:
            rotation = 0
        else:
            rotation = 0 - self.trace_list[1 - self.trace_back_step][0]
        movement = 0 - self.trace_list[-self.trace_back_step][1]
        self.trace_back_step += 1
        # perform rotation
        if rotation == -90:
            heading = dir_sensors[self.heading][0]
        elif rotation == 90:
            heading = dir_sensors[self.heading][2]
        else:
            heading = self.heading
        logging.info("TRACE STEP" + str(self.trace_back_step))
        return movement, heading


    @staticmethod
    def more_pos(sensors):
        '''
        Has more than one possible direction
        :param sensors:
        :return:
        '''
        pos_mov = 0
        for pos in sensors:
            if pos != 0:
                pos_mov += 1
        if pos_mov > 1:
            return True
        else:
            return False

    def next_pos_move(self, sensors):
        '''
        Use this function to determine the possible next move
        :return:
        '''
        dir_possible = dict()
        # not dead end
        if not self.dead_end:
            # has possible movement
            if sum(sensors) != 0:
                for idx, possible_heading in enumerate(dir_sensors[self.heading]):
                    wall_distance = sensors[idx]
                    if wall_distance != 0:
                        logging.info("+++pos h:" + possible_heading + "|||wal d:" + str(wall_distance))
                        dir_possible[possible_heading] = wall_distance
            # no possible movement, dead end
            elif sum(sensors) == 0:
                self.dead_end = True
                dir_possible[self.heading] = 0 - self.trace_list[-1][1]
                del self.trace_list[-1]
        # dead end
        else:
            # has more than one possible movement
            if self.more_pos(sensors):
                self.dead_end = False
                for idx, possible_heading in enumerate(dir_sensors[self.heading]):
                    wall_distance = sensors[idx]
                    if wall_distance != 0:
                        logging.info("+++pos h:" + possible_heading + "|||wal d:" + str(wall_distance))
                        dir_possible[possible_heading] = wall_distance
            else:
                logging.info("keep step back" + str(self.trace_list[-1]))
                dir_possible[self.heading] = 0 - self.trace_list[-1][1]
                del self.trace_list[-1]

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
        self.trace_back = False
        self.trace_back_step = 1
        self.trace_list = []

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

        ####################################
        # check if the robot is in a special location
        ####################################
        goal_bounds = [self.maze_dim/2 - 1, self.maze_dim/2]
        # check for goal entered
        logging.info("cur loc " + str(self.location) + ",head " + str(self.heading) + ",track back " + str(self.trace_back))
        if self.location[0] in goal_bounds and self.location[1] in goal_bounds:
        # if self.location[0] in goal_bounds and self.location[1] in goal_bounds:
            logging.info("GOAL")
            print "GOAL"
            self.trace_back = True
        if self.move > self.train_deadline:
            logging.info("DEADLINE")
            print "DEADLINE"
            self.trace_back = True

        # reset traceback after the robot back to the initial location
        if self.location == [0, 0] and self.move > 0:

            if "u" not in self.heading:
                self.rotate = True
                movement = 0
                if "d" in self.heading:
                    heading = 'l'
                elif "r" in self.heading or "l" in self.heading:
                    heading = 'u'
                movement, rotation = self.decide_move_n_rotation(heading, movement)
                self.move += 1

            else:
                self.rotate = False
                self.move = 0
            self.reset_traceback()

        if not self.rotate:
            if not self.trace_back:
                # get distance to every direction wall
                # collect possible heading and movement
                dir_possible = self.next_pos_move(sensors)
                # random select heading and movement
                random_m = True
                if random_m:
                    heading, movement = self.random_move(dir_possible)

                # set movement and rotation
                movement, rotation = self.decide_move_n_rotation(heading, movement)
                if not self.dead_end:
                    self.trace_list += [(rotation, movement)]
                self.move += 1

            # traceback
            else:
                movement, heading = self.traceback_move()
                movement, rotation = self.decide_move_n_rotation(heading, movement)

        # update location and heading
        self.update_location(movement, heading)
        logging.info(str(self.move) + "next loc" + str(self.location) + "end step")  ###
        logging.info(str(self.trace_list))

        return rotation, movement
