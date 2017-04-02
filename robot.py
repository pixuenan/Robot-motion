import numpy as np
import random
from maze import Maze

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

        self.max_time = 1000
        self.Q = dict()

    def update_location(self, movement):
        '''
        Update robot location based on the movement and rotation
        '''

        # perform movement
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
        # in the first run
        # create a list to record all of the movements to trace back to the start position
        # empty this list when restart

        # get distance to every direction wall
        # collect possible heading and movement
        dir_possible = dict()
        if sum(sensors) != 0:
            for idx, possible_heading in enumerate(dir_sensors[self.heading]):
                wall_distance = sensors[idx]
                # print "+++", possible_heading, wall_distance
                if wall_distance != 0:
                    dir_possible[possible_heading] = wall_distance
        # dead end, move back 1 step
        else:
            dir_possible[dir_reverse[self.heading]] = 1

        # random select heading and movement
        if len(dir_possible.keys()) > 1:
            heading = random.choice(dir_possible.keys())
            movement = random.choice(range(1, min(dir_possible[heading], 3)+1))
        else:
            heading = dir_possible.keys()[0]
            movement = min(dir_possible[heading], 3)

        if dir_reverse[self.heading] == heading:
            movement = -movement
        if (self.heading, heading) in dir_rotation_heading[90]:
            rotation = 90
        elif (self.heading, heading) in dir_rotation_heading[-90]:
            rotation = -90
        else:
            rotation = 0
        print self.heading, heading, rotation, movement

        # update location and heading
        if dir_reverse[self.heading] != heading:
            self.heading = heading
        self.update_location(movement)
        print self.location

        return rotation, movement
