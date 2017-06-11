import numpy as np
import math
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

class TraceBack(object):
    def __init__(self):
        self.trace_list = []
        self.trace_back = False
        self.trace_back_step = 1
        self.rotate_degree = "0"
        self.dead_end = False
        self.dead_end_back_step = 1
        self.next_rotation = 0

    def update_list(self, location, heading, num_pos, rotation, next_heading, next_movement):
        self.trace_list += [(location, heading, num_pos, rotation, next_heading, next_movement)]

    def reset_dead_end_traceback(self):
        self.rotate_degree = "0"
        self.dead_end = False
        self.dead_end_back_step = 1

    def dead_end_trace_back(self):
        # last step of trace back
        if self.rotate_degree != "0":
            rotation, movement = self.rotate_degree, 0
            self.reset_dead_end_traceback()
        else:
            if self.trace_list[-1][2] > 1:
                self.rotate_degree = 0 - self.trace_list[-1][-3]
            rotation = self.trace_rotation(self.dead_end_back_step)
            movement = self.trace_movement()
            del self.trace_list[-1]
            self.dead_end_back_step += 1

        return rotation, movement

    def trace_movement(self):
        return 0 - self.trace_list[-1][-1]

    def trace_rotation(self, step):
        if step == 1:
            rotation = 0
        else:
            rotation = self.next_rotation
        self.next_rotation = 0 - self.trace_list[-1][-3]
        return rotation

    def trace_back_move(self):
        rotation = self.trace_rotation(self.trace_back_step)
        movement = self.trace_movement()
        del self.trace_list[-1]
        self.trace_back_step += 1
        return rotation, movement

    def last_multiple_pos(self):
        '''
        :return: the reverse index of the last step that has multiple possible direction
        '''
        index = 1
        while self.trace_list[-index][2] < 2:
            index += 1
        return index

    def reset_traceback(self):
        '''
        Reset variables after traceback
        :return:
        '''
        self.trace_back = False
        self.trace_back_step = 1
        self.trace_list = []


class Score(object):
    def __init__(self):
        self.reward = 0
        self.penalty = 0

    def deadline_penalty(self, deadline, move):
        fnc = move * 1.0 / (move + deadline)
        # print "fnc", fnc
        gradient = 10
        self.penalty = (math.pow(gradient, fnc) - 1) / (gradient - 1)
        # print "penalty", self.penalty

    def get_score(self, deadline, move, dead_end=False, repeat=False):
        self.reward = 2 * random.random() - 1
        if dead_end:
            self.reward -= 10
        elif repeat:
            self.reward -= 5
        # self.deadline_penalty(deadline, move)
        # self.reward += 2 - self.penalty
        return self.reward


class Robot(TraceBack, Score):
    def __init__(self, maze_dim):
        '''
        Use the initialization function to set up attributes that your robot
        will use to learn and navigate the maze. Some initial attributes are
        provided based on common information, including the size of the maze
        the robot is placed in.
        '''
        TraceBack.__init__(self)
        Score.__init__(self)
        self.location = [0, 0]
        self.heading = 'up'
        self.maze_dim = maze_dim
        self.goal_loc = [[self.maze_dim/2-1, self.maze_dim/2-1],
                         [self.maze_dim/2, self.maze_dim/2-1],
                         [self.maze_dim/2-1, self.maze_dim/2],
                         [self.maze_dim/2, self.maze_dim/2]]
        self.alpha = 0.8
        self.test = 0
        self.step = 0
        self.epsilon = 0.5

        self.max_time = 1000
        self.Q_dict = dict()
        self.rotate = False
        self.move = 0
        self.train_deadline = 30 * self.maze_dim / 2
        self.initial_location_pos_move = dict()
        logging.basicConfig(filename='test.log', filemode='w', level=logging.DEBUG)  ###
        self.build_Q_dict()


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
                        goal_dist = abs(next_loc[0]-((self.maze_dim-1)/2)) + abs(next_loc[1]-((self.maze_dim-1)/2))
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
        movement = max(min(int(movement), 3), -3) # fix to range [-3, 3]
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

    def update_Q_dict(self, dir_possible, dead_end=False, repeat=False, goal=False):
        '''
        Update Q dictionary for the previous action
        :return:
        '''
        self.remove_action(dir_possible)
        location = tuple(self.trace_list[-1][0])
        action = tuple([self.trace_list[-1][-2], abs(self.trace_list[-1][-1])])
        if dead_end:
            reward = self.get_score(self.train_deadline, self.move-1, dead_end=True, repeat=False)
        elif repeat:
            reward = self.get_score(self.train_deadline, self.move-1, dead_end=False, repeat=True)
        else:
            reward = self.get_score(self.train_deadline, self.move-1, dead_end, repeat)
        goal_dist = abs(self.location[0]-((self.maze_dim-1)/2)) + abs(self.location[1]-((self.maze_dim-1)/2))
        dist_reward = self.maze_dim - goal_dist
        reward += dist_reward
        if goal:
            reward += 30
        # if self.closer(self.location, location):
        #     reward += 10
        original_Qvaule = self.Q_dict[location][action]
        max_cur_Qvalue = self.Q_dict[tuple(self.location)].copy().values() and max(self.Q_dict[tuple(self.location)].copy().values()) or 0
        # Qvalue = original_Qvaule + self.alpha * (reward - original_Qvaule)
        Qvalue = original_Qvaule + self.alpha * (reward + 0.5 * max_cur_Qvalue - original_Qvaule)
        # print '+++', Qvalue
        self.Q_dict[location][action] = Qvalue

    def remove_action(self, dir_possible):
        '''
        Remove actions that is possible in the Q_dict
        :return:
        '''
        action_dict = self.Q_dict[tuple(self.location)]#.copy()
        for key, value in action_dict.items():
            if key[0] != dir_reverse[self.heading]:
                if key[0] not in dir_possible.keys() or key[1] > dir_possible[key[0]]:
                    del self.Q_dict[tuple(self.location)][key]

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
        for idx, possible_heading in enumerate(dir_sensors[self.heading]):
            wall_distance = sensors[idx]
            if wall_distance != 0:
                logging.info("+++pos h:" + possible_heading + "|||wal d:" + str(wall_distance))
                dir_possible[possible_heading] = min(wall_distance, 3)
        return dir_possible

    @staticmethod
    def random_move(dir_possible):
        # random select heading and movement
        if len(dir_possible.keys()) > 1:
            heading = random.choice(dir_possible.keys())
            movement = random.choice(range(1, min(dir_possible[heading], 3) + 1))
        else:
            heading = dir_possible.keys()[0]
            movement = random.choice(range(1, min(dir_possible[heading], 3) + 1))
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

    def rotation_to_heading(self, rotation):
        if rotation == 0:
            heading = self.heading
        else:
            cur_heading_list, next_heading_list = zip(*dir_rotation_heading[rotation])
            heading = next_heading_list[cur_heading_list.index(self.heading)]
        return heading

    def closer(self, location1, location2):
        '''
        Check if location1 is closer or equal to the destination than location2
        :return: True or False
        '''
        goal_location = [(self.maze_dim + 1) * 1.0 / 2, (self.maze_dim + 1) * 1.0 / 2]
        loc1_dist = abs(location1[0] - goal_location[0]) + abs(location1[1] - goal_location[1])
        loc2_dist = abs(location2[0] - goal_location[0]) + abs(location2[1] - goal_location[1])
        if loc1_dist <= loc2_dist:
            return True
        else:
            return False

    def act(self, dir_possible):
        action_dict = self.Q_dict[tuple(self.location)].copy()
        for key, value in action_dict.items():
            if key[0] not in dir_possible.keys() or key[1] > dir_possible[key[0]]:
                del action_dict[key]
        max_Q_action = []
        max_Q_value = max(action_dict.values())
        logging.info("action+ " + str(action_dict))
        for key, value in action_dict.items():
            if value == max_Q_value:
                max_Q_action += [key]
        heading, movement = random.choice(max_Q_action)
        return heading, movement

    def into_goal(self, pre_location_list, dir_possible):
        '''
        Choose the movement and rotation that the next location is not be visited yet.
        And force the robot to enter the goal zone
        :param pre_location_list:
        :param dir_possible:
        :return:
        '''
        if len(dir_possible) > 1 or dir_possible.values()[0] > 1:
            print dir_possible
            next_loc_dict = dict()
            # record dead end location
            # get all the next loactions from dir_possible
            # only the movement not in the pre_location_list, dead_end_location_list will be included
            for heading, movement in dir_possible.items():
                for i in range(1, movement+1):
                    next_loc = self.update_location(dir_sensors[heading][0], self.location, movement, heading)[1]
                    # if goal zone is in the next locations, choose that action
                    if next_loc in self.goal_loc:
                        print "***Goal"
                        return heading, movement
                    goal_dist = abs(next_loc[0]-((self.maze_dim-1)/2)) + abs(next_loc[1]-((self.maze_dim-1)/2))
                    if not next_loc in pre_location_list:
                        next_loc_dict[(heading, movement)] = [next_loc, goal_dist]
            # elif choose the movement not in the pre_location_list, dead_end_location_list and closer to the goal_zone
            if next_loc_dict:
                min_dist = min(zip(*next_loc_dict.values())[1])
                for action, result in next_loc_dict.items():
                    if result[1] == min_dist:
                        print "***Closet", action
                        return action
            else:
                # else random choose one action
                print "***Random"
                return self.random_move(dir_possible)
        else:
            print "***Only option"
            return dir_possible.items()[0]

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
        # self.epsilon = 0.5*math.cos(math.pi*self.step/4000)
        print "###", self.location
        dir_possible = self.next_pos_move(sensors)

        # rest_step = self.train_deadline - self.move
        # check for goal entered
        if self.location in self.goal_loc:
        # if self.location[0] in goal_bounds and self.location[1] in goal_bounds:
            logging.info("GOAL")
            print "GOAL"
            print self.step
            self.trace_back = True
            self.test += 1
            print "Test %d" % self.test
            self.move = 0
            self.update_Q_dict(dir_possible, goal=True)
        #
        # if self.move >= self.train_deadline:
        #     logging.info("DEADLINE")
        #     print "DEADLINE"
        #     self.test += 1
        #     print "Test %d" % self.test
        #     self.trace_back = True
        #     self.move = 0
        logging.info("cur loc " + str(self.location) + ",head " + str(self.heading) + ",trace back " + str(self.trace_back))
        logging.info("move " + str(self.move))

        # rotate the robot to the original direction after traceback
        if self.location == [0, 0]:
            #
            if "u" not in self.heading:
                logging.info("ROTATING")
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
        #
        if not self.rotate:
            # if self.trace_back:
            #     rotation, movement = self.trace_back_move()
            #     heading = self.rotation_to_heading(rotation)
            #     logging.info("TRACE STEP: %d, %d, %d" % (self.trace_back_step - 1, rotation, movement))
        #
            # else:
                # print '---', self.location
                if sum(sensors) == 0:
                    self.dead_end = True
                    self.update_Q_dict(dir_possible, dead_end=True)

                if self.dead_end:
                    rotation, movement = self.dead_end_trace_back()
                    heading = self.rotation_to_heading(rotation)
                    logging.info("DEAD END TRACE STEP: %d" % self.dead_end_back_step)
                    logging.info("DEAD END TRACE BACK: %d, %d" % (rotation, movement))
                # start dead end trace back

                else:
                    pre_location_list = []
                    if self.location != [0, 0]:
                        pre_location_list = zip(*self.trace_list)[0]
                        if self.location in pre_location_list:
                            self.update_Q_dict(dir_possible, repeat=True)
                        else:
                            self.update_Q_dict(dir_possible)
                    else:
                        self.remove_action(dir_possible)

                    # collect possible heading and movement
                    logging.info(sensors)
                    # random select heading and movement
                    # print self.test, self.train
                    # if self.epsilon > random.random():
                    #     heading, movement = self.random_move(dir_possible)
                    # else:
                        # print "act"
                        # heading, movement = self.act(dir_possible)
                    # set movement and rotation
                    # decide heading and movement when force the robot to enter the goal zone
                    heading, movement = self.into_goal(pre_location_list, dir_possible)
                    print "++++++", heading, movement
                    movement, rotation = self.decide_move_n_rotation(heading, movement)
                    cur_location = self.location[:]
                    self.update_list(cur_location, self.heading, len(dir_possible.keys()), rotation, heading, movement)
                self.move += 1

        # update location and heading
        self.heading, self.location = self.update_location(self.heading, self.location, movement, heading)
        logging.info("next loc" + str(self.location) + "end step")  ###
        logging.info(str(self.trace_list))
        self.step += 1

        return rotation, movement
