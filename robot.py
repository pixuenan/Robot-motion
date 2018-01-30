import numpy as np
import heapq
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


class Vertex(object):
    def __init__(self, loc, dim):
        """
        :param loc: location in the maze such as (0, 0)
        """
        self.loc = loc
        self.dim = dim
        self.rhs = float("inf")
        self.g = float("inf")
        self.key = [0, 0]
        self.parents = dict()
        self.children = dict()
        self._init_children_parent()

    def calculate_key(self, start_vertex):
        self.key[0] = min([self.g, self.rhs]) + self.calculate_dist(start_vertex)
        self.key[1] = min([self.g, self.rhs])

    def del_children(self, children_loc):
        del self.children[children_loc]

    def del_parent(self, parent_loc):
        del self.parents[parent_loc]

    def calculate_dist(self, start_vertex):
        """Heuristic distance from current vertex to the start vertex"""
        return max([abs(self.loc[0]-start_vertex.loc[0]), abs(self.loc[1]-start_vertex.loc[1])])

    def _init_children_parent(self):
        x, y = self.loc
        self.add_vertex_loc(x, y, self.dim, self.children)
        self.add_vertex_loc(x, y, self.dim, self.parents)

    @staticmethod
    def add_vertex_loc(x, y, dim, the_dict, edge_cost=1):
        if x - 1 > 0:
            the_dict[(x-1, y)] = edge_cost
        if x + 1 < dim - 1:
            the_dict[(x+1, y)] = edge_cost
        if y - 1 > 0:
            the_dict[(x, y-1)] = edge_cost
        if y + 1 < dim - 1:
            the_dict[(x, y+1)] = edge_cost


class Graph(object):
    def __init__(self, dim):
        self.dim = dim
        self.graph = {}
        self.goal_loc = []
        self._get_goal_loc()
        self.km = 0
        self.priority_queue = []

        self._init_vertex()
        self.start_vertex = self.graph[(0, 0)]
        self._init_D_lite()

    def top_key(self):
        self.priority_queue.sort()
        if self.priority_queue:
            return self.priority_queue[0][:2]
        else:
            return [float("inf"), float("inf")]

    def _get_goal_loc(self):
        left_goal_pos = self.dim/2-1
        right_goal_pos = self.dim/2
        self.goal_loc = zip([left_goal_pos]*2 + [right_goal_pos]*2,
                            [left_goal_pos, right_goal_pos, left_goal_pos, right_goal_pos])

    def _init_vertex(self):
        for x in range(self.dim):
            for y in range(self.dim):
                node = Vertex((x, y), self.dim)
                self.graph[(x, y)] = node

    def _init_D_lite(self):
        for goal_loc in self.goal_loc:
            goal_vertex = self.graph[goal_loc]
            goal_vertex.rhs = 0
            goal_vertex.calculate_key(self.start_vertex)
            heapq.heappush(self.priority_queue, goal_vertex.key + [goal_vertex.loc])

    def _calculate_h(self):
        for vertex in self.graph.values():
            vertex.h = max([min(abs(vertex.loc[0]-self.dim/2-1), abs(vertex.loc[0]-self.dim/2)),
                            min(abs(vertex.loc[1]-self.dim/2-1), abs(vertex.loc[1]-self.dim/2))])

    def update_vertex(self, vertex):
        if vertex.loc not in self.goal_loc:
            vertex.rhs = min([self.graph[i].g + vertex.children[i] for i in vertex.children.keys()])
        priority_loc = [loc for loc in self.priority_queue if vertex.loc in loc]
        if priority_loc and len(priority_loc) == 1:
            self.priority_queue.remove(priority_loc[0])
        if vertex.g != vertex.rhs:
            heapq.heappush(self.priority_queue, vertex.calculate_key(self.start_vertex) + [vertex.loc])

    def compute_shortest_path(self):
        while self.top_key() < self.start_vertex.calculate_key(self.start_vertex) or \
              self.start_vertex.rhs != self.start_vertex.g:
            old_key = self.top_key()
            least_vertex = heapq.heappop(self.priority_queue)
            if old_key < least_vertex.calculate_key(self.start_vertex):
                heapq.heappush(self.priority_queue, least_vertex.calculate_key(self.start_vertex) + [least_vertex.loc])
            elif least_vertex.g > least_vertex.rhs:
                least_vertex.g = least_vertex.rhs
                for loc in least_vertex.parents.keys():
                    parent_vertex = self.graph[loc]
                    self.update_vertex(parent_vertex)
            else:
                least_vertex.g = float("inf")
                self.update_vertex(least_vertex)
                for loc in least_vertex.parents.keys():
                    parent_vertex = self.graph[loc]
                    self.update_vertex(parent_vertex)


class Robot(Graph):
    def __init__(self, maze_dim):
        '''
        Use the initialization function to set up attributes that your robot
        will use to learn and navigate the maze. Some initial attributes are
        provided based on common information, including the size of the maze
        the robot is placed in.
        '''
        Graph.__init__(self, maze_dim)
        self.location = [0, 0]
        self.heading = 'up'
        self.maze_dim = maze_dim

        self.possible_action = dict()
        self.move = 0

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
