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
dir_reverse_rotation = {'r': {'l': 180, 'u': 90, 'd': -90, 'r': 0},
                        'l': {'r': 180, 'u': -90, 'd': 90, 'l': 0},
                        'u': {'d': 180, 'r': -90, 'l': 90, 'u': 0},
                        'd': {'u': 180, 'l': -90, 'r': 90, 'd': 0}}


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
        self.heading = 'u'
        self.maze_dim = maze_dim

        self.possible_action = []
        self.action_list = [[self.location, self.heading]]
        self.move = 0

    @staticmethod
    def update_location(cur_heading, cur_location, movement, heading):
        '''
        Update robot location based on the movement and heading chosen
        '''
        cur_location = list(cur_location)
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
        self.possible_action = []
        for idx, possible_heading in enumerate(dir_sensors[self.heading]):
            wall_distance = sensors[idx]
            if wall_distance:
                next_loc = self.update_location(self.heading, self.location, 1, possible_heading)[1]
                self.possible_action.append(next_loc)

    def decide_move_n_rotation(self, next_loc):
        x, y = self.location
        next_x, next_y = next_loc
        if next_x != x:
            movement = abs(next_x - x)
            if next_x > x:
                heading = 'r'
            else:
                heading = 'l'
        else:
            movement = abs(next_y - y)
            if next_y > y:
                heading = 'u'
            else:
                heading = 'd'
        rotation = dir_reverse_rotation[heading][self.heading]
        if rotation == 180:
            rotation, movement = 0, -movement
        return movement, rotation, heading

    def update_obstacle_change(self, current_loc, next_loc):
        current_vertex = self.graph[current_loc]
        next_vertex = self.graph[next_loc]
        current_vertex.children[next_loc] = float("inf")
        next_vertex.children[next_loc] = float("inf")

    def obstacle_change_vertex(self, sensors):
        last_loc, last_heading = self.action_list[-1]
        #dead end
        if sensors.count(0) == 3:
            self.update_obstacle_change(tuple(self.location), tuple(last_loc))
        else:
            for idx, possible_heading in enumerate(dir_sensors[last_heading]):
                wall_distance = sensors[idx]
                if wall_distance:
                    next_loc = self.update_location(last_heading, last_loc, 1, possible_heading)[1]
                    self.update_obstacle_change(tuple(last_loc), tuple(next_loc))
                    self.update_vertex(self.graph[tuple(last_loc)])

    def get_next_loc(self):
        current_vertex = self.graph[tuple(self.location)]
        min_cost = min([self.graph[i].g + current_vertex.children[i] for i in current_vertex.children.keys() if list(i) in self.possible_action])
        print("###", min_cost)
        for node_loc in current_vertex.children.keys():
            node = self.graph[node_loc]
            if node.g + current_vertex.children[node_loc] == min_cost and list(node_loc) in self.possible_action:
                next_loc = node_loc
        return next_loc

    def action(self, sensors):
        if self.location == [0, 0]:
            self.compute_shortest_path()
        #dead end
        if not self.possible_action:
            movement, rotation, heading = -1, 0, self.heading
            next_loc = self.action_list[-2][0]
        else:
            next_loc = self.get_next_loc()
            movement, rotation, heading = self.decide_move_n_rotation(next_loc)
        
        self.km += self.start_vertex.calculate_dist(self.graph[tuple(next_loc)])
        self.location, self.heading = next_loc, heading
        self.start_vertex = self.graph[tuple(self.location)]
        self.obstacle_change_vertex(sensors)

        self.compute_shortest_path()

        if self.possible_action:
            self.action_list.append([self.location, heading])
        else:
            del self.action_list[-1]
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
        print self.location
        if not self.location in self.goal_loc:
            self.next_pos_move(sensors)
            movement, rotation = self.action(sensors)
            print("---", movement, rotation)

            return rotation, movement
        else:
            return "Reset", "Reset"
