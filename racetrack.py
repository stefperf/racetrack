# solving Riddler Classic @ https://fivethirtyeight.com/features/can-you-zoom-around-the-race-track/

import functools
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from collections import namedtuple
from math import atan2, pi


P = namedtuple('P', 'x y')  # 2-D Point

# to label moves
ACCELERATIONS = {i: P(x, y) for i, (x, y) in enumerate([(x, y) for x in range(-1, 2) for y in range(-1, 2)], 1)}


def calc_angle(x, y):
    """get angle of polar coordinates from Cartesian coordinates"""
    return (atan2(y, x) + (2 * pi if y < 0 else 0.)) / (2 * pi)


def point_sum(p0, p1):
    """vector sum of two Points"""
    return P(p0.x + p1.x, p0.y + p1.y)


def point_dist(p0, p1):
    """Euclidean distance between two Points"""
    return ((p0.x - p1.x) ** 2 + (p0.y - p1.y) ** 2) ** 0.5


def point_rotate(p):
    """rotate Point by 90% clockwise"""
    return P(p.y, -p.x)


class S:
    """race State = <2-D position: Point, 2-D velocity: Point>"""
    def __init__(self, p, v):
        self.p = p
        self.v = v

    def __key(self):
        return (self.p.x, self.p.y, self.v.x, self.v.y)

    def __eq__(self, other):
        return self.__key() == other.__key()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.__key())

    def __repr__(self):
        return f'S(p={self.p}, v={self.v})'

    def get_potential_moves(self):
        """get all potential next states in a boundless track"""
        potential_moves = {}
        for move, acc in ACCELERATIONS.items():
            v1 = point_sum(self.v, acc)
            p1 = point_sum(self.p, v1)
            potential_moves[move] = S(p1, v1)
        return potential_moves


A = namedtuple('A', 'moves prevs time')  # attributes of a state: prev( State)s, moves leading to it in given time


class RaceTrack:
    """representing a race track and the graph of possible race states to optimize upon"""
    def __init__(self, half_side=7, r=3, start_x=None, accept=None):
        """init the track and the graph of possible race states"""
        self.half_side = half_side
        self.r = r
        self.r2 = r ** 2
        self.accept = accept if accept else self.angle_doesnt_worsen
        self.points_to_angles = {
            P(x, y): calc_angle(x, y)
            for x in range(-half_side, half_side + 1)
                for y in range(-half_side, half_side + 1)
                    if x**2 + y**2 >= self.r2
        }
        self.start_point = P(x=start_x if start_x else (half_side + r) // 2, y=0)
        self.start_node = S(self.start_point, P(0, 0))
        self.nodes = {self.start_node: A(moves=[''], prevs=[], time=0.)}
        self.starts = set([self.start_node])
        self.ends = set()
        self.best_routes = []

    @functools.lru_cache(maxsize=None)
    def segment_crosses_circle(self, p0, p1):
        """True iff the segment between the two given Points crosses the circle in two crossing points"""
        if p0.x == p1.x:
            delta = self.r2 - p0.x ** 2
            if delta <= 0.:
                return False
            else:
                (y_min, y_max) = (p0.y, p1.y) if p0.y < p1.y else (p1.y, p0.y)
                y_plus = delta ** 0.5
                y_minus = - y_plus
                return y_min <= y_minus <= y_max and y_min <= y_plus <= y_max
        elif p0.y == p1.y:
            delta = self.r2 - p0.y ** 2
            if delta <= 0.:
                return False
            else:
                (x_min, x_max) = (p0.x, p1.x) if p0.x < p1.x else (p1.x, p0.x)
                x_plus = delta ** 0.5
                x_minus = -x_plus
                return x_min <= x_minus <= x_max and x_min <= x_plus <= x_max
        else:
            m = (p1.y - p0.y) / (p1.x - p0.x)
            k = p0.y - m * p0.x
            km = k * m
            m2_1 = m ** 2 + 1
            delta = km ** 2 - m2_1 * (k ** 2 - self.r2)
            if delta <= 0.:
                return False
            else:
                (x_min, x_max) = (p0.x, p1.x) if p0.x < p1.x else (p1.x, p0.x)
                delta_sqrt = delta ** 0.5
                x_plus = (-km + delta_sqrt) / m2_1
                x_minus = (-km - delta_sqrt) / m2_1
                return x_min <= x_minus <= x_max and x_min <= x_plus <= x_max

    @functools.lru_cache(maxsize=None)
    def arrow_crosses_finish(self, p0, p1):
        """True iff Point p1 is on or beyond the finish line, but p0 is not"""
        if p1.y < 0:
            return False
        elif p0.y >= 0:
            return False
        elif p0.x == p1.x:
            return self.r <= p0.x <= self.half_side
        else:
            x_cross = p0.x - p0.y * (p1.x - p0.x) / (p1.y - p0.y)
            return self.r <= x_cross <= self.half_side

    def get_admissible_moves(self, s0, accept=None):
        """
        If no potential move crosses the finish line, then return:
            (list of all potential next states within the track boundaries, False)
        otherwise return:
            (list of all potential next states within the track boundaries and across the finish line, True)
        """
        if not accept:
            accept = self.accept
        admissible_moves = {}
        finish_possible = False
        for move, s1 in s0.get_potential_moves().items():
            if not s1.p in self.points_to_angles or self.segment_crosses_circle(s0.p, s1.p):
                continue
            finishing = self.arrow_crosses_finish(s0.p, s1.p)
            if not finish_possible and finishing:
                admissible_moves = {}
                finish_possible = True
            if finish_possible and finishing or not finish_possible and accept(s0, s1):
                admissible_moves[move] = s1
        return admissible_moves, finish_possible

    def angle_compares_ok(self, s0, s1, comparison_function):
        """True iff the angle improves between the two given states, as measured by the given comparison_function"""
        phi0, phi1 = self.points_to_angles[s0.p], self.points_to_angles[s1.p]
        return comparison_function(phi0, phi1) and phi1 - phi0 < 0.5

    def angle_improves(self, s0, s1):
        """True iff the angle improves between the two given states"""
        return self.angle_compares_ok(s0, s1, lambda phi0, phi1: phi1 > phi0)

    def angle_doesnt_worsen(self, s0, s1):
        """True iff the angle does not worsen between the two given states"""
        return self.angle_compares_ok(s0, s1, lambda phi0, phi1: phi1 >= phi0)

    def move_forward(self):
        """enrich the graph of race states by one move forward from every last reached state"""
        new_starts = set()
        finish_reached = False
        for s0 in self.starts:
            a0 = self.nodes[s0]
            admissible_moves, finishing = self.get_admissible_moves(s0)
            finish_reached = finish_reached or finishing
            for move1, s1 in admissible_moves.items():
                new_time = a0.time + 1.
                if s1 not in self.nodes:
                    a1 = A(moves=[move1], prevs=[s0], time=new_time)
                    self.nodes[s1] = a1
                    new_starts.add(s1)
                elif self.nodes[s1].time == new_time:
                    a1 = self.nodes[s1]
                    a1.moves.append(move1)
                    a1.prevs.append(s0)
                if finishing:
                    self.ends.add(s1)
        if finish_reached:
            self.starts = set()
        else:
            self.starts = new_starts
        return finish_reached

    def retrace_paths_back_to_start(self, s1):
        """get all bakward sequences of points from state s1 to the start point"""
        if s1 == self.start_node:
            return [[('', s1.p)]]
        paths = []
        a1 = self.nodes[s1]
        for move01, s0 in zip(a1.moves, a1.prevs):
            path_end = [(move01, s1.p)]
            prev_paths = self.retrace_paths_back_to_start(s0)
            for prev_path in prev_paths:
                paths.append(path_end + prev_path)
        return paths

    def score_path(self, points):
        """
        compute the time and the length of a path, given as a sequence of points;
        for the move crossing the finish line, only the fraction of time and length before crossing is considered
        """
        time, length = 0., 0.
        for p0, p1 in zip(points[:-1], points[1:]):
            fraction_before_finish = -p0.y / (p1.y - p0.y) if self.arrow_crosses_finish(p0, p1) else 1.
            time += fraction_before_finish
            length += fraction_before_finish * point_dist(p0, p1)
        return time, length

    def optimize_route(self, consider_length, do_print=True, do_plot=True):
        """
        identify and return the best routes [(time, length, <string_of_move_labels>, <list_of_points>)]
        :param consider_length: True iff the min. length should be a secondary optimization criterion, after min. time
        :param do_print: True iff the best routes should be printed
        :param do_plot: True iff the best routes should be plotted
        :return: the best found routes [(time, length, <string_of_move_labels>, <list_of_points>)]
        """
        if do_print:
            print(f'\nFINDING THE {"SHORTEST OF THE " if consider_length else ""}FASTEST ROUTES...')
        n_moves = 0
        while self.starts:
            n_moves += 1
            self.move_forward()
            n_open_paths = len(self.starts)
            if n_open_paths:
                if do_print:
                    print(f'After {n_moves} moves, {len(self.starts)} states/nodes have been reached...')
        backward_paths = []
        for s_end in self.ends:
            backward_paths.extend(self.retrace_paths_back_to_start(s_end))
        best_routes = []
        for backward_path in backward_paths:
            moves, points = zip(*reversed(backward_path))
            moves = ''.join([str(move) for move in moves])
            time, length = self.score_path(points)
            best_routes.append((time, length, moves, [point_rotate(point) for point in points]))
        best_routes = sorted(best_routes)
        filter_elems = 2 if consider_length else 1
        best_score = best_routes[0][:filter_elems]
        n_best = 0
        for route in best_routes:
            if route[:filter_elems] > best_score:
                break
            else:
                n_best += 1
                if do_print:
                    (time, length, moves, points) = route
                    print(f'time = {time}, length = {length:.3f}, '
                          f'moves = {moves}, points = {[(p.x, p.y) for p in points]}')
        self.best_routes = best_routes[:n_best]
        if do_print:
            print(f'After {n_moves} moves, the finish line has been reached. '
                  f'These are the {n_best} {"shortest of the " if consider_length else ""}fastest routes:')
        if do_plot:
            (time, length) = self.best_routes[0][:2]
            if consider_length:
                title = f'The {n_best} shortest of the fastest routes: time = {time:.3f}, length = {length:.3f}'
            else:
                title = f'The {n_best} fastest routes: time = {time:.3f}'
            self.plot_best_routes(self.best_routes, title)
        return self.best_routes

    def plot_best_routes(self, routes, title, do_annotate=None):
        """
        plot the given routes
        :param routes: sequence of routes [(time, length, <string_of_move_labels>, <list_of_points>)]
        :param title: figure title
        :param do_annotate: True iff moves should be labelled in the plot
        :return: nothing
        """
        if do_annotate is None:
            do_annotate = len(routes) <= 10
        foreground = 10
        background = 0
        hs = self.half_side
        margin = 1
        plt_hs = hs + margin
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(plt_hs, plt_hs))
        fig.suptitle(title, fontsize=16)
        plt.xlim(-plt_hs, +plt_hs)
        plt.ylim(-plt_hs, +plt_hs)
        major_ticks = np.linspace(-plt_hs, plt_hs, 2 * plt_hs + 1)
        ax.set_xticks(major_ticks)
        ax.set_yticks(major_ticks)
        wall_col = 'gray'
        ax.grid(color=wall_col, zorder=background)
        ax.plot([-hs, +hs], [-hs, -hs], color=wall_col, zorder=background)
        ax.plot([-hs, +hs], [+hs, +hs], color=wall_col, zorder=background)
        ax.plot([-hs, -hs], [-hs, +hs], color=wall_col, zorder=background)
        ax.plot([+hs, +hs], [-hs, +hs], color=wall_col, zorder=background)
        ax.add_patch(plt.Circle((0, 0), self.r, color=wall_col, zorder=background))
        ax.add_patch(plt.Rectangle((-plt_hs, -plt_hs), margin, 2 * plt_hs, color=wall_col, zorder=background))
        ax.add_patch(plt.Rectangle((hs, -plt_hs), margin, 2 * plt_hs, color=wall_col, zorder=background))
        ax.add_patch(plt.Rectangle((-plt_hs, -plt_hs), 2 * plt_hs,  margin, color=wall_col, zorder=background))
        ax.add_patch(plt.Rectangle((-plt_hs, hs), 2 * plt_hs,  margin, color=wall_col, zorder=background))
        ax.plot([0, 0], [-self.r, -hs], linewidth=3, color=wall_col, zorder=background)
        for (time, length, moves, points) in routes:
            colors = reversed(cm.rainbow(np.linspace(0, 1, len(points) - 1)))
            for arrow_count, (p0, p1, move, arrow_color) in enumerate(zip(points[:-1], points[1:], moves, colors)):
                arrow_zorder = foreground + arrow_count
                ax.plot([p0.x], [p0.y], marker='o', markersize=5, color=arrow_color, zorder=arrow_zorder - 2)
                ax.arrow(p0.x, p0.y, p1.x - p0.x, p1.y - p0.y, color=arrow_color, linewidth=2, zorder=arrow_zorder,
                                 head_width=0.2, head_length=0.4, length_includes_head=True)
                if do_annotate:
                    ax.annotate(move, ((p0.x + p1.x) / 2, (p0.y + p1.y) / 2), zorder=arrow_zorder + 10)
        plt.show()


if __name__ == '__main__':
    RaceTrack().optimize_route(consider_length=False)
    RaceTrack().optimize_route(consider_length=True)
