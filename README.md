# racetrack.py
This Python 3.9 script finds all solutions to Riddler Classic @ https://fivethirtyeight.com/features/can-you-zoom-around-the-race-track/ with an adapted version of the [Dijkstra algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm) where each node of the graph to be navigated is a state = 2-D position and 2-D velocity, and where moves with a negative angular velocity are not considered because they are clearly suboptimal.


Accounting for all path forks and path mergings, there are 322 fastest routes reaching the finish line in the same minimum time, as listed [here](https://github.com/stefperf/racetrack/blob/main/output.txt). Moves are colored based on their ordinal number.

![figure showing all the fastest routes](https://github.com/stefperf/racetrack/blob/main/Fastest_routes.png)


If we add a secondary goal of minimizing route length, then there are only the 2 optimal routes 352447988663 and 352448879663.

![figure showing all the shortest of the fastest routes](https://github.com/stefperf/racetrack/blob/main/Shortest_fastest_routes.png)
