# racetrack
Finding all solutions to Riddler Classic @ https://fivethirtyeight.com/features/can-you-zoom-around-the-race-track/ with an adapted version of the [Dijkstra algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm) where each node of the graph to be navigated is a state including both 2-D position and 2-D velocity.

There are many 322 fastest routes reaching the finish line in the same minimum time. 

![figure showing all the fastest routes](https://github.com/stefperf/racetrack/blob/main/Fastest_routes.png)

If we add a secondary goal of minimizing route length, then there are only 2 optimal routes.

![figure showing all the shortest of the fastest routes](https://github.com/stefperf/racetrack/blob/main/Shortest_fastest_routes.png)
