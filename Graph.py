import random, sys, copy, os, pygame
import io
from collections import defaultdict
from collections import deque
from heapq import heappush, heappop, heapify

class Graph():
    def __init__(self, edges, weights = None, nvertices = 0, directed = False, clean = False):
        self.nvertices = nvertices
        self.edges = edges
        self.directed = directed
        if weights is None:
            self.weights = {}
        else:
            self.weights = weights

        if clean:
            self.cleanup()
        self.update_undirected_edges()
    
    def update_undirected_edges(self):
        if not self.directed:
            # make updated adjacency list
            newEdges = defaultdict(set)
            for a, paths in self.edges.items():
                newEdges[a].update(paths)
                for b in paths:
                    newEdges[b].add(a)
            
            # make updated weights dict
            newWeights = {}
            for (a,b), w in self.weights.items():
                newWeights[(a,b)] = w
                newWeights[(b,a)] = w
            
            self.edges = newEdges
            self.weights = newWeights

    def cleanup(self):
        newV = {}
        vertices = set() # keeps track of all old vertices
        for v, paths in self.edges.items():
            vertices.add(v)
            vertices.update(paths)

        # pairs old vertices with new number, in sorted order because set
        for i, v in enumerate(vertices):
            newV[v] = i
        
        # make new cleaned up adjacency list
        newEdges = defaultdict(set)
        for a, paths in self.edges.items():
            newA = newV[a]
            for b in paths:
                newB = newV[b]
                newEdges[newA].add(newB)

        # make new cleaned up weights dict
        newWeights = {}
        for (a,b), w in self.weights.items():
            newA, newB = newV[a], newV[b]
            newWeights[(newA, newB)] = w

        self.edges = newEdges
        self.weights = newWeights
    
    # helper for the file reading in the from_file that gets rid of comments and leading/trailing whitespace
    @staticmethod
    def stripLine(line):
        for i in range(len(line)):
            if line[i] == '#':
                line = line[:i]
                break
        line = line.strip()
        return line

    @classmethod
    def from_file(cls, openfile, directed = False):
        f = openfile.readlines()
        weightLine = len(f) # default value so if not found it won't look for weights

        # find special comment "# WEIGHTS" and strip comments from other lines
        for i in range(len(f)):
            if "# WEIGHTS" in f[i]:
                weightLine = i
            
            f[i] = Graph.stripLine(f[i])
        
        # make adjacency lists and keep track of all the vertices
        vertices = set()
        edges = defaultdict(set)
        for i in range(weightLine):
            if f[i] == "":
                continue
            
            adjList = list(map(int, f[i].split(" ")))
            edges[adjList[0]] = set(adjList[1:])
            vertices.update(adjList)

        nvertices = len(vertices)

        # initialize weights if they exist
        weights = {}
        for i in range(weightLine+1, len(f)):
            if f[i] == "":
                continue

            a, b, w = map(int, f[i].split(" "))
            weights[(a,b)] = w

        return Graph(edges, weights, nvertices, directed)

    @classmethod
    def from_str(cls, txt, directed = False):
        openfile = io.StringIO(txt)
        
        graph = Graph.from_file(openfile, directed)
        openfile.close()

        return graph
    
    def dfs_path(self, start, end):
        vis = [False] * self.nvertices
        disc = [False] * self.nvertices
        prev = [-1] * self.nvertices
        order = []

        def dfs(v):
            disc[v] = True
            order.append((v, 0))
            for dest in self.edges[v]:
                if not disc[dest]:
                    prev[dest] = v
                    dfs(dest)
                if (vis[end]): return
            vis[v] = True
            order.append((v, 1))
        
        dfs(start)

        path = []
        if vis[end]:
            v = end
            while (v != -1):
                path.append(v)
                v = prev[v]
            
            path.reverse()
            
        return order, path
    
    def bfs_path(self, start, end):
        vis = [False] * self.nvertices
        disc = [False] * self.nvertices
        prev = [-1] * self.nvertices
        order = []

        def bfs(start, end):
            q = deque()
            q.append(start)
            disc[start] = True
            order.append((start, 0))

            while q and not vis[end]:
                cur = q.popleft()
                assert not vis[cur]

                for dest in self.edges[cur]:
                    if not disc[dest]:
                        disc[dest] = True
                        prev[dest] = cur
                        order.append((dest, 0))
                        q.append(dest)

                vis[cur] = True
                order.append((cur, 1))
        
        bfs(start, end)

        path = []
        if vis[end]:
            v = end
            while (v != -1):
                path.append(v)
                v = prev[v]
            
            path.reverse()
            
        return order, path
    
    def bfs2way_path(self, start, end):
        vis = [False] * self.nvertices
        disc = [False] * self.nvertices
        prev = [-1] * self.nvertices
        goal = [-1] * self.nvertices
        order = []

        def bfs2way(start, end):
            q = deque()
            q.append(start)
            q.append(end)
            disc[start] = True
            disc[end] = True
            goal[start] = end
            goal[end] = start
            order.append((start, 0))
            order.append((end, 0))

            while q:
                cur = q.popleft()
                assert not vis[cur]

                for dest in self.edges[cur]:
                    if vis[dest] and goal[dest] != goal[cur]:
                        vis[cur] = True
                        order.append((cur, 1))
                        if goal[cur] == end: return cur, dest
                        else: return dest, cur
                        
                    if not disc[dest]:
                        disc[dest] = True
                        prev[dest] = cur
                        goal[dest] = goal[cur]
                        order.append((dest, 0))
                        q.append(dest)

                vis[cur] = True
                order.append((cur, 1))

            return -1, -1
        
        p1, p2 = bfs2way(start, end)

        path = []
        if p1 != -1:
            v = p1
            while (v != -1):
                path.append(v)
                v = prev[v]
            
            path.reverse()

            v = p2
            while (v != -1):
                path.append(v)
                v = prev[v]
            
        return order, path

    def astar_path(self, start, end, heuristic, weight = 1):
        vis = [False] * self.nvertices
        disc = [False] * self.nvertices
        dist = [0] * self.nvertices
        prev = [-1] * self.nvertices
        order = []
        path = []

        pq = []
        # cost, heuristic, moves, node
        startHeur = heuristic(start, end)
        heappush(pq, (weight * startHeur, startHeur, 0, start))
        disc[start] = True
        order.append((start, 0))
        
        while pq:
            curCost, curHeur, moves, cur = heappop(pq)
            if dist[cur] < moves or vis[cur]:
                continue

            if cur == end:
                vis[cur] = True
                order.append((cur, 1))
                break

            for nbr in self.edges[cur]:
                if vis[nbr]: continue
                newHeur = heuristic(nbr, end)
                newCost = weight * newHeur + moves + 1
                if not disc[nbr] or dist[nbr] > moves + 1:
                    dist[nbr] = moves + 1
                    disc[nbr] = True
                    order.append((nbr, 0))
                    prev[nbr] = cur
                    heappush(pq, (newCost, newHeur, moves + 1, nbr))

            vis[cur] = True
            order.append((cur, 1))
        
        if vis[end]:
            v = end
            
            while (v != -1):
                path.append(v)
                v = prev[v]
            
            path.reverse()

        return order, path
    
    def astar2way_path(self, start, end, heuristic, weight=1):
        n = self.nvertices
        INF = 10**18

        # forward
        dist_f = [INF] * n
        prev_f = [-1] * n
        disc_f = [False] * n
        vis_f  = [False] * n

        # backward
        dist_b = [INF] * n
        prev_b = [-1] * n
        disc_b = [False] * n
        vis_b  = [False] * n

        pq_f = []
        pq_b = []

        order = []

        dist_f[start] = 0
        dist_b[end]   = 0

        hf = heuristic(start, end)
        hb = heuristic(end, start)

        heappush(pq_f, (hf, 0, start))
        heappush(pq_b, (hb, 0, end))

        disc_f[start] = True
        disc_b[end]   = True

        order.append((start, 0))
        order.append((end,   0))

        best = INF
        meet = -1

        while pq_f and pq_b:
            # 🔑 optimal termination condition
            if pq_f[0][0] + pq_b[0][0] >= best:
                break

            # ---------- forward ----------
            fcost, g, u = heappop(pq_f)
            if vis_f[u]:
                pass
            else:
                vis_f[u] = True
                order.append((u, 1))

                if vis_b[u]:
                    if dist_f[u] + dist_b[u] < best:
                        best = dist_f[u] + dist_b[u]
                        meet = u

                for v in self.edges[u]:
                    if vis_f[v]:
                        continue
                    ng = dist_f[u] + 1
                    if ng < dist_f[v]:
                        dist_f[v] = ng
                        prev_f[v] = u
                        if not disc_f[v]:
                            disc_f[v] = True
                            order.append((v, 0))
                        heappush(pq_f, (ng + weight * heuristic(v, end), ng, v))

            # ---------- backward ----------
            fcost, g, u = heappop(pq_b)
            if vis_b[u]:
                continue

            vis_b[u] = True
            order.append((u, 1))

            if vis_f[u]:
                if dist_f[u] + dist_b[u] < best:
                    best = dist_f[u] + dist_b[u]
                    meet = u

            for v in self.edges[u]:
                if vis_b[v]:
                    continue
                ng = dist_b[u] + 1
                if ng < dist_b[v]:
                    dist_b[v] = ng
                    prev_b[v] = u
                    if not disc_b[v]:
                        disc_b[v] = True
                        order.append((v, 0))
                    heappush(pq_b, (ng + weight * heuristic(v, start), ng, v))

        # ---------- reconstruct ----------
        path = []
        if meet != -1:
            v = meet
            while v != -1:
                path.append(v)
                v = prev_f[v]
            path.reverse()

            v = prev_b[meet]
            while v != -1:
                path.append(v)
                v = prev_b[v]

        return order, path

    # count how many connected components in the graph
    def connected_components(self):
        vis = [False] * self.nvertices

        def dfs(v):
            assert vis[v] == False
            vis[v] = True
            for dest in self.edges[v]:
                if not vis[dest]:
                    dfs(dest)
        
        cnt = 0
        for v in range(self.nvertices):
            if not vis[v]:
                dfs(v)
                cnt += 1
        
        return cnt

    def topo_sort(self):
        vis = [False] * self.nvertices
        order = []

        def dfs(v):
            vis[v] = True
            
            for adj in self.edges[v]:
                if not vis[adj]:
                    dfs(adj)
            
            order.append(v)

        for v in range(self.nvertices):
            if not vis[v]:
                dfs(v)
        
        order.reverse()

        return order

    def __repr__(self):
        return f"[Graph, V={self.nvertices}, E={dict(self.edges)}, W={self.weights}]"
