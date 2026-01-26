import Graph
import random, sys, copy, os, pygame
from collections import defaultdict
import pygame
from pygame.locals import *

# ----------MAZE MAKER----------

QUIT = "EXIT"
gridSize = (50,50)
tileSize = 11
halfSize = tileSize // 2
sidebarWidth = 220
size = (gridSize[0] * tileSize + sidebarWidth, gridSize[1] * tileSize)

def init():
    global screen, clock
    sys.setrecursionlimit(10000)
    clock = pygame.time.Clock()
    pygame.init()
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("maze thing")

def main():
    init()
    
    running = True
    model = {
            "state" : "build", 
            "start" : (-1,-1),
            "end" : (gridSize[0] - 1, gridSize[1] - 1), 
            "search" : "astar",
            "order_index" : -1, 
            "order" : [], 
            "path" : [], 
            "grid" : defaultdict(list)
            }
    
    while (running):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or model == QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                model = key_handler(model, event)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                model = mouse_handler(model, event)
            
        # increment path animation every tick
        if (model["state"] == "graph"):
            model["order_index"] = min(model["order_index"] + 1, len(model["order"]) + len(model["path"]))

        # removing the negative values (the hover values) from the dict
        for pos, dirs in model["grid"].items():
            for i, d in enumerate(dirs):
                if d < 0:
                    dirs.pop(i)
        
        draw_handler(model)
        pygame.display.flip()
        clock.tick(1000)

def getWallDir(dx, dy):
    dists = [dy, tileSize-dx, tileSize-dy, dx]
    minIdx = 0
    minDist = dy
    for i in range(4):
        if (dists[i] < minDist):
                minIdx = i
                minDist = dists[i]
    return minIdx

def draw_sidebar(model):
    panel_x = gridSize[0] * tileSize
    panel_rect = pygame.Rect(panel_x, 0, sidebarWidth, size[1])

    pygame.draw.rect(screen, pygame.Color("lightgray"), panel_rect)
    
    font_big = pygame.font.SysFont(None, 22)
    font_small = pygame.font.SysFont(None, 15)
    color = pygame.Color("black")

    info = [
        "Current Algorithm:",
        f"  {model['search']}",
        "",
        "Path Length:",
        f"  {len(model['path'])}",
        "",
        "Explored Nodes:",
        f"  {min(model['order_index'], len(model['order']))}",
        "",
    ]

    hotkeys = [
        
        "Hotkeys:",
        "  1  DFS",
        "  2  BFS",
        "  3  A*",
        "  4  A* Weighted",
        "  5  BFS 2-Way",
        "  6  A* 2-Way",
        "",
        "  SPACE  Build / Solve",
        "  R      Reset",
        "  G      Gen Maze 1",
        "  H      Gen Maze 2",
        "  Q      Quit",
    ]

    y = 20
    for line in info:
        text = font_big.render(line, True, color)
        screen.blit(text, (panel_x + 15, y))
        y += 28

    
    for line in hotkeys:
        text = font_small.render(line, True, color)
        screen.blit(text, (panel_x + 15, y))
        y += 20


def draw_handler(model):
    global screen
    bgcolor = pygame.color.Color("orange")
    screen.fill(bgcolor)

    draw_sidebar(model)
    
    wallDir = {0 : (0,-1), 1 : (1,0), 2 : (0,1), 3 : (-1, 0)}
    
    # color the nodes in the traversal order
    if model["state"] == "graph" and model["order_index"] != -1:
        visColor = pygame.color.Color("blue")
        discColor = pygame.color.Color("aqua")
        idx = model["order_index"]
        order = model["order"]

        for i in range(min(idx+1, len(order))):
            curNode, nodeState = order[i]
            gridX, gridY = nodeToGrid(curNode)
            x, y = gridX * tileSize, gridY * tileSize
            square = pygame.Rect(x, y, tileSize, tileSize)
            if (nodeState == 1): pygame.draw.rect(screen, visColor, square)
            if (nodeState == 0): pygame.draw.rect(screen, discColor, square)
        
        if idx > len(order):
            path = model["path"]
            pathIdx = min(idx - len(order), len(path))
            pathColor = pygame.color.Color("green")
            for i in range(pathIdx):
                curNode = path[i]
                gridX, gridY = nodeToGrid(curNode)
                x, y = gridX * tileSize, gridY * tileSize
                square = pygame.Rect(x, y, tileSize, tileSize)
                pygame.draw.rect(screen, pathColor, square)
    
    # get mouse grid pos and closest wall and draw hover position during "build" state
    if model["state"] == "build":
        mx, my = pygame.mouse.get_pos()
        mdx, mdy = mx % tileSize, my % tileSize
        mouseDir = getWallDir(mdx, mdy)
        mgx, mgy= mx // tileSize, my // tileSize
        mx, my = mgx * tileSize + halfSize, mgy * tileSize + halfSize
        
        wall = wallDir[mouseDir]
        mStart = pygame.math.Vector2(mx + wall[0] * halfSize, my + wall[1] * halfSize)
        mEnd = pygame.math.Vector2(mx + wall[0] * halfSize, my + wall[1] * halfSize)
        if wall[0] == 0:
            mStart[0] += halfSize
            mEnd[0] += -halfSize
        if wall[1] == 0:
            mStart[1] += halfSize
            mEnd[1] += -halfSize
        
        if inBound((mgx, mgy)):
            if mouseDir in model["grid"][(mgx,mgy)]:
                    pygame.draw.line(screen, "red", mStart, mEnd)
            else:
                    pygame.draw.line(screen, "blue", mStart, mEnd)
                
    # draw walls
    wall_color = pygame.color.Color("black")
    
    for pos, dirs in model["grid"].items():
        (gridX, gridY) = pos
        x, y = gridX * tileSize + halfSize, gridY * tileSize + halfSize
        for d in dirs:
            wall = wallDir[d]
            start = pygame.math.Vector2(x + wall[0] * halfSize, y + wall[1] * halfSize)
            end = pygame.math.Vector2(x + wall[0] * halfSize, y + wall[1] * halfSize)
            if wall[0] == 0:
                start[0] += halfSize
                end[0] += -halfSize
            if wall[1] == 0:
                start[1] += halfSize
                end[1] += -halfSize
                    
            start = pygame.math.Vector2(start)
            end = pygame.math.Vector2(end)
            pygame.draw.line(screen, wall_color, start, end)

def inBound(gridPos):
    x, y = gridPos
    xBound = 0 <= x and x < gridSize[0]
    yBound = 0 <= y and y < gridSize[1]
    return xBound and yBound

def reset(model):
    model["grid"] = defaultdict(list)
    model["state"] = "build"
    model["order_index"] = -1
    model["order"] = []
    model["path"] = []

def genMaze1(model):
    wallDir = {0 : (0,-1), 1 : (1,0), 2 : (0,1), 3 : (-1, 0)}
    reset(model)
    model["state"] = "graph"
    for x in range(gridSize[0]):
        for y in range(gridSize[1]):
            for dir in range(4):
                if dir in model["grid"][(x, y)]:
                    continue
                pairPos = (x + wallDir[dir][0], y + wallDir[dir][1])
                pairDir = (dir + 2) % 4
                if not inBound(pairPos):
                    continue
                makeWall = random.randint(1,4)
                if makeWall == 4:
                    model["grid"][(x, y)].append(dir)
                    model["grid"][pairPos].append(pairDir)

def break_wall(model, x, y, d):
    DIRS = [(0,-1),(1,0),(0,1),(-1,0)]
    OPP = {0:2, 1:3, 2:0, 3:1}

    nx = x + DIRS[d][0]
    ny = y + DIRS[d][1]

    if not inBound((nx, ny)):
        return

    if d in model["grid"][(x, y)]:
        model["grid"][(x, y)].remove(d)
    if OPP[d] in model["grid"][(nx, ny)]:
        model["grid"][(nx, ny)].remove(OPP[d])

def genMaze2(model, break_wall_prob=0.05):
    reset(model)
    model["state"] = "graph"

    DIRS = [(0,-1),(1,0),(0,1),(-1,0)]
    OPP = {0:2, 1:3, 2:0, 3:1}

    w, h = gridSize
    visited = [[False]*h for _ in range(w)]

    # start with all walls
    for x in range(w):
        for y in range(h):
            model["grid"][(x,y)] = [0,1,2,3]

    def dfs(x, y):
        visited[x][y] = True
        dirs = [0,1,2,3]
        random.shuffle(dirs)

        for d in dirs:
            nx = x + DIRS[d][0]
            ny = y + DIRS[d][1]

            if not inBound((nx,ny)) or visited[nx][ny]:
                continue

            break_wall(model, x, y, d)
            dfs(nx, ny)

    # build perfect maze
    dfs(0, 0)

    # ---------- ADD MULTIPLE PATHS ----------
    for x in range(w):
        for y in range(h):
            for d in range(4):
                if d in model["grid"][(x,y)]:
                    if random.random() < break_wall_prob:
                        break_wall(model, x, y, d)

        
def key_handler(model, event):
    if len(model) == 0:
        return model
    
    k = event.key
    
    if k == pygame.K_q:
        model = QUIT
    elif k == pygame.K_a or k == pygame.K_LEFT:
        model["order_index"] = max(model["order_index"] - 1, 0)
    elif k == pygame.K_d or k == pygame.K_RIGHT:
        model["order_index"] = min(model["order_index"] + 1, len(model["order"]) + len(model["path"]) - 1)
    elif k == pygame.K_r:
        reset(model)
    elif k == pygame.K_g:
        genMaze1(model)
    elif k == pygame.K_h:
        genMaze2(model)
    elif k == pygame.K_SPACE and model["state"] == "build":
        model["state"] = "graph"
        model["order_index"] = -1
        model["order"] = []
        model["path"] = []
    elif k == pygame.K_SPACE and model["state"] == "graph":
        model["state"] = "build"
    elif k == pygame.K_p:
        model["order_index"] = len(model["order"]) - 1
    elif k == pygame.K_1:
        model["search"] = "dfs"
    elif k == pygame.K_2:
        model["search"] = "bfs"
    elif k == pygame.K_3:
        model["search"] = "astar"
    elif k == pygame.K_4:
        model["search"] = "astar_weighted"
    elif k == pygame.K_5:
        model["search"] = "bfs2way"
    elif k == pygame.K_6:
        model["search"] = "astar2way"
            
    return model

def gridToNode(pos):
    x, y = pos
    node = y * gridSize[0] + x

    return node

def nodeToGrid(node):
    x = node % gridSize[0]
    y = node // gridSize[0]

    return (x,y)

def heuristic(node, goal):
    nodeX, nodeY = nodeToGrid(node)
    goalX, goalY = nodeToGrid(goal)

    return abs(nodeX - goalX) + abs(nodeY - goalY)

def isLeftClick(event):
    return event.type == pygame.MOUSEBUTTONDOWN and event.button == 1

def isRightClick(event):
    return event.type == pygame.MOUSEBUTTONDOWN and event.button == 3

def makeGraph(model):
    wallDir = {0 : (0,-1), 1 : (1,0), 2 : (0,1), 3 : (-1, 0)}

    edges = defaultdict(list)
    nvertices = gridSize[0] * gridSize[1]
    for node in range(nvertices):
        curGridPos = nodeToGrid(node)
        for dir in range(4):
            if dir not in model["grid"][curGridPos]:
                shift = wallDir[dir]
                neighborGrid = (curGridPos[0] + shift[0], curGridPos[1] + shift[1])
                if inBound(neighborGrid):
                    neighborNode = gridToNode(neighborGrid)
                    edges[node].append(neighborNode)
    
    graph = Graph.Graph(edges, nvertices = nvertices)
    return graph

def setPath(model, graph, start, end):
    if model["search"] == "dfs": model["order"], model["path"] = graph.dfs_path(start, end)
    if model["search"] == "bfs": model["order"], model["path"] = graph.bfs_path(start, end)
    if model["search"] == "astar": model["order"], model["path"] = graph.astar_path(start, end, heuristic)
    if model["search"] == "astar_weighted": model["order"], model["path"] = graph.astar_path(start, end, heuristic, weight = 100)
    if model["search"] == "bfs2way": model["order"], model["path"] = graph.bfs2way_path(start, end)
    if model["search"] == "astar2way": model["order"], model["path"] = graph.astar2way_path(start, end, heuristic)
    
def mouse_handler(model, event):
    x, y = event.pos
    dx = x % tileSize
    dy = y % tileSize
    
    # find out which wall the click was closest to
    minIdx = getWallDir(dx, dy)
    wallDir = {0 : (0,-1), 1 : (1,0), 2 : (0,1), 3 : (-1, 0)}
    wall = wallDir[minIdx]

    gridX = x // tileSize
    gridY = y // tileSize
    gridPos = (gridX, gridY)
    pairPos = (gridX + wall[0], gridY + wall[1])
    pairDir = (minIdx + 2) % 4

    # exit early if the click wasnt in bounds
    if not inBound(gridPos): return model

    if isLeftClick(event) and model["state"] == "build":
        # remove the wall if its already there and add it if not
        if minIdx in model["grid"][gridPos]:
            for i, d in enumerate(model["grid"][gridPos]):
                if minIdx == d:
                    model["grid"][gridPos].pop(i)
            for i, d in enumerate(model["grid"][pairPos]):
                if pairDir == d:
                    model["grid"][pairPos].pop(i)
        else:
            model["grid"][gridPos].append(minIdx)
            model["grid"][pairPos].append(pairDir)
        
    elif isLeftClick(event) and model["state"] == "graph":
        model["start"] = gridPos
        model["order_index"] = 0
        graph = makeGraph(model)
        start = gridToNode(model["start"])
        end = gridToNode(model["end"])
        setPath(model, graph, start, end)
        
        print(len(model["path"]))
    
    elif isRightClick(event) and model["state"] == "graph":
        model["end"] = gridPos

    return model
        
if __name__ == '__main__':
    main()