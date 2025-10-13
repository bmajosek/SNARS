import random
import math
import matplotlib.pyplot as plt

def p1(height, width):
    graph = {i: [] for i in range(height * width)}
    for i in range(height):
        for j in range(width):
            if j < width - 1:
                graph[j + i * width].append(j + 1 + i * width)
            if i < height - 1:
                graph[j + i * width].append(j + (i + 1) * width)
    return graph
        

def p2(n):
    graph = {i: [] for i in range(n)}
    for i in range(n - 1):
        graph[i].append((i + 1) %n)
    return graph

def p3(n):
    graph = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i):
            graph[i].append(j)
            graph[j].append(i)
                
    return graph

def p4(n):
    graph = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i):
            if random.random() >= 0.5:
                graph[i].append((j, (i + j) % n))
                graph[j].append((i, (i + j) % n))
    return graph
            
def p5(n):
    graph = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j and random.random() >= 0.5:
                graph[i].append(j)                
    return graph
                
def p6(graph):
    in_degrees = {i: 0 for i in graph} 
    
    for i in graph:
        print(f"out degree of {i}-th vertex: {len(graph[i])}")
        for neigh in graph[i]:
            if isinstance(neigh, tuple):
                node, _ = neigh
            else:
                node = neigh
            in_degrees[node] += 1
    for i in in_degrees:
        print(f"in degree of {i}-th vertex: {in_degrees[i]}")

def draw_graph(graph, weighted=False, directed=False, grid=False, grid_rows=None, grid_cols=None):
    positions = {}

    if grid:
        for node in graph:
            row = node // grid_cols
            col = node % grid_cols
            positions[node] = (col * 2, -row * 2)
    else:
        radius = 5
        angle_step = 2 * math.pi / len(graph)
        for node in graph:
            angle = node * angle_step
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            positions[node] = (x, y)

    _, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.axis('off')

    for node, neighbors in graph.items():
        for neighbor in neighbors:
            if weighted:
                target, weight = neighbor
            else:
                target = neighbor
                weight = None

            x1, y1 = positions[node]
            x2, y2 = positions[target]

            if directed:
                dx = x2 - x1
                dy = y2 - y1
                ax.arrow(x1, y1, dx * 0.95, dy * 0.95,
                         head_width=0.2, length_includes_head=True,
                         color='black')
            else:
                ax.plot([x1, x2], [y1, y2], color='black')

            if weight is not None:
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                ax.text(mid_x, mid_y, str(weight), fontsize=14, color='red')

    for node, (x, y) in positions.items():
        ax.plot(x, y, 'o', markersize=12, color='skyblue')
        ax.text(x, y, str(node), fontsize=10, ha='center', va='center')

    plt.show()
    
    
print("#"*50)
g = p1(3, 4)
draw_graph(g, grid=True, grid_cols=4, grid_rows=3)

print("#"*50)
g = p2(6)
draw_graph(g)

print("#"*50)
g = p3(6)
draw_graph(g)
p6(g)

print("#"*50)
g = p4(7)
draw_graph(g, weighted=True)
p6(g)

print("#"*50)
g = p5(6)
draw_graph(g, directed=True)
p6(g)