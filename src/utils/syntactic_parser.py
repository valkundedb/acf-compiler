from collections import deque

def get_syntactic_distance(doc, span1_start, span1_end, span2_start, span2_end):
    """Calculates shortest path through the ScispaCy grammatical dependency tree using BFS"""
    tokens1 = [t for t in doc if t.idx < span1_end and (t.idx + len(t.text)) > span1_start and not t.is_space and not t.is_punct]
    tokens2 = [t for t in doc if t.idx < span2_end and (t.idx + len(t.text)) > span2_start and not t.is_space and not t.is_punct]
    
    if not tokens1 or not tokens2:
        return float('inf')

    # Build undirected adjacency list from dependency tree
    adj = {t.i: [] for t in doc}
    for t in doc:
        if t.head.i != t.i:
            adj[t.i].append(t.head.i)
            adj[t.head.i].append(t.i)

    # BFS traversal
    queue = deque([(t.i, 0) for t in tokens1])
    visited = set(t.i for t in tokens1)
    target_indices = set(t.i for t in tokens2)

    while queue:
        curr, dist = queue.popleft()
        if curr in target_indices:
            return dist
        for neighbor in adj[curr]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return float('inf')