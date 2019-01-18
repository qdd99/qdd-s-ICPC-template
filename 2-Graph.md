## 4.2 图论

### BFS

```cpp
// vector
// MAXN开点数
int dist[MAXN];
bool vis[MAXN];
vector<int> G[MAXN];

void bfs(int s) {
    queue<int> q;
    vis[s] = true;
    dist[s] = 0;
    q.push(s);
    while (!q.empty()) {
        int now = q.front();
        for (int i = 0; i < G[now].size(); i++) {
            if (!vis[G[now][i]]) {
                vis[G[now][i]] = true;
                q.push(G[now][i]);
                dist[G[now][i]] = dist[now] + 1;
            }
        }
        q.pop();
    }
}
 
// 前向星
// MAXN开边数
int ecnt;
int mp[MAXN], dist[MAXN];
bool vis[MAXN];

struct Edge {
    int to, nxt;
    Edge(int to = 0, int nxt = 0) : to(to), nxt(nxt) {};
} es[MAXN];

void mp_init() {
    memset(mp, -1, sizeof(mp));
    ecnt = 0;
}

void mp_link(int u, int v) {
    es[ecnt] = Edge(v, mp[u]);
    mp[u] = ecnt++;
}

void bfs(int s) {
    queue<int> q;
    vis[s] = true;
    dist[s] = 0;
    q.push(s);
    while (!q.empty()) {
        int now = q.front();
        for (int i = mp[now]; i != -1; i = es[i].nxt) {
            if (!vis[es[i].to]) {
                vis[es[i].to] = true;
                q.push(es[i].to);
                dist[es[i].to] = dist[now] + 1;
            }
        }
        q.pop();
    }
}
```

### Dijkstra

```cpp
// vector
// MAXN开点数
struct Edge {
    int to, val;
    Edge(int to = 0, int val = 0) : to(to), val(val) {};
};
vector<Edge> G[MAXN];
long long dist[MAXN];

void dijkstra(int s) {
    using pii = pair<long long, int>;
    memset(dist, 0x3f, sizeof(dist));
    priority_queue<pii, vector<pii>, greater<pii> > q;
    dist[s] = 0;
    q.push({0, s});
    while (!q.empty()) {
        pii p = q.top();
        q.pop();
        int now = p.second;
        if (dist[now] < p.first) continue;
        for (int i = 0; i < G[now].size(); i++) {
            int to = G[now][i].to;
            if (dist[to] > dist[now] + G[now][i].val) {
                dist[to] = dist[now] + G[now][i].val;
                q.push({dist[to], to});
            }
        }
    }
}

// 前向星
// MAXN开边数
int ecnt;
int mp[MAXN];
long long dist[MAXN];

struct Edge {
    int to, nxt, val;
    Edge(int to = 0, int nxt = 0, int val = 0) : to(to), nxt(nxt), val(val) {};
} es[MAXN];

void mp_init() {
    memset(mp, -1, sizeof(mp));
    ecnt = 0;
}

void mp_link(int u, int v, int val) {
    es[ecnt] = Edge(v, mp[u], val);
    mp[u] = ecnt++;
}

void dijkstra(int s) {
    using pii = pair<long long, int>;
    memset(dist, 0x3f, sizeof(dist));
    priority_queue<pii, vector<pii>, greater<pii> > q;
    dist[s] = 0;
    q.push({0, s});
    while (!q.empty()) {
        pii p = q.top();
        q.pop();
        int now = p.second;
        if (dist[now] < p.first) continue;
        for (int i = mp[now]; i != -1; i = es[i].nxt) {
            int to = es[i].to;
            if (dist[to] > dist[now] + es[i].val) {
                dist[to] = dist[now] + es[i].val;
                q.push({dist[to], to});
            }
        }
    }
}
```

### LCA

```cpp
int dep[MAXN], up[MAXN][22]; // 22 = ((int)log2(MAXN) + 1)

void dfs(int now, int pa) {
    dep[now] = dep[pa] + 1;
    up[now][0] = pa;
    for (int i = 1; i < 22; i++) {
        up[now][i] = up[up[now][i - 1]][i - 1];
    }
    for (int i = 0; i < G[now].size(); i++) {
        if (G[now][i] != pa) {
            dfs(G[now][i], now);
        }
    }
}

int lca(int u, int v) {
    if (dep[u] > dep[v]) swap(u, v);
    int t = dep[v] - dep[u];
    for (int i = 0; i < 22; i++) {
        if ((t >> i) & 1) v = up[v][i];
    }
    if (u == v) return u;
    for (int i = 21; i >= 0; i--) {
        if (up[u][i] != up[v][i]) {
            u = up[u][i];
            v = up[v][i];
        }
    }
    return up[u][0];
}
```
