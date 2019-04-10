## 图论

### 链式前向星

```cpp
int ecnt, mp[MAXN];

struct Edge {
    int to, nxt;
    Edge(int to = 0, int nxt = 0) : to(to), nxt(nxt) {}
} es[MAXM];

void mp_init() {
    memset(mp, -1, (n + 2) * sizeof(int));
    ecnt = 0;
}

void mp_link(int u, int v) {
    es[ecnt] = Edge(v, mp[u]);
    mp[u] = ecnt++;
}

for (int i = mp[now]; i != -1; i = es[i].nxt)
```

### Dijkstra

```cpp
struct Edge {
    int to, val;
    Edge(int to = 0, int val = 0) : to(to), val(val) {}
};
vector<Edge> G[MAXN];
ll dist[MAXN];

void dijkstra(int s) {
    using pii = pair<ll, int>;
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
```

### 拓扑排序

```cpp
int n, deg[MAXN], dis[MAXN];
vector<int> G[MAXN];

bool topo(vector<int>& ans) {
    queue<int> q;
    for (int i = 1; i <= n; i++) {
        if (deg[i] == 0) {
            q.push(i);
            dis[i] = 1;
        }
    }
    while (!q.empty()) {
        int now = q.front();
        q.pop();
        ans.push_back(now);
        for (int nxt : G[now]) {
            deg[nxt]--;
            dis[nxt] = max(dis[nxt], dis[now] + 1);
            if (deg[nxt] == 0) q.push(nxt);
        }
    }
    return ans.size() == n;
}
```

### 最小生成树

```cpp
// 前置：并查集
struct Edge {
    int from, to, val;
    Edge(int from = 0, int to = 0, int val = 0) : from(from), to(to), val(val) {}
};

vector<Edge> es;

ll kruskal() {
    sort(es.begin(), es.end(), [](Edge& x, Edge& y) { return x.val < y.val; });
    iota(pa, pa + n + 1, 0);
    ll ans = 0;
    for (Edge& e : es) {
        if (find(e.from) != find(e.to)) {
            merge(e.from, e.to);
            ans += e.val;
        }
    }
    return ans;
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

### 网络流

```cpp
// 最大流
const int INF = 0x7fffffff;

struct Edge {
    int to, cap;
    Edge(int to, int cap) : to(to), cap(cap) {}
};

struct Dinic {
    int n, s, t;
    vector<Edge> es;
    vector<vector<int> > G;
    vector<int> dist, cur;

    Dinic(int n, int s, int t) : n(n), s(s), t(t), G(n + 1), dist(n + 1), cur(n + 1) {}

    void addEdge(int u, int v, int cap) {
        G[u].push_back(es.size());
        es.emplace_back(v, cap);
        G[v].push_back(es.size());
        es.emplace_back(u, 0);
    }

    bool bfs() {
        dist.assign(n + 1, 0);
        queue<int> q;
        q.push(s);
        dist[s] = 1;
        while (!q.empty()) {
            int now = q.front();
            q.pop();
            for (int i : G[now]) {
                Edge& e = es[i];
                if (!dist[e.to] && e.cap > 0) {
                    dist[e.to] = dist[now] + 1;
                    q.push(e.to);
                }
            }
        }
        return dist[t];
    }

    int dfs(int now, int cap) {
        if (now == t || cap == 0) return cap;
        int tmp = cap, f;
        for (int& i = cur[now]; i < G[now].size(); i++) {
            Edge& e = es[G[now][i]];
            if (dist[e.to] == dist[now] + 1) {
                f = dfs(e.to, min(cap, e.cap));
                e.cap -= f;
                es[G[now][i] ^ 1].cap += f;
                cap -= f;
                if (cap == 0) break;
            }
        }
        return tmp - cap;
    }

    ll solve() {
        ll flow = 0;
        while (bfs()) {
            cur.assign(n + 1, 0);
            flow += dfs(s, INF);
        }
        return flow;
    }
};

// 最小费用流
const int INF = 0x7fffffff;

struct Edge {
    int from, to, cap, cost;
    Edge(int from, int to, int cap, int cost) : from(from), to(to), cap(cap), cost(cost) {}
};

struct MCMF {
    int n, s, t, flow, cost;
    vector<Edge> es;
    vector<vector<int> > G;
    vector<int> d, p, a;  // dist, prev, add
    deque<bool> in;

    MCMF(int n, int s, int t) : n(n), s(s), t(t), flow(0), cost(0), G(n + 1), p(n + 1), a(n + 1) {}

    void addEdge(int u, int v, int cap, int cost) {
        G[u].push_back(es.size());
        es.emplace_back(u, v, cap, cost);
        G[v].push_back(es.size());
        es.emplace_back(v, u, 0, -cost);
    }

    bool spfa() {
        d.assign(n + 1, INF);
        in.assign(n + 1, false);
        d[s] = 0;
        in[s] = 1;
        a[s] = INF;
        queue<int> q;
        q.push(s);
        while (!q.empty()) {
            int now = q.front();
            q.pop();
            in[now] = false;
            for (int& i : G[now]) {
                Edge& e = es[i];
                if (e.cap && d[e.to] > d[now] + e.cost) {
                    d[e.to] = d[now] + e.cost;
                    p[e.to] = i;
                    a[e.to] = min(a[now], e.cap);
                    if (!in[e.to]) {
                        q.push(e.to);
                        in[e.to] = true;
                    }
                }
            }
        }
        return d[t] != INF;
    }

    void solve() {
        while (spfa()) {
            flow += a[t];
            cost += a[t] * d[t];
            int u = t;
            while (u != s) {
                es[p[u]].cap -= a[t];
                es[p[u] ^ 1].cap += a[t];
                u = es[p[u]].from;
            }
        }
    }
};
```

### 树链剖分

```cpp
// 点权
vector<int> G[MAXN];
int pa[MAXN], sz[MAXN], dep[MAXN], dfn[MAXN], maxc[MAXN], top[MAXN];

void dfs1(int now) {
    sz[now] = 1;
    maxc[now] = -1;
    int maxs = 0;
    for (int& nxt : G[now]) {
        if (nxt != pa[now]) {
            pa[nxt] = now;
            dep[nxt] = dep[now] + 1;
            dfs1(nxt);
            sz[now] += sz[nxt];
            if (updmax(maxs, sz[nxt])) maxc[now] = nxt;
        }
    }
}

void dfs2(int now, int tp) {
    static int cnt = 0;
    top[now] = tp;
    dfn[now] = ++cnt;
    if (maxc[now] != -1) dfs2(maxc[now], tp);
    for (int& nxt : G[now]) {
        if (nxt != pa[now] && nxt != maxc[now]) {
            dfs2(nxt, nxt);
        }
    }
}

void init() {
    dep[1] = 1;
    dfs1(1);
    dfs2(1, 1);
}

ll go(int u, int v) {
    int uu = top[u], vv = top[v];
    ll res = 0;
    while (uu != vv) {
        if (dep[uu] < dep[vv]) {
            swap(u, v);
            swap(uu, vv);
        }
        res += segt.query(dfn[uu], dfn[u]);
        u = pa[uu];
        uu = top[u];
    }
    if (dep[u] > dep[v]) swap(u, v);
    res += segt.query(dfn[u], dfn[v]);
    return res;
}
```
