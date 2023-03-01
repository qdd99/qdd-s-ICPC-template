## Graph Theory

### Shortest Path

+ Dijkstra

```cpp
struct Dijkstra {
  struct Edge {
    int to, val;
    Edge(int to = 0, int val = 0) : to(to), val(val) {}
  };

  int n;
  vector<vector<Edge>> g;

  Dijkstra(int n) : n(n), g(n) {}

  void add_edge(int u, int v, int val) { g[u].emplace_back(v, val); }

  vector<i64> solve(int s) {
    using pii = pair<i64, int>;
    vector<i64> dis(n, 1LL << 60);
    priority_queue<pii, vector<pii>, greater<pii>> q;
    dis[s] = 0;
    q.emplace(0, s);
    while (!q.empty()) {
      pii p = q.top();
      q.pop();
      int u = p.second;
      if (dis[u] < p.first) continue;
      for (Edge& e : g[u]) {
        int v = e.to;
        if (dis[v] > dis[u] + e.val) {
          dis[v] = dis[u] + e.val;
          q.emplace(dis[v], v);
        }
      }
    }
    return dis;
  }
};
```

+ Bellman-Ford

```cpp
struct SPFA {
  struct Edge {
    int to, val;
    Edge(int to = 0, int val = 0) : to(to), val(val) {}
  };

  int n;
  vector<vector<Edge>> g;

  SPFA(int n) : n(n), g(n) {}

  void add_edge(int u, int v, int val) { g[u].emplace_back(v, val); }

  vector<i64> solve(int s) {
    queue<int> q;
    vector<i64> dis(n, 1LL << 60);
    vector<bool> in(n, 0);
    q.push(s);
    dis[s] = 0;
    in[s] = 1;
    while (!q.empty()) {
      int u = q.front();
      q.pop();
      in[u] = 0;
      for (Edge& e : g[u]) {
        int v = e.to;
        if (dis[v] > dis[u] + e.val) {
          dis[v] = dis[u] + e.val;
          if (!in[v]) {
            in[v] = 1;
            q.push(v);
          }
        }
      }
    }
    return dis;
  }
};
```

+ Floyd–Warshall, with Shortest Cycle

```cpp
// Note: INF should not exceed 1/3 LLONG_MAX
for (int k = 0; k < n; k++) {
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < i; j++) {
      ans = min(ans, g[i][k] + g[k][j] + dis[i][j]);
    }
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      dis[i][j] = min(dis[i][j], dis[i][k] + dis[k][j]);
    }
  }
}
```

### Topological Sorting

```cpp
// Use max/min heap for lexicographically largest/smallest
int n, deg[N], dis[N];
vector<int> g[N];

bool topo(vector<int>& ans) {
  queue<int> q;
  for (int i = 1; i <= n; i++) {
    if (deg[i] == 0) {
      q.push(i);
      dis[i] = 1;
    }
  }
  while (!q.empty()) {
    int u = q.front();
    q.pop();
    ans.push_back(u);
    for (int v : g[u]) {
      deg[v]--;
      dis[v] = max(dis[v], dis[u] + 1);
      if (deg[v] == 0) q.push(v);
    }
  }
  return ans.size() == n;
}
```

### Minimum Spanning Tree

```cpp
// Prerequisite: Disjoint Set Union
struct Edge {
  int u, v, w;
  Edge(int u = 0, int v = 0, int w = 0) : u(u), v(v), w(w) {}
};

i64 kruskal(vector<Edge>& es, int n) {
  sort(es.begin(), es.end(), [](Edge& x, Edge& y) { return x.w < y.w; });
  dsu d(n + 1);
  i64 ans = 0;
  for (Edge& e : es) {
    if (d.merge(e.u, e.v)) {
      ans += e.w;
    }
  }
  return ans;
}
```

### Lowest Common Ancestor

```cpp
struct LCA {
  int n, log;
  vector<vector<int>> g;
  vector<int> dep;
  vector<vector<int>> up;
  LCA(int n)
      : n(n), log(__lg(n) + 1), g(n), dep(n), up(n, vector<int>(log, -1)) {}

  void add_edge(int u, int v) {
    g[u].push_back(v);
    g[v].push_back(u);
  }

  void dfs(int u, int p) {
    up[u][0] = p;
    for (int i = 1; i < log; i++) {
      up[u][i] = (up[u][i - 1] == -1 ? -1 : up[up[u][i - 1]][i - 1]);
    }
    for (int v : g[u]) {
      if (v == p) continue;
      dep[v] = dep[u] + 1;
      dfs(v, u);
    }
  }

  void build(int root = 0) { dfs(root, -1); }

  int lca(int u, int v) {
    if (dep[u] < dep[v]) swap(u, v);
    for (int i = log - 1; i >= 0; i--) {
      if (dep[u] - (1 << i) >= dep[v]) u = up[u][i];
    }
    if (u == v) return u;
    for (int i = log - 1; i >= 0; i--) {
      if (up[u][i] != up[v][i]) u = up[u][i], v = up[v][i];
    }
    return up[u][0];
  }
};
```

### Network Flow

+ Max Flow

```cpp
const int INF = 0x7fffffff;

struct Dinic {
  struct Edge {
    int to, cap;
    Edge(int to, int cap) : to(to), cap(cap) {}
  };

  int n, s, t;
  vector<Edge> es;
  vector<vector<int>> g;
  vector<int> dis, cur;

  Dinic(int n, int s, int t) : n(n), s(s), t(t), g(n), dis(n), cur(n) {}

  void add_edge(int u, int v, int cap) {
    g[u].push_back(es.size());
    es.emplace_back(v, cap);
    g[v].push_back(es.size());
    es.emplace_back(u, 0);
  }

  bool bfs() {
    dis.assign(n, 0);
    queue<int> q;
    q.push(s);
    dis[s] = 1;
    while (!q.empty()) {
      int u = q.front();
      q.pop();
      for (int i : g[u]) {
        Edge& e = es[i];
        if (!dis[e.to] && e.cap > 0) {
          dis[e.to] = dis[u] + 1;
          q.push(e.to);
        }
      }
    }
    return dis[t];
  }

  int dfs(int u, int cap) {
    if (u == t || cap == 0) return cap;
    int tmp = cap;
    for (int& i = cur[u]; i < (int)g[u].size(); i++) {
      Edge& e = es[g[u][i]];
      if (dis[e.to] == dis[u] + 1) {
        int f = dfs(e.to, min(cap, e.cap));
        e.cap -= f;
        es[g[u][i] ^ 1].cap += f;
        cap -= f;
        if (cap == 0) break;
      }
    }
    return tmp - cap;
  }

  i64 solve() {
    i64 flow = 0;
    while (bfs()) {
      cur.assign(n, 0);
      flow += dfs(s, INF);
    }
    return flow;
  }
};
```

+ Minimum Cost Flow

```cpp
const i64 INF = 1e15;

struct MCMF {
  struct Edge {
    int from, to;
    i64 cap, cost;
    Edge(int from, int to, i64 cap, i64 cost) : from(from), to(to), cap(cap), cost(cost) {}
  };

  int n, s, t;
  i64 flow, cost;
  vector<Edge> es;
  vector<vector<int>> g;
  vector<i64> d, a;  // dis, add, prev
  vector<int> p, in;

  MCMF(int n, int s, int t) : n(n), s(s), t(t), flow(0), cost(0), g(n), p(n), a(n) {}

  void add_edge(int u, int v, i64 cap, i64 cost) {
    g[u].push_back(es.size());
    es.emplace_back(u, v, cap, cost);
    g[v].push_back(es.size());
    es.emplace_back(v, u, 0, -cost);
  }

  bool spfa() {
    d.assign(n, INF);
    in.assign(n, 0);
    d[s] = 0;
    in[s] = 1;
    a[s] = INF;
    queue<int> q;
    q.push(s);
    while (!q.empty()) {
      int u = q.front();
      q.pop();
      in[u] = 0;
      for (int& i : g[u]) {
        Edge& e = es[i];
        if (e.cap && d[e.to] > d[u] + e.cost) {
          d[e.to] = d[u] + e.cost;
          p[e.to] = i;
          a[e.to] = min(a[u], e.cap);
          if (!in[e.to]) {
            q.push(e.to);
            in[e.to] = 1;
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

### Minimum Cut of Undirected Graph

```cpp
namespace stoer_wagner {
  bool vis[N], in[N];
  int g[N][N], w[N];

  void init() {
    memset(g, 0, sizeof(g));
    memset(in, 0, sizeof(in));
  }

  void add_edge(int u, int v, int w) {
    g[u][v] += w;
    g[v][u] += w;
  }

  int search(int& s, int& t) {
    memset(vis, 0, sizeof(vis));
    memset(w, 0, sizeof(w));
    int maxw, tt = n + 1;
    for (int i = 0; i < n; i++) {
      maxw = -INF;
      for (int j = 0; j < n; j++) {
        if (!in[j] && !vis[j] && w[j] > maxw) {
          maxw = w[j];
          tt = j;
        }
      }
      if (t == tt) return w[t];
      s = t; t = tt;
      vis[tt] = true;
      for (int j = 0; j < n; j++) {
        if (!in[j] && !vis[j]) {
          w[j] += g[tt][j];
        }
      }
    }
    return w[t];
  }

  int go() {
    int s, t, ans = INF;
    for (int i = 0; i < n - 1; i++) {
      s = t = -1;
      ans = min(ans, search(s, t));
      if (ans == 0) return 0;
      in[t] = true;
      for (int j = 0; j < n; j++) {
        if (!in[j]) {
          g[s][j] += g[t][j];
          g[j][s] += g[j][t];
        }
      }
    }
    return ans;
  }
}
```

### Heavy-Light Decomposition

```cpp
// jiangly
struct HLD {
  int n;
  vector<int> siz, top, dep, parent, in, out, seq;
  vector<vector<int>> adj;
  int cur;

  explicit HLD(int n) : n(n), siz(n), top(n), dep(n), parent(n), in(n), out(n), seq(n), adj(n), cur(0) {}

  void addEdge(int u, int v) {
    adj[u].push_back(v);
    adj[v].push_back(u);
  }

  void work(int root = 0) {
    top[root] = root;
    dep[root] = 0;
    parent[root] = -1;
    dfs1(root);
    dfs2(root);
  }

  void dfs1(int u) {
    if (parent[u] != -1) {
      adj[u].erase(find(adj[u].begin(), adj[u].end(), parent[u]));
    }
    siz[u] = 1;
    for (auto& v : adj[u]) {
      parent[v] = u;
      dep[v] = dep[u] + 1;
      dfs1(v);
      siz[u] += siz[v];
      if (siz[v] > siz[adj[u][0]]) {
        swap(v, adj[u][0]);
      }
    }
  }

  void dfs2(int u) {
    in[u] = cur++;
    seq[in[u]] = u;
    for (auto v : adj[u]) {
      top[v] = v == adj[u][0] ? top[u] : v;
      dfs2(v);
    }
    out[u] = cur;
  }

  int lca(int u, int v) {
    while (top[u] != top[v]) {
      if (dep[top[u]] > dep[top[v]]) {
        u = parent[top[u]];
      } else {
        v = parent[top[v]];
      }
    }
    return dep[u] < dep[v] ? u : v;
  }

  int dist(int u, int v) { return dep[u] + dep[v] - 2 * dep[lca(u, v)]; }

  int jump(int u, int k) {
    if (dep[u] < k) {
      return -1;
    }
    int d = dep[u] - k;
    while (dep[top[u]] > d) {
      u = parent[top[u]];
    }
    return seq[in[u] - dep[u] + d];
  }

  bool isAncester(int u, int v) { return in[u] <= in[v] && in[v] < out[u]; }

  int rootedParent(int u, int v) {
    swap(u, v);
    if (u == v) {
      return u;
    }
    if (!isAncester(u, v)) {
      return parent[u];
    }
    auto it = upper_bound(adj[u].begin(), adj[u].end(), v, [&](int x, int y) { return in[x] < in[y]; }) - 1;
    return *it;
  }

  int rootedSize(int u, int v) {
    if (u == v) {
      return n;
    }
    if (!isAncester(v, u)) {
      return siz[v];
    }
    return n - siz[rootedParent(u, v)];
  }

  int rootedLca(int a, int b, int c) { return lca(a, b) ^ lca(b, c) ^ lca(c, a); }

  // [u, v] if inclusive is true, otherwise [u, v)
  vector<pair<int, int>> path(int u, int v, bool inclusive) {
    assert(isAncester(v, u));
    vector<pair<int, int>> res;
    while (top[u] != top[v]) {
      res.emplace_back(u, top[u]);
      u = parent[top[u]];
    }
    if (inclusive) {
      res.emplace_back(u, v);
    } else if (u != v) {
      res.emplace_back(u, seq[in[v] + 1]);
    }
    return res;
  }
};
```

### Tree Hash

```cpp
const uint64_t mask = chrono::steady_clock::now().time_since_epoch().count();

uint64_t f(uint64_t x) {
  x ^= mask;
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  x ^= mask;
  return x;
}

uint64_t dfs(int u, int p) {
  uint64_t h = 1;
  for (int v : g[u]) {
    if (v == p) continue;
    h += f(dfs(v, u));
  }
  return h;
}
```

### Tarjan

+ Cut Points

```cpp
int dfn[N], low[N], clk;

void init() { clk = 0; memset(dfn, 0, sizeof(dfn)); }

void tarjan(int u, int pa) {
  low[u] = dfn[u] = ++clk;
  int cc = (pa != 0);
  for (int v : g[u]) {
    if (v == pa) continue;
    if (!dfn[v]) {
      tarjan(v, u);
      low[u] = min(low[u], low[v]);
      cc += low[v] >= dfn[u];
    } else low[u] = min(low[u], dfn[v]);
  }
  if (cc > 1) // ...
}
```

+ Bridges

```cpp
int dfn[N], low[N], clk;

void init() { clk = 0; memset(dfn, 0, sizeof(dfn)); }

void tarjan(int u, int pa) {
  low[u] = dfn[u] = ++clk;
  int f = 0;
  for (int v : g[u]) {
    if (v == pa && ++f == 1) continue;
    if (!dfn[v]) {
      tarjan(v, u);
      if (low[v] > dfn[u]) // ...
      low[u] = min(low[u], low[v]);
    } else low[u] = min(low[u], dfn[v]);
  }
}
```

+ Strongly Connected Components (SCC)

```cpp
struct SCC {
  int n, tot;
  vector<vector<int>> g;
  vector<int> color;

  SCC(int n) : n(n), g(n) {}

  void add_edge(int u, int v) { g[u].push_back(v); }

  void work() {
    tot = 0;
    color.assign(n, -1);
    vector<int> dfn(n), low(n), st;
    int clk = 0;
    function<void(int)> dfs = [&](int u) {
      dfn[u] = low[u] = ++clk;
      st.push_back(u);
      for (int v : g[u]) {
        if (!dfn[v]) {
          dfs(v);
          low[u] = min(low[u], low[v]);
        } else if (color[v] == -1) {
          low[u] = min(low[u], dfn[v]);
        }
      }
      if (dfn[u] == low[u]) {
        for (;;) {
          int x = st.back();
          st.pop_back();
          color[x] = tot;
          if (x == u) break;
        }
        tot++;
      }
    };
    for (int i = 0; i < n; i++) {
      if (!dfn[i]) dfs(i);
    }
    for (int& x : color) {
      x = tot - 1 - x;
    }
  }

  vector<vector<int>> scc() {
    vector<vector<int>> res(tot);
    for (int i = 0; i < n; i++) {
      res[color[i]].push_back(i);
    }
    return res;
  }

  vector<vector<int>> dag() {
    vector<vector<int>> res(tot);
    for (int i = 0; i < n; i++) {
      for (int j : g[i]) {
        if (color[i] != color[j]) {
          res[color[i]].push_back(color[j]);
        }
      }
    }
    for (auto& v : res) {
      sort(v.begin(), v.end());
      v.erase(unique(v.begin(), v.end()), v.end());
    }
    return res;
  }
};
```

+ 2-SAT

```cpp
struct two_sat {
  int n;
  SCC scc;

  two_sat(int n) : n(n), scc(n * 2) {}

  void add_clause(int u, bool f, int v, bool g) {
    u = u * 2 + f;
    v = v * 2 + g;
    scc.add_edge(u ^ 1, v);
    scc.add_edge(v ^ 1, u);
  }

  bool solve() {
    scc.work();
    for (int i = 0; i < n; i++) {
      if (scc.color[i * 2] == scc.color[i * 2 + 1]) {
        return false;
      }
    }
    return true;
  }

  vector<bool> answer() {
    vector<bool> res(n);
    for (int i = 0; i < n; i++) {
      res[i] = scc.color[i * 2 + 1] > scc.color[i * 2];
    }
    return res;
  }
};
```

### Eulerian Path

```cpp
struct EulerPath {
  int n;
  vector<vector<int>> g;
  vector<pair<int, int>> es;

  EulerPath(int n) : n(n), g(n) {}

  void add_edge(int u, int v, bool directed = false) {
    g[u].push_back(es.size());
    if (!directed) g[v].push_back(es.size());
    es.emplace_back(u, v);
  }

  vector<int> solve(int s) {
    vector<int> path, ptr(n), st;
    vector<bool> used(es.size());
    st.push_back(s);
    while (!st.empty()) {
      int u = st.back();
      int& i = ptr[u];
      while (i < g[u].size() && used[g[u][i]]) i++;
      if (i == g[u].size()) {
        path.push_back(u);
        st.pop_back();
      } else {
        int e = g[u][i];
        used[e] = 1;
        int v = es[e].first ^ es[e].second ^ u;
        st.push_back(v);
      }
    }
    reverse(path.begin(), path.end());
    return path;
  }
};
```

### Dominator Tree

```cpp
vector<int> g[N], rg[N];
vector<int> dt[N];

namespace tl {
  int pa[N], dfn[N], clk, rdfn[N];
  int c[N], best[N], sdom[N], idom[N];

  void init(int n) {
    clk = 0;
    fill(c, c + n + 1, -1);
    fill(dfn, dfn + n + 1, 0);
    for (int i = 1; i <= n; i++) {
      dt[i].clear();
      sdom[i] = best[i] = i;
    }
  }

  void dfs(int u) {
    dfn[u] = ++clk;
    rdfn[clk] = u;
    for (int& v : g[u]) {
      if (!dfn[v]) {
        pa[v] = u;
        dfs(v);
      }
    }
  }

  int fix(int x) {
    if (c[x] == -1) return x;
    int& f = c[x], rt = fix(f);
    if (dfn[sdom[best[x]]] > dfn[sdom[best[f]]]) best[x] = best[f];
    return f = rt;
  }

  void go(int rt) {
    dfs(rt);
    for (int i = clk; i > 1; i--) {
      int x = rdfn[i], mn = clk + 1;
      for (int& u : rg[x]) {
        if (!dfn[u]) continue; // may not reach all vertices
        fix(u);
        mn = min(mn, dfn[sdom[best[u]]]);
      }
      c[x] = pa[x];
      dt[sdom[x] = rdfn[mn]].push_back(x);
      x = rdfn[i - 1];
      for (int& u: dt[x]) {
        fix(u);
        idom[u] = (sdom[best[u]] == x) ? x : best[u];
      }
      dt[x].clear();
    }
    for (int i = 2; i <= clk; i++) {
      int u = rdfn[i];
      if (idom[u] != sdom[u]) idom[u] = idom[idom[u]];
      dt[idom[u]].push_back(u);
    }
  }
}
```
