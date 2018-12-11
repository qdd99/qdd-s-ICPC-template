## 4.2 图论

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
