## 待验证

**版权归原作者所有 部分代码有风格调整 不保证内容的正确性**

### 最长上升子序列

```cpp
// Chestnut
S[d[i] = lower_bound(S, S + i, a[i] - 1) - S] = min(S[d[i]], a[i]);
ans = max(ans, d[i]);
```

### 约瑟夫问题

```cpp
// n个人，1至m报数，问最后留下来的人的编号
// 公式：f(n,m)=(f(n−1,m)+m)%n，f(0,m)=0;
// O(n)
ll calc(int n, ll m) {
    ll p = 0;
    for (int i = 2; i <= n; i++) {
        p = (p + m) % i;
    }
    return p + 1;
}

// n个人，1至m报数，问第k个出局的人的编号
// 公式：f(n,k)=(f(n−1,k−1)+m−1)%n+1
// f(n−k+1,1)=m%(n−k+1)
// if (f==0) f=n−k+1
// O(k)
ll cal1(ll n, ll m, ll k) {  // (k == n) equal(calc)
    ll p = m % (n - k + 1);
    if (p == 0) p = n - k + 1;
    for (ll i = 2; i <= k; i++) {
        p = (p + m - 1) % (n - k + i) + 1;
    }
    return p;
}

// n个人，1至m报数，问第k个出局的人的编号
// O(m*log(m))
ll cal2(ll n, ll m, ll k) {
    if (m == 1)
        return k;
    else {
        ll a = n - k + 1, b = 1;
        ll c = m % a, x = 0;
        if (c == 0) c = a;
        while (b + x <= k) {
            a += x, b += x, c += m * x;
            c %= a;
            if (c == 0) c = a;
            x = (a - c) / (m - 1) + 1;
        }
        c += (k - b) * m;
        c %= n;
        if (c == 0) c = n;
        return c;
    }
}

// n个人，1至m报数，问编号为k的人是第几个出局的
// O(n)
ll n, k;  //可做n<=4e7,询问个数<=100,下标范围[0,n-1]
ll dieInXturn(int n, int k, int x) {  // n个人，报数k，下标为X的人第几个死亡
    ll tmp = 0;
    while (n) {
        x = (x + n) % n;
        if (k > n) x += (k - x - 1 + n - 1) / n * n;
        if ((x + 1) % k == 0) {
            tmp += (x + 1) / k;
            break;
        } else {
            if (k > n) {
                tmp += x / k;
                ll ttmp = x;
                x = x - (x / n + 1) * (x / k) + (x + n) / n * n - k;
                n -= ttmp / k;
            } else {
                tmp += n / k;
                x = x - x / k;
                x += n - n / k * k;
                n -= n / k;
            }
        }
    }
    return tmp;
}
```

### 二分图最大权匹配KM

```cpp
// ECNU
namespace R {
    int n;
    int w[N][N], kx[N], ky[N], py[N], vy[N], slk[N], pre[N];

    ll go() {
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= n; j++)
                kx[i] = max(kx[i], w[i][j]);
        for (int i = 1; i <= n; i++) {
            fill(vy, vy + n + 1, 0);
            fill(slk, slk + n + 1, INF);
            fill(pre, pre + n + 1, 0);
            int k = 0, p = -1;
            for (py[k = 0] = i; py[k]; k = p) {
                int d = INF;
                vy[k] = 1;
                int x = py[k];
                for (int j = 1; j <= n; j++) {
                    if (!vy[j]) {
                        int t = kx[x] + ky[j] - w[x][j];
                        if (t < slk[j]) { slk[j] = t; pre[j] = k; }
                        if (slk[j] < d) { d = slk[j]; p = j; }
                    }
                }
                for (int j = 0; j <= n; j++) {
                    if (vy[j]) { kx[py[j]] -= d; ky[j] += d; }
                    else slk[j] -= d;
                }
            }
            for (; k; k = pre[k]) py[k] = py[pre[k]];
        }
        ll ans = 0;
        for (int i = 1; i <= n; i++) ans += kx[i] + ky[i];
        return ans;
    }
}
```

### HLPP

```cpp
struct HLPP {
    struct Edge {
        int v, rev;
        ll cap;
    };
    int n, sp, tp, lim, ht, lcnt;
    ll exf[N];
    vector<Edge> G[N];
    vector<int> hq[N], gap[N], h, sum;
    void init(int nn, int s, int t) {
        sp = s, tp = t, n = nn, lim = n + 1, ht = lcnt = 0;
        for (int i = 1; i <= n; ++i) G[i].clear(), exf[i] = 0;
    }
    void add_edge(int u, int v, ll cap) {
        G[u].push_back({v, int(G[v].size()), cap});
        G[v].push_back({u, int(G[u].size()) - 1, 0});
    }
    void update(int u, int nh) {
        ++lcnt;
        if (h[u] != lim) --sum[h[u]];
        h[u] = nh;
        if (nh == lim) return;
        ++sum[ht = nh];
        gap[nh].push_back(u);
        if (exf[u] > 0) hq[nh].push_back(u);
    }
    void relabel() {
        queue<int> q;
        for (int i = 0; i <= lim; ++i) hq[i].clear(), gap[i].clear();
        h.assign(lim, lim), sum.assign(lim, 0), q.push(tp);
        lcnt = ht = h[tp] = 0;
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            for (Edge& e : G[u])
                if (h[e.v] == lim && G[e.v][e.rev].cap) update(e.v, h[u] + 1), q.push(e.v);
            ht = h[u];
        }
    }
    void push(int u, Edge& e) {
        if (!exf[e.v]) hq[h[e.v]].push_back(e.v);
        ll df = min(exf[u], e.cap);
        e.cap -= df, G[e.v][e.rev].cap += df;
        exf[u] -= df, exf[e.v] += df;
    }
    void discharge(int u) {
        int nh = lim;
        if (h[u] == lim) return;
        for (Edge& e : G[u]) {
            if (!e.cap) continue;
            if (h[u] == h[e.v] + 1) {
                push(u, e);
                if (exf[u] <= 0) return;
            } else if (nh > h[e.v] + 1)
                nh = h[e.v] + 1;
        }
        if (sum[h[u]] > 1)
            update(u, nh);
        else {
            for (; ht >= h[u]; gap[ht--].clear())
                for (int& i : gap[ht]) update(i, lim);
        }
    }
    ll hlpp() {
        exf[sp] = INF, exf[tp] = -INF, relabel();
        for (Edge& e : G[sp]) push(sp, e);
        for (; ~ht; --ht) {
            while (!hq[ht].empty()) {
                int u = hq[ht].back();
                hq[ht].pop_back();
                discharge(u);
                if (lcnt > (n << 2)) relabel();
            }
        }
        return exf[tp] + INF;
    }
};
```

### 上下界网络流

```cpp
const int INF = 0x3f3f3f3f;

struct edge {
    int to, cap, rev;
};

const int N = 60003;
const int M = 400003;

struct graph {
    int n, m;
    edge w[M];
    int fr[M];
    int num[N], cur[N], first[N];
    edge e[M];

    void init(int n) {
        this->n = n;
        m = 0;
    }

    void add_edge(int from, int to, int cap) {
        w[++m] = (edge){to, cap};
        num[from]++, fr[m] = from;
        w[++m] = (edge){from, 0};
        num[to]++, fr[m] = to;
    }

    void prepare() {
        first[1] = 1;
        for (int i = 2; i <= n; i++) first[i] = first[i - 1] + num[i - 1];
        for (int i = 1; i < n; i++) num[i] = first[i + 1] - 1;
        num[n] = m;
        for (int i = 1; i <= m; i++) {
            e[first[fr[i]] + (cur[fr[i]]++)] = w[i];

            if (!(i % 2)) {
                e[first[fr[i]] + cur[fr[i]] - 1].rev =
                    first[w[i].to] + cur[w[i].to] - 1;
                e[first[w[i].to] + cur[w[i].to] - 1].rev =
                    first[fr[i]] + cur[fr[i]] - 1;
            }
        }
    }

    int q[N];
    int dist[N];
    int t;

    bool bfs(int s) {
        int l = 1, r = 1;
        q[1] = s;
        memset(dist, -1, (n + 1) * 4);
        dist[s] = 0;
        while (l <= r) {
            int u = q[l++];
            for (int i = first[u]; i <= num[u]; i++) {
                int v = e[i].to;
                if ((dist[v] != -1) || (!e[i].cap)) continue;
                dist[v] = dist[u] + 1;
                if (v == t) return true;
                q[++r] = v;
            }
        }
        return dist[t] != -1;
    }

    int dfs(int u, int flow) {
        if (u == t) return flow;
        for (int& i = cur[u]; i <= num[u]; i++) {
            int v = e[i].to;
            if (!e[i].cap || dist[v] != dist[u] + 1) continue;
            int t = dfs(v, min(flow, e[i].cap));
            if (t) {
                e[i].cap -= t;
                e[e[i].rev].cap += t;
                return t;
            }
        }
        return 0;
    }

    ll dinic(int s, int t) {
        ll ans = 0;
        this->t = t;
        while (bfs(s)) {
            int flow;
            for (int i = 1; i <= n; i++) cur[i] = first[i];
            while (flow = dfs(s, INF)) ans += (ll)flow;
        }
        return ans;
    }
};

struct graph_bounds {
    int in[N];
    int S, T, sum, cur;
    graph g;
    int n;

    void init(int n) {
        this->n = n;
        S = n + 1;
        T = n + 2;
        sum = 0;
        g.init(n + 2);
    }

    void add_edge(int from, int to, int low, int up) {
        g.add_edge(from, to, up - low);
        in[to] += low;
        in[from] -= low;
    }

    void build() {
        for (int i = 1; i <= n; i++)
            if (in[i] > 0)
                g.add_edge(S, i, in[i]), sum += in[i];
            else if (in[i])
                g.add_edge(i, T, -in[i]);
        g.prepare();
    }

    bool canflow() {
        build();
        int flow = g.dinic(S, T);
        return flow >= sum;
    }

    bool canflow(int s, int t) {
        g.add_edge(t, s, INF);
        build();
        for (int i = 1; i <= g.m; i++) {
            edge& e = g.e[i];
            if (e.to == s && e.cap == INF) {
                cur = i;
                break;
            }
        }
        int flow = g.dinic(S, T);
        return flow >= sum;
    }

    int maxflow(int s, int t) {
        if (!canflow(s, t)) return -1;
        return g.dinic(s, t);
    }

    int minflow(int s, int t) {
        if (!canflow(s, t)) return -1;
        edge& e = g.e[cur];
        int flow = INF - e.cap;
        e.cap = g.e[e.rev].cap = 0;
        return flow - g.dinic(t, s);
    }
} g;

void solve() {
    int n = read(), m = read(), s = read(), t = read();
    g.init(n);
    while (m--) {
        int u = read(), v = read(), low = read(), up = read();
        g.add_edge(u, v, low, up);
    }
}
```

### Link-Cut Tree

```cpp
// Chestnut
const int N = 50005;

#define lc son[x][0]
#define rc son[x][1]

struct Splay {
    int fa[N], son[N][2];
    int st[N];
    bool rev[N];
    inline int which(int x) {
        for (int i = 0; i < 2; i++)
            if (son[fa[x]][i] == x) return i;
        return -1;
    }

    inline void pushdown(int x) {
        if (rev[x]) {
            rev[x] ^= 1;
            rev[lc] ^= 1;
            rev[rc] ^= 1;
            swap(lc, rc);
        }
    }
    
    inline void rotate(int x) {
        int f = fa[x], w = which(x) ^ 1, c = son[x][w];
        fa[x] = fa[f];
        if (which(f) != -1) son[fa[f]][which(f)] = x;
        fa[c] = f;
        son[f][w ^ 1] = c;
        fa[f] = x;
        son[x][w] = f;
    }

    inline void splay(int x) {
        int top = 0;
        st[++top] = x;
        for (int i = x; which(i) != -1; i = fa[i]) {
            st[++top] = fa[i];
        }
        for (int i = top; i; i--) pushdown(st[i]);
        while (which(x) != -1) {
            int f = fa[x];
            if (which(f) != -1) {
                if (which(x) ^ which(f)) rotate(x);
                else rotate(f);
            }
            rotate(x);
        }
    }

    void access(int x) {
        int t = 0;
        while (x) {
            splay(x);
            rc = t;
            t = x;
            x = fa[x];
        }
    }

    void rever(int x) {
        access(x);
        splay(x);
        rev[x] ^= 1;
    }

    void link(int x, int y) {
        rever(x);
        fa[x] = y;
        splay(x);
    }

    void cut(int x, int y) {
        rever(x);
        access(y);
        splay(y);
        son[y][0] = fa[x] = 0;
    }

    int find(int x) {
        access(x);
        splay(x);
        int y = x;
        while (son[y][0]) y = son[y][0];
        return y;
    }
} T;

int n, m;

int main() {
    char ch[10];
    int x, y;
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= m; i++) {
        scanf("%s", ch);
        scanf("%d%d", &x, &y);
        if (ch[0] == 'C') T.link(x, y);
        else if (ch[0] == 'D') T.cut(x, y);
        else {
            if (T.find(x) == T.find(y)) printf("Yes\n");
            else printf("No\n");
        }
    }
}
```

### 任意模数 NTT

```cpp
// memset0
const int N = 4e5 + 10, G = 3, P[3] = {469762049, 998244353, 1004535809};
int n1, n2, k, n, p, p1, p2, M2;
int a[N], b[N], f[3][N], g[N], rev[N], ans[N];

void ntt(int *a, int g, int p) {
    for (int i = 0; i < n; i++) if (i < rev[i]) swap(a[i], a[rev[i]]);
    for (int len = 1; len < n; len <<= 1) {
        int wn = qk(g, (p - 1) / (len << 1), p);
        for (int i = 0; i < n; i += (len << 1)) {
            int w = 1;
            for (int j = 0; j < len; j++, w = (ll)w * wn % p) {
                int x = a[i + j], y = (ll)w * a[i + j + len] % p;
                a[i + j] = (x + y) % p, a[i + j + len] = (x - y + p) % p;
            }
        }
    }
}

int merge(int a1, int a2, int A2) {
    ll M1 = (ll)p1 * p2;
    ll A1 = ((ll)inv(p2, p1) * a1 % p1 * p2 + (ll)inv(p1, p2) * a2 % p2 * p1) % M1;
    ll K = ((A2 - A1) % M2 + M2) % M2 * inv(M1 % M2, M2) % M2;
    int ans = (A1 + M1 % p * K) % p;
    return ans;
}

void go() {
    read(n1), read(n2), read(p);
    p1 = P[0], p2 = P[1], M2 = P[2];
    for (int i = 0; i <= n1; i++) read(a[i]);
    for (int i = 0; i <= n2; i++) read(b[i]);
    n = 1; while (n <= (n1 + n2)) n <<= 1, ++k;
    for (int i = 0; i < n; i++) {
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (k - 1));
    }
    for (int k = 0; k < 3; k++) {
        for (int i = 0; i < n; i++) f[k][i] = a[i] % P[k];
        for (int i = 0; i < n; i++) g[i] = b[i] % P[k];
        ntt(f[k], G, P[k]), ntt(g, G, P[k]);
        for (int i = 0; i < n; i++) f[k][i] = (ll)f[k][i] * g[i] % P[k];
        ntt(f[k], inv(G, P[k]), P[k]);
        for (int i = 0; i < n; i++) f[k][i] = (ll)f[k][i] * inv(n, P[k]) % P[k];
    }
    for (int i = 0; i <= n1 + n2; i++) ans[i] = merge(f[0][i], f[1][i], f[2][i]);
}
```

### 计算几何

```cpp
// 经纬度球面最短距离
// Voleking
ld Dist(ld la1, ld lo1, ld la2, ld lo2, ld R) {
    la1 *= PI / 180, lo1 *= PI / 180, la2 *= PI / 180, lo2 *= PI / 180;
    ld x1 = cos(la1) * sin(lo1), y1 = cos(la1) * cos(lo1), z1 = sin(la1); 
    ld x2 = cos(la2) * sin(lo2), y2 = cos(la2) * cos(lo2), z1 = sin(la2); 
    return R * acos(x1 * x2 + y1 * y2 + z1 * z2);
}

// jiry_2
int cmp(ld k1, ld k2) {
    return sgn(k1 - k2);
}
V proj(V k1, V k2, V q) { // q 到直线 k1,k2 的投影 
    V k = k2 - k1;
    return k1 + k * (dot(q - k1, k) / k.abs2());
}
V reflect(V k1, V k2, V q) {
    return proj(k1, k2, q) * 2 - q;
}
int clockwise(V k1, V k2, V k3) { // k1 k2 k3 逆时针 1 顺时针 -1 否则 0  
    return sgn(det(k2 - k1, k3 - k1));
}
int checkLL(V k1, V k2, V k3, V k4) { // 求直线 (L) 线段 (S) k1,k2 和 k3,k4 的交点 
    return cmp(det(k3 - k1, k4 - k1), det(k3 - k2, k4 - k2)) != 0;
}
V getLL(V k1, V k2, V k3, V k4) {
    ld w1 = det(k1 - k3, k4 - k3), w2 = det(k4 - k3, k2 - k3);
    return (k1 * w2 + k2 * w1) / (w1 + w2);
}
vector<line> getHL(vector<line>& L) { // 求半平面交, 半平面是逆时针方向, 输出按照逆时针
    sort(L.begin(), L.end());
    deque<line> q;
    for (int i = 0; i < (int) L.size(); i++) {
        if (i && sameDir(L[i], L[i - 1])) continue;
        while (q.size() > 1 && !checkpos(q[q.size() - 2], q[q.size() - 1], L[i])) q.pop_back();
        while (q.size() > 1 && !checkpos(q[1], q[0], L[i])) q.pop_front();
        q.push_back(L[i]);
    }
    while (q.size() > 2 && !checkpos(q[q.size() - 2], q[q.size() - 1], q[0])) q.pop_back();
    while (q.size() > 2 && !checkpos(q[1], q[0], q[q.size() - 1])) q.pop_front();
    vector<line> ans;
    for (int i = 0; i < q.size(); i++) ans.push_back(q[i]);
    return ans;
}
int checkposCC(circle k1, circle k2) { // 返回两个圆的公切线数量
    if (cmp(k1.r, k2.r) == -1) swap(k1, k2);
    ld dis = k1.o.dis(k2.o);
    int w1 = cmp(dis, k1.r + k2.r), w2 = cmp(dis, k1.r - k2.r);
    if (w1 > 0) return 4;
    else if (w1 == 0) return 3;
    else if (w2 > 0) return 2;
    else if (w2 == 0) return 1;
    else return 0;
}
vector<V> getCL(circle k1, V k2, V k3) { // 沿着 k2->k3 方向给出, 相切给出两个 
    V k = proj(k2, k3, k1.o);
    ld d = k1.r * k1.r - (k - k1.o).abs2();
    if (sgn(d) == -1) return {};
    V del = (k3 - k2).unit() * sqrt(max((ld) 0.0, d));
    return {k - del, k + del};
}
vector<line> TangentoutCC(circle k1, circle k2) {
    int pd = checkposCC(k1, k2);
    if (pd == 0) return {};
    if (pd == 1) {
        V k = getCC(k1, k2)[0];
        return { (line){k, k} };
    }
    if (cmp(k1.r, k2.r) == 0) {
        V del = (k2.o - k1.o).unit().turn90().getdel();
        return {
            (line){k1.o - del * k1.r, k2.o - del * k2.r},
            (line){k1.o + del * k1.r, k2.o + del * k2.r}
        };
    } else {
        V p = (k2.o * k1.r - k1.o * k2.r) / (k1.r - k2.r);
        vector<V> A = TangentCP(k1, p), B = TangentCP(k2, p);
        vector<line> ans;
        for (int i = 0; i < A.size(); i++) ans.push_back((line){A[i], B[i]});
        return ans;
    }
}
vector<line> TangentinCC(circle k1, circle k2) {
    int pd = checkposCC(k1, k2);
    if (pd <= 2) return {};
    if (pd == 3) {
        V k = getCC(k1, k2)[0];
        return { (line){k, k} };
    }
    V p = (k2.o * k1.r + k1.o * k2.r) / (k1.r + k2.r);
    vector<V> A = TangentCP(k1, p), B = TangentCP(k2, p);
    vector<line> ans;
    for (int i = 0; i < A.size(); i++) ans.push_back((line){A[i], B[i]});
    return ans;
}
vector<line> TangentCC(circle k1, circle k2) {
    int flag = 0;
    if (k1.r < k2.r) swap(k1, k2), flag = 1;
    vector<line> A = TangentoutCC(k1, k2), B = TangentinCC(k1, k2);
    for (line k: B) A.push_back(k);
    if (flag) for (line& k: A) swap(k[0], k[1]);
    return A;
}
ld convexDiameter(vector<V> A) {
    int now = 0, n = A.size();
    ld ans = 0;
    for (int i = 0; i < A.size(); i++) {
        now = max(now, i);
        while (1) {
            ld k1 = A[i].dis(A[now % n]), k2 = A[i].dis(A[(now + 1) % n]);
            ans = max(ans, max(k1, k2));
            if (k2 > k1) now++;
            else break;
        }
    }
    return ans;
}
vector<V> convexcut(vector<V> A, V k1, V k2) { // 保留 k1,k2,p 逆时针的所有点
    int n = A.size();
    A.push_back(A[0]);
    vector<V> ans;
    for (int i = 0; i < n; i++) {
        int w1 = clockwise(k1, k2, A[i]), w2 = clockwise(k1, k2, A[i + 1]);
        if (w1 >= 0) ans.push_back(A[i]);
        if (w1 * w2 < 0) ans.push_back(getLL(k1, k2, A[i], A[i + 1]));
    }
    return ans;
}
```

### 本模板未涉及的专题

+ ECNU

**数据结构**

均摊复杂度线段树 K-DTree 树状数组套主席树 左偏树 Treap-序列 可回滚并查集 舞蹈链 笛卡尔树 莫队

**数学**

min_25 杜教筛 伯努利数和等幂求和 单纯形 数论分块

**图论**

zkw费用流 树上点分治 二分图匹配 虚树 欧拉路径 一般图匹配 点双连通分量/广义圆方树
圆方树 最小树形图 三元环、四元环

**计算几何**

圆与多边形交 圆的离散化、面积并 圆的反演 三维计算几何 旋转 线、面 凸包

+ kuangbin

**数学**

整数拆分 求A^B的约数之和对MOD取模 斐波那契数列取模循环节

**图论**

次小生成树 生成树计数 曼哈顿最小生成树
