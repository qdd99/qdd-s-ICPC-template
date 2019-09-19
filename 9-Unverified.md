## 待验证

**版权归原作者所有 部分代码有风格调整 不保证内容的正确性**

### 最长上升子序列

```cpp
// Chestnut
S[d[i] = lower_bound(S, S + i, a[i] - 1) - S] = min(S[d[i]], a[i]);
ans = max(ans, d[i]);
```

### Link-Cut Tree

```cpp
// Chestnut
const int MAXN = 50005;

#define lc son[x][0]
#define rc son[x][1]

struct Splay {
    int fa[MAXN], son[MAXN][2];
    int st[MAXN];
    bool rev[MAXN];
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

### 后缀自动机

```cpp
// Chestnut
char s[50100];

struct samnode {
    samnode *par, *ch[26];
    int val;
    samnode() {
        par = 0;
        memset(ch, 0, sizeof(ch));
        val = 0;
    }
} node[100100], *root, *last;

int size = 0;

inline void init() { last = root = &node[0]; }

inline void add(int c) {
    samnode *p = last;
    samnode *np = &node[++size];
    np->val = p->val + 1;
    while (p && !p->ch[c])
        p->ch[c] = np, p = p->par;
    if (!p) np->par = root;
    else {
        samnode *q = p->ch[c];
        if (q->val == p->val + 1)
            np->par = q;
        else {
            samnode *nq = &node[++size];
            nq->val = p->val + 1;
            memcpy(nq->ch, q->ch, sizeof(q->ch));
            nq->par = q->par;
            q->par = np->par = nq;
            while (p && p->ch[c] == q)
                p->ch[c] = nq, p = p->par;
        }
    }
    last = np;
}

int main() {
    init();
    scanf("%s", s);
    int n = strlen(s), ans = 0;
    for (int i = 0; i < n; i++) add(s[i] - 'A');
    for (int i = 1; i <= size; i++) ans += node[i].val - node[i].par->val;
    printf("%d\n", ans);
    return 0;
}
```

+ 广义后缀自动机

```cpp
// Chestnut
int v[100005], head[100005], tot, d[100005];

struct node {
    node *fa, *go[11];
    int max;
} *root, pool[4000005], *cnt;

struct edge {
    int go, next;
} e[100005];

void add(int x, int y) {
    e[++tot] = (edge){y, head[x]}; head[x] = tot;
    e[++tot] = (edge){x, head[y]}; head[y] = tot;
}

void init() { cnt = root = pool + 1; }

node *newnode(int _val) {
    (++cnt)->max = _val;
    return cnt;
}

ostream& operator , (ostream& os, int a) {}

node *extend(node *p, int c) {
    node *np = newnode(p->max + 1);
    while (p && !p->go[c]) p->go[c] = np, p = p->fa;
    if (!p) np->fa = root;
    else {
        node *q = p->go[c];
        if (p->max + 1 == q->max) np->fa = q;
        else {
            node *nq = newnode(p->max + 1);
            memcpy(nq->go, q->go, sizeof q->go);
            nq->fa = q->fa;
            np->fa = q->fa = nq;
            while (p && p->go[c] == q) p->go[c] = nq, p = p->fa;
        }
    }
    return np;
}

ll solve() {
    ll ans = 0;
    for (node *i = root + 1; i <= cnt; i++)
        ans += i->max - i->fa->max;
    return ans;
}

void dfs(int x, int fa, node *p) {
    node *t = extend(p, v[x]);
    for (int i = head[x]; i; i = e[i].next)
        if (e[i].go != fa)
            dfs(e[i].go, x, t);
}

int n, c, x, y;

int main() {
    init();
    scanf("%d%d", &n, &c);
    for (int i = 1; i <= n; i++) scanf("%d", &v[i]);
    for (int i = 1; i < n; i++) {
        scanf("%d%d", &x, &y);
        add(x, y);
        d[x]++, d[y]++;
    }
    for (int i = 1; i <= n; i++)
        if (d[i] == 1) dfs(i, 0, pool + 1);
    printf("%lld", solve());
}
```

### 任意模数 NTT

```cpp
// memset0
const int MAXN = 4e5 + 10, G = 3, P[3] = {469762049, 998244353, 1004535809};
int n1, n2, k, n, p, p1, p2, M2;
int a[MAXN], b[MAXN], f[3][MAXN], g[MAXN], rev[MAXN], ans[MAXN];

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
ld closepoint(vector<V>& A, int l, int r) { // 最近点对, 先要按照 x 坐标排序 
    if (r - l <= 5) {
        ld ans = 1e20;
        for (int i = l; i <= r; i++)
            for (int j = i + 1; j <= r; j++) ans = min(ans, A[i].dis(A[j]));
        return ans;
    }
    int mid = l + r >> 1;
    ld ans = min(closepoint(A, l, mid), closepoint(A, mid + 1, r));
    vector<V> B;
    for (int i = l; i <= r; i++)
        if (abs(A[i].x - A[mid].x) <= ans) B.push_back(A[i]);
    sort(B.begin(), B.end(), [](V k1, V k2) {
        return k1.y < k2.y;
    });
    for (int i = 0; i < B.size(); i++)
        for (int j = i + 1; j < B.size() && B[j].y - B[i].y < ans; j++) ans = min(ans, B[i].dis(B[j]));
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

min_25 杜教筛 高斯消元 伯努利数和等幂求和 单纯形 数论分块

**图论**

zkw费用流 树上点分治 二分图匹配 虚树 欧拉路径 一般图匹配 点双连通分量/广义圆方树
圆方树 最小树形图 三元环、四元环

**计算几何**

圆与多边形交 圆的离散化、面积并 圆的反演 三维计算几何 旋转 线、面 凸包

**杂项**

数位DP

+ kuangbin

**数学**

整数拆分 求A^B的约数之和对MOD取模 斐波那契数列取模循环节 大区间素数筛选

**图论**

次小生成树 生成树计数 2-SAT 曼哈顿最小生成树
