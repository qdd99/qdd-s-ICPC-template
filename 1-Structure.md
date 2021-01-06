## 数据结构

### 并查集

```cpp
struct dsu {
    vector<int> p;
    dsu(int n) : p(n) { iota(p.begin(), p.end(), 0); }
    int find(int x) { return (x == p[x]) ? x : p[x] = find(p[x]); }
    void merge(int a, int b) { p[find(a)] = find(b); }
};
```

+ 动态开点并查集

```cpp
// pa 为负数表示集合大小
unordered_map<int, int> pa;

void _set(int x) { if (!pa.count(x)) pa[x] = -1; }
int find(int x) { return (pa[x] < 0) ? x : pa[x] = find(pa[x]); }

void merge(int a, int b) {
    int x = find(a), y = find(b);
    if (x == y) return;
    if (pa[x] > pa[y]) swap(x, y);
    pa[x] += pa[y];
    pa[y] = x;
}
```

### RMQ

+ 一维

```cpp
// 下标从0开始
struct RMQ {
    int st[MAXN][22]; // 22 = ((int)log2(MAXN) + 1)

    int xlog(int x) { return 31 - __builtin_clz(x); }

    void init(int *a, int n) {
        for (int i = 0; i < n; i++) {
            st[i][0] = a[i];
        }
        for (int j = 1; (1 << j) <= n; j++) {
            for (int i = 0; i + (1 << j) - 1 < n; i++) {
                st[i][j] = max(st[i][j - 1], st[i + (1 << (j - 1))][j - 1]);
            }
        }
    }

    int query(int l, int r) {
        int x = xlog(r - l + 1);
        return max(st[l][x], st[r - (1 << x) + 1][x]);
    }
};
```

+ 二维

```cpp
struct RMQ {
    int st[11][11][MAXN][MAXN]; // 11 = ((int)log2(MAXN) + 1)

    int xlog(int x) { return 31 - __builtin_clz(x); }

    void init(int n, int m) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                st[0][0][i][j] = a[i][j];
            }
        }
        for (int i = 0; (1 << i) <= n; i++) {
            for (int j = 0; (1 << j) <= m; j++) {
                if (i == 0 && j == 0) continue;
                for (int r = 0; r + (1 << i) - 1 < n; r++) {
                    for (int c = 0; c + (1 << j) - 1 < m; c++) {
                        if (i == 0) {
                            st[i][j][r][c] = max(st[i][j - 1][r][c], st[i][j - 1][r][c + (1 << (j - 1))]);
                        } else {
                            st[i][j][r][c] = max(st[i - 1][j][r][c], st[i - 1][j][r + (1 << (i - 1))][c]);
                        }
                    }
                }
            }
        }
    }

    int query(int r1, int c1, int r2, int c2) {
        int x = xlog(r2 - r1 + 1);
        int y = xlog(c2 - c1 + 1);
        int m1 = st[x][y][r1][c1];
        int m2 = st[x][y][r1][c2 - (1 << y) + 1];
        int m3 = st[x][y][r2 - (1 << x) + 1][c1];
        int m4 = st[x][y][r2 - (1 << x) + 1][c2 - (1 << y) + 1];
        return max({m1, m2, m3, m4});
    }
};
```

+ 滑动窗口 RMQ

```cpp
// k 为滑动窗口的大小
deque<int> q;
for (int i = 0, j = 0; i + k <= n; i++) {
    while (j < i + k) {
        while (!q.empty() && a[q.back()] < a[j]) q.pop_back(); // 最小值取'>'号
        q.push_back(j++);
    }
    while (q.front() < i) q.pop_front();
    rmq.push_back(a[q.front()]);
}
```

### 树状数组

+ 单点修改，区间和

```cpp
// 支持第k大的BIT
// 下标从1开始
struct Tbit {
    int size;
    ll t[MAXN];

    int lowbit(int x) { return x & (-x); }

    void init(int sz) {
        size = sz + 1;
        memset(t, 0, (sz + 2) * sizeof(ll));
    }

    void add(int p, ll x) {
        if (p <= 0) return;
        for (; p <= size; p += lowbit(p)) t[p] += x;
    }

    ll get(int p) {
        ll sum = 0;
        for (; p > 0; p -= lowbit(p)) sum += t[p];
        return sum;
    }

    void update(int p, ll x) { add(p, x - query(p, p)); }
    ll query(int l, int r) { return get(r) - get(l - 1); }

    int kth(ll k) {
        int p = 0;
        for (int i = 20; i >= 0; i--) {
            int p_ = p + (1 << i);
            if (p_ <= size && t[p_] < k) {
                k -= t[p_];
                p = p_;
            }
        }
        return p + 1;
    }
};
```

+ 区间加，单点查询

```cpp
void range_add(int l, int r, ll x) {
    add(l, x);
    add(r + 1, -x);
}
```

+ 区间加，区间和

```cpp
Tbit t1, t2;

void range_add(int l, int r, ll x) {
    t1.add(l, x);
    t2.add(l, l * x);
    t1.add(r + 1, -x);
    t2.add(r + 1, (r + 1) * -x);
}

ll range_sum(int l, int r) {
    return (r + 1) * t1.get(r) - t2.get(r) - l * t1.get(l - 1) + t2.get(l - 1);
}
```

+ 二维

```cpp
struct Tbit {
    ll t[MAXN][MAXN];

    int lowbit(int x) { return x & (-x); }

    void add(int x, int y, int d) {
        for (int i = x; i <= n; i += lowbit(i))
            for (int j = y; j <= m; j += lowbit(j)) t[i][j] += d;
    }

    ll get(int x, int y) {
        ll sum = 0;
        for (int i = x; i > 0; i -= lowbit(i))
            for (int j = y; j > 0; j -= lowbit(j)) sum += t[i][j];
        return sum;
    }

    ll query(int x, int y, int xx, int yy) {
        return get(xx, yy) - get(x - 1, yy) - get(xx, y - 1) + get(x - 1, y - 1);
    }
};
```

+ 二维区间加，区间和

```cpp
Tbit t0, t1, t2, t3;

void add4(int x, int y, ll d) {
    t0.add(x, y, d);
    t1.add(x, y, d * x);
    t2.add(x, y, d * y);
    t3.add(x, y, d * x * y);
}

void range_add(int x, int y, int xx, int yy, ll d) {
    add4(x, y, d);
    add4(x, yy + 1, -d);
    add4(xx + 1, y, -d);
    add4(xx + 1, yy + 1, d);
}

ll get4(int x, int y) {
    return (x + 1) * (y + 1) * t0.get(x, y)
    - (y + 1) * t1.get(x, y)
    - (x + 1) * t2.get(x, y)
    + t3.get(x, y);
}

ll range_sum(int x, int y, int xx, int yy) {
    return get4(xx, yy) - get4(x - 1, yy) - get4(xx, y - 1) + get4(x - 1, y - 1);
}
```

### 线段树

+ 单点修改，RMQ

```cpp
// 下标从1开始
struct Node {
    int val;
    Node(int val = -INF) : val(val) {}
};

Node merge(const Node& a, const Node& b) {
    return Node(max(a.val, b.val));
}

struct SegT {
#define lc (p << 1)
#define rc (p << 1 | 1)
#define mid ((pl + pr) >> 1)

    int size;
    vector<Node> t;

    SegT(int sz) {
        size = 1;
        while (size < sz) size <<= 1;
        t.resize(2 * size);
    }

    Node ask(int p, int pl, int pr, int l, int r) {
        if (l > pr || r < pl) return Node();
        if (l <= pl && r >= pr) return t[p];
        return merge(ask(lc, pl, mid, l, r), ask(rc, mid + 1, pr, l, r));
    }

    void update(int k, int val) {
        int p = size + k - 1;
        t[p] = Node(val);
        for (p >>= 1; p > 0; p >>= 1) {
            t[p] = merge(t[lc], t[rc]);
        }
    }

    Node query(int l, int r) { return ask(1, 1, size, l, r); }

#undef lc
#undef rc
#undef mid
};
```

+ 权值线段树：单点修改，第k大

```cpp
void add(int x, ll val) {
    int p = size + x - 1;
    t[p].val += val;
    for (p >>= 1; p > 0; p >>= 1) {
        t[p].val += val;
    }
}

int ask(int p, int pl, int pr, ll k) {
    if (pl == pr) return pl;
    if (k <= t[lc].val) return ask(lc, pl, mid, k);
    return ask(rc, mid + 1, pr, k - t[lc].val);
}

int query(ll k) { return ask(1, 1, size, k); }
```

+ 区间加，区间和

```cpp
struct Node {
    ll val, lazy;
};

void pushdown(int p, int pl, int pr) {
    if (!t[p].lazy) return;  // 如果是区间赋值，选取一个数据范围外的值
    t[lc].val += t[p].lazy * (mid - pl + 1);
    t[rc].val += t[p].lazy * (pr - mid);
    t[lc].lazy += t[p].lazy;
    t[rc].lazy += t[p].lazy;
    t[p].lazy = 0;
}

ll ask(int p, int pl, int pr, int l, int r) {
    if (l > pr || r < pl) return 0;
    if (l <= pl && r >= pr) return t[p].val;
    pushdown(p, pl, pr);
    ll vl = ask(lc, pl, mid, l, r);
    ll vr = ask(rc, mid + 1, pr, l, r);
    return vl + vr;
}

void modify(int p, int pl, int pr, int l, int r, int val) {
    if (l > pr || r < pl) return;
    if (l <= pl && r >= pr) {
        t[p].val += 1LL * val * (pr - pl + 1);
        t[p].lazy += val;
        return;
    }
    pushdown(p, pl, pr);
    modify(lc, pl, mid, l, r, val);
    modify(rc, mid + 1, pr, l, r, val);
    t[p].val = t[lc].val + t[rc].val;
}

void update(int l, int r, int val) { modify(1, 1, size, l, r, val); }
ll query(int l, int r) { return ask(1, 1, size, l, r); }
```

+ 区间乘混加，区间和取模

```cpp
struct Node {
    ll val, mul, add;
    Node() : val(0), add(0), mul(1) {}
};

void pushdown(int p, int pl, int pr) {
    if (t[p].mul == 1 && t[p].add == 0) return;
    t[lc].val = (t[lc].val * t[p].mul % MOD + (mid - pl + 1) * t[p].add % MOD) % MOD;
    t[rc].val = (t[rc].val * t[p].mul % MOD + (pr - mid) * t[p].add % MOD) % MOD;
    t[lc].mul = t[p].mul * t[lc].mul % MOD;
    t[rc].mul = t[p].mul * t[rc].mul % MOD;
    t[lc].add = (t[lc].add * t[p].mul % MOD + t[p].add) % MOD;
    t[rc].add = (t[rc].add * t[p].mul % MOD + t[p].add) % MOD;
    t[p].mul = 1;
    t[p].add = 0;
}

ll ask(int p, int pl, int pr, int l, int r) {
    if (l > pr || r < pl) return 0;
    if (l <= pl && r >= pr) return t[p].val;
    pushdown(p, pl, pr);
    ll vl = ask(lc, pl, mid, l, r);
    ll vr = ask(rc, mid + 1, pr, l, r);
    return (vl + vr) % MOD;
}

// x' = ax + b
void modify(int p, int pl, int pr, int l, int r, int a, int b) {
    if (l > pr || r < pl) return;
    if (l <= pl && r >= pr) {
        t[p].val = (t[p].val * a % MOD + 1LL * (pr - pl + 1) * b % MOD) % MOD;
        t[p].mul = t[p].mul * a % MOD;
        t[p].add = (t[p].add * a % MOD + b) % MOD;
        return;
    }
    pushdown(p, pl, pr);
    modify(lc, pl, mid, l, r, a, b);
    modify(rc, mid + 1, pr, l, r, a, b);
    t[p].val = (t[lc].val + t[rc].val) % MOD;
}

void update(int l, int r, int a, int b) { modify(1, 1, size, l, r, a, b); }
ll query(int l, int r) { return ask(1, 1, size, l, r); }
```

### 动态开点线段树

```cpp
struct Node {
    int lc, rc, val;
    Node(int lc = 0, int rc = 0, int val = 0) : lc(lc), rc(rc), val(val) {}
} t[20 * MAXN];

int cnt;

struct SegT {
#define mid ((pl + pr) >> 1)

    int rt, size;

    SegT(int sz) : rt(0) {
        size = 1;
        while (size < sz) size <<= 1;
    }

    int modify(int p, int pl, int pr, int k, int val) {
        if (pl > k || pr < k) return p;
        if (!p) p = ++cnt;
        if (pl == pr) t[p].val = val;
        else {
            t[p].lc = modify(t[p].lc, pl, mid, k, val);
            t[p].rc = modify(t[p].rc, mid + 1, pr, k, val);
            t[p].val = max(t[t[p].lc].val, t[t[p].rc].val);
        }
        return p;
    }

    int ask(int p, int pl, int pr, int l, int r) {
        if (l > pr || r < pl) return -INF;
        if (l <= pl && r >= pr) return t[p].val;
        int vl = ask(t[p].lc, pl, mid, l, r);
        int vr = ask(t[p].rc, mid + 1, pr, l, r);
        return max(vl, vr);
    }

    void update(int k, int val) { rt = modify(rt, 1, size, k, val); }
    int query(int l, int r) { return ask(rt, 1, size, l, r); }

#undef mid
};
```

### 主席树

```cpp
struct Node {
    int lc, rc, val;
    Node(int lc = 0, int rc = 0, int val = 0) : lc(lc), rc(rc), val(val) {}
} t[40 * MAXN]; // (4 + log(size)) * MAXN 小心 MLE

int cnt;

struct FST {
#define mid ((pl + pr) >> 1)

    int size;
    vector<int> root;

    FST(int sz) {
        size = 1;
        while (size < sz) size <<= 1;
        root.push_back(N(0, 0, 0));
    }

    int N(int lc, int rc, int val) {
        t[cnt] = Node(lc, rc, val);
        return cnt++;
    }

    int ins(int p, int pl, int pr, int x) {
        if (pl > x || pr < x) return p;
        if (pl == pr) return N(0, 0, t[p].val + 1);
        return N(ins(t[p].lc, pl, mid, x), ins(t[p].rc, mid + 1, pr, x), t[p].val + 1);
    }

    int ask(int p1, int p2, int pl, int pr, int k) {
        if (pl == pr) return pl;
        ll vl = t[t[p2].lc].val - t[t[p1].lc].val;
        if (k <= vl) return ask(t[p1].lc, t[p2].lc, pl, mid, k);
        return ask(t[p1].rc, t[p2].rc, mid + 1, pr, k - vl);
    }

    void add(int x) {
        root.push_back(ins(root.back(), 1, size, x));
    }

    int query(int l, int r, int k) {
        return ask(root[l - 1], root[r], 1, size, k);
    }

#undef mid
};
```

### Splay

```cpp
// 正常Splay
struct Node {
    int val, size;
    Node *pa, *lc, *rc;
    Node(int val = 0, Node *pa = nullptr) : val(val), size(1), pa(pa), lc(nullptr), rc(nullptr) {}
    Node*& c(bool x) { return x ? lc : rc; }
    bool d() { return pa ? this == pa->lc : 0; }
} pool[MAXN << 2], *tail = pool;

struct Splay {
    Node *root;

    Splay() : root(nullptr) {}

    Node* N(int val, Node *pa) {
        return new (tail++) Node(val, pa);
    }

    void upd(Node *o) {
        o->size = (o->lc ? o->lc->size : 0) + (o->rc ? o->rc->size : 0) + 1;
    }

    void link(Node *x, Node *y, bool d) {
        if (x) x->pa = y;
        if (y) y->c(d) = x;
    }

    void rotate(Node *o) {
        bool dd = o->d();
        Node *x = o->pa, *xx = x->pa, *y = o->c(!dd);
        link(o, xx, x->d());
        link(y, x, dd);
        link(x, o, !dd);
        upd(x);
        upd(o);
    }

    void splay(Node *o) {
        for (Node *x = o->pa; x = o->pa, x; rotate(o)) {
            if (x->pa) rotate(o->d() == x->d() ? x : o);
        }
        root = o;
    }
};
```

### Treap

```cpp
// split_x 左侧元素 < x
// split_k 左侧分割出 k 个元素
namespace tr {
    using uint = unsigned int;

    uint rnd() {
        static uint A = 1 << 16 | 3, B = 33333331, C = 1091;
        return C = A * C + B;
    }

    struct Node {
        uint key;
        int val, size;
        Node *lc, *rc;
        Node(int val = 0) : key(rnd()), val(val), size(1), lc(nullptr), rc(nullptr) {}
    } pool[MAXN << 2], *tail = pool;

    Node* N(int val) {
        return new (tail++) Node(val);
    }

    void upd(Node *o) {
        o->size = (o->lc ? o->lc->size : 0) + (o->rc ? o->rc->size : 0) + 1;
    }

    Node* merge(Node *l, Node *r) {
        if (!l) return r;
        if (!r) return l;
        if (l->key > r->key) {
            l->rc = merge(l->rc, r);
            upd(l);
            return l;
        } else {
            r->lc = merge(l, r->lc);
            upd(r);
            return r;
        }
    }

    void split_x(Node *o, int x, Node*& l, Node*& r) {
        if (!o) { l = r = nullptr; return; }
        if (o->val < x) {
            l = o;
            split_x(o->rc, x, l->rc, r);
            upd(l);
        } else {
            r = o;
            split_x(o->lc, x, l, r->lc);
            upd(r);
        }
    }

    void split_k(Node *o, int k, Node*& l, Node*& r) {
        if (!o) { l = r = nullptr; return; }
        int lsize = o->lc ? o->lc->size : 0;
        if (lsize < k) {
            l = o;
            split_k(o->rc, k - lsize - 1, o->rc, r);
            upd(l);
        } else {
            r = o;
            split_k(o->lc, k, l, o->lc);
            upd(r);
        }
    }
}
```

### CDQ 分治

+ 三维偏序（不严格）

```cpp
struct Node {
    int x, y, z, sum, ans;
} p[MAXN], q[MAXN];

void CDQ(int l, int r) {
    if (l == r) return;
    int mid = (l + r) >> 1;
    CDQ(l, mid);
    CDQ(mid + 1, r);
    int i = l, j = mid + 1;
    for (int t = l; t <= r; t++) {
        if (j > r || (i <= mid && p[i].y <= p[j].y)) {
            q[t] = p[i++];
            bit.add(q[t].z, q[t].sum);
        } else {
            q[t] = p[j++];
            q[t].ans += bit.get(q[t].z);
        }
    }
    for (i = l; i <= r; i++) {
        p[i] = q[i];
        bit.update(p[i].z, 0);
    }
}

void go() {
    sort(p + 1, p + n + 1, [](const Node &a, const Node &b) {
        if (a.x != b.x) return a.x < b.x;
        if (a.y != b.y) return a.y < b.y;
        return a.z < b.z;
    });
    auto eq = [](const Node& a, const Node& b) {
        return a.x == b.x && a.y == b.y && a.z == b.z;
    };
    int k = n;
    for (int i = 1, j = 1; i <= n; i++, j++) {
        if (eq(p[i], p[j - 1])) j--, k--;
        else if (i != j) p[j] = p[i];
        p[j].sum++;
    }
    bit.init(m);
    CDQ(1, k);
}
```
