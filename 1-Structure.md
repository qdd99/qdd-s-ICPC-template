## 数据结构

### 并查集

```cpp
int find(int x) { return (x == pa[x]) ? x : pa[x] = find(pa[x]); }
void merge(int a, int b) { pa[find(a)] = find(b); }
```

### RMQ

```cpp
// 下标从0开始
// 一维
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

// 二维
struct RMQ {
    int st[MAXN][MAXN][11][11]; // 11 = ((int)log2(MAXN) + 1)

    int xlog(int x) { return 31 - __builtin_clz(x); }

    void init(int n, int m) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                st[i][j][0][0] = a[i][j];
            }
        }
        for (int i = 0; (1 << i) <= n; i++) {
            for (int j = 0; (1 << j) <= m; j++) {
                if (i == 0 && j == 0) continue;
                for (int r = 0; r + (1 << i) - 1 < n; r++) {
                    for (int c = 0; c + (1 << j) - 1 < m; c++) {
                        if (i == 0) {
                            st[r][c][i][j] = max(st[r][c][i][j - 1], st[r][c + (1 << (j - 1))][i][j - 1]);
                        } else {
                            st[r][c][i][j] = max(st[r][c][i - 1][j], st[r + (1 << (i - 1))][c][i - 1][j]);
                        }
                    }
                }
            }
        }
    }

    int query(int r1, int c1, int r2, int c2) {
        int x = xlog(r2 - r1 + 1);
        int y = xlog(c2 - c1 + 1);
        int m1 = st[r1][c1][x][y];
        int m2 = st[r1][c2 - (1 << y) + 1][x][y];
        int m3 = st[r2 - (1 << x) + 1][c1][x][y];
        int m4 = st[r2 - (1 << x) + 1][c2 - (1 << y) + 1][x][y];
        return max({m1, m2, m3, m4});
    }
};
```

### 树状数组

```cpp
// 支持第k大的BIT
// 下标从1开始
// 修改：单点
// 查询：区间和
struct Tbit {
    int size;
    ll t[MAXN];

    int lowbit(int x) { return x & (-x); }

    void init(int sz) {
        size = sz + 1;
        memset(t, 0, (sz + 2) * sizeof(ll));
    }

    void add(int pos, ll val) {
        if (pos <= 0) return;
        while (pos <= size) {
            t[pos] += val;
            pos += lowbit(pos);
        }
    }

    ll get(int pos) {
        ll sum = 0;
        while (pos > 0) {
            sum += t[pos];
            pos -= lowbit(pos);
        }
        return sum;
    }

    void update(int pos, ll val) { add(pos, val - query(pos, pos)); }
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

// 修改：区间加
// 查询：单点
struct Tbit {
    int size;
    ll t[MAXN];

    int lowbit(int x) { return x & (-x); }

    void init(int sz) {
        size = sz + 1;
        memset(t, 0, (sz + 2) * sizeof(ll));
    }

    void add(int pos, ll val) {
        if (pos <= 0) return;
        while (pos <= size) {
            t[pos] += val;
            pos += lowbit(pos);
        }
    }

    ll get(int pos) {
        ll sum = 0;
        while (pos > 0) {
            sum += t[pos];
            pos -= lowbit(pos);
        }
        return sum;
    }

    void update(int l, int r, ll val) {
        add(l, val);
        add(r + 1, -val);
    }
};

// 修改：区间加
// 查询：区间和
Tbit t1, t2;

void range_add(int l, int r, ll val) {
    t1.add(l, val);
    t2.add(l, l * val);
    t1.add(r + 1, -val);
    t2.add(r + 1, (r + 1) * -val);
}

ll range_sum(int l, int r) {
    return (r + 1) * t1.get(r) - t2.get(r) - l * t1.get(l - 1) + t2.get(l - 1);
}
```

### 线段树

```cpp
// 下标从1开始
// 修改：单点
// 查询：RMQ
struct Node {
    int val;
};

struct SegT {
#define lc (p << 1)
#define rc (p << 1 | 1)
#define mid (pl + pr >> 1)

    int size;
    Node *t;

    SegT(int sz) {
        size = 1;
        while (size < sz) size <<= 1;
        t = new Node[2 * size]();
    }

    ~SegT() {
        delete [] t;
    }

    int ask(int p, int l, int r, int pl, int pr) {
        if (l > pr || r < pl) return -INF;
        if (l <= pl && r >= pr) return t[p].val;
        int vl = ask(lc, l, r, pl, mid);
        int vr = ask(rc, l, r, mid + 1, pr);
        return max(vl, vr);
    }

    void update(int k, int val) {
        int p = size + k - 1;
        t[p].val = val;
        for (p >>= 1; p > 0; p >>= 1) {
            t[p].val = max(t[lc].val, t[rc].val);
        }
    }

    int query(int l, int r) { return ask(1, l, r, 1, size); }

#undef lc
#undef rc
#undef mid
};

// 权值线段树
// 修改：单点加
// 查询：第k大
void add(int x, ll val) {
    int p = size + x - 1;
    t[p].val += val;
    for (p >>= 1; p > 0; p >>= 1) {
        t[p].val += val;
    }
}

int ask(int p, ll k, int pl, int pr) {
    if (pl == pr) return pl;
    if (k <= t[lc].val) return ask(lc, k, pl, mid);
    return ask(rc, k - t[lc].val, mid + 1, pr);
}

int query(ll k) { return ask(1, k, 1, size); }

// 修改：区间加
// 查询：区间和
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

ll ask(int p, int l, int r, int pl, int pr) {
    if (l > pr || r < pl) return 0;
    if (l <= pl && r >= pr) return t[p].val;
    pushdown(p, pl, pr);
    ll vl = ask(lc, l, r, pl, mid);
    ll vr = ask(rc, l, r, mid + 1, pr);
    return vl + vr;
}

void modify(int p, int l, int r, int val, int pl, int pr) {
    if (l > pr || r < pl) return;
    if (l <= pl && r >= pr) {
        t[p].val += 1LL * val * (pr - pl + 1);
        t[p].lazy += val;
        return;
    }
    pushdown(p, pl, pr);
    modify(lc, l, r, val, pl, mid);
    modify(rc, l, r, val, mid + 1, pr);
    t[p].val = t[lc].val + t[rc].val;
}

void update(int l, int r, int val) { modify(1, l, r, val, 1, size); }
ll query(int l, int r) { return ask(1, l, r, 1, size); }
 
// 修改：区间乘混加
// 查询：区间和取模
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

ll ask(int p, int l, int r, int pl, int pr) {
    if (l > pr || r < pl) return 0;
    if (l <= pl && r >= pr) return t[p].val;
    pushdown(p, pl, pr);
    ll vl = ask(lc, l, r, pl, mid);
    ll vr = ask(rc, l, r, mid + 1, pr);
    return (vl + vr) % MOD;
}

// x' = ax + b
void modify(int p, int l, int r, int a, int b, int pl, int pr) {
    if (l > pr || r < pl) return;
    if (l <= pl && r >= pr) {
        t[p].val = (t[p].val * a % MOD + 1LL * (pr - pl + 1) * b % MOD) % MOD;
        t[p].mul = t[p].mul * a % MOD;
        t[p].add = (t[p].add * a % MOD + b) % MOD;
        return;
    }
    pushdown(p, pl, pr);
    modify(lc, l, r, a, b, pl, mid);
    modify(rc, l, r, a, b, mid + 1, pr);
    t[p].val = (t[lc].val + t[rc].val) % MOD;
}

void update(int l, int r, int a, int b) { modify(1, l, r, a, b, 1, size); }
ll query(int l, int r) { return ask(1, l, r, 1, size); }
```

### 主席树

```cpp
struct Node {
    int lc, rc, val;
    Node(int lc = 0, int rc = 0, int val = 0) : lc(lc), rc(rc), val(val) {}
} t[40 * MAXN];

int cnt;

struct FST {
#define mid (pl + pr >> 1)

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

    int ins(int p, int x, int pl, int pr) {
        if (pl > x || pr < x) return p;
        if (pl == pr) return N(0, 0, t[p].val + 1);
        return N(ins(t[p].lc, x, pl, mid), ins(t[p].rc, x, mid + 1, pr), t[p].val + 1);
    }

    int ask(int p1, int p2, int k, int pl, int pr) {
        if (pl == pr) return pl;
        ll vl = t[t[p2].lc].val - t[t[p1].lc].val;
        if (k <= vl) return ask(t[p1].lc, t[p2].lc, k, pl, mid);
        return ask(t[p1].rc, t[p2].rc, k - vl, mid + 1, pr);
    }

    void add(int x) {
        root.push_back(ins(root.back(), x, 1, size));
    }

    int query(int l, int r, int k) {
        return ask(root[l - 1], root[r], k, 1, size);
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
} pool[MAXN], *tail = pool;

struct Splay {
    Node *root;

    Splay() : root(nullptr) {}

    Node* N(int val, Node *pa) {
        return new (tail++) Node(val, pa);
    }

    void pushup(Node *o) {
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
        pushup(x);
        pushup(o);
    }

    void splay(Node *o) {
        for (Node *x = o->pa; x = o->pa, x; rotate(o)) {
            if (x->pa) rotate(o->d() == x->d() ? x : o);
        }
        root = o;
    }
};
```
