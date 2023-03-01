## Data Structures

**Unless specified, all data structure interfaces are 0-indexed, [l, r]**

### Disjoint Set Union (DSU)

```cpp
struct dsu {
  vector<int> p;
  dsu(int n) : p(n, -1) {}
  int find(int x) { return (p[x] < 0) ? x : p[x] = find(p[x]); }
  bool merge(int x, int y) {
    x = find(x), y = find(y);
    if (x == y) return 0;
    p[y] += p[x];
    p[x] = y;
    return 1;
  }
};
```

+ Dynamic Allocation DSU

```cpp
struct dsu {
  unordered_map<int, int> p;
  int data(int x) {
    auto it = p.find(x);
    return it == p.end() ? p[x] = -1 : it->second;
  }
  int find(int x) {
    int n = data(x);
    return n < 0 ? x : p[x] = find(n);
  }
  bool merge(int x, int y) {
    x = find(x), y = find(y);
    if (x == y) return 0;
    auto itx = p.find(x), ity = p.find(y);
    if (itx->second > ity->second) swap(itx, ity), swap(x, y);
    itx->second += ity->second;
    ity->second = x;
    return 1;
  }
};
```

+ Rollback DSU

```cpp
struct dsu {
  vector<int> p, sz, undo;
  dsu(int n) : p(n, -1), sz(n, 1) {}
  int find(int x) { return (p[x] < 0) ? x : find(p[x]); }
  bool merge(int x, int y) {
    x = find(x), y = find(y);
    if (x == y) return 0;
    if (sz[x] > sz[y]) swap(x, y);
    undo.push_back(x);
    sz[y] += sz[x];
    p[x] = y;
    return 1;
  }
  void rollback() {
    int x = undo.back();
    undo.pop_back();
    sz[p[x]] -= sz[x];
    p[x] = -1;
  }
};
```

### Sparse Table

+ 1D

```cpp
struct SparseTable {
  vector<vector<int>> st;

  SparseTable(const vector<int>& a) {
    int n = a.size();
    st.assign(__lg(n) + 1, vector<int>(n));
    for (int i = 0; i < n; i++) {
      st[0][i] = a[i];
    }
    for (int k = 1; (1 << k) <= n; k++) {
      for (int i = 0; i + (1 << k) <= n; i++) {
        st[k][i] = max(st[k - 1][i], st[k - 1][i + (1 << (k - 1))]);
      }
    }
  }

  int query(int l, int r) {
    int k = __lg(r - l + 1);
    return max(st[k][l], st[k][r - (1 << k) + 1]);
  }
};
```

+ 2D

```cpp
struct SparseTable {
  int st[11][11][N][N]; // 11 = ((int)log2(N) + 1)

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
    int x = __lg(r2 - r1 + 1);
    int y = __lg(c2 - c1 + 1);
    int m1 = st[x][y][r1][c1];
    int m2 = st[x][y][r1][c2 - (1 << y) + 1];
    int m3 = st[x][y][r2 - (1 << x) + 1][c1];
    int m4 = st[x][y][r2 - (1 << x) + 1][c2 - (1 << y) + 1];
    return max({m1, m2, m3, m4});
  }
};
```

+ Sliding Window RMQ

```cpp
// k is the size of the sliding window
deque<int> q;
for (int i = 0, j = 0; i + k <= n; i++) {
  while (j < i + k) {
    while (!q.empty() && a[q.back()] < a[j]) q.pop_back(); // Change '<' to '>' for minimum
    q.push_back(j++);
  }
  while (q.front() < i) q.pop_front();
  rmq.push_back(a[q.front()]);
}
```

### Fenwick Tree (Binary Indexed Tree)

+ Point Update, Range Sum

```cpp
template <class T>
struct fenwick {
  int n;
  vector<T> t;
  fenwick(int n) : n(n), t(n) {}
  void add(int p, T x) {
    // assert(0 <= p && p < n);
    for (; p < n; p |= p + 1) t[p] += x;
  }
  T sum(int p) {
    T ans = 0;
    for (; p >= 0; p = (p & (p + 1)) - 1) ans += t[p];
    return ans;
  }
  void set(int p, T x) { add(p, x - range_sum(p, p)); }
  T range_sum(int l, int r) { return sum(r) - sum(l - 1); }

  // smallest is 0-th
  int kth(T k) {
    int p = 0;
    for (int i = 1 << __lg(n); i; i /= 2) {
      if (p + i <= n && k >= t[p + i - 1]) {
        p += i;
        k -= t[p - 1];
      }
    }
    return p;
  }
};
```

+ Range Add, Point Query

```cpp
void range_add(int l, int r, i64 x) {
  add(l, x);
  add(r + 1, -x);
}
```

+ Range Add, Range Sum

```cpp
template <class T>
struct fenwick_range {
  fenwick<T> t0, t1;
  fenwick_range(int n) : t0(n), t1(n) {}
  void range_add(int l, int r, T x) {
    t0.add(l, x);
    t1.add(l, l * x);
    t0.add(r + 1, -x);
    t1.add(r + 1, (r + 1) * -x);
  }
  T range_sum(int l, int r) {
    return (r + 1) * t0.sum(r) - t1.sum(r) - l * t0.sum(l - 1) + t1.sum(l - 1);
  }
};
```

+ 2D

```cpp
template <class T>
struct fenwick_2d {
  int n, m;
  vector<vector<T>> t;
  fenwick_2d(int n, int m) : n(n), m(m), t(n, vector<T>(m)) {}
  void add(int x, int y, T d) {
    for (int i = x; i < n; i |= i + 1)
      for (int j = y; j < m; j |= j + 1) t[i][j] += d;
  }
  T sum(int x, int y) {
    T ans = 0;
    for (int i = x; i >= 0; i = (i & (i + 1)) - 1)
      for (int j = y; j >= 0; j = (j & (j + 1)) - 1) ans += t[i][j];
    return ans;
  }
  T range_sum(int x, int y, int xx, int yy) {
    return sum(xx, yy) - sum(x - 1, yy) - sum(xx, y - 1) + sum(x - 1, y - 1);
  }
};
```

+ 2D Range Add, Range Sum

```cpp
template <class T>
struct fenwick_range_2d {
  int n, m;
  fenwick_2d<T> t0, t1, t2, t3;
  fenwick_range_2d(int n, int m) : n(n), m(m), t0(n, m), t1(n, m), t2(n, m), t3(n, m) {}
  void range_add(int x, int y, int xx, int yy, T d) {
    add4(x, y, d);
    add4(x, yy + 1, -d);
    add4(xx + 1, y, -d);
    add4(xx + 1, yy + 1, d);
  }
  T range_sum(int x, int y, int xx, int yy) {
    return sum4(xx, yy) - sum4(x - 1, yy) - sum4(xx, y - 1) + sum4(x - 1, y - 1);
  }

private:
  void add4(int x, int y, T d) {
    t0.add(x, y, d);
    t1.add(x, y, d * x);
    t2.add(x, y, d * y);
    t3.add(x, y, d * x * y);
  }
  T sum4(int x, int y) {
    return (x + 1) * (y + 1) * t0.sum(x, y)
    - (y + 1) * t1.sum(x, y)
    - (x + 1) * t2.sum(x, y)
    + t3.sum(x, y);
  }
};
```

### Segment Tree

+ ZKW Segment Tree

```cpp
struct segt {
  using S = int;

  S op(S a, S b) { return max(a, b); }
  S e() { return (int)-2e9; }

  int n;
  vector<S> d;
  segt(int n) : n(n), d(2 * n, e()) {}

  void set(int p, S x) {
    for (d[p += n] = x; p > 1; p /= 2) {
      d[p / 2] = op(d[p], d[p ^ 1]);
    }
  }

  S query(int l, int r) {
    S lp = e(), rp = e();
    for (l += n, r += n + 1; l < r; l /= 2, r /= 2) {
      if (l & 1) lp = op(lp, d[l++]);
      if (r & 1) rp = op(d[--r], rp);
    }
    return op(lp, rp);
  }
};
```

+ Point Update, Range Max

```cpp
// Use const S& when necessary to optimize constants
template <class S>
struct segtree {
  using OP = function<S(S, S)>;
  using E = function<S()>;

  int _n, size;
  vector<S> d;

  const OP op;
  const E e;

  segtree(int n, OP op, E e) : _n(n), op(op), e(e) {
    size = 1;
    while (size < n) size <<= 1;
    d.assign(2 * size, e());
  }

  segtree(const vector<S>& v, OP op, E e) : segtree(v.size(), op, e) {
    copy(v.begin(), v.end(), d.begin() + size);
    for (int i = size - 1; i >= 1; i--) pull(i);
  }

  vector<S> operator()() const { return vector<S>(d.begin() + size, d.begin() + size + _n); }

  void pull(int p) { d[p] = op(d[p * 2], d[p * 2 + 1]); }

  void set(int p, S x) {
    p += size;
    d[p] = x;
    for (p >>= 1; p > 0; p >>= 1) pull(p);
  }

  S query(int l, int r) {
    S lp = e(), rp = e();
    for (l += size, r += size + 1; l < r; l >>= 1, r >>= 1) {
      if (l & 1) lp = op(lp, d[l++]);
      if (r & 1) rp = op(d[--r], rp);
    }
    return op(lp, rp);
  }

  S operator[](int i) { return d[i + size]; }

  // f(e()) = false
  // find the smallest r such that f(sum([l...r])) = true
  template <class F>
  int find_right(int l, F f) {
    l += size;
    S s = e();
    do {
      while (l % 2 == 0) l >>= 1;
      if (f(op(s, d[l]))) {
        while (l < size) {
          l *= 2;
          if (!f(op(s, d[l]))) {
            s = op(s, d[l]);
            l++;
          }
        }
        return l - size;
      }
      s = op(s, d[l]);
      l++;
    } while ((l & -l) != l);
    return _n;
  }

  // find the largest l such that f(sum([l...r])) = true
  template <class F>
  int find_left(int r, F f) {
    r += size + 1;
    S s = e();
    do {
      r--;
      while (r > 1 && (r % 2)) r >>= 1;
      if (f(op(d[r], s))) {
        while (r < size) {
          r = 2 * r + 1;
          if (!f(op(d[r], s))) {
            s = op(d[r], s);
            r--;
          }
        }
        return r - size;
      }
      s = op(d[r], s);
    } while ((r & -r) != r);
    return -1;
  }
};

auto op = [](int a, int b) { return max(a, b); };
auto e = []() { return (int)-2e9; };

segtree<int> st(n, op, e);
```

+ Range Add, Range Sum

```cpp
template <class S, class T>
struct lazy_segtree {
  using OP = function<S(S, S)>;
  using E = function<S()>;
  using MAP = function<S(S, T)>;
  using COM = function<T(T, T)>;
  using ID = function<T()>;

  int _n, size, log;
  vector<S> d;
  vector<T> lz;

  const OP op;
  const E e;
  const MAP mapping;
  const COM composition;
  const ID id;

  lazy_segtree(int n, OP op, E e, MAP mapping, COM composition, ID id)
      : _n(n), op(op), e(e), mapping(mapping), composition(composition), id(id) {
    size = 1, log = 0;
    while (size < n) size <<= 1, log++;
    d.assign(2 * size, e());
    lz.assign(size, id());
  }

  lazy_segtree(const vector<S>& v, OP op, E e, MAP mapping, COM composition, ID id)
      : lazy_segtree(v.size(), op, e, mapping, composition, id) {
    copy(v.begin(), v.end(), d.begin() + size);
    for (int i = size - 1; i >= 1; i--) pull(i);
  }

  vector<S> operator()() const {
    lazy_segtree cp = *this;
    for (int i = 1; i < size; i++) cp.push(i);
    return vector<S>(cp.d.begin() + size, cp.d.begin() + size + _n);
  }

  void pull(int p) { d[p] = op(d[p * 2], d[p * 2 + 1]); }

  void apply(T t, int p) {
    d[p] = mapping(d[p], t);
    if (p < size) lz[p] = composition(t, lz[p]);
  }

  void push(int p) {
    if (lz[p] == id()) return;
    apply(lz[p], p * 2);
    apply(lz[p], p * 2 + 1);
    lz[p] = id();
  }

  void set(int p, S x) {
    p += size;
    for (int i = log; i >= 1; i--) push(p >> i);
    d[p] = x;
    for (int i = 1; i <= log; i++) pull(p >> i);
  }

  void update(int l, int r, T t) {
    function<void(int, int, int)> rec = [&](int p, int x, int y) {
      if (l > y || r < x) return;
      if (l <= x && r >= y) {
        apply(t, p);
        return;
      }
      push(p);
      int z = (x + y) >> 1;
      rec(p * 2, x, z);
      rec(p * 2 + 1, z + 1, y);
      pull(p);
    };
    rec(1, 0, size - 1);
  }

  S query(int l, int r) {
    function<S(int, int, int)> rec = [&](int p, int x, int y) -> S {
      if (l > y || r < x) return e();
      if (l <= x && r >= y) return d[p];
      push(p);
      int z = (x + y) >> 1;
      return op(rec(p * 2, x, z), rec(p * 2 + 1, z + 1, y));
    };
    return rec(1, 0, size - 1);
  }

  S operator[](int i) { return query(i, i); }

  // f(e()) = false
  // find the smallest r such that f(sum([l...r])) = true
  template <class F>
  int find_right(int l, F f) {
    l += size;
    for (int i = log; i >= 1; i--) push(l >> i);
    S s = e();
    do {
      while (l % 2 == 0) l >>= 1;
      if (f(op(s, d[l]))) {
        while (l < size) {
          push(l);
          l *= 2;
          if (!f(op(s, d[l]))) {
            s = op(s, d[l]);
            l++;
          }
        }
        return l - size;
      }
      s = op(s, d[l]);
      l++;
    } while ((l & -l) != l);
    return _n;
  }

  // find the largest l such that f(sum([l...r])) = true
  template <class F>
  int find_left(int r, F f) {
    r += size + 1;
    for (int i = log; i >= 1; i--) push((r - 1) >> i);
    S s = e();
    do {
      r--;
      while (r > 1 && (r % 2)) r >>= 1;
      if (f(op(d[r], s))) {
        while (r < size) {
          push(r);
          r = 2 * r + 1;
          if (!f(op(d[r], s))) {
            s = op(d[r], s);
            r--;
          }
        }
        return r - size;
      }
      s = op(d[r], s);
    } while ((r & -r) != r);
    return -1;
  }
};

struct Info {
  i64 sum, len;
};

auto op = [](Info a, Info b) { return Info{a.sum + b.sum, a.len + b.len}; };
auto e = []() { return Info{0LL, 0LL}; };
auto mapping = [](Info a, i64 f) { return Info{a.sum + a.len * f, a.len}; };
auto composition = [](i64 f, i64 g) { return f + g; };  // f(g())
auto id = []() { return 0LL; };  // For range assignment, choose a value outside the data range

lazy_segtree<Info, i64> st(vector<Info>(n, Info{0LL, 1LL}), op, e, mapping, composition, id);
```

+ Range Affine, Range Sum

```cpp
const i64 md = 1e9 + 7;

struct Info {
  i64 sum, len;
};

struct Tag {
  i64 mul, add;

  bool operator==(const Tag& t) const { return mul == t.mul && add == t.add; }
};

auto op = [](Info a, Info b) { return Info{(a.sum + b.sum) % md, a.len + b.len}; };
auto e = []() { return Info{0, 0}; };
auto mapping = [](Info a, Tag t) { return Info{(a.sum * t.mul + a.len * t.add) % md, a.len}; };
auto composition = [](Tag f, Tag g) { return Tag{(f.mul * g.mul) % md, (f.mul * g.add + f.add) % md}; };
auto id = []() { return Tag{1, 0}; };
```

### Functional Segment Tree

```cpp
struct Node {
  int lc, rc, val;
  Node(int lc = 0, int rc = 0, int val = 0) : lc(lc), rc(rc), val(val) {}
} t[40 * MAXN]; // Be careful with MLE: (4 + log(size)) * MAXN

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
    i64 vl = t[t[p2].lc].val - t[t[p1].lc].val;
    if (k <= vl) return ask(t[p1].lc, t[p2].lc, pl, mid, k);
    return ask(t[p1].rc, t[p2].rc, mid + 1, pr, k - vl);
  }

  void add(int x) {
    root.push_back(ins(root.back(), 0, size - 1, x));
  }

  int query(int l, int r, int k) {
    return ask(root[l - 1], root[r], 0, size - 1, k);
  }

#undef mid
};
```

### pb_ds

```cpp
// Balanced Tree
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
template<class T>
using rank_set = tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;
template<class Key, class T>
using rank_map = tree<Key, T, less<Key>, rb_tree_tag, tree_order_statistics_node_update>;

void example() {
  rank_set<int> t, t2;
  t.insert(8);
  auto it = t.insert(10).first;
  assert(it == t.lower_bound(9));
  assert(t.order_of_key(10) == 1);
  assert(t.order_of_key(11) == 2);
  assert(*t.find_by_order(0) == 8);
  t.join(t2);  // assuming t < t2 or t > t2, merge t2 into t
}

// Priority Queue
#include <ext/pb_ds/priority_queue.hpp>
using namespace __gnu_pbds;
template<class T, class Cmp = less<T>>
using pair_heap = __gnu_pbds::priority_queue<T, Cmp>;

void example() {
  pair_heap<int> q1, q2;
  q1.push(5);
  q1.push(10);
  q2.push(1);
  q2.push(7);
  q1.join(q2);
}
```

### Splay

```cpp
// Normal Splay
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
// split_x: elements on the left are < x
// split_k: split out k elements on the left
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

### Interval Set

```cpp
// For Q assign operation it takes Qlogn time in total
template <class T>
struct interval_set {
  map<pair<int, int>, T> mp;  // {r,l}=val
  interval_set(int l, int r, T v = T()) { mp[{r, l}] = v; }

  // assign a[i]=val for l<=i<=r
  // returns affected ranges before performing this assign operation
  vector<pair<pair<int, int>, T>> assign(int l, int r, T v) {
    auto b = mp.lower_bound({l, 0})->first;
    if (b.second != l) {
      T z = mp[b];
      mp.erase(b);
      mp[{l - 1, b.second}] = z;
      mp[{b.first, l}] = z;
    }

    auto e = mp.lower_bound({r, 0})->first;
    if (e.first != r) {
      T z = mp[e];
      mp.erase(e);
      mp[{e.first, r + 1}] = z;
      mp[{r, e.second}] = z;
    }

    vector<pair<pair<int, int>, T>> ret;
    for (auto it = mp.lower_bound({l, 0}); it != mp.end() && it->first.first <= r; ++it) {
      ret.push_back({{it->first.second, it->first.first}, it->second});
    }
    for (auto it : ret) mp.erase({it.first.second, it.first.first});
    mp[{r, l}] = v;
    return ret;
  }

  T operator[](int i) const { return mp.lower_bound({i, 0})->second; }
};
```

### CDQ Divide and Conquer

+ Three-dimensional partial order (non-strict)

```cpp
struct Node {
  int x, y, z, sum, ans;
} p[N], q[N];

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
    bit.set(p[i].z, 0);
  }
}

void go() {
  sort(p, p + n, [](const Node &a, const Node &b) {
    if (a.x != b.x) return a.x < b.x;
    if (a.y != b.y) return a.y < b.y;
    return a.z < b.z;
  });
  auto eq = [](const Node& a, const Node& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
  };
  int k = n;
  for (int i = 0, j = 0; i < n; i++, j++) {
    if (eq(p[i], p[j - 1])) j--, k--;
    else if (i != j) p[j] = p[i];
    p[j].sum++;
  }
  bit.init(m);
  CDQ(0, k - 1);
}
```

### Mo's Algorithm

```cpp
int unit = max(1, int(n / sqrt(q)));
sort(qry.begin(), qry.end(), [&](auto &a, auto &b) {
  if (a.l / unit != b.l / unit) return a.l < b.l;
  return ((a.l / unit) & 1) ? a.r < b.r : a.r > b.r;
});

// [l, r)
while (l > q.l) add_left(--l);
while (r < q.r) add_right(r++);
while (l < q.l) delete_left(l++);
while (r > q.r) delete_right(--r);
```