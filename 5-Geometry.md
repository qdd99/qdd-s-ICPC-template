## 计算几何

### 二维几何基础

```cpp
#define y1 qwq

const double PI = acos(-1);
const double EPS = 1e-8;

int sgn(double x) { return x < -EPS ? -1 : x > EPS; }

// 不要直接使用sgn
bool eq(double x, double y) { return sgn(x - y) == 0; }
bool lt(double x, double y) { return sgn(x - y) < 0; }
bool gt(double x, double y) { return sgn(x - y) > 0; }
bool leq(double x, double y) { return sgn(x - y) <= 0; }
bool geq(double x, double y) { return sgn(x - y) >= 0; }

struct V {
    double x, y;
    V(double x = 0, double y = 0) : x(x), y(y) {}
    V(const V& a, const V& b) : x(b.x - a.x), y(b.y - a.y) {}
    V operator + (const V &b) const { return V(x + b.x, y + b.y); }
    V operator - (const V &b) const { return V(x - b.x, y - b.y); }
    V operator * (double k) const { return V(x * k, y * k); }
    V operator / (double k) const { return V(x / k, y / k); }
    double len() const { return hypot(x, y); }
    double len2() const { return x * x + y * y; }
};

double dist(const V& a, const V& b) { return (b - a).len(); }
double dot(const V& a, const V& b) { return a.x * b.x + a.y * b.y; }
double det(const V& a, const V& b) { return a.x * b.y - a.y * b.x; }
double cross(const V& s, const V& t, const V& o) { return det(V(o, s), V(o, t)); }
```

### 多边形

```cpp
// 构建凸包 点不可以重复
// lt(cross(...), 0) 边上可以有点 leq(cross(...), 0) 则不能
// 会改变输入点的顺序
vector<V> convex_hull(vector<V>& s) {
    // assert(s.size() >= 3);
    sort(s.begin(), s.end(), [](V &a, V &b) { return eq(a.x, b.x) ? lt(a.y, b.y) : lt(a.x, b.x); });
    vector<V> ret(2 * s.size());
    int sz = 0;
    for (int i = 0; i < s.size(); i++) {
        while (sz > 1 && leq(cross(ret[sz - 1], s[i], ret[sz - 2]), 0)) sz--;
        ret[sz++] = s[i];
    }
    int k = sz;
    for (int i = s.size() - 2; i >= 0; i--) {
        while (sz > k && leq(cross(ret[sz - 1], s[i], ret[sz - 2]), 0)) sz--;
        ret[sz++] = s[i];
    }
    ret.resize(sz - (s.size() > 1));
    return ret;
}
```

### 圆

### 三维几何
