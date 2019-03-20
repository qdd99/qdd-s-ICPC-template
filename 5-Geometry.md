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

// 逆时针旋转 r 弧度
V rot(const V& p, double r) {
    return V(p.x * cos(r) - p.y * sin(r), p.x * sin(r) + p.y * cos(r));
}
V rot_ccw90(const V& p) { return V(-p.y, p.x); }
V rot_cw90(const V& p) { return V(p.y, -p.x); }

// 点在线段上 leq(dot(...), 0) 包含端点 lt(dot(...), 0) 则不包含
bool p_on_seg(const V& p, const V& a, const V& b) {
    return eq(det(p - a, b - a), 0) && leq(dot(p - a, p - b), 0);
}

// 点在射线上 geq(dot(...), 0) 包含端点 gt(dot(...), 0) 则不包含
bool p_on_ray(const V& p, const V& a, const V& b) {
    return eq(det(p - a, b - a), 0) && geq(dot(p - a, b - a), 0);
}

// 点到直线距离
double dist_to_line(const V& p, const V& a, const V& b) {
    return abs(cross(a, b, p) / dist(a, b));
}

// 点到线段距离
double dist_to_seg(const V& p, const V& a, const V& b) {
    if (lt(dot(b - a, p - a), 0)) return dist(p, a);
    if (lt(dot(a - b, p - b), 0)) return dist(p, b);
    return dist_to_line(p, a, b);
}

// 求直线交点
V intersect(const V& a, const V& b, const V& c, const V& d) {
    double s1 = cross(c, d, a), s2 = cross(c, d, b);
    return (a * s2 - b * s1) / (s2 - s1);
}
```

### 多边形

```cpp
// 多边形面积
double area(const vector<V>& s) {
    double ret = 0;
    for (int i = 0; i < s.size(); i++) {
        ret += det(s[i], s[(i + 1) % s.size()]);
    }
    return ret / 2;
}

// 多边形重心
V centroid(const vector<V>& s) {
    V c;
    for (int i = 0; i < s.size(); i++) {
        c = c + (s[i] + s[(i + 1) % s.size()]) * det(s[i], s[(i + 1) % s.size()]);
    }
    return c / 6.0 / area(s);
}

// 点是否在多边形中
// 1 inside 0 on border -1 outside
int inside(const vector<V>& s, const V& p) {
    int cnt = 0;
    for (int i = 0; i < s.size(); i++) {
        V a = s[i], b = s[(i + 1) % s.size()];
        if (p_on_seg(p, a, b)) return 0;
        if (leq(a.y, b.y)) swap(a, b);
        if (gt(p.y, a.y)) continue;
        if (leq(p.y, b.y)) continue;
        cnt += gt(cross(b, a, p), 0);
    }
    return (cnt & 1) ? 1 : -1;
}

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

// 多边形是否为凸包
bool is_convex(const vector<V>& s) {
    for (int i = 0; i < s.size(); i++) {
        if (lt(cross(s[(i + 1) % s.size()], s[(i + 2) % s.size()], s[i]), 0)) return false;
    }
    return true;
}

// 点是否在凸包中
// 1 inside 0 on border -1 outside
int inside(const vector<V>& s, const V& p) {
    for (int i = 0; i < s.size(); i++) {
        if (lt(cross(s[i], s[(i + 1) % s.size()], p), 0)) return -1;
        if (p_on_seg(p, s[i], s[(i + 1) % s.size()])) return 0;
    }
    return 1;
}
```

### 圆

### 三维几何
