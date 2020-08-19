## 计算几何

### 二维几何基础

```cpp
#define y1 qwq

using ld = double;

const ld PI = acos(-1);
const ld EPS = 1e-8;

int sgn(ld x) { return x < -EPS ? -1 : x > EPS; }

// 不要直接使用sgn
bool eq(ld x, ld y) { return sgn(x - y) == 0; }
bool lt(ld x, ld y) { return sgn(x - y) < 0; }
bool gt(ld x, ld y) { return sgn(x - y) > 0; }
bool leq(ld x, ld y) { return sgn(x - y) <= 0; }
bool geq(ld x, ld y) { return sgn(x - y) >= 0; }

struct V {
    ld x, y;
    constexpr V(ld x = 0, ld y = 0) : x(x), y(y) {}
    constexpr V(const V& a, const V& b) : x(b.x - a.x), y(b.y - a.y) {}
    V operator + (const V& b) const { return V(x + b.x, y + b.y); }
    V operator - (const V& b) const { return V(x - b.x, y - b.y); }
    V operator * (ld k) const { return V(x * k, y * k); }
    V operator / (ld k) const { return V(x / k, y / k); }
    ld len() const { return hypot(x, y); }
    ld len2() const { return x * x + y * y; }
};

ostream& operator << (ostream& os, const V& p) { return os << "(" << p.x << ", " << p.y << ")"; }
istream& operator >> (istream& is, V& p) { return is >> p.x >> p.y; }

ld dist(const V& a, const V& b) { return (b - a).len(); }
ld dot(const V& a, const V& b) { return a.x * b.x + a.y * b.y; }
ld det(const V& a, const V& b) { return a.x * b.y - a.y * b.x; }
ld cross(const V& s, const V& t, const V& o) { return det(V(o, s), V(o, t)); }

ld to_rad(ld deg) { return deg / 180 * PI; }

// 象限
int quad(const V& p) {
    int x = sgn(p.x), y = sgn(p.y);
    if (x > 0 && y >= 0) return 1;
    if (x <= 0 && y > 0) return 2;
    if (x < 0 && y <= 0) return 3;
    if (x >= 0 && y < 0) return 4;
    assert(0);
}

// 极角排序
struct cmp_angle {
    V p;
    cmp_angle(const V& p = V()) : p(p) {}
    bool operator () (const V& a, const V& b) const {
        int qa = quad(a - p), qb = quad(b - p);
        if (qa != qb) return qa < qb;
        int d = sgn(cross(a, b, p));
        if (d) return d > 0;
        return dist(a, p) < dist(b, p);
    }
};

// 单位向量
V unit(const V& p) { return eq(p.len(), 0) ? V(1, 0) : p / p.len(); }

// 逆时针旋转 r 弧度
V rot(const V& p, ld r) {
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
ld dist_to_line(const V& p, const V& a, const V& b) {
    return abs(cross(a, b, p) / dist(a, b));
}

// 点到线段距离
ld dist_to_seg(const V& p, const V& a, const V& b) {
    if (lt(dot(b - a, p - a), 0)) return dist(p, a);
    if (lt(dot(a - b, p - b), 0)) return dist(p, b);
    return dist_to_line(p, a, b);
}

// 求直线交点
V intersect(const V& a, const V& b, const V& c, const V& d) {
    ld s1 = cross(c, d, a), s2 = cross(c, d, b);
    return (a * s2 - b * s1) / (s2 - s1);
}

// 点在直线上的投影点
V proj(const V& p, const V& a, const V& b) {
    return a + (b - a) * dot(b - a, p - a) / (b - a).len2();
}

// 点关于直线的对称点
V reflect(const V& p, const V& a, const V& b) {
    return proj(p, a, b) * 2 - p;
}

// 三角形重心
V centroid(const V& a, const V& b, const V& c) {
    return (a + b + c) / 3;
}

// 内心
V incenter(const V& a, const V& b, const V& c) {
    ld AB = dist(a, b), AC = dist(a, c), BC = dist(b, c);
    // ld r = abs(cross(b, c, a)) / (AB + AC + BC);
    return (a * BC + b * AC + c * AB) / (AB + BC + AC);
}

// 外心
V circumcenter(const V& a, const V& b, const V& c) {
    V mid1 = (a + b) / 2, mid2 = (a + c) / 2;
    // ld r = dist(a, b) * dist(b, c) * dist(c, a) / 2 / abs(cross(b, c, a));
    return intersect(mid1, mid1 + rot_ccw90(b - a), mid2, mid2 + rot_ccw90(c - a));
}

// 垂心
V orthocenter(const V& a, const V& b, const V& c) {
    return centroid(a, b, c) * 3 - circumcenter(a, b, c) * 2;
}

// 旁心（三个）
vector<V> escenter(const V& a, const V& b, const V& c) {
    ld AB = dist(a, b), AC = dist(a, c), BC = dist(b, c);
    V p1 = (a * (-BC) + b * AC + c * AB) / (AB + AC - BC);
    V p2 = (a * BC + b * (-AC) + c * AB) / (AB - AC + BC);
    V p3 = (a * BC + b * AC + c * (-AB)) / (-AB + AC + BC);
    return {p1, p2, p3};
}
```

### 多边形

```cpp
// 多边形面积
ld area(const vector<V>& s) {
    ld ret = 0;
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

```cpp
struct C {
    V o;
    ld r;
    C(const V& o, ld r) : o(o), r(r) {}
};

// 过一点求圆的切线，返回切点
vector<V> tangent_point(const C& c, const V& p) {
    ld k = c.r / dist(c.o, p);
    if (gt(k, 1)) return vector<V>();
    if (eq(k, 1)) return {p};
    V a = V(c.o, p) * k;
    return {c.o + rot(a, acos(k)), c.o + rot(a, -acos(k))};
}

// 最小圆覆盖
C min_circle_cover(vector<V> a) {
    shuffle(a.begin(), a.end(), mt_rand);
    V o = a[0];
    ld r = 0;
    int n = a.size();
    for (int i = 1; i < n; i++) if (gt(dist(a[i], o), r)) {
        o = a[i]; r = 0;
        for (int j = 0; j < i; j++) if (gt(dist(a[j], o), r)) {
            o = (a[i] + a[j]) / 2;
            r = dist(a[j], o);
            for (int k = 0; k < j; k++) if (gt(dist(a[k], o), r)) {
                o = circumcenter(a[i], a[j], a[k]);
                r = dist(a[k], o);
            }
        }
    }
    return C(o, r);
}
```

### 三维几何
