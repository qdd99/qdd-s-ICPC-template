## Geometry

### 2D Geometry Basics

```cpp
#define y1 qwq

using ld = double;
using V = complex<ld>;

const ld eps = 1e-8;

int sign(ld x) { return x < -eps ? -1 : x > eps; }

ld dot(V a, V b) { return (conj(a) * b).real(); }
ld det(V a, V b) { return (conj(a) * b).imag(); }

ld to_rad(ld deg) { return deg / 180 * numbers::pi; }

// quadrant
int quad(V p) {
  int x = sign(p.real()), y = sign(p.imag());
  if (x > 0 && y >= 0) return 1;
  if (x <= 0 && y > 0) return 2;
  if (x < 0 && y <= 0) return 3;
  if (x >= 0 && y < 0) return 4;
  assert(0);
}

// sorting by polar angle
struct cmp_angle {
  V p;
  cmp_angle(V p = V()) : p(p) {}
  bool operator()(V a, V b) const {
    int qa = quad(a - p), qb = quad(b - p);
    if (qa != qb) return qa < qb;
    int s = sign(det(a - p, b - p));
    return s ? s > 0 : abs(a - p) < abs(b - p);
  }
};

// unit vector
V unit(V p) { return sign(abs(p)) == 0 ? V(1, 0) : p / abs(p); }

// rotate conterclockwise by r radians
V rot(V p, ld r) {
  return p * polar(1.0, r);
}
V rot_ccw90(V p) { return p * V(0, 1); }
V rot_cw90(V p) { return p * V(0, -1); }

// point on segment, dot(...) <= 0 contains endpoints, < otherwise
bool p_on_seg(V p, V a, V b) {
  return sign(det(p - a, b - a)) == 0 && sign(dot(p - a, p - b)) <= 0;
}

// point on ray, dot(...) >= 0 contains the endpoint, > otherwise
bool p_on_ray(V p, V a, V b) {
  return sign(det(p - a, b - a)) == 0 && sign(dot(p - a, b - a)) >= 0;
}

// intersection of lines
V intersect(V a, V b, V c, V d) {
  ld s1 = det(c - a, d - a), s2 = det(c - b, d - b);
  return (a * s2 - b * s1) / (s2 - s1);
}

// projection of a point onto a line
V proj(V p, V a, V b) {
  return a + (b - a) * dot(b - a, p - a) / norm(b - a);
}

// symmetric point about a line
V reflect(V p, V a, V b) {
  return proj(p, a, b) * 2.0 - p;
}

// closest point on segment
V closest_point_on_seg(V p, V a, V b) {
  if (sign(dot(b - a, p - a)) < 0) return a;
  if (sign(dot(a - b, p - b)) < 0) return b;
  return proj(p, a, b);
}

// centroid
V centroid(V a, V b, V c) {
  return (a + b + c) / 3.0;
}

// incenter
V incenter(V a, V b, V c) {
  ld AB = abs(a - b), AC = abs(a - c), BC = abs(b - c);
  // ld r = abs(det(b - a, c - a)) / (AB + AC + BC);
  return (a * BC + b * AC + c * AB) / (AB + BC + AC);
}

// circumcenter
V circumcenter(V a, V b, V c) {
  V mid1 = (a + b) / 2.0, mid2 = (a + c) / 2.0;
  // ld r = dist(a, b) * dist(b, c) * dist(c, a) / 2 / abs(det(b - a, c - a));
  return intersect(mid1, mid1 + rot_ccw90(b - a), mid2, mid2 + rot_ccw90(c - a));
}

// orthocenter
V orthocenter(V a, V b, V c) {
  return centroid(a, b, c) * 3.0 - circumcenter(a, b, c) * 2.0;
}

// excenters (opposite to a, b, c)
vector<V> excenters(V a, V b, V c) {
  ld AB = abs(a - b), AC = abs(a - c), BC = abs(b - c);
  V p1 = (a * (-BC) + b * AC + c * AB) / (AB + AC - BC);
  V p2 = (a * BC + b * (-AC) + c * AB) / (AB - AC + BC);
  V p3 = (a * BC + b * AC + c * (-AB)) / (-AB + AC + BC);
  return {p1, p2, p3};
}
```

### Polygons

```cpp
// polygon area
ld area(const vector<V>& s) {
  ld ret = 0;
  for (int i = 0; i < s.size(); i++) {
    ret += det(s[i], s[(i + 1) % s.size()]);
  }
  return ret / 2;
}

// polygon centroid
V centroid(const vector<V>& s) {
  V c;
  for (int i = 0; i < s.size(); i++) {
    c = c + (s[i] + s[(i + 1) % s.size()]) * det(s[i], s[(i + 1) % s.size()]);
  }
  return c / 6.0 / area(s);
}

// point and polygon
// 1 inside 0 on border -1 outside
int inside(const vector<V>& s, V p) {
  int cnt = 0;
  for (int i = 0; i < s.size(); i++) {
    V a = s[i], b = s[(i + 1) % s.size()];
    if (p_on_seg(p, a, b)) return 0;
    if (sign(a.imag() - b.imag()) <= 0) swap(a, b);
    if (sign(p.imag() - a.imag()) > 0) continue;
    if (sign(p.imag() - b.imag()) <= 0) continue;
    cnt += sign(det(b - p, a - p)) > 0;
  }
  return (cnt & 1) ? 1 : -1;
}

// convex hull, points cannot be duplicated
// det(...) < 0 allow point on edges det(...) <= 0 otherwise
// will change the order of the input points
vector<V> convex_hull(vector<V>& s) {
  // assert(s.size() >= 3);
  sort(s.begin(), s.end(), [](V &a, V &b) {
    return a.real() == b.real() ? a.imag() < b.imag() : a.real() < b.real();
  });
  vector<V> ret(2 * s.size());
  int sz = 0;
  for (int i = 0; i < s.size(); i++) {
    while (sz > 1 && sign(det(ret[sz - 1] - ret[sz - 2], s[i] - ret[sz - 2])) <= 0) sz--;
    ret[sz++] = s[i];
  }
  int k = sz;
  for (int i = s.size() - 2; i >= 0; i--) {
    while (sz > k && sign(det(ret[sz - 1] - ret[sz - 2], s[i] - ret[sz - 2])) <= 0) sz--;
    ret[sz++] = s[i];
  }
  ret.resize(sz - (s.size() > 1));
  return ret;
}

// is convex?
bool is_convex(const vector<V>& s) {
  for (int i = 0; i < s.size(); i++) {
    int j = (i + 1) % s.size(), k = (i + 2) % s.size();
    if (sign(det(s[j] - s[i], s[k] - s[i])) < 0) return false;
  }
  return true;
}

// point and convex hull
// 1 inside 0 on border -1 outside
int inside(const vector<V>& s, V p) {
  for (int i = 0; i < s.size(); i++) {
    int j = (i + 1) % s.size();
    if (sign(det(s[i] - p, s[j] - p)) < 0) return -1;
    if (p_on_seg(p, s[i], s[j])) return 0;
  }
  return 1;
}

// closest pair of points, sort by x-coordinate first
// min_dist(s, 0, s.size())
ld min_dist(const vector<V>& s, int l, int r) {
  if (r - l <= 5) {
    ld ret = 1e100;
    for (int i = l; i < r; i++) {
      for (int j = i + 1; j < r; j++) {
        ret = min(ret, abs(s[i] - s[j]));
      }
    }
    return ret;
  }
  int m = (l + r) >> 1;
  ld ret = min(min_dist(s, l, m), min_dist(s, m, r));
  vector<V> q;
  for (int i = l; i < r; i++) {
    if (abs(s[i].real() - s[m].real()) <= ret) q.push_back(s[i]);
  }
  sort(q.begin(), q.end(), [](auto& a, auto& b) { return a.imag() < b.imag(); });
  for (int i = 1; i < q.size(); i++) {
    for (int j = i - 1; j >= 0 && q[j].imag() >= q[i].imag() - ret; j--) {
      ret = min(ret, abs(q[i] - q[j]));
    }
  }
  return ret;
}
```

### Circles

```cpp
struct C {
  V o;
  ld r;
  C(V o, ld r) : o(o), r(r) {}
};

// sector area, radius r, angle d
ld area_sector(ld r, ld d) { return r * r * d / 2; }

// find the tangent line to a circle and return the tangent point
vector<V> tangent_point(C c, V p) {
  ld k = c.r / abs(c.o - p);
  if (sign(k - 1) > 0) return {};
  if (sign(k - 1) == 0) return {p};
  V a = (p - c.o) * k;
  return {c.o + rot(a, acos(k)), c.o + rot(a, -acos(k))};
}

vector<V> circle_line_inter(C c, V a, V b) {
  V p = proj(c.o, a, b);
  ld d = abs(p - c.o);
  if (sign(d - c.r) > 0) return {};
  if (sign(d - c.r) == 0) return {p};
  ld l = sqrt(c.r * c.r - d * d);
  V v = unit(b - a) * l;
  return {p + v, p - v};
}

// 0: disjoint 1: touch externally 2: intersect 3: touch internally 4: contain
int circle_circle_relation(C c1, C c2) {
  ld d = abs(c1.o - c2.o);
  if (sign(d - c1.r - c2.r) > 0) return 0;
  if (sign(d - c1.r - c2.r) == 0) return 1;
  if (sign(d - abs(c1.r - c2.r)) < 0) return 4;
  if (sign(d - abs(c1.r - c2.r)) == 0) return 3;
  return 2;
}

vector<V> circle_circle_inter(C c1, C c2) {
  if (c1.r < c2.r) swap(c1, c2);
  int rel = circle_circle_relation(c1, c2);
  if (rel == 0 || rel == 4) return {};
  if (rel == 1 || rel == 3) return {c1.o + unit(c2.o - c1.o) * c1.r};
  ld d = abs(c1.o - c2.o);
  ld a = (c1.r * c1.r + d * d - c2.r * c2.r) / 2 / d;
  V p = c1.o + (c2.o - c1.o) * a / d;
  V v = rot_ccw90(unit(c2.o - c1.o)) * sqrt(c1.r * c1.r - a * a);
  return {p + v, p - v};
}

// min circle cover
C min_circle_cover(vector<V> a) {
  shuffle(a.begin(), a.end(), rng);
  V o = a[0];
  ld r = 0;
  int n = a.size();
  for (int i = 1; i < n; i++) if (sign(abs(a[i] - o) - r) > 0) {
    o = a[i]; r = 0;
    for (int j = 0; j < i; j++) if (sign(abs(a[j] - o) - r) > 0) {
      o = (a[i] + a[j]) / 2.0;
      r = abs(a[j] - o);
      for (int k = 0; k < j; k++) if (sign(abs(a[k] - o) - r) > 0) {
        o = circumcenter(a[i], a[j], a[k]);
        r = abs(a[k] - o);
      }
    }
  }
  return C(o, r);
}
```

### 3D Geometry

```cpp
struct V {
  ld x, y, z;
  constexpr V(ld x = 0, ld y = 0, ld z = 0) : x(x), y(y), z(z) {}
  V operator+(V b) const { return V(x + b.x, y + b.y, z + b.z); }
  V operator-(V b) const { return V(x - b.x, y - b.y, z - b.z); }
  V operator*(ld k) const { return V(x * k, y * k, z * k); }
  V operator/(ld k) const { return V(x / k, y / k, z / k); }
  ld len() const { return sqrt(len2()); }
  ld len2() const { return x * x + y * y + z * z; }
};

ostream& operator<<(ostream& os, V p) { return os << "(" << p.x << "," << p.y << "," << p.z << ")"; }
istream& operator>>(istream& is, V& p) { return is >> p.x >> p.y >> p.z; }

ld dist(V a, V b) { return (b - a).len(); }
ld dot(V a, V b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
V det(V a, V b) { return V(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
ld mix(V a, V b, V c) { return dot(a, det(b, c)); }
```
