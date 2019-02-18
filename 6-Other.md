## 4.6 杂项

### 二分答案

```cpp
// 可行下界
while (l < r) {
    mid = (l + r) / 2;
    if (check(mid)) r = mid;
    else l = mid + 1;
}

// 可行上界
while (l < r) {
    mid = (l + r + 1) / 2;
    if (check(mid)) l = mid;
    else r = mid - 1;
}
```

### 三分

```cpp
// 实数范围
double l, r, mid1, mid2;
for (int i = 0; i < 75; i++) {
    mid1 = (l * 5 + r * 4) / 9;
    mid2 = (l * 4 + r * 5) / 9;
    if (f(mid1) > f(mid2)) r = mid2; // 单峰函数取'>'号，单谷函数取'<'号
    else l = mid1;
}

// 整数范围
int l, r, mid1, mid2;
while (l < r - 2) {
    mid1 = (l + r) / 2;
    mid2 = mid1 + 1;
    if (f(mid1) > f(mid2)) r = mid2; // 单峰函数取'>'号，单谷函数取'<'号
    else l = mid1;
}
int maxval = f(l), ans = l;
for (int i = l + 1; i <= r; i++) {
    if (updmax(maxval, f(i))) ans = i;
}
```

### 日期

```cpp
int date_to_int(int y, int m, int d) {
    return
    1461 * (y + 4800 + (m - 14) / 12) / 4 +
    367 * (m - 2 - (m - 14) / 12 * 12) / 12 -
    3 * ((y + 4900 + (m - 14) / 12) / 100) / 4 +
    d - 32075;
}

void int_to_date(int jd, int &y, int &m, int &d) {
    int x, n, i, j;

    x = jd + 68569;
    n = 4 * x / 146097;
    x -= (146097 * n + 3) / 4;
    i = (4000 * (x + 1)) / 1461001;
    x -= 1461 * i / 4 - 31;
    j = 80 * x / 2447;
    d = x - 2447 * j / 80;
    x = j / 11;
    m = j + 2 - 12 * x;
    y = 100 * (n - 49) + i + x;
}
```

### upd

```cpp
template<typename T> inline bool updmax(T &a, T b) { return a < b ? a = b, 1 : 0; }
template<typename T> inline bool updmin(T &a, T b) { return a > b ? a = b, 1 : 0; }
```

### Debug

```cpp
// 标准版
#define dbg(x) cerr << #x << " = ", dprint(x), cerr << endl

void dprint(string s) { cerr << '"' << s << '"'; }

template <class T> void dprint(T x) { cerr << x; }

template <class T1, class T2>
void dprint(pair<T1, T2> p) {
    cerr << "(";
    dprint(p.first);
    cerr << ", ";
    dprint(p.second);
    cerr << ")";
}

template <template <class...> class T, class t>
void dprint(T<t> v) {
    bool first = true;
    cerr << "{";
    for (auto it : v) {
        if (!first) cerr << ", ";
        first = false;
        dprint(it);
    }
    cerr << "}";
}

template <class T1, class T2>
void dprint(map<T1, T2> v) {
    bool first = true;
    cerr << "{";
    for (auto it : v) {
        if (!first) cerr << ", ";
        first = false;
        dprint(it);
    }
    cerr << "}";
}

template <class T>
void dprint(priority_queue<T> q) {
    cerr << "{";
    while (!q.empty()) {
        dprint(q.top());
        cerr << (q.size() > 1 ? ", " : "");
        q.pop();
    }
    cerr << "}";
}

template <class T>
void dprint(priority_queue<T, vector<T>, greater<T> > q) {
    cerr << "{";
    while (!q.empty()) {
        dprint(q.top());
        cerr << (q.size() > 1 ? ", " : "");
        q.pop();
    }
    cerr << "}";
}

template <class T>
void dprint(queue<T> q) {
    cerr << "{";
    while (!q.empty()) {
        dprint(q.front());
        cerr << (q.size() > 1 ? ", " : "");
        q.pop();
    }
    cerr << "}";
}

template <class T>
void dprint(stack<T> q_) {
    stack<T> q;
    while (!q_.empty()) {
        q.push(q_.top());
        q_.pop();
    }
    cerr << "{";
    while (!q.empty()) {
        dprint(q.top());
        cerr << (q.size() > 1 ? ", " : "");
        q.pop();
    }
    cerr << "}";
}

// 压行版
#define dbg(x) cerr << #x << " = ", dprint(x), cerr << endl
void dprint(string s) { cerr << '"' << s << '"'; }
template<class T> void dprint(T x) { cerr << x; }
template<class T1, class T2> void dprint(pair<T1, T2> p) { cerr << "("; dprint(p.first); cerr << ", "; dprint(p.second); cerr << ")"; }
template<template<class...> class T, class t> void dprint(T<t> v) { bool first = true; cerr << "{"; for (auto it : v) { if (!first) cerr << ", "; first = false; dprint(it); } cerr << "}"; }
template<class T1, class T2> void dprint(map<T1, T2> v) { bool first = true; cerr << "{"; for (auto it : v) { if (!first) cerr << ", "; first = false; dprint(it); } cerr << "}"; }
template<class T> void dprint(priority_queue<T> q) { cerr << "{"; while (!q.empty()) { dprint(q.top()); cerr << (q.size() > 1 ? ", " : ""); q.pop(); } cerr << "}"; }
template<class T> void dprint(priority_queue<T, vector<T>, greater<T> > q) { cerr << "{"; while (!q.empty()) { dprint(q.top()); cerr << (q.size() > 1 ? ", " : ""); q.pop(); } cerr << "}"; }
template<class T> void dprint(queue<T> q) { cerr << "{"; while (!q.empty()) { dprint(q.front()); cerr << (q.size() > 1 ? ", " : ""); q.pop(); } cerr << "}"; }
template<class T> void dprint(stack<T> q_) { stack<T> q; while (!q_.empty()) { q.push(q_.top()); q_.pop(); } cerr << "{"; while (!q.empty()) { dprint(q.top()); cerr << (q.size() > 1 ? ", " : ""); q.pop(); } cerr << "}"; }
```
