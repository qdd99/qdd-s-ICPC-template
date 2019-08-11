## 字符串

### 哈希

```cpp
// open hack不要用哈希
using ull = unsigned long long;

const int x = 135, p1 = 1e9 + 7, p2 = 1e9 + 9;

int n;
char s[N];
ull xp1[N], xp2[N], h[N];

void init_xp() {
    xp1[0] = xp2[0] = 1;
    for (int i = 1; i < N; i++) {
        xp1[i] = xp1[i - 1] * x % p1;
        xp2[i] = xp2[i - 1] * x % p2;
    }
}

void init_hash() {
    ull res1 = 0, res2 = 0;
    h[n + 1] = 0;
    for (int i = n; i >= 0; i--) {
        res1 = (res1 * x + s[i]) % p1;
        res2 = (res2 * x + s[i]) % p2;
        h[i] = (res1 << 32) | res2;
    }
}

ull get_hash(int l, int r) {
    r++;
    int len = r - l;
    unsigned int mask32 = ~(0u);
    ull l1 = h[l] >> 32, r1 = h[r] >> 32;
    ull l2 = h[l] & mask32, r2 = h[r] & mask32;
    ull res1 = (l1 - r1 * xp1[len] % p1 + p1) % p1;
    ull res2 = (l2 - r2 * xp2[len] % p2 + p2) % p2;
    return (res1 << 32) | res2;
}
```

### Manacher

```cpp
// "aba" => "#a#b#a#"
string make(string& s) {
    string t = "#";
    for (int i = 0; i < s.size(); i++) {
        t.push_back(s[i]);
        t.push_back('#');
    }
    return t;
}

void manacher(string& s, vector<int>& d) {
    int n = s.size();
    d.resize(n);
    for (int i = 0, l = 0, r = -1; i < n; i++) {
        int k = (i > r) ? 1 : min(d[l + r - i], r - i);
        while (i - k >= 0 && i + k < n && s[i - k] == s[i + k]) k++;
        d[i] = --k;
        if (i + k > r) {
            l = i - k;
            r = i + k;
        }
    }
}
```

### KMP

```cpp
// 前缀函数（每一个前缀的最长公共前后缀）
void get_pi(const string& s, vector<int>& a) {
    int n = s.size(), j = 0;
    a.resize(n);
    for (int i = 1; i < n; i++) {
        while (j && s[j] != s[i]) j = a[j - 1];
        if (s[j] == s[i]) j++;
        a[i] = j;
    }
}

void kmp(const string& s, vector<int>& a, const string& t) {
    int j = 0;
    for (int i = 0; i < t.size(); i++) {
        while (j && s[j] != t[i]) j = a[j - 1];
        if (s[j] == t[i]) j++;
        if (j == s.size()) {
            // ...
            j = a[j - 1]; // 允许重叠匹配 j = 0 不允许
        }
    }
}

// Z函数（每一个后缀和该字符串的最长公共前缀）
void get_z(const string& s, vector<int>& z) {
    int n = s.size(), l = 0, r = 0;
    z.resize(n);
    for (int i = 1; i < n; i++) {
        if (i <= r) z[i] = min(r - i + 1, z[i - l]);
        while (i + z[i] < n && s[z[i]] == s[i + z[i]]) z[i]++;
        if (i + z[i] - 1 > r) {
            l = i;
            r = i + z[i] - 1;
        }
    }
}
```

### Trie

```cpp
// 01 Trie
struct Trie {
    int t[31 * N][2], sz;

    void init() {
        memset(t, 0, 2 * (sz + 2) * sizeof(int));
        sz = 1;
    }

    void insert(int x) {
        int p = 0;
        for (int i = 30; i >= 0; i--) {
            bool d = (x >> i) & 1;
            if (!t[p][d]) t[p][d] = sz++;
            p = t[p][d];
        }
    }
};

// 正常Trie
struct Trie {
    int t[N][26], sz, cnt[N];

    void init() {
        memset(t, 0, 26 * (sz + 2) * sizeof(int));
        memset(cnt, 0, (sz + 2) * sizeof(int));
        sz = 1;
    }

    void insert(const string& s) {
        int p = 0;
        for (char c : s) {
            int d = c - 'a';
            if (!t[p][d]) t[p][d] = sz++;
            p = t[p][d];
        }
        cnt[p]++;
    }
};
```

### AC 自动机

```cpp
struct ACA {
    int t[N][26], sz, fail[N], nxt[N], cnt[N];

    void init() {
        memset(t, 0, 26 * (sz + 2) * sizeof(int));
        memset(fail, 0, (sz + 2) * sizeof(int));
        memset(nxt, 0, (sz + 2) * sizeof(int));
        memset(cnt, 0, (sz + 2) * sizeof(int));
        sz = 1;
    }

    void insert(const string& s) {
        int p = 0;
        for (char c : s) {
            int d = c - 'a';
            if (!t[p][d]) t[p][d] = sz++;
            p = t[p][d];
        }
        cnt[p]++;
    }

    void build() {
        queue<int> q;
        for (int i = 0; i < 26; i++) {
            if (t[0][i]) q.push(t[0][i]);
        }
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            for (int i = 0; i < 26; i++) {
                int& v = t[u][i];
                if (v) {
                    fail[v] = t[fail[u]][i];
                    nxt[v] = cnt[fail[v]] ? fail[v] : nxt[fail[v]];
                    q.push(v);
                } else {
                    v = t[fail[u]][i];
                }
            }
        }
    }
};
```

### 后缀数组

```cpp
// 下标从1开始
// sa[i]: 排名为i的后缀位置
// rk[i]: 第i个后缀的排名
// ht[i]: LCP(sa[i], sa[i - 1])
struct SA {
    int n, m;
    vector<int> a, d, sa, rk, ht;

    void rsort() {
        vector<int> c(m + 1);
        for (int i = 1; i <= n; i++) c[rk[d[i]]]++;
        for (int i = 1; i <= m; i++) c[i] += c[i - 1];
        for (int i = n; i; i--) sa[c[rk[d[i]]]--] = d[i];
    }

    SA(const string& s) : n(s.size()), m(128), a(n + 1), d(n + 1), sa(n + 1), rk(n + 1), ht(n + 1) {
        for (int i = 1; i <= n; i++) { rk[i] = a[i] = s[i - 1]; d[i] = i; }
        rsort();
        for (int j = 1, i, k; k < n; m = k, j <<= 1) {
            for (i = n - j + 1, k = 0; i <= n; i++) d[++k] = i;
            for (i = 1; i <= n; i++) if (sa[i] > j) d[++k] = sa[i] - j;
            rsort(); swap(rk, d); rk[sa[1]] = k = 1;
            for (i = 2; i <= n; i++) {
                rk[sa[i]] = (d[sa[i]] == d[sa[i - 1]] && d[sa[i] + j] == d[sa[i - 1] + j]) ? k : ++k;
            }
        }
        int j, k = 0;
        for (int i = 1; i <= n; ht[rk[i++]] = k) {
            for (k ? k-- : k, j = sa[rk[i] - 1]; a[i + k] == a[j + k]; ++k);
        }
    }
};
```
