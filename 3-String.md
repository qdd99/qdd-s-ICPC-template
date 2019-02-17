## 4.3 字符串

### 哈希

```cpp
// open hack不要用哈希
using ull = unsigned long long;

const int x = 135, p1 = 1e9 + 7, p2 = 1e9 + 9;

int n;
char s[MAXN];
ull xp1[MAXN], xp2[MAXN], h[MAXN];

void init_xp() {
    xp1[0] = xp2[0] = 1;
    for (int i = 1; i < MAXN; i++) {
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

### Trie

```cpp
// 01 Trie
struct Trie {
    int t[31 * MAXN][2], sz;

    void init() {
        memset(t, 0, sizeof(t));
        sz = 2;
    }

    void insert(int x) {
        int p = 1;
        for (int i = 30; i >= 0; i--) {
            bool d = (x >> i) & 1;
            if (!t[p][d]) t[p][d] = sz++;
            p = t[p][d];
        }
    }
};
```
