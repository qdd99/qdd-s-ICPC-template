## 4.3 字符串

### 哈希

```cpp
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
    for (int j = n; j >= 0; j--) {
        res1 = (res1 * x + s[j]) % p1;
        res2 = (res2 * x + s[j]) % p2;
        h[j] = (res1 << 32) | res2;
    }
}

ull get_substring_hash(int l, int r) {
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
