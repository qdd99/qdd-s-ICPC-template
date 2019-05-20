## 数学

### GCD & LCM

```cpp
ll gcd(ll a, ll b) { return b ? gcd(b, a % b) : a; }
ll lcm(ll a, ll b) { return a / gcd(a, b) * b; }
```

### 快速幂 & 快速乘

```cpp
// 注意 b = 0, MOD = 1 的情况
ll powMod(ll a, ll b) {
    ll ans = 1;
    for (a %= MOD; b; b >>= 1) {
        if (b & 1) ans = ans * a % MOD;
        a = a * a % MOD;
    }
    return ans;
}

// 模数爆int时使用
ll mul(ll a, ll b) {
    ll ans = 0;
    for (a %= MOD; b; b >>= 1) {
        if (b & 1) ans = (ans + a) % MOD;
        a = (a << 1) % MOD;
    }
    return ans;
}

// O(1)
ll mul(ll a, ll b) {
    return (ll)(__int128(a) * b % MOD);
}
```

### 矩阵快速幂

```cpp
const int MAT_SZ = 3;

struct Mat {
    ll m[MAT_SZ][MAT_SZ] = {{0}};
    ll * operator [] (int i) { return m[i]; }
    void one() { for (int i = 0; i < MAT_SZ; i++) m[i][i] = 1; }
};

Mat mul(Mat &a, Mat &b) {
    Mat ans;
    for (int i = 0; i < MAT_SZ; i++)
        for (int j = 0; j < MAT_SZ; j++)
            if (a[i][j])
                for (int k = 0; k < MAT_SZ; k++)
                    ans[i][k] = (ans[i][k] + a[i][j] * b[j][k]) % MOD;
    return ans;
}

Mat pow(Mat &a, ll b) {
    Mat ans;
    ans.one();
    while (b) {
        if (b & 1) ans = mul(a, ans);
        b >>= 1;
        a = mul(a, a);
    }
    return ans;
}
```

### 素数判断

```cpp
bool isPrime(int x) {
    if (x < 2) return false;
    for (int i = 2; i * i <= x; i++) if (x % i == 0) return false;
    return true;
}

// O(logn)
// 前置：快速幂、快速乘
// int范围只需检查2, 7, 61
bool Rabin_Miller(ll p, ll a) {
    if (p == 2) return 1;
    if (p & 1 == 0 || p == 1) return 0;
    ll d = p - 1;
    while (!(d & 1)) d >>= 1;
    ll m = powMod(a, d, p);
    if (m == 1) return 1;
    while (d < p) {
        if (m == p - 1) return 1;
        d <<= 1;
        m = mul(m, m, p);
    }
    return 0;
}

bool isPrime(ll x) {
    if (x == 3 || x == 5) return 1;
    static ll prime[7] = {2, 307, 7681, 36061, 555097, 4811057, 1007281591};
    for (int i = 0; i < 7; i++) {
        if (x == prime[i]) return 1;
        if (!Rabin_Miller(x, prime[i])) return 0;
    }
    return 1;
}
```

### 线性筛

```cpp
// 注意 0 和 1 不是素数
bool vis[MAXN];
int prime[MAXN];

void get_prime() {
    int tot = 0;
    for (int i = 2; i < MAXN - 5; i++) {
        if (!vis[i]) prime[tot++] = i;
        for (int j = 0; j < tot; j++) {
            int d = i * prime[j];
            if (d >= MAXN - 5) break;
            vis[d] = true;
            if (i % prime[j] == 0) break;
        }
    }
}

// 最小素因子
bool vis[MAXN];
int spf[MAXN], prime[MAXN];

void get_spf() {
    int tot = 0;
    for (int i = 2; i < MAXN - 5; i++) {
        if (!vis[i]) {
            prime[tot++] = i;
            spf[i] = i;
        }
        for (int j = 0; j < tot; j++) {
            int d = i * prime[j];
            if (d >= MAXN - 5) break;
            vis[d] = true;
            spf[d] = prime[j];
            if (i % prime[j] == 0) break;
        }
    }
}

// 欧拉函数
bool vis[MAXN];
int phi[MAXN], prime[MAXN];

void get_phi() {
    int tot = 0;
    phi[1] = 1;
    for (int i = 2; i < MAXN - 5; i++) {
        if (!vis[i]) {
            prime[tot++] = i;
            phi[i] = i - 1;
        }
        for (int j = 0; j < tot; j++) {
            int d = i * prime[j];
            if (d >= MAXN - 5) break;
            vis[d] = true;
            if (i % prime[j] == 0) {
                phi[d] = phi[i] * prime[j];
                break;
            }
            else phi[d] = phi[i] * (prime[j] - 1);
        }
    }
}

// 莫比乌斯函数
bool vis[MAXN];
int mu[MAXN], prime[MAXN];

void get_mu() {
    int tot = 0;
    mu[1] = 1;
    for (int i = 2; i < MAXN - 5; i++) {
        if (!vis[i]) {
            prime[tot++] = i;
            mu[i] = -1;
        }
        for (int j = 0; j < tot; j++) {
            int d = i * prime[j];
            if (d >= MAXN - 5) break;
            vis[d] = true;
            if (i % prime[j] == 0) {
                mu[d] = 0;
                break;
            }
            else mu[d] = -mu[i];
        }
    }
}
```

### 找因数

```cpp
// O(sqrt(n))
vector<int> getf(int x) {
    vector<int> v;
    for (int i = 1; i * i <= x; i++) {
        if (x % i == 0) {
            v.push_back(i);
            if (x / i != i) v.push_back(x / i);
        }
    }
    sort(v.begin(), v.end());
    return v;
}
```

### 找质因数

```cpp
// O(sqrt(n))，无重复
vector<int> getf(int x) {
    vector<int> v;
    for (int i = 2; i * i <= x; i++) {
        if (x % i == 0) {
            v.push_back(i);
            while (x % i == 0) x /= i;
        }
    }
    if (x != 1) v.push_back(x);
    return v;
}

// O(sqrt(n))，有重复
vector<int> getf(int x) {
    vector<int> v;
    for (int i = 2; i * i <= x; i++) {
        while (x % i == 0) {
            v.push_back(i);
            x /= i;
        }
    }
    if (x != 1) v.push_back(x);
    return v;
}

// 前置：线性筛
// O(logn)，无重复
vector<int> getf(int x) {
    vector<int> v;
    while (x > 1) {
        int p = spf[x];
        v.push_back(p);
        while (x % p == 0) x /= p;
    }
    return v;
}

// O(logn)，有重复
vector<int> getf(int x) {
    vector<int> v;
    while (x > 1) {
        int p = spf[x];
        while (x % p == 0) {
            v.push_back(p);
            x /= p;
        }
    }
    return v;
}
```

### 欧拉函数

```cpp
// 前置：找质因数（无重复）
int phi(int x) {
    int ret = x;
    vector<int> v = getf(x);
    for (int f : v) ret = ret / f * (f - 1);
    return ret;
}

// O(nloglogn)
int phi[MAXN];

void get_phi() {
    phi[1] = 1;
    for (int i = 2; i < MAXN - 5; i++) {
        if (!phi[i]) {
            for (int j = i; j < MAXN - 5; j += i) {
                if (!phi[j]) phi[j] = j;
                phi[j] = phi[j] / i * (i - 1);
            }
        }
    }
}
```

### EXGCD

```cpp
// ax + by = gcd(a, b)
ll exgcd(ll a, ll b, ll &x, ll &y) {
    if (b == 0) {
        x = 1;
        y = 0;
        return a;
    }
    ll d = exgcd(b, a % b, y, x);
    y -= a / b * x;
    return d;
}
```

### 逆元

```cpp
ll inv(ll x) { return powMod(x, MOD - 2); }

// EXGCD
// gcd(a, p) = 1时有逆元
ll inv(ll a, ll p) {
    ll x, y;
    ll d = exgcd(a, p, x, y);
    if (d == 1) return (x % p + p) % p;
    return -1;
}

// 逆元打表
ll inv[MAXN];

void initInv() {
    inv[1] = 1;
    for (int i = 2; i < MAXN - 5; i++) {
        inv[i] = 1LL * (MOD - MOD / i) * inv[MOD % i] % MOD;
    }
}
```

### 组合数

```cpp
// 组合数打表
ll C[MAXN][MAXN];

void initC() {
    C[0][0] = 1;
    for (int i = 1; i < MAXN - 5; i++) {
        C[i][0] = 1;
        for (int j = 1; j <= i; j++) {
            C[i][j] = (C[i - 1][j] + C[i - 1][j - 1]) % MOD;
        }
    }
}

// 快速组合数取模
// MAXN开2倍上限
ll fac[MAXN], ifac[MAXN];

void initInv() {
    fac[0] = 1;
    for (int i = 1; i < MAXN; i++) {
        fac[i] = fac[i - 1] * i % MOD;
    }
    ifac[MAXN - 1] = powMod(fac[MAXN - 1], MOD - 2);
    for (int i = MAXN - 2; i >= 0; i--) {
        ifac[i] = ifac[i + 1] * (i + 1);
        ifac[i] %= MOD;
    }
}

ll C(int n, int m) {
    if (n < m || m < 0) return 0;
    return fac[n] * ifac[m] % MOD * ifac[n - m] % MOD;
}

// Lucas
ll C(ll n, ll m) {
    if (n < m || m < 0) return 0;
    if (n < MOD && m < MOD) return fac[n] * ifac[m] % MOD * ifac[n - m] % MOD;
    return C(n / MOD, m / MOD) * C(n % MOD, m % MOD) % MOD;
}
```

### 康托展开

```cpp
// 需要预处理阶乘
int cantor(vector<int>& s) {
    int n = s.size(), ans = 0;
    for (int i = 0; i < n - 1; i++) {
        int cnt = 0;
        for (int j = i + 1; j < n; j++) {
            if (s[j] < s[i]) cnt++;
        }
        ans += cnt * fac[n - i - 1];
    }
    return ans + 1;
}

vector<int> inv_cantor(int x, int n) {
    x--;
    vector<int> ans(n), rk(n);
    iota(rk.begin(), rk.end(), 1);
    for (int i = 0; i < n; i++) {
        int t = x / fac[n - i - 1];
        x %= fac[n - i - 1];
        ans[i] = rk[t];
        for (int j = t; rk[j] < n; j++) {
            rk[j] = rk[j + 1];
        }
    }
    return ans;
}
```

### 线性基

```cpp
ll a[65];

void insert(ll x) {
    for (int i = 60; i >= 0; i--) {
        if ((x >> i) & 1) {
            if (a[i]) x ^= a[i];
            else { a[i] = x; break; }
        }
    }
}
```

### 中国剩余定理

```cpp
// 前置：exgcd
ll excrt(vector<ll>& m, vector<ll>& r) {
    ll M = m[0], R = r[0], x, y, d;
    for (int i = 1; i < m.size(); i++) {
        d = exgcd(M, m[i], x, y);
        if ((r[i] - R) % d) return -1;
        x = mul(x, (r[i] - R) / d, m[i] / d);
        R += x * M;
        M = M / d * m[i];
        R %= M;
    }
    return R >= 0 ? R : R + M;
}
```

### 原根

```cpp
// 前置：找质因数（无重复）
ll primitive_root(ll p) {
    vector<ll> facs = getf(p - 1);
    for (ll i = 2; i < p; i++) {
        bool flag = true;
        for (ll x : facs) {
            if (powMod(i, (p - 1) / x, p) == 1) {
                flag = false;
                break;
            }
        }
        if (flag) return i;
    }
    return -1;
}
```

### 离散对数

```cpp
// a ^ x = b (mod p)，要求模数为素数
ll BSGS(ll a, ll b, ll p) {
    a %= p;
    if (!a && !b) return 1;
    if (!a) return -1;
    map<ll, ll> mp;
    ll m = ceil(sqrt(p)), v = 1;
    for (int i = 1; i <= m; i++) {
        (v *= a) %= p;
        mp[v * b % p] = i;
    }
    ll vv = v;
    for (int i = 1; i <= m; i++) {
        auto it = mp.find(vv);
        if (it != mp.end()) return i * m - it->second;
        (vv *= v) %= p;
    }
    return -1;
}

// 模数可以非素数
ll exBSGS(ll a, ll b, ll p) {
    a %= p; b %= p;
    if (a == 0) return b > 1 ? -1 : (b == 0 && p != 1);
    ll c = 0, q = 1;
    for (;;) {
        ll g = gcd(a, p);
        if (g == 1) break;
        if (b == 1) return c;
        if (b % g) return -1;
        ++c; b /= g; p /= g; q = a / g * q % p;
    }
    map<ll, ll> mp;
    ll m = ceil(sqrt(p)), v = 1;
    for (int i = 1; i <= m; i++) {
        (v *= a) %= p;
        mp[v * b % p] = i;
    }
    for (int i = 1; i <= m; i++) {
        (q *= v) %= p;
        auto it = mp.find(q);
        if (it != mp.end()) return i * m - it->second + c;
    }
    return -1;
}

// 已知 x, b, p，求 a
ll SGSB(ll x, ll b, ll p) {
    ll g = primitive_root(p);
    return powMod(g, BSGS(powMod(g, x, p), b, p), p);
}
```

### FFT & NTT & FWT

+ FFT

```cpp
const double PI = acos(-1);
using cp = complex<double>;

int n1, n2, n, k, rev[MAXN];

void fft(vector<cp>& a, int p) {
    for (int i = 0; i < n; i++) if (i < rev[i]) swap(a[i], a[rev[i]]);
    for (int h = 1; h < n; h <<= 1) {
        cp wn(cos(PI / h), p * sin(PI / h));
        for (int i = 0; i < n; i += (h << 1)) {
            cp w(1, 0);
            for (int j = 0; j < h; j++, w *= wn) {
                cp x = a[i + j], y = w * a[i + j + h];
                a[i + j] = x + y, a[i + j + h] = x - y;
            }
        }
    }
    if (p == -1) for (int i = 0; i < n; i++) a[i] /= n;
}

void go(vector<cp>& a, vector<cp>& b) {
    n = 1, k = 0;
    while (n <= n1 + n2) n <<= 1, k++;
    a.resize(n); b.resize(n);
    for (int i = 0; i < n; i++) rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (k - 1));
    fft(a, 1); fft(b, 1);
    for (int i = 0; i < n; i++) a[i] *= b[i];
    fft(a, -1);
}
```

+ NTT

```cpp
const int MOD = 998244353, G = 3, IG = 332748118;

int n1, n2, n, k, rev[MAXN];

void ntt(vector<ll>& a, int p) {
    for (int i = 0; i < n; i++) if (i < rev[i]) swap(a[i], a[rev[i]]);
    for (int h = 1; h < n; h <<= 1) {
        ll wn = powMod(p == 1 ? G : IG, (MOD - 1) / (h << 1));
        for (int i = 0; i < n; i += (h << 1)) {
            ll w = 1;
            for (int j = 0; j < h; j++, (w *= wn) %= MOD) {
                ll x = a[i + j], y = w * a[i + j + h] % MOD;
                a[i + j] = (x + y) % MOD, a[i + j + h] = (x - y + MOD) % MOD;
            }
        }
    }
    if (p == -1) {
        ll ninv = powMod(n, MOD - 2);
        for (int i = 0; i < n; i++) (a[i] *= ninv) %= MOD;
    }
}

void go(vector<ll>& a, vector<ll>& b) {
    n = 1, k = 0;
    while (n <= n1 + n2) n <<= 1, k++;
    a.resize(n); b.resize(n);
    for (int i = 0; i < n; i++) rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (k - 1));
    ntt(a, 1); ntt(b, 1);
    for (int i = 0; i < n; i++) (a[i] *= b[i]) %= MOD;
    ntt(a, -1);
}
```

+ FWT

```cpp
void AND(ll& a, ll& b) { a += b; }
void rAND(ll& a, ll& b) { a -= b; }

void OR(ll& a, ll& b) { b += a; }
void rOR(ll& a, ll& b) { b -= a; }

void XOR (ll& a, ll& b) {
    ll x = a, y = b;
    a = (x + y) % MOD;
    b = (x - y + MOD) % MOD;
}
void rXOR(ll& a, ll& b) {
    static ll inv2 = (MOD + 1) / 2;
    ll x = a, y = b;
    a = (x + y) * inv2 % MOD;
    b = (x - y + MOD) * inv2 % MOD;
}

template<class T>
void fwt(vector<ll>& a, int n, T f) {
    for (int d = 1; d < n; d <<= 1) {
        for (int i = 0; i < n; i += (d << 1)) {
            for (int j = 0; j < d; j++) {
                f(a[i + j], a[i + j + d]);
            }
        }
    }
}
```

### 自适应Simpson积分

```cpp
double simpson(double l, double r) {
    double c = (l + r) / 2;
    return (f(l) + 4 * f(c) + f(r)) * (r - l) / 6;
}

double asr(double l, double r, double eps, double S) {
    double mid = (l + r) / 2;
    double L = simpson(l, mid), R = simpson(mid, r);
    if (fabs(L + R - S) < 15 * eps) return L + R + (L + R - S) / 15;
    return asr(l, mid, eps / 2, L) + asr(mid, r, eps / 2, R);
}

double asr(double l, double r) { return asr(l, r, EPS, simpson(l, r)); }
```

### 拉格朗日插值

```cpp
vector<double> La(vector<pair<double, double> > v) {
    int n = v.size(), t;
    vector<double> ret(n);
    double p, q;
    for (int i = 0; i < n; i++) {
        p = v[i].second;
        for (int j = 0; j < n; j++) {
            p /= (i == j) ? 1 : (v[i].first - v[j].first);
        }
        for (int j = 0; j < (1 << n); j++) {
            q = 1, t = 0;
            for (int k = 0; k < n; k++) {
                if (i == k) continue;
                if ((j >> k) & 1) q *= -v[k].first;
                else t++;
            }
            ret[t] += p * q / 2;
        }
    }
    return ret;
}
```
