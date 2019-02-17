## 4.4 数学

### GCD & LCM

```cpp
long long gcd(long long a, long long b) { return b ? gcd(b, a % b) : a; }
long long lcm(long long a, long long b) { return a / gcd(a, b) * b; }
```

### 快速幂 & 快速乘

```cpp
// 注意 b = 0, MOD = 1 的情况
long long powMod(long long a, long long b) {
    long long ans = 1;
    for (a %= MOD; b; b >>= 1) {
        if (b & 1) ans = ans * a % MOD;
        a = a * a % MOD;
    }
    return ans;
}

// 模数爆int时使用
long long mul(long long a, long long b) {
    long long ans = 0;
    for (a %= MOD; b; b >>= 1) {
        if (b & 1) ans = (ans + a) % MOD;
        a = (a << 1) % MOD;
    }
    return ans;
}
```

### 矩阵快速幂

```cpp
struct Mat {
    long long m[3][3] = {0};
    long long * operator [] (int i) { return m[i]; }
    void one() { for (int i = 0; i < 3; i++) m[i][i] = 1; }
};

Mat mul(Mat &a, Mat &b) {
    Mat ans;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            if (a[i][j])
                for (int k = 0; k < 3; k++)
                    ans[i][k] = (ans[i][k] + a[i][j] * b[j][k]) % MOD;
    return ans;
}

Mat pow(Mat &a, long long b) {
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
bool Rabin_Miller(long long p, long long a) {
    if (p == 2) return 1;
    if (p & 1 == 0 || p == 1) return 0;
    long long d = p - 1;
    while (!(d & 1)) d >>= 1;
    long long m = powMod(a, d, p);
    if (m == 1) return 1;
    while (d < p) {
        if (m == p - 1) return 1;
        d <<= 1;
        m = mul(m, m, p);
    }
    return 0;
}

bool isPrime(long long x) {
    if (x == 3 || x == 5) return 1;
    static long long prime[7] = {2, 307, 7681, 36061, 555097, 4811057, 1007281591};
    for (int i = 0; i < 7; i++) {
        if (x == prime[i]) return 1;
        if (!Rabin_Miller(x, prime[i])) return 0;
    }
    return 1;
}
```

### 素数筛

```cpp
bitset<MAXN> isprime;

void init_prime() {
    isprime.set();
    isprime[0] = isprime[1] = 0;
    for (int i = 2; i * i < MAXN - 5; i++) {
        if (isprime[i]) {
            for (int j = i * i; j < MAXN - 5; j += i) isprime[j] = 0;
        }
    }
}
```

### 找因数

```cpp
// O(sqrt(n))
void getf(int x, vector<int> &v) {
    for (int i = 1; i * i <= x; i++) {
        if (x % i == 0) {
            v.push_back(i);
            if (x / i != i) v.push_back(x / i);
        }
    }
    sort(v.begin(), v.end());
}
```

### 找质因数

```cpp
// O(sqrt(n))，无重复
void getf(int x, vector<int> &v) {
    for (int i = 2; i * i <= x; i++) {
        if (x % i == 0) {
            v.push_back(i);
            while (x % i == 0) x /= i;
        }
    }
    if (x != 1) v.push_back(x);
}

// O(sqrt(n))，有重复
void getf(int x, vector<int> &v) {
    for (int i = 2; i * i <= x; i++) {
        while (x % i == 0) {
            v.push_back(i);
            x /= i;
        }
    }
    if (x != 1) v.push_back(x);
}

// 预处理 O(nloglogn)
int spf[MAXN];

void init_spf() {
    for (int i = 2; i < MAXN - 5; i++) {
        if (!spf[i]) {
            for (int j = i; j < MAXN - 5; j += i) {
                if (!spf[j]) spf[j] = i;
            }
        }
    }
}

// O(logn)，无重复
void getf(int x, vector<int> &v) {
    while (x > 1) {
        int p = spf[x];
        v.push_back(p);
        while (x % p == 0) x /= p;
    }
}

// O(logn)，有重复
void getf(int x, vector<int> &v) {
    while (x > 1) {
        int p = spf[x];
        while (x % p == 0) {
            v.push_back(p);
            x /= p;
        }
    }
}
```

### 欧拉函数

```cpp
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

### 线性筛

```cpp
// 欧拉线性筛
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

// 莫比乌斯线性筛
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

### EXGCD

```cpp
long long exgcd(long long a, long long b, long long &x, long long &y) {
    if (b == 0) {
        x = 1;
        y = 0;
        return a;
    }
    long long d = exgcd(b, a % b, y, x);
    y -= a / b * x;
    return d;
}
```

### 逆元

```cpp
long long inv(long long x) { return powMod(x, MOD - 2); }

// EXGCD
// gcd(a, p) = 1时有逆元
long long inv(long long a, long long p) {
    long long x, y;
    long long d = exgcd(a, p, x, y);
    if (d == 1) return (x % p + p) % p;
    return -1;
}

// 逆元打表
long long inv[MAXN];

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
long long C[MAXN][MAXN];

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
long long fac[MAXN], ifac[MAXN];

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

long long C(int n, int m) {
    if (n < m || m < 0) return 0;
    return fac[n] * ifac[m] % MOD * ifac[n - m] % MOD;
}

// Lucas
long long C(long long n, long long m) {
    if (n < m || m < 0) return 0;
    if (n < MOD && m < MOD) return fac[n] * ifac[m] % MOD * ifac[n - m] % MOD;
    return C(n / MOD, m / MOD) * C(n % MOD, m % MOD) % MOD;
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
