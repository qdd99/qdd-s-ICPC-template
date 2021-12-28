## 杂项

### 防爆vector

```cpp
template<class T>
class vector_s : public vector<T> {
public:
    vector_s(size_t n = 0, const T& x = T()) : vector<T>(n, x) {}
    T& operator [] (size_t n) { return this->at(n); }
    const T& operator [] (size_t n) const { return this->at(n); }
};

#define vector vector_s
```

### pair_hash

```cpp
template<class T1, class T2>
struct pair_hash {
    size_t operator () (const pair<T1, T2>& p) const {
        return hash<T1>()(p.first) * 19260817 + hash<T2>()(p.second);
    }
};

unordered_set<pair<int, int>, pair_hash<int, int> > st;
unordered_map<pair<int, int>, int, pair_hash<int, int> > mp;
```

### updmax/min

```cpp
template<class T> inline bool updmax(T &a, T b) { return a < b ? a = b, 1 : 0; }
template<class T> inline bool updmin(T &a, T b) { return a > b ? a = b, 1 : 0; }
```

### 离散化

```cpp
// 重复元素id不同
template<class T>
vector<int> dc(const vector<T>& a, int start_id) {
    int n = a.size();
    vector<pair<T, int> > t(n);
    for (int i = 0; i < n; i++) {
        t[i] = make_pair(a[i], i);
    }
    sort(t.begin(), t.end());
    vector<int> id(n);
    for (int i = 0; i < n; i++) {
        id[t[i].second] = start_id + i;
    }
    return id;
}

// 重复元素id相同
template<class T>
vector<int> unique_dc(const vector<T>& a, int start_id) {
    int n = a.size();
    vector<T> t(a);
    sort(t.begin(), t.end());
    t.resize(unique(t.begin(), t.end()) - t.begin());
    vector<int> id(n);
    for (int i = 0; i < n; i++) {
        id[i] = start_id + lower_bound(t.begin(), t.end(), a[i]) - t.begin();
    }
    return id;
}
```

### 合并同类项

```cpp
template <class T>
vector<pair<T, int> > norm(vector<T>& v) {
    // sort(v.begin(), v.end());
    vector<pair<T, int> > p;
    for (int i = 0; i < (int)v.size(); i++) {
        if (i == 0 || v[i] != v[i - 1])
            p.emplace_back(v[i], 1);
        else
            p.back().second++;
    }
    return p;
}
```

### 加强版优先队列

```cpp
struct heap {
    priority_queue<int> q1, q2;
    void push(int x) { q1.push(x); }
    void erase(int x) { q2.push(x); }
    int top() {
        while (q2.size() && q1.top() == q2.top()) q1.pop(), q2.pop();
        return q1.top();
    }
    void pop() {
        while (q2.size() && q1.top() == q2.top()) q1.pop(), q2.pop();
        q1.pop();
    }
    int size() { return q1.size() - q2.size(); }
};
```

### 分数

```cpp
struct Frac {
    ll x, y;

    Frac(ll p = 0, ll q = 1) {
        ll d = __gcd(p, q);
        x = p / d, y = q / d;
        if (y < 0) x = -x, y = -y;
    }

    Frac operator + (const Frac& b) { return Frac(x * b.y + y * b.x, y * b.y); }
    Frac operator - (const Frac& b) { return Frac(x * b.y - y * b.x, y * b.y); }
    Frac operator * (const Frac& b) { return Frac(x * b.x, y * b.y); }
    Frac operator / (const Frac& b) { return Frac(x * b.y, y * b.x); }
};

ostream& operator << (ostream& os, const Frac& f) {
    if (f.y == 1) return os << f.x;
    else return os << f.x << '/' << f.y;
}
```

### 二分答案

```cpp
// 二分闭区间[l, r]
template <class T, class F>
T min_left(T l, T r, F f) {
    while (l < r) {
        T p = l + (r - l) / 2;
        f(p) ? r = p : l = p + 1;
    }
    return l;
}

template <class T, class F>
T max_right(T l, T r, F f) {
    while (l < r) {
        T p = l + (r - l + 1) / 2;
        f(p) ? l = p : r = p - 1;
    }
    return l;
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
// 0 ~ 6 对应 周一 ~ 周日
int zeller(int y, int m, int d) {
    if (m <= 2) m += 12, y--;
    return (d + 2 * m + 3 * (m + 1) / 5 + y + y / 4 - y / 100 + y / 400) % 7;
}

// date_to_int(1, 1, 1) = 1721426
// date_to_int(2019, 10, 27) = 2458784
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

### 子集枚举

```cpp
// 枚举真子集
for (int t = (x - 1) & x; t; t = (t - 1) & x)

// 枚举大小为 k 的子集
// 注意 k 不能为 0
void subset(int k, int n) {
    int t = (1 << k) - 1;
    while (t < (1 << n)) {
        // do something
        int x = t & -t, y = t + x;
        t = ((t & ~y) / x >> 1) | y;
    }
}
```

### 数位 dp

```cpp
// 小于等于 x 的 base 进制下回文数个数
ll dp[20][20][20][2], tmp[20], a[20];

ll dfs(ll base, ll pos, ll len, ll s, bool limit) {
    if (pos == -1) return s;
    if (!limit && dp[base][pos][len][s] != -1) return dp[base][pos][len][s];
    ll ret = 0;
    ll ed = limit ? a[pos] : base - 1;
    for (int i = 0; i <= ed; i++) {
        tmp[pos] = i;
        if (len == pos)
            ret += dfs(base, pos - 1, len - (i == 0), s, limit && i == a[pos]);
        else if (s && pos < (len + 1) / 2)
            ret += dfs(base, pos - 1, len, tmp[len - pos] == i, limit && i == a[pos]);
        else
            ret += dfs(base, pos - 1, len, s, limit && i == a[pos]);
    }
    if (!limit) dp[base][pos][len][s] = ret;
    return ret;
}

ll solve(ll x, ll base) {
    memset(dp, -1, sizeof(dp));
    ll sz = 0;
    while (x) {
        a[sz++] = x % base;
        x /= base;
    }
    return dfs(base, sz - 1, sz - 1, 1, true);
}
```

### 大范围洗牌算法

```cpp
vector<int> randset(int l, int r, int k) {
    // assert(l <= r && k <= r - l + 1);
    unordered_map<int, int> p;
    for (int i = l; i < l + k; i++) p[i] = i;
    for (int i = l; i < l + k; i++) {
        int j = randint(i, r);
        if (!p.count(j)) p[j] = j;
        swap(p[i], p[j]);
    }
    vector<int> a(k);
    for (int i = 0; i < k; i++) a[i] = p[i + l];
    return a;
}
```

### 表达式求值

```py
print(input()) # Python2
print(eval(input())) # Python3
```

### 对拍

+ *unix

```bash
#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"

g++ gen.cpp -o gen -O2 -std=c++11
g++ my.cpp -o my -O2 -std=c++11
g++ std.cpp -o std -O2 -std=c++11

while true
do
    ./gen > in.txt
    ./std < in.txt > stdout.txt
    ./my < in.txt > myout.txt

    if test $? -ne 0
    then
        printf "RE\n"
        exit 0
    fi

    if diff stdout.txt myout.txt
    then
        printf "AC\n"
    else
        printf "WA\n"
        exit 0
    fi
done
```

+ Windows

```
@echo off

g++ gen.cpp -o gen.exe -O2 -std=c++11
g++ my.cpp -o my.exe -O2 -std=c++11
g++ std.cpp -o std.exe -O2 -std=c++11

:loop
    gen.exe > in.txt
    std.exe < in.txt > stdout.txt
    my.exe < in.txt > myout.txt
    if errorlevel 1 (
        echo RE
        pause
        exit
    )
    fc stdout.txt myout.txt
    if errorlevel 1 (
        echo WA
        pause
        exit
    )
goto loop
```

### Java

+ Main

```java
import java.io.*;
import java.util.*;

public class Main {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        PrintStream out = System.out;

    }
}
```

+ 皮特老师读入挂

```java
public class Main {
    public static void main(String[] args) {
        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        InputReader in = new InputReader(inputStream);
        PrintWriter out = new PrintWriter(outputStream);

        out.close();
    }

    static class InputReader {
        public BufferedReader reader;
        public StringTokenizer tokenizer;

        public InputReader(InputStream stream) {
            reader = new BufferedReader(new InputStreamReader(stream), 32768);
            tokenizer = null;
        }

        public String next() {
            while (tokenizer == null || !tokenizer.hasMoreTokens()) {
                try {
                    tokenizer = new StringTokenizer(reader.readLine());
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
            return tokenizer.nextToken();
        }

        public int nextInt() {
            return Integer.parseInt(next());
        }
    }
}
```

+ 大整数

```java
import java.math.BigInteger;

BigInteger.ZERO
BigInteger.ONE
BigInteger.TWO // since Java 9
BigInteger.TEN
BigInteger.valueOf(2)

BigInteger abs()
BigInteger negate() // -this

BigInteger add​(BigInteger x)
BigInteger subtract​(BigInteger x)
BigInteger multiply​(BigInteger x)
BigInteger divide​(BigInteger x)

BigInteger pow​(int exp)
BigInteger sqrt() // since Java 9

BigInteger mod​(BigInteger m)
BigInteger modPow​(BigInteger exp, BigInteger m)
BigInteger modInverse​(BigInteger m)

boolean isProbablePrime​(int certainty) // probability: 1 - (1/2) ^ (certainty)

BigInteger gcd​(BigInteger x)

BigInteger not() // ~this
BigInteger and​(BigInteger x)
BigInteger or​(BigInteger x)
BigInteger xor​(BigInteger x)
BigInteger shiftLeft​(int n)
BigInteger shiftRight​(int n)

int compareTo​(BigInteger x) // -1, 0, 1
BigInteger max​(BigInteger x)
BigInteger min​(BigInteger x)

int intValue()
long longValue()
String toString()

public static BigInteger getsqrt(BigInteger n) {
    if (n.compareTo(BigInteger.ZERO) <= 0) return n;
    BigInteger x, xx, txx;
    xx = x = BigInteger.ZERO;
    for (int t = n.bitLength() / 2; t >= 0; t--) {
        txx = xx.add(x.shiftLeft(t + 1)).add(BigInteger.ONE.shiftLeft(t + t));
        if (txx.compareTo(n) <= 0) {
            x = x.add(BigInteger.ONE.shiftLeft(t));
            xx = txx;
        }
    }
    return x;
}
```

+ 浮点数格式

```java
import java.text.DecimalFormat;

DecimalFormat fmt;

// String s = fmt.format(...)

// round to at most 2 digits, leave of digits if not needed
fmt = new DecimalFormat("#.##");
// 12345.6789 -> "12345.68"
// 12345.0 -> "12345"
// 0.0 -> "0"
// 0.01 -> ".1"

// round to precisely 2 digits
fmt = new DecimalFormat("#.00");
// 12345.6789 -> "12345.68"
// 12345.0 -> "12345.00"
// 0.0 -> ".00"

// round to precisely 2 digits, force leading zero
fmt = new DecimalFormat("0.00");
// 12345.6789 -> "12345.68"
// 12345.0 -> "12345.00"
// 0.0 -> "0.00"

// round to precisely 2 digits, force leading zeros
fmt = new DecimalFormat("000000000.00");
// 12345.6789 -> "000012345.68"
// 12345.0 -> "000012345.00"
// 0.0 -> "000000000.00"
```

### pb_ds

```cpp
// 平衡树
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
template<class T>
using rank_set = tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;
template<class Key, class T>
using rank_map = tree<Key, T, less<Key>, rb_tree_tag, tree_order_statistics_node_update>;

// 优先队列
#include <ext/pb_ds/priority_queue.hpp>
using namespace __gnu_pbds;
template<class T, class Cmp = less<T> >
using pair_heap = __gnu_pbds::priority_queue<T, Cmp>;
```
