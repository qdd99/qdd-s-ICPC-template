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

### updmax/min

```cpp
template<class T> inline bool updmax(T &a, T b) { return a < b ? a = b, 1 : 0; }
template<class T> inline bool updmin(T &a, T b) { return a > b ? a = b, 1 : 0; }
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

### bits/stdc++.h

```cpp
// C
#include <cassert>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

// C++
#include <algorithm>
#include <bitset>
#include <complex>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <utility>
#include <vector>

// C++11
#include <chrono>
#include <random>
#include <unordered_map>
#include <unordered_set>
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
