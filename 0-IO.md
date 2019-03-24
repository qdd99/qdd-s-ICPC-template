## 输入 & 输出

### 特殊格式

```cpp
long double %Lf
unsigned int %u
unsigned long long %llu

cout << fixed << setprecision(15);
```

### 文件和流同步

```cpp
freopen("in.txt", "r", stdin);

ios::sync_with_stdio(false);
cin.tie(0);
```

### 程序计时

```cpp
fprintf(stderr, "%f\n", (double)clock() / CLOCKS_PER_SEC);
```

### 整行读入

```cpp
scanf("%[^\n]", s)  // 需测试是否可用
getline(cin, s)
```

### 读到文件尾

```cpp
while (cin) {}
while (~scanf) {}
```

### int128

```cpp
// 需测试是否可用
inline __int128 get128() {
    __int128 x = 0, sgn = 1;
    char c;
    for (c = getchar(); c < '0' || c > '9'; c = getchar()) if (c == '-') sgn = -1;
    for (; c >= '0' && c <= '9'; c = getchar()) x = x * 10 + c - '0';
    return sgn * x;
}

inline void print128(__int128 x) {
    if (x < 0) {
        putchar('-');
        x = -x;
    }
    if (x >= 10) print128(x / 10);
    putchar(x % 10 + '0');
}
```

### 读入挂

```cpp
// 本机测试需要EOF才能看到输出结果
#define BUF_SIZE 1048576

inline char nc() {
    static char buf[BUF_SIZE], *p1 = buf, *p2 = buf;
    if (p1 == p2) {
        p1 = buf;
        p2 = buf + fread(buf, 1, BUF_SIZE, stdin);
        assert(p1 != p2);
    }
    return *p1++;
}

inline bool blank(char c) { return c == ' ' || c == '\n' || c == '\r' || c == '\t'; }

// non-negative integer
inline int getint() {
    int x = 0;
    char c = nc();
    while (blank(c)) c = nc();
    for (; c >= '0' && c <= '9'; c = nc()) x = x * 10 + c - '0';
    return x;
}

// integer
inline int getint() {
    int x = 0, sgn = 1;
    char c = nc();
    while (blank(c)) c = nc();
    if (c == '-') sgn = -1, c = nc();
    for (; c >= '0' && c <= '9'; c = nc()) x = x * 10 + c - '0';
    return sgn * x;
}

#undef BUF_SIZE
```



# test