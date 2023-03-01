## Input & Output

### Setup

```cpp
#include <bits/stdc++.h>

using namespace std;

using i64 = long long;

int main() {
  ios::sync_with_stdio(false);
  cin.tie(nullptr);

  return 0;
}
```

### Debug

```cpp
// Heltion
template <class T, size_t size = tuple_size<T>::value>
string to_debug(T, string s = "")
  requires(!ranges::range<T>);
string to_debug(auto x)
  requires requires(ostream &os) { os << x; }
{
  return static_cast<ostringstream>(ostringstream() << x).str();
}
string to_debug(ranges::range auto x, string s = "")
  requires(!is_same_v<decltype(x), string>)
{
  for (auto xi : x) s += ", " + to_debug(xi);
  return "[" + s.substr(s.empty() ? 0 : 2) + "]";
}
template <class T, size_t size>
string to_debug(T x, string s)
  requires(!ranges::range<T>)
{
  [&]<size_t... I>(index_sequence<I...>) { ((s += ", " + to_debug(get<I>(x))), ...); }(make_index_sequence<size>());
  return "(" + s.substr(s.empty() ? 0 : 2) + ")";
}
#define dbg(x...) ([&] { cerr << __FILE__ ":" << __LINE__ << ": (" #x ") = " << to_debug(tuple(x)) << "\n"; }())
```

### Special Formats

```cpp
cout << fixed << setprecision(15);  // 15 decimal places

format("{:04d}-{:02d}-{:02d}", y, m, d)  // 0001-01-01
```

### File I/O

```cpp
freopen("in.txt", "r", stdin);
freopen("out.txt", "w", stdout);
```

### Timing

```cpp
(double)clock() / CLOCKS_PER_SEC
```

### Read a Line

```cpp
getline(cin, s)
```

### Read Until End of File

```cpp
while (cin) {}
```

### int128

```cpp
// Need to test if usable
istream& operator>>(istream& is, __int128& x) {
  string s;
  is >> s;
  x = 0;
  for (char c : s) {
    if (c == '-') continue;
    x = x * 10 + c - '0';
  }
  if (s[0] == '-') x = -x;
  return is;
}

ostream& operator<<(ostream& os, __int128 x) {
  if (x < 0) os << '-', x = -x;
  if (x > 9) os << x / 10;
  os << char(x % 10 + '0');
  return os;
}
```

### Fast Input

```cpp
class Scanner {
#ifdef qdd
  static constexpr int BUF_SIZE = 1;
#else
  static constexpr int BUF_SIZE = 1048576; // 1MB
#endif

  char buf[BUF_SIZE], *p1 = buf, *p2 = buf;

  char nc() {
    if (p1 == p2) {
      p1 = buf; p2 = buf + fread(buf, 1, BUF_SIZE, stdin);
      // assert(p1 != p2);
    }
    return p1 == p2 ? EOF : *p1++;
  }

public:
  Scanner& operator>>(string& s) {
    s.clear();
    char c = nc();
    while (c <= 32) c = nc();
    for (; c > 32; c = nc()) s += c;
    return *this;
  }

  Scanner& operator>>(int& x) {
    x = 0;
    int sgn = 1;
    char c = nc();
    for (; c < '0' || c > '9'; c = nc()) if (c == '-') sgn = -1;
    for (; c >= '0' && c <= '9'; c = nc()) x = x * 10 + (c - '0');
    return x *= sgn, *this;
  }

  Scanner& operator>>(double& x) {
    x = 0;
    double base = 0.1;
    int sgn = 1;
    char c = nc();
    for (; c < '0' || c > '9'; c = nc()) if (c == '-') sgn = -1;
    for (; c >= '0' && c <= '9'; c = nc()) x = x * 10 + (c - '0');
    for (; c < '0' || c > '9'; c = nc()) if (c != '.') return x *= sgn, *this;
    for (; c >= '0' && c <= '9'; c = nc()) x += base * (c - '0'), base *= 0.1;
    return x *= sgn, *this;
  }
} in;
```
