# 🐍 Advanced Python DSA Cheat Sheet

##   Table of Contents
1. [Basic Data Structures](#basic-data-structures)
2. [Advanced Collections](#advanced-collections)
3. [String Manipulation](#string-manipulation)
4. [Mathematical Operations](#mathematical-operations)
5. [Algorithm Patterns](#algorithm-patterns)
6. [Time & Space Complexity](#time--space-complexity)
7. [Built-in Functions & Libraries](#built-in-functions--libraries)

---

## 🎯 Basic Data Structures

### 1. Numbers और Advanced Operations
```python
# Basic number types
age = 25                    # Integer
height = 5.8               # Float
big_num = 10**100          # Python handles arbitrarily large integers!
complex_num = 3 + 4j       # Complex numbers

# Advanced operations
import math
print(math.gcd(12, 18))    # Greatest Common Divisor: 6
print(math.lcm(12, 18))    # Least Common Multiple: 36
print(pow(2, 10, 1000))    # Modular exponentiation: 2^10 % 1000

# Bitwise operations (Important for competitive programming!)
a, b = 5, 3                # 5 = 101, 3 = 011 in binary
print(a & b)               # AND: 1
print(a | b)               # OR: 7
print(a ^ b)               # XOR: 6
print(a << 1)              # Left shift: 10
print(a >> 1)              # Right shift: 2
print(~a)                  # NOT: -6
```

### 2. Lists (Advanced Operations)
```python
# List creation और slicing
numbers = [1, 2, 3, 4, 5]
matrix = [[1, 2], [3, 4], [5, 6]]  # 2D list
nested = [[i*j for j in range(3)] for i in range(3)]  # Nested comprehension

# Advanced operations
numbers.extend([6, 7, 8])   # Add multiple elements
numbers.clear()             # Remove all elements
index = numbers.index(3)    # Find index of element
count = numbers.count(2)    # Count occurrences

# Slicing tricks
arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(arr[::2])            # Every 2nd element: [0, 2, 4, 6, 8]
print(arr[1::2])           # Odd indices: [1, 3, 5, 7, 9]
print(arr[::-1])           # Reverse: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
print(arr[2:8:2])          # Slice with step: [2, 4, 6]

# List comprehensions (Power user techniques!)
squares = [x**2 for x in range(10)]
matrix_flat = [item for row in matrix for item in row]  # Flatten 2D list
filtered = [x for x in range(20) if x % 3 == 0 and x % 2 == 1]
dict_comp = {x: x**2 for x in range(5)}  # Dictionary comprehension
set_comp = {x % 3 for x in range(10)}    # Set comprehension

# Sorting with custom keys
students = [('Alice', 85), ('Bob', 90), ('Charlie', 78)]
students.sort(key=lambda x: x[1])        # Sort by grade
students.sort(key=lambda x: x[1], reverse=True)  # Descending order
```

### 3. Advanced String Operations
```python
text = "Hello World! This is a test string."

# String methods
print(text.startswith("Hello"))    # True
print(text.endswith("string."))    # True
print(text.find("World"))          # Index: 6
print(text.rfind("i"))            # Last occurrence of 'i'
print(text.count("l"))            # Count occurrences: 3

# String formatting (Multiple ways!)
name, age = "Danish", 25
# f-strings (Python 3.6+) - PREFERRED!
print(f"Name: {name}, Age: {age}")
print(f"Calculation: {2 + 3 = }")  # Shows equation and result
print(f"Percentage: {0.1234:.2%}") # Format as percentage: 12.34%

# str.format() method
print("Name: {}, Age: {}".format(name, age))
print("Name: {n}, Age: {a}".format(n=name, a=age))

# % formatting (old style)
print("Name: %s, Age: %d" % (name, age))

# Advanced string operations
words = text.split()
joined = " | ".join(words)         # Join with separator
stripped = "  hello  ".strip()     # Remove whitespace
padded = "hello".center(20, "*")   # Center with padding
```

### 4. Dictionaries (Hash Maps) - Advanced
```python
# Dictionary creation methods
dict1 = {"a": 1, "b": 2}
dict2 = dict(a=1, b=2)
dict3 = dict([("a", 1), ("b", 2)])
dict4 = {chr(97+i): i+1 for i in range(3)}  # {'a': 1, 'b': 2, 'c': 3}

# Advanced operations
student = {"name": "Rahul", "age": 22, "marks": [85, 90, 78]}

# Merge dictionaries (Python 3.9+)
extra_info = {"city": "Delhi", "phone": "1234567890"}
merged = student | extra_info      # New dict with merged data
student |= extra_info              # Update in place

# Dictionary methods
student.setdefault("grade", "A")   # Set if key doesn't exist
popped_value = student.pop("age", 0)  # Pop with default
student.update({"semester": 5})    # Update multiple keys

# Nested dictionaries
nested_dict = {
    "students": {
        "CS": ["Alice", "Bob"],
        "EE": ["Charlie", "David"]
    }
}
```

---

##  ️ Advanced Collections

### 1. Sets - O(1) Operations!
```python
# Set creation
set1 = {1, 2, 3, 4, 5}
set2 = set([1, 2, 2, 3, 3, 4])  # Duplicates removed: {1, 2, 3, 4}
empty_set = set()               # Empty set (NOT {})

# Set operations
set1.add(6)                     # Add element
set1.remove(3)                  # Remove (raises error if not found)
set1.discard(10)               # Remove (no error if not found)

# Mathematical set operations
a, b = {1, 2, 3, 4}, {3, 4, 5, 6}
union = a | b                   # {1, 2, 3, 4, 5, 6}
intersection = a & b            # {3, 4}
difference = a - b              # {1, 2}
symmetric_diff = a ^ b          # {1, 2, 5, 6}

# Set comprehension
squares_set = {x**2 for x in range(10)}
```

### 2. Tuples - Immutable Sequences
```python
# Tuple creation
point = (3, 4)
coordinates = 10, 20           # Parentheses optional
single_element = (5,)          # Comma needed for single element

# Tuple unpacking
x, y = point                   # x=3, y=4
first, *middle, last = (1, 2, 3, 4, 5)  # first=1, middle=[2,3,4], last=5

# Named tuples (Advanced!)
from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(3, 4)
print(p.x, p.y)               # Access by name: 3 4
```

### 3. Deques - Double-ended Queues
```python
from collections import deque

# Deque operations - O(1) at both ends!
dq = deque([1, 2, 3])
dq.appendleft(0)              # Add to left: deque([0, 1, 2, 3])
dq.append(4)                  # Add to right: deque([0, 1, 2, 3, 4])
left_item = dq.popleft()      # Remove from left: 0
right_item = dq.pop()         # Remove from right: 4

# Rotate operations
dq.rotate(2)                  # Rotate right by 2
dq.rotate(-1)                 # Rotate left by 1
```

### 4. Counter - Count Hashable Objects
```python
from collections import Counter

# Count elements
text = "hello world"
char_count = Counter(text)    # Counter({'l': 3, 'o': 2, 'h': 1, ...})
word_count = Counter(['apple', 'banana', 'apple', 'cherry'])

# Useful methods
print(char_count.most_common(2))  # [('l', 3), ('o', 2)]
print(char_count['l'])            # 3
char_count.update("aaa")          # Add more counts
```

### 5. DefaultDict - Dictionary with Default Values
```python
from collections import defaultdict

# Group items by some criteria
dd = defaultdict(list)
items = [('fruit', 'apple'), ('vegetable', 'carrot'), ('fruit', 'banana')]

for category, item in items:
    dd[category].append(item)
# Result: defaultdict(<class 'list'>, {'fruit': ['apple', 'banana'], 'vegetable': ['carrot']})

# Graph adjacency list
graph = defaultdict(list)
edges = [(1, 2), (1, 3), (2, 4)]
for u, v in edges:
    graph[u].append(v)
```

---

## 🎛️ Control Flow (Advanced)

### 1. Advanced If-Else and Match-Case
```python
# Multiple conditions
score = 85
attendance = 90

# Complex conditions
if score >= 80 and attendance >= 85:
    status = "Excellent"
elif score >= 70 or attendance >= 80:
    status = "Good"
else:
    status = "Needs Improvement"

# Walrus operator (Python 3.8+)
if (n := len(some_list)) > 10:
    print(f"List is long with {n} items")

# Match-case (Python 3.10+) - Switch statement
def handle_response(status_code):
    match status_code:
        case 200:
            return "Success"
        case 404 | 403:  # Multiple values
            return "Not Found or Forbidden"
        case code if 400 <= code < 500:
            return "Client Error"
        case _:  # Default case
            return "Unknown"
```

### 2. Advanced Loops और Iterators
```python
# Nested loops with else clause
def find_prime_factors(n):
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return [i] + find_prime_factors(n // i)
    else:
        return [n] if n > 1 else []

# Multiple iterables
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
for name, age in zip(names, ages):
    print(f"{name} is {age} years old")

# Enumerate with start parameter
for i, item in enumerate(items, start=1):
    print(f"{i}. {item}")

# Itertools - Powerful iteration tools
from itertools import permutations, combinations, product, chain

# Permutations and combinations
perms = list(permutations([1, 2, 3], 2))      # [(1,2), (1,3), (2,1), (2,3), (3,1), (3,2)]
combs = list(combinations([1, 2, 3, 4], 2))   # [(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
prod = list(product([1, 2], ['a', 'b']))      # [(1,'a'), (1,'b'), (2,'a'), (2,'b')]
chained = list(chain([1, 2], [3, 4]))         # [1, 2, 3, 4]
```

### 3. Advanced Functions
```python
# Function annotations (Type hints)
def calculate_area(length: float, width: float) -> float:
    """Calculate rectangle area with type hints."""
    return length * width

# *args and **kwargs
def flexible_function(*args, **kwargs):
    print(f"Args: {args}")
    print(f"Kwargs: {kwargs}")

# Lambda functions with advanced usage
students = [('Alice', 85), ('Bob', 90), ('Charlie', 78)]
top_student = max(students, key=lambda x: x[1])  # ('Bob', 90)

# Higher-order functions
def apply_operation(arr, operation):
    return [operation(x) for x in arr]

squared = apply_operation([1, 2, 3, 4], lambda x: x**2)

# Decorators (Advanced topic!)
from functools import wraps
import time

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
    return "Done"

# Generators (Memory efficient!)
def fibonacci_generator():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# Use first 10 fibonacci numbers
fib_gen = fibonacci_generator()
first_10 = [next(fib_gen) for _ in range(10)]
```

---

## 🧮 Mathematical Operations

### 1. Advanced Math Functions
```python
import math
import random

# Mathematical constants
print(math.pi)      # 3.141592653589793
print(math.e)       # 2.718281828459045

# Power and logarithms
print(math.pow(2, 3))        # 8.0
print(math.sqrt(16))         # 4.0
print(math.log(100, 10))     # 2.0 (log base 10)
print(math.log2(8))          # 3.0 (log base 2)
print(math.log(math.e))      # 1.0 (natural log)

# Trigonometric functions
print(math.sin(math.pi/2))   # 1.0
print(math.cos(0))           # 1.0
print(math.tan(math.pi/4))   # 1.0

# Ceiling and floor
print(math.ceil(4.3))        # 5
print(math.floor(4.7))       # 4

# Random numbers (Important for algorithms!)
random.seed(42)              # Set seed for reproducibility
print(random.randint(1, 10)) # Random integer between 1-10
print(random.choice([1, 2, 3, 4, 5]))  # Random choice from list
random.shuffle([1, 2, 3, 4, 5])        # Shuffle list in place

# Modular arithmetic (Competitive programming!)
def mod_exp(base, exp, mod):
    """Fast modular exponentiation"""
    result = 1
    base = base % mod
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp = exp >> 1
        base = (base * base) % mod
    return result

# Extended Euclidean Algorithm
def extended_gcd(a, b):
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd, x, y
```

### 2. Number Theory Functions
```python
def is_prime(n):
    """Optimized primality test"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def sieve_of_eratosthenes(n):
    """Generate all primes up to n"""
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    
    return [i for i in range(n + 1) if is_prime[i]]

def prime_factors(n):
    """Find prime factorization"""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors
```

---

## 🚀 Algorithm Patterns

### 1. Two Pointers Technique (Advanced)
```python
def two_pointers_example(arr, target):
    """Basic two pointers - sorted array"""
    left, right = 0, len(arr) - 1
    
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return [-1, -1]

def three_sum(nums, target):
    """Three sum using two pointers"""
    nums.sort()
    result = []
    
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i-1]:  # Skip duplicates
            continue
            
        left, right = i + 1, len(nums) - 1
        while left < right:
            current_sum = nums[i] + nums[left] + nums[right]
            if current_sum == target:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif current_sum < target:
                left += 1
            else:
                right -= 1
    
    return result

def remove_duplicates(arr):
    """Remove duplicates from sorted array in-place"""
    if not arr:
        return 0
    
    write_idx = 1
    for read_idx in range(1, len(arr)):
        if arr[read_idx] != arr[read_idx - 1]:
            arr[write_idx] = arr[read_idx]
            write_idx += 1
    
    return write_idx
```

### 2. Sliding Window (Advanced Patterns)
```python
def max_sum_subarray(arr, k):
    """Fixed size sliding window"""
    if len(arr) < k:
        return None
    
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i-k] + arr[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum

def longest_substring_k_distinct(s, k):
    """Variable size sliding window"""
    if not s or k == 0:
        return 0
    
    char_count = {}
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        # Expand window
        char_count[s[right]] = char_count.get(s[right], 0) + 1
        
        # Shrink window if needed
        while len(char_count) > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1
        
        max_length = max(max_length, right - left + 1)
    
    return max_length

def min_window_substring(s, t):
    """Minimum window substring"""
    if not s or not t:
        return ""
    
    dict_t = {}
    for char in t:
        dict_t[char] = dict_t.get(char, 0) + 1
    
    required = len(dict_t)
    formed = 0
    window_counts = {}
    
    left = right = 0
    ans = float("inf"), None, None
    
    while right < len(s):
        char = s[right]
        window_counts[char] = window_counts.get(char, 0) + 1
        
        if char in dict_t and window_counts[char] == dict_t[char]:
            formed += 1
        
        while left <= right and formed == required:
            char = s[left]
            
            if right - left + 1 < ans[0]:
                ans = (right - left + 1, left, right)
            
            window_counts[char] -= 1
            if char in dict_t and window_counts[char] < dict_t[char]:
                formed -= 1
            
            left += 1
        
        right += 1
    
    return "" if ans[0] == float("inf") else s[ans[1]:ans[2] + 1]
```

### 3. Hash Map Patterns (Advanced)
```python
def two_sum(nums, target):
    """Classic two sum problem"""
    num_to_index = {}
    
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_to_index:
            return [num_to_index[complement], i]
        num_to_index[num] = i
    
    return []

def group_anagrams(strs):
    """Group anagrams using hash map"""
    anagrams = {}
    
    for s in strs:
        # Sort string to get canonical form
        key = ''.join(sorted(s))
        if key not in anagrams:
            anagrams[key] = []
        anagrams[key].append(s)
    
    return list(anagrams.values())

def subarray_sum_equals_k(nums, k):
    """Count subarrays with sum equal to k"""
    count = 0
    prefix_sum = 0
    sum_count = {0: 1}  # Important: empty subarray has sum 0
    
    for num in nums:
        prefix_sum += num
        
        # Check if (prefix_sum - k) exists
        if prefix_sum - k in sum_count:
            count += sum_count[prefix_sum - k]
        
        # Add current prefix sum to map
        sum_count[prefix_sum] = sum_count.get(prefix_sum, 0) + 1
    
    return count

def longest_consecutive_sequence(nums):
    """Find longest consecutive sequence"""
    if not nums:
        return 0
    
    num_set = set(nums)
    longest_streak = 0
    
    for num in num_set:
        # Only start counting if it's the beginning of a sequence
        if num - 1 not in num_set:
            current_num = num
            current_streak = 1
            
            while current_num + 1 in num_set:
                current_num += 1
                current_streak += 1
            
            longest_streak = max(longest_streak, current_streak)
    
    return longest_streak
```

### 4. Binary Search Patterns
```python
def binary_search_basic(arr, target):
    """Basic binary search"""
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

def find_first_occurrence(arr, target):
    """Find first occurrence of target"""
    left, right = 0, len(arr) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            result = mid
            right = mid - 1  # Continue searching left
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result

def find_peak_element(arr):
    """Find any peak element"""
    left, right = 0, len(arr) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        if arr[mid] > arr[mid + 1]:
            right = mid  # Peak is in left half
        else:
            left = mid + 1  # Peak is in right half
    
    return left

def search_rotated_array(arr, target):
    """Search in rotated sorted array"""
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        
        # Left half is sorted
        if arr[left] <= arr[mid]:
            if arr[left] <= target < arr[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right half is sorted
        else:
            if arr[mid] < target <= arr[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1
```

### 5. Dynamic Programming Patterns
```python
def fibonacci_dp(n):
    """Fibonacci with dynamic programming"""
    if n <= 1:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]

def coin_change(coins, amount):
    """Minimum coins needed for amount"""
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

def longest_increasing_subsequence(arr):
    """Length of longest increasing subsequence"""
    if not arr:
        return 0
    
    dp = [1] * len(arr)
    
    for i in range(1, len(arr)):
        for j in range(i):
            if arr[j] < arr[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)

def knapsack_01(weights, values, capacity):
    """0-1 Knapsack problem"""
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(
                    values[i-1] + dp[i-1][w - weights[i-1]],  # Include item
                    dp[i-1][w]  # Exclude item
                )
            else:
                dp[i][w] = dp[i-1][w]
    
    return dp[n][capacity]
```

### 6. Graph Algorithms
```python
from collections import deque, defaultdict

def bfs_shortest_path(graph, start, end):
    """BFS for shortest path in unweighted graph"""
    if start == end:
        return [start]
    
    queue = deque([(start, [start])])
    visited = {start}
    
    while queue:
        node, path = queue.popleft()
        
        for neighbor in graph[node]:
            if neighbor == end:
                return path + [neighbor]
            
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return []

def dfs_recursive(graph, node, visited=None):
    """DFS traversal (recursive)"""
    if visited is None:
        visited = set()
    
    visited.add(node)
    result = [node]
    
    for neighbor in graph[node]:
        if neighbor not in visited:
            result.extend(dfs_recursive(graph, neighbor, visited))
    
    return result

def has_cycle_undirected(graph):
    """Detect cycle in undirected graph"""
    visited = set()
    
    def dfs(node, parent):
        visited.add(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs(neighbor, node):
                    return True
            elif neighbor != parent:
                return True
        
        return False
    
    for node in graph:
        if node not in visited:
            if dfs(node, -1):
                return True
    
    return False

def topological_sort(graph):
    """Topological sort using Kahn's algorithm"""
    in_degree = defaultdict(int)
    
    # Calculate in-degrees
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1
    
    # Find nodes with no incoming edges
    queue = deque([node for node in graph if in_degree[node] == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return result if len(result) == len(graph) else []
```

### 7. Tree Algorithms
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def inorder_traversal(root):
    """Inorder traversal (iterative)"""
    result = []
    stack = []
    current = root
    
    while stack or current:
        while current:
            stack.append(current)
            current = current.left
        
        current = stack.pop()
        result.append(current.val)
        current = current.right
    
    return result

def level_order_traversal(root):
    """Level order traversal (BFS)"""
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result

def max_depth(root):
    """Maximum depth of binary tree"""
    if not root:
        return 0
    
    return 1 + max(max_depth(root.left), max_depth(root.right))

def is_balanced(root):
    """Check if binary tree is balanced"""
    def check_height(node):
        if not node:
            return 0
        
        left_height = check_height(node.left)
        if left_height == -1:
            return -1
        
        right_height = check_height(node.right)
        if right_height == -1:
            return -1
        
        if abs(left_height - right_height) > 1:
            return -1
        
        return max(left_height, right_height) + 1
    
    return check_height(root) != -1

def lowest_common_ancestor(root, p, q):
    """LCA in binary tree"""
    if not root or root == p or root == q:
        return root
    
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)
    
    if left and right:
        return root
    
    return left if left else right
```

---

## ⏱️ Time & Space Complexity

### Big O Notation Reference
```python
# O(1) - Constant Time
def get_first_element(arr):
    return arr[0] if arr else None

# O(log n) - Logarithmic Time
def binary_search(arr, target):
    # See implementation above
    pass

# O(n) - Linear Time  
def find_maximum(arr):
    return max(arr)

# O(n log n) - Log-linear Time
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

# O(n²) - Quadratic Time
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

# O(2^n) - Exponential Time (Naive Fibonacci)
def fibonacci_naive(n):
    if n <= 1:
        return n
    return fibonacci_naive(n-1) + fibonacci_naive(n-2)
```

### Space Complexity Examples
```python
# O(1) Space - Constant
def sum_array(arr):
    total = 0  # Only one variable
    for num in arr:
        total += num
    return total

# O(n) Space - Linear
def create_copy(arr):
    return arr.copy()  # New array of same size

# O(log n) Space - Recursive call stack
def binary_search_recursive(arr, target, left=0, right=None):
    if right is None:
        right = len(arr) - 1
    
    if left > right:
        return -1
    
    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)
```

---

## 🛠️ Built-in Functions & Libraries

### 1. Essential Built-ins for DSA
```python
# min, max with custom key
points = [(1, 2), (3, 1), (5, 4)]
closest_to_origin = min(points, key=lambda p: p[0]**2 + p[1]**2)

# sorted() vs sort()
arr = [3, 1, 4, 1, 5]
sorted_arr = sorted(arr)           # Returns new list
arr.sort()                         # Sorts in-place

# Custom sorting
students = [('Alice', 85), ('Bob', 90), ('Charlie', 78)]
by_grade = sorted(students, key=lambda x: x[1], reverse=True)
by_name = sorted(students, key=lambda x: x[0])

# all() and any()
numbers = [2, 4, 6, 8]
print(all(n % 2 == 0 for n in numbers))  # True (all even)
print(any(n > 5 for n in numbers))       # True (some > 5)

# sum() with start value
print(sum([1, 2, 3], 10))         # 16 (starts from 10)

# zip() and enumerate()
names = ['Alice', 'Bob', 'Charlie']
ages = [25, 30, 35]
for i, (name, age) in enumerate(zip(names, ages)):
    print(f"{i+1}. {name} is {age} years old")
```

### 2. Heaps (Priority Queues)
```python
import heapq

# Min heap (default)
heap = [3, 1, 4, 1, 5, 9, 2, 6]
heapq.heapify(heap)                # Convert list to heap in O(n)

heapq.heappush(heap, 0)           # Add element
min_element = heapq.heappop(heap)  # Remove and return smallest

# For max heap, use negative values
max_heap = [-x for x in [3, 1, 4, 1, 5]]
heapq.heapify(max_heap)
max_element = -heapq.heappop(max_heap)

# K largest/smallest elements
nums = [3, 1, 4, 1, 5, 9, 2, 6]
k_largest = heapq.nlargest(3, nums)      # [9, 6, 5]
k_smallest = heapq.nsmallest(3, nums)    # [1, 1, 2]

# Heap with custom key
points = [(1, 2), (3, 1), (5, 4)]
closest = heapq.nsmallest(2, points, key=lambda p: p[0]**2 + p[1]**2)
```

### 3. Bisect Module (Binary Search)
```python
import bisect

# Sorted list
arr = [1, 3, 3, 5, 7, 9]

# Find insertion points
left_pos = bisect.bisect_left(arr, 3)   # 1 (leftmost position)
right_pos = bisect.bisect_right(arr, 3) # 3 (rightmost position)

# Insert elements
bisect.insort(arr, 4)  # Inserts 4 in sorted position

# Find range of target
def find_range(arr, target):
    left = bisect.bisect_left(arr, target)
    right = bisect.bisect_right(arr, target)
    return [left, right - 1] if left < len(arr) and arr[left] == target else [-1, -1]
```

### 4. String Algorithms
```python
# String matching
def kmp_search(text, pattern):
    """KMP algorithm for pattern matching"""
    def compute_lps(pattern):
        lps = [0] * len(pattern)
        length = 0
        i = 1
        
        while i < len(pattern):
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        return lps
    
    lps = compute_lps(pattern)
    i = j = 0
    matches = []
    
    while i < len(text):
        if pattern[j] == text[i]:
            i += 1
            j += 1
        
        if j == len(pattern):
            matches.append(i - j)
            j = lps[j - 1]
        elif i < len(text) and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    
    return matches

# Regular expressions
import re

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def extract_numbers(text):
    return re.findall(r'\d+', text)
```

---

## 🎯 Input/Output Templates

### 1. Competitive Programming I/O
```python
import sys
from collections import *

# Fast I/O for competitive programming
def fast_io():
    input = sys.stdin.readline
    
    # Single integer
    n = int(input())
    
    # Multiple integers
    a, b, c = map(int, input().split())
    
    # List of integers
    arr = list(map(int, input().split()))
    
    # String input (strip newline)
    s = input().strip()

# Template for multiple test cases
def solve():
    n = int(input())
    arr = list(map(int, input().split()))
    
    # Your solution here
    result = your_algorithm(arr)
    
    print(result)

def main():
    t = int(input())
    for _ in range(t):
        solve()

if __name__ == "__main__":
    main()

# Matrix input
def read_matrix():
    n, m = map(int, input().split())
    matrix = []
    for _ in range(n):
        row = list(map(int, input().split()))
        matrix.append(row)
    return matrix

# Graph input (adjacency list)
def read_graph():
    n, m = map(int, input().split())  # nodes, edges
    graph = defaultdict(list)
    
    for _ in range(m):
        u, v = map(int, input().split())
        graph[u].append(v)
        graph[v].append(u)  # For undirected graph
    
    return graph
```

### 2. File I/O
```python
# Reading from file
def read_from_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        return [line.strip() for line in lines]

# Writing to file
def write_to_file(filename, data):
    with open(filename, 'w') as f:
        for item in data:
            f.write(f"{item}\n")

# CSV handling
import csv

def read_csv(filename):
    data = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    return data

# JSON handling
import json

def read_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)
```

---

## 💡 Advanced Tips for DSA

### 1. Memory Optimization
```python
# Use generators for large datasets
def read_large_file(filename):
    with open(filename, 'r') as f:
        for line in f:
            yield line.strip()

# Slots for memory-efficient classes
class Point:
    __slots__ = ['x', 'y']
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Array module for numeric arrays
import array
int_array = array.array('i', [1, 2, 3, 4, 5])  # 'i' for integers
```

### 2. Performance Tips
```python
# Use local variables in loops (faster access)
def optimized_sum(arr):
    total = 0
    local_sum = total  # Local reference
    for num in arr:
        local_sum += num
    return local_sum

# List comprehensions are faster than loops
# Fast
squares = [x**2 for x in range(1000)]

# Slower
squares = []
for x in range(1000):
    squares.append(x**2)

# Use appropriate data structures
# Set for O(1) lookups
seen = set()
for item in data:
    if item not in seen:  # O(1) average case
        seen.add(item)

# Deque for queue operations
from collections import deque
queue = deque()
queue.append(item)      # O(1)
item = queue.popleft()  # O(1)
```

### 3. Debugging और Testing
```python
# Assert statements for debugging
def binary_search_with_asserts(arr, target):
    assert all(arr[i] <= arr[i+1] for i in range(len(arr)-1)), "Array must be sorted"
    
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

# Unit testing
import unittest

class TestAlgorithms(unittest.TestCase):
    def test_binary_search(self):
        arr = [1, 3, 5, 7, 9]
        self.assertEqual(binary_search(arr, 5), 2)
        self.assertEqual(binary_search(arr, 2), -1)
    
    def test_two_sum(self):
        nums = [2, 7, 11, 15]
        self.assertEqual(two_sum(nums, 9), [0, 1])

if __name__ == '__main__':
    unittest.main()

# Timing functions
import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.6f} seconds")
        return result
    return wrapper

@timer
def slow_algorithm(n):
    return sum(range(n))
```

---

## 🚨 Common Mistakes to Avoid

### 1. Python Pitfalls
```python
# ❌ Mutable default arguments
def bad_function(arr=[]):  # Don't do this!
    arr.append(1)
    return arr

# ✅ Correct way
def good_function(arr=None):
    if arr is None:
        arr = []
    arr.append(1)
    return arr

# ❌ Late binding closures
funcs = []
for i in range(3):
    funcs.append(lambda: i)  # All will return 2!

# ✅ Correct way
funcs = []
for i in range(3):
    funcs.append(lambda x=i: x)  # Capture current value

# ❌ Modifying list while iterating
arr = [1, 2, 3, 4, 5]
for i, item in enumerate(arr):
    if item % 2 == 0:
        arr.remove(item)  # Can cause issues!

# ✅ Correct way
arr = [item for item in arr if item % 2 != 0]
```

### 2. Algorithm Mistakes
```python
# ❌ Off-by-one errors in binary search
def bad_binary_search(arr, target):
    left, right = 0, len(arr)  # Should be len(arr) - 1
    while left < right:        # Should be left <= right
        # ... rest of implementation

# ❌ Infinite loops
def bad_while_loop():
    i = 0
    while i < 10:
        print(i)
        # Forgot to increment i!

# ❌ Integer overflow (less common in Python)
# Python handles big integers automatically, but be aware in other languages

# ❌ Wrong operator precedence
if not x == 0:  # This is (not x) == 0, not not (x == 0)
if not (x == 0):  # Correct way
```

---

## 📚 Additional Resources

### Must-Know Libraries
```python
# Data science libraries (for advanced problems)
import numpy as np
import pandas as pd

# Algorithm libraries
from heapq import *
from bisect import *
from collections import *
from itertools import *
from functools import *

# System libraries
import sys
import os
import math
import random

# Regular expressions
import re

# Date and time
from datetime import datetime, timedelta
```

### Quick Reference
- **Sorting**: `sorted()`, `list.sort()`, `key=lambda`, `reverse=True`
- **Searching**: `in` operator, `list.index()`, `bisect` module
- **String**: `str.join()`, `str.split()`, `str.strip()`, regex
- **Math**: `math` module, `//` vs `/`, `**` for power, `%` for modulo
- **Collections**: `Counter`, `defaultdict`, `deque`, `namedtuple`
- **Itertools**: `permutations`, `combinations`, `product`, `chain`

---

**Practice File**: Use `Python_Templates.py` for hands-on coding! 🚀

**Remember**: 
- Practice implementing these algorithms from scratch
- Understand time and space complexity
- Test with edge cases
- Use appropriate data structures
- Write clean, readable code