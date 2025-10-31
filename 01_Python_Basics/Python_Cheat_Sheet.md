# 🐍 Python Basics Cheat Sheet for DSA

## 📝 Essential Python Syntax (Hinglish Comments)

### 1. Variables और Data Types
```python
# Numbers
age = 25                    # Integer
height = 5.8               # Float
name = "Dinesh"            # String  
is_student = True          # Boolean

# Type checking
print(type(age))           # <class 'int'>
```

### 2. Lists (सबसे important for DSA!)
```python
# List creation
numbers = [1, 2, 3, 4, 5]
fruits = ["apple", "banana", "orange"]
mixed = [1, "hello", 3.14, True]

# Common operations
numbers.append(6)          # Add element at end
numbers.insert(0, 0)       # Insert at specific index
numbers.remove(3)          # Remove first occurrence
popped = numbers.pop()     # Remove and return last element
numbers.sort()             # Sort in place
numbers.reverse()          # Reverse in place

# List comprehension (बहुत useful!)
squares = [x**2 for x in range(5)]        # [0, 1, 4, 9, 16]
evens = [x for x in range(10) if x % 2 == 0]  # [0, 2, 4, 6, 8]
```

### 3. Strings
```python
text = "Hello World"
print(len(text))           # Length
print(text.upper())        # Uppercase
print(text.lower())        # Lowercase
print(text.split())        # Split into list
print(text.replace("Hello", "Hi"))  # Replace

# String slicing (important for DSA)
print(text[0])             # First character: 'H'
print(text[-1])            # Last character: 'd'  
print(text[0:5])           # Substring: 'Hello'
print(text[::-1])          # Reverse: 'dlroW olleH'
```

### 4. Dictionaries (Hash Maps)
```python
# Creation
student = {
    "name": "Rahul",
    "age": 22,
    "marks": [85, 90, 78]
}

# Operations
print(student["name"])      # Access value
student["city"] = "Delhi"   # Add new key
student.get("grade", "A")   # Safe access with default

# Useful methods
keys = student.keys()       # All keys
values = student.values()   # All values  
items = student.items()     # Key-value pairs
```

## 🔄 Control Flow

### 1. If-Else Statements
```python
score = 85

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"  
elif score >= 70:
    grade = "C"
else:
    grade = "F"

# Ternary operator
result = "Pass" if score >= 60 else "Fail"
```

### 2. Loops
```python
# For loop with range
for i in range(5):          # 0 to 4
    print(i)

for i in range(1, 6):       # 1 to 5  
    print(i)

# For loop with list
fruits = ["apple", "banana", "orange"]
for fruit in fruits:
    print(fruit)

# Enumerate (index के साथ)
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")

# While loop
count = 0
while count < 5:
    print(count)
    count += 1
```

## 🔧 Functions

### 1. Basic Functions
```python
def greet(name):
    return f"Hello, {name}!"

# Function with default parameter
def power(base, exp=2):
    return base ** exp

# Multiple return values
def divide_and_remainder(a, b):
    quotient = a // b
    remainder = a % b
    return quotient, remainder
```

## 📊 Common DSA Patterns in Python

### 1. Two Pointers Technique
```python
def two_pointers_example(arr, target):
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
```

### 2. Sliding Window
```python
def max_sum_subarray(arr, k):
    # Maximum sum of k consecutive elements
    if len(arr) < k:
        return None
    
    # First window
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    # Slide the window
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i-k] + arr[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum
```

### 3. Hash Map Pattern
```python
def two_sum(nums, target):
    # Find two numbers that add up to target
    num_to_index = {}
    
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_to_index:
            return [num_to_index[complement], i]
        num_to_index[num] = i
    
    return []
```

## 🎯 Input/Output Templates

### 1. Basic I/O
```python
# Single input
n = int(input())                    # Read integer
name = input().strip()              # Read string

# Multiple inputs in one line  
a, b = map(int, input().split())    # Read two integers
numbers = list(map(int, input().split()))  # Read list of integers
```

### 2. Multiple Test Cases
```python
# Template for competitive programming
t = int(input())  # Number of test cases
for _ in range(t):
    n = int(input())
    arr = list(map(int, input().split()))
    
    # Solve problem here
    result = solve_problem(arr)
    print(result)
```

## 💡 Quick Tips for DSA

1. **Index out of bounds**: Always check `if i < len(arr)`
2. **Integer division**: Use `//` for floor division, `/` for float division  
3. **List methods**: `append()`, `pop()`, `insert()`, `remove()`
4. **String slicing**: `s[start:end:step]`
5. **Dictionary methods**: `get()`, `keys()`, `values()`, `items()`

## 🚨 Common Mistakes to Avoid

- **Mutable default arguments**: Use `arr=None` instead of `arr=[]`
- **Infinite loops**: Make sure loop condition changes
- **Wrong operators**: `=` vs `==`, `and` vs `&`
- **Indentation**: Python is strict about indentation

---

**Practice File**: Use `Python_Templates.py` for hands-on coding! 🚀