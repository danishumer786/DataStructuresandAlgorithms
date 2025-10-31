"""
🐍 Python Basics for DSA - Practice Templates
This file contains working Python code examples for DSA practice
"""

# ===============================================
# BASIC DATA TYPES AND OPERATIONS
# ===============================================

# Variables और Data Types
def basic_data_types():
    age = 25                    # Integer
    height = 5.8               # Float
    name = "Dinesh"            # String  
    is_student = True          # Boolean
    
    print(f"Age: {age}, Type: {type(age)}")
    print(f"Name: {name}, Type: {type(name)}")

# ===============================================
# LIST OPERATIONS (Most Important for DSA!)
# ===============================================

def list_operations():
    # List creation
    numbers = [1, 2, 3, 4, 5]
    fruits = ["apple", "banana", "orange"]
    
    # Common operations
    numbers.append(6)          # Add element at end
    numbers.insert(0, 0)       # Insert at specific index
    numbers.remove(3)          # Remove first occurrence
    popped = numbers.pop()     # Remove and return last element
    
    print("After operations:", numbers)
    
    # List comprehension (बहुत useful!)
    squares = [x**2 for x in range(5)]        # [0, 1, 4, 9, 16]
    evens = [x for x in range(10) if x % 2 == 0]  # [0, 2, 4, 6, 8]
    
    print("Squares:", squares)
    print("Even numbers:", evens)

# ===============================================
# STRING OPERATIONS
# ===============================================

def string_operations():
    text = "Hello World"
    
    print(f"Length: {len(text)}")
    print(f"Uppercase: {text.upper()}")
    print(f"Lowercase: {text.lower()}")
    print(f"Split: {text.split()}")
    
    # String slicing (important for DSA)
    print(f"First character: {text[0]}")
    print(f"Last character: {text[-1]}")
    print(f"Substring: {text[0:5]}")
    print(f"Reverse: {text[::-1]}")

# ===============================================
# DICTIONARY OPERATIONS (Hash Maps)
# ===============================================

def dictionary_operations():
    student = {
        "name": "Rahul",
        "age": 22,
        "marks": [85, 90, 78]
    }
    
    # Operations
    print(f"Name: {student['name']}")
    student["city"] = "Delhi"   # Add new key
    grade = student.get("grade", "A")   # Safe access with default
    
    print("Updated student:", student)
    print("Keys:", list(student.keys()))
    print("Values:", list(student.values()))

# ===============================================
# COMMON DSA PATTERNS
# ===============================================

def two_pointers_pattern(arr, target):
    """Two Pointers Technique - Find pair with given sum"""
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

def sliding_window_pattern(arr, k):
    """Sliding Window - Maximum sum of k consecutive elements"""
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

def hash_map_pattern(nums, target):
    """Hash Map Pattern - Two Sum Problem"""
    num_to_index = {}
    
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_to_index:
            return [num_to_index[complement], i]
        num_to_index[num] = i
    
    return []

# ===============================================
# INPUT/OUTPUT TEMPLATES
# ===============================================

def basic_io_template():
    """Template for basic input/output"""
    print("Enter a number:")
    # n = int(input())                    # Read integer
    # name = input().strip()              # Read string
    
    print("Enter two numbers separated by space:")
    # a, b = map(int, input().split())    # Read two integers
    
    print("Enter list of numbers:")
    # numbers = list(map(int, input().split()))  # Read list of integers

def competitive_programming_template():
    """Template for multiple test cases"""
    # t = int(input())  # Number of test cases
    # for _ in range(t):
    #     n = int(input())
    #     arr = list(map(int, input().split()))
    #     
    #     # Solve problem here
    #     result = solve_problem(arr)
    #     print(result)
    pass

# ===============================================
# PRACTICE PROBLEMS - EASY LEVEL
# ===============================================

def find_maximum(arr):
    """Find maximum element in array"""
    if not arr:
        return None
    
    max_val = arr[0]
    for num in arr:
        if num > max_val:
            max_val = num
    return max_val

def reverse_string(s):
    """Reverse a string"""
    return s[::-1]

def count_vowels(s):
    """Count vowels in a string"""
    vowels = "aeiouAEIOU"
    count = 0
    for char in s:
        if char in vowels:
            count += 1
    return count

def is_palindrome(s):
    """Check if string is palindrome"""
    s = s.lower().replace(" ", "")
    return s == s[::-1]

def factorial(n):
    """Calculate factorial of a number"""
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n):
    """Generate first n fibonacci numbers"""
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib

def binary_search(arr, target):
    """Binary search implementation"""
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

# ===============================================
# TESTING FUNCTIONS
# ===============================================

def test_functions():
    """Test all the functions"""
    print("=== Testing Basic Functions ===")
    
    # Test basic operations
    basic_data_types()
    print()
    
    list_operations()
    print()
    
    string_operations()
    print()
    
    dictionary_operations()
    print()
    
    # Test DSA patterns
    print("=== Testing DSA Patterns ===")
    arr = [1, 2, 3, 4, 5, 6]
    result = two_pointers_pattern(arr, 9)
    print(f"Two pointers result: {result}")
    
    window_result = sliding_window_pattern([1, 4, 2, 10, 23, 3, 1, 0, 20], 4)
    print(f"Sliding window max sum: {window_result}")
    
    two_sum_result = hash_map_pattern([2, 7, 11, 15], 9)
    print(f"Two sum result: {two_sum_result}")
    
    # Test practice problems
    print("\n=== Testing Practice Problems ===")
    test_arr = [3, 1, 4, 1, 5, 9, 2, 6]
    print(f"Maximum in {test_arr}: {find_maximum(test_arr)}")
    
    test_string = "hello world"
    print(f"Reverse of '{test_string}': {reverse_string(test_string)}")
    
    print(f"Vowels in '{test_string}': {count_vowels(test_string)}")
    
    print(f"Is 'racecar' palindrome: {is_palindrome('racecar')}")
    
    print(f"Factorial of 5: {factorial(5)}")
    
    print(f"First 8 fibonacci numbers: {fibonacci(8)}")
    
    sorted_arr = [1, 3, 5, 7, 9, 11, 13]
    print(f"Binary search for 7 in {sorted_arr}: {binary_search(sorted_arr, 7)}")

if __name__ == "__main__":
    test_functions()