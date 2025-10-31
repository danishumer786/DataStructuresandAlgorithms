# 📝 Week 1 Practice Problems - Python Basics

## 🎯 Day 1-2: Arrays और Basic Operations

### Problem 1: Find Maximum Element
**Difficulty**: Easy  
**Topic**: Arrays  
**Hindi Explanation**: Array में सबसे बड़ा element find करना है।

```python
def find_maximum(arr):
    """
    Input: [3, 1, 4, 1, 5, 9, 2, 6]
    Output: 9
    """
    # Your code here
    pass

# Test cases
test1 = [3, 1, 4, 1, 5, 9, 2, 6]
test2 = [10]
test3 = [-5, -2, -10, -1]

# Expected outputs: 9, 10, -1
```

### Problem 2: Reverse Array  
**Difficulty**: Easy  
**Topic**: Arrays  
**Hindi Explanation**: Array को reverse करना है without using built-in functions।

```python
def reverse_array(arr):
    """
    Input: [1, 2, 3, 4, 5]
    Output: [5, 4, 3, 2, 1]
    """
    # Method 1: Using two pointers
    # Method 2: Using extra space
    pass
```

### Problem 3: Find Second Largest
**Difficulty**: Easy-Medium  
**Topic**: Arrays  
**Hindi Explanation**: Array में दूसरा सबसे बड़ा element find करना है।

```python
def second_largest(arr):
    """
    Input: [12, 35, 1, 10, 34, 1]
    Output: 34
    """
    # Handle edge cases: array with less than 2 elements
    pass
```

---

## 🎯 Day 3-4: Strings और Pattern Matching

### Problem 4: Check Palindrome
**Difficulty**: Easy  
**Topic**: Strings  
**Hindi Explanation**: String palindrome है या नहीं check करना है।

```python
def is_palindrome(s):
    """
    Input: "racecar"
    Output: True
    
    Input: "hello"  
    Output: False
    """
    # Method 1: Compare with reverse
    # Method 2: Two pointers approach
    pass
```

### Problem 5: Count Vowels
**Difficulty**: Easy  
**Topic**: Strings  
**Hindi Explanation**: String में कितने vowels हैं count करना है।

```python
def count_vowels(s):
    """
    Input: "hello world"
    Output: 3 (e, o, o)
    """
    vowels = "aeiouAEIOU"
    # Your code here
    pass
```

### Problem 6: Remove Duplicates
**Difficulty**: Easy-Medium  
**Topic**: Strings  
**Hindi Explanation**: String से duplicate characters हटाना है।

```python
def remove_duplicates(s):
    """
    Input: "programming"
    Output: "progamin"
    """
    # Maintain order of first occurrence
    pass
```

---

## 🎯 Day 5-6: Mathematics और Logic

### Problem 7: Prime Number Check
**Difficulty**: Easy  
**Topic**: Mathematics  
**Hindi Explanation**: Number prime है या नहीं check करना है।

```python
def is_prime(n):
    """
    Input: 17
    Output: True
    
    Input: 15
    Output: False
    """
    # Optimize: check only up to sqrt(n)
    pass
```

### Problem 8: Factorial
**Difficulty**: Easy  
**Topic**: Mathematics/Recursion  
**Hindi Explanation**: Number का factorial calculate करना है।

```python
def factorial(n):
    """
    Input: 5
    Output: 120
    """
    # Method 1: Iterative
    # Method 2: Recursive
    pass
```

### Problem 9: Fibonacci Series
**Difficulty**: Easy-Medium  
**Topic**: Mathematics/Recursion  
**Hindi Explanation**: First n fibonacci numbers generate करना है।

```python
def fibonacci(n):
    """
    Input: 7
    Output: [0, 1, 1, 2, 3, 5, 8]
    """
    # Method 1: Iterative (efficient)
    # Method 2: Recursive (inefficient for large n)
    pass
```

---

## 🎯 Day 7: Mixed Practice

### Problem 10: Two Sum
**Difficulty**: Easy  
**Topic**: Arrays + Hash Map  
**Hindi Explanation**: Array में दो numbers find करना है जिनका sum target के बराबर हो।

```python
def two_sum(nums, target):
    """
    Input: nums = [2, 7, 11, 15], target = 9
    Output: [0, 1] (indices of 2 and 7)
    """
    # Use hash map for O(n) solution
    pass
```

---

## ✅ Solutions (देखने से पहले खुद try करें!)

<details>
<summary>Click to see solutions</summary>

### Solution 1: Find Maximum
```python
def find_maximum(arr):
    if not arr:
        return None
    
    max_val = arr[0]
    for num in arr:
        if num > max_val:
            max_val = num
    return max_val

# Alternative using built-in
def find_maximum_builtin(arr):
    return max(arr) if arr else None
```

### Solution 2: Reverse Array
```python
def reverse_array(arr):
    # Method 1: Two pointers
    left, right = 0, len(arr) - 1
    while left < right:
        arr[left], arr[right] = arr[right], arr[left]
        left += 1
        right -= 1
    return arr

# Method 2: Create new array
def reverse_array_new(arr):
    return arr[::-1]
```

### Solution 3: Second Largest
```python
def second_largest(arr):
    if len(arr) < 2:
        return None
    
    first = second = float('-inf')
    
    for num in arr:
        if num > first:
            second = first
            first = num
        elif num > second and num != first:
            second = num
    
    return second if second != float('-inf') else None
```

### Solution 4: Check Palindrome
```python
def is_palindrome(s):
    # Method 1: Compare with reverse
    return s == s[::-1]

# Method 2: Two pointers
def is_palindrome_two_pointers(s):
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True
```

### Solution 5: Count Vowels
```python
def count_vowels(s):
    vowels = "aeiouAEIOU"
    count = 0
    for char in s:
        if char in vowels:
            count += 1
    return count

# Alternative using list comprehension
def count_vowels_compact(s):
    return sum(1 for char in s if char in "aeiouAEIOU")
```

</details>

---

## 🎯 Practice Strategy

### Daily Routine:
1. **समझें** problem statement को carefully
2. **सोचें** different approaches के बारे में  
3. **लिखें** brute force solution पहले
4. **Optimize** करें time और space complexity
5. **Test** करें different test cases से

### Time Management:
- **Easy problems**: 15-20 minutes
- **Medium problems**: 30-45 minutes  
- अगर stuck हो जाएं तो solution देखें और समझें

### Key Points:
- हमेशा **edge cases** handle करें
- **Time complexity** को analyze करें
- **Space complexity** को भी consider करें
- Code को **readable** रखें

---

**Next Week**: Advanced arrays और basic algorithms! 🚀