"""
🐍 Advanced Python DSA - Comprehensive Practice Templates
===========================================================

This file contains working Python code examples covering:
1. Data Structures & Operations
2. Algorithm Patterns & Techniques  
3. Graph & Tree Algorithms
4. Dynamic Programming
5. Mathematical Algorithms
6. String Processing
7. Competitive Programming Templates
8. Advanced Python Features for DSA

Author: Danish Umer
Repository: DataStructuresandAlgorithms
"""

import sys
import math
import heapq
import bisect
import random
from collections import defaultdict, deque, Counter, namedtuple
from itertools import permutations, combinations, product
from functools import lru_cache, wraps
import time

# ===============================================
# 1. ADVANCED DATA STRUCTURES
# ===============================================

class AdvancedDataStructures:
    """Advanced data structure implementations and operations"""
    
    def __init__(self):
        self.demo()
    
    def basic_data_types_advanced(self):
        """Advanced number operations और data types"""
        # Big integers (Python handles automatically!)
        big_num = 10**100
        print(f"Big number: {big_num}")
        
        # Complex numbers
        z = 3 + 4j
        print(f"Complex: {z}, Magnitude: {abs(z)}")
        
        # Bitwise operations
        a, b = 5, 3
        print(f"Bitwise AND {a} & {b} = {a & b}")
        print(f"Bitwise OR {a} | {b} = {a | b}")
        print(f"Bitwise XOR {a} ^ {b} = {a ^ b}")
        print(f"Left shift {a} << 1 = {a << 1}")
        print(f"Right shift {a} >> 1 = {a >> 1}")
    
    def advanced_list_operations(self):
        """Advanced list operations और techniques"""
        # 2D list creation
        matrix = [[0] * 3 for _ in range(3)]  # Correct way
        # Wrong: [[0] * 3] * 3  # This creates references!
        
        # List slicing tricks
        arr = list(range(10))
        print(f"Original: {arr}")
        print(f"Every 2nd: {arr[::2]}")
        print(f"Reverse: {arr[::-1]}")
        print(f"Last 3: {arr[-3:]}")
        
        # Advanced list comprehensions
        matrix_flat = [item for row in matrix for item in row]
        nested_list = [[i*j for j in range(3)] for i in range(3)]
        filtered = [x for x in range(20) if x % 3 == 0 and x % 2 == 1]
        
        print(f"Nested list: {nested_list}")
        print(f"Filtered: {filtered}")
        
        # Sorting with custom keys
        students = [('Alice', 85), ('Bob', 90), ('Charlie', 78)]
        by_grade = sorted(students, key=lambda x: x[1], reverse=True)
        print(f"Sorted by grade: {by_grade}")
    
    def advanced_string_operations(self):
        """Advanced string manipulation techniques"""
        text = "Hello World! This is Python DSA."
        
        # String methods
        print(f"Starts with 'Hello': {text.startswith('Hello')}")
        print(f"Count of 'l': {text.count('l')}")
        print(f"Find 'World': {text.find('World')}")
        
        # String formatting (multiple ways)
        name, age = "Danish", 25
        print(f"F-string: {name} is {age} years old")
        print("Format: {} is {} years old".format(name, age))
        print("Old style: %s is %d years old" % (name, age))
        
        # Advanced formatting
        pi = math.pi
        print(f"Pi to 2 decimals: {pi:.2f}")
        print(f"Percentage: {0.1234:.2%}")
        
        # String operations for algorithms
        s = "racecar"
        print(f"Is palindrome: {s == s[::-1]}")
        
        # Join and split
        words = text.split()
        joined = " | ".join(words)
        print(f"Joined: {joined[:50]}...")
    
    def advanced_dictionary_operations(self):
        """Advanced dictionary operations और patterns"""
        # Different ways to create dictionaries
        dict1 = {"a": 1, "b": 2}
        dict2 = dict(a=1, b=2)
        dict3 = dict([("a", 1), ("b", 2)])
        dict4 = {chr(97+i): i+1 for i in range(3)}
        
        print(f"Dict comprehension: {dict4}")
        
        # Dictionary merging (Python 3.9+)
        student = {"name": "Rahul", "age": 22}
        extra = {"city": "Delhi", "grade": "A"}
        merged = student | extra
        print(f"Merged: {merged}")
        
        # Useful dictionary methods
        student.setdefault("phone", "N/A")
        print(f"With default: {student}")
        
        # Counter usage (very important for DSA!)
        text = "hello world"
        char_count = Counter(text)
        print(f"Character count: {char_count}")
        print(f"Most common: {char_count.most_common(3)}")
    
    def sets_and_advanced_collections(self):
        """Sets और advanced collections का usage"""
        # Set operations (O(1) average case!)
        set1 = {1, 2, 3, 4, 5}
        set2 = {4, 5, 6, 7, 8}
        
        print(f"Union: {set1 | set2}")
        print(f"Intersection: {set1 & set2}")
        print(f"Difference: {set1 - set2}")
        print(f"Symmetric difference: {set1 ^ set2}")
        
        # Deque (double-ended queue) - O(1) operations at both ends
        dq = deque([1, 2, 3])
        dq.appendleft(0)
        dq.append(4)
        print(f"Deque: {list(dq)}")
        
        # DefaultDict - automatic default values
        dd = defaultdict(list)
        items = [('fruit', 'apple'), ('vegetable', 'carrot'), ('fruit', 'banana')]
        for category, item in items:
            dd[category].append(item)
        print(f"DefaultDict: {dict(dd)}")
        
        # Named tuples
        Point = namedtuple('Point', ['x', 'y'])
        p = Point(3, 4)
        print(f"Point: {p}, Distance from origin: {math.sqrt(p.x**2 + p.y**2)}")
    
    def heap_operations(self):
        """Heap operations using heapq module"""
        # Min heap (default)
        heap = [3, 1, 4, 1, 5, 9, 2, 6]
        heapq.heapify(heap)
        print(f"Heapified: {heap}")
        
        heapq.heappush(heap, 0)
        print(f"After push 0: {heap}")
        
        min_element = heapq.heappop(heap)
        print(f"Popped min: {min_element}, Heap: {heap}")
        
        # K largest/smallest
        nums = [3, 1, 4, 1, 5, 9, 2, 6]
        print(f"3 largest: {heapq.nlargest(3, nums)}")
        print(f"3 smallest: {heapq.nsmallest(3, nums)}")
        
        # Max heap using negative values
        max_heap = [-x for x in [1, 3, 2, 7, 5]]
        heapq.heapify(max_heap)
        max_element = -heapq.heappop(max_heap)
        print(f"Max element: {max_element}")
    
    def demo(self):
        """Run all demonstrations"""
        print("=== ADVANCED DATA STRUCTURES DEMO ===\n")
        
        print("1. Advanced Data Types:")
        self.basic_data_types_advanced()
        print()
        
        print("2. Advanced List Operations:")
        self.advanced_list_operations()
        print()
        
        print("3. Advanced String Operations:")
        self.advanced_string_operations()
        print()
        
        print("4. Advanced Dictionary Operations:")
        self.advanced_dictionary_operations()
        print()
        
        print("5. Sets and Advanced Collections:")
        self.sets_and_advanced_collections()
        print()
        
        print("6. Heap Operations:")
        self.heap_operations()
        print()

# ===============================================
# 2. ALGORITHM PATTERNS
# ===============================================

class AlgorithmPatterns:
    """Implementation of common DSA patterns"""

    
    def two_pointers_techniques(self):
        """Advanced two pointers implementations"""
        
        def two_sum_sorted(arr, target):
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
        
        def three_sum(nums, target=0):
            """Three sum using two pointers"""
            nums.sort()
            result = []
            
            for i in range(len(nums) - 2):
                if i > 0 and nums[i] == nums[i-1]:
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
        
        def remove_duplicates_inplace(arr):
            """Remove duplicates from sorted array in-place"""
            if not arr:
                return 0
            
            write_idx = 1
            for read_idx in range(1, len(arr)):
                if arr[read_idx] != arr[read_idx - 1]:
                    arr[write_idx] = arr[read_idx]
                    write_idx += 1
            return write_idx
        
        # Test cases
        print("=== Two Pointers Techniques ===")
        arr = [1, 2, 3, 4, 5, 6]
        print(f"Two sum in {arr} with target 7: {two_sum_sorted(arr, 7)}")
        
        nums = [-1, 0, 1, 2, -1, -4]
        print(f"Three sum in {nums}: {three_sum(nums)}")
        
        duplicates = [1, 1, 2, 2, 3, 3, 4]
        new_length = remove_duplicates_inplace(duplicates)
        print(f"Array after removing duplicates: {duplicates[:new_length]}")
    
    def sliding_window_techniques(self):
        """Advanced sliding window implementations"""
        
        def max_sum_subarray(arr, k):
            """Fixed size sliding window - max sum"""
            if len(arr) < k:
                return None
            
            window_sum = sum(arr[:k])
            max_sum = window_sum
            
            for i in range(k, len(arr)):
                window_sum = window_sum - arr[i-k] + arr[i]
                max_sum = max(max_sum, window_sum)
            return max_sum
        
        def longest_substring_k_distinct(s, k):
            """Variable size window - longest substring with k distinct chars"""
            if not s or k == 0:
                return 0
            
            char_count = {}
            left = 0
            max_length = 0
            
            for right in range(len(s)):
                char_count[s[right]] = char_count.get(s[right], 0) + 1
                
                while len(char_count) > k:
                    char_count[s[left]] -= 1
                    if char_count[s[left]] == 0:
                        del char_count[s[left]]
                    left += 1
                
                max_length = max(max_length, right - left + 1)
            return max_length
        
        def min_window_substring(s, t):
            """Minimum window substring containing all chars of t"""
            if not s or not t:
                return ""
            
            dict_t = Counter(t)
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
        
        # Test cases
        print("=== Sliding Window Techniques ===")
        arr = [1, 4, 2, 10, 23, 3, 1, 0, 20]
        print(f"Max sum subarray of size 4: {max_sum_subarray(arr, 4)}")
        
        s = "eceba"
        print(f"Longest substring with 2 distinct chars in '{s}': {longest_substring_k_distinct(s, 2)}")
        
        s, t = "ADOBECODEBANC", "ABC"
        print(f"Minimum window substring: {min_window_substring(s, t)}")
    
    def hash_map_techniques(self):
        """Advanced hash map pattern implementations"""
        
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
                key = ''.join(sorted(s))
                if key not in anagrams:
                    anagrams[key] = []
                anagrams[key].append(s)
            
            return list(anagrams.values())
        
        def subarray_sum_equals_k(nums, k):
            """Count subarrays with sum equal to k"""
            count = 0
            prefix_sum = 0
            sum_count = {0: 1}
            
            for num in nums:
                prefix_sum += num
                if prefix_sum - k in sum_count:
                    count += sum_count[prefix_sum - k]
                sum_count[prefix_sum] = sum_count.get(prefix_sum, 0) + 1
            
            return count
        
        def longest_consecutive_sequence(nums):
            """Find longest consecutive sequence"""
            if not nums:
                return 0
            
            num_set = set(nums)
            longest_streak = 0
            
            for num in num_set:
                if num - 1 not in num_set:  # Start of sequence
                    current_num = num
                    current_streak = 1
                    
                    while current_num + 1 in num_set:
                        current_num += 1
                        current_streak += 1
                    
                    longest_streak = max(longest_streak, current_streak)
            
            return longest_streak
        
        # Test cases
        print("=== Hash Map Techniques ===")
        nums = [2, 7, 11, 15]
        print(f"Two sum in {nums} with target 9: {two_sum(nums, 9)}")
        
        strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
        print(f"Grouped anagrams: {group_anagrams(strs)}")
        
        nums = [1, 1, 1]
        print(f"Subarrays with sum 2: {subarray_sum_equals_k(nums, 2)}")
        
        nums = [100, 4, 200, 1, 3, 2]
        print(f"Longest consecutive sequence: {longest_consecutive_sequence(nums)}")
    
    def binary_search_techniques(self):
        """Binary search pattern implementations"""
        
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
        
        def find_first_last_occurrence(arr, target):
            """Find first and last occurrence"""
            def find_first():
                left, right = 0, len(arr) - 1
                result = -1
                while left <= right:
                    mid = left + (right - left) // 2
                    if arr[mid] == target:
                        result = mid
                        right = mid - 1
                    elif arr[mid] < target:
                        left = mid + 1
                    else:
                        right = mid - 1
                return result
            
            def find_last():
                left, right = 0, len(arr) - 1
                result = -1
                while left <= right:
                    mid = left + (right - left) // 2
                    if arr[mid] == target:
                        result = mid
                        left = mid + 1
                    elif arr[mid] < target:
                        left = mid + 1
                    else:
                        right = mid - 1
                return result
            
            return [find_first(), find_last()]
        
        def search_rotated_array(arr, target):
            """Search in rotated sorted array"""
            left, right = 0, len(arr) - 1
            
            while left <= right:
                mid = left + (right - left) // 2
                
                if arr[mid] == target:
                    return mid
                
                if arr[left] <= arr[mid]:  # Left half sorted
                    if arr[left] <= target < arr[mid]:
                        right = mid - 1
                    else:
                        left = mid + 1
                else:  # Right half sorted
                    if arr[mid] < target <= arr[right]:
                        left = mid + 1
                    else:
                        right = mid - 1
            return -1
        
        def find_peak_element(arr):
            """Find any peak element"""
            left, right = 0, len(arr) - 1
            
            while left < right:
                mid = left + (right - left) // 2
                if arr[mid] > arr[mid + 1]:
                    right = mid
                else:
                    left = mid + 1
            return left
        
        # Test cases
        print("=== Binary Search Techniques ===")
        arr = [1, 3, 5, 7, 9, 11, 13]
        print(f"Search 7 in {arr}: index {binary_search_basic(arr, 7)}")
        
        arr = [5, 7, 7, 8, 8, 10]
        print(f"First and last occurrence of 8: {find_first_last_occurrence(arr, 8)}")
        
        arr = [4, 5, 6, 7, 0, 1, 2]
        print(f"Search 0 in rotated array: index {search_rotated_array(arr, 0)}")
        
        arr = [1, 2, 1, 3, 5, 6, 4]
        print(f"Peak element at index: {find_peak_element(arr)}")
    
    def demo(self):
        """Run all algorithm pattern demonstrations"""
        print("=== ALGORITHM PATTERNS DEMO ===\n")
        
        self.two_pointers_techniques()
        print()
        
        self.sliding_window_techniques()
        print()
        
        self.hash_map_techniques()
        print()
        
        self.binary_search_techniques()
        print()

# ===============================================
# 3. DYNAMIC PROGRAMMING
# ===============================================

class DynamicProgramming:
    """Dynamic programming implementations"""

    
    def basic_dp_problems(self):
        """Basic dynamic programming problems"""
        
        @lru_cache(maxsize=None)
        def fibonacci_memo(n):
            """Fibonacci with memoization"""
            if n <= 1:
                return n
            return fibonacci_memo(n-1) + fibonacci_memo(n-2)
        
        def fibonacci_dp(n):
            """Fibonacci with tabulation"""
            if n <= 1:
                return n
            
            dp = [0] * (n + 1)
            dp[1] = 1
            
            for i in range(2, n + 1):
                dp[i] = dp[i-1] + dp[i-2]
            return dp[n]
        
        def coin_change(coins, amount):
            """Minimum coins needed"""
            dp = [float('inf')] * (amount + 1)
            dp[0] = 0
            
            for coin in coins:
                for i in range(coin, amount + 1):
                    dp[i] = min(dp[i], dp[i - coin] + 1)
            
            return dp[amount] if dp[amount] != float('inf') else -1
        
        def longest_increasing_subsequence(arr):
            """Length of LIS"""
            if not arr:
                return 0
            
            dp = [1] * len(arr)
            
            for i in range(1, len(arr)):
                for j in range(i):
                    if arr[j] < arr[i]:
                        dp[i] = max(dp[i], dp[j] + 1)
            
            return max(dp)
        
        # Test cases
        print("=== Basic DP Problems ===")
        print(f"Fibonacci(10): {fibonacci_dp(10)}")
        print(f"Coin change for amount 11 with [1,2,5]: {coin_change([1, 2, 5], 11)}")
        print(f"LIS of [10,9,2,5,3,7,101,18]: {longest_increasing_subsequence([10,9,2,5,3,7,101,18])}")
    
    def advanced_dp_problems(self):
        """Advanced DP problems"""
        
        def knapsack_01(weights, values, capacity):
            """0-1 Knapsack problem"""
            n = len(weights)
            dp = [[0] * (capacity + 1) for _ in range(n + 1)]
            
            for i in range(1, n + 1):
                for w in range(capacity + 1):
                    if weights[i-1] <= w:
                        dp[i][w] = max(
                            values[i-1] + dp[i-1][w - weights[i-1]],
                            dp[i-1][w]
                        )
                    else:
                        dp[i][w] = dp[i-1][w]
            
            return dp[n][capacity]
        
        def edit_distance(word1, word2):
            """Minimum edit distance"""
            m, n = len(word1), len(word2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            # Initialize base cases
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if word1[i-1] == word2[j-1]:
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        dp[i][j] = 1 + min(
                            dp[i-1][j],    # Delete
                            dp[i][j-1],    # Insert
                            dp[i-1][j-1]   # Replace
                        )
            
            return dp[m][n]
        
        def longest_common_subsequence(text1, text2):
            """LCS using DP"""
            m, n = len(text1), len(text2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if text1[i-1] == text2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        def max_subarray_sum(arr):
            """Kadane's algorithm for max subarray sum"""
            if not arr:
                return 0
            
            max_ending_here = max_so_far = arr[0]
            
            for i in range(1, len(arr)):
                max_ending_here = max(arr[i], max_ending_here + arr[i])
                max_so_far = max(max_so_far, max_ending_here)
            
            return max_so_far
        
        # Test cases
        print("=== Advanced DP Problems ===")
        weights, values, capacity = [1, 3, 4, 5], [1, 4, 5, 7], 7
        print(f"Knapsack result: {knapsack_01(weights, values, capacity)}")
        
        word1, word2 = "horse", "ros"
        print(f"Edit distance between '{word1}' and '{word2}': {edit_distance(word1, word2)}")
        
        text1, text2 = "abcde", "ace"
        print(f"LCS of '{text1}' and '{text2}': {longest_common_subsequence(text1, text2)}")
        
        arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
        print(f"Max subarray sum: {max_subarray_sum(arr)}")
    
    def demo(self):
        """Run all DP demonstrations"""
        print("=== DYNAMIC PROGRAMMING DEMO ===\n")
        
        self.basic_dp_problems()
        print()
        
        self.advanced_dp_problems()
        print()

# ===============================================
# 4. GRAPH ALGORITHMS
# ===============================================

class GraphAlgorithms:
    """Graph algorithm implementations"""
    
    def graph_representations(self):
        """Different ways to represent graphs"""
        # Adjacency List
        graph_adj_list = {
            1: [2, 3],
            2: [1, 4],
            3: [1, 4],
            4: [2, 3]
        }
        
        # Adjacency Matrix
        n = 5
        graph_adj_matrix = [[0] * n for _ in range(n)]
        edges = [(0, 1), (0, 2), (1, 3), (2, 4)]
        for u, v in edges:
            graph_adj_matrix[u][v] = 1
            graph_adj_matrix[v][u] = 1
        
        # Edge List
        edge_list = [(1, 2), (1, 3), (2, 4), (3, 4)]
        
        print("=== Graph Representations ===")
        print(f"Adjacency List: {graph_adj_list}")
        print(f"Edge List: {edge_list}")
        
        return graph_adj_list
    
    def bfs_dfs_implementations(self):
        """BFS and DFS implementations"""
        
        def bfs_traversal(graph, start):
            """BFS using queue"""
            visited = set()
            queue = deque([start])
            result = []
            
            while queue:
                node = queue.popleft()
                if node not in visited:
                    visited.add(node)
                    result.append(node)
                    
                    for neighbor in graph.get(node, []):
                        if neighbor not in visited:
                            queue.append(neighbor)
            
            return result
        
        def dfs_recursive(graph, node, visited=None):
            """DFS recursive implementation"""
            if visited is None:
                visited = set()
            
            visited.add(node)
            result = [node]
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    result.extend(dfs_recursive(graph, neighbor, visited))
            
            return result
        
        def dfs_iterative(graph, start):
            """DFS using stack"""
            visited = set()
            stack = [start]
            result = []
            
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    result.append(node)
                    
                    for neighbor in graph.get(node, []):
                        if neighbor not in visited:
                            stack.append(neighbor)
            
            return result
        
        def bfs_shortest_path(graph, start, end):
            """BFS for shortest path in unweighted graph"""
            if start == end:
                return [start]
            
            queue = deque([(start, [start])])
            visited = {start}
            
            while queue:
                node, path = queue.popleft()
                
                for neighbor in graph.get(node, []):
                    if neighbor == end:
                        return path + [neighbor]
                    
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
            
            return []
        
        # Test cases
        graph = {1: [2, 3], 2: [1, 4], 3: [1, 4], 4: [2, 3]}
        
        print("=== BFS/DFS Implementations ===")
        print(f"BFS from node 1: {bfs_traversal(graph, 1)}")
        print(f"DFS recursive from node 1: {dfs_recursive(graph, 1)}")
        print(f"DFS iterative from node 1: {dfs_iterative(graph, 1)}")
        print(f"Shortest path from 1 to 4: {bfs_shortest_path(graph, 1, 4)}")
    
    def cycle_detection(self):
        """Cycle detection algorithms"""
        
        def has_cycle_undirected(graph):
            """Detect cycle in undirected graph using DFS"""
            visited = set()
            
            def dfs(node, parent):
                visited.add(node)
                
                for neighbor in graph.get(node, []):
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
        
        def has_cycle_directed(graph):
            """Detect cycle in directed graph using DFS"""
            WHITE, GRAY, BLACK = 0, 1, 2
            color = defaultdict(int)
            
            def dfs(node):
                if color[node] == GRAY:
                    return True
                if color[node] == BLACK:
                    return False
                
                color[node] = GRAY
                
                for neighbor in graph.get(node, []):
                    if dfs(neighbor):
                        return True
                
                color[node] = BLACK
                return False
            
            for node in graph:
                if color[node] == WHITE:
                    if dfs(node):
                        return True
            
            return False
        
        # Test cases
        undirected_graph = {1: [2], 2: [1, 3], 3: [2, 4], 4: [3, 1]}
        directed_graph = {1: [2], 2: [3], 3: [4], 4: [2]}
        
        print("=== Cycle Detection ===")
        print(f"Undirected graph has cycle: {has_cycle_undirected(undirected_graph)}")
        print(f"Directed graph has cycle: {has_cycle_directed(directed_graph)}")
    
    def topological_sort(self):
        """Topological sorting implementations"""
        
        def topological_sort_dfs(graph):
            """Topological sort using DFS"""
            visited = set()
            stack = []
            
            def dfs(node):
                visited.add(node)
                for neighbor in graph.get(node, []):
                    if neighbor not in visited:
                        dfs(neighbor)
                stack.append(node)
            
            for node in graph:
                if node not in visited:
                    dfs(node)
            
            return stack[::-1]
        
        def topological_sort_kahn(graph):
            """Kahn's algorithm for topological sort"""
            in_degree = defaultdict(int)
            all_nodes = set()
            
            # Calculate in-degrees
            for node in graph:
                all_nodes.add(node)
                for neighbor in graph[node]:
                    all_nodes.add(neighbor)
                    in_degree[neighbor] += 1
            
            # Find nodes with no incoming edges
            queue = deque([node for node in all_nodes if in_degree[node] == 0])
            result = []
            
            while queue:
                node = queue.popleft()
                result.append(node)
                
                for neighbor in graph.get(node, []):
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
            
            return result if len(result) == len(all_nodes) else []
        
        # Test case - DAG
        dag = {5: [2, 0], 4: [0, 1], 2: [3], 3: [1]}
        
        print("=== Topological Sort ===")
        print(f"DFS topological sort: {topological_sort_dfs(dag)}")
        print(f"Kahn's algorithm: {topological_sort_kahn(dag)}")
    
    def demo(self):
        """Run all graph algorithm demonstrations"""
        print("=== GRAPH ALGORITHMS DEMO ===\n")
        
        self.graph_representations()
        print()
        
        self.bfs_dfs_implementations()
        print()
        
        self.cycle_detection()
        print()
        
        self.topological_sort()
        print()

# ===============================================
# 5. TREE ALGORITHMS
# ===============================================

class TreeNode:
    """Binary tree node definition"""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
    def __repr__(self):
        return f"TreeNode({self.val})"

class TreeAlgorithms:
    """Tree algorithm implementations"""

    
    def tree_traversals(self):
        """Tree traversal implementations"""
        
        def inorder_recursive(root):
            """Inorder traversal - recursive"""
            if not root:
                return []
            return (inorder_recursive(root.left) + 
                   [root.val] + 
                   inorder_recursive(root.right))
        
        def inorder_iterative(root):
            """Inorder traversal - iterative"""
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
        
        def preorder_iterative(root):
            """Preorder traversal - iterative"""
            if not root:
                return []
            
            result = []
            stack = [root]
            
            while stack:
                node = stack.pop()
                result.append(node.val)
                
                if node.right:
                    stack.append(node.right)
                if node.left:
                    stack.append(node.left)
            
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
        
        # Create test tree
        #       3
        #      / \
        #     9   20
        #        /  \
        #       15   7
        root = TreeNode(3)
        root.left = TreeNode(9)
        root.right = TreeNode(20)
        root.right.left = TreeNode(15)
        root.right.right = TreeNode(7)
        
        print("=== Tree Traversals ===")
        print(f"Inorder recursive: {inorder_recursive(root)}")
        print(f"Inorder iterative: {inorder_iterative(root)}")
        print(f"Preorder iterative: {preorder_iterative(root)}")
        print(f"Level order: {level_order_traversal(root)}")
    
    def tree_properties(self):
        """Tree property calculations"""
        
        def max_depth(root):
            """Maximum depth of tree"""
            if not root:
                return 0
            return 1 + max(max_depth(root.left), max_depth(root.right))
        
        def is_balanced(root):
            """Check if tree is height-balanced"""
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
        
        def diameter_of_tree(root):
            """Diameter of binary tree"""
            self.diameter = 0
            
            def depth(node):
                if not node:
                    return 0
                
                left_depth = depth(node.left)
                right_depth = depth(node.right)
                
                # Update diameter at each node
                self.diameter = max(self.diameter, left_depth + right_depth)
                
                return max(left_depth, right_depth) + 1
            
            depth(root)
            return self.diameter
        
        def lowest_common_ancestor(root, p, q):
            """LCA in binary tree"""
            if not root or root == p or root == q:
                return root
            
            left = lowest_common_ancestor(root.left, p, q)
            right = lowest_common_ancestor(root.right, p, q)
            
            if left and right:
                return root
            
            return left if left else right
        
        # Create test tree
        root = TreeNode(3)
        root.left = TreeNode(9)
        root.right = TreeNode(20)
        root.right.left = TreeNode(15)
        root.right.right = TreeNode(7)
        
        print("=== Tree Properties ===")
        print(f"Max depth: {max_depth(root)}")
        print(f"Is balanced: {is_balanced(root)}")
        print(f"Diameter: {diameter_of_tree(root)}")
    
    def demo(self):
        """Run all tree algorithm demonstrations"""
        print("=== TREE ALGORITHMS DEMO ===\n")
        
        self.tree_traversals()
        print()
        
        self.tree_properties()
        print()

# ===============================================
# 6. MATHEMATICAL ALGORITHMS
# ===============================================

class MathematicalAlgorithms:
    """Mathematical algorithm implementations"""
    
    def number_theory(self):
        """Number theory algorithms"""
        
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
        
        def gcd_euclidean(a, b):
            """Greatest Common Divisor"""
            while b:
                a, b = b, a % b
            return a
        
        def extended_gcd(a, b):
            """Extended Euclidean Algorithm"""
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y
        
        def prime_factors(n):
            """Prime factorization"""
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
        
        # Test cases
        print("=== Number Theory ===")
        print(f"Is 17 prime: {is_prime(17)}")
        print(f"Primes up to 20: {sieve_of_eratosthenes(20)}")
        print(f"GCD of 48 and 18: {gcd_euclidean(48, 18)}")
        print(f"Prime factors of 60: {prime_factors(60)}")
        print(f"2^10 mod 1000: {mod_exp(2, 10, 1000)}")
    
    def combinatorics(self):
        """Combinatorial algorithms"""
        
        def factorial(n):
            """Calculate factorial"""
            if n <= 1:
                return 1
            result = 1
            for i in range(2, n + 1):
                result *= i
            return result
        
        def nCr_optimized(n, r):
            """Optimized combination calculation"""
            if r > n - r:
                r = n - r
            
            result = 1
            for i in range(r):
                result = result * (n - i) // (i + 1)
            return result
        
        def generate_permutations(arr):
            """Generate all permutations"""
            if len(arr) <= 1:
                return [arr]
            
            result = []
            for i in range(len(arr)):
                rest = arr[:i] + arr[i+1:]
                for p in generate_permutations(rest):
                    result.append([arr[i]] + p)
            return result
        
        def generate_subsets(arr):
            """Generate all subsets (power set)"""
            result = []
            n = len(arr)
            
            for i in range(2**n):
                subset = []
                for j in range(n):
                    if i & (1 << j):
                        subset.append(arr[j])
                result.append(subset)
            
            return result
        
        # Test cases
        print("=== Combinatorics ===")
        print(f"5! = {factorial(5)}")
        print(f"C(5,2) = {nCr_optimized(5, 2)}")
        print(f"Permutations of [1,2,3]: {generate_permutations([1, 2, 3])}")
        print(f"Subsets of [1,2]: {generate_subsets([1, 2])}")
    
    def demo(self):
        """Run all mathematical algorithm demonstrations"""
        print("=== MATHEMATICAL ALGORITHMS DEMO ===\n")
        
        self.number_theory()
        print()
        
        self.combinatorics()
        print()

# ===============================================
# 7. STRING ALGORITHMS
# ===============================================

class StringAlgorithms:
    """String algorithm implementations"""
    
    def pattern_matching(self):
        """String pattern matching algorithms"""
        
        def naive_search(text, pattern):
            """Naive string matching"""
            matches = []
            n, m = len(text), len(pattern)
            
            for i in range(n - m + 1):
                j = 0
                while j < m and text[i + j] == pattern[j]:
                    j += 1
                if j == m:
                    matches.append(i)
            
            return matches
        
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
        
        # Test cases
        text = "ABABDABACDABABCABCABCABCABC"
        pattern = "ABABCABCABCABC"
        
        print("=== Pattern Matching ===")
        print(f"Naive search: {naive_search(text, pattern)}")
        print(f"KMP search: {kmp_search(text, pattern)}")
    
    def string_processing(self):
        """String processing algorithms"""
        
        def is_palindrome_optimized(s):
            """Check palindrome ignoring case and spaces"""
            # Clean string
            cleaned = ''.join(char.lower() for char in s if char.isalnum())
            
            left, right = 0, len(cleaned) - 1
            while left < right:
                if cleaned[left] != cleaned[right]:
                    return False
                left += 1
                right -= 1
            return True
        
        def longest_palindromic_substring(s):
            """Find longest palindromic substring"""
            if not s:
                return ""
            
            start = 0
            max_len = 1
            
            def expand_around_center(left, right):
                while left >= 0 and right < len(s) and s[left] == s[right]:
                    left -= 1
                    right += 1
                return right - left - 1
            
            for i in range(len(s)):
                # Odd length palindromes
                len1 = expand_around_center(i, i)
                # Even length palindromes
                len2 = expand_around_center(i, i + 1)
                
                curr_max = max(len1, len2)
                if curr_max > max_len:
                    max_len = curr_max
                    start = i - (curr_max - 1) // 2
            
            return s[start:start + max_len]
        
        def anagram_groups(strs):
            """Group anagrams together"""
            groups = defaultdict(list)
            
            for s in strs:
                key = ''.join(sorted(s))
                groups[key].append(s)
            
            return list(groups.values())
        
        def string_compression(s):
            """Basic string compression"""
            if not s:
                return s
            
            compressed = []
            count = 1
            
            for i in range(1, len(s)):
                if s[i] == s[i-1]:
                    count += 1
                else:
                    compressed.append(s[i-1] + str(count))
                    count = 1
            
            compressed.append(s[-1] + str(count))
            result = ''.join(compressed)
            
            return result if len(result) < len(s) else s
        
        # Test cases
        print("=== String Processing ===")
        print(f"Is 'A man a plan a canal Panama' palindrome: {is_palindrome_optimized('A man a plan a canal Panama')}")
        print(f"Longest palindrome in 'babad': {longest_palindromic_substring('babad')}")
        
        strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
        print(f"Anagram groups: {anagram_groups(strs)}")
        print(f"Compressed 'aabcccccaaa': {string_compression('aabcccccaaa')}")
    
    def demo(self):
        """Run all string algorithm demonstrations"""
        print("=== STRING ALGORITHMS DEMO ===\n")
        
        self.pattern_matching()
        print()
        
        self.string_processing()
        print()

# ===============================================
# 8. COMPETITIVE PROGRAMMING TEMPLATES
# ===============================================

class CompetitiveProgrammingTemplates:
    """Templates for competitive programming"""
    
    def fast_io_template(self):
        """Fast I/O template for competitive programming"""
        template = '''
import sys
from collections import *
from bisect import *
from heapq import *

def main():
    # Fast I/O
    input = sys.stdin.readline
    
    # Single integer
    n = int(input())
    
    # Multiple integers
    a, b, c = map(int, input().split())
    
    # List of integers
    arr = list(map(int, input().split()))
    
    # String (strip newline)
    s = input().strip()
    
    # Your solution here
    result = solve(arr)
    print(result)

def solve(arr):
    # Implementation goes here
    return sum(arr)

# Multiple test cases template
def solve_multiple():
    t = int(input())
    for _ in range(t):
        n = int(input())
        arr = list(map(int, input().split()))
        
        result = solve(arr)
        print(result)

if __name__ == "__main__":
    main()
'''
        print("=== Fast I/O Template ===")
        print("Template for competitive programming:")
        print(template)
    
    def common_snippets(self):
        """Common code snippets for competitive programming"""
        
        def read_matrix(n, m):
            """Read n x m matrix"""
            matrix = []
            for _ in range(n):
                row = list(map(int, input().split()))
                matrix.append(row)
            return matrix
        
        def read_graph_adjacency_list(n, m):
            """Read graph as adjacency list"""
            graph = defaultdict(list)
            for _ in range(m):
                u, v = map(int, input().split())
                graph[u].append(v)
                graph[v].append(u)  # For undirected
            return graph
        
        def binary_search_answer(check_function, left, right):
            """Binary search on answer"""
            while left < right:
                mid = (left + right) // 2
                if check_function(mid):
                    right = mid
                else:
                    left = mid + 1
            return left
        
        def merge_intervals(intervals):
            """Merge overlapping intervals"""
            if not intervals:
                return []
            
            intervals.sort(key=lambda x: x[0])
            merged = [intervals[0]]
            
            for current in intervals[1:]:
                if current[0] <= merged[-1][1]:
                    merged[-1] = [merged[-1][0], max(merged[-1][1], current[1])]
                else:
                    merged.append(current)
            
            return merged
        
        print("=== Common Snippets ===")
        print("Matrix reading, graph input, binary search on answer, etc.")
        
        # Example usage
        intervals = [[1, 3], [2, 6], [8, 10], [15, 18]]
        print(f"Merged intervals: {merge_intervals(intervals)}")
    
    def demo(self):
        """Run competitive programming demonstrations"""
        print("=== COMPETITIVE PROGRAMMING TEMPLATES ===\n")
        
        self.fast_io_template()
        print()
        
        self.common_snippets()
        print()

# ===============================================
# 9. PRACTICE PROBLEMS
# ===============================================

class PracticeProblems:
    """Collection of practice problems with solutions"""
    
    def easy_problems(self):
        """Easy level practice problems"""
        
        def find_maximum(arr):
            """Find maximum element in array"""
            return max(arr) if arr else None
        
        def reverse_string(s):
            """Reverse a string"""
            return s[::-1]
        
        def count_vowels(s):
            """Count vowels in a string"""
            vowels = "aeiouAEIOU"
            return sum(1 for char in s if char in vowels)
        
        def is_palindrome(s):
            """Check if string is palindrome"""
            cleaned = ''.join(char.lower() for char in s if char.isalnum())
            return cleaned == cleaned[::-1]
        
        def factorial_iterative(n):
            """Calculate factorial iteratively"""
            result = 1
            for i in range(1, n + 1):
                result *= i
            return result
        
        def fibonacci_sequence(n):
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
        
        def binary_search_implementation(arr, target):
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
        
        # Test all easy problems
        print("=== Easy Practice Problems ===")
        test_arr = [3, 1, 4, 1, 5, 9, 2, 6]
        print(f"Maximum in {test_arr}: {find_maximum(test_arr)}")
        
        test_string = "hello world"
        print(f"Reverse of '{test_string}': {reverse_string(test_string)}")
        print(f"Vowels in '{test_string}': {count_vowels(test_string)}")
        print(f"Is 'racecar' palindrome: {is_palindrome('racecar')}")
        print(f"Factorial of 5: {factorial_iterative(5)}")
        print(f"First 8 fibonacci numbers: {fibonacci_sequence(8)}")
        
        sorted_arr = [1, 3, 5, 7, 9, 11, 13]
        print(f"Binary search for 7: index {binary_search_implementation(sorted_arr, 7)}")
    
    def medium_problems(self):
        """Medium level practice problems"""
        
        def merge_sorted_arrays(arr1, arr2):
            """Merge two sorted arrays"""
            result = []
            i = j = 0
            
            while i < len(arr1) and j < len(arr2):
                if arr1[i] <= arr2[j]:
                    result.append(arr1[i])
                    i += 1
                else:
                    result.append(arr2[j])
                    j += 1
            
            result.extend(arr1[i:])
            result.extend(arr2[j:])
            return result
        
        def rotate_array(arr, k):
            """Rotate array to the right by k steps"""
            n = len(arr)
            k = k % n
            return arr[-k:] + arr[:-k]
        
        def valid_parentheses(s):
            """Check if parentheses are valid"""
            stack = []
            mapping = {')': '(', '}': '{', ']': '['}
            
            for char in s:
                if char in mapping:
                    if not stack or stack.pop() != mapping[char]:
                        return False
                else:
                    stack.append(char)
            
            return not stack
        
        def generate_pascals_triangle(n):
            """Generate Pascal's triangle"""
            triangle = []
            
            for i in range(n):
                row = [1] * (i + 1)
                for j in range(1, i):
                    row[j] = triangle[i-1][j-1] + triangle[i-1][j]
                triangle.append(row)
            
            return triangle
        
        # Test medium problems
        print("=== Medium Practice Problems ===")
        arr1, arr2 = [1, 3, 5], [2, 4, 6]
        print(f"Merged arrays: {merge_sorted_arrays(arr1, arr2)}")
        
        arr = [1, 2, 3, 4, 5, 6, 7]
        print(f"Rotated by 3: {rotate_array(arr, 3)}")
        
        print(f"Valid parentheses '{{[()]}}': {valid_parentheses('{[()]}')}") 
        print(f"Pascal's triangle (5 rows): {generate_pascals_triangle(5)}")
    
    def demo(self):
        """Run all practice problem demonstrations"""
        print("=== PRACTICE PROBLEMS DEMO ===\n")
        
        self.easy_problems()
        print()
        
        self.medium_problems()
        print()

# ===============================================
# MAIN DEMONSTRATION FUNCTION
# ===============================================

def run_all_demonstrations():
    """Run all demonstrations"""
    print("🐍 ADVANCED PYTHON DSA - COMPREHENSIVE DEMO")
    print("=" * 60)
    print()
    
    # Run all demonstrations
    ads = AdvancedDataStructures()
    print()
    
    ap = AlgorithmPatterns()
    ap.demo()
    
    dp = DynamicProgramming()
    dp.demo()
    
    ga = GraphAlgorithms()
    ga.demo()
    
    ta = TreeAlgorithms()
    ta.demo()
    
    ma = MathematicalAlgorithms()
    ma.demo()
    
    sa = StringAlgorithms()
    sa.demo()
    
    cpt = CompetitiveProgrammingTemplates()
    cpt.demo()
    
    pp = PracticeProblems()
    pp.demo()
    
    print("🚀 All demonstrations completed!")
    print("Check out the Python_Cheat_Sheet.md for more detailed explanations!")

if __name__ == "__main__":
    run_all_demonstrations()