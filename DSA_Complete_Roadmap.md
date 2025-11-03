# 🚀 Complete DSA Roadmap - From Zero to MNC Software Engineer

## 📋 Table of Contents
1. [Journey Overview](#journey-overview)
2. [Phase-wise Learning Plan](#phase-wise-learning-plan)
3. [Essential Data Structures](#essential-data-structures)
4. [Core Algorithms](#core-algorithms)
5. [Important Patterns & Tricks](#important-patterns--tricks)
6. [Company-wise Preparation](#company-wise-preparation)
7. [Study Schedule Template](#study-schedule-template)

---

## 🎯 Journey Overview

**Goal**: MNC Software Engineer Position  
**Timeline**: 6 months intensive preparation  
**Approach**: Problem-solving focused with strong conceptual foundation

### Success Metrics
- **Month 1-2**: 150+ Easy problems solved
- **Month 3-4**: 100+ Medium problems solved  
- **Month 5-6**: 50+ Hard problems + Mock interviews
- **Final**: Ready for Google, Microsoft, Amazon, etc.

---

## 📚 Phase-wise Learning Plan

### Phase 1: Foundation Building (Month 1-2)
**Focus**: Python mastery + Basic DSA concepts

#### Week 1-2: Python Advanced Concepts
```
✅ Already Done: Python basics (variables, loops, functions)
🎯 Current Focus: Advanced Python for DSA
```

**Priority Topics:**
- **List Comprehensions & Generator Expressions**
- **Dictionary & Set Operations** (O(1) lookups)
- **String Manipulation** (slicing, formatting)
- **Built-in Functions** (sorted, min, max, zip, enumerate)
- **Collections Module** (Counter, defaultdict, deque)

#### Week 3-4: Basic Data Structures
1. **Arrays & Lists** (2-3 days)
   - Dynamic arrays, 2D arrays
   - Prefix sums, difference arrays
   - **Pattern**: Kadane's algorithm for max subarray

2. **Strings** (2-3 days)
   - String algorithms, pattern matching
   - **Tricks**: Sliding window on strings, palindrome checks

3. **Hash Maps & Sets** (2-3 days)
   - Frequency counting, grouping
   - **Pattern**: Two sum, anagram grouping

#### Week 5-6: Linear Data Structures
1. **Stacks** (2 days)
   - Monotonic stacks, expression evaluation
   - **Problems**: Next greater element, valid parentheses

2. **Queues & Deques** (2 days)
   - BFS implementation, circular queue
   - **Pattern**: Sliding window maximum

3. **Linked Lists** (3 days)
   - Singly, doubly linked lists
   - **Tricks**: Two pointers (fast/slow), cycle detection

#### Week 7-8: Sorting & Searching
1. **Binary Search** (3 days)
   - Template mastery, search in rotated array
   - **Pattern**: First/last occurrence, peak finding

2. **Sorting Algorithms** (2 days)
   - Quick sort, merge sort, counting sort
   - **Application**: Custom comparators

3. **Two Pointers** (3 days)
   - Opposite direction, same direction
   - **Problems**: 3Sum, remove duplicates

### Phase 2: Algorithm Mastery (Month 3-4)

#### Week 9-10: Tree Algorithms
1. **Binary Trees** (4 days)
   - Traversals (recursive + iterative)
   - **Patterns**: Level order, tree construction

2. **Binary Search Trees** (3 days)
   - Insert, delete, search operations
   - **Problems**: Validate BST, LCA

#### Week 11-12: Graph Algorithms  
1. **Graph Representation** (2 days)
   - Adjacency list/matrix, weighted graphs

2. **BFS & DFS** (3 days)
   - Connected components, cycle detection
   - **Applications**: Shortest path, topological sort

3. **Advanced Graph** (3 days)
   - Dijkstra's algorithm, Union-Find
   - **Problems**: Minimum spanning tree

#### Week 13-14: Dynamic Programming
1. **1D DP** (3 days)
   - Fibonacci, climbing stairs, house robber
   - **Pattern**: State transition identification

2. **2D DP** (4 days)
   - Grid problems, longest common subsequence
   - **Advanced**: Knapsack variations

#### Week 15-16: Advanced Topics
1. **Backtracking** (3 days)
   - N-Queens, sudoku solver, permutations
   - **Template**: Choice, constraint, goal

2. **Greedy Algorithms** (2 days)
   - Activity selection, fractional knapsack
   - **Pattern**: Local optimal → Global optimal

3. **Sliding Window** (2 days)
   - Fixed size, variable size windows
   - **Master Pattern**: Expand-contract technique

### Phase 3: Interview Readiness (Month 5-6)

#### Week 17-18: Company-Specific Practice
- **Google**: Trees, graphs, system design basics
- **Microsoft**: Arrays, strings, dynamic programming  
- **Amazon**: Leadership principles + coding
- **Meta**: Trees, graphs, behavioral questions

#### Week 19-20: Mock Interviews & System Design
- 2-3 mock interviews per week
- Basic system design concepts
- Behavioral question preparation

#### Week 21-24: Final Polish
- Revision of all patterns
- Speed improvement (aim for 20-30 min per medium problem)
- Advanced topics based on target companies

---

## 🏗️ Essential Data Structures

### 1. Arrays & Strings
```python
# Key Patterns
- Prefix Sums: cumsum[i] = sum(arr[:i+1])
- Two Pointers: left, right moving strategy
- Sliding Window: maintain window with property
- Kadane's Algorithm: max subarray sum

# Important Problems:
- Maximum Subarray (Kadane's)
- Longest Substring Without Repeating Characters
- Minimum Window Substring
- 3Sum Problem
```

### 2. Hash Tables (Dictionaries/Sets)
```python
# Key Patterns
- Frequency Counting: Counter(arr)
- Index Mapping: {value: index}
- Prefix Sum + HashMap: subarray sum problems
- Anagram Detection: sorted(word) as key

# Important Problems:
- Two Sum
- Group Anagrams  
- Subarray Sum Equals K
- Longest Consecutive Sequence
```

### 3. Linked Lists
```python
# Key Patterns
- Two Pointers: slow/fast for cycle detection
- Reverse: iterative and recursive
- Merge: merge sorted lists technique
- Dummy Node: simplify edge cases

# Important Problems:
- Reverse Linked List
- Merge Two Sorted Lists
- Detect Cycle
- Remove Nth Node from End
```

### 4. Stacks & Queues
```python
# Stack Patterns
- Monotonic Stack: next greater/smaller element
- Expression Evaluation: parentheses matching
- DFS Implementation: using explicit stack

# Queue Patterns  
- BFS Implementation: level-order traversal
- Sliding Window Maximum: using deque
- Producer-Consumer: queue as buffer

# Important Problems:
- Valid Parentheses
- Next Greater Element
- Sliding Window Maximum
- Binary Tree Level Order
```

### 5. Trees
```python
# Key Patterns
- DFS: preorder, inorder, postorder
- BFS: level-order traversal
- Tree Construction: from traversals
- Lowest Common Ancestor: bottom-up approach

# Important Problems:
- Binary Tree Inorder Traversal
- Maximum Depth of Binary Tree
- Validate Binary Search Tree
- Lowest Common Ancestor
- Serialize and Deserialize Binary Tree
```

### 6. Graphs
```python
# Key Patterns
- DFS: connected components, cycle detection
- BFS: shortest path in unweighted graph
- Topological Sort: Kahn's algorithm
- Union-Find: dynamic connectivity

# Important Problems:
- Number of Islands
- Course Schedule (Topological Sort)
- Shortest Path in Binary Matrix
- Accounts Merge (Union-Find)
```

### 7. Heaps
```python
# Key Patterns
- Min/Max Heap: priority operations
- K-way Merge: merge k sorted arrays
- Top K Elements: heap of size k
- Median Finding: two heaps technique

# Important Problems:
- Kth Largest Element
- Merge k Sorted Lists
- Find Median from Data Stream
- Top K Frequent Elements
```

---

## ⚙️ Core Algorithms

### 1. Sorting Algorithms
```python
# Must Know:
1. Quick Sort: O(n log n) average, O(n²) worst
2. Merge Sort: O(n log n) stable sorting
3. Heap Sort: O(n log n) in-place
4. Counting Sort: O(n+k) for limited range

# When to Use:
- Quick Sort: general purpose, in-place
- Merge Sort: stable sort needed, linked lists
- Heap Sort: guaranteed O(n log n), in-place
- Counting Sort: small range of integers
```

### 2. Searching Algorithms
```python
# Binary Search Template (Master This!)
def binary_search(arr, target):
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

# Variations:
- Find First/Last Occurrence
- Search in Rotated Array
- Find Peak Element
- Square Root (Binary Search on Answer)
```

### 3. Graph Algorithms
```python
# DFS Template
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    
    visited.add(start)
    
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    
    return visited

# BFS Template
from collections import deque

def bfs(graph, start):
    visited = set([start])
    queue = deque([start])
    
    while queue:
        node = queue.popleft()
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return visited
```

### 4. Dynamic Programming
```python
# DP Pattern Recognition:
1. Optimal Substructure: problem can be broken down
2. Overlapping Subproblems: same subproblems occur
3. State Definition: dp[i] represents what?
4. Recurrence Relation: dp[i] = f(dp[j] where j < i)
5. Base Cases: dp[0], dp[1], etc.

# Common DP Patterns:
- Linear DP: dp[i] depends on dp[i-1], dp[i-2]
- Grid DP: dp[i][j] depends on dp[i-1][j], dp[i][j-1]
- Interval DP: dp[i][j] for subarray [i...j]
- Tree DP: DP on trees
```

---

## 🎯 Important Patterns & Tricks

### 1. Two Pointers Pattern
```python
# When to Use: Arrays, strings, linked lists
# Types: Opposite direction, same direction

# Template 1: Opposite Direction
def two_sum_sorted(arr, target):
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

# Template 2: Same Direction (Slow-Fast)
def remove_duplicates(arr):
    if not arr:
        return 0
    
    slow = 0
    for fast in range(1, len(arr)):
        if arr[fast] != arr[slow]:
            slow += 1
            arr[slow] = arr[fast]
    
    return slow + 1

# Applications:
- 3Sum, 4Sum problems
- Remove duplicates
- Cycle detection in linked list
- Palindrome validation
```

### 2. Sliding Window Pattern
```python
# When to Use: Subarray/substring problems
# Types: Fixed size, variable size

# Template 1: Fixed Size Window
def max_sum_subarray(arr, k):
    if len(arr) < k:
        return 0
    
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i-k] + arr[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum

# Template 2: Variable Size Window  
def longest_substring_k_distinct(s, k):
    char_count = {}
    left = 0
    max_len = 0
    
    for right in range(len(s)):
        # Expand window
        char_count[s[right]] = char_count.get(s[right], 0) + 1
        
        # Contract window
        while len(char_count) > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1
        
        max_len = max(max_len, right - left + 1)
    
    return max_len

# Applications:
- Maximum sum subarray of size K
- Longest substring without repeating characters
- Minimum window substring
- Sliding window maximum
```

### 3. Hash Map Pattern
```python
# When to Use: Counting, grouping, fast lookups

# Template: Prefix Sum + HashMap
def subarray_sum_equals_k(nums, k):
    count = 0
    prefix_sum = 0
    sum_count = {0: 1}  # Important: empty prefix
    
    for num in nums:
        prefix_sum += num
        
        # Check if (prefix_sum - k) exists
        if prefix_sum - k in sum_count:
            count += sum_count[prefix_sum - k]
        
        sum_count[prefix_sum] = sum_count.get(prefix_sum, 0) + 1
    
    return count

# Applications:
- Two sum variations
- Anagram problems
- Subarray sum problems
- Frequency counting
```

### 4. Binary Search Pattern
```python
# When to Use: Sorted arrays, search space reduction

# Template: Search for condition
def binary_search_condition(arr, condition):
    left, right = 0, len(arr) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if condition(arr[mid]):
            result = mid
            # Search for better answer
            right = mid - 1  # or left = mid + 1
        else:
            left = mid + 1   # or right = mid - 1
    
    return result

# Applications:
- Find first/last occurrence
- Search in rotated array
- Find peak element
- Binary search on answer
```

### 5. Backtracking Pattern
```python
# When to Use: Generate all possibilities
# Template: Choice, Constraint, Goal

def backtrack(path, choices, constraints, goal):
    # Base case: reached goal
    if goal_reached(path):
        result.append(path[:])  # Copy current path
        return
    
    # Try all choices
    for choice in choices:
        # Check constraints
        if is_valid(path, choice, constraints):
            # Make choice
            path.append(choice)
            
            # Recursively explore
            backtrack(path, choices, constraints, goal)
            
            # Undo choice (backtrack)
            path.pop()

# Applications:
- Generate permutations/combinations
- N-Queens problem
- Sudoku solver
- Word search in grid
```

### 6. Dynamic Programming Patterns
```python
# Pattern 1: Linear DP
def climb_stairs(n):
    if n <= 2:
        return n
    
    dp = [0] * (n + 1)
    dp[1], dp[2] = 1, 2
    
    for i in range(3, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]

# Pattern 2: 2D DP (Grid)
def unique_paths(m, n):
    dp = [[1] * n for _ in range(m)]
    
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    
    return dp[m-1][n-1]

# Pattern 3: String DP
def longest_common_subsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]
```

---

## 🏢 Company-wise Preparation

### Google (Alphabet)
**Focus Areas:**
- Trees and Graphs (40%)
- Arrays and Strings (30%) 
- Dynamic Programming (20%)
- System Design (10%)

**Key Problems:**
- Maximum Depth of Binary Tree
- Number of Islands
- Longest Increasing Subsequence
- Merge Intervals
- Design LRU Cache

**Interview Style:**
- 2 coding rounds (45 min each)
- 1 system design (senior roles)
- Strong emphasis on optimization

### Microsoft
**Focus Areas:**
- Arrays and Strings (35%)
- Trees and Linked Lists (30%)
- Dynamic Programming (25%)
- Object-Oriented Design (10%)

**Key Problems:**
- Reverse Linked List
- Merge Two Sorted Arrays
- Word Break Problem
- Design Parking Lot
- Rotate Image

**Interview Style:**
- 4-5 rounds total
- Behavioral questions important
- Focus on clean, readable code

### Amazon
**Focus Areas:**
- Trees and Graphs (30%)
- Arrays and Strings (30%)
- System Design (25%)
- Leadership Principles (15%)

**Key Problems:**
- Two Sum
- Merge k Sorted Lists
- Design Amazon Locker
- Copy List with Random Pointer
- Trapping Rain Water

**Interview Style:**
- Strong emphasis on Leadership Principles
- Bar raiser round
- Customer obsession scenarios

### Meta (Facebook)
**Focus Areas:**
- Trees and Graphs (35%)
- Arrays and Hash Tables (30%)
- Dynamic Programming (20%)
- System Design (15%)

**Key Problems:**
- Valid Parentheses
- Clone Graph  
- Product of Array Except Self
- Design News Feed
- Binary Tree Level Order Traversal

**Interview Style:**
- Fast-paced coding rounds
- Focus on edge cases
- System design for E4+ levels

---

## 📅 Study Schedule Template

### Daily Schedule (2-3 hours/day)
```
🌅 Morning (1 hour):
- Theory study (concepts, patterns)
- Watch educational videos
- Read algorithm explanations

🌆 Evening (1-2 hours):  
- Solve 1-2 problems
- Implement learned concepts
- Review and optimize solutions

📝 Weekend (3-4 hours):
- Mock interviews
- Revision of week's topics
- Solve contest problems
```

### Weekly Progress Tracking
```markdown
## Week X Progress

### Concepts Learned:
- [ ] Topic 1: Understanding level (1-5)
- [ ] Topic 2: Understanding level (1-5)

### Problems Solved:
- [ ] Easy: X/Y target
- [ ] Medium: X/Y target  
- [ ] Hard: X/Y target

### Areas Need Improvement:
- 
- 

### Next Week Focus:
- 
```

### Monthly Milestones
**Month 1:** Python + Basic DSA (150 easy problems)  
**Month 2:** Linear structures (100 easy + 50 medium)  
**Month 3:** Trees + Graphs (80 medium problems)  
**Month 4:** DP + Advanced topics (60 medium + 20 hard)  
**Month 5:** Company-specific (40 hard problems)  
**Month 6:** Mock interviews + polish (System design)

---

## 🎯 Success Tips

### 1. Problem-Solving Approach
```
1. 🤔 Understand the problem (5 min)
   - Read carefully, identify constraints
   - Ask clarifying questions
   - Work through examples

2. 💡 Plan the solution (10 min)
   - Identify pattern/algorithm
   - Think of edge cases
   - Estimate time/space complexity

3. ⌨️ Code the solution (20 min)
   - Start with brute force if needed
   - Write clean, readable code
   - Use meaningful variable names

4. 🧪 Test and optimize (10 min)
   - Test with examples
   - Check edge cases
   - Optimize if possible
```

### 2. Learning Strategy
- **Spaced Repetition**: Review problems after 1 day, 1 week, 1 month
- **Pattern Recognition**: Group similar problems together
- **Teaching Others**: Explain solutions to solidify understanding
- **Mock Interviews**: Practice with peers or platforms

### 3. Avoiding Common Mistakes
- Don't memorize solutions, understand patterns
- Don't skip easy problems, they build confidence
- Don't ignore edge cases (empty arrays, single elements)
- Don't optimize prematurely, get working solution first

---

## 📚 Recommended Resources

### Online Platforms
1. **LeetCode**: Primary problem source
2. **GeeksforGeeks**: Theory and explanations
3. **InterviewBit**: Structured curriculum
4. **Pramp**: Free mock interviews

### Books
1. **"Cracking the Coding Interview"** - Gayle McDowell
2. **"Elements of Programming Interviews"** - Aziz, Lee, Prakash
3. **"Algorithm Design Manual"** - Steven Skiena

### YouTube Channels
1. **NeetCode**: Pattern-based explanations
2. **Abdul Bari**: Algorithm fundamentals
3. **Tushar Roy**: Detailed problem walkthroughs

---

## 🚀 Final Words

DSA mastery is a journey, not a destination. Focus on understanding patterns rather than memorizing solutions. With consistent practice and the right strategy, you'll be ready for any MNC interview!

**Remember**: 
- Quality > Quantity in problem solving
- Consistency beats intensity
- Learn from failures, they're stepping stones
- Stay motivated, your dream job awaits! 

**Good luck with your journey to becoming a Software Engineer at a top MNC!** 🎯

---

*Keep updating this roadmap as you progress. Mark completed topics and add personal notes for future reference.*