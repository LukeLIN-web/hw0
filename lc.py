from typing import List,Optional
from collections import Counter
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def minSteps(self, s: str, t: str) -> int:
        l1 = [0]*26
        l2 = [0]*26
        for i in s:
            l1[ord(i)-ord('a')] += 1
        for i in t:
            l2[ord(i)-ord('a')] += 1
        res = 0
        for i in range(26):
            res += abs(l1[i]-l2[i])
        return res//2



        
        

if __name__ == '__main__':
    s = Solution()
    res = s.minSteps("leetcode","practice")
    print(res )



