from typing import Any, List
list1: List[Any] = [0] * 5
list1[0] = ''
list2 = [0 for n in range(5)]
list2[0] = 2
print(list1, list2)
