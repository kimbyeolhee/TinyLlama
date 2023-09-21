import sys
import pickle
import warnings
from contextlib import contextmanager

def find_multiple(n: int, k: int) -> int:
    """
    n과 같거나 큰 k의 최소 배수를 반환
    """
    assert k > 0
    return n if n % k == 0 else n + k - (n % k)