# Official Implementation of "Less is More: DocString Compression in Code Generation"

## Introduction
This paper is currently under review. Therefore, we only open-source the core code for reference. The complete project code will be released after the paper is accepted.

Please refer to *usage_demo.py* for usage instructions.

You can change the model and tokenizer in *usage_demo.py* to test different models(e.g., Qwen2.5-Coder-Instruct, CodeGPT-py, etc.).

## Comparison Results

Original docstring:
```python
def tri(n):
    """Everyone knows Fibonacci sequence, it was studied deeply by mathematicians in 
    the last couple centuries. However, what people don't know is Tribonacci sequence.
    Tribonacci sequence is defined by the recurrence:
    tri(1) = 3
    tri(n) = 1 + n / 2, if n is even.
    tri(n) =  tri(n - 1) + tri(n - 2) + tri(n + 1), if n is odd.
    For example:
    tri(2) = 1 + (2 / 2) = 2
    tri(4) = 3
    tri(3) = tri(2) + tri(1) + tri(4)
           = 2 + 3 + 3 = 8 
    You are given a non-negative integer number n, you have to a return a list of the 
    first n + 1 numbers of the Tribonacci sequence.
    Examples:
    tri(3) = [1, 3, 2, 8]
    """
```

Compressed docstring:
```python
def tri(n):
    """    
    Everyone knows studied by mathematicians don't know tribonacci. tribonacci defined by tri(1) =3 tri(n) 1 + n / 2 if n is even tri(n) = tri(n - 1) + tri(n - 2) + tri(n + 1), if n is odd. example tri(2) = 1 + (2 / 2) = 2 tri() tri(3) = tri(2) + tri(1) + tri(4) = 2 + 3 + 3 = 8 non n +1 numbers of tribonacci sequence examples tri(3) [1, 3, 2, 8]
    """
```

Compression ratio: 0.35746606334841624