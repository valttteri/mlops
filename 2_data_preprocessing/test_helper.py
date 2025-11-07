import pandas as pd

def series_approximately_same(s1: pd.Series, s2: pd.Series, tolerance: float) -> bool:
    """
    Compares whether each value in two pandas series is approximately same
    Args:
        s1 (Series): A pandas Series
        s2 (Series): A pandas Series
        tolerance (float): The tolerance level
    Returns:
        True if each value in s1 and s2 is approximately same, False otherwise
    """
    return (s1 - s2).abs().max() < tolerance






