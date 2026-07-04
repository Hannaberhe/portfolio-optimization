"""Basic tests for portfolio optimization."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that required packages can be imported."""
    import pandas
    import numpy
    import yfinance
    assert True

def test_sharpe_calculation():
    """Test Sharpe ratio calculation."""
    from src.eda_utils import calculate_sharpe_ratio
    import pandas as pd
    returns = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01])
    sharpe = calculate_sharpe_ratio(returns)
    assert sharpe is not None

def test_var_calculation():
    """Test VaR calculation."""
    from src.eda_utils import calculate_var
    import pandas as pd
    returns = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01])
    var = calculate_var(returns)
    assert var is not None
    assert var < 0
