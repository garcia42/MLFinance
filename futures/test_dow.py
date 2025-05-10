import unittest
import pandas as pd
import numpy as np
from dow import calculate_equity_curve_dow_theory

class TestDowTheory(unittest.TestCase):
    def test_trend_short_to_long_transition(self):
        # Create sample data that will trigger a trend change from short to long
        # We'll create a series that:
        # 1. Makes a swing high, Initial uptrend to a swing high (112)
        # 2. Makes a lower swing low (establishing short trend), Downtrend establishing a short position
        # 3. Then breaks above the previous swing high (triggering long trend), Break above previous high (115) triggering the trend change to long
        
        # Create enough data points to establish clear swing points and trend changes
        dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
        data = {
            # Pattern: Initial uptrend -> Swing high -> Downtrend -> Swing low -> Break above previous high
            'Open':  [100, 102, 105, 108, 110, 108, 105, 102, 98,  95,  92,  90,  88,  90,  92,  95,  98,  102, 105, 108],
            'High':  [102, 104, 107, 110, 112, 110, 107, 104, 100, 97,  94,  92,  90,  92,  94,  97,  100, 104, 107, 115], # Final break above 112
            'Low':   [98,  100, 103, 106, 108, 105, 102, 98,  95,  92,  90,  88,  86,  88,  90,  93,  96,  100, 103, 107],
            'Close': [101, 103, 106, 109, 111, 107, 104, 100, 97,  94,  91,  89,  87,  90,  93,  96,  99,  103, 106, 113]
        }
        df = pd.DataFrame(data, index=dates)
        
        # Run the analysis
        result_df, trades_df = calculate_equity_curve_dow_theory(df)
        
        # Verify the trend change was detected
        # The signal should change from -1 (short) to 1 (long) when trend changes
        self.assertTrue(any(result_df['signal'] == -1), "Short position was not detected")
        self.assertTrue(any(result_df['signal'] == 1), "Long position was not detected")
        
        # Verify we have at least one trade in the trades DataFrame
        self.assertGreater(len(trades_df), 0, "No trades were generated")
        
        # Find the transition trade (should be the last trade)
        last_trade = trades_df.iloc[-1]
        
        # Verify the last trade shows a transition from short to long
        self.assertEqual(last_trade['position'], 'long', 
                        "Last trade should be a long position after trend change")
        
        # Verify the entry price is at the point where we broke above the swing high
        self.assertAlmostEqual(last_trade['entry_price'], 113, 
                             msg="Entry price should be at the trend change point")

if __name__ == '__main__':
    unittest.main()
