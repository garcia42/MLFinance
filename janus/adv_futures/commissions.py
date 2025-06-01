# Interactive Brokers Futures Commission Mapping
# Updated as of 2024 - always verify current rates with IB
FUTURES_COMMISSION_MAP = {
    # Add these to your FUTURES_COMMISSION_MAP:
    'BZ': {  # Brent Crude Oil
        'name': 'Brent Crude Oil',
        'exchange': 'NYMEX',
        'commission': 1.25,
        'currency': 'USD',
        'contract_size': '1,000 barrels',
        'tick_size': 0.01,
        'tick_value': 10.00,
        'contract_months': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        'months_per_year': 12
    },

    'KC': {  # Coffee
        'name': 'Coffee',
        'exchange': 'ICE',
        'commission': 1.25,
        'currency': 'USD',
        'contract_size': '37,500 pounds',
        'tick_size': 0.05,
        'tick_value': 18.75,
        'contract_months': ['Mar', 'May', 'Jul', 'Sep', 'Dec'],
        'months_per_year': 5
    },

    'CT': {  # Cotton
        'name': 'Cotton',
        'exchange': 'ICE',
        'commission': 1.25,
        'currency': 'USD',
        'contract_size': '50,000 pounds',
        'tick_size': 0.01,
        'tick_value': 5.00,
        'contract_months': ['Mar', 'May', 'Jul', 'Oct', 'Dec'],
        'months_per_year': 5
    },

    'SB': {  # Sugar
        'name': 'Sugar',
        'exchange': 'ICE',
        'commission': 1.25,
        'currency': 'USD',
        'contract_size': '112,000 pounds',
        'tick_size': 0.01,
        'tick_value': 11.20,
        'contract_months': ['Mar', 'May', 'Jul', 'Oct'],
        'months_per_year': 4
    },

    'PA': {  # Palladium
        'name': 'Palladium',
        'exchange': 'NYMEX',
        'commission': 1.25,
        'currency': 'USD',
        'contract_size': '100 troy ounces',
        'tick_size': 0.05,
        'tick_value': 5.00,
        'contract_months': ['Mar', 'Jun', 'Sep', 'Dec'],
        'months_per_year': 4
    },

    'PL': {  # Platinum
        'name': 'Platinum',
        'exchange': 'NYMEX',
        'commission': 1.25,
        'currency': 'USD',
        'contract_size': '50 troy ounces',
        'tick_size': 0.10,
        'tick_value': 5.00,
        'contract_months': ['Jan', 'Apr', 'Jul', 'Oct'],
        'months_per_year': 4
    },

    'HO': {  # Heating Oil
        'name': 'Heating Oil',
        'exchange': 'NYMEX',
        'commission': 1.25,
        'currency': 'USD',
        'contract_size': '42,000 gallons',
        'tick_size': 0.0001,
        'tick_value': 4.20,
        'contract_months': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        'months_per_year': 12
    },

    'RB': {  # RBOB Gasoline
        'name': 'RBOB Gasoline',
        'exchange': 'NYMEX',
        'commission': 1.25,
        'currency': 'USD',
        'contract_size': '42,000 gallons',
        'tick_size': 0.0001,
        'tick_value': 4.20,
        'contract_months': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        'months_per_year': 12
    },
    # ===== EQUITY INDEX FUTURES =====
    'ES': {  # E-mini S&P 500
        'name': 'E-mini S&P 500',
        'exchange': 'CME',
        'commission': 0.25,
        'currency': 'USD',
        'contract_size': '$50 × S&P 500 Index',
        'tick_size': 0.25,
        'tick_value': 12.50,
        'contract_months': ['Mar', 'Jun', 'Sep', 'Dec'],  # Quarterly (H, M, U, Z)
        'months_per_year': 4
    },
    'NQ': {  # E-mini NASDAQ-100
        'name': 'E-mini NASDAQ-100',
        'exchange': 'CME',
        'commission': 0.25,
        'currency': 'USD',
        'contract_size': '$20 × NASDAQ-100 Index',
        'tick_size': 0.25,
        'tick_value': 5.00,
        'contract_months': ['Mar', 'Jun', 'Sep', 'Dec'],  # Quarterly (H, M, U, Z)
        'months_per_year': 4
    },
    'YM': {  # E-mini Dow Jones
        'name': 'E-mini Dow Jones',
        'exchange': 'CBOT',
        'commission': 0.25,
        'currency': 'USD',
        'contract_size': '$5 × Dow Jones Index',
        'tick_size': 1.0,
        'tick_value': 5.00,
        'contract_months': ['Mar', 'Jun', 'Sep', 'Dec'],  # Quarterly (H, M, U, Z)
        'months_per_year': 4
    },
    'RTY': {  # E-mini Russell 2000
        'name': 'E-mini Russell 2000',
        'exchange': 'CME',
        'commission': 0.25,
        'currency': 'USD',
        'contract_size': '$50 × Russell 2000 Index',
        'tick_size': 0.10,
        'tick_value': 5.00,
        'contract_months': ['Mar', 'Jun', 'Sep', 'Dec'],  # Quarterly (H, M, U, Z)
        'months_per_year': 4
    },
    'VX': {  # VIX Futures
        'name': 'VIX Futures',
        'exchange': 'CFE',
        'commission': 1.25,
        'currency': 'USD',
        'contract_size': '$1000 × VIX Index',
        'tick_size': 0.05,
        'tick_value': 50.00,
        'contract_months': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],  # Monthly
        'months_per_year': 12
    },
    
    # ===== CURRENCY FUTURES =====
    '6E': {  # Euro FX
        'name': 'Euro FX',
        'exchange': 'CME',
        'commission': 0.85,
        'currency': 'USD',
        'contract_size': '€125,000',
        'tick_size': 0.00005,
        'tick_value': 6.25,
        'contract_months': ['Mar', 'Jun', 'Sep', 'Dec'],  # Quarterly (H, M, U, Z)
        'months_per_year': 4
    },
    '6B': {  # British Pound
        'name': 'British Pound',
        'exchange': 'CME',
        'commission': 0.85,
        'currency': 'USD',
        'contract_size': '£62,500',
        'tick_size': 0.0001,
        'tick_value': 6.25,
        'contract_months': ['Mar', 'Jun', 'Sep', 'Dec'],  # Quarterly (H, M, U, Z)
        'months_per_year': 4
    },
    '6J': {  # Japanese Yen
        'name': 'Japanese Yen',
        'exchange': 'CME',
        'commission': 0.85,
        'currency': 'USD',
        'contract_size': '¥12,500,000',
        'tick_size': 0.000001,
        'tick_value': 12.50,
        'contract_months': ['Mar', 'Jun', 'Sep', 'Dec'],  # Quarterly (H, M, U, Z)
        'months_per_year': 4
    },
    '6A': {  # Australian Dollar
        'name': 'Australian Dollar',
        'exchange': 'CME',
        'commission': 0.85,
        'currency': 'USD',
        'contract_size': 'A$100,000',
        'tick_size': 0.0001,
        'tick_value': 10.00,
        'contract_months': ['Mar', 'Jun', 'Sep', 'Dec'],  # Quarterly (H, M, U, Z)
        'months_per_year': 4
    },
    '6C': {  # Canadian Dollar
        'name': 'Canadian Dollar',
        'exchange': 'CME',
        'commission': 0.85,
        'currency': 'USD',
        'contract_size': 'C$100,000',
        'tick_size': 0.0001,
        'tick_value': 10.00,
        'contract_months': ['Mar', 'Jun', 'Sep', 'Dec'],  # Quarterly (H, M, U, Z)
        'months_per_year': 4
    },
    
    # ===== COMMODITY FUTURES =====
    'CL': {  # Crude Oil
        'name': 'Crude Oil',
        'exchange': 'NYMEX',
        'commission': 1.25,
        'currency': 'USD',
        'contract_size': '1,000 barrels',
        'tick_size': 0.01,
        'tick_value': 10.00,
        'contract_months': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],  # Monthly
        'months_per_year': 12
    },
    'QM': {  # E-mini Crude Oil
        'name': 'E-mini Crude Oil',
        'exchange': 'NYMEX',
        'commission': 1.25,
        'currency': 'USD',
        'contract_size': '500 barrels',
        'tick_size': 0.025,
        'tick_value': 12.50,
        'contract_months': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],  # Monthly
        'months_per_year': 12
    },
    'NG': {  # Natural Gas
        'name': 'Natural Gas',
        'exchange': 'NYMEX',
        'commission': 1.25,
        'currency': 'USD',
        'contract_size': '10,000 MMBtu',
        'tick_size': 0.001,
        'tick_value': 10.00,
        'contract_months': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],  # Monthly
        'months_per_year': 12
    },
    'GC': {  # Gold
        'name': 'Gold',
        'exchange': 'COMEX',
        'commission': 1.25,
        'currency': 'USD',
        'contract_size': '100 troy ounces',
        'tick_size': 0.10,
        'tick_value': 10.00,
        'contract_months': ['Feb', 'Apr', 'Jun', 'Aug', 'Oct', 'Dec'],  # Bi-monthly (G, J, M, Q, V, Z)
        'months_per_year': 6
    },
    'QO': {  # E-micro Gold
        'name': 'E-micro Gold',
        'exchange': 'COMEX',
        'commission': 0.25,
        'currency': 'USD',
        'contract_size': '10 troy ounces',
        'tick_size': 0.10,
        'tick_value': 1.00,
        'contract_months': ['Feb', 'Apr', 'Jun', 'Aug', 'Oct', 'Dec'],  # Bi-monthly (G, J, M, Q, V, Z)
        'months_per_year': 6
    },
    'SI': {  # Silver
        'name': 'Silver',
        'exchange': 'COMEX',
        'commission': 1.25,
        'currency': 'USD',
        'contract_size': '5,000 troy ounces',
        'tick_size': 0.005,
        'tick_value': 25.00,
        'contract_months': ['Mar', 'May', 'Jul', 'Sep', 'Dec'],  # (H, K, N, U, Z)
        'months_per_year': 5
    },
    'QI': {  # E-micro Silver
        'name': 'E-micro Silver',
        'exchange': 'COMEX',
        'commission': 0.25,
        'currency': 'USD',
        'contract_size': '1,000 troy ounces',
        'tick_size': 0.005,
        'tick_value': 5.00,
        'contract_months': ['Mar', 'May', 'Jul', 'Sep', 'Dec'],  # (H, K, N, U, Z)
        'months_per_year': 5
    },
    'HG': {  # Copper
        'name': 'Copper',
        'exchange': 'COMEX',
        'commission': 1.25,
        'currency': 'USD',
        'contract_size': '25,000 pounds',
        'tick_size': 0.0005,
        'tick_value': 12.50,
        'contract_months': ['Mar', 'May', 'Jul', 'Sep', 'Dec'],  # (H, K, N, U, Z)
        'months_per_year': 5
    },
    
    # ===== AGRICULTURAL FUTURES =====
    'ZC': {  # Corn
        'name': 'Corn',
        'exchange': 'CBOT',
        'commission': 1.25,
        'currency': 'USD',
        'contract_size': '5,000 bushels',
        'tick_size': 0.0025,
        'tick_value': 12.50,
        'contract_months': ['Mar', 'May', 'Jul', 'Sep', 'Dec'],  # (H, K, N, U, Z)
        'months_per_year': 5
    },
    'ZS': {  # Soybeans
        'name': 'Soybeans',
        'exchange': 'CBOT',
        'commission': 1.25,
        'currency': 'USD',
        'contract_size': '5,000 bushels',
        'tick_size': 0.0025,
        'tick_value': 12.50,
        'contract_months': ['Jan', 'Mar', 'May', 'Jul', 'Aug', 'Sep', 'Nov'],  # (F, H, K, N, Q, U, X)
        'months_per_year': 7
    },
    'ZW': {  # Wheat
        'name': 'Wheat',
        'exchange': 'CBOT',
        'commission': 1.25,
        'currency': 'USD',
        'contract_size': '5,000 bushels',
        'tick_size': 0.0025,
        'tick_value': 12.50,
        'contract_months': ['Mar', 'May', 'Jul', 'Sep', 'Dec'],  # (H, K, N, U, Z)
        'months_per_year': 5
    },
    'LE': {  # Live Cattle
        'name': 'Live Cattle',
        'exchange': 'CME',
        'commission': 1.25,
        'currency': 'USD',
        'contract_size': '40,000 pounds',
        'tick_size': 0.00025,
        'tick_value': 10.00,
        'contract_months': ['Feb', 'Apr', 'Jun', 'Aug', 'Oct', 'Dec'],  # Bi-monthly (G, J, M, Q, V, Z)
        'months_per_year': 6
    },
    'HE': {  # Lean Hogs
        'name': 'Lean Hogs',
        'exchange': 'CME',
        'commission': 1.25,
        'currency': 'USD',
        'contract_size': '40,000 pounds',
        'tick_size': 0.00025,
        'tick_value': 10.00,
        'contract_months': ['Feb', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Oct', 'Dec'],  # (G, J, K, M, N, Q, V, Z)
        'months_per_year': 8
    },
    
    # ===== INTEREST RATE FUTURES =====
    'ZB': {  # 30-Year Treasury Bond
        'name': '30-Year Treasury Bond',
        'exchange': 'CBOT',
        'commission': 0.85,
        'currency': 'USD',
        'contract_size': '$100,000',
        'tick_size': 0.03125,
        'tick_value': 31.25,
        'contract_months': ['Mar', 'Jun', 'Sep', 'Dec'],  # Quarterly (H, M, U, Z)
        'months_per_year': 4
    },
    'ZN': {  # 10-Year Treasury Note
        'name': '10-Year Treasury Note',
        'exchange': 'CBOT',
        'commission': 0.85,
        'currency': 'USD',
        'contract_size': '$100,000',
        'tick_size': 0.015625,
        'tick_value': 15.625,
        'contract_months': ['Mar', 'Jun', 'Sep', 'Dec'],  # Quarterly (H, M, U, Z)
        'months_per_year': 4
    },
    'ZF': {  # 5-Year Treasury Note
        'name': '5-Year Treasury Note',
        'exchange': 'CBOT',
        'commission': 0.85,
        'currency': 'USD',
        'contract_size': '$100,000',
        'tick_size': 0.0078125,
        'tick_value': 7.8125,
        'contract_months': ['Mar', 'Jun', 'Sep', 'Dec'],  # Quarterly (H, M, U, Z)
        'months_per_year': 4
    },
    'ZT': {  # 2-Year Treasury Note
        'name': '2-Year Treasury Note',
        'exchange': 'CBOT',
        'commission': 0.85,
        'currency': 'USD',
        'contract_size': '$200,000',
        'tick_size': 0.00390625,
        'tick_value': 7.8125,
        'contract_months': ['Mar', 'Jun', 'Sep', 'Dec'],  # Quarterly (H, M, U, Z)
        'months_per_year': 4
    },
    
    # ===== CRYPTO FUTURES =====
    'BTC': {  # Bitcoin
        'name': 'Bitcoin',
        'exchange': 'CME',
        'commission': 6.00,
        'currency': 'USD',
        'contract_size': '5 Bitcoin',
        'tick_size': 5.00,
        'tick_value': 25.00,
        'contract_months': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],  # Monthly
        'months_per_year': 12
    },
    'MBT': {  # Micro Bitcoin
        'name': 'Micro Bitcoin',
        'exchange': 'CME',
        'commission': 2.50,
        'currency': 'USD',
        'contract_size': '0.1 Bitcoin',
        'tick_size': 1.00,
        'tick_value': 0.10,
        'contract_months': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],  # Monthly
        'months_per_year': 12
    },
    'ETH': {  # Ether
        'name': 'Ether',
        'exchange': 'CME',
        'commission': 6.00,
        'currency': 'USD',
        'contract_size': '50 Ether',
        'tick_size': 0.05,
        'tick_value': 2.50,
        'contract_months': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],  # Monthly
        'months_per_year': 12
    },
    'MET': {  # Micro Ether
        'name': 'Micro Ether',
        'exchange': 'CME',
        'commission': 2.50,
        'currency': 'USD',
        'contract_size': '0.1 Ether',
        'tick_size': 0.01,
        'tick_value': 0.001,
        'contract_months': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],  # Monthly
        'months_per_year': 12
    }
}

# Micro futures with lower commissions
MICRO_FUTURES_MAP = {
    'MES': {  # Micro E-mini S&P 500
        'name': 'Micro E-mini S&P 500',
        'exchange': 'CME',
        'commission': 0.25,
        'currency': 'USD',
        'contract_size': '$5 × S&P 500 Index',
        'tick_size': 0.25,
        'tick_value': 1.25,
        'contract_months': ['Mar', 'Jun', 'Sep', 'Dec'],  # Quarterly (H, M, U, Z)
        'months_per_year': 4
    },
    'MNQ': {  # Micro E-mini NASDAQ-100
        'name': 'Micro E-mini NASDAQ-100',
        'exchange': 'CME',
        'commission': 0.25,
        'currency': 'USD',
        'contract_size': '$2 × NASDAQ-100 Index',
        'tick_size': 0.25,
        'tick_value': 0.50,
        'contract_months': ['Mar', 'Jun', 'Sep', 'Dec'],  # Quarterly (H, M, U, Z)
        'months_per_year': 4
    },
    'MYM': {  # Micro E-mini Dow
        'name': 'Micro E-mini Dow',
        'exchange': 'CBOT',
        'commission': 0.25,
        'currency': 'USD',
        'contract_size': '$0.50 × Dow Jones Index',
        'tick_size': 1.0,
        'tick_value': 0.50,
        'contract_months': ['Mar', 'Jun', 'Sep', 'Dec'],  # Quarterly (H, M, U, Z)
        'months_per_year': 4
    },
    'M2K': {  # Micro E-mini Russell 2000
        'name': 'Micro E-mini Russell 2000',
        'exchange': 'CME',
        'commission': 0.25,
        'currency': 'USD',
        'contract_size': '$5 × Russell 2000 Index',
        'tick_size': 0.10,
        'tick_value': 0.50,
        'contract_months': ['Mar', 'Jun', 'Sep', 'Dec'],  # Quarterly (H, M, U, Z)
        'months_per_year': 4
    }
}

def get_futures_commission(symbol):
    """Get commission for a futures symbol"""
    if symbol in FUTURES_COMMISSION_MAP:
        return FUTURES_COMMISSION_MAP[symbol]['commission']
    elif symbol in MICRO_FUTURES_MAP:
        return MICRO_FUTURES_MAP[symbol]['commission']
    else:
        return None

def get_futures_details(symbol):
    """Get full details for a futures symbol"""
    if symbol in FUTURES_COMMISSION_MAP:
        return FUTURES_COMMISSION_MAP[symbol]
    elif symbol in MICRO_FUTURES_MAP:
        return MICRO_FUTURES_MAP[symbol]
    else:
        return None

def get_contract_months(symbol):
    """Get contract months for a futures symbol"""
    details = get_futures_details(symbol)
    if details:
        return details.get('contract_months', []), details.get('months_per_year', 0)
    return [], 0

def calculate_total_cost(symbol, contracts, entry_price, exit_price):
    """Calculate total trading cost including commission and slippage"""
    details = get_futures_details(symbol)
    if not details:
        return None
    
    commission_per_contract = details['commission']
    tick_size = details['tick_size']
    tick_value = details['tick_value']
    
    # Calculate P&L
    price_diff = exit_price - entry_price
    ticks = price_diff / tick_size
    pnl = ticks * tick_value * contracts
    
    # Calculate total commission (round trip)
    total_commission = commission_per_contract * contracts * 2  # Entry + Exit
    
    # Net P&L after commission
    net_pnl = pnl - total_commission
    
    return {
        'gross_pnl': pnl,
        'total_commission': total_commission,
        'net_pnl': net_pnl,
        'commission_per_contract': commission_per_contract,
        'round_trip_commission': commission_per_contract * 2
    }

def print_commission_summary():
    """Print a summary of all futures commissions"""
    print("FUTURES COMMISSION SUMMARY (Per Contract)")
    print("=" * 60)
    
    # Group by commission rates
    commission_groups = {}
    all_futures = {**FUTURES_COMMISSION_MAP, **MICRO_FUTURES_MAP}
    
    for symbol, details in all_futures.items():
        rate = details['commission']
        if rate not in commission_groups:
            commission_groups[rate] = []
        commission_groups[rate].append((symbol, details['name']))
    
    for rate in sorted(commission_groups.keys()):
        print(f"\n${rate:.2f} per contract:")
        for symbol, name in commission_groups[rate]:
            print(f"  {symbol:6} - {name}")

def print_contract_months_summary():
    """Print a summary of contract months for all futures"""
    print("\nCONTRACT MONTHS SUMMARY")
    print("=" * 80)
    
    # Group by number of contract months per year
    months_groups = {}
    all_futures = {**FUTURES_COMMISSION_MAP, **MICRO_FUTURES_MAP}
    
    for symbol, details in all_futures.items():
        months_per_year = details.get('months_per_year', 0)
        if months_per_year not in months_groups:
            months_groups[months_per_year] = []
        months_groups[months_per_year].append((symbol, details['name'], details.get('contract_months', [])))
    
    for months_count in sorted(months_groups.keys(), reverse=True):
        print(f"\n{months_count} Contract Months Per Year:")
        print("-" * 40)
        for symbol, name, months in months_groups[months_count]:
            months_str = ', '.join(months)
            print(f"  {symbol:6} - {name}")
            print(f"         Months: {months_str}")

# Example usage
if __name__ == "__main__":
    print_commission_summary()
    print_contract_months_summary()
    
    print("\n" + "=" * 60)
    print("EXAMPLE CALCULATIONS")
    print("=" * 60)
    
    # Example trade calculations
    examples = [
        ('ES', 1, 4200.00, 4210.00),
        ('NQ', 2, 15000.00, 15050.00),
        ('CL', 1, 80.00, 82.50),
        ('GC', 1, 2000.0, 2025.0),
        ('MES', 10, 4200.00, 4210.00)  # Micro contract
    ]
    
    for symbol, contracts, entry, exit in examples:
        result = calculate_total_cost(symbol, contracts, entry, exit)
        if result:
            details = get_futures_details(symbol)
            months, months_per_year = get_contract_months(symbol)
            print(f"\n{symbol} ({details['name']}):")
            print(f"  Contracts: {contracts}")
            print(f"  Entry: ${entry}")
            print(f"  Exit: ${exit}")
            print(f"  Contract Months: {', '.join(months)} ({months_per_year}/year)")
            print(f"  Gross P&L: ${result['gross_pnl']:.2f}")
            print(f"  Commission: ${result['total_commission']:.2f}")
            print(f"  Net P&L: ${result['net_pnl']:.2f}")