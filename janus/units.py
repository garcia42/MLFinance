from typing import Final
from enum import Enum

class ContractType(Enum):
    ENERGY = "Energy"
    METALS = "Metals"
    AGRICULTURE = "Agriculture"
    FINANCIAL = "Financial"

class ContractUnit:
    def __init__(self, size: int, unit: str, type: ContractType):
        self.size: Final = size
        self.unit: Final = unit
        self.type: Final = type

# Contract specifications
NATURAL_GAS: Final = ContractUnit(10000, "MMBtu", ContractType.ENERGY)
GOLD: Final = ContractUnit(100, "troy oz", ContractType.METALS)
PLATINUM: Final = ContractUnit(50, "troy oz", ContractType.METALS)
COTTON: Final = ContractUnit(50000, "pounds", ContractType.AGRICULTURE)
COFFEE: Final = ContractUnit(37500, "pounds", ContractType.AGRICULTURE)
SOYBEAN: Final = ContractUnit(5000, "bushels", ContractType.AGRICULTURE)
FIVE_YEAR_NOTE: Final = ContractUnit(100000, "face value", ContractType.FINANCIAL)
TEN_YEAR_NOTE: Final = ContractUnit(100000, "face value", ContractType.FINANCIAL)

# Mapping of filenames to contract specifications
CONTRACT_UNITS: Final = {
    'Natural_Gas_data.csv': NATURAL_GAS.size,
    'Gold_data.csv': GOLD.size,
    'Platinum_data.csv': PLATINUM.size,
    'Cotton.csv': COTTON.size,
    'Coffee.csv': COFFEE.size,
    'Soybean_data.csv': SOYBEAN.size,
    'US 5 Year T-Note Futures Historical Data.csv': FIVE_YEAR_NOTE.size,
    'US 10 Year T-Note Futures Historical Data.csv': TEN_YEAR_NOTE.size,
}
