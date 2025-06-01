from typing import Final
from enum import Enum

class ContractType(Enum):
    ENERGY = "Energy"
    METALS = "Metals"
    AGRICULTURE = "Agriculture"
    FINANCIAL = "Financial"

class ContractUnit:
    def __init__(self, size: int, unit: str, type: ContractType, symbol: str):
        self.size: Final = size
        self.unit: Final = unit
        self.type: Final = type
        self.symbol: Final = symbol

# Contract specifications
NATURAL_GAS: Final = ContractUnit(10000, "MMBtu", ContractType.ENERGY, "NG")
GOLD: Final = ContractUnit(100, "troy oz", ContractType.METALS, "GC")
PLATINUM: Final = ContractUnit(50, "troy oz", ContractType.METALS, "PL")
COTTON: Final = ContractUnit(50000, "pounds", ContractType.AGRICULTURE, "CT")
COFFEE: Final = ContractUnit(37500, "pounds", ContractType.AGRICULTURE, "KC")
SOYBEAN: Final = ContractUnit(5000, "bushels", ContractType.AGRICULTURE, "ZS")
FIVE_YEAR_NOTE: Final = ContractUnit(100000, "face value", ContractType.FINANCIAL, "ZF")
TEN_YEAR_NOTE: Final = ContractUnit(100000, "face value", ContractType.FINANCIAL, "ZN")

BRENT_CRUDE: Final = ContractUnit(1000, "barrels", ContractType.ENERGY, "BZ")
CRUDE_OIL: Final = ContractUnit(1000, "barrels", ContractType.ENERGY, "CL")
HEATING_OIL: Final = ContractUnit(42000, "gallons", ContractType.ENERGY, "HO")
RBOB_GASOLINE: Final = ContractUnit(42000, "gallons", ContractType.ENERGY, "RB")
COPPER: Final = ContractUnit(25000, "pounds", ContractType.METALS, "HG")
PALLADIUM: Final = ContractUnit(100, "troy oz", ContractType.METALS, "PA")
SILVER: Final = ContractUnit(5000, "troy oz", ContractType.METALS, "SI")
CORN: Final = ContractUnit(5000, "bushels", ContractType.AGRICULTURE, "ZC")
FEEDER_CATTLE: Final = ContractUnit(50000, "pounds", ContractType.AGRICULTURE, "GF")
KC_WHEAT: Final = ContractUnit(5000, "bushels", ContractType.AGRICULTURE, "KE")
LEAN_HOGS: Final = ContractUnit(40000, "pounds", ContractType.AGRICULTURE, "HE")
LIVE_CATTLE: Final = ContractUnit(40000, "pounds", ContractType.AGRICULTURE, "LE")
OATS: Final = ContractUnit(5000, "bushels", ContractType.AGRICULTURE, "ZO")
ROUGH_RICE: Final = ContractUnit(2000, "hundredweight", ContractType.AGRICULTURE, "ZR")
SOYBEAN_OIL: Final = ContractUnit(60000, "pounds", ContractType.AGRICULTURE, "ZL")
SUGAR: Final = ContractUnit(112000, "pounds", ContractType.AGRICULTURE, "SB")
THIRTY_YEAR_BOND: Final = ContractUnit(100000, "face value", ContractType.FINANCIAL, "ZB")

# Mapping of filenames to contract specifications
CONTRACTS: Final = {
    'Natural_Gas_data.csv': NATURAL_GAS,
    'Gold_data.csv': GOLD,
    'Platinum_data.csv': PLATINUM,
    'Cotton.csv': COTTON,
    'Coffee.csv': COFFEE,
    'Soybean_data.csv': SOYBEAN,
    'US 5 Year T-Note Futures Historical Data.csv': FIVE_YEAR_NOTE,
    'US 10 Year T-Note Futures Historical Data.csv': TEN_YEAR_NOTE,
    'Brent_Crude_Oil_data.csv': BRENT_CRUDE,
    'Copper_data.csv': COPPER,
    'Corn_data.csv': CORN,
    'Crude_Oil_data.csv': CRUDE_OIL,
    'Feeder Cattle.csv': FEEDER_CATTLE,
    'Heating_Oil_data.csv': HEATING_OIL,
    'KC_HRW_Wheat_data.csv': KC_WHEAT,
    'Lean Hogs.csv': LEAN_HOGS,
    'Live Cattle.csv': LIVE_CATTLE,
    'Oat_data.csv': OATS,
    'Palladium_data.csv': PALLADIUM,
    'RBOB_Gasoline_data.csv': RBOB_GASOLINE,
    'Rough_Rice_data.csv': ROUGH_RICE,
    'Silver_data.csv': SILVER,
    'Soybean_Oil_data.csv': SOYBEAN_OIL,
    'Sugar.csv': SUGAR,
    'US 30 Year T-Bond Futures Historical Data.csv': THIRTY_YEAR_BOND
}

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
    'Brent_Crude_Oil_data.csv': BRENT_CRUDE.size,
    'Copper_data.csv': COPPER.size,
    'Corn_data.csv': CORN.size,
    'Crude_Oil_data.csv': CRUDE_OIL.size,
    'Feeder Cattle.csv': FEEDER_CATTLE.size,
    'Heating_Oil_data.csv': HEATING_OIL.size,
    'KC_HRW_Wheat_data.csv': KC_WHEAT.size,
    'Lean Hogs.csv': LEAN_HOGS.size,
    'Live Cattle.csv': LIVE_CATTLE.size,
    'Oat_data.csv': OATS.size,
    'Palladium_data.csv': PALLADIUM.size,
    'RBOB_Gasoline_data.csv': RBOB_GASOLINE.size,
    'Rough_Rice_data.csv': ROUGH_RICE.size,
    'Silver_data.csv': SILVER.size,
    'Soybean_Oil_data.csv': SOYBEAN_OIL.size,
    'Sugar.csv': SUGAR.size,
    'US 30 Year T-Bond Futures Historical Data.csv': THIRTY_YEAR_BOND.size
}

# Mapping of contracts to their trading symbols
TRADING_SYMBOLS: Final = {
    contract.symbol: contract for contract in [
        NATURAL_GAS, GOLD, PLATINUM, COTTON, COFFEE, SOYBEAN,
        FIVE_YEAR_NOTE, TEN_YEAR_NOTE, BRENT_CRUDE, CRUDE_OIL,
        HEATING_OIL, RBOB_GASOLINE, COPPER, PALLADIUM, SILVER,
        CORN, FEEDER_CATTLE, KC_WHEAT, LEAN_HOGS, LIVE_CATTLE,
        OATS, ROUGH_RICE, SOYBEAN_OIL, SUGAR, THIRTY_YEAR_BOND
    ]
}
