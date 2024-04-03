class SolPosSchema:
  AZIMUTH = 'azimuth'
  APPARENT_ZENITH = 'apparent_zenith'
  ZENITH = 'zenith'
  ELEVATION = 'elevation'
  APPARENT_EVELATION = 'apparent_elevation'
  EQUATION_OF_TIME = 'equation_of_time'


class PVEconomicDataSchema:
  MIN_SIZE = 'Min size (kW)'
  MAX_SIZE = 'Max size (kW)'
  MEAN = 'Mean (GBP/kW_installed)'


class PVSystemDataSchema:
  OUTPUT = 'PV_electricity_generation', 'kWh'