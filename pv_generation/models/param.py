from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pvlib

from pv_generation.structure import schema


def get_pv_capital_costs() -> pd.DataFrame:
  """Get the capital costs of a solar pv system in GBP/kWp.
  
  Returns:
      pd.DataFrame: The capital costs of a solar pv system in GBP/kWp.
  """
  economic_data = {
      'Min size (kW)': {
          0: 0,
          1: 4,
          2: 10
      },
      'Max size (kW)': {
          0: 4,
          1: 10,
          2: 50
      },
      'Mean (GBP/kW_installed)': {
          0: 1860.1666666666667,
          1: 1591.6666666666667,
          2: 1133.0
      }
  }

  return pd.DataFrame(economic_data)


@dataclass
class PvPanelParam:
  """Class to store the parameters of a solar pv panel.

  Attributes:
      module_data: pd.DataFrame | None
          The data of the solar pv panel.
      lifetime: float
          The lifetime of the solar pv panel in years.
      a: float
      
      b: float
      
      deltaT: float
      
      height_installation: float
          The height from the ground to the bottom of the panel.
      pv_width: float
          The width of the solar pv panel in meters.
      pv_length: float
          The length of the solar pv panel in meters.
      specific_capital_cost: float
          The cost rate of the solar pv system in GBP/kWp.
      specific_maintenance_cost: float
          The maintenance cost rate of the solar pv system in GBP/kW/year.
          
  Methods:
      size_module: float
          The size of a single module in kWp.
      temperature_model_parameters: dict[str, float]
          The parameters of the temperature model.
      set_default_module_data: None
          Set the default module data.
      get_foot_width_pv_panel: float
          Calculate the foot width of a solar pv based on its tilt angle.
      get_max_number_panels_by_row: float
          Calculate the maximum number of panels in a row.
      get_max_number_rows: int
          Calculate the maximum number of rows.
      get_number_modules: float
          Calculate the number of modules required to meet size of the system.
      update_capital_cost: None
          Update the capital cost of the solar pv system.
  """
  module_data: pd.DataFrame | None = None
  lifetime: float = 20  #year
  a: float = -3.47
  b: float = -0.0594
  deltaT: float = 3
  height_installation: float = 0.3  #height from ground to bottom of panel
  pv_width: float = 0.996  #m
  pv_length: float = 1.68  #m
  specific_capital_cost: float = 1500  #GBP/kWp
  specific_maintenance_cost: float = 0.01 * specific_capital_cost  #GBP/kW/year

  def __post_init__(self):
    """Initialise the scaling factor based on the module data and the size of the system."""
    if self.module_data is None:
      self.set_default_module_data()

  @property
  def area_module(self) -> float:
    """Calculate the area of a single module in m2.

    Returns:
        float: The area of a single module in m2."""
    return self.pv_width * self.pv_length

  @property
  def size_module(self) -> float:
    """Calculate the size of a single module in kWp.

    Returns:
        float: The size of a single module in kWp."""
    kWp_pv = (self.module_data.Impo * self.module_data.Vmpo) / 1000
    print(
        f'The size of a single module is {kWp_pv} kWp or {kWp_pv/self.area_module} kWp/m2'
    )
    return kWp_pv

  @property
  def temperature_model_parameters(self) -> dict[str, float]:
    """Return the parameters of the temperature model.

    Returns:
        dict[str, float]: The parameters of the temperature model."""
    return {'a': self.a, 'b': self.b, 'deltaT': self.deltaT}

  def set_default_module_data(self) -> None:
    """Set the default module data.
    """
    sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
    self.module_data = sandia_modules['Canadian_Solar_CS5P_220M___2009_']

  def get_foot_width_pv_panel(self, tilt: float) -> float:
    """Calculate the foot width of a solar pv based on its tilt angle

    Arguments:
        tilt (float): The tilt angle of the solar pv panel in degrees.
      
    Returns:
        float: The foot width of the solar pv panel in meters."""
    return np.cos(tilt * np.pi / 180) * self.pv_width

  def get_max_number_panels_by_row(self, length_available: float) -> float:
    """Calculate the maximum number of panels in a row.

    Arguments:
        length_available (float): The available length to install the solar pv panels.
    
    Returns:
        float: The maximum number of panels in a row."""
    return length_available // self.pv_length

  def get_max_number_rows(self, distance_btw_rows: float,
                          width_available: float, tilt: int) -> int:
    """Calculate the maximum number of rows of solar pv panels.
    
    Arguments:
        distance_btw_rows (float): The distance between rows of solar pv panels.
        width_available (float): The available width to install the solar pv panels.
        tilt (int): The tilt angle of the solar pv panel in degrees.
    
    Returns:
        int: The maximum number of rows of solar pv panels.
    """
    nb_rows = (width_available + distance_btw_rows) / (
        distance_btw_rows + self.get_foot_width_pv_panel(tilt))
    return round(nb_rows, 0)

  def get_number_modules(self, size_system: float) -> float:
    """Calculate the number of modules required to meet size of the system
    
    Arguments:
        size_system (float): The size of the solar pv system in kWp.
      
    Returns:
        float: The number of modules required to meet size of the system."""
    return size_system / self.size_module

  def update_capital_cost(self, size_system: float) -> None:
    """Update the capital cost of the solar pv system.

    Arguments:
        size_system (float): The size of the solar pv system in kWp.
    """

    capital_dataf = get_pv_capital_costs()
    filt = (
        capital_dataf[schema.PVEconomicDataSchema.MIN_SIZE] <= size_system) & (
            size_system < capital_dataf[schema.PVEconomicDataSchema.MAX_SIZE])
    if any(x == True for x in filt):
      self.specific_capital_cost = capital_dataf.loc[
          filt, schema.PVEconomicDataSchema.MEAN].values[0]
    else:
      print(
          "No economic data provided. The capital cost is set to the default value."
      )
    return None
