from pathlib import Path

import pandas as pd
from weather import enums as weather_enums

from pv_generation.models.param import PvPanelParam
from pv_generation.models.pvsystem import PvInstallation
from pv_generation.utils import functions as pv_functions


def create_demo_pv_installation() -> PvInstallation:
  """Create a demo PV installation with a size of 250 kWp.
  
  Returns:
      PvInstallation: A PV installation with a size of 250 kWp."""
  path_analysis_results = Path.cwd().parent / 'data/pv_analysis'
  simulation_year = 2020
  pv_param = PvPanelParam()
  demo_location = pv_functions.get_location_obj("Demo_site", 53.514, -1.143,
                                                114.6, 'UTC')
  demo_weather = pv_functions.get_weatherdata_obj(
      demo_location, simulation_year, weather_enums.WeatherDataSource.ERA5,
      path_analysis_results)

  demo_pv_installation = PvInstallation(pv_param,
                                        weather_data_gen=demo_weather,
                                        geolocation=demo_location)
  demo_pv_installation.size_system = 250
  return demo_pv_installation


def main() -> pd.DataFrame:
  """Run the demo PV installation and return the results.

  Returns:
      pd.DataFrame: The results of the demo PV installation"""
  demo_pv_installation = create_demo_pv_installation()
  demo_pv_installation.run_model()
  return demo_pv_installation.export_results()


if __name__ == '__main__':
  results = main()
  print(results)
