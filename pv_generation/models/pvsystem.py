from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import pvlib
from e2slib.analysis import location
from e2slib.common import common
from e2slib.structures import enums as e2s_enums
from e2slib.structures import site_schema
from pvlib.modelchain import ModelChain
from pvlib.pvsystem import PVSystem
from weather.api import functions
from weather.structure import enums as weather_enums

from pv_generation.models import param
from pv_generation.structure import schema


class WeatherDataGenerator(Protocol):

  def get_weather_data(self) -> pd.DataFrame:
    ...

  def get_weather_data_source(self) -> weather_enums.WeatherDataSource:
    ...


@dataclass
class PvInstallation:
  """Class to store the parameters of a solar pv installation and estimate its electricity generation.

  Attributes:
      pv_panel_param (param.PvPanelParam): The parameters of the solar pv panel.
      weather_data_gen (WeatherDataGenerator): The weather data generator to use.
      geolocation (location.GeoLocation): The geolocation of the installation.
      name (str): The name of the installation.
      distance_btw_rows (float): The distance between rows of panels in meters.
      width_available (float): The width available for the installation in meters.
      length_available (float): The length available for the installation in meters.
      tilt (int): The tilt of the panels in degrees.
      azimuth (int): The azimuth of the panels in degrees.
      inverter_data (pd.DataFrame | None): The data of the inverter.
      system_losses (float): The system losses in percentage.
      _size_system (float): The size of the system in kW peak.
      model_chain (ModelChain | None): The model chain to use.
      scaling_factor (float): The scaling factor to use.
      technology_type (e2s_enums.TechnologyType): The technology type of the installation.
      results (pd.DataFrame): The results of the installation.

  Methods:
      lifetime(): The lifetime of the installation.
      capital_cost(): The capital cost of the installation.
      annual_maintenance_cost(): The annual maintenance cost of the installation.
      size_system(): The size of the system.
      size_system.setter(): Set the size of the system.
      additional_demand(): The additional demand of the installation.
      onsite_generation(): The onsite generation of the installation.
      get_assets(): Get the assets of the installation.
      export_results(): Export the results of the installation.
      set_default_inverter_data(): Set the default inverter data.
      get_max_number_panels(): Get the maximum number of panels that can be installed.
      get_max_capacity_installed_on_flat_surface(): Get the maximum capacity that can be installed on a flat surface.
      create_pv_system(): Create the pv system.
      get_pvlib_location(): Get the pvlib location.
      run_model(): Run the model.
      plot_solar_elevation_against_azimuth(): Plot the solar elevation against azimuth.
      get_solar_position(): Get the solar position.
      compare_pv_weather_data_outputs(): Compare the results from the weather source with results from TMY and clearsky.
      size_pv_system_for_no_export(): Size the pv system for no export.
      calculate_export_demand(): Calculate the export demand.
  """
  pv_panel_param: param.PvPanelParam
  weather_data_gen: WeatherDataGenerator
  geolocation: location.GeoLocation
  name: str = 'PV_system'
  distance_btw_rows: float = 0  #m
  width_available: float = 1  #m
  length_available: float = 1  #m
  tilt: int = 10  #degree
  azimuth: int = 180  #degree
  inverter_data: pd.DataFrame | None = None
  system_losses: float = 0.  #default system_losses of 14%
  _size_system: float = 10  #kW peak
  model_chain: ModelChain | None = None
  scaling_factor: float = 1
  technology_type: e2s_enums.TechnologyType = e2s_enums.TechnologyType.PV
  results: pd.DataFrame = field(default_factory=pd.DataFrame)

  def __post_init__(self):
    if self.inverter_data is None:
      self.set_default_inverter_data()
    self.scaling_factor = self.pv_panel_param.get_number_modules(
        sum(self.size_system.values()))
    self.pv_panel_param.update_capital_cost(sum(self.size_system.values()))
    self.create_pv_system()

  @property
  def lifetime(self) -> float:
    """The lifetime of the installation.
    
    Returns:
        float: The lifetime of the installation."""
    return self.pv_panel_param.lifetime

  @property
  def capital_cost(self) -> dict[e2s_enums.TechnologyType, float]:
    """The capital cost of the installation.
    
    Returns:
        dict[e2s_enums.TechnologyType, float]: The capital cost of the installation."""
    return {
        self.technology_type:
        self.pv_panel_param.specific_capital_cost *
        sum(self.size_system.values())
    }

  @property
  def annual_maintenance_cost(self) -> dict[e2s_enums.TechnologyType, float]:
    """The annual maintenance cost of the installation.

    Returns:
        dict[e2s_enums.TechnologyType, float]: The annual maintenance cost of the installation."""
    return {
        self.technology_type:
        self.pv_panel_param.specific_maintenance_cost *
        sum(self.size_system.values())
    }

  @property
  def size_system(self) -> dict[e2s_enums.TechnologyType, float]:
    """The size of the system.

    Returns:
        dict[e2s_enums.TechnologyType, float]: The size of the system."""
    return {self.technology_type: self._size_system}

  @size_system.setter
  def size_system(self, size_system: float) -> None:
    """Set the size of the system.

    Arguments:
        size_system (float): The size of the system.
    """

    self._size_system = size_system
    self.scaling_factor = self.pv_panel_param.get_number_modules(size_system)
    self.pv_panel_param.update_capital_cost(sum(self.size_system.values()))
    self.create_pv_system()

  @property
  def additional_demand(self) -> pd.DataFrame:
    """The additional demand of the installation.

    Returns:
        pd.DataFrame: The additional demand of the installation."""
    columns = common.get_multiindex_single_column(
        site_schema.SiteDataSchema.ADDITIONAL_ELECTRICITY_DEMAND)
    additional_demand = pd.DataFrame(index=self.results.index, columns=columns)
    additional_demand.iloc[:, 0] = 0.
    return additional_demand

  @property
  def onsite_generation(self) -> pd.DataFrame:
    """The onsite generation of the installation.
    
    Returns:
        pd.DataFrame: The onsite generation of the installation."""
    return self.results

  def get_assets(self) -> dict[str, PvInstallation]:
    """Get the assets of the installation.
    
    Returns:
        dict[str, PvInstallation]: The assets of the installation."""
    return {self.name: self}

  def export_results(self):
    """Export the results of the installation.
    
    Returns:
        pd.DataFrame: The results of the installation."""
    additional_demand = self.additional_demand
    on_site_gen = self.results
    return pd.concat([additional_demand, on_site_gen], axis=1).astype(float)

  def set_default_inverter_data(self) -> None:
    """Set the default inverter data.
    
    """
    sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
    self.inverter_data = sapm_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']

  def get_max_number_panels(self):
    """Calculate the number of panels that can be installed.
    
    Returns:
        float: The number of panels that can be installed."""
    return self.pv_panel_param.get_max_number_panels_by_row(
        self.length_available) * self.pv_panel_param.get_max_number_rows(
            self.distance_btw_rows, self.width_available, self.tilt)

  def get_max_capacity_installed_on_flat_surface(self):
    """Calculate the number of panels that can be installed on a ground flat surface.
    
    Returns:
        float: The number of panels that can be installed on a ground flat surface."""
    return self.pv_panel_param.get_max_number_panels_by_row(
        self.length_available) * self.pv_panel_param.get_max_number_rows(
            self.distance_btw_rows, self.width_available,
            self.tilt) * self.pv_panel_param.size_module  #in kWp

  def create_pv_system(self) -> None:
    """Create the pv system and appends it to the `self.model_chain` attribute.
    """
    # https://pvlib-python.readthedocs.io/en/stable/_modules/pvlib/temperature.html
    module_data = self.pv_panel_param.module_data
    temperature_model_parameters = self.pv_panel_param.temperature_model_parameters
    system = PVSystem(
        surface_tilt=self.tilt,
        surface_azimuth=self.azimuth,
        module_parameters=module_data,
        inverter_parameters=self.inverter_data,
        temperature_model_parameters=temperature_model_parameters)
    pvlib_location = self.get_pvlib_location()
    self.model_chain = ModelChain(system, pvlib_location, aoi_model='ashrae')

  def get_pvlib_location(self) -> pvlib.location.Location:
    """Get the pvlib location.

    Returns:
        pvlib.location.Location: The pvlib location object."""
    return pvlib.location.Location(
        self.geolocation.latitude,
        self.geolocation.longitude,
        name=self.name,
        altitude=int(self.geolocation.altitude),
        tz=self.geolocation.timezone,
    )

  def run_model(self, weather_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Run the pv system model.

    Arguments:
        weather_df (pd.DataFrame | None): The weather data to use.
    
    Returns:
        pd.DataFrame: The results of the model."""

    if weather_df is None:
      weather_df = self.weather_data_gen.get_weather_data()
    weather_data_format = functions.get_weather_data_format(
        list(weather_df.columns))
    if weather_data_format is weather_enums.WeatherDataFormat.POA:
      self.model_chain.run_model_from_poa(weather_df)
    else:
      self.model_chain.run_model(weather_df)
    energy_outputs = self.model_chain.results.ac.to_frame(
    ) * self.scaling_factor * (1 - self.system_losses) / 1000
    energy_outputs = energy_outputs / 2  # from kW per 30min to kWh
    energy_outputs.columns = common.get_multiindex_single_column(
        site_schema.SiteDataSchema.ELECTRICITY_GENERATION)
    energy_outputs = energy_outputs.astype(float)
    self.results = energy_outputs
    return self.results

  def plot_solar_elevation_against_azimuth(self):
    """Plot the solar elevation against azimuth.

    Returns:
        plt.Figure: The figure of the ploplt.Axes: The axes of the plot."""
    # https://pvlib-python.readthedocs.io/en/v0.7.1/auto_examples/plot_sunpath_diagrams.html
    solpos = self.get_solar_position()
    fig, ax = plt.subplots()
    solpos_index: pd.DatetimeIndex = solpos.index
    points = ax.scatter(solpos[schema.SolPosSchema.AZIMUTH].values,
                        solpos[schema.SolPosSchema.APPARENT_EVELATION].values,
                        s=2,
                        c=solpos_index.dayofyear,
                        label=None)
    fig.colorbar(points)

    for hour in np.unique(solpos_index.hour):
      # choose label position by the largest elevation for each hour
      subset = solpos.loc[solpos_index.hour == hour, :]
      height = subset[schema.SolPosSchema.APPARENT_EVELATION]
      pos = solpos.loc[height.idxmax(), :]
      ax.text(pos[schema.SolPosSchema.AZIMUTH],
              pos[schema.SolPosSchema.APPARENT_EVELATION], str(hour))

    list_dates: pd.DatetimeIndex = pd.to_datetime(
        ['2020-03-21', '2020-06-21', '2020-12-21'], format="%Y-%m-%d")
    for date in list_dates:
      times = pd.date_range(start=date,
                            end=date + pd.Timedelta('24h'),
                            freq='5min',
                            tz=self.geolocation.timezone)
      solpos = pvlib.solarposition.get_solarposition(
          times, self.geolocation.latitude, self.geolocation.longitude)
      solpos = solpos.loc[
          solpos[schema.SolPosSchema.APPARENT_EVELATION] > 0, :]
      label = date.strftime('%Y-%m-%d')
      ax.plot(solpos[schema.SolPosSchema.AZIMUTH],
              solpos[schema.SolPosSchema.APPARENT_EVELATION],
              label=label)

    ax.figure.legend(loc='upper left')
    ax.set_xlabel('Solar Azimuth (degrees)')
    ax.set_ylabel('Solar Elevation (degrees)')
    plt.close()
    return fig, ax

  def get_solar_position(self) -> pd.DataFrame:
    """Get the solar position.

    Returns:
        pd.DataFrame: The solar position."""
    times = pd.date_range('2020-01-01 00:00:00',
                          '2021-01-01',
                          freq='H',
                          tz=self.geolocation.timezone)
    solpos = pvlib.solarposition.get_solarposition(times,
                                                   self.geolocation.latitude,
                                                   self.geolocation.longitude)
    solpos = solpos.loc[solpos['apparent_elevation'] > 0, :]
    return solpos

  def compare_pv_weather_data_outputs(self) -> pd.DataFrame:
    """Compare the results from the org_weather_source with results from TMY and clearsky
    
    Returns:
        pd.DataFrame: The results of the comparison."""
    org_weather_data = self.weather_data_gen.get_weather_data(
    )  #Default weather source
    tmy_weather_data = self.weather_data_gen.get_weather_data(
        weather_enums.WeatherDataSource.TMY)
    clearsky_weather_data = self.weather_data_gen.get_weather_data(
        weather_enums.WeatherDataSource.PVLIB_CLEARSKY)
    results: dict[str, dict[weather_enums.WeatherDataSource, float]] = {}

    sub_results: dict[weather_enums.WeatherDataSource, float] = {}
    temp_results = self.run_model(org_weather_data)
    sub_results[self.weather_data_gen.get_weather_data_source()] = float(
        temp_results.sum()[0])
    temp_results = self.run_model(tmy_weather_data)
    sub_results[weather_enums.WeatherDataSource.TMY] = float(
        temp_results.sum()[0])
    temp_results = self.run_model(clearsky_weather_data)
    sub_results[weather_enums.WeatherDataSource.PVLIB_CLEARSKY] = float(
        temp_results.sum()[0])
    results[self.name] = sub_results
    return pd.DataFrame(results)

  def size_pv_system_for_no_export(
      self,
      site_load_demand: pd.DataFrame,
      weather_dataf: pd.DataFrame | None = None) -> float:
    """Size the pv system for no export.

    Arguments:
        site_load_demand (pd.DataFrame): The site load demand.
        weather_dataf (pd.DataFrame | None): The weather data to use.

    Returns:
        float: The size of the pv system for no export."""

    if weather_dataf is None:
      weather_dataf = self.weather_data_gen.get_weather_data()
    weather_dataf = weather_dataf.loc[site_load_demand.index]
    energy_gen = self.run_model(weather_dataf)  # in kWh
    energy_gen = energy_gen * 2 / sum(self.size_system.values(
    ))  #specific energy generation from power plant kW_output/kW_installed
    filt = energy_gen.iloc[:, 0].values < 0
    energy_gen.loc[filt] = 0
    dataf = pd.concat([site_load_demand, energy_gen], axis=1)
    new_col = 'Site demand/specific pv gen'
    dataf[new_col] = dataf.iloc[:, 0].div(dataf.iloc[:, 1].values, axis=0)
    dataf[~np.isfinite(dataf)] = np.nan
    dataf.fillna(0, inplace=True)
    filt = dataf.loc[:, new_col].values > 0
    return dataf.loc[filt, new_col].sort_values()[0]

  def calculate_export_demand(self, org_import_elec: npt.NDArray[np.float64]):
    """Calculate the export demand.

    Arguments:
        org_import_elec (npt.NDArray[np.float64]): The original imported electricity.

    Returns:
        npt.NDArray[np.float64]: The exported electricity."""
    elec_gen = self.results.values.flatten()
    export_elec = elec_gen - org_import_elec
    return np.where(export_elec < 0, 0, export_elec)
