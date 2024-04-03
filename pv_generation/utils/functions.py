from pathlib import Path

import folium
from e2slib.analysis import location
from weather import WeatherData
from weather import enums as weather_enums


def get_location_obj(name: str, latitude: float, longitude: float,
                     altitude: float, timezone: str) -> location.GeoLocation:
  """ Create a location object with the given parameters 
  
  Arguments:
      name (str): The name of the location
      latitude (float): The latitude of the location
      longitude (float): The longitude of the location
      altitude (float): The altitude of the location
      timezone (str): The timezone of the location
      
  Returns:
      location.GeoLocation: A location object with the given parameters
  """
  return location.GeoLocation(name=name,
                              latitude=latitude,
                              longitude=longitude,
                              altitude=altitude,
                              timezone=timezone)


def get_weatherdata_obj(geolocation: location.GeoLocation, year: int,
                        weather_source: weather_enums.WeatherDataSource,
                        saving_path: Path) -> WeatherData:
  """ Create a weather data object with the given parameters 
  
  Arguments:
      geolocation (location.GeoLocation): The location of the weather data
      year (int): The year of the weather data
      weather_source (weather_enums.WeatherDataSource): The source of the weather data
      saving_path (Path): The path to save the weather data
    
  Returns:
      WeatherData: A weather data object with the given parameters"""
  return WeatherData(geolocation=geolocation,
                     simulation_year=year,
                     weather_data_source=weather_source,
                     saving_path=saving_path)


def create_map(geoloc: location.GeoLocation) -> folium.Map:
  """ Create a map centered at the given latitude and longitude 
  
  Arguments:
      geoloc (location.GeoLocation): The location to center the map on
  
  Returns:
      folium.Map: A map centered at the given location"""
  latitude = geoloc.latitude
  longitude = geoloc.longitude
  m = folium.Map(location=[latitude, longitude], zoom_start=13)
  # add a marker at the given location
  folium.Marker([latitude, longitude]).add_to(m)
  return m


def create_pv_saving_path(saving_path: Path | None = None) -> Path:
  """ Create a directory to save the results of the pv analysis 
  
  Arguments:
      saving_path (Path | None): The path to save the results of the pv analysis
  
  Returns:
      Path: The path to save the results of the pv analysis"""
  if saving_path is None:
    path_analysis = Path(r'../pv_analysis')
  else:
    path_analysis = saving_path / 'pv_analysis'
  path_analysis.mkdir(parents=True, exist_ok=True)
  return path_analysis
