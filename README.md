pv-generation
==============================

Documentation available at: https://stunning-adventure-qk6zqe2.pages.github.io/

The PV-Generation tool can be used to model the potential solar generation ability of a site. The tool uses solar panel, weather and location data to estimate the potential energy generation of a commercial site. There is a `Demo_notebook.ipynb` that users can run to get a basic understanding of how the tool works and see various functionalities.

Data is provided in the `data` folder with both example site energy consumption data, weather data examples from the weather tool and capital cost data to do with the solar pv panels.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── demo                                                        
    │   │   └── demo_elec_data.csv                                      <- Demonstration site electricity data for cost/consumption/carbon estimations
    │   └── pv_analysis                                                 
    │       ├── economic_data                                           
    │       │   └── capital_costs_pv_2022.csv                           <- PV cost table for 2022
    │       └── weather_data            
    │           ├── Demo_site_netcdf_data                               <- 
    │           │   ├── Demo_site_2m_temperature_2020.nc                <- 
    │           │   └── Demo_site_surface_net_solar_radiation_2020.nc   <- 
    │           ├── Weather_data_era5_2020.csv                          <- Weather data from era5 API
    │           ├── Weather_data_pvgissarah2_2020.csv                   <- Weather data from Sarah2 API
    │           └── Weather_data_pvlib_clearsky_2020.csv                <- Weather data from Clearsky API
    │
    ├── notebooks          <- Jupyter notebooks
    │   └── Demo_notebook.ipynb       <- Data from third party sources
    │
    ├── pv_generation                <- Source code for use in this project
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── models           <- Scripts to download or generate data
    │   │   ├── param.py       <- Script for creating a pv panel object
    │   │   └── pvsystem.py       <- Script for generating a PV System object
    │   │
    │   ├── structure       <- Scripts used for structuring, naming and typing in project
    │   │   ├── enums.py       <- Enums script for pv_generation
    │   │   └── schema.py       <- Schema script for pv_generation
    │   │
    │   ├── utils
    │   │   ├── config.py       <- Script for the path to the economic data [Might be able to delete]
    │   │   └── functions.py       <- Script containing functions to create objects necessary for modeling
    │   │
    │   └── main.py  <- Main script for running a demo pv system
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    └── requirements.txt   <- The requirements file for reproducing the analysis environment.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
