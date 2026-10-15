"""Registry of real-world API sources that make up the StreamArena v2 catalog."""

from .binance import BinanceKlines
from .bts_flights import FlightDelays
from .energy import CarbonIntensityGB, EnergyChartsLoad, OctopusAgile
from .mobility import CitiBikeDemand, CitiBikeRiderType, TaxiDemandNYC
from .noaa_coops import NOAATides
from .open_meteo import OpenMeteoAirQuality, OpenMeteoWeather
from .space_weather import GeomagneticStorm
from .usgs_quake import EarthquakeSignificance
from .usgs_water import USGSWater
from .wikimedia import WikipediaReverts


def catalog() -> dict:
    """name -> Source instance (26 real-world datasets)."""
    srcs = [
        # regression (14)
        *(OpenMeteoWeather(c) for c in ("auckland", "berlin", "phoenix")),
        OpenMeteoWeather("hamburg", target="wind_speed_10m"),
        OpenMeteoAirQuality("delhi"), OpenMeteoAirQuality("losangeles"),
        USGSWater(), NOAATides(), CarbonIntensityGB(), EnergyChartsLoad(), OctopusAgile(),
        CitiBikeDemand(), TaxiDemandNYC(), BinanceKlines(),
        # classification (12)
        *(OpenMeteoWeather(c, classify_rain=True) for c in ("auckland", "berlin", "phoenix")),
        OpenMeteoAirQuality("delhi", classify_aqi=True),
        CarbonIntensityGB(classify_index=True), GeomagneticStorm(), USGSWater(flood=True),
        EarthquakeSignificance(), FlightDelays(), CitiBikeRiderType(), WikipediaReverts(), BinanceKlines(direction=True),
    ]
    return {s.name: s for s in srcs}


#: sources whose archives are multi-GB; build these on the server, not a laptop
HEAVY = {"bike_demand_nyc", "bike_rider_type_nyc", "taxi_demand_nyc", "flight_delay_bts", "wiki_revert_simple"}
