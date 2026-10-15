# StreamArena — Dataset Tables

Counts are computed directly from each built CSV. Real datasets are capped at 1M rows by uniform
thinning; `h` is the prediction horizon in steps of the stream's native cadence. Every file has a leading
`timestamp` column (not a feature) and a trailing `target`. Sources are cited in `paper/references.bib`.

## Classification

### Real-world (API archives)

| Dataset | #Instances | #Features | #Classes | Span | Source |
|---|---:|---:|---:|---|---|
| airquality_aqi_delhi | 36,449 | 25 | 6 | 2022-08-04 → 2026-09-30 | Open-Meteo air-quality API (CAMS global/European reanalysis), CC BY 4.0 |
| crypto_direction | 918,533 | 21 | 2 | 2018-01-01 → 2026-09-30 | Binance public market data archive (data.binance.vision) |
| flood_potomac | 371,905 | 24 | 2 | 2016-01-01 → 2026-10-01 | USGS NWIS Instantaneous Values web service, public domain |
| grid_carbon_index_gb | 142,779 | 24 | 5 | 2018-05-11 → 2026-09-30 | NESO Carbon Intensity API, CC BY 4.0 |
| quake_significant | 452,947 | 19 | 2 | 2010-01-01 → 2026-09-30 | USGS Earthquake Hazards Program ComCat, FDSN event service, public domain |
| space_storm_kp | 468,949 | 20 | 2 | 2000-01-01 → 2026-10-01 | GFZ German Research Centre for Geosciences, Kp/Hp30 indices, CC BY 4.0 |
| weather_rain_auckland | 94,218 | 32 | 2 | 2016-01-01 → 2026-09-30 | Open-Meteo historical weather API (ERA5), Zippenfenig 2023, CC BY 4.0 |
| weather_rain_berlin | 94,218 | 32 | 2 | 2016-01-01 → 2026-09-30 | Open-Meteo historical weather API (ERA5), Zippenfenig 2023, CC BY 4.0 |
| weather_rain_phoenix | 94,218 | 32 | 2 | 2016-01-01 → 2026-09-30 | Open-Meteo historical weather API (ERA5), Zippenfenig 2023, CC BY 4.0 |

### Synthetic

| Dataset | #Instances | #Features | #Classes | Drift | Generator |
|---|---:|---:|---:|---|---|
| agrawal_abrupt | 500,000 | 9 | 2 | abrupt @ 250000 | Agrawal |
| agrawal_gradual | 500,000 | 9 | 2 | gradual @ 250000 | Agrawal |
| hyperplane_abrupt | 500,000 | 10 | 2 | abrupt @ 250000 | Hyperplane |
| hyperplane_gradual | 500,000 | 10 | 2 | gradual @ 250000 | Hyperplane |
| led_abrupt | 500,000 | 24 | 10 | abrupt @ 250000 | LED |
| led_gradual | 500,000 | 24 | 10 | gradual @ 250000 | LED |
| mixed_abrupt | 500,000 | 4 | 2 | abrupt @ 250000 | Mixed |
| mixed_gradual | 500,000 | 4 | 2 | gradual @ 250000 | Mixed |
| randomtree_abrupt | 500,000 | 10 | 3 | abrupt @ 250000 | RandomTree |
| randomtree_gradual | 500,000 | 10 | 3 | gradual @ 250000 | RandomTree |
| rbf_abrupt | 500,000 | 10 | 4 | abrupt @ 250000 | RandomRBF |
| rbf_gradual | 500,000 | 10 | 4 | gradual @ 250000 | RandomRBF |
| sea_abrupt | 500,000 | 3 | 2 | abrupt @ 250000 | SEA |
| sea_gradual | 500,000 | 3 | 2 | gradual @ 250000 | SEA |
| stagger_abrupt | 500,000 | 3 | 2 | abrupt @ 250000 | STAGGER |
| stagger_gradual | 500,000 | 3 | 2 | gradual @ 250000 | STAGGER |
| waveform_abrupt | 500,000 | 40 | 3 | abrupt @ 250000 | Waveform |
| waveform_gradual | 500,000 | 40 | 3 | gradual @ 250000 | Waveform |

## Regression

### Real-world (API archives)

| Dataset | #Instances | #Features | Horizon | Span | Source |
|---|---:|---:|---:|---|---|
| airquality_pm25_delhi | 36,449 | 25 | h=1 | 2022-08-04 → 2026-09-30 | Open-Meteo air-quality API (CAMS global/European reanalysis), CC BY 4.0 |
| airquality_pm25_losangeles | 36,449 | 25 | h=1 | 2022-08-04 → 2026-09-30 | Open-Meteo air-quality API (CAMS global/European reanalysis), CC BY 4.0 |
| crypto_btc_1m | 918,533 | 21 | h=1 | 2018-01-01 → 2026-09-30 | Binance public market data archive (data.binance.vision) |
| energy_price_agile | 150,816 | 55 | h=48 | 2018-02-22 → 2026-09-29 | Octopus Energy public tariff API (Agile Octopus, region C London) |
| grid_carbon_gb | 142,779 | 24 | h=1 | 2018-05-11 → 2026-09-30 | NESO Carbon Intensity API, CC BY 4.0 |
| grid_load_de | 376,880 | 25 | h=4 | 2016-01-01 → 2026-09-30 | Energy-Charts (Fraunhofer ISE) public_power API, CC BY 4.0 |
| river_discharge_potomac | 371,901 | 24 | h=4 | 2016-01-01 → 2026-10-01 | USGS NWIS Instantaneous Values web service, public domain |
| tide_level_battery | 934,841 | 20 | h=10 | 2016-01-01 → 2026-09-30 | NOAA CO-OPS Tides & Currents API, public domain |
| weather_temp_auckland | 94,217 | 20 | h=1 | 2016-01-01 → 2026-09-30 | Open-Meteo historical weather API (ERA5), Zippenfenig 2023, CC BY 4.0 |
| weather_temp_berlin | 94,217 | 20 | h=1 | 2016-01-01 → 2026-09-30 | Open-Meteo historical weather API (ERA5), Zippenfenig 2023, CC BY 4.0 |
| weather_temp_phoenix | 94,217 | 20 | h=1 | 2016-01-01 → 2026-09-30 | Open-Meteo historical weather API (ERA5), Zippenfenig 2023, CC BY 4.0 |
| weather_wind_hamburg | 94,217 | 20 | h=1 | 2016-01-01 → 2026-09-30 | Open-Meteo historical weather API (ERA5), Zippenfenig 2023, CC BY 4.0 |

### Synthetic

| Dataset | #Instances | #Features | — | Drift | Generator |
|---|---:|---:|---:|---|---|
| friedman_abrupt | 500,000 | 10 | — | abrupt @ 250000 | Friedman |
| friedman_gradual | 500,000 | 10 | — | gradual @ 250000 | Friedman |
| hyperplane_reg_abrupt | 500,000 | 10 | — | abrupt @ 250000 | HyperplaneRegression |
| hyperplane_reg_gradual | 500,000 | 10 | — | gradual @ 250000 | HyperplaneRegression |
| mv_abrupt | 500,000 | 10 | — | abrupt @ 250000 | Mv |
| mv_gradual | 500,000 | 10 | — | gradual @ 250000 | Mv |
| planes2d_abrupt | 500,000 | 10 | — | abrupt @ 250000 | Planes2D |
| planes2d_gradual | 500,000 | 10 | — | gradual @ 250000 | Planes2D |
| poisson_demand_abrupt | 500,000 | 31 | — | abrupt @ 250000 | PoissonDemand |
| poisson_demand_gradual | 500,000 | 31 | — | gradual @ 250000 | PoissonDemand |
| regime_walk_abrupt | 500,000 | 19 | — | abrupt @ 250000 | RegimeSwitchingWalk |
| regime_walk_gradual | 500,000 | 19 | — | gradual @ 250000 | RegimeSwitchingWalk |
| seasonal_sensor_abrupt | 500,000 | 13 | — | abrupt @ 250000 | SeasonalSensor |
| seasonal_sensor_gradual | 500,000 | 13 | — | gradual @ 250000 | SeasonalSensor |
