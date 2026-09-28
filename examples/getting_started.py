from dataclasses import asdict
import datetime
import zoneinfo

import numpy

# import pandas
import pvlib

import pv_workflows.common
import pv_workflows.irradiance
import pv_workflows.location
import pv_workflows.weather


location = pv_workflows.location.Location(
    latitude=pv_workflows.location.Latitude(value=45.677, units="deg"),
    longitude=pv_workflows.location.Longitude(value=-111.043, units="deg"),
    altitude=pv_workflows.location.Altitude(value=4820, units="m"),
    tzinfo=zoneinfo.ZoneInfo("America/Denver"),
)
timestamps = pv_workflows.common.Timestamps(
    sequence=(datetime.datetime(2026, 6, 22, 12, tzinfo=location.tzinfo),)
)
location_timestamps = pv_workflows.location.LocationTimestamps(
    location=location, timestamps=timestamps
)

print("")

print("pvlib solar position calculated without weather:")
print(
    pv_workflows.location.pvlib_solar_position_from_location_timestamps(
        location_timestamps=location_timestamps
    )
)

print("")

weather_ghi = pv_workflows.weather.WeatherGhi(
    dew_point_temperatures=None,
    dry_bulb_temperatures=pv_workflows.weather.Temperatures(
        array=numpy.array(25.0), units="degC"
    ),
    ghi=pv_workflows.weather.Irradiances(
        array=numpy.array(1000.0),
        units="W m-2",
    ),
    pressures=pv_workflows.weather.Pressures(
        array=numpy.asarray(
            pvlib.atmosphere.alt2pres(location_timestamps.location.altitude.value)
        ),
        units="Pa",
    ),
    wind_speeds=pv_workflows.weather.WindSpeeds(
        array=numpy.array(1.0),
        units="m s-1",
        height=pv_workflows.common.Height(value=10, units="m"),
    ),
)
location_timestamps_weather = pv_workflows.location.LocationTimestampsWeather(
    location=location, timestamps=timestamps, weather=weather_ghi
)

solar_position_location_timestamps_weather = (
    pv_workflows.location.pvlib_solar_position_from_location_timestamps_weather(
        location_timestamps_weather=location_timestamps_weather
    )
)

print("pvlib solar position calculated with weather:")
print(solar_position_location_timestamps_weather)

# DIRINT is a WIP: Requires at least two timesteps.
# print("")

# print("pvlib DIRINT decomposition of GHI:")
# print(
#     pv_workflows.irradiance.pvlib_dirint_decomposition_from_solar_position_timestamps_weather(
#         solar_position_timestamps_weather=solar_position_location_timestamps_weather
#     )
# )

print("")

print("pvlib DISC decomposition of GHI:")
print(
    pv_workflows.irradiance.pvlib_disc_decomposition_from_solar_position_timestamps_weather(
        solar_position_timestamps_weather=solar_position_location_timestamps_weather
    )
)

print("")

print("pvlib Erbs decomposition of GHI:")
print(
    pv_workflows.irradiance.pvlib_erbs_decomposition_from_solar_position_timestamps_weather(
        solar_position_timestamps_weather=solar_position_location_timestamps_weather
    )
)

print("")

dhi_dni_ghi = pv_workflows.irradiance.pvlib_erbs_driesse_decomposition_from_solar_position_timestamps_weather(
    solar_position_timestamps_weather=solar_position_location_timestamps_weather
)

print("pvlib Erbs-Driesse decomposition of GHI:")
print(
    pv_workflows.irradiance.pvlib_erbs_driesse_decomposition_from_solar_position_timestamps_weather(
        solar_position_timestamps_weather=solar_position_location_timestamps_weather
    )
)

print("")

weather_dhi_dni_ghi = pv_workflows.weather.WeatherDhiDniGhi(
    **asdict(weather_ghi), dhi=dhi_dni_ghi.dhi, dni=dhi_dni_ghi.dni
)

print(weather_dhi_dni_ghi)

# print(
#     type(
#         pandas.date_range(
#             start="2026-01-24 12:00:00",
#             end="2026-01-24 12:00:00",
#             freq="60min",
#             tz="America/Denver",
#         )
#         .to_pydatetime()
#         .tolist()[0]
#     )
# )
