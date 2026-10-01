"""Getting started with a basic mono-facial PV energy simulation."""

import datetime
import zoneinfo

import numpy

import pv_workflows.atmosphere
import pv_workflows.common
import pv_workflows.location
import pv_workflows_pvlib.atmosphere
import pv_workflows_pvlib.irradiance
import pv_workflows_pvlib.location

# Alternative method to create compatible timestamps sequence using pandas.
# timestamps = pv_workflows.common.Timestamps(
#     sequence=tuple(
#         pandas.date_range(
#             start="2026-01-24 12:00:00",
#             end="2026-01-24 12:00:00",
#             freq="60min",
#             tz="America/Denver",
#         )
#         .to_pydatetime()
#         .tolist()
#     )
# )

# Energy-simulation configuration.
timestamps = pv_workflows.common.Timestamp(
    sequence=(
        datetime.datetime(2026, 6, 22, 12, tzinfo=zoneinfo.ZoneInfo("America/Denver")),
    )
)
latitude = pv_workflows.location.Latitude(value=45.677, units="deg")
longitude = pv_workflows.location.Longitude(value=-111.043, units="deg")
altitude = pv_workflows.location.Altitude(value=4820, units="m")

poa_tilt = pv_workflows.common.Angle(array=45.0, units="deg")
poa_azimuth = pv_workflows.common.Angle(array=180.0, units="deg")

dry_bulb_temperature = pv_workflows.atmosphere.Temperature(
    array=numpy.array(25.0), units="degC"
)
wind_speed = pv_workflows.atmosphere.WindSpeed(
    array=numpy.array(1.0),
    units="m s-1",
    height=pv_workflows.common.Height(value=10, units="m"),
)

ground_ghi = pv_workflows.atmosphere.Irradiance(
    array=numpy.array(1000.0), units="W m-2"
)
ground_albedo = pv_workflows.atmosphere.Albedo(array=numpy.array(0.124), units="")

print()

sun_position = pv_workflows_pvlib.location.sun_position_nrel_numpy(
    timestamp=timestamps,
    latitude=latitude,
    longitude=longitude,
    altitude=altitude,
    dry_bulb_temperature=dry_bulb_temperature,
)

print(f"Sun position calculated with temperature via pvlib:\n{sun_position}")

print()

decomposition = pv_workflows_pvlib.irradiance.erbs_driesse_decomposition(
    timestamps=timestamps,
    ground_ghi=ground_ghi,
    **sun_position,  # Contains sun_zenith. Extra aguments ignored.
)

print(f"Erbs-Driesse decomposition of GHI via pvlib:\n{decomposition}")

print()

# FIXME Compute dni_extraterrestrial and use above in erbs_driesse_decomposition.
extraterrestrial_dni = pv_workflows.atmosphere.Irradiance(
    array=numpy.array(1366.1), units="W m-2"
)

# TODO It's unclear if using sun position at sea level is technically correct here.
sun_position_sea_level = pv_workflows_pvlib.location.sun_position_nrel_numpy(
    timestamp=timestamps,
    latitude=latitude,
    longitude=longitude,
    altitude=pv_workflows.location.Altitude(value=0, units="m"),
    dry_bulb_temperature=dry_bulb_temperature,
)

# Compute relative air mass at sea level.
air_mass_relative = pv_workflows_pvlib.atmosphere.air_mass_relative_kastenyoung1989(
    **sun_position_sea_level,  # Contains sun_apparent_zenith.
)

poa_components = pv_workflows_pvlib.irradiance.perez_driesse_poa_components(
    poa_tilt=poa_tilt,
    poa_azimuth=poa_azimuth,
    **sun_position_sea_level,  # Contains sun_zenith_apparent and sun_azimuth.
    **decomposition,  # Contains ground_dhi and ground_dni.
    ground_ghi=ground_ghi,
    extraterrestrial_dni=extraterrestrial_dni,
    ground_albedo=ground_albedo,
    air_mass_relative=air_mass_relative,
)

print(f"Perez-Driesse POA components via pvlib:\n{poa_components}")

print()
