"""Getting started with a basic mono-facial PV energy simulation."""

import datetime
import zoneinfo

import numpy

import pv_workflows.atmosphere
import pv_workflows.common
import pv_workflows.location
import pv_workflows.temperature
import pv_workflows_pvlib.atmosphere
import pv_workflows_pvlib.irradiance
import pv_workflows_pvlib.location

# Alternative method to create compatible timestamp sequence using pandas.
# timestamp = pv_workflows.common.Timestamps(
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

# Energy-simulation configuration, with related elements collected into dictionaries.
timestamp = pv_workflows.common.Timestamps(
    sequence=(
        datetime.datetime(2026, 6, 22, 12, tzinfo=zoneinfo.ZoneInfo("America/Denver")),
    )
)
location = {
    "latitude": pv_workflows.location.Latitude(value=45.677, units="deg"),
    "longitude": pv_workflows.location.Longitude(value=-111.043, units="deg"),
    "altitude": pv_workflows.location.Altitude(value=4820, units="m"),
}
poa_geometry = {
    "poa_tilt": pv_workflows.common.Angles(array=45.0, units="deg"),
    "poa_azimuth": pv_workflows.common.Angles(array=180.0, units="deg"),
}
weather = {
    "dry_bulb_temperature": pv_workflows.common.Temperatures(
        array=numpy.array(25.0), units="degC"
    ),
    "wind_speed": pv_workflows.atmosphere.WindSpeeds(
        array=numpy.array(1.0),
        units="m s-1",
        height=pv_workflows.common.Height(value=10, units="m"),
    ),
    "ghi": pv_workflows.atmosphere.Irradiances(
        array=numpy.array(1000.0), units="W m-2"
    ),
    "albedo": pv_workflows.atmosphere.Albedos(array=numpy.array(0.124)),
}
heat_balance_coefficients = {
    "thermal_conduction_coefficient": pv_workflows.temperature.Uc(
        value=25 / (0.9 * (1 - 0.2)), units="W m-2 degC-1"
    ),
    "thermal_convection_coefficient": pv_workflows.temperature.Uv(
        value=1.2 / (0.9 * (1 - 0.2)), units="W s m-3 degC-1"
    ),
}

print()

sun_position = pv_workflows_pvlib.location.compute_sun_position_nrel_numpy(
    timestamp=timestamp,
    **location,  # Contains only latitude, longitude, and altitude.
    dry_bulb_temperature=weather["dry_bulb_temperature"],
)

print(f"Sun position calculated with temperature via pvlib:\n{sun_position}")

print()

decomposition = pv_workflows_pvlib.irradiance.decompose_ghi_erbs_driesse(
    timestamp=timestamp,
    ghi=weather["ghi"],
    sun_zenith=sun_position.zenith,
)

print(f"Erbs-Driesse decomposition of GHI via pvlib:\n{decomposition}")

print()

extraterrestrial_dni = (
    pv_workflows_pvlib.irradiance.compute_extraterrestrial_dni_spencer(
        timestamp=timestamp, solar_constant=None
    )
)

# TODO It's unclear if using sun position at sea level is technically correct here.
sun_position_sea_level = pv_workflows_pvlib.location.compute_sun_position_nrel_numpy(
    timestamp=timestamp,
    latitude=location["latitude"],
    longitude=location["longitude"],
    altitude=pv_workflows.location.Altitude(value=0, units="m"),
    dry_bulb_temperature=weather["dry_bulb_temperature"],
)

# Compute relative air mass (at sea level).
relative_air_mass = (
    pv_workflows_pvlib.atmosphere.compute_relative_air_mass_kastenyoung1989(
        sun_zenith_apparent=sun_position_sea_level.zenith_apparent,
    )
)

poa_irradiance_components = (
    pv_workflows_pvlib.irradiance.compute_poa_irradiance_components_perez_driesse(
        **poa_geometry,  # Contains only poa_tilt and poa_azimuth.
        sun_zenith_apparent=sun_position_sea_level.zenith_apparent,
        sun_azimuth=sun_position_sea_level.azimuth,
        dhi=decomposition.dhi,
        dni=decomposition.dni,
        extraterrestrial_dni=extraterrestrial_dni,
        relative_air_mass=relative_air_mass,
        ghi=weather["ghi"],
        albedo=weather["albedo"],
    )
)

print(f"Perez-Driesse POA components via pvlib:\n{poa_irradiance_components}")

print()

cell_temperature = pv_workflows.temperature.compute_cell_temperature_heat_balance(
    dry_bulb_temperature=weather["dry_bulb_temperature"],
    wind_speed=weather["wind_speed"],
    poa_irradiance=poa_irradiance_components.total,
    **heat_balance_coefficients,  # Contains only thermal_conduction_coefficient and thermal_convection_coefficient.
)

print(f"Cell temperature from heat balance:\n{cell_temperature}")

print()
