"""
Getting started with a single-panel, mono-facial, fixed-tilt, PV energy simulation.
"""

import datetime
import os
import zoneinfo

import numpy
import pandas
import pvlib  # FIXME Direct reference to implementation should not be necessary.

import pv_workflows.atmosphere
import pv_workflows.common
import pv_workflows.location
import pv_workflows.temperature
import pv_workflows_pvlib.atmosphere
import pv_workflows_pvlib.irradiance
import pv_workflows_pvlib.location

# Energy-simulation configuration, with related elements collected into dictionaries.

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

weather_filepath = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "weather",
    "346177_45.69_-111.02_tgy-2025.csv",
)
weather_nsrdb_header = pandas.read_csv(weather_filepath, nrows=1)

weather_nsrdb_data = pandas.read_csv(weather_filepath, skiprows=2)

# Construct timestamps from localized weather timestamps.
# Note that NSRDB does not shift hours for daylight-savings-time events.
tzinfo = zoneinfo.ZoneInfo(
    f"Etc/GMT{-int(weather_nsrdb_header['Time Zone'].item()):+}"  # Flip sign.
)
timestamp = pv_workflows.common.Timestamps(
    sequence=tuple(
        datetime.datetime(year, month, day, hour=hour, minute=minute, tzinfo=tzinfo)
        for year, month, day, hour, minute in zip(
            weather_nsrdb_data["Year"],
            weather_nsrdb_data["Month"],
            weather_nsrdb_data["Day"],
            weather_nsrdb_data["Hour"],
            weather_nsrdb_data["Minute"],
        )
    )
)

location = {
    "latitude": pv_workflows.location.Latitude(value=45.677, units="deg"),
    "longitude": pv_workflows.location.Longitude(value=-111.043, units="deg"),
    "altitude": pv_workflows.location.Altitude(value=4820, units="m"),
}
poa_geometry = {
    "poa_tilt": pv_workflows.common.TiltAngles(
        value=location["latitude"].value, units="deg"
    ),
    "poa_azimuth": pv_workflows.common.AzimuthAngles(value=180.0, units="deg"),
}
weather = {
    "dry_bulb_temperature": pv_workflows.common.Temperatures(
        value=weather_nsrdb_data["Temperature"].to_numpy(), units="degC"
    ),
    "wind_speed": pv_workflows.atmosphere.WindSpeeds(
        value=weather_nsrdb_data["Wind Speed"].to_numpy(),
        units="m s-1",
        height=pv_workflows.common.Height(value=10, units="m"),
    ),
    "pressure": pv_workflows.atmosphere.Pressures(
        value=100 * weather_nsrdb_data["Pressure"].to_numpy(),  # mbar to Pa
        units="Pa",
    ),
    "ghi": pv_workflows.atmosphere.Irradiances(
        value=weather_nsrdb_data["GHI"].to_numpy(), units="W m-2"
    ),
    "albedo": pv_workflows.atmosphere.Albedos(
        value=weather_nsrdb_data["Surface Albedo"].to_numpy()
    ),
}
heat_balance_coefficients = {
    "thermal_conduction_coefficient": pv_workflows.temperature.Uc(
        value=25 / (0.9 * (1 - 0.2)), units="W m-2 degC-1"
    ),
    "thermal_convection_coefficient": pv_workflows.temperature.Uv(
        value=1.2 / (0.9 * (1 - 0.2)), units="W s m-3 degC-1"
    ),
}
iam_profile = {
    "incident_angle": pv_workflows.irradiance.IncidentAngles(
        value=numpy.array(
            (
                0.0,
                10.0,
                20.0,
                30.0,
                40.0,
                45.0,
                50.0,
                55.0,
                60.0,
                65.0,
                70.0,
                75.0,
                80.0,
                85.0,
                90.0,
            )
        ),
        units="deg",
    ),
    "incident_angle_modifier": pv_workflows.irradiance.IncidentAngleModifiers(
        value=numpy.array(
            (
                1.0,
                1.0,
                1.0,
                1.0,
                0.998,
                0.994,
                0.991,
                0.982,
                0.963,
                0.935,
                0.892,
                0.813,
                0.678,
                0.454,
                0.0,
            )
        )
    ),
}

print()

sun_position = pv_workflows_pvlib.location.compute_sun_position_nrel_numpy(
    timestamp=timestamp,
    **location,  # Contains only latitude, longitude, and altitude.
    dry_bulb_temperature=weather["dry_bulb_temperature"],
    pressure=weather["pressure"],
)

print(f"Sun position:\n{sun_position}")

print()

# TODO Compare results using this decomposition with weather-file DHI and DNI.
decomposition = pv_workflows_pvlib.irradiance.decompose_ghi_erbs_driesse(
    timestamp=timestamp,
    ghi=weather["ghi"],
    sun_zenith=sun_position.zenith,
)

print(f"Decomposition of GHI:\n{decomposition}")

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
    pressure=weather["pressure"],
)

# Compute relative air mass (at sea level).
relative_air_mass = (
    pv_workflows_pvlib.atmosphere.compute_relative_air_mass_kastenyoung1989(
        sun_zenith_apparent=sun_position.zenith_apparent,
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

print(f"POA irradiance components:\n{poa_irradiance_components}")
print(f"POA irradiance total:\n{poa_irradiance_components.total}")

print()

cell_temperature = pv_workflows.temperature.compute_cell_temperature_heat_balance(
    dry_bulb_temperature=weather["dry_bulb_temperature"],
    wind_speed=weather["wind_speed"],
    poa_irradiance=poa_irradiance_components.total,
    **heat_balance_coefficients,  # Contains only thermal_conduction_coefficient and thermal_convection_coefficient.
)

print(f"Cell temperature from heat balance:\n{cell_temperature}")

print()

compute_iam_from_incident_angle = (
    pv_workflows.irradiance.construct_compute_iam_from_incident_angle_pchip(
        **iam_profile
    )
)

incident_angle = pv_workflows.irradiance.IncidentAngles(
    value=pvlib.irradiance.aoi(
        numpy.asarray(poa_geometry["poa_tilt"].value),
        numpy.asarray(poa_geometry["poa_azimuth"].value),
        numpy.asarray(sun_position.zenith.value),
        numpy.asarray(sun_position.azimuth.value),
    ),
    units="deg",
)

poa_iam_direct_circumsolar = pv_workflows.irradiance.compute_poa_iam_direct_circumsolar(
    compute_iam=compute_iam_from_incident_angle, incident_angle=incident_angle
)

poa_iam_components = pv_workflows.irradiance.PoaIamComponents(
    direct=poa_iam_direct_circumsolar,
    circumsolar=poa_iam_direct_circumsolar,
    isotropic=pv_workflows_pvlib.irradiance.construct_compute_iam_from_tilt_angle_marion_pchip(
        compute_iam=compute_iam_from_incident_angle, region="sky"
    )(tilt_angle=poa_geometry["poa_tilt"]),
    horizon=pv_workflows_pvlib.irradiance.construct_compute_iam_from_tilt_angle_marion_pchip(
        compute_iam=compute_iam_from_incident_angle, region="horizon"
    )(tilt_angle=poa_geometry["poa_tilt"]),
    ground=pv_workflows_pvlib.irradiance.construct_compute_iam_from_tilt_angle_marion_pchip(
        compute_iam=compute_iam_from_incident_angle, region="ground"
    )(tilt_angle=poa_geometry["poa_tilt"]),
)

effective_poa_irradiance_components = (
    pv_workflows.irradiance.compute_effective_poa_irradiance_components(
        poa_irradiance_components=poa_irradiance_components,
        poa_iam_components=poa_iam_components,
        losses=(),
    )
)

print(
    "Effective POA irradiance components, no losses, including IAM effects:\n"
    f"{effective_poa_irradiance_components}"
)
print(
    "Effective POA irradiance total, no losses, including IAM effects:\n"
    f"{effective_poa_irradiance_components.total}"
)

print()

poa_iam_components_no_iam = pv_workflows.irradiance.PoaIamComponents(
    direct=pv_workflows.irradiance.Irradiances(value=numpy.array(1), units="W m-2"),
    circumsolar=pv_workflows.irradiance.Irradiances(
        value=numpy.array(1), units="W m-2"
    ),
    isotropic=pv_workflows.irradiance.Irradiances(value=numpy.array(1), units="W m-2"),
    horizon=pv_workflows.irradiance.Irradiances(value=numpy.array(1), units="W m-2"),
    ground=pv_workflows.irradiance.Irradiances(value=numpy.array(1), units="W m-2"),
)

effective_poa_irradiance_components_no_iam = (
    pv_workflows.irradiance.compute_effective_poa_irradiance_components(
        poa_irradiance_components=poa_irradiance_components,
        poa_iam_components=poa_iam_components_no_iam,
        losses=(),
    )
)

print(
    "Effective POA irradiance components, no losses, excluding IAM effects:\n"
    f"{effective_poa_irradiance_components_no_iam}"
)
print(
    "Effective POA irradiance total, no losses, excluding IAM effects:\n"
    f"{effective_poa_irradiance_components_no_iam.total}"
)

print()
