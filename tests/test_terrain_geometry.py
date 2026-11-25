"""Check the signed local incidence and terrain flags on analytic DEM profiles."""

import math

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from data.preprocessing import _local_incidence, terrain_masks


@pytest.mark.parametrize("orbit,look_east", [("ASCENDING", 1), ("DESCENDING", -1)])
@pytest.mark.parametrize(
    "east_slope_deg",
    [0.0, 45.0, -45.0, 60.0, -60.0],
    ids=["flat", "rise-east-45", "rise-west-45", "rise-east-60", "rise-west-60"],
)
def test_terrain_flags_follow_sensor_look(tmp_path, orbit, look_east, east_slope_deg):
    size = 9
    step = 0.00025
    mid_latitude = 40 - step * size / 2
    dx_m = step * 111_320 * math.cos(math.radians(mid_latitude))
    dy_m = step * 110_540
    east_distance = dx_m * np.arange(size)
    profile = east_distance * math.tan(math.radians(east_slope_deg))
    elevation = np.tile(profile, (size, 1)).astype(np.float32)
    transform = from_origin(49, 40, step, step)
    reference = tmp_path / "dem"
    reference.mkdir()
    dem_path = reference / "ramp.tif"
    with rasterio.open(
        dem_path, "w", driver="GTiff", width=size, height=size, count=1,
        dtype=elevation.dtype, crs="EPSG:4326", transform=transform,
    ) as dataset:
        dataset.write(elevation, 1)

    # Along a planar east/west ramp the surface normal rotates toward the
    # sensor by exactly the signed rise angle along the sensor-to-ground ray.
    # Neither an aspect calculation nor a production azimuth feeds this value.
    expected_incidence = 37.55 - look_east * east_slope_deg
    expected_exclusion = expected_incidence <= 0 or expected_incidence >= 90
    incidence = _local_incidence(elevation, dx_m, dy_m, orbit, 37.55)
    np.testing.assert_allclose(np.degrees(incidence), expected_incidence, atol=1e-4)

    invalid, valid, gaps = terrain_masks(dem_path, reference, orbit, 29.1, 46.0)
    assert valid.all()
    assert not gaps.any()
    assert np.all(invalid == expected_exclusion)
