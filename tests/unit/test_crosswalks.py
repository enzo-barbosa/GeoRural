"""Testes dos crosswalks de rótulos (MapBiomas / WorldCover)."""

from src.gee.ground_truth import (
    MAPBIOMAS_CROP_CROSSWALK,
    MAPBIOMAS_CROSSWALK,
    WORLDCOVER_CROSSWALK,
)


def test_mapbiomas_water_is_not_vegetation():
    assert MAPBIOMAS_CROSSWALK[26] == 5
    assert MAPBIOMAS_CROSSWALK[33] == 5
    assert MAPBIOMAS_CROSSWALK[31] == 5


def test_mapbiomas_pasture_and_crops():
    assert MAPBIOMAS_CROSSWALK[15] == 1
    assert MAPBIOMAS_CROSSWALK[20] == 2
    assert MAPBIOMAS_CROSSWALK[39] == 2
    assert MAPBIOMAS_CROSSWALK[47] == 2


def test_mapbiomas_ambiguous_codes_are_ignored():
    assert 21 not in MAPBIOMAS_CROSSWALK  # mosaico de usos
    assert 34 not in MAPBIOMAS_CROSSWALK  # geleira


def test_worldcover_crosswalk():
    assert WORLDCOVER_CROSSWALK[10] == 0
    assert WORLDCOVER_CROSSWALK[30] == 1
    assert WORLDCOVER_CROSSWALK[40] == 2
    assert WORLDCOVER_CROSSWALK[60] == 3
    assert WORLDCOVER_CROSSWALK[50] == 4
    assert WORLDCOVER_CROSSWALK[80] == 5
    assert WORLDCOVER_CROSSWALK[70] == -1


def test_crop_crosswalk_matches_config_order():
    # Cana, Citros, Pastagem, Soja, Milho
    assert MAPBIOMAS_CROP_CROSSWALK[20] == 0
    assert MAPBIOMAS_CROP_CROSSWALK[47] == 1
    assert MAPBIOMAS_CROP_CROSSWALK[15] == 2
    assert MAPBIOMAS_CROP_CROSSWALK[39] == 3
    assert MAPBIOMAS_CROP_CROSSWALK[41] == 4
