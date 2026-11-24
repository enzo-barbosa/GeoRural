# Changelog

## [0.2.0](https://github.com/enzo-barbosa/GeoRural/compare/georural-v0.1.0...georural-v0.2.0) (2026-11-24)


### Features

* align RF/SVM features with U-Net channels and train on Sentinel-2 ([5a36b33](https://github.com/enzo-barbosa/GeoRural/commit/5a36b33de30ae7ed98c667450a472df062c1afe8))
* build full U-Net raster dataset and align input channels ([7ebf91e](https://github.com/enzo-barbosa/GeoRural/commit/7ebf91ede2826a26be4ce24e7260cee169c51472))
* publish trained models through github releases ([#5](https://github.com/enzo-barbosa/GeoRural/issues/5)) ([06a3d50](https://github.com/enzo-barbosa/GeoRural/commit/06a3d5063cb6dbc6d26e1e673691729a476a9e43))


### Bug Fixes

* correct S2 cloud mask, NDVI validity, balanced GEE sampling, and chunked raster download ([8e52cbc](https://github.com/enzo-barbosa/GeoRural/commit/8e52cbc0bf91fc3dc6c02e185b5562c1ceee34ae))
* unify land-cover class order and fix U-Net data prep ([eca9087](https://github.com/enzo-barbosa/GeoRural/commit/eca90877bbf04e05f3f7f0550ea1d17f50716699))


### Documentation

* add license, contributing guide, and github templates ([240eed7](https://github.com/enzo-barbosa/GeoRural/commit/240eed7ff59b3f48c416e855d8aca82b150b4aca))
* align stack to PyTorch, fix classifier features in app, add U-Net training script ([d0ef37c](https://github.com/enzo-barbosa/GeoRural/commit/d0ef37c26f6b7e6f18e0d135795c87ed9094272f))
* record session continuity state and GEE account decision ([01f1d10](https://github.com/enzo-barbosa/GeoRural/commit/01f1d107b23c5949f8db5c79d9177f93a5df368e))
* require one pull request at a time ([#12](https://github.com/enzo-barbosa/GeoRural/issues/12)) ([366265a](https://github.com/enzo-barbosa/GeoRural/commit/366265a87220f6ffc918781039c077b2c08d54a4))
