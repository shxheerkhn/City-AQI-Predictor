from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Protocol

import numpy as np
import requests

from aqi_predictor.configs.settings import CityProfile
from aqi_predictor.models.domain import AQIMeasurement
from aqi_predictor.utils.logging import get_logger


LOGGER = get_logger(__name__)


class AQIDataSource(Protocol):
    def fetch_observation(self, city: CityProfile, timestamp: datetime) -> AQIMeasurement:
        raise NotImplementedError


@dataclass(slots=True)
class SyntheticAQIDataSource:
    seed_namespace: str = "pearls-aqi"

    def fetch_observation(self, city: CityProfile, timestamp: datetime) -> AQIMeasurement:
        try:
            base_hash = sha256(f"{self.seed_namespace}:{city.city_id}:{timestamp.isoformat()}".encode("utf-8")).hexdigest()
            seed = int(base_hash[:16], 16) % (2**32)
            rng = np.random.default_rng(seed)
            hour = timestamp.hour + timestamp.minute / 60.0
            day_of_year = timestamp.timetuple().tm_yday
            hour_cycle = math.sin(2.0 * math.pi * hour / 24.0)
            day_cycle = math.sin(2.0 * math.pi * day_of_year / 365.25)
            seasonal_cos = math.cos(2.0 * math.pi * day_of_year / 365.25)
            traffic_wave = 1.0 + 0.35 * math.sin(2.0 * math.pi * (hour - 8.0) / 24.0)
            temperature = 25.0 + 10.0 * hour_cycle + 7.0 * day_cycle + 0.1 * city.latitude / 10.0 + rng.normal(0.0, 0.8)
            humidity = 58.0 + 16.0 * seasonal_cos - 8.0 * hour_cycle + rng.normal(0.0, 2.0)
            humidity = float(np.clip(humidity, 10.0, 100.0))
            wind_speed = float(np.clip(2.4 + 1.7 * math.cos(2.0 * math.pi * hour / 24.0) + rng.normal(0.0, 0.35), 0.1, 12.0))
            pressure = float(np.clip(1008.0 + 4.0 * seasonal_cos + rng.normal(0.0, 1.0), 980.0, 1035.0))
            rainfall = float(max(0.0, rng.normal(0.9 + 0.7 * (1.0 - seasonal_cos), 0.6)))
            dispersion_penalty = 18.0 / (wind_speed + 0.8) + 0.25 * humidity + 5.0 * (1.0 if rainfall == 0.0 else 0.0)
            pollution_base = 70.0 + 20.0 * traffic_wave + 11.0 * day_cycle + city.aqi_bias
            pm25 = float(max(5.0, pollution_base + dispersion_penalty + rng.normal(0.0, 8.0)))
            pm10 = float(max(pm25 + 8.0, pm25 * 1.35 + rng.normal(0.0, 10.0)))
            co = float(max(0.1, 0.9 + 0.04 * pm25 + rng.normal(0.0, 0.08)))
            no2 = float(max(3.0, 16.0 + 0.12 * pm25 + rng.normal(0.0, 2.0)))
            so2 = float(max(1.0, 5.0 + 0.06 * pm10 + rng.normal(0.0, 1.2)))
            o3 = float(max(4.0, 30.0 + 0.5 * temperature - 0.16 * humidity + rng.normal(0.0, 4.0)))
            aqi = self._compute_aqi(pm25, pm10, co, no2, so2, o3)
            return AQIMeasurement(
                city_id=city.city_id,
                timestamp=timestamp,
                aqi=aqi,
                temperature=float(temperature),
                humidity=float(humidity),
                wind_speed=float(wind_speed),
                pressure=float(pressure),
                rainfall=float(rainfall),
                pm25=float(pm25),
                pm10=float(pm10),
                co=float(co),
                no2=float(no2),
                so2=float(so2),
                o3=float(o3),
                source="synthetic",
            )
        except Exception as exc:
            LOGGER.exception("Synthetic observation generation failed")
            raise RuntimeError(f"Synthetic observation generation failed: {exc}") from exc

    @staticmethod
    def _compute_aqi(pm25: float, pm10: float, co: float, no2: float, so2: float, o3: float) -> float:
        try:
            components = [
                pm25 * 1.25,
                pm10 * 0.7,
                co * 38.0,
                no2 * 3.0,
                so2 * 4.5,
                o3 * 1.6,
            ]
            aqi = max(components)
            return float(np.clip(aqi, 1.0, 500.0))
        except Exception as exc:
            LOGGER.exception("AQI computation failed")
            raise RuntimeError(f"AQI computation failed: {exc}") from exc


@dataclass(slots=True)
class OpenWeatherAQIDataSource:
    api_key: str
    base_url: str = "https://api.openweathermap.org/data/2.5"
    timeout_seconds: int = 20

    def fetch_observation(self, city: CityProfile, timestamp: datetime) -> AQIMeasurement:
        try:
            if not self.api_key:
                raise ValueError("OpenWeather API key is required")
            weather_url = f"{self.base_url}/weather"
            params = {
                "lat": city.latitude,
                "lon": city.longitude,
                "appid": self.api_key,
                "units": "metric",
            }
            response = requests.get(weather_url, params=params, timeout=self.timeout_seconds)
            response.raise_for_status()
            payload = response.json()
            weather = payload.get("main", {})
            wind = payload.get("wind", {})
            aqi_url = f"{self.base_url}/air_pollution"
            air_response = requests.get(aqi_url, params=params, timeout=self.timeout_seconds)
            air_response.raise_for_status()
            air_payload = air_response.json()
            components = air_payload.get("list", [{}])[0].get("components", {})
            measurement = AQIMeasurement(
                city_id=city.city_id,
                timestamp=timestamp,
                aqi=float(air_payload.get("list", [{}])[0].get("main", {}).get("aqi", 1)) * 50.0,
                temperature=float(weather.get("temp", 0.0)),
                humidity=float(weather.get("humidity", 0.0)),
                wind_speed=float(wind.get("speed", 0.0)),
                pressure=float(weather.get("pressure", 0.0)),
                rainfall=float(payload.get("rain", {}).get("1h", 0.0) or 0.0),
                pm25=float(components.get("pm2_5", 0.0)),
                pm10=float(components.get("pm10", 0.0)),
                co=float(components.get("co", 0.0)),
                no2=float(components.get("no2", 0.0)),
                so2=float(components.get("so2", 0.0)),
                o3=float(components.get("o3", 0.0)),
                source="openweather",
            )
            return measurement
        except Exception as exc:
            LOGGER.exception("OpenWeather fetch failed")
            raise RuntimeError(f"OpenWeather fetch failed: {exc}") from exc


@dataclass(slots=True)
class AQICNAQIDataSource:
    api_token: str
    base_url: str = "http://api.waqi.info"
    timeout_seconds: int = 20
    station_id: str | None = None

    def fetch_observation(self, city: CityProfile, timestamp: datetime) -> AQIMeasurement:
        try:
            if not self.api_token:
                raise ValueError("AQICN API token is required")
            payload = self._fetch_payload(city)
            if payload.get("status") != "ok":
                raise ValueError(f"AQICN API returned non-ok status: {payload.get('status')}")
            data = payload.get("data", {})
            iaqi = data.get("iaqi", {})
            pm25 = self._extract_iaqi(iaqi, "pm25")
            pm10 = self._extract_iaqi(iaqi, "pm10")
            co = self._extract_iaqi(iaqi, "co")
            no2 = self._extract_iaqi(iaqi, "no2")
            so2 = self._extract_iaqi(iaqi, "so2")
            o3 = self._extract_iaqi(iaqi, "o3")
            temperature = self._extract_iaqi(iaqi, "t") or 0.0
            humidity = self._extract_iaqi(iaqi, "h") or 0.0
            wind_speed = self._extract_iaqi(iaqi, "w") or 0.0
            rainfall = 0.0
            pressure = 1000.0
            aqi = float(data.get("aqi", 0.0))
            if aqi <= 0.0:
                aqi = SyntheticAQIDataSource._compute_aqi(pm25, pm10, co, no2, so2, o3)
            return AQIMeasurement(
                city_id=city.city_id,
                timestamp=timestamp,
                aqi=float(aqi),
                temperature=float(temperature),
                humidity=float(humidity),
                wind_speed=float(wind_speed),
                pressure=float(pressure),
                rainfall=float(rainfall),
                pm25=float(pm25),
                pm10=float(pm10),
                co=float(co),
                no2=float(no2),
                so2=float(so2),
                o3=float(o3),
                source="aqicn",
            )
        except Exception as exc:
            LOGGER.exception("AQICN fetch failed")
            raise RuntimeError(f"AQICN fetch failed: {exc}") from exc

    def _fetch_payload(self, city: CityProfile) -> dict[str, object]:
        try:
            preferred_ids: list[str] = []
            if self.station_id:
                preferred_ids.append(self.station_id)
            if city.city_id.lower() == "karachi":
                preferred_ids.extend(["@11790", "11790", "A471613"])

            tried_urls: list[str] = []
            for station_id in preferred_ids:
                normalized_station_id = station_id.lstrip("@")
                url = f"{self.base_url.rstrip('/')}/feed/@{normalized_station_id}/"
                tried_urls.append(url)
                try:
                    response = requests.get(url, params={"token": self.api_token}, timeout=self.timeout_seconds)
                    response.raise_for_status()
                    payload = response.json()
                    if payload.get("status") == "ok":
                        LOGGER.info("AQICN station lookup succeeded for %s using %s", city.city_id, url)
                        return payload
                    LOGGER.warning("AQICN station lookup returned %s for %s", payload.get("status"), url)
                except Exception as exc:
                    LOGGER.warning("AQICN station lookup failed for %s: %s", url, exc)

            geo_url = f"{self.base_url.rstrip('/')}/feed/geo:{city.latitude:.4f};{city.longitude:.4f}/"
            tried_urls.append(geo_url)
            response = requests.get(geo_url, params={"token": self.api_token}, timeout=self.timeout_seconds)
            response.raise_for_status()
            payload = response.json()
            if payload.get("status") != "ok":
                raise ValueError(f"AQICN geo lookup returned non-ok status: {payload.get('status')}")
            LOGGER.info("AQICN geo lookup succeeded for %s using %s", city.city_id, geo_url)
            return payload
        except Exception as exc:
            raise RuntimeError(f"AQICN lookup failed after trying direct station and geo endpoints: {exc}") from exc

    @staticmethod
    def _extract_iaqi(iaqi: dict[str, dict[str, float] | float], key: str) -> float:
        value = iaqi.get(key, 0.0)
        if isinstance(value, dict):
            return float(value.get("v", 0.0))
        return float(value)
