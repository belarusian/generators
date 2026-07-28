#!/usr/bin/env python3
"""weather_discover.py - A complete weather discovery program.

Usage:
    python weather_discover.py                         # Interactive mode
    python weather_discover.py current <city>           # Current weather
    python weather_discover.py forecast <city> [days]   # Forecast
    python weather_discover.py compare <city1,city2>    # Compare cities

Requires: requests (pip install requests)
Uses wttr.in public API - no API key needed.
"""

import sys
import json
import requests
from datetime import datetime


class WeatherDiscovery:
    """Discovers weather conditions and forecasts for any location worldwide."""

    BASE_URL = "https://wttr.in"

    def __init__(self):
        self.cache = {}
        self.history = []

    def get_current_weather(self, city: str) -> dict:
        try:
            url = f"{self.BASE_URL}/{city}?format=j1"
            resp = requests.get(url, timeout=10, headers={"User-Agent": "WeatherDiscovery/1.0"})
            resp.raise_for_status()
            data = resp.json()
            current = data.get("current_condition", [{}])[0]
            result = {
                "city": city,
                "timestamp": datetime.now().isoformat(),
                "temperature_c": current.get("temp_C", "N/A"),
                "temperature_f": current.get("temp_F", "N/A"),
                "feels_like_c": current.get("FeelsLikeC", "N/A"),
                "feels_like_f": current.get("FeelsLikeF", "N/A"),
                "humidity": current.get("humidity", "N/A"),
                "description": current.get("weatherDesc", [{}])[0].get("value", "N/A"),
                "wind_speed_kmph": current.get("windspeedKmph", "N/A"),
                "wind_direction": current.get("winddir16Point", "N/A"),
                "pressure_mb": current.get("pressure", "N/A"),
                "visibility_km": current.get("visibility", "N/A"),
                "uv_index": current.get("uvIndex", "N/A"),
                "cloud_cover": current.get("cloudcover", "N/A"),
            }
            self.cache[city] = result
            self.history.append({"action": "current", "city": city, "time": result["timestamp"]})
            return result
        except requests.RequestException as e:
            return {"city": city, "error": str(e)}

    def get_forecast(self, city: str, days: int = 3) -> dict:
        try:
            url = f"{self.BASE_URL}/{city}?format=j1"
            resp = requests.get(url, timeout=10, headers={"User-Agent": "WeatherDiscovery/1.0"})
            resp.raise_for_status()
            data = resp.json()
            forecasts = []
            for day_data in data.get("weather", [])[:days]:
                hourly = day_data.get("hourly", [])
                mid = hourly[len(hourly) // 2] if hourly else {}
                forecasts.append({
                    "date": day_data.get("date", "N/A"),
                    "max_temp_c": day_data.get("maxtempC", "N/A"),
                    "min_temp_c": day_data.get("mintempC", "N/A"),
                    "max_temp_f": day_data.get("maxtempF", "N/A"),
                    "min_temp_f": day_data.get("mintempF", "N/A"),
                    "description": mid.get("weatherDesc", [{}])[0].get("value", "N/A"),
                    "humidity": mid.get("humidity", "N/A"),
                    "sunrise": day_data.get("astronomy", [{}])[0].get("sunrise", "N/A"),
                    "sunset": day_data.get("astronomy", [{}])[0].get("sunset", "N/A"),
                })
            result = {"city": city, "forecast_days": len(forecasts), "forecasts": forecasts}
            self.history.append({"action": "forecast", "city": city, "time": datetime.now().isoformat()})
            return result
        except requests.RequestException as e:
            return {"city": city, "error": str(e)}

    def compare_cities(self, cities: list) -> list:
        return [self.get_current_weather(c.strip()) for c in cities]

    def format_current(self, w: dict) -> str:
        if "error" in w:
            return f"Error for {w['city']}: {w['error']}"
        return (
            f"=== Weather for {w['city']} ===\n"
            f"  {w['description']}\n"
            f"  Temperature : {w['temperature_c']}\u00b0C ({w['temperature_f']}\u00b0F)\n"
            f"  Feels Like  : {w['feels_like_c']}\u00b0C ({w['feels_like_f']}\u00b0F)\n"
            f"  Humidity    : {w['humidity']}%\n"
            f"  Wind        : {w['wind_speed_kmph']} km/h {w['wind_direction']}\n"
            f"  Pressure    : {w['pressure_mb']} mb\n"
            f"  Visibility  : {w['visibility_km']} km\n"
            f"  UV Index    : {w['uv_index']}\n"
            f"  Cloud Cover : {w['cloud_cover']}%"
        )

    def format_forecast(self, f: dict) -> str:
        if "error" in f:
            return f"Error for {f['city']}: {f['error']}"
        lines = [f"=== {f['forecast_days']}-Day Forecast for {f['city']} ==="]
        for day in f.get("forecasts", []):
            lines.append(f"\n  {day['date']}")
            lines.append(f"    High: {day['max_temp_c']}\u00b0C  Low: {day['min_temp_c']}\u00b0C")
            lines.append(f"    {day['description']}  Humidity: {day['humidity']}%")
            lines.append(f"    Sunrise: {day['sunrise']}  Sunset: {day['sunset']}")
        return "\n".join(lines)


def main():
    wd = WeatherDiscovery()
    args = sys.argv[1:]

    if not args:
        print("\n  Weather Discovery v1.0")
        print("  " + "=" * 40)
        print("  Commands:")
        print("    current <city>          - Current weather")
        print("    forecast <city> [days]  - Weather forecast")
        print("    compare <c1,c2,...>      - Compare cities")
        print("    quit                    - Exit\n")
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if not line or line.lower() == "quit":
                print("Goodbye!")
                break
            parts = line.split()
            cmd = parts[0].lower()
            if cmd == "current" and len(parts) >= 2:
                w = wd.get_current_weather(parts[1])
                print(wd.format_current(w))
            elif cmd == "forecast" and len(parts) >= 2:
                days = int(parts[2]) if len(parts) > 2 else 3
                f = wd.get_forecast(parts[1], days)
                print(wd.format_forecast(f))
            elif cmd == "compare" and len(parts) >= 2:
                cities = parts[1].split(",")
                for w in wd.compare_cities(cities):
                    print(wd.format_current(w))
                    print()
            else:
                print("Unknown command. Try: current London")
        return

    cmd = args[0].lower()
    if cmd == "current" and len(args) >= 2:
        w = wd.get_current_weather(args[1])
        print(wd.format_current(w))
    elif cmd == "forecast" and len(args) >= 2:
        days = int(args[2]) if len(args) > 2 else 3
        f = wd.get_forecast(args[1], days)
        print(wd.format_forecast(f))
    elif cmd == "compare" and len(args) >= 2:
        cities = args[1].split(",")
        for w in wd.compare_cities(cities):
            print(wd.format_current(w))
            print()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
