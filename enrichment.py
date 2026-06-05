import os
import requests
import pandas as pd
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataEnricher:
    """
    Enriches the base unified dataset with external API data.
    - FRED API: Macro-economic indicators (Inflation, Retail Sales, Savings Rate)
    - Open-Meteo API: Geocoding and historical weather to engineer a precipitation flag.
    """

    def __init__(self):
        self.fred_api_key = os.getenv("FRED_API_KEY")
        if not self.fred_api_key:
            logger.warning("FRED_API_KEY environment variable is not set. FRED enrichment may fail.")

    def fetch_fred_series(self, series_id: str) -> pd.DataFrame:
        """
        Fetches monthly observations for a given FRED series ID.
        """
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": self.fred_api_key,
            "file_type": "json"
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        df = pd.DataFrame(data["observations"])
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Format date as Year-Month string for merging
        df['YearMonth'] = df['date'].dt.to_period('M').astype(str)
        df.rename(columns={'value': series_id}, inplace=True)
        return df[['YearMonth', series_id]]

    def get_macro_indicators(self) -> pd.DataFrame:
        """
        Fetches and merges CPIAUCSL (Inflation), RSXFS (Retail Sales), and PSAVERT (Savings Rate).
        """
        logger.info("Fetching macro-economic indicators from FRED...")
        try:
            cpi = self.fetch_fred_series("CPIAUCSL")
            rsxfs = self.fetch_fred_series("RSXFS")
            psavert = self.fetch_fred_series("PSAVERT")
            
            macro_df = pd.merge(cpi, rsxfs, on='YearMonth', how='outer')
            macro_df = pd.merge(macro_df, psavert, on='YearMonth', how='outer')
            return macro_df
        except Exception as e:
            logger.error(f"Failed to fetch FRED data: {e}")
            return pd.DataFrame()

    def get_coordinates(self, location_name: str) -> tuple[Optional[float], Optional[float]]:
        """
        Resolves a location string to lat/long using Open-Meteo Geocoding API.
        """
        url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {
            "name": location_name,
            "count": 1,
            "language": "en",
            "format": "json"
        }
        response = requests.get(url, params=params)
        if response.status_code == 200 and "results" in response.json():
            result = response.json()["results"][0]
            return result["latitude"], result["longitude"]
        return None, None

    def get_weather_precipitation_flag(self, lat: float, lon: float, start_date: str, end_date: str) -> int:
        """
        Calls Open-Meteo Archive API to check if precipitation occurred.
        Returns 1 if precipitation > 0 during the period, 0 otherwise.
        """
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "precipitation_sum",
            "models": "best_match"
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            precip_sums = data.get("daily", {}).get("precipitation_sum", [])
            # If any day had precipitation > 0
            if any(p is not None and p > 0 for p in precip_sums):
                return 1
            return 0
        except Exception as e:
            logger.error(f"Failed to fetch weather data for {lat},{lon}: {e}")
            return 0

    def enrich_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Orchestrates the enrichment of the unified dataframe.
        Expects df to have 'Country' or location data, and date fields.
        """
        logger.info("Starting data enrichment...")
        
        # 1. Macro Indicators
        macro_df = self.get_macro_indicators()
        if not macro_df.empty and 'InvoiceDate' in df.columns:
            # We assume InvoiceDate was retained or we have a generic 'Last_Transaction_Date'
            # Fallback to current YearMonth if no date column exists for simplistic join
            df['YearMonth'] = pd.to_datetime(df.get('InvoiceDate', 'today')).dt.to_period('M').astype(str)
            df = pd.merge(df, macro_df, on='YearMonth', how='left')
            df.drop(columns=['YearMonth'], inplace=True)
            logger.info("Merged FRED macro indicators.")

        # 2. Weather Precipitation Flag (Mock/Example implementation for top countries to avoid API limits)
        # In a real pipeline, this would be mapped per unique country/date combination.
        if 'Country' in df.columns:
            unique_countries = df['Country'].dropna().unique()
            # Just resolving the first one as an example to prevent massive API spam
            for country in unique_countries[:1]:
                lat, lon = self.get_coordinates(country)
                if lat and lon:
                    # Using a dummy recent date range for the archive API
                    precip_flag = self.get_weather_precipitation_flag(lat, lon, "2023-01-01", "2023-01-31")
                    df['purchased_during_precipitation'] = precip_flag
                    logger.info(f"Resolved weather for {country}. Precip flag: {precip_flag}")
        else:
            # Add default 0 if mapping fails
            df['purchased_during_precipitation'] = 0

        logger.info("Enrichment complete.")
        return df

if __name__ == "__main__":
    enricher = DataEnricher()
    # Dummy df test
    test_df = pd.DataFrame({"Country": ["United Kingdom"], "InvoiceDate": ["2010-12-01"]})
    res = enricher.enrich_data(test_df)
    print(res)
