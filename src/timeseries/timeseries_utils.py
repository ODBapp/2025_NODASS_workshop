
import requests
import pandas as pd

# Function to get Niño 3.4 SST anomalies from NOAA data
def get_nino34_anomalies(start="1950", end="2025"):
    url = "https://psl.noaa.gov/data/timeseries/month/data/nino34.long.anom.data"
    response = requests.get(url, timeout=30)
    lines = response.text.strip().split('\n')[1:]  # skip header

    data = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 13:
            continue
        try:
            year = int(parts[0])
            if year < int(start[:4]) or year > int(end[:4]):
                continue
            for i, val in enumerate(parts[1:13]):
                if '-99.99' in val:
                    break
                ts = pd.Timestamp(year=year, month=i + 1, day=15)
                data.append((ts, float(val)))
        except ValueError:
            continue

    return pd.DataFrame(data, columns=["date", "ssta"]).set_index("date")