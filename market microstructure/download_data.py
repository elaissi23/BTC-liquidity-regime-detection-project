import requests, os

base = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/"
months = [f"BTCUSDT-1m-{y}-{m:02d}.zip"
          for y in [2020, 2021, 2022] for m in range(1, 13)]

os.makedirs("data", exist_ok=True)

for f in months:
    r = requests.get(base + f)
    if r.status_code == 200:
        open(f"data/{f}", "wb").write(r.content)
        print(f"✅ downloaded {f}")
    else:
        print(f"⏭ skipped {f}")