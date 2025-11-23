import pandas as pd
import numpy as np


def generate_csv_files():
    print("generant dades...")

    # configuració de llavor per reproductibilitat
    np.random.seed(42)

    # --- 1. generar camions (transports.csv) ---
    trucks_data = [
        {
            "truck_id": "T-01",
            "capacity_kg": 10000,
            "cost_per_km": 1.15,
            "maintenance_cost_daily": 50,
            "type": "Small",
        },
        {
            "truck_id": "T-02",
            "capacity_kg": 10000,
            "cost_per_km": 1.15,
            "maintenance_cost_daily": 50,
            "type": "Small",
        },
        {
            "truck_id": "T-03",
            "capacity_kg": 15000,
            "cost_per_km": 1.25,
            "maintenance_cost_daily": 80,
            "type": "Large",
        },
        {
            "truck_id": "T-04",
            "capacity_kg": 15000,
            "cost_per_km": 1.25,
            "maintenance_cost_daily": 80,
            "type": "Large",
        },
    ]
    df_trucks = pd.DataFrame(trucks_data)
    df_trucks.to_csv("transports.csv", index=False)
    print("transports generats")

    # --- configuració geogràfica ---
    LAT_MIN, LAT_MAX = 41.30, 41.90
    LON_MIN, LON_MAX = 0.30, 1.00

    # --- 2. generar escorxadors (slaughterhouses.csv) ---
    slaughter_data = [
        {"slaughterhouse_id": "M-Nord", "lat": 41.85, "lon": 0.50, "daily_limit": 2500},
        {"slaughterhouse_id": "M-Sud", "lat": 41.35, "lon": 0.60, "daily_limit": 2500},
        {"slaughterhouse_id": "M-Est", "lat": 41.60, "lon": 0.95, "daily_limit": 3000},
        {"slaughterhouse_id": "M-Oest", "lat": 41.60, "lon": 0.35, "daily_limit": 2000},
    ]
    df_slaughter = pd.DataFrame(slaughter_data)
    df_slaughter.to_csv("slaughterhouses.csv", index=False)
    print("slaughterhouses generats")

    # --- 3. generar granges (farms.csv) ---
    # augmentem lleugerament el nombre de granges per assegurar rutes
    num_farms = 60
    farms_data = []

    for i in range(1, num_farms + 1):
        f_id = f"F-{i:03d}"
        lat = np.random.uniform(LAT_MIN, LAT_MAX)
        lon = np.random.uniform(LON_MIN, LON_MAX)

        # més porcs per granja per omplir camions ràpid
        pigs = np.random.randint(500, 1200)

        # --- lògica de pesos optimitzada per beneficis ---
        # evitem porcs molt petits (<80kg) que generen despesa però no venda

        rand_val = np.random.rand()

        if rand_val < 0.35:
            # grup 1: venda immediata (setmana 1)
            # porcs que ja estan llestos o gairebé llestos
            # generen cash flow positiu els primers dies
            avg_weight = np.random.uniform(105.0, 115.0)

        elif rand_val < 0.75:
            # grup 2: venda a mig termini (setmana 2)
            # pesen ~95kg. en 7-10 dies arribaran a 105kg
            # aquests ompliran els camions la segona setmana
            avg_weight = np.random.uniform(92.0, 102.0)

        else:
            # grup 3: el "boom" del dia 15
            # pesen ~85kg
            # creixen durant 14 dies x ~0.9kg/dia = +12.6kg
            # 85 + 12.6 = 97.6kg (gairebé llestos)
            # 88 + 12.6 = 100.6kg (llestos per penalització baixa)
            # això assegura que hi hagi estoc, però minimitza el temps de "només menjar"
            avg_weight = np.random.uniform(85.0, 91.0)

        # càlcul edat
        age_weeks = int(avg_weight / 5.6) + 3

        farms_data.append(
            {
                "farm_id": f_id,
                "lat": round(lat, 5),
                "lon": round(lon, 5),
                "pigs": pigs,
                "age_weeks": age_weeks,
                "avg_weight": round(avg_weight, 2),
            }
        )

    df_farms = pd.DataFrame(farms_data)
    df_farms.to_csv("farms.csv", index=False)
    print(f"{len(df_farms)} farms generades")
    print("dades optimitzades generades correctament")


if __name__ == "__main__":
    generate_csv_files()
