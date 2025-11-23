import sqlite3
import pandas as pd
import os


def init_db_from_csv():
    DB_PATH = "logistics.db"
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # --------------------- 1. NETEJA ---------------------------
    print("Netejant taules...")
    cursor.execute("DROP TABLE IF EXISTS farms")
    cursor.execute("DROP TABLE IF EXISTS slaughterhouses")
    cursor.execute("DROP TABLE IF EXISTS trucks")
    cursor.execute("DROP TABLE IF EXISTS results_simulation")
    cursor.execute("DROP TABLE IF EXISTS results_routes")

    cursor.execute("""
    CREATE TABLE results_simulation (
        dia INTEGER,
        beneficio_bruto REAL,
        cerdos_entregados INTEGER,
        coste_pienso REAL,
        coste_alquiler REAL
    )
    """)
    cursor.execute("""
    CREATE TABLE results_routes (
        route_id INTEGER PRIMARY KEY AUTOINCREMENT,
        dia INTEGER,
        truck_id TEXT,
        cerdos INTEGER,
        pes_total REAL,
        distancia_km REAL,
        coste_ruta REAL,
        n_granges_visitades INTEGER,
        truck_capacity REAL,
        load_factor REAL,
        lat_origen REAL,
        lon_origen REAL,
        lat_dest REAL,
        lon_dest REAL,
        slaughterhouse_id TEXT
    )
    """)

    # ------------------- 2. DATOS DESDE CSV ----------------------------------

    # definicio d'arxius esperats
    files_to_load = {
        "farms.csv": "farms",
        "slaughterhouses.csv": "slaughterhouses",
        "transports.csv": "trucks",
    }

    for filename, table_name in files_to_load.items():
        if not os.path.exists(filename):
            print(f" ERROR: no se troba el arxiu '{filename}'")
            conn.close()
            return
        try:
            # llegir el .csv
            df = pd.read_csv(filename)
            # guardar amb SQL
            df.to_sql(table_name, conn, if_exists="append", index=False)
            print(f"cargat '{filename}' en taula '{table_name}'. Files: {len(df)}")

        except Exception as e:
            print(f"ERROR al procesar '{filename}': {e}")
            conn.close()
            return

    conn.commit()
    conn.close()
    print("carga de dades finalitzada")


if __name__ == "__main__":
    init_db_from_csv()
