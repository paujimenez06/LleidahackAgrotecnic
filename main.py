import sqlite3
import pandas as pd
import numpy as np
from scipy.stats import norm

# --- importació crítica d'optimització ---
from logistics import planificar_rutas_dia

# --- configuració ---
DB_PATH = "logistics.db"
# paràmetres econòmics corregits
PRECIO_BASE_KG = 1.56
PESO_INICIAL_LECHON = 20.0
COST_PIENSO_KG_GAIN = 1.10
COST_MANTENIMIENTO_DIA = 0.05
PESO_OBJETIVO = 105.0
RANGO_IDEAL = (105, 115)
RANGO_PENALIZACION_15_LOW = (100, 105)
RANGO_PENALIZACION_15_HIGH = (115, 120)


# --- funcions de càrrega i pre-processament de dades de referència ---


def load_and_process_references():
    """carrega i calcula taules de creixement i consum diari des de fitxers csv"""

    # --- 1. carregar guany de pes (weight) ---
    df_weight = pd.read_csv("weight.csv", delimiter=";", decimal=",")
    df_weight.columns = df_weight.columns.str.strip()

    weight_cols = [
        col for col in df_weight.columns if "Weight" in col or "weight" in col
    ]
    if len(weight_cols) == 0:
        if df_weight.shape[1] < 2:
            raise ValueError("l'arxiu weight.csv no té suficients columnes")
        df_weight.columns = ["age_weeks", "weight"] + list(df_weight.columns[2:])
        weight_col = "weight"
        week_col = "age_weeks"
    else:
        weight_col = weight_cols[0]
        week_col = "Week" if "Week" in df_weight.columns else df_weight.columns[0]

    df_weight.rename(
        columns={week_col: "age_weeks", weight_col: "weight"}, inplace=True
    )
    df_weight.set_index("age_weeks", inplace=True)

    if len(df_weight) <= 1:
        raise ValueError(
            "l'arxiu weight.csv ha de contenir almenys dues files de dades"
        )

    growth_dict = {}
    for i in range(len(df_weight) - 1):
        w_curr = df_weight.iloc[i]["weight"]
        w_next = df_weight.iloc[i + 1]["weight"]
        week = df_weight.index[i]
        growth_dict[week] = (w_next - w_curr) / 7.0

    # --- 2. carregar consum de pinso (consumption) ---
    df_cons = pd.read_csv("consumption1.csv", delimiter=";", decimal=",")
    df_cons.columns = df_cons.columns.str.strip()

    consumption_cols = [
        col for col in df_cons.columns if "consumtion" in col or "intake" in col
    ]

    if len(consumption_cols) == 0:
        if df_cons.shape[1] < 2:
            raise ValueError("l'arxiu consumption1.csv no té suficients columnes")

        df_cons.columns = ["age_weeks", "cumulative_intake"] + list(df_cons.columns[2:])
        consumption_col = "cumulative_intake"
        week_col_cons = "age_weeks"
    else:
        consumption_col = consumption_cols[0]
        week_col_cons = "Week" if "Week" in df_cons.columns else df_cons.columns[0]

    df_cons.rename(
        columns={week_col_cons: "age_weeks", consumption_col: "cumulative_intake"},
        inplace=True,
    )
    df_cons.set_index("age_weeks", inplace=True)

    if len(df_cons) <= 1:
        raise ValueError(
            "l'arxiu consumption1.csv ha de contenir almenys dues files de dades"
        )

    consumption_dict = {}
    for i in range(len(df_cons) - 1):
        accum_curr = df_cons.iloc[i]["cumulative_intake"]
        accum_next = df_cons.iloc[i + 1]["cumulative_intake"]
        week = df_cons.index[i]
        weekly_intake = accum_next - accum_curr
        consumption_dict[week] = weekly_intake / 7.0

    max_growth = list(growth_dict.values())[-1]
    max_consumption = list(consumption_dict.values())[-1]

    return growth_dict, consumption_dict, max_growth, max_consumption


class FarmBatch:
    def __init__(
        self,
        batch_id,
        farm_id,
        num_pigs,
        growth_ref,
        consumption_ref,
        max_growth,
        max_consumption,
        current_weight=None,
    ):
        self.batch_id = batch_id
        self.farm_id = farm_id
        self.num_pigs = num_pigs
        self.growth_ref = growth_ref
        self.consumption_ref = consumption_ref
        self.max_growth = max_growth
        self.max_consumption = max_consumption

        if current_weight is None:
            self.mean_weight = np.random.uniform(
                PESO_INICIAL_LECHON, PESO_OBJETIVO - 10
            )
        else:
            self.mean_weight = current_weight

        daily_growth_proxy = growth_ref.get(10, 0.95)

        self.age_weeks = int(self.mean_weight / (daily_growth_proxy * 7))
        if self.age_weeks < 1:
            self.age_weeks = 1

        self.std_dev = self.mean_weight * 0.10

    def get_daily_growth(self):
        current_week = int(self.age_weeks)
        return self.growth_ref.get(current_week, self.max_growth)

    def get_daily_consumption(self):
        current_week = int(self.age_weeks)
        return self.consumption_ref.get(current_week, self.max_consumption)

    def grow_one_day(self):
        if self.num_pigs <= 0:
            return 0

        stat_gain = self.get_daily_growth()

        gain = np.random.normal(stat_gain, 0.1)
        if gain < 0:
            gain = 0.05

        self.mean_weight += gain
        self.std_dev = self.mean_weight * 0.10

        self.age_weeks += 1 / 7

        daily_intake = self.get_daily_consumption()
        pienso_cost = daily_intake * COST_PIENSO_KG_GAIN * self.num_pigs

        maintenance_cost = COST_MANTENIMIENTO_DIA * self.num_pigs

        return pienso_cost + maintenance_cost

    @property
    def is_ready(self):
        return self.mean_weight >= PESO_OBJETIVO and self.num_pigs > 0

    def select_pigs_by_weight(self, max_amount):
        if self.num_pigs <= 0 or max_amount <= 0:
            return 0, 0, 0

        n_pigs = min(self.num_pigs, max_amount)
        mu = self.mean_weight
        sigma = self.std_dev

        current_distribution = norm.rvs(loc=mu, scale=sigma, size=self.num_pigs)
        selected_pigs_weights = np.sort(current_distribution)[-n_pigs:]

        removed_kg = selected_pigs_weights.sum()
        avg_removed_weight = removed_kg / n_pigs

        self.num_pigs -= n_pigs

        return n_pigs, removed_kg, avg_removed_weight


def calculate_penalized_income_corrected(
    kg_total, num_pigs, avg_extracted_weight, price_per_kg=PRECIO_BASE_KG
):
    if num_pigs == 0:
        return 0

    avg_weight = avg_extracted_weight
    penalty = 0.0

    if RANGO_IDEAL[0] <= avg_weight <= RANGO_IDEAL[1]:
        penalty = 0.0
    elif (
        RANGO_PENALIZACION_15_LOW[0] <= avg_weight < RANGO_PENALIZACION_15_LOW[1]
    ) or (RANGO_PENALIZACION_15_HIGH[0] < avg_weight <= RANGO_PENALIZACION_15_HIGH[1]):
        penalty = 0.15
    else:
        penalty = 0.20

    return kg_total * price_per_kg * (1.0 - penalty)


class LogisticsFarmProxy:
    def __init__(self, farm_id, num_pigs, mean_weight, lat, lon):
        self.farm_id = farm_id
        self.num_pigs = num_pigs
        self.mean_weight = mean_weight
        self.lat = lat
        self.lon = lon


# --- funció de simulació principal ---
def run_simulation():
    growth_ref, consumption_ref, max_growth, max_consumption = (
        load_and_process_references()
    )

    conn = sqlite3.connect(DB_PATH)

    df_farms_db = pd.read_sql("SELECT * FROM farms", conn)
    df_slaughter = pd.read_sql("SELECT * FROM slaughterhouses", conn)
    df_trucks = pd.read_sql("SELECT * FROM trucks", conn)

    df_trucks.columns = df_trucks.columns.str.strip()
    if len(df_trucks.columns) == 4:
        df_trucks.columns = [
            "truck_id",
            "capacity_kg",
            "cost_per_km",
            "maintenance_cost_daily",
        ]

    # init batches
    batches = []
    batch_counter = 0
    farm_last_visit = {fid: -7 for fid in df_farms_db["farm_id"]}

    for _, farm in df_farms_db.iterrows():
        farm_pigs = farm.get("pigs", 300)
        farm_avg_weight = farm.get("avg_weight", 80.0)

        # lot 1 (més pesat)
        pigs_b1 = int(farm_pigs * 0.3)
        weight_b1 = farm_avg_weight * np.random.uniform(1.05, 1.15)
        if weight_b1 > PESO_OBJETIVO * 1.1:
            weight_b1 = PESO_OBJETIVO * 1.1

        batch_counter += 1
        b1 = FarmBatch(
            f"B-{batch_counter}",
            farm["farm_id"],
            pigs_b1,
            growth_ref,
            consumption_ref,
            max_growth,
            max_consumption,
            current_weight=weight_b1,
        )
        batches.append(b1)

        # lot 2 (més lleuger)
        pigs_b2 = farm_pigs - pigs_b1
        weight_b2 = farm_avg_weight * np.random.uniform(0.75, 0.95)

        batch_counter += 1
        b2 = FarmBatch(
            f"B-{batch_counter}",
            farm["farm_id"],
            pigs_b2,
            growth_ref,
            consumption_ref,
            max_growth,
            max_consumption,
            current_weight=weight_b2,
        )
        batches.append(b2)

    simulation_results = []
    routes_results = []

    FIXED_COST_WEEKLY = 2000.0
    # inicialitzem per a 3 setmanes per suportar el dia 15
    max_trucks_used_per_week = {1: 0, 2: 0, 3: 0}

    # configuració de flota
    try:
        id_small = df_trucks[df_trucks["capacity_kg"] == 10000].iloc[0]["truck_id"]
    except:
        id_small = "T-Small"

    try:
        id_large = df_trucks[df_trucks["capacity_kg"] > 12000].iloc[0]["truck_id"]
    except:
        id_large = "T-Large"

    df_transports_logistics = pd.DataFrame(
        [
            {
                "type": "Small",
                "capacity_kg": 10000,
                "cost_km": 1.15,
            },
            {
                "type": "Large",
                "capacity_kg": 15000,  # 15t
                "cost_km": 1.25,
            },
        ]
    )

    truck_map = {
        "Small": id_small,
        "Large": id_large,
    }
    capacity_map = {
        "Small": 10000,
        "Large": 15000,
    }

    print("iniciant simulació ...")

    # bucle dels 15 dies
    for dia in range(1, 16):
        daily_pienso_cost = 0
        daily_slaughter_usage = {sid: 0 for sid in df_slaughter["slaughterhouse_id"]}

        # --- 1. creixement (ocorre tots els dies) ---
        for b in batches:
            daily_pienso_cost += b.grow_one_day()
            if b.num_pigs <= 0:
                b.num_pigs = np.random.randint(100, 200)
                b.mean_weight = PESO_INICIAL_LECHON

        # --- determinar si és dia laborable ---
        dia_semana = (dia - 1) % 7 + 1
        es_dia_laborable = dia_semana <= 5  # true per dilluns-divendres

        rutas_optimizadas = []
        num_camiones_usados = 0

        # només executem logística si és dia laborable
        if es_dia_laborable:
            # 2. estat granges (pre-planificació)
            farm_status = {}
            for b in batches:
                if b.is_ready:
                    if b.farm_id not in farm_status:
                        farm_status[b.farm_id] = {
                            "ready_pigs": 0,
                            "batches": [],
                            "avg_weight": 0,
                        }
                    farm_status[b.farm_id]["batches"].append(b)

            for fid, data in farm_status.items():
                ready_batches = [
                    b_i
                    for b_i in data["batches"]
                    if b_i.is_ready and b_i.farm_id == fid
                ]
                total_ready_pigs = sum(b_i.num_pigs for b_i in ready_batches)

                if total_ready_pigs > 0:
                    total_ready_weight = sum(
                        b_i.num_pigs * b_i.mean_weight for b_i in ready_batches
                    )
                    farm_status[fid]["ready_pigs"] = total_ready_pigs
                    farm_status[fid]["avg_weight"] = (
                        total_ready_weight / total_ready_pigs
                    )
                else:
                    farm_status[fid]["ready_pigs"] = 0
                    farm_status[fid]["avg_weight"] = 0

            # 3. preparació de logística
            eligible_farms_objects = []
            for fid, data in farm_status.items():
                last_visit = farm_last_visit.get(fid, -7)

                if (dia - last_visit) >= 7 and data["ready_pigs"] > 15:
                    farm_info = df_farms_db[df_farms_db["farm_id"] == fid].iloc[0]
                    eligible_farms_objects.append(
                        LogisticsFarmProxy(
                            farm_id=fid,
                            num_pigs=data["ready_pigs"],
                            mean_weight=data["avg_weight"],
                            lat=farm_info["lat"],
                            lon=farm_info["lon"],
                        )
                    )

            # 4. crida a l'optimitzador
            df_slaughter_logistics = df_slaughter.copy()
            df_slaughter_logistics.rename(
                columns={"daily_limit": "capacity"}, inplace=True
            )

            rutas_optimizadas, num_camiones_usados = planificar_rutas_dia(
                eligible_farms_objects, df_slaughter_logistics, df_transports_logistics
            )

        # actualitzar màxim de camions (incloent setmana 3 per al dia 15)
        semana_actual = (dia - 1) // 7 + 1
        current_max = max_trucks_used_per_week.get(semana_actual, 0)
        max_trucks_used_per_week[semana_actual] = max(current_max, num_camiones_usados)

        # 5. processar resultats
        total_income_day = 0
        total_pigs_day = 0

        for ruta in rutas_optimizadas:
            pigs_to_extract = ruta["pigs_collected"]
            kg_removed = 0

            batches_to_extract_from = []
            for fid in ruta["stops"]:
                farm_last_visit[fid] = dia
                f_data = farm_status.get(fid)
                if f_data:
                    batches_to_extract_from.extend(
                        sorted(
                            [b for b in batches if b.is_ready and b.farm_id == fid],
                            key=lambda x: x.mean_weight,
                            reverse=True,
                        )
                    )

            unique_batches_to_extract = []
            seen_ids = set()
            for b in batches_to_extract_from:
                if b.batch_id not in seen_ids:
                    unique_batches_to_extract.append(b)
                    seen_ids.add(b.batch_id)

            for batch in unique_batches_to_extract:
                if pigs_to_extract > 0 and batch.num_pigs > 0:
                    n_extracted, k_extracted, _ = batch.select_pigs_by_weight(
                        pigs_to_extract
                    )

                    if n_extracted > 0:
                        pigs_to_extract -= n_extracted
                        kg_removed += k_extracted

            if ruta["pigs_collected"] > 0:
                avg_removed_weight_route = kg_removed / ruta["pigs_collected"]
            else:
                avg_removed_weight_route = 0

            income = calculate_penalized_income_corrected(
                kg_removed, ruta["pigs_collected"], avg_removed_weight_route
            )

            total_income_day += income
            total_pigs_day += ruta["pigs_collected"]
            daily_slaughter_usage[ruta["slaughter_id"]] += ruta["pigs_collected"]

            truck_type = ruta["truck_type"]
            routes_results.append(
                {
                    "dia": dia,
                    "truck_id": truck_map.get(truck_type, "N/A"),
                    "cerdos": ruta["pigs_collected"],
                    "pes_total": round(kg_removed, 2),
                    "distancia_km": ruta["trip_distance"],
                    "coste_ruta": ruta["trip_cost"],
                    "n_granges_visitades": len(ruta["stops"]),
                    "truck_capacity": capacity_map.get(truck_type, 0),
                    "load_factor": round(
                        kg_removed / capacity_map.get(truck_type, 1)
                        if capacity_map.get(truck_type, 1) > 0
                        else 0,
                        4,
                    ),
                    "lat_origen": ruta["route_coords"][0][0],
                    "lon_origen": ruta["route_coords"][0][1],
                    "lat_dest": ruta["route_coords"][-1][0],
                    "lon_dest": ruta["route_coords"][-1][1],
                    "slaughterhouse_id": ruta["slaughter_id"],
                    "stops_str": ",".join(ruta["stops"]),
                }
            )

        # 6. resultats diaris
        daily_fixed_cost = 0
        daily_variable_routes = sum(
            [r["coste_ruta"] for r in routes_results if r["dia"] == dia]
        )

        beneficio_neto = (
            total_income_day
            - daily_pienso_cost
            - daily_fixed_cost
            - daily_variable_routes
        )

        simulation_results.append(
            {
                "dia": dia,
                "beneficio_bruto": round(beneficio_neto, 2),
                "cerdos_entregados": total_pigs_day,
                "coste_pienso": round(daily_pienso_cost, 2),
                "coste_alquiler": round(daily_fixed_cost, 2),
            }
        )

        tipo_dia = "laborable" if es_dia_laborable else "descans"
        print(
            f"dia {dia} ({tipo_dia}): {total_pigs_day} porcs. benefici: {beneficio_neto:,.0f} eur"
        )

    # post-processament: aplicar cost fix setmanal real
    total_coste_fijo_alquiler = 0

    max_camiones_s1 = max_trucks_used_per_week[1]
    coste_fijo_s1 = max_camiones_s1 * FIXED_COST_WEEKLY
    total_coste_fijo_alquiler += coste_fijo_s1

    max_camiones_s2 = max_trucks_used_per_week[2]
    coste_fijo_s2 = max_camiones_s2 * FIXED_COST_WEEKLY
    total_coste_fijo_alquiler += coste_fijo_s2

    # cost setmana 3 (només pel dia 15, però es paga la setmana si s'activa)
    max_camiones_s3 = max_trucks_used_per_week.get(3, 0)
    coste_fijo_s3 = max_camiones_s3 * FIXED_COST_WEEKLY
    total_coste_fijo_alquiler += coste_fijo_s3

    DIAS_SIMULADOS = 15
    coste_fijo_diario_ajustado = total_coste_fijo_alquiler / DIAS_SIMULADOS

    print("\n--- resum cost fix ---")
    print(f"màxim camions s1: {max_camiones_s1}")
    print(f"màxim camions s2: {max_camiones_s2}")
    print(f"màxim camions s3: {max_camiones_s3}")
    print(f"cost fix total: {total_coste_fijo_alquiler:,.2f} eur")

    for res in simulation_results:
        res["coste_alquiler"] = round(coste_fijo_diario_ajustado, 2)
        res["beneficio_bruto"] = round(
            res["beneficio_bruto"] - coste_fijo_diario_ajustado, 2
        )

    pd.DataFrame(simulation_results).to_sql(
        "results_simulation", conn, if_exists="replace", index=False
    )

    ROUTE_COLUMNS = [
        "dia",
        "truck_id",
        "cerdos",
        "pes_total",
        "distancia_km",
        "coste_ruta",
        "n_granges_visitades",
        "truck_capacity",
        "load_factor",
        "lat_origen",
        "lon_origen",
        "lat_dest",
        "lon_dest",
        "slaughterhouse_id",
        "stops_str",
    ]
    df_routes = pd.DataFrame(routes_results, columns=ROUTE_COLUMNS)

    df_routes.to_sql("results_routes", conn, if_exists="replace", index=False)

    final_status = []
    for fid in df_farms_db["farm_id"].unique():
        total = sum([b.num_pigs for b in batches if b.farm_id == fid])
        waiting = sum([b.num_pigs for b in batches if b.farm_id == fid and b.is_ready])
        final_status.append(
            {"farm_id": fid, "inventory": total, "pigs_waiting": waiting}
        )

    df_stat = pd.DataFrame(final_status)
    df_m = pd.merge(
        df_farms_db[["farm_id", "lat", "lon"]], df_stat, on="farm_id", how="left"
    ).fillna(0)

    df_m["inventory"] = df_m["inventory"].astype(int)
    df_m["pigs_waiting"] = df_m["pigs_waiting"].astype(int)

    df_m.to_sql("farms", conn, if_exists="replace", index=False)

    conn.close()
    print("simulació finalitzada")


if __name__ == "__main__":
    run_simulation()
