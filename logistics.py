import math
import numpy as np
import copy
from itertools import permutations

# --- configuració de temps ---
VELOCIDAD_MEDIA_KMH = 60
TIEMPO_CARGA_HORAS = 0.5
TIEMPO_DESCARGA_HORAS = 0.75
JORNADA_MAX_HORAS = 8.0

# per al pes
BUFFER_SEGURIDAD = 0.93


# --- 1. funcions base ---
def haversine(lat1, lon1, lat2, lon2):
    """calcula distància en km"""
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def calcular_metricas_ruta(
    paradas_ordenadas, matadero_data, coste_km, capacidad_kg, carga_actual_kg
):
    distancia = 0
    m_lat, m_lon = matadero_data["lat"], matadero_data["lon"]
    puntos = (
        [(m_lat, m_lon)]
        + [(p.lat, p.lon) for p in paradas_ordenadas]
        + [(m_lat, m_lon)]
    )
    ruta_coords = [puntos[0]]

    for i in range(len(puntos) - 1):
        p1 = puntos[i]
        p2 = puntos[i + 1]
        distancia += haversine(p1[0], p1[1], p2[0], p2[1])
        if i < len(puntos) - 1:
            ruta_coords.append(p2)

    tiempo_conduccion = distancia / VELOCIDAD_MEDIA_KMH
    num_paradas = len(paradas_ordenadas)
    tiempo_servicio = (num_paradas * TIEMPO_CARGA_HORAS) + TIEMPO_DESCARGA_HORAS
    tiempo_total = tiempo_conduccion + tiempo_servicio

    if capacidad_kg > 0:
        ratio_carga = carga_actual_kg / capacidad_kg
    else:
        ratio_carga = 1.0

    if ratio_carga > 1.0:
        ratio_carga = 1.0

    coste_total = distancia * coste_km * ratio_carga

    return coste_total, distancia, tiempo_total, ruta_coords


# --- 2. optimitzador d'ordre (tsp) ---
def optimizar_orden_paradas(
    granjas_camion, matadero_optimo, coste_km, capacidad_kg, carga_actual_kg
):
    if len(granjas_camion) <= 1:
        coste, dist, tiempo, coords = calcular_metricas_ruta(
            granjas_camion, matadero_optimo, coste_km, capacidad_kg, carga_actual_kg
        )
        return coste, dist, tiempo, coords, granjas_camion

    mejor_distancia = float("inf")
    mejor_pack = None
    mejor_orden = None

    for perm in permutations(granjas_camion):
        coste, dist, tiempo, coords = calcular_metricas_ruta(
            perm, matadero_optimo, coste_km, capacidad_kg, carga_actual_kg
        )
        if dist < mejor_distancia:
            mejor_distancia = dist
            mejor_pack = (coste, dist, tiempo, coords)
            mejor_orden = list(perm)

    return mejor_pack[0], mejor_pack[1], mejor_pack[2], mejor_pack[3], mejor_orden


# --- 3. optimitzador de flota ---
def optimizar_flota_8h(rutas_generadas):
    rutas_ordenadas = sorted(
        rutas_generadas, key=lambda x: x["trip_time"], reverse=True
    )
    camiones = []
    for ruta in rutas_ordenadas:
        duracion = ruta["trip_time"]
        asignado = False
        for i in range(len(camiones)):
            if camiones[i] + duracion <= JORNADA_MAX_HORAS:
                camiones[i] += duracion
                ruta["real_truck_id"] = f"T-{i + 1:02d}"
                asignado = True
                break
        if not asignado:
            nuevo_id = len(camiones) + 1
            camiones.append(duracion)
            ruta["real_truck_id"] = f"T-{nuevo_id:02d}"
    return len(camiones)


# --- 4. helpers ---
def asignar_matadero_disponible(granjas_camion, df_slaughter, capacidades_restantes):
    carga_total = sum(g.num_pigs for g in granjas_camion)
    avg_lat = np.mean([g.lat for g in granjas_camion])
    avg_lon = np.mean([g.lon for g in granjas_camion])

    candidatos = df_slaughter.copy()
    candidatos["distancia"] = np.sqrt(
        (candidatos["lat"] - avg_lat) ** 2 + (candidatos["lon"] - avg_lon) ** 2
    )
    candidatos = candidatos.sort_values("distancia")

    for _, matadero in candidatos.iterrows():
        m_id = matadero["slaughterhouse_id"]
        if capacidades_restantes.get(m_id, 0) >= carga_total:
            return matadero, m_id
    return None, None


# --- 5. planificador principal ---
def planificar_rutas_dia(
    lista_granjas, df_slaughter, df_transports, capacidad_dummy=None
):
    rutas_brutas = []
    capacidades_restantes = df_slaughter.set_index("slaughterhouse_id")[
        "capacity"
    ].to_dict()

    try:
        truck_small = df_transports[df_transports["type"] == "Small"].iloc[0]
        truck_large = df_transports[df_transports["type"] == "Large"].iloc[0]
        MAX_CAPACITY_REAL = float(truck_large["capacity_kg"])
    except:
        truck_small = {"capacity_kg": 10000, "cost_km": 1.15}
        truck_large = {"capacity_kg": 15000, "cost_km": 1.25}
        MAX_CAPACITY_REAL = 15000.0

    # Apliquem el buffer de seguridad
    MAX_CAPACITY_PLANIFICACION = MAX_CAPACITY_REAL * BUFFER_SEGURIDAD
    CAPACIDAD_MAX_PIGS = 170

    pool_candidatas = [
        f for f in lista_granjas if f.num_pigs > 0 and f.mean_weight >= 105
    ]
    pool_candidatas.sort(key=lambda x: x.mean_weight, reverse=True)

    while pool_candidatas:
        semilla = pool_candidatas[0]
        peso_cerdo = semilla.mean_weight

        max_pigs_weight = int(MAX_CAPACITY_PLANIFICACION / peso_cerdo)
        max_pigs_space = CAPACIDAD_MAX_PIGS
        max_pigs_truck = min(max_pigs_weight, max_pigs_space)

        pigs_to_take = min(semilla.num_pigs, max_pigs_truck)

        parada_actual = copy.copy(semilla)
        parada_actual.num_pigs = pigs_to_take
        camion_actual = [parada_actual]

        if pigs_to_take == semilla.num_pigs:
            pool_candidatas.pop(0)
            truck_is_full = False
        else:
            semilla.num_pigs -= pigs_to_take
            truck_is_full = True

        current_weight = pigs_to_take * peso_cerdo
        current_pigs = pigs_to_take

        if not truck_is_full:
            while len(camion_actual) < 3 and pool_candidatas:
                ultima_parada = camion_actual[-1]
                distancias = []
                for idx, g in enumerate(pool_candidatas):
                    d = (g.lat - ultima_parada.lat) ** 2 + (
                        g.lon - ultima_parada.lon
                    ) ** 2
                    distancias.append((d, idx, g))

                distancias.sort(key=lambda x: x[0])
                vecino_encontrado = None
                indice_a_borrar = -1

                for _, idx, vecino in distancias:
                    peso_vecino = vecino.num_pigs * vecino.mean_weight

                    if (
                        current_weight + peso_vecino <= MAX_CAPACITY_PLANIFICACION
                    ) and (current_pigs + vecino.num_pigs <= CAPACIDAD_MAX_PIGS):
                        vecino_encontrado = vecino
                        indice_a_borrar = idx
                        break

                if vecino_encontrado:
                    camion_actual.append(vecino_encontrado)
                    current_weight += (
                        vecino_encontrado.num_pigs * vecino_encontrado.mean_weight
                    )
                    current_pigs += vecino_encontrado.num_pigs
                    pool_candidatas.pop(indice_a_borrar)
                else:
                    break

        matadero, m_id = asignar_matadero_disponible(
            camion_actual, df_slaughter, capacidades_restantes
        )

        if matadero is not None:
            capacidades_restantes[m_id] -= current_pigs
            peso_total_ruta = sum([g.mean_weight * g.num_pigs for g in camion_actual])

            if peso_total_ruta <= truck_small["capacity_kg"]:
                tipo = "Small"
                coste_km = truck_small["cost_km"]
                capacidad_camion_kg = truck_small["capacity_kg"]
            else:
                tipo = "Large"
                coste_km = truck_large["cost_km"]
                capacidad_camion_kg = truck_large["capacity_kg"]

            coste, dist, tiempo, coords, orden = optimizar_orden_paradas(
                camion_actual, matadero, coste_km, capacidad_camion_kg, peso_total_ruta
            )

            rutas_brutas.append(
                {
                    "stops": [g.farm_id for g in orden],
                    "pigs_collected": current_pigs,
                    "slaughter_id": m_id,
                    "trip_cost": round(coste, 2),
                    "trip_distance": round(dist, 2),
                    "trip_time": round(tiempo, 2),
                    "truck_type": tipo,
                    "route_coords": coords,
                }
            )

    num_camiones = optimizar_flota_8h(rutas_brutas)
    return rutas_brutas, num_camiones
