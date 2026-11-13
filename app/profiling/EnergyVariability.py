"""Statistiche descrittive della variabilita' energetica, senza nuove misure."""

import math
import statistics


def sample_statistics(values) -> dict:
    """Calcola la varianza *campionaria* (n-1) senza trasformare i dati mancanti in zero.

    I valori sono osservazioni descrittive: campioni Scaphandre successivi
    possono essere temporalmente correlati, quindi la varianza non e' un
    intervallo di confidenza ne' una stima dell'incertezza del sensore.
    """
    samples = [float(value) for value in values
               if isinstance(value, (int, float)) and not isinstance(value, bool)
               and math.isfinite(value)]
    count = len(samples)
    variance = statistics.variance(samples) if count >= 2 else None
    return {
        "count": count,
        "mean": statistics.mean(samples) if samples else None,
        "sample_variance": variance,
        "sample_std": math.sqrt(variance) if variance is not None else None,
        "min": min(samples) if samples else None,
        "max": max(samples) if samples else None,
    }


def _point(row: dict, name: str, key: str):
    return (row.get(name) or {}).get(key)


def _average_rapl_power(row: dict, name: str):
    point = row.get(name) or {}
    joules = point.get("rapl_package_gross_j")
    duration = point.get("elapsed_seconds")
    if (isinstance(joules, (float, int)) and isinstance(duration, (float, int))
            and math.isfinite(joules) and math.isfinite(duration) and duration > 0):
        return joules / duration
    return None


def summarize_energy_variability(rows: list[dict], *, scaphandre_key: str) -> dict:
    """Separa la variabilità entro una finestra da quella tra repliche.

    E1 ed EN sono serie distinte: la varianza di due valori relativi a
    quantità di lavoro diverse non rappresenta la variabilita' energetica.
    """
    series = {
        "rapl_one_gross_j": [_point(row, "one", "rapl_package_gross_j") for row in rows],
        "rapl_many_gross_j": [_point(row, "many", "rapl_package_gross_j") for row in rows],
        "rapl_one_incremental_j": [_point(row, "one", "rapl_package_incremental_estimate_j") for row in rows],
        "rapl_many_incremental_j": [_point(row, "many", "rapl_package_incremental_estimate_j") for row in rows],
        "rapl_one_mean_power_w": [_average_rapl_power(row, "one") for row in rows],
        "rapl_many_mean_power_w": [_average_rapl_power(row, "many") for row in rows],
        "rapl_incremental_j_per_invocation": [row.get("rapl_incremental_j_per_invocation") for row in rows],
        "rapl_gross_j_per_invocation": [row.get("rapl_gross_j_per_invocation") for row in rows],
        "scaphandre_attributed_j_per_invocation": [row.get("scaphandre_attributed_j_per_invocation") for row in rows],
        "scaphandre_observed_mean_power_w": [
            (row.get(scaphandre_key) or {}).get("observed_mean_power_w")
            if scaphandre_key == "scaphandre_separate_run"
            else ((row.get("many") or {}).get("scaphandre") or {}).get("observed_mean_power_w")
            for row in rows
        ],
        "rapl_idle_before_w": [(row.get("idle_bracketing") or {}).get("before_one_w") for row in rows],
        "rapl_idle_between_w": [(row.get("idle_bracketing") or {}).get("between_one_and_n_w") for row in rows],
        "rapl_idle_after_w": [(row.get("idle_bracketing") or {}).get("after_n_w") for row in rows],
    }
    idle_all = []
    within = []
    for row in rows:
        idle = row.get("idle_bracketing") or {}
        idle_readings = [idle.get(key) for key in
                         ("before_one_w", "between_one_and_n_w", "after_n_w")]
        idle_all.extend(idle_readings)
        scaph = ((row.get("scaphandre_separate_run") or {})
                 if scaphandre_key == "scaphandre_separate_run"
                 else ((row.get("many") or {}).get("scaphandre") or {}))
        within.append({
            "repetition": row.get("repetition"),
            "rapl_idle_three_points_w": sample_statistics(idle_readings),
            # La varianza temporale delle potenze viene calcolata dal sampler,
            # senza aggiungere polling RAPL all'interno della finestra EN.
            "scaphandre_power_samples_w": scaph.get("power_sample_statistics_w"),
            "scaphandre_status": scaph.get("status"),
        })
    series["rapl_idle_all_points_w"] = idle_all
    return {
        "definition": "Varianza campionaria (n-1), descrittiva; E1 ed EN non sono mescolati.",
        "across_repetitions": {key: sample_statistics(values) for key, values in series.items()},
        "within_repetitions": within,
    }


def energy_variability_lines(report: dict) -> list[str]:
    """Formatta un riepilogo terminale per tutti i percorsi di profilazione."""
    languages = report.get("energy_variability") or {}
    lines = []
    keys = (
        ("rapl_incremental_j_per_invocation", "RAPL incrementale", "J^2/chiamata^2"),
        ("rapl_gross_j_per_invocation", "RAPL lordo", "J^2/chiamata^2"),
        ("scaphandre_attributed_j_per_invocation", "Scaphandre attribuito", "J^2/chiamata^2"),
        ("rapl_one_gross_j", "RAPL E1 lordo", "J^2"),
        ("rapl_many_gross_j", "RAPL EN lordo", "J^2"),
        ("rapl_one_incremental_j", "RAPL E1 incrementale", "J^2"),
        ("rapl_many_incremental_j", "RAPL EN incrementale", "J^2"),
        ("rapl_one_mean_power_w", "RAPL potenza media E1", "W^2"),
        ("rapl_many_mean_power_w", "RAPL potenza media EN", "W^2"),
        ("rapl_idle_all_points_w", "RAPL idle, tutti i punti", "W^2"),
        ("scaphandre_observed_mean_power_w", "Scaphandre potenza media EN", "W^2"),
    )

    def number(value):
        return f"{value:.6g}" if value is not None else "N/D"

    for language, dataset in languages.items():
        series = dataset.get("across_repetitions", {})
        lines.append(f"VARIABILITA' ENERGETICA {language.upper()} (varianza campionaria, n-1):")
        for key, label, unit in keys:
            stats = series.get(key) or {}
            lines.append(f"  {label}: n={stats.get('count', 0)}, "
                         f"var={number(stats.get('sample_variance'))} {unit}, "
                         f"dev.std={number(stats.get('sample_std'))}")
        for rep in dataset.get("within_repetitions", []):
            idle = rep.get("rapl_idle_three_points_w") or {}
            power = rep.get("scaphandre_power_samples_w") or {}
            lines.append(
                f"  Replica {rep.get('repetition')}: "
                f"varianza RAPL idle 3 punti={number(idle.get('sample_variance'))} W^2; "
                f"varianza temporale Scaphandre="
                f"{number(power.get('sample_variance'))} W^2 "
                f"(n={power.get('count', 0)}, stato={rep.get('scaphandre_status')})")
    return lines
