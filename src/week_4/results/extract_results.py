import pandas as pd
import os
import json
from datetime import datetime


def get_all_historical_runs():
    """
    Haalt ALLE Ray training runs op uit het verleden, ook afgebroken runs.
    Gebaseerd op het principe: alle data is waardevol voor analyse.
    """

    # Lijst voor alle runs (succesvol, afgebroken, gefaald)
    all_runs = []

    # Ray results directory
    ray_results_dir = os.path.expanduser("~/ray_results")

    print(f"Scanning alle historische runs in: {ray_results_dir}")

    if not os.path.exists(ray_results_dir):
        print("Geen Ray results directory gevonden.")
        return pd.DataFrame()

    # Loop door alle experiment directories
    for experiment_name in os.listdir(ray_results_dir):
        experiment_path = os.path.join(ray_results_dir, experiment_name)

        if not os.path.isdir(experiment_path):
            continue

        print(f"Processing experiment: {experiment_name}")

        # Loop door alle trial directories
        for trial_name in os.listdir(experiment_path):
            trial_path = os.path.join(experiment_path, trial_name)

            if not os.path.isdir(trial_path):
                continue

            try:
                # Basis trial informatie
                trial_data = {
                    "experiment_name": experiment_name,
                    "trial_name": trial_name,
                    "trial_path": trial_path,
                    "status": "unknown",
                    "has_params": False,
                    "has_results": False,
                    "last_update": None,
                }

                # Check for verschillende bestanden
                params_file = os.path.join(trial_path, "params.json")
                result_file = os.path.join(trial_path, "result.json")
                progress_file = os.path.join(trial_path, "progress.csv")

                # Lees parameters (hyperparameters)
                if os.path.exists(params_file):
                    trial_data["has_params"] = True
                    with open(params_file, "r") as f:
                        params = json.load(f)
                        for key, value in params.items():
                            trial_data[f"param_{key}"] = value

                # Lees resultaten (ook van afgebroken runs)
                if os.path.exists(result_file):
                    trial_data["has_results"] = True
                    with open(result_file, "r") as f:
                        lines = f.readlines()

                        if lines:
                            # Lees ALLE results, niet alleen de laatste
                            first_result = json.loads(lines[0])
                            last_result = json.loads(lines[-1])

                            # Bepaal status
                            if last_result.get("done", False):
                                trial_data["status"] = "completed"
                            elif "error" in last_result:
                                trial_data["status"] = "failed"
                            else:
                                trial_data["status"] = (
                                    "stopped"  # Waarschijnlijk afgebroken
                                )

                            # Voeg metrics toe van laatste bekende staat
                            for key, value in last_result.items():
                                if key not in ["trial_id", "experiment_id"]:
                                    trial_data[f"metric_{key}"] = value

                            # Voeg aantal epochs/iterations toe
                            trial_data["total_iterations"] = len(lines)
                            trial_data["first_timestamp"] = first_result.get(
                                "timestamp", None
                            )
                            trial_data["last_timestamp"] = last_result.get(
                                "timestamp", None
                            )

                # Check progress.csv voor extra info
                if os.path.exists(progress_file):
                    try:
                        progress_df = pd.read_csv(progress_file)
                        if not progress_df.empty:
                            trial_data["progress_rows"] = len(progress_df)
                            # Voeg laatste bekende waarden toe
                            last_row = progress_df.iloc[-1]
                            for col in progress_df.columns:
                                if col not in trial_data:
                                    trial_data[f"progress_{col}"] = last_row[col]
                    except (
                        pd.errors.EmptyDataError,
                        pd.errors.ParserError,
                        FileNotFoundError,
                        PermissionError,
                    ) as e:
                        print(f"    Kan progress.csv niet lezen: {e}")

                # File timestamps voor wanneer run laatst geupdate was
                if os.path.exists(trial_path):
                    mod_time = os.path.getmtime(trial_path)
                    trial_data["last_update"] = datetime.fromtimestamp(mod_time)

                all_runs.append(trial_data)
                status_emoji = {
                    "completed": "✓",
                    "failed": "✗",
                    "stopped": "⏸",
                    "unknown": "?",
                }
                print(
                    f"  {status_emoji.get(trial_data['status'], '?')} {trial_name} - {trial_data['status']}"
                )

            except Exception as e:
                print(f"  ⚠ Error processing {trial_name}: {str(e)}")
                # Voeg minimale info toe zelfs bij errors
                all_runs.append(
                    {
                        "experiment_name": experiment_name,
                        "trial_name": trial_name,
                        "trial_path": trial_path,
                        "status": "error",
                        "error_message": str(e),
                    }
                )

    # Maak DataFrame
    if all_runs:
        df = pd.DataFrame(all_runs)
        print(f"\nGevonden: {len(df)} totale runs")

        # Print status overzicht
        status_counts = df["status"].value_counts()
        print("Status overzicht:")
        for status, count in status_counts.items():
            print(f"  {status}: {count}")

        return df
    else:
        print("Geen runs gevonden.")
        return pd.DataFrame()


def analyze_historical_data():
    """
    Hoofdfunctie om alle historische data te analyseren en opslaan.
    """
    print("=== Ray Historical Data Extractor ===")
    print("Haalt ALLE runs op, ook afgebroken en gefaalde runs\n")

    # Haal alle data op
    df = get_all_historical_runs()

    if df.empty:
        print("Geen data gevonden.")
        return None

    # Sla op naar CSV
    filename = f"ray_all_runs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)
    print(f"\n📁 Data opgeslagen als: {filename}")

    # Basis statistieken
    print("\n📊 Dataset info:")
    print(f"   Totaal runs: {len(df)}")
    print(f"   Totaal columns: {len(df.columns)}")
    print(f"   Experimenten: {df['experiment_name'].nunique()}")

    # Toon eerste paar rijen
    print("\n🔍 Eerste 3 runs:")
    display_cols = [
        "experiment_name",
        "trial_name",
        "status",
        "has_params",
        "has_results",
    ]
    available_cols = [col for col in display_cols if col in df.columns]
    print(df[available_cols].head(3).to_string(index=False))

    return df


def filter_by_status(df, status_list=["completed", "stopped"]):
    """
    Filter DataFrame op specifieke statussen.

    Args:
        df: DataFrame met alle runs
        status_list: Lijst van gewenste statussen
    """
    if df.empty:
        return df

    filtered = df[df["status"].isin(status_list)]
    print(f"Gefilterd op {status_list}: {len(filtered)} van {len(df)} runs")
    return filtered


def get_experiments_summary(df):
    """
    Geeft overzicht per experiment.
    """
    if df.empty:
        return pd.DataFrame()

    summary = (
        df.groupby("experiment_name")
        .agg(
            {
                "trial_name": "count",
                "status": lambda x: x.value_counts().to_dict(),
                "last_update": "max",
            }
        )
        .rename(columns={"trial_name": "total_trials"})
    )

    return summary


# Gebruik voorbeelden
if __name__ == "__main__":
    # Haal alle historische data op
    all_data = analyze_historical_data()

    if all_data is not None:
        print("\n" + "=" * 50)
        print("EXTRA ANALYSES:")

        # Filter alleen nuttige runs (completed + stopped)
        useful_runs = filter_by_status(all_data, ["completed", "stopped"])
        if not useful_runs.empty:
            useful_runs.to_csv("ray_useful_runs.csv", index=False)
            print("✓ Nuttige runs opgeslagen als: ray_useful_runs.csv")

        # Experiment overzicht
        summary = get_experiments_summary(all_data)
        if not summary.empty:
            print("\n📋 Experiment overzicht:")
            print(summary.to_string())

        print("\n✅ Klaar! Check de CSV bestanden voor je data.")
