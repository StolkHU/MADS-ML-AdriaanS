import sqlite3
import pandas as pd
from datetime import datetime


def extract_mlflow_data(db_path="mlflow.db", output_path="mlflow_overview.csv"):
    """
    Extract run data from MLflow database and create a CSV overview with all parameters and metrics
    """
    # Connect to the database
    conn = sqlite3.connect(db_path)

    try:
        # Get run information
        runs_query = """
        SELECT 
            run_uuid,
            name,
            start_time,
            end_time,
            status,
            artifact_uri,
            experiment_id
        FROM runs
        """
        runs_df = pd.read_sql_query(runs_query, conn)

        # Calculate runtime in seconds
        def calculate_runtime(row):
            if pd.notna(row["start_time"]) and pd.notna(row["end_time"]):
                # Assuming timestamps are in milliseconds (common in MLflow)
                # If they're in seconds, remove the /1000
                start = datetime.fromtimestamp(row["start_time"] / 1000)
                end = datetime.fromtimestamp(row["end_time"] / 1000)
                runtime_seconds = (end - start).total_seconds()
                return runtime_seconds
            return None

        runs_df["runtime_seconds"] = runs_df.apply(calculate_runtime, axis=1)

        # Convert runtime to more readable format
        def format_runtime(seconds):
            if pd.isna(seconds) or seconds is None:
                return ""

            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = int(seconds % 60)

            if hours > 0:
                return f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                return f"{minutes}m {seconds}s"
            else:
                return f"{seconds}s"

        runs_df["runtime_formatted"] = runs_df["runtime_seconds"].apply(format_runtime)

        # Get all parameters for each run
        params_query = """
        SELECT 
            run_uuid,
            key,
            value
        FROM params
        """
        params_df = pd.read_sql_query(params_query, conn)

        # Get all metrics for each run
        metrics_query = """
        SELECT 
            run_uuid,
            key,
            value,
            timestamp,
            step
        FROM metrics
        """
        metrics_df = pd.read_sql_query(metrics_query, conn)

        # Create pivot tables for parameters and metrics
        params_pivot = params_df.pivot(index="run_uuid", columns="key", values="value")

        # For metrics, get the latest value (highest step) for each metric
        latest_metrics = (
            metrics_df.sort_values("step").groupby(["run_uuid", "key"]).tail(1)
        )
        metrics_pivot = latest_metrics.pivot(
            index="run_uuid", columns="key", values="value"
        )

        # Merge all dataframes
        result = runs_df.set_index("run_uuid")
        result = result.join(params_pivot, how="left")
        result = result.join(metrics_pivot, how="left")

        # Debug: Print all available columns in the merged result
        print("\nAll available columns in merged data:")
        all_columns = sorted(result.columns)
        for col in all_columns:
            print(f"- {col}")

        # Separate parameters and metrics columns
        param_columns = list(params_pivot.columns) if params_pivot is not None else []
        metric_columns = (
            list(metrics_pivot.columns) if metrics_pivot is not None else []
        )

        print(
            f"\nFound {len(param_columns)} parameters and {len(metric_columns)} metrics"
        )

        # Create the desired columns overview with specific mappings
        overview_columns = {}

        # Add the specific columns you want with special handling
        overview_columns["Name"] = result.get("name", "")
        overview_columns["Runtime"] = result.get("runtime_formatted", "")
        overview_columns["Runtime (seconds)"] = result.get("runtime_seconds", "")
        overview_columns["Batchsize"] = result.get(
            "batchsize", result.get("batch_size", result.get("Batchsize", ""))
        )
        overview_columns["filters"] = result.get("filters", result.get("Filters", ""))
        overview_columns["units1"] = result.get("units1", result.get("Units1", ""))
        overview_columns["units2"] = result.get("units2", result.get("Units2", ""))
        overview_columns["Loss/test"] = result.get(
            "Loss/test",
            result.get(
                "loss/test",
                result.get(
                    "test_loss", result.get("Test Loss", result.get("Test_Loss", ""))
                ),
            ),
        )
        overview_columns["Loss/train"] = result.get(
            "Loss/train",
            result.get(
                "loss/train",
                result.get(
                    "train_loss", result.get("Train Loss", result.get("Train_Loss", ""))
                ),
            ),
        )
        overview_columns["Learning_rate"] = result.get(
            "Learning_rate",
            result.get(
                "learning_rate",
                result.get(
                    "lr", result.get("Learning Rate", result.get("LearningRate", ""))
                ),
            ),
        )
        overview_columns["metric/Accuracy"] = result.get(
            "metric_Accuracy",
            result.get(
                "metric/Accuracy",
                result.get(
                    "metric/accuracy",
                    result.get("accuracy", result.get("Accuracy", "")),
                ),
            ),
        )

        # Special handling for iteration column - fill from iteration, filters, or name
        iteration_values = []
        filters_values = result.get("filters", result.get("Filters", ""))
        name_values = result.get("name", "")
        iteration_param_values = result.get(
            "iteration", result.get("iterations", result.get("Iteration", ""))
        )

        # Create iteration column by priority: iteration param > filters > name
        for idx in result.index:
            iteration_val = ""
            if (
                idx in iteration_param_values
                and pd.notna(iteration_param_values.loc[idx])
                and str(iteration_param_values.loc[idx]).strip()
            ):
                iteration_val = iteration_param_values.loc[idx]
            elif (
                idx in filters_values
                and pd.notna(filters_values.loc[idx])
                and str(filters_values.loc[idx]).strip()
            ):
                iteration_val = filters_values.loc[idx]
            elif (
                idx in name_values
                and pd.notna(name_values.loc[idx])
                and str(name_values.loc[idx]).strip()
            ):
                iteration_val = name_values.loc[idx]
            iteration_values.append(iteration_val)

        overview_columns["iteration"] = iteration_values

        # Add all other parameters (excluding the ones already mapped)
        already_mapped_params = {
            "name",
            "batchsize",
            "batch_size",
            "Batchsize",
            "filters",
            "Filters",
            "units1",
            "Units1",
            "units2",
            "Units2",
            "Learning_rate",
            "learning_rate",
            "lr",
            "Learning Rate",
            "LearningRate",
            "iteration",
            "iterations",
            "Iteration",
        }

        for param in param_columns:
            if param not in already_mapped_params:
                overview_columns[f"param_{param}"] = result[param]

        # Add all other metrics (excluding the ones already mapped)
        already_mapped_metrics = {
            "Loss/test",
            "loss/test",
            "test_loss",
            "Test Loss",
            "Test_Loss",
            "Loss/train",
            "loss/train",
            "train_loss",
            "Train Loss",
            "Train_Loss",
            "metric_Accuracy",
            "metric/Accuracy",
            "metric/accuracy",
            "accuracy",
            "Accuracy",
        }

        for metric in metric_columns:
            if metric not in already_mapped_metrics:
                overview_columns[f"metric_{metric}"] = result[metric]

        overview_df = pd.DataFrame(overview_columns)

        # Round loss and accuracy values to 2 decimal places
        numeric_columns = ["Loss/test", "Loss/train", "metric/Accuracy"]
        for col in numeric_columns:
            if col in overview_df.columns:
                overview_df[col] = pd.to_numeric(
                    overview_df[col], errors="coerce"
                ).round(2)

        # Round all other metric columns to 2 decimal places
        for col in overview_df.columns:
            if col.startswith("metric_"):
                overview_df[col] = pd.to_numeric(
                    overview_df[col], errors="coerce"
                ).round(2)

        # Round runtime seconds to 2 decimal places
        if "Runtime (seconds)" in overview_df.columns:
            overview_df["Runtime (seconds)"] = pd.to_numeric(
                overview_df["Runtime (seconds)"], errors="coerce"
            ).round(2)

        # Reorder columns to put priority columns first
        priority_columns = [
            "Name",
            "Runtime",
            "Runtime (seconds)",
            "Batchsize",
            "filters",
            "units1",
            "units2",
            "Loss/test",
            "Loss/train",
            "Learning_rate",
            "metric/Accuracy",
            "iteration",
        ]

        # Get all columns and separate them
        all_cols = list(overview_df.columns)
        priority_cols_available = [col for col in priority_columns if col in all_cols]
        param_cols = [col for col in all_cols if col.startswith("param_")]
        metric_cols = [
            col
            for col in all_cols
            if col.startswith("metric_") and col != "metric/Accuracy"
        ]
        other_cols = [
            col
            for col in all_cols
            if col not in priority_cols_available
            and not col.startswith("param_")
            and not col.startswith("metric_")
        ]

        # Final column order
        final_column_order = (
            priority_cols_available + param_cols + metric_cols + other_cols
        )
        overview_df = overview_df[final_column_order]

        # Save to CSV
        overview_df.to_csv(output_path, index=True)

        print(f"\nOverview saved to {output_path}")
        print(f"Shape: {overview_df.shape}")
        print(f"Columns included: {len(overview_df.columns)}")
        print(f"- Priority columns: {len(priority_cols_available)}")
        print(f"- Parameter columns: {len(param_cols)}")
        print(f"- Metric columns: {len(metric_cols)}")
        print(f"- Other columns: {len(other_cols)}")

        print("\nFirst 5 rows:")
        # Show subset of columns for readable output
        display_cols = priority_cols_available[:5] + param_cols[:3] + metric_cols[:3]
        print(overview_df[display_cols].head())

        print("\nColumn names:")
        for i, col in enumerate(final_column_order):
            print(f"{i + 1:3d}. {col}")

        return overview_df

    finally:
        conn.close()


if __name__ == "__main__":
    # Extract the data
    df = extract_mlflow_data()

    # Optional: Create a separate file with just the column mapping for reference
    conn = sqlite3.connect("mlflow.db")

    # Get all unique parameter and metric keys
    params_query = "SELECT DISTINCT key FROM params ORDER BY key"
    metrics_query = "SELECT DISTINCT key FROM metrics ORDER BY key"

    all_params = pd.read_sql_query(params_query, conn)
    all_metrics = pd.read_sql_query(metrics_query, conn)

    print("\nAll parameters found in database:")
    for param in all_params["key"]:
        print(f"- {param}")

    print("\nAll metrics found in database:")
    for metric in all_metrics["key"]:
        print(f"- {metric}")

    # Save parameter and metric lists for reference
    with open("mlflow_columns_reference.txt", "w") as f:
        f.write("MLflow Database Column Reference\n")
        f.write("=" * 40 + "\n\n")
        f.write("Parameters:\n")
        for param in all_params["key"]:
            f.write(f"- {param}\n")
        f.write("\nMetrics:\n")
        for metric in all_metrics["key"]:
            f.write(f"- {metric}\n")

    conn.close()
    print("\nColumn reference saved to 'mlflow_columns_reference.txt'")
