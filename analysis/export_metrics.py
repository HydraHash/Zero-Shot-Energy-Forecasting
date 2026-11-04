import ast
import gc
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import false_discovery_control, rankdata, wilcoxon


def export_all_raw_maes(base_output_dir: str | Path, output_file: str | Path):
    base_path = Path(base_output_dir)

    # Filter out master files and the target output file before processing
    all_parquet_files = list(base_path.rglob("*.parquet"))
    parquet_files = [
        f
        for f in all_parquet_files
        if not f.name.startswith("master_") and f.name != Path(output_file).name
    ]

    if not parquet_files:
        print(f"No valid prediction parquet files found in {base_path}")
        return

    print(
        f"Extracting MAE from {len(parquet_files)} files with direct-to-disk flushing..."
    )

    writer = None

    for file_path in parquet_files:
        if len(file_path.parts) < 4:
            continue

        model_name = file_path.parent.name
        combo_name = file_path.parent.parent.name.split("_")[0]
        size_folder = file_path.parent.parent.parent.name
        size = "small" if "small" in size_folder.lower() else "normal"

        # Read only required columns into memory
        df = pd.read_parquet(
            file_path,
            engine="pyarrow",
            columns=["id", "horizon_start", "actual_values", "predicted_values"],
        )

        maes = []
        for actuals, preds in zip(df["actual_values"], df["predicted_values"]):
            if isinstance(actuals, str):
                actuals = ast.literal_eval(actuals)
            if isinstance(preds, str):
                preds = ast.literal_eval(preds)

            act_arr = np.array(actuals, dtype=float)
            pred_arr = np.array(preds, dtype=float)

            maes.append(np.mean(np.abs(act_arr - pred_arr)))

        batch_df = pd.DataFrame(
            {
                "id": df["id"],
                "horizon_start": df["horizon_start"],
                "model_name": model_name,
                "combo": combo_name,
                "size": size,
                "mae": maes,
            }
        )

        table = pa.Table.from_pandas(batch_df)

        if writer is None:
            writer = pq.ParquetWriter(output_file, table.schema)

        writer.write_table(table)

        del df, batch_df, table, maes
        gc.collect()

    if writer:
        writer.close()

    print(f"Raw MAEs successfully compiled and saved to {output_file}")


def append_advanced_metrics(
    df: pd.DataFrame, baseline_mae_df: pd.DataFrame
) -> pd.DataFrame:
    rmses, smapes_mean, smapes_median, rmaes, das = [], [], [], [], []

    # Merge baseline MAEs to align exactly on ID and horizon_start
    df = df.merge(baseline_mae_df, on=["id", "horizon_start"], how="left")

    for _, row in df.iterrows():
        actuals = row["actual_values"]
        preds = row["predicted_values"]
        baseline_mae = row["baseline_mae"]

        if isinstance(actuals, str):
            actuals = ast.literal_eval(actuals)
        if isinstance(preds, str):
            preds = ast.literal_eval(preds)

        act_arr = np.array(actuals, dtype=float)
        pred_arr = np.array(preds, dtype=float)
        errors = act_arr - pred_arr

        model_mae = np.mean(np.abs(errors))

        # rMAE (Relative MAE)
        if pd.isna(baseline_mae):
            rmae = np.nan
        elif baseline_mae == 0 and model_mae == 0:
            rmae = 0.0
        elif baseline_mae == 0 and model_mae > 0:
            rmae = np.inf
        else:
            rmae = model_mae / baseline_mae

        # RMSE
        rmse = np.sqrt(np.mean(errors**2))

        # sMAPE (Mean & Median)
        denominator = (np.abs(act_arr) + np.abs(pred_arr)) / 2
        smape_arr = (
            np.divide(
                np.abs(errors),
                denominator,
                out=np.zeros_like(errors, dtype=float),
                where=denominator != 0,
            )
            * 100
        )
        smape_mean = np.mean(smape_arr)
        smape_median = np.median(smape_arr)

        # Directional Accuracy (DA)
        if len(act_arr) > 1:
            da = np.mean(np.sign(np.diff(act_arr)) == np.sign(np.diff(pred_arr))) * 100
        else:
            da = np.nan

        rmaes.append(rmae)
        rmses.append(rmse)
        smapes_mean.append(smape_mean)
        smapes_median.append(smape_median)
        das.append(da)

    df["horizon_rmae"] = rmaes
    df["horizon_rmse"] = rmses
    df["horizon_smape_mean"] = smapes_mean
    df["horizon_smape_median"] = smapes_median
    df["horizon_da"] = das

    df.drop(columns=["baseline_mae"], inplace=True)
    return df


def process_and_export_batches(
    base_output_dir: str | Path,
    mae_file: str | Path,
    baseline_model_name: str,
    invalid_ids: list = None,
):
    base_path = Path(base_output_dir)
    output_file = base_path / "master_metrics.parquet"

    all_maes = pd.read_parquet(mae_file)

    # Exclude invalid IDs upfront with dynamic type matching
    if invalid_ids:
        safe_invalid_ids = pd.Series(invalid_ids).astype(all_maes["id"].dtype).tolist()
        all_maes = all_maes[~all_maes["id"].isin(safe_invalid_ids)]

    batch_groups = defaultdict(list)
    parquet_files = list(base_path.rglob("*.parquet"))

    for file_path in parquet_files:
        if file_path.name == Path(mae_file).name or "master_" in file_path.name:
            continue
        batch_groups[file_path.parent].append(file_path)

    all_company_metrics = []

    for parquet_dir, files in batch_groups.items():
        if len(parquet_dir.parts) < 3:
            continue

        model_name = parquet_dir.name
        combo_name = parquet_dir.parent.name
        size_folder_name = parquet_dir.parent.parent.name

        size = "small" if "small" in size_folder_name.lower() else "normal"
        combo_prefix = combo_name.split("_")[0]

        print(
            f"Processing metrics for {model_name} (Size: {size}, Combo: {combo_prefix})..."
        )

        # Isolate the baseline MAE for this exact cohort
        cohort_baseline_maes = all_maes[
            (all_maes["model_name"] == baseline_model_name)
            & (all_maes["combo"] == combo_prefix)
            & (all_maes["size"] == size)
        ][["id", "horizon_start", "mae"]].rename(columns={"mae": "baseline_mae"})

        if cohort_baseline_maes.empty and model_name != baseline_model_name:
            warnings.warn(
                f"No MAE data found for baseline '{baseline_model_name}' in Size: {size}, Combo: {combo_prefix}. rMAE will result in NaN."
            )

        all_batches = [pd.read_parquet(f, engine="pyarrow") for f in files]
        df = pd.concat(all_batches, ignore_index=True)

        if "actual_values" not in df.columns:
            print(f"Skipping {model_name} due to missing 'actual_values' column.")
            continue

        # Exclude invalid IDs from the model predictions
        if invalid_ids:
            safe_invalid_ids = pd.Series(invalid_ids).astype(df["id"].dtype).tolist()
            df = df[~df["id"].isin(safe_invalid_ids)]

        df.sort_values(by=["id", "horizon_start"], ascending=[True, True], inplace=True)
        df.reset_index(drop=True, inplace=True)

        df = append_advanced_metrics(df, cohort_baseline_maes)

        agg_funcs = {
            "horizon_rmae": ["mean", "median"],
            "horizon_rmse": ["mean", "median"],
            "horizon_smape_mean": ["mean"],
            "horizon_smape_median": ["median"],
            "horizon_da": ["mean"],
        }
        company_metrics = df.groupby("id").agg(agg_funcs)

        # Flatten MultiIndex columns
        company_metrics.columns = [
            "_".join(col).strip() for col in company_metrics.columns.values
        ]
        company_metrics.reset_index(inplace=True)

        # Append identifying columns
        company_metrics["model_name"] = model_name
        company_metrics["size_category"] = size
        company_metrics["combo_prefix"] = combo_prefix

        all_company_metrics.append(company_metrics)

        del df, company_metrics, all_batches
        gc.collect()

    # Consolidate into a single master parquet file
    if all_company_metrics:
        master_metrics_df = pd.concat(all_company_metrics, ignore_index=True)
        master_metrics_df.to_parquet(output_file, engine="pyarrow", index=False)
        print(f"All metrics exported successfully to {output_file.name}")
    else:
        print("No valid metrics were processed.")


LOWER_IS_BETTER = {
    "horizon_smape_median_median": True,
    "horizon_smape_median": True,
    "horizon_rmae_median": True,
    "horizon_mae_median": True,
    "horizon_rmse_median": True,
    "horizon_da_mean": False,
    "horizon_da_median": False,
}


def _is_lower_better(metric_name: str) -> bool:
    if metric_name in LOWER_IS_BETTER:
        return LOWER_IS_BETTER[metric_name]
    return "_da_" not in metric_name.lower() and not metric_name.lower().endswith("_da")


def _run_single_wilcoxon(
    df_sub: pd.DataFrame, metric_model: str, metric_baseline: str, lower_better: bool
):
    """Helper to compute paired Wilcoxon and Rank-Biserial r on any slice of paired data."""
    valid_mask = np.isfinite(df_sub[metric_model]) & np.isfinite(
        df_sub[metric_baseline]
    )
    df_clean = df_sub[valid_mask]
    n_dropped = len(df_sub) - len(df_clean)
    n_valid = len(df_clean)

    if n_valid < 10:
        return None

    raw_diffs = (df_clean[metric_model] - df_clean[metric_baseline]).to_numpy()
    advantage_diffs = -raw_diffs if lower_better else raw_diffs

    wins = np.sum(advantage_diffs > 0)
    losses = np.sum(advantage_diffs < 0)
    ties = np.sum(advantage_diffs == 0)
    win_rate_pct = ((wins + 0.5 * ties) / n_valid) * 100.0

    non_zero_mask = advantage_diffs != 0
    n_non_zero = int(np.sum(non_zero_mask))

    if n_non_zero == 0:
        stat, p_value, r_rb = 0.0, 1.0, 0.0
    else:
        stat, p_value = wilcoxon(raw_diffs, zero_method="wilcox")
        nz_adv = advantage_diffs[non_zero_mask]
        ranks = rankdata(np.abs(nz_adv))
        w_pos = np.sum(ranks[nz_adv > 0])
        w_neg = np.sum(ranks[nz_adv < 0])
        r_rb = (w_pos - w_neg) / (w_pos + w_neg)

    return {
        "n_valid_samples": int(n_valid),
        "dropped_infs_nans": int(n_dropped),
        "median_difference": float(np.median(raw_diffs)),
        "win_rate_pct": float(win_rate_pct),
        "rank_biserial_r": float(r_rb),
        "w_statistic": float(stat),
        "p_value": float(p_value),
    }


def export_unified_wilcoxon_tests(
    base_output_dir: str | Path,
    baseline_model_name: str,
    masterdata_path: str | Path = None,
    target_metrics: list[str] = None,
    alpha: float = 0.05,
    min_effect_size: float = 0.10,
    force_recompute: bool = False,
):
    if target_metrics is None:
        target_metrics = ["horizon_smape_median_median", "horizon_rmae_median"]

    base_path = Path(base_output_dir)
    output_file = base_path / "master_wilcoxon_tests.parquet"
    metrics_file = base_path / "master_metrics.parquet"

    if output_file.exists() and not force_recompute:
        try:
            existing_df = pd.read_parquet(output_file)
            if (
                "stratification_type" in existing_df.columns
                and "p_value_fdr" in existing_df.columns
            ):
                print(
                    f"Skipping statistical testing - '{output_file.name}' is up to date."
                )
                return
        except Exception:
            pass

    # Load Masterdata if provided for Sector / Grid_level / Imputed stratification
    df_master = None
    strat_cols = []
    if masterdata_path is not None and Path(masterdata_path).exists():
        m_path = Path(masterdata_path)
        sep = "\t" if m_path.suffix in [".tsv", ".csv"] else ","
        df_master = (
            pd.read_csv(m_path, sep=sep)
            if m_path.suffix != ".parquet"
            else pd.read_parquet(m_path)
        )
        if "Id" in df_master.columns:
            df_master.rename(columns={"Id": "id"}, inplace=True)
        # Create compact Sector label: e.g., "B: Bergbau und Gewinnung..."
        if (
            "Sector_group_id" in df_master.columns
            and "Sector_group" in df_master.columns
        ):
            df_master["Sector_Label"] = (
                df_master["Sector_group_id"].astype(str)
                + ": "
                + df_master["Sector_group"].astype(str)
            )
            strat_cols.append(("Sector", "Sector_Label"))
        if "Grid_level" in df_master.columns:
            strat_cols.append(("Grid Level", "Grid_level"))
        if "Imputed" in df_master.columns:
            strat_cols.append(("Imputed Status", "Imputed"))

    if not metrics_file.exists():
        print(
            "No master_metrics.parquet found for statistical testing. Run process_and_export_batches first."
        )
        return

    df_metrics = pd.read_parquet(metrics_file)
    baseline_df = df_metrics[df_metrics["model_name"] == baseline_model_name]

    if baseline_df.empty:
        warnings.warn(f"No baseline metrics found for '{baseline_model_name}'.")
        return

    competing_df = df_metrics[df_metrics["model_name"] != baseline_model_name]

    results = []

    # Iterate dynamically over every Size/Combo pairing available in the baseline dataset
    for (size, combo), b_group in baseline_df.groupby(
        ["size_category", "combo_prefix"]
    ):
        c_group = competing_df[
            (competing_df["size_category"] == size)
            & (competing_df["combo_prefix"] == combo)
        ]

        for model_name, m_group in c_group.groupby("model_name"):
            df_paired = m_group.merge(
                b_group, on="id", suffixes=("_model", "_baseline")
            )

            if df_master is not None:
                df_paired = df_paired.merge(df_master, on="id", how="left")

            for metric in target_metrics:
                metric_model, metric_baseline = f"{metric}_model", f"{metric}_baseline"
                if (
                    metric_model not in df_paired.columns
                    or metric_baseline not in df_paired.columns
                ):
                    continue
                lower_better = _is_lower_better(metric)

                # Overall Cohort Test
                res_all = _run_single_wilcoxon(
                    df_paired, metric_model, metric_baseline, lower_better
                )
                if res_all:
                    results.append(
                        {
                            "target_metric": metric,
                            "model_name": model_name,
                            "baseline_model": baseline_model_name,
                            "size_category": size,
                            "combo_prefix": combo,
                            "stratification_type": "Overall",
                            "stratification_value": "All Profiles",
                            **res_all,
                        }
                    )

                # Stratified Tests (Sector, Grid Level, Imputed)
                for strat_label, col_name in strat_cols:
                    for group_val, df_group in df_paired.groupby(col_name, dropna=True):
                        res_grp = _run_single_wilcoxon(
                            df_group, metric_model, metric_baseline, lower_better
                        )
                        if res_grp:
                            results.append(
                                {
                                    "target_metric": metric,
                                    "model_name": model_name,
                                    "baseline_model": baseline_model_name,
                                    "size_category": size,
                                    "combo_prefix": combo,
                                    "stratification_type": strat_label,
                                    "stratification_value": str(group_val),
                                    **res_grp,
                                }
                            )

    if results:
        results_df = pd.DataFrame(results)
        # Apply Benjamini-Hochberg FDR within each evaluation family
        results_df["p_value_fdr"] = results_df.groupby(
            ["size_category", "combo_prefix", "target_metric", "stratification_type"]
        )["p_value"].transform(
            lambda p: (
                false_discovery_control(np.clip(p.to_numpy(), 0.0, 1.0), method="bh")
                if len(p) > 1
                else p
            )
        )
        results_df["neg_log10_p_fdr"] = -np.log10(
            np.clip(results_df["p_value_fdr"], 1e-300, 1.0)
        )
        results_df["is_significant"] = results_df["p_value_fdr"] < alpha

        conclusions = []
        for _, row in results_df.iterrows():
            if not row["is_significant"] or row["rank_biserial_r"] == 0:
                conclusions.append("No Significant Difference")
            else:
                r = row["rank_biserial_r"]
                if abs(r) < min_effect_size:
                    conclusions.append(
                        "Model Superior (Negligible Effect)"
                        if r > 0
                        else "Baseline Superior (Negligible Effect)"
                    )
                else:
                    conclusions.append(
                        "Model is Superior" if r > 0 else "Baseline is Superior"
                    )
        results_df["conclusion"] = conclusions
        results_df.to_parquet(output_file, engine="pyarrow", index=False)
        print(f"Wilcoxon tests successfully exported to {output_file.name}")


if __name__ == "__main__":
    PREDICTION_DIR = r"C:\Users\lsb-ps\Documents\predictions_overview_lean"
    MAE_FILE = Path(PREDICTION_DIR) / "all_models_raw_mae.parquet"
    MASTERDATA_PATH = r"C:\Users\lsb-ps\Documents\VEA_Playground\zero_shot_data\08_zero_shot_output\master_data_imputed.csv"

    INVALID_IDS = [49, 401, 2325, 3425, 3986, 4297]

    BASELINE_NAME = "SeasonNaive"
    TARGET_METRICS = ["horizon_rmae_median", "horizon_smape_median_median"]

    if not MAE_FILE.exists():
        print("Calculating raw mae for all models...")
        export_all_raw_maes(PREDICTION_DIR, MAE_FILE)

    # print("Calculating all metrics...")
    # process_and_export_batches(PREDICTION_DIR, MAE_FILE, BASELINE_NAME, invalid_ids=INVALID_IDS)

    # export_unified_wilcoxon_tests(PREDICTION_DIR, BASELINE_NAME, MASTERDATA_PATH, TARGET_METRICS)
