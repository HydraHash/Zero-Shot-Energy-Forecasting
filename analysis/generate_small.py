import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def split_specific_batches(base_output_dir: str | Path):
    base_path = Path(base_output_dir)
    all_batches = []
    
    target_files = ["batch_0.parquet", "batch_500.parquet"]
    parquet_files = [f for f in base_path.rglob("*.parquet") if f.name in target_files]
    
    if not parquet_files:
        logging.warning(f"Keine der gesuchten Dateien ({target_files}) in {base_path} gefunden.")
        return

    for file_path in parquet_files:
        model_name = file_path.parent.name
        combo_name = file_path.parent.parent.name
        
        df_batch = pd.read_parquet(file_path, engine="pyarrow")
        df_batch["model"] = model_name
        df_batch["combo"] = combo_name
        
        all_batches.append(df_batch)

    full_df = pd.concat(all_batches, ignore_index=True)
    
    # Direkte Filterung über die bestehende 'id'-Spalte (mit on-the-fly Konvertierung zu int für den sicheren Vergleich)
    mask_id = full_df['id'].astype(int)
    df_part1 = full_df[(mask_id >= 0) & (mask_id < 268)].copy()
    df_part2 = full_df[(mask_id >= 268) & (mask_id < 535)].copy()
    
    for df in [df_part1, df_part2]:
        df.sort_values(by=["id", "horizon_start"], ascending=[True, True], inplace=True)
        df.reset_index(drop=True, inplace=True)
        
    out_file1 = base_path / "batch_ids_0_to_267.parquet"
    out_file2 = base_path / "batch_ids_268_to_534.parquet"
    
    df_part1.to_parquet(out_file1, engine="pyarrow", index=False)
    df_part2.to_parquet(out_file2, engine="pyarrow", index=False)
    
    logging.info(f"Erfolgreich gespeichert: {len(df_part1)} Einträge in {out_file1.name}")
    logging.info(f"Erfolgreich gespeichert: {len(df_part2)} Einträge in {out_file2.name}")

if __name__ == "__main__":
    split_specific_batches(r"C:\Users\lsb-ps\Documents\model_TimeLLM\Time-LLM\prediction_output\c1_train60_horizon1_step1\dlinear")
