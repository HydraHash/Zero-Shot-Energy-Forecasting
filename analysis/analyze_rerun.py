import pandas as pd
import ast
from pathlib import Path

def identify_rerun_requirements(base_output_dir: str | Path):
    base_path = Path(base_output_dir)
    
    # Check for parquet first, fallback to tsv if needed
    data_files = list(base_path.rglob("*.parquet")) or list(base_path.rglob("*.tsv"))
    
    if not data_files:
        print("No output files found.")
        return

    checked_signatures = set()
    print(f"{'MODEL':<25} | {'COMBINATION':<25} | {'STATUS':<15} | {'ACTUAL LEN':<12} | {'PRED LEN'}")
    print("-" * 100)

    for file_path in data_files:
        model_name = file_path.parent.name
        combo_name = file_path.parent.parent.name
        signature = f"{model_name}_{combo_name}"
        
        if signature in checked_signatures:
            continue
            
        checked_signatures.add(signature)
        
        # Determine engine dynamically based on file type
        if file_path.suffix == ".parquet":
            df = pd.read_parquet(file_path, engine="pyarrow")
        else:
            df = pd.read_csv(file_path, sep="\t")
            
        if df.empty:
            continue
            
        first_row = df.iloc[0]
        
        actual_val = first_row["actual_values"]
        pred_val = first_row["predicted_values"]
        
        # CRUCIAL FIX: If the file was read as a string, safely evaluate it back into a Python list
        if isinstance(actual_val, str):
            actual_val = ast.literal_eval(actual_val)
        if isinstance(pred_val, str):
            pred_val = ast.literal_eval(pred_val)
            
        actual_len = len(actual_val) if hasattr(actual_val, "__iter__") else 1
        pred_len = len(pred_val) if hasattr(pred_val, "__iter__") else 1
        
        status = "❌ RERUN" if actual_len != pred_len else "✅ VALID"
        print(f"{model_name:<25} | {combo_name:<25} | {status:<15} | {actual_len:<12} | {pred_len}")


if __name__ == "__main__":
    PREDICTION_DIR = r"C:\Users\lsb-ps\Documents\predictions_overview_lean\5_DLinear\dlinear_prediction_output\c1_lag1_horizon1_step1"
    #PREDICTION_DIR = r"H:\model_Kairos\small10m_kairos_prediction_output"
    identify_rerun_requirements(PREDICTION_DIR)