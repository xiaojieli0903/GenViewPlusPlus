import argparse
import pandas as pd

def merge_two_csv(csv_gs, csv_nl, output_dir):
    df_gs = pd.read_csv(csv_gs)
    df_nl = pd.read_csv(csv_nl)

    merged = pd.merge(df_gs, df_nl, on=["image", "prompt"], how="outer")
    merged.to_csv(f"{output_dir}/infos.csv", index=False)

    # only exists in noise_level CSV
    only_in_gs = merged[merged["noise_level"].isna()][["image", "prompt", "guidance_scale"]]
    only_in_gs.to_csv(f"{output_dir}/only_in_gs.csv", index=False)

    # only exists in noise_level CSV
    only_in_nl = merged[merged["guidance_scale"].isna()][["image", "prompt", "noise_level"]]
    only_in_nl.to_csv(f"{output_dir}/only_in_nl.csv", index=False)

    print(f"✅ The merge is complete and the results are saved to {output_dir}")
    print(f"- merged.csv     ({len(merged)} rows)")
    print(f"- only_in_gs.csv ({len(only_in_gs)} rows)")
    print(f"- only_in_nl.csv ({len(only_in_nl)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two CSV files with guidance_scale and noise_level info")
    parser.add_argument("--csv_gs", type=str, required=True, help="Path to the CSV file with guidance_scale (image, prompt, guidance_scale)")
    parser.add_argument("--csv_nl", type=str, required=True, help="Path to the CSV file with noise_level (image, prompt, noise_level)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the merged and unmatched CSV files")

    args = parser.parse_args()

    merge_two_csv(args.csv_gs, args.csv_nl, args.output_dir)
