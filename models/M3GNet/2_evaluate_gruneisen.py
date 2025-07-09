import os
from glob import glob
import pandas as pd

from gamma_SRME import glob2df, ID, DFT_NONAC_REF
from gamma_SRME.benchmark import get_metrics, process_gruneisen_descriptors
from gamma_SRME.utils import check_imaginary_freqs


model_name = "M3GNet"

json_file = "gamma_srme.json.gz"
txt_path = "metrics_gamma.txt"
in_file = "mode_gruneisen_0-???.json.gz"

in_folder = f"2025-07-07-{model_name}-phononDB-GAMMA-FIRE_2SR_force0.0001_sym1e-05/"

DFT_RESULTS_FILE = DFT_NONAC_REF

module_dir = os.path.dirname(__file__)
in_pattern = f"{module_dir}/{in_folder}/{in_file}"
out_path = f"{module_dir}/{in_folder}/{json_file}"


# Read MLP results
if not glob(in_pattern):
    if os.path.exists(out_path):
        df_mlp_results = pd.read_json(out_path).set_index(ID)
else:
    df_mlp_results = glob2df(in_pattern, max_files=None).set_index(ID)


# Read DFT results
df_dft_results = pd.read_json(DFT_RESULTS_FILE).set_index(ID)


df_mlp_filtered = df_mlp_results[df_mlp_results.index.isin(df_dft_results.index)]
# df_mlp_filtered = df_mlp_filtered.reindex(df_dft_results.index)


df_mlp_processed = process_gruneisen_descriptors(df_mlp_filtered, df_dft_results, [300])

mSRE, mSRME, rmseSRE, rmseSRME = get_metrics(df_mlp_filtered)


# Save results
df_mlp_processed.round(5)
df_mlp_processed.index.name = ID
df_mlp_processed.reset_index().to_json(out_path)

df_mlp_filtered.sort_values("desc", inplace=True)


# Print
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
df_mlp_print = df_mlp_filtered.loc[
    :, ["desc", "SRME", "SRE", "gamma", "DFT_gamma", "imaginary_freqs"]
].copy()

df_mlp_print["DFT_gamma"] = df_mlp_print["DFT_gamma"].apply(
    lambda x: x[0] if not pd.isna(x) else x
)
df_mlp_print["gamma"] = df_mlp_print["gamma"].apply(
    lambda x: x[0] if not pd.isna(x) else x
)

# df_mlp_print["SRME_failed"] = df_mlp_print["SRME"].apply(lambda x: x == 2)

with open(txt_path, "w") as f:
    print(f"MODEL: {model_name}", file=f)
    print(f"\tmean SRME: {mSRME}", file=f)
    print(f"\tmean SRE: {mSRE}", file=f)

    print(df_mlp_print.round(4), file=f)


df_mlp_print = df_mlp_print[["desc", "SRME", "SRE", "gamma", "DFT_gamma"]]
print(df_mlp_print.round(3))

print(f"MODEL: {model_name}")
print(f"\tmean SRME: {mSRME}")
print(f"\tmean SRE: {mSRE}")
