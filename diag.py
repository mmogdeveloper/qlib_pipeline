#!/usr/bin/env python3
import os, sys
if sys.platform == 'darwin':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')

import multiprocessing
if sys.platform == "darwin":
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

sys.path.insert(0, os.path.dirname(__file__))

from data.data_loader import get_data_loader
loader = get_data_loader()
loader.init_qlib()

from qlib.workflow import R
from qlib.data import D
import pandas as pd

exp = R.get_exp(experiment_name='qlib_pipeline_lgbm')
recs = exp.list_recorders(rtype=exp.RT_L)

if isinstance(recs, list):
    finished = [r for r in recs if r.info.get("status") == "FINISHED"]
    rec = finished[-1] if finished else recs[-1]
elif isinstance(recs, dict):
    rec = recs[list(recs.keys())[-1]]

pred = rec.load_object('pred.pkl')
print(f"pred shape (full): {pred.shape}, index names: {pred.index.names}")

# 只保留最近 1 个月数据，避免全量数据撑爆内存
cutoff = pred.index.get_level_values(0).max() - pd.DateOffset(months=1)
pred = pred[pred.index.get_level_values(0) >= cutoff]
print(f"pred shape (1m):   {pred.shape}")

# Shift
pred_shifted = pred.reset_index()
date_col = pred_shifted.columns[0]
dates = sorted(pred_shifted[date_col].unique())
date_map = dict(zip(dates[:-1], dates[1:]))
pred_shifted[date_col] = pred_shifted[date_col].map(date_map)
pred_shifted = pred_shifted.dropna(subset=[date_col])
pred_shifted = pred_shifted.set_index(pred.index.names)
print(f"shifted shape: {pred_shifted.shape}")

instruments = pred_shifted.index.get_level_values(1).unique().tolist()[:3]
start_date = str(pred_shifted.index.get_level_values(0).min())[:10]
end_date = str(pred_shifted.index.get_level_values(0).max())[:10]

volume_df = D.features(instruments, ['$volume'], start_time=start_date, end_time=end_date)
print(f"\nvolume_df index names: {volume_df.index.names}")
print(f"pred_shifted index names: {pred_shifted.index.names}")

if list(volume_df.index.names) != list(pred_shifted.index.names):
    print("Swapping levels...")
    volume_df = volume_df.swaplevel(0, 1).sort_index()

inst = instruments[0]
pred_dates = set(pred_shifted.loc[pred_shifted.index.get_level_values(1) == inst].index.get_level_values(0))

actual_dates_per_inst = volume_df.groupby(level="instrument").apply(
    lambda g: set(g.index.get_level_values("datetime"))
)
vol_dates = actual_dates_per_inst.get(inst, set())

print(f"\nInst: {inst}")
print(f"pred dates count: {len(pred_dates)}, vol dates count: {len(vol_dates)}")
p_sample = sorted(list(pred_dates))[:2]
v_sample = sorted(list(vol_dates))[:2]
print(f"pred sample: {[repr(x) for x in p_sample]}")
print(f"vol sample: {[repr(x) for x in v_sample]}")
missing = pred_dates - vol_dates
print(f"missing: {len(missing)} / {len(pred_dates)}")
if missing and vol_dates:
    p0, v0 = p_sample[0], v_sample[0]
    print(f"p0 type: {type(p0)}, v0 type: {type(v0)}")
    print(f"p0 == v0: {p0 == v0}")
    print(f"p0 hash: {hash(p0)}, v0 hash: {hash(v0)}")
