import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import joblib

ROLL_K = 6
PAD = 1.0
TRAIN_CUTOFF = pd.Timestamp('2026-06-15')

GAP_THRESH = {
    'Vac_Ext': 6.0,
    'Lazer_65W': 35.0,
}
SUB_DEVICES = list(GAP_THRESH.keys())
FEATURES = ['dt0', 'dt1', 'dt2', 'r01', 'roll_mean', 'roll_std', 'tod_sin', 'tod_cos']

df = pd.read_csv('/mnt/user-data/uploads/merged.csv')
df['Timestamp_signal'] = pd.to_datetime(df['Timestamp_signal'])
df['Device_utiliser'] = df['Device_utiliser'].replace({
    'Signal5': 'Lazer_65W', 'Lazer_60W': 'Lazer_65W',
    'Signal2': 'Inverter_resin_print', 'Grand_Lazer': 'Inverter_resin_print'
})
df = df.sort_values('Timestamp_signal').drop_duplicates(subset='Timestamp_signal', keep='first')

main_df = df[df['Device_utiliser'] == 'Main_Mtr'].sort_values('Timestamp_signal').reset_index(drop=True)
T_main = main_df['Timestamp_signal'].values.astype('datetime64[ns]').astype('int64') / 1e9

def build_runs(t_sorted, gap_thresh, pad):
    if len(t_sorted) == 0:
        return np.empty((0, 2))
    breaks = np.where(np.diff(t_sorted) > gap_thresh)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [len(t_sorted) - 1]))
    return np.column_stack([t_sorted[starts] - pad, t_sorted[ends] + pad])

runs = {}
for d in SUB_DEVICES:
    t = np.sort(df[df['Device_utiliser'] == d]['Timestamp_signal'].values.astype('datetime64[ns]').astype('int64') / 1e9)
    runs[d] = build_runs(t, GAP_THRESH[d], PAD)
    total = (runs[d][:, 1] - runs[d][:, 0]).sum()
    print(f"{d}: {len(runs[d])} runs, total active time = {total/3600:.2f} hrs ({total/3600/1104*100:.2f}% of span)")

n = len(T_main)
in_run = {}
for d in SUB_DEVICES:
    starts, ends = runs[d][:, 0], runs[d][:, 1]
    idx = np.searchsorted(starts, T_main, side='right') - 1
    idx = np.clip(idx, 0, len(starts) - 1)
    inside = np.zeros(n, dtype=bool)
    valid = idx >= 0
    inside[valid] = (T_main[valid] >= starts[idx[valid]]) & (T_main[valid] <= ends[idx[valid]])
    in_run[d] = inside

# NOTE: completely ignoring Prntr_3D here — a Main_Mtr pulse can be inside Vac_Ext's
# run AND inside Prntr_3D's run simultaneously; we only care about Vac_Ext/Lazer_65W overlap with EACH OTHER
n_matches = in_run['Vac_Ext'].astype(int) + in_run['Lazer_65W'].astype(int)

labels = np.full(n, 'neither', dtype=object)
labels[in_run['Vac_Ext'] & (n_matches == 1)] = 'Vac_Ext'
labels[in_run['Lazer_65W'] & (n_matches == 1)] = 'Lazer_65W'
labels[n_matches >= 2] = 'overlap'  # Vac_Ext and Lazer_65W both active at once (rare, real conflict)

print("\nLabel distribution (Problem 2: Vac_Ext / Lazer_65W / neither, Prntr_3D ignored):")
print(pd.Series(labels).value_counts())

# ---------------- Features (same construction as before) ----------------
dt = np.diff(T_main, prepend=T_main[0]); dt[0] = np.nan
dt_series = pd.Series(dt)
roll_mean = dt_series.rolling(ROLL_K, min_periods=ROLL_K).mean().values
roll_std = dt_series.rolling(ROLL_K, min_periods=ROLL_K).std().values
dt1 = np.roll(dt, 1); dt1[0:2] = np.nan
dt2 = np.roll(dt, 2); dt2[0:3] = np.nan
with np.errstate(divide='ignore', invalid='ignore'):
    r01 = np.where(dt1 > 0, dt / dt1, 1.0)

ts = main_df['Timestamp_signal']
tod_sec = (ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second).values
tod_sin = np.sin(2 * np.pi * tod_sec / 86400)
tod_cos = np.cos(2 * np.pi * tod_sec / 86400)

feat_df = pd.DataFrame({
    'dt0': dt, 'dt1': dt1, 'dt2': dt2, 'r01': r01,
    'roll_mean': roll_mean, 'roll_std': roll_std,
    'tod_sin': tod_sin, 'tod_cos': tod_cos,
    'label': labels, 'timestamp': ts.values
}).dropna().reset_index(drop=True)

print(f"\nFeature rows after warm-up drop: {len(feat_df)}")

feat_df['timestamp'] = pd.to_datetime(feat_df['timestamp'])
train_mask = (feat_df['timestamp'] < TRAIN_CUTOFF) & (feat_df['label'] != 'overlap')
test_mask = feat_df['timestamp'] >= TRAIN_CUTOFF
train = feat_df[train_mask]
test = feat_df[test_mask]
test_eval = test[test['label'] != 'overlap']
test_overlap = test[test['label'] == 'overlap']

print(f"\nTrain rows: {len(train)}")
print("Train label distribution:")
print(train['label'].value_counts())
print(f"\nTest rows (eval): {len(test_eval)}, overlap held out: {len(test_overlap)}")
print("Test label distribution:")
print(test_eval['label'].value_counts())

X_train, y_train = train[FEATURES].values, train['label'].values
X_test, y_test = test_eval[FEATURES].values, test_eval['label'].values

model = RandomForestClassifier(
    n_estimators=30, max_depth=6, class_weight='balanced',
    random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT (Problem 2: Vac_Ext / Lazer_65W / neither)")
print("=" * 60)
preds = model.predict(X_test)
print(classification_report(y_test, preds, digits=3, zero_division=0))
print(f"Balanced accuracy: {balanced_accuracy_score(y_test, preds):.3f}")

labels_order = ['Vac_Ext', 'Lazer_65W', 'neither']
cm = confusion_matrix(y_test, preds, labels=labels_order)
print("\nConfusion matrix (rows=true, cols=pred), order:", labels_order)
print(cm)

print("\nFeature importances:")
for f, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {f}: {imp:.3f}")

joblib.dump(model, '/home/claude/nilm_rf_model_problem2.joblib')
import os
print(f"\nModel size: {os.path.getsize('/home/claude/nilm_rf_model_problem2.joblib')} bytes")
