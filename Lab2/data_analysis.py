import pandas as pd

def summarize_numeric(df_num):
    rows = []
    total = len(df_num)
    for col in df_num.columns:
        s = df_num[col]
        n_nonnull = s.notna().sum()
        miss_pct = 100 * (1 - n_nonnull / total)
        card = s.nunique(dropna=True)
        rows.append({
            "column": col,
            "N": total,
            "% missing": round(miss_pct, 2),
            "cardinality": card,
            "min": s.min(),
            "Q1": s.quantile(0.25),
            "median": s.median(),
            "Q3": s.quantile(0.75),
            "max": s.max(),
            "mean": s.mean(),
            #"mode": s.mode().iloc[0] if not s.mode().empty else None,
            "std": s.std()
        })
    return pd.DataFrame(rows)

def summarize_categorical(df_cat):
    rows = []
    total = len(df_cat)
    for col in df_cat.columns:
        s = df_cat[col].astype("object")
        n_nonnull = s.notna().sum()
        miss_pct = 100 * (1 - n_nonnull / total)
        card = s.nunique(dropna=True)
        vc = s.value_counts(dropna=True) #value_counts - counts occurences of values
        mode1, mode2 = (vc.index[0] if len(vc) > 0 else np.nan,
                        vc.index[1] if len(vc) > 1 else np.nan)
        rows.append({
            "column": col,
            "N": total,
            "% missing": round(miss_pct, 2),
            "cardinality": card,
            "mode": mode1,
            "mode freq": int(vc.iloc[0]) if len(vc) > 0 else np.nan, #iloc-to get position in dataframe, e.g iloc[0]-get 1st row
            "mode %": round(100 * vc.iloc[0] / total, 2) if len(vc) > 0 else np.nan,
            "mode 2": mode2,
            "mode 2 freq": int(vc.iloc[1]) if len(vc) > 1 else np.nan,
            "mode 2 %": round(100 * vc.iloc[1] / total, 2) if len(vc) > 1 else np.nan
        })
    return pd.DataFrame(rows)