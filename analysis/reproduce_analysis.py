"""Reproduce core analyses for the PLOS Water submission.

Usage
-----
python analysis/reproduce_analysis.py --data-dir /path/to/SIHBS/files --output-dir derived_outputs

Required SIHBS files (original filenames):
- hh(2).dta
- hhm(2).dta
- agland(2).dta
- crops(2).dta
- livestock_own(1).dta

The script does not redistribute SIHBS microdata. It reconstructs the final livelihood
classification, seasonal water-source transition outcome, weighted descriptive outputs,
full regression model outputs, sensitivity models, destination models, and predictive
validation results used in the manuscript.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance


def water_family(x):
    if pd.isna(x):
        return np.nan
    x = str(x)
    if x in {"Piped water into dwelling", "Piped water to yard/plot"}:
        return "Private piped"
    if x == "Public tap/standpipe":
        return "Public network"
    if x in {"Tubewell/borehole", "Protected dug well", "Unprotected dug well",
             "Protected spring", "Unprotected spring"}:
        return "Groundwater"
    if x in {"Rainwater collection", "Natural surface water (river, dam, lake)",
             "Surface water (pond, stream, canal, irrigation channels)", "Water catchment"}:
        return "Rain/surface"
    if x in {"Tanker-truck", "Cart with small tank/drum", "Bottled water"}:
        return "Delivered/vendor"
    return "Other/social"


def broad_family(x):
    f = water_family(x)
    return "Piped/network" if f in {"Private piped", "Public network"} else f


def wmean(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(w)
    return np.average(x[m], weights=w[m]) if m.any() else np.nan


def model_to_rows(model, model_name):
    rows = []
    for term, beta in model.params.items():
        se = model.bse[term]
        rows.append({
            "model": model_name,
            "term": term,
            "beta": beta,
            "SE": se,
            "OR": np.exp(beta),
            "lower95_model_based": np.exp(beta - 1.96 * se),
            "upper95_model_based": np.exp(beta + 1.96 * se),
            "p_model_based": model.pvalues[term],
            "N": int(model.nobs),
            "AIC": model.aic,
            "deviance": model.deviance,
        })
    return rows


def main(data_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    hh = pd.read_stata(data_dir / "hh(2).dta", convert_categoricals=True)
    hh_raw = pd.read_stata(data_dir / "hh(2).dta", convert_categoricals=False)
    hhm = pd.read_stata(data_dir / "hhm(2).dta", columns=["hid","qsn1_03","qsn1_04","qsn2_05"], convert_categoricals=True)
    agland = pd.read_stata(data_dir / "agland(2).dta", convert_categoricals=True)
    crops_raw = pd.read_stata(data_dir / "crops(2).dta", convert_categoricals=False)
    livestock = pd.read_stata(data_dir / "livestock_own(1).dta", convert_categoricals=False)

    df = hh.copy()
    df["rain_source"] = df["hh_water_type"].astype(object)
    df["dry_source"] = np.where(df["hh2_10"].eq("Yes"), df["hh_water_type"].astype(object), df["hh2_11"].astype(object))
    df["rain_family"] = df["rain_source"].map(water_family)
    df["dry_family"] = df["dry_source"].map(water_family)
    df["substantive_transition"] = (df["rain_family"] != df["dry_family"]).astype(int)
    df["any_source_change"] = (~df["hh2_10"].eq("Yes")).astype(int)
    df["rain_family_broad"] = df["rain_source"].map(broad_family)
    df["dry_family_broad"] = df["dry_source"].map(broad_family)
    df["transition_broad"] = (df["rain_family_broad"] != df["dry_family_broad"]).astype(int)

    agland_hids = set(agland["hid"].unique())
    crop_hids = set(crops_raw.loc[crops_raw["crop6_21"].eq(1), "hid"].unique())
    agriculture_hids = agland_hids | crop_hids
    livestock_hids = set(livestock["hid"].unique())
    df["agriculture_hh"] = df["hid"].isin(agriculture_hids)
    df["livestock_hh"] = df["hid"].isin(livestock_hids)
    def livgrp(r):
        if r["agriculture_hh"] and r["livestock_hh"]: return "Mixed crop-livestock"
        if r["livestock_hh"]: return "Livestock only"
        if r["agriculture_hh"]: return "Agriculture only"
        return "Other"
    df["livelihood"] = df.apply(livgrp, axis=1)

    heads = hhm[hhm["qsn1_04"].astype(str).eq("Head")].copy()
    heads["female_head"] = heads["qsn1_03"].astype(str).eq("Female").astype(float)
    heads["school_cat"] = np.where(heads["qsn2_05"].isna(), "Unknown", np.where(heads["qsn2_05"].astype(str).eq("Yes"), "Ever attended", "Never attended"))
    df = df.merge(heads[["hid","female_head","school_cat"]], on="hid", how="left", validate="1:1")
    df["road_km"] = pd.to_numeric(hh_raw["hh1_17"], errors="coerce").values
    df["road_cap95"] = df["road_km"].clip(upper=df["road_km"].quantile(.95))
    df["road_cap99"] = df["road_km"].clip(upper=df["road_km"].quantile(.99))
    df["w_norm"] = df["wgt"] / df["wgt"].mean()

    # Core audit
    audit = pd.DataFrame({"metric":["Households","Any source change weighted %","Substantive transition weighted %","Agriculture only N","Livestock only N","Mixed crop-livestock N","Other N"],
                          "value":[len(df),100*wmean(df["any_source_change"],df["wgt"]),100*wmean(df["substantive_transition"],df["wgt"]),
                                   int((df["livelihood"]=="Agriculture only").sum()),int((df["livelihood"]=="Livestock only").sum()),
                                   int((df["livelihood"]=="Mixed crop-livestock").sum()),int((df["livelihood"]=="Other").sum())]})
    audit.to_csv(output_dir/"core_audit_counts.csv", index=False)

    # Pathways and destination distribution
    sw = df[df["substantive_transition"].eq(1)].copy()
    sw["pathway"] = sw["rain_family"] + " -> " + sw["dry_family"]
    pth = sw.groupby("pathway").agg(N=("hid","size"), W=("wgt","sum")).reset_index()
    pth["weighted_share_percent"] = 100*pth["W"]/pth["W"].sum()
    pth.sort_values("weighted_share_percent", ascending=False).drop(columns="W").to_csv(output_dir/"weighted_transition_pathways.csv", index=False)

    dest = sw.groupby("dry_family").agg(N=("hid","size"),W=("wgt","sum")).reset_index()
    dest["weighted_share_percent"] = 100*dest["W"]/dest["W"].sum()
    dest.sort_values("weighted_share_percent",ascending=False).drop(columns="W").to_csv(output_dir/"destination_distribution.csv",index=False)

    # Nested models
    base = """C(livelihood, Treatment(reference='Other'))
    + C(ea_type_n, Treatment(reference='Urban'))
    + hhsize + female_head
    + C(school_cat, Treatment(reference='Never attended'))
    + C(region_n)"""
    nested_specs = {
        "Model 1: livelihood + context":"substantive_transition ~ "+base,
        "Model 2: + rainy-season source":"substantive_transition ~ "+base+" + C(rain_family, Treatment(reference='Private piped'))",
        "Model 3: + road distance":"substantive_transition ~ "+base+" + C(rain_family, Treatment(reference='Private piped')) + road_cap99",
    }
    rows=[]
    for name,formula in nested_specs.items():
        dat = df.dropna(subset=["road_cap99"]) if "road" in name else df
        m=smf.glm(formula,data=dat,family=sm.families.Binomial(),freq_weights=dat["w_norm"]).fit()
        rows += model_to_rows(m,name)
    pd.DataFrame(rows).to_csv(output_dir/"S1_Table_Full_Nested_Logistic_Models.csv",index=False)

    # Destination models
    sw["to_groundwater"] = sw["dry_family"].eq("Groundwater").astype(int)
    sw["to_vendor"] = sw["dry_family"].eq("Delivered/vendor").astype(int)
    common = """C(livelihood, Treatment(reference='Other'))
    + C(ea_type_n, Treatment(reference='Urban')) + hhsize + female_head
    + C(school_cat, Treatment(reference='Never attended')) + C(rain_family) + C(region_n)"""
    rows=[]
    for outcome,name in [("to_groundwater","Groundwater destination"),("to_vendor","Delivered/vendor destination")]:
        m=smf.glm(f"{outcome} ~ {common}",data=sw,family=sm.families.Binomial(),freq_weights=sw["w_norm"]).fit()
        rows += model_to_rows(m,name)
    pd.DataFrame(rows).to_csv(output_dir/"S2_Table_Full_Destination_Models.csv",index=False)

    # Robustness models - full output
    base_terms = """C(livelihood, Treatment(reference='Other'))
    + C(ea_type_n, Treatment(reference='Urban')) + hhsize + female_head
    + C(school_cat, Treatment(reference='Never attended'))"""
    with_region = base_terms + " + C(region_n)"
    with_source_region = with_region + " + C(rain_family, Treatment(reference='Private piped'))"
    specs = [
        ("Primary weighted","substantive_transition ~ "+with_source_region,df,"w_norm"),
        ("Unweighted","substantive_transition ~ "+with_source_region,df,None),
        ("No region controls","substantive_transition ~ "+base_terms+" + C(rain_family, Treatment(reference='Private piped'))",df,"w_norm"),
        ("Alternative broad source grouping","transition_broad ~ "+with_region+" + C(rain_family_broad, Treatment(reference='Piped/network'))",df,"w_norm"),
        ("Any questionnaire source change","any_source_change ~ "+with_source_region,df,"w_norm"),
        ("Road 95th-pct cap","substantive_transition ~ "+with_source_region+" + road_cap95",df.dropna(subset=["road_cap95"]),"w_norm"),
        ("Road 99th-pct cap","substantive_transition ~ "+with_source_region+" + road_cap99",df.dropna(subset=["road_cap99"]),"w_norm"),
        ("Road uncapped","substantive_transition ~ "+with_source_region+" + road_km",df.dropna(subset=["road_km"]),"w_norm"),
    ]
    rows=[]
    for name,formula,dat,wt in specs:
        m=smf.glm(formula,data=dat,family=sm.families.Binomial(),freq_weights=(dat[wt] if wt else None)).fit()
        rows += model_to_rows(m,name)
    pd.DataFrame(rows).to_csv(output_dir/"S3_Table_Full_Robustness_Models.csv",index=False)

    # ML validation
    features=["rain_family","livelihood","ea_type_n","road_cap99","hhsize","female_head","school_cat"]
    cat_features=["rain_family","livelihood","ea_type_n","school_cat"]
    num_features=["road_cap99","hhsize","female_head"]
    X=df[features].copy(); y=df["substantive_transition"].astype(int).to_numpy(); groups=df["region_n"].astype(str).to_numpy(); weights=df["wgt"].astype(float).to_numpy()
    prep_sparse=ColumnTransformer([("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat_features),
                                   ("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num_features)])
    prep_dense=ColumnTransformer([("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cat_features),
                                  ("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num_features)])
    estimators={"Logistic regression":LogisticRegression(max_iter=1000),
                "Random forest":RandomForestClassifier(n_estimators=400,min_samples_leaf=5,random_state=42,n_jobs=-1),
                "Gradient boosting":HistGradientBoostingClassifier(learning_rate=.06,max_iter=250,max_leaf_nodes=15,l2_regularization=1.0,random_state=42)}
    pooled=[]; folds=[]
    for validation, splitter in [("Stratified random 5-fold",StratifiedKFold(n_splits=5,shuffle=True,random_state=42)),("Region-grouped 5-fold",GroupKFold(n_splits=5))]:
        for model_name,est in estimators.items():
            preds=np.full(len(y),np.nan)
            split_iter=splitter.split(X,y,groups=groups) if isinstance(splitter,GroupKFold) else splitter.split(X,y)
            for fold,(tr,te) in enumerate(split_iter,1):
                prep=prep_dense if model_name=="Gradient boosting" else prep_sparse
                pipe=Pipeline([("prep",prep),("model",est)])
                pipe.fit(X.iloc[tr],y[tr],model__sample_weight=weights[tr])
                p=pipe.predict_proba(X.iloc[te])[:,1]; preds[te]=p
                folds.append({"validation":validation,"model":model_name,"fold":fold,"ROC_AUC":roc_auc_score(y[te],p,sample_weight=weights[te]),"PR_AUC":average_precision_score(y[te],p,sample_weight=weights[te]),"Brier":brier_score_loss(y[te],p,sample_weight=weights[te]),"N_test":len(te)})
            mask=np.isfinite(preds)
            pooled.append({"validation":validation,"model":model_name,"ROC_AUC":roc_auc_score(y[mask],preds[mask],sample_weight=weights[mask]),"PR_AUC":average_precision_score(y[mask],preds[mask],sample_weight=weights[mask]),"Brier":brier_score_loss(y[mask],preds[mask],sample_weight=weights[mask])})
    pd.DataFrame(pooled).to_csv(output_dir/"S4_Table_ML_Validation_Pooled.csv",index=False)
    pd.DataFrame(folds).to_csv(output_dir/"S4_Table_ML_Validation_Fold_Level.csv",index=False)

    # Agricultural subsystem
    ag2=agland.copy()
    ag2["rain_irrig"] = ag2["land6_07"].astype(str).eq("Rain")
    ag2["seasons_num"] = pd.to_numeric(ag2["land6_05"].astype(str), errors="coerce")
    agh=ag2.groupby("hid").agg(all_rain=("rain_irrig","all"),max_seasons=("seasons_num","max")).reset_index()
    agh["irrigation_group"]=np.where(agh["all_rain"],"Rain-dependent","Any non-rain source")
    agh=agh.merge(df[["hid","substantive_transition","wgt"]],on="hid",how="left")
    rows=[]
    for grp,g in agh.groupby("irrigation_group"):
        rows.append({"irrigation_group":grp,"N":len(g),"weighted_substantive_transition_percent":100*np.average(g["substantive_transition"],weights=g["wgt"]),"weighted_mean_max_seasons_utilized":np.average(g["max_seasons"],weights=g["wgt"])})
    pd.DataFrame(rows).to_csv(output_dir/"agricultural_subsystem.csv",index=False)

    print(f"Wrote reproducibility outputs to {output_dir}")

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,default=Path("derived_outputs"))
    args=ap.parse_args()
    main(args.data_dir,args.output_dir)
