from typing_extensions import Literal, Any, Callable, List
from collections import Counter

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import glob
from catboost_worker import *
from datetime import datetime


features2drop = ['specialization_sum_agg_payments__g_contract__sum__all__sum__ALL_TIME',
 'specialization_sum_agg_ks2__g_contract__total_sum__all__sum__ALL_TIME',
 'agg_spass_applications__g_specialization__appl_count_week__mean__ALL_TIME',
 'building_id',
 'agg_scontrol__g_contractor__close_delay__defect_type_labour_protection__mean__ALL_TIME',
 'agg_scontrol__g_contractor__close_delay__defect_type_arch_supervision__mean__ALL_TIME',
 'counteragent_sum_agg_payments__g_contract__sum__all__sum__ALL_TIME',
 'agg_sroomer__g_contractor__sroomer_id__count__6M',
 'agg_ArbitrationCases__g_contractor__DefendantSum__sum__ALL_TIME',
 'agg_tender_proposal__g_contractor__id__ALL__countDistinct__52W',
 'counteragent_mean_agg_spass_applications__g_contract__appl_count_week__mean__ALL_TIME',
 'agg_scontrol__g_contractor__close_delay__defect_type_tech_supervision__mean__ALL_TIME',
 'agg_tender_proposal__g_contractor__id__ALL__countDistinct__ALL_TIME',
'agg_all_contracts__g_contract__abs_change_price_last_ds__isMain__last__ALL_TIME',
'agg_all_contracts__g_contract__abs_change_price_last_ds__isMain__mean__ALL_TIME',
'agg_all_contracts__g_contract__rel_change_price_last_ds__isMain__last__ALL_TIME',
'agg_all_contracts__g_contract__rel_change_price_last_ds__isMain__mean__ALL_TIME',
'agg_sroomer__g_contractor__sroomer_id__count__3M',
'agg_sroomer__g_contractor__sroomer_id__count__6M',
'agg_sroomer__g_contractor__sroomer_id__count__12M',
'agg_sroomer__g_contractor__sroomer_id__count__ALL_TIME',
'agg_FinanceAndTaxesFTS__g_contractor__TaxArrearsSum__last__ALL_TIME',
'agg_FinanceAndTaxesFTS__g_contractor__TaxPenaltiesSum__last__ALL_TIME'
]

unique_ids_features = ["contractor_id", "project_id", "specialization_id"]
contract_id_col = "contract_id"
target_column = "default6"

graph_features = pd.read_csv("contractor_graph_features_v2.csv")
graph_features2 = pd.read_csv("contractor_graph_features_v3.csv")

inference_worker = CatBoostWorker(
    task='class',
    seed=1,
    device='CPU'
)

PATH_TO_CHECKPOINTS = "models_pipeline2"

loaded_models = []
for model_path in sorted(glob.glob(f"{PATH_TO_CHECKPOINTS}/*")):
  print(model_path)
  model = catboost.CatBoostClassifier()
  model.load_model(model_path)
  loaded_models.append(model)

inference_worker.folds_models = loaded_models
inference_worker.folds_scores = [0.9687122905399921,
 0.9668696984343785,
 0.9588232580304867,
 0.9605463484648421,
 0.9690963623437471] # подставить веса



def get_predict(df):
    df = df.merge(graph_features, on="contractor_id", how="left")
    df = df.merge(graph_features2, on="contractor_id", how="left")

    isna_stat = df.isna().sum()[df.isna().sum() > 0]
    for col in isna_stat.index:
        if col in ["contract_date", "report_date"]:
            continue
        else:
            df[col] = SimpleImputer(strategy="median").fit_transform(df[col].values.reshape(-1, 1))

    df["contract_length"] = 0
    df["sample_num"] = 0

    for i in df["contract_id"].unique():
        df.loc[df["contract_id"] == i, "contract_length"] = (df["contract_id"] == i).sum()

    dates_columns = df.select_dtypes('object').columns
    for date_col in dates_columns:
        df[date_col] = df[date_col].apply(lambda x: datetime.fromisoformat(x).timestamp())

    for cat_feat in unique_ids_features + ["contract_id"]:
        df[cat_feat] = df[cat_feat].astype(np.int16)

    test_pool = catboost.Pool(data=df.drop(columns=contract_id_col))

    probs = inference_worker.inference_model(
        return_probs=True,
        use_kfold_models=True,
        test_pool=test_pool
    )

    w = inference_worker.folds_scores
    probs = (np.stack(probs) * np.array(w).reshape(-1, 1)).sum(axis=0) / np.sum(w)
    return probs

