from typing_extensions import Literal, Any, Callable, List
import catboost
import numpy as np
from sklearn.model_selection import KFold, StratifiedGroupKFold, StratifiedKFold
import sklearn
import pandas as pd


class CatBoostWorker:
  def __init__(
      self,
      task: Literal['reg', 'class'] = "reg",
      seed: int = 1,
      device: Literal['CPU', 'GPU'] ="CPU"
  ) -> None:
      pass

      """
        Column name:  | Default value:      | Column description:
                      |                     |
        task          | default: 'reg'      | whether this catboost will solve regression('reg') on classification('class') task
        seed          | default: 1          | random state to make result reproducable
        device        | default: 'CPU'      | device to compute ('CPU'/'GPU')

      """

      # Init-params
      self.seed = seed
      self.device = device
      self.task = task


      # Default splits data
      self.train_data = None
      self.val_data = None
      self.test_data = None


      # CV setting
      self.num_folds = None
      self.folds = None
      self.cross_val_test_pool = None
      self.cross_val_score = None

      # Models settings
      self.default_model_params = {
          "iterations" : 100,
          "border_count" : 254
      }
      self.default_train_params = {}


      # Models

      self.model = None
      self.folds_models = None


  def split_data(
      self,
      data: Any = None,
      target_column: str = "target" ,
      fold: bool = False,
      splits: dict = {"train" : 1.0, "val" : None, "test" : None},
      extra_features_columns: dict  = {},
      verbose: bool = True
  ) -> None:

    """

      Column name:           | Default value:                                        | Column description:
                             |                                                       |
      data                   | default: None                                         | dataframe
      target_column          | default: 'target'                                     | name of target column
      splits                 | default: {"train" : 1.0, "val" : None, "test" : None} | train/val/test splits sizes
      extra_features_columns | default: {}                                           | columns with specifical data for catboost like cat_features, embedding_features, text_features
      verbose                | default: True                                         | whether show progress of data initalizing or not

    """

    self.extra_features_columns = extra_features_columns

    # getting train/val/test actual sizes
    train_size = int(len(data) * splits["train"])

    val_size = 0.0
    test_size = 0.0
    if splits["val"] != None:
      val_size = int(len(data) * splits["val"])

    if splits["test"] != None:
      test_size = int(len(data) * splits["test"])

    if verbose:

      print(f"train size: {train_size}, val size: {val_size}, test_size: {test_size}")

    # shake data
    np.random.seed(self.seed)
    data = data.sample(frac=1).reset_index(drop=True)

    # saving train/val/test splits to catboost Pools
    X = data.drop(columns=[target_column])
    Y = data[target_column]

    X_train_split = X.iloc[:train_size, :]
    Y_train_split = Y.iloc[:train_size]
    self.train_data = (X_train_split, Y_train_split)

    if val_size != 0.0:

      X_val_split = X.iloc[train_size:train_size + val_size, :].reset_index(drop=True)
      Y_val_split = Y.iloc[train_size:train_size + val_size].reset_index(drop=True)
      self.val_data = (X_val_split, Y_val_split)

    if test_size != 0.0:

      X_test_split = X.iloc[train_size + val_size:, :].reset_index(drop=True)
      Y_test_split = Y.iloc[train_size + val_size:].reset_index(drop=True)
      self.test_data = (X_test_split, Y_test_split)

    if verbose:

      print("all pools have been succesfully saved :)")


  def split_folds(
      self,
      data: Any = None,
      target_column: str = "target" ,
      cv_type: Literal['Classic', 'Stratified'] = "Classic",
      num_folds: int = 5,
      test_size: int = 0.1,
      extra_features_columns: dict  = {},
      verbose: bool = True,
      groups : List = []
  ) -> None:

    """

      Column name:           | Default value:                                        | Column description:
                             |                                                       |
      data                   | default: None                                         | dataframe
      target_column          | default: 'target'                                     | name of target column
      num_folds              | default: 5                                            | number of folds for create
      cv_type                | default: 'Classic'                                    | type of cross-validation
      test_size              | default: 0.1                                          | test-size
      verbose                | default: True                                         | whether show progress of data initalizing or not
      groups                 | default: []                                           | Groups for StratifiedGroupKFold
    """

    self.extra_features_columns = extra_features_columns

    self.num_folds = num_folds

    # test split
    if test_size != 0.0:

      cv_size = int(len(data) * (1 - test_size))

      #shake data
      np.random.seed(self.seed)
      data = data.sample(frac=1).reset_index(drop=True)

      # saving train/val/test splits to catboost Pools
      X = data.drop(columns=[target_column])
      Y = data[target_column]

      X_cv_data = X.iloc[:cv_size, :]
      Y_cv_data = Y.iloc[:cv_size]

      X_test_data = X.iloc[cv_size:, :]
      Y_test_data = Y.iloc[cv_size:]
      self.cross_val_test_data = (X_test_data, Y_test_data)

      if verbose:

        print(f"cv_size: {cv_size}, test_size: {len(data) - cv_size}")

    else:

      X_cv_data = data.drop(columns=[target_column])
      Y_cv_data = data[target_column]

    # getting folds
    if cv_type == "Classic":

      kf = sklearn.model_selection.KFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)

    if cv_type == "Stratified":

      kf = sklearn.model_selection.StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)

    if cv_type == "StratifiedGroup":

      kf = sklearn.model_selection.StratifiedGroupKFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)

    if cv_type == "Group":

      kf = sklearn.model_selection.GroupKFold(n_splits=self.num_folds)

    if self.task == "reg":

      Y_to_split_on = pd.qcut(Y_cv_data, self.num_folds, labels = np.arange(self.num_folds))

    folds_data = {}

    split_args = dict(
        X=X_cv_data,
        y=(Y_to_split_on if (self.task == "reg" and cv_type == "Stratified") else Y_cv_data)
    )

    if cv_type in ["Group", "StratifiedGroup"]:
      split_args["groups"] = groups

    for fold, (train_idxs, val_idxs) in enumerate(kf.split(
        **split_args
    )):

      X_train_fold, y_train_fold = X_cv_data.iloc[train_idxs, :], Y_cv_data.iloc[train_idxs]
      X_val_fold, y_val_fold = X_cv_data.iloc[val_idxs, ], Y_cv_data.iloc[val_idxs]


      folds_data[f"fold_{fold}"] = ((X_train_fold, y_train_fold), (X_val_fold, y_val_fold))

      if verbose:

        print(f"fold {fold} saved")

    self.folds = folds_data

  def init_model_params(
      self,
      model_params: dict = {"iterations" : 100},
      train_params: dict = {}
  ) -> None:

      """
        Column name:           | Default value:                                        | Column description:
                               |                                                       |
        model_params           | default: {"iterations" : 100}                         | dict with catboost model initialization params
        train_params           | default: {}                                           | dict with catboost model .fit params

      """

      self.default_model_params = model_params
      self.default_train_params = train_params


  def train_model(self) -> None:

    if self.task == "reg":

      self.model = catboost.CatBoostRegressor(
          **self.default_model_params,
          random_state=self.seed,
          task_type=self.device
      )

    if self.task == "class":

      self.model = catboost.CatBoostClassifier(
          **self.default_model_params,
          random_state=self.seed,
          task_type=self.device
      )

    train_pool = catboost.Pool(data=self.train_data[0], label=self.train_data[1], **self.extra_features_columns)

    eval_sets = [train_pool] + ([] if self.val_data == None else [catboost.Pool(data=self.val_data[0], label=self.val_data[1], **self.extra_features_columns)])

    self.model.fit(X=train_pool, eval_set=eval_sets, **self.default_train_params)

  def train_kfold(
      self,
      eval_metric: Callable = None,
      verbose: bool = True,
  ) -> None:

    """
        Column name:           | Default value:                       | Column description:

        eval_metric            | default: None                        | metric to evaluate cross-valadation on
        verbose                | default: True                        | whether to verbose evaluation process or no
    """

    folds_scores = []
    folds_models = []

    for fold in range(self.num_folds):
      train_fold_data, val_fold_data = self.folds[f"fold_{fold}"]

      train_fold_pool = catboost.Pool(data=train_fold_data[0], label=train_fold_data[1], **self.extra_features_columns)
      val_fold_pool = catboost.Pool(data=val_fold_data[0], label=val_fold_data[1], **self.extra_features_columns)

      if self.task == "reg":

        fold_model = catboost.CatBoostRegressor(
            **self.default_model_params,
            random_state=self.seed,
            task_type=self.device
        )

      if self.task == "class":

        fold_model = catboost.CatBoostClassifier(
            **self.default_model_params,
            random_state=self.seed,
            task_type=self.device
        )

      eval_sets = [train_fold_pool, val_fold_pool]

      fold_model.fit(X=train_fold_pool, eval_set=eval_sets, **self.default_train_params)

      fold_val_preds = fold_model.predict_proba(val_fold_pool)[:, 1]
      fold_val_score = eval_metric(val_fold_pool.get_label(), fold_val_preds)

      folds_scores.append(fold_val_score)
      folds_models.append(fold_model)

      if verbose:
        print(f"FOLD {fold}, VAL SCORE: {fold_val_score}")

    if verbose:
      print()
      print(f"mean val score per folds {np.mean(folds_scores)}")
      print(f"mean val score per folds with regularization {np.mean(folds_scores) - np.std(folds_scores)}")

    self.cross_val_score = np.mean(folds_scores)
    self.folds_models = folds_models
    self.folds_scores = folds_scores


  def inference_model(
    self,
    return_probs: bool = False,
    target_threshold: float = 0.5,
    use_kfold_models: bool = False,
    test_pool = None,
  ):

    """
        Column name:           | Default value:                       | Column description:

        return_probs           | default: False                       | whether to return probes or preds
        target_threshold       | default: 0.5                         | threshold to make preds from probes
        use_kfold_models       | default: False                       | use or not KFold models to eval on test_set
        test_pool              | default: None                        | custom test_set to evaluate on
    """

    data_to_inference = None

    if test_pool is not None:
      data_to_inference = test_pool
    elif use_kfold_models:
      data_to_inference = catboost.Pool(data=self.cross_val_test_pool[0], label=self.cross_val_test_pool[1], **self.extra_features_columns)
    else:
      data_to_inference = catboost.Pool(data=self.test_data[0], label=self.test_data[1], **self.extra_features_columns)

    if data_to_inference is None:
      raise Exception("no data to inference")


    if use_kfold_models:

      all_models_preds = []
      for model in self.folds_models:

        model_preds = model.predict_proba(data_to_inference)[:, 1]
        all_models_preds.append(model_preds)

      if return_probs:
        return all_models_preds
      else:

        probes = np.stack(all_models_preds).mean(axis=0)
        final_preds = (probes > target_threshold).astype(int)
        return final_preds

    else:

      probes = self.model.predict_proba(data_to_inference)

      if return_probs:

        return probes

      else:

        return (probes[:, 1] > target_threshold).astype(int)