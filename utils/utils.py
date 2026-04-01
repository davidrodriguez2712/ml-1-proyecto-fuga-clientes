import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
import shap

class PreprocessingData:
    def __init__(self):
        pass

    def split_type_features(self, dataset: pd.DataFrame):
        """Permite separar variables categóricas, numéricas y de tiempo"""

        numerical = ["int64", "float64"]
        categorical = ["object"]
        datetime = ["datetime64[ns]"]
        self.numerical_features = []
        self.categorial_features = []
        self.datetime_features = []
        self.wo_feature = []
        self.dataset_origin = dataset

        features_names = dataset.columns.to_list()
        #print(features_names)
        features_type = dataset.dtypes.astype("str").values.tolist()
        #print(features_type)

        for (feature_name, feature_type) in list(zip(features_names, features_type)):
            if feature_type in datetime:
                self.datetime_features.append(feature_name)
            if feature_type in numerical:
                self.numerical_features.append(feature_name)
            if feature_type in categorical:
                self.categorial_features.append(feature_name)

        return self.numerical_features, self.categorial_features, self.datetime_features
    
    def unique_values(self, dataset: pd.DataFrame):
        """Devuelve la cantidad de valores únicos para cada feature proporcionada"""
        features = dataset.columns.to_list()
        cant_unicos = []

        for i in features:
            unicos = len(dataset[i].unique().tolist())
            cant_unicos.append(unicos)

        reporte_unicos = {
            "feature": features,
            "cantidad de únicos": cant_unicos
        }

        reporte_unicos = pd.DataFrame(reporte_unicos).sort_values(by= "cantidad de únicos", ascending= False)

        return reporte_unicos

            
    def report_missings(self, dataset: pd.DataFrame):
        """Analiza la cantidad y porcentaje de missing en cada variable"""
        ## Columnas tentativas: Variable, Cant. No Nulos, Cant. Nulos, % Nulos
        
        features_col = dataset.columns.to_list()
        cant_nulos = [dataset[d].isna().sum() for d in features_col]
        mean_nulos = [dataset[d].isna().mean() for d in features_col]

        reporte = {
            "Columnas Tentativas": features_col,
            "Cant. Nulos": cant_nulos,
            "% Nulos": mean_nulos
        }

        reporte_df = pd.DataFrame(
            reporte,
        ).sort_values(by= "Cant. Nulos", ascending= False)

        fig, ax = plt.subplots()
        sns.heatmap(
            dataset.isna().transpose(),
            cmap= "YlGnBu",
            cbar_kws= {"label": "Valores perdidos"}
        )
        ax.set_title("Distribución de valores perdidos")
        plt.tight_layout()
        return reporte_df, fig, ax

    def limpieza_en_blancos(self):
        """Remueve espacios en blanco y saltos de línea al inicio y final de cada elemento del dataset"""
        dataset = self.dataset_origin

        for i in self.categorial_features:
            dataset[i] = dataset[i].astype("str").str.strip()
    
        return dataset
    
    def outliers(self, dataset: pd.DataFrame):
        """Visualizador de boxplot para cada feature numérica"""
        fig, axes = plt.subplots(len(self.numerical_features), 1, figsize = (10, 20))
        axes = axes.flatten()

        for i, ax in zip(self.numerical_features, axes):
            sns.boxplot(
                x = i,
                data= dataset,
                whis= 1.5,
                ax= ax
            )
            ax.set_title(f"Distribución en {i}")
        fig.tight_layout()
        fig.subplots_adjust(hspace= 0.6)

        return fig, axes
    
    def histogram(self, dataset: pd.DataFrame):
        """Muestra la distribución de cada feature numérico"""
        fig, axes = plt.subplots(len(self.numerical_features), 1, figsize= (6, 30))
        axes = axes.flatten()

        for i, ax in zip(self.numerical_features, axes):
            ax.hist(
                dataset[i],
                bins = "auto"
            )
            ax.set_title(f"{i}")

        return fig, axes

    def iqr_tecnica(self, dataset: pd.DataFrame):
        """Calcular numéricamente la cantidad de outliers con la técnica IQR"""
        
        numerical_features = []
        upper_list = []
        lower_list = []
        cantidad_outlier = []
        porcentaje_outlier = []
        remove_vals = ["late_payments", "marketing_emails_opened", "complaints_last_3m" ,"support_calls"]

        numerical_features_wo_payment = [i for i in self.numerical_features if i not in remove_vals]
        dataset_wins = dataset.copy()

        for i in numerical_features_wo_payment:
            
            Q1 = dataset[i].quantile(0.25)
            Q3 = dataset[i].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - (IQR * 1.5)
            upper = Q3 + (IQR * 1.5)
            mask_outliers = (dataset[i] < lower) | (dataset[i] > upper)
            cantidad_outliers = mask_outliers.sum()
            porcentaje_outliers = cantidad_outliers / (dataset[i].count())

            numerical_features.append(i)
            upper_list.append(upper)
            lower_list.append(lower)
            cantidad_outlier.append(cantidad_outliers)
            porcentaje_outlier.append(porcentaje_outliers)
            dataset_wins[i] = dataset_wins[i].clip(lower, upper)

        reporte_iqr = {
            "Feature Numérica": numerical_features,
            "Límite Inferior": lower_list,
            "Límite Superior": upper_list,
            "Cantidad de Outliers": cantidad_outlier,
            "Porcentaje Outlier": porcentaje_outlier
        }

        iqr_tabla = pd.DataFrame(reporte_iqr).sort_values(by= "Cantidad de Outliers", ascending= False)
        
        

        return iqr_tabla, dataset_wins
    
    def create_var_features(self, dataset: pd.DataFrame, features: list, mode = "numerico"):
        """Creación de las nuevas features/columnas que comparan moth over month (mom)"""
        var_features = ["var_"+i for i in features]
        dataset_other = dataset.copy().sort_values(by=["customer_id", "period"])

        for var, feat in zip(var_features, features):
            
            if mode == "numerico":
                dataset_other[var] = (
                    dataset_other[feat] - dataset_other.groupby("customer_id")[feat].shift(1)
                )
            elif mode == "porcentage":
                dataset_other[var] = (
                    dataset_other.groupby("customer_id")[feat].pct_change()
                )

        print(f"Features Creados: {var_features}")
        return dataset_other


class Modeling:
    def __init__(self):
        pass

    def objective(self, trial, X_train, y_train):
        params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "random_state": 42,
        "n_jobs": -1
    }

        model = RandomForestClassifier(**params)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        score = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",  
            n_jobs=-1
        ).mean()

        return score
    
    def objective_xgboost(self, trial, X_train, y_train):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist"
        }

        model = XGBClassifier(**params)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        score = cross_val_score(
            estimator=model,
            X=X_train,
            y=y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1
        ).mean()

        return score

    
    def objective_lightlgb(self, trial, X_train, y_train):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1
        }

        model = LGBMClassifier(**params)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        score = cross_val_score(
            estimator=model,
            X=X_train,
            y=y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1
        ).mean()

        return score
    

class Explicabilidad():
    def __init__(self):
        pass

    def shap_binary_class1(self, model, X_sample):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_sample)

        if len(shap_values.values.shape) == 3:
            shap_values = shap_values[:, :, 1]

        return shap_values




















