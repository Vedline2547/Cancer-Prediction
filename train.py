import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,make_scorer,f1_score,recall_score,ConfusionMatrixDisplay,accuracy_score
from imblearn.over_sampling import SMOTE
import optuna
from optuna.samplers import TPESampler
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
# Prepeare data
df = pd.read_csv(r"C:\Users\HP\Desktop\New ML\Cancer-prediction\cancer-risk-factors.csv")
# print(df.head())
X = df.drop(columns=['Risk_Level','Patient_ID','Cancer_Type'])
le = LabelEncoder()
y = le.fit_transform(df['Risk_Level'])
# Split dataset
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# model = RandomForestClassifier(random_state=42,n_estimators=100)
# model.fit(X_train,y_train)
# y_pred = model.predict(X_test)
# print("Classification report")
# print(classification_report(y_test,y_pred,target_names=le.classes_))
# print("Confusion Matrix")
# print(confusion_matrix(y_test,y_pred))
# The model accurately distinguises all risk levels with imbalanced dataset meaning we have an overtraining issue
# log_reg = LogisticRegression(random_state=42)
# log_reg.fit(X_train,y_train)
# y_pred_lr = log_reg.predict(X_test)
# print("Classification report")
# print(classification_report(y_test,y_pred_lr,target_names=le.classes_))
# print("Confusion Matrix")
# print(confusion_matrix(y_test,y_pred_lr))
# Removing Overall Risk Score in an aim to prevent overfitiitng
# X_new = df.drop(columns=['Risk_Level','Patient_ID','Cancer_Type','Overall_Risk_Score'])
# # Encode target variable
# le_new = LabelEncoder()
# y_new = le_new.fit_transform(df['Risk_Level'])
# # Split dataset
# X_train_new,X_test_new,y_train_new,y_test_new = train_test_split(X_new,y_new,test_size=0.2,random_state=42,stratify=y)
# # Feature scaling
# scaler = StandardScaler()
# X_train_new = scaler.fit_transform(X_train_new)
# X_test_new = scaler.transform(X_test_new)
# # model = RandomForestClassifier(random_state=42,n_estimators=100)
# model.fit(X_train_new,y_train_new)
# y_pred_new = model.predict(X_test_new)
# print("Classification report")
# print(classification_report(y_test,y_pred_new,target_names=le_new.classes_))
# print("Confusion Matrix")
# print(confusion_matrix(y_test_new,y_pred_new))
# We now have a more reliable output after removing Overall Risk Score

# MODEL BUULDING 2
# Trying to balance dataset
# SMOTE
# smote = SMOTE(random_state=42)
# X_train_res,y_train_res = smote.fit_resample(X_train_new,y_train_new)
# # print("Class distribution after SMOTE")
# # print(pd.Series(y_train_res).value_counts())
# rf_model = RandomForestClassifier(random_state=41,n_estimators=1000)
# rf_model.fit(X_train_res,y_train_res)
# y_pred2 = rf_model.predict(X_test_new)
# print("Classification report")
# print(classification_report(y_test,y_pred2,target_names=le_new.classes_))
# print("Confusion Matrix")
# print(confusion_matrix(y_test_new,y_pred2))

# MODEL BUILDING 3
# Using Optuna
# Define macro F1 scorer explicitly for multiclass
# def macro_f1(y_true, y_pred):
#     return f1_score(y_true, y_pred, average='macro')

# scorer = make_scorer(macro_f1)
# def objective(trial):
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 50, 500),
#         'max_depth': trial.suggest_int('max_depth', 3, 30),
#         'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
#         'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
#         'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
#         'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
#         'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
#         'random_state': 42,
#         'n_jobs': -1
#     }

#     model = RandomForestClassifier(**params)

#     cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
#     scores = []

#     for train_idx, val_idx in cv.split(X_train_res, y_train_res):
#         X_t, X_v = X_train_res[train_idx], X_train_res[val_idx]
#         y_t, y_v = y_train_res[train_idx], y_train_res[val_idx] 
#         model.fit(X_t, y_t)
#         y_pred = model.predict(X_v)
#         score = f1_score(y_v, y_pred, average='macro')  # explicit multiclass handling
#         scores.append(score)

#     return np.mean(scores)

# # Create study 
# sampler = TPESampler(seed=42)
# study = optuna.create_study(direction='maximize', sampler=sampler, study_name='rf_macro_f1')
# # Run optimization 
# n_trials = 50  # change to 100+ if you want more thorough search
# study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
# print("Best trial:")
# print("  Value (macro F1):", study.best_value)
# print("  Params:")
# for k, v in study.best_params.items():
#     print(f"    {k}: {v}")
    
# RUN ABOVE CODE FIRST
# Train final model with best params and full training data, then evaluate on test
# best_params = {
#     'n_estimators': 347,
#     'max_depth': 19,
#     'min_samples_split': 2,
#     'min_samples_leaf': 1,
#     'max_features': 'sqrt',
#     'bootstrap': False,
#     'criterion': 'gini',
#     'random_state': 42,
#     'n_jobs': -1
# }


# final_rf = RandomForestClassifier(**best_params)
# final_rf.fit(X_train_res, y_train_res)         # resampled training set
# y_pred_test = final_rf.predict(X_test_new)        # untouched test set
# print(classification_report(y_test_new, y_pred_test))
# print(confusion_matrix(y_test_new, y_pred_test))

# MODEL BUILDING 4
# Using IMDB Pipeline with optuna ans SMOTE
# scorer (macro f1)
# def macro_f1(y_true, y_pred):
#     return f1_score(y_true, y_pred, average='macro')

# macro_f1_scorer = make_scorer(macro_f1)

# def objective(trial):
#     # RF params to try
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 100, 400),
#         'max_depth': trial.suggest_int('max_depth', 6, 20),
#         'min_samples_split': trial.suggest_int('min_samples_split', 2, 8),
#         'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 6),
#         'max_features': trial.suggest_categorical('max_features', ['sqrt','log2']),
#         'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
#         'criterion': trial.suggest_categorical('criterion', ['gini','entropy']),
#         'random_state': 42,
#         'n_jobs': 1   # avoid nested parallelism inside cross_val_score
#     }

#     sm = SMOTE(random_state=42)
#     rf = RandomForestClassifier(**params)
#     pipe = ImbPipeline([('smote', sm), ('rf', rf)])

#     cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
#     scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=macro_f1_scorer, n_jobs=1)
#     return float(np.mean(scores))

# study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
# study.optimize(objective, n_trials=30)
# print(study.best_value, study.best_params)
# RUN ABOVE CODE FIRST

# Suppose best_params from study:
# best = study.best_params
# rf = RandomForestClassifier(**best, random_state=42, n_jobs=-1)
# pipe_final = ImbPipeline([('smote', SMOTE(random_state=42)), ('rf', rf)])
# pipe_final.fit(X_train, y_train)   # fit on original X_train (pipeline will SMOTE inside)
# y_test_pred = pipe_final.predict(X_test)
# print(classification_report(y_test, y_test_pred))
# print(confusion_matrix(y_test, y_test_pred))
# THIS IS SO FAR THE BEST MODEL - It has an accuracy of 1 and the high risk ia all corrrectly predicted

# MODEL BUILDING 5
# XGBOOST
# le = LabelEncoder()
# le.fit(y_train)

# y_train_enc = le.transform(y_train)
# y_test_enc = le.transform(y_test)

# # Build Pipeline
# xgb = XGBClassifier(use_label_encoder=False, eval_metrics='mlogloss', random_state=42, n_jobs=-1)
# pipe = ImbPipeline([
#     ('smote', SMOTE(random_state=42)),
#     ('xgb', xgb)
# ])

# pipe.fit(X_train, y_train_enc)

# y_pred_enc = pipe.predict(X_test)

# y_pred = le.inverse_transform(y_pred_enc)
# y_test_orig = le.inverse_transform(y_test_enc)

# print("Baseline XGB Classifier")
# print(classification_report(y_test_orig, y_pred))
# print("Confusion Matrix")
# print(confusion_matrix(y_test_orig, y_pred))

# MODEL BUILDING 6
# =====================================================
# 1. ENCODE TARGET LABELS
# =====================================================
# le = LabelEncoder()
# y_train_enc = le.fit_transform(y_train)
# y_test_enc = le.transform(y_test)
# classes = np.unique(y_train_enc)
# print("Original Classes:", le.classes_)
# print("Encoded Classes:", classes)


# # =====================================================
# # 2. COMPUTE CLASS WEIGHTS
# # =====================================================

# class_weights = compute_sample_weight(
#     class_weight="balanced",
#     y=y_train_enc
# )

# print("\nClass Weights:")

# for cls in classes:
#     weight = class_weights[y_train_enc == cls][0]
#     print(f"{le.classes_[cls]}: {weight:.4f}")


# # =====================================================
# # 3. INITIALIZE XGBOOST
# # =====================================================

# xgb_weighted = XGBClassifier(
#     objective="multi:softmax",
#     num_class=len(classes),
#     eval_metric="mlogloss",

#     random_state=42,

#     n_estimators=105,
#     max_depth=3,
#     learning_rate=0.014843806162322944,
#     subsample=0.7730982444721965,
#     colsample_bytree=0.8509237341976973,
#     gamma=4.134336554829186,
#     reg_alpha=4.453508368364819,
#     reg_lambda=2.616923339470738,

#     n_jobs=-1
# )


# # =====================================================
# # 4. TRAIN MODEL USING SAMPLE WEIGHTS
# # =====================================================

# xgb_weighted.fit(
#     X_train,
#     y_train_enc,
#     sample_weight=class_weights
# )


# # =====================================================
# # 5. MAKE PREDICTIONS
# # =====================================================

# y_pred_enc = xgb_weighted.predict(X_test)

# # Convert predictions back to original labels
# y_pred = le.inverse_transform(y_pred_enc.astype(int))


# # =====================================================
# # 6. CLASSIFICATION REPORT
# # =====================================================

# print("\n" + "=" * 55)
# print("CLASS-WEIGHTED XGBOOST - CLASSIFICATION REPORT")
# print("=" * 55)

# print(
#     classification_report(
#         y_test_enc,
#         y_pred_enc,
#         labels=classes,
#         target_names=le.classes_.astype(str),
#         zero_division=0
#     )
# )


# # =====================================================
# # 7. CONFUSION MATRIX
# # =====================================================

# cm = confusion_matrix(
#     y_test_enc,
#     y_pred_enc,
#     labels=classes
# )

# print("\nConfusion Matrix:")
# print(cm)

# disp = ConfusionMatrixDisplay(
#     confusion_matrix=cm,
#     display_labels=le.classes_.astype(str)
# )

# disp.plot(cmap="Blues", values_format="d")

# plt.title("Class-Weighted XGBoost - Confusion Matrix")
# plt.tight_layout()
# plt.show()


# # =====================================================
# # 8. OVERALL MODEL PERFORMANCE
# # =====================================================

# accuracy = accuracy_score(y_test_enc, y_pred_enc)

# macro_f1 = f1_score(
#     y_test_enc,
#     y_pred_enc,
#     average="macro"
# )

# print("\n" + "=" * 55)
# print("OVERALL MODEL PERFORMANCE")
# print("=" * 55)

# print(f"Accuracy: {accuracy:.4f}")
# print(f"Macro F1-score: {macro_f1:.4f}")


# # =====================================================
# # 9. HIGH-RISK RECALL
# # =====================================================

# TARGET_LABEL = "High"

# high_index = le.transform([TARGET_LABEL])[0]

# high_recall = recall_score(
#     y_test_enc,
#     y_pred_enc,
#     labels=[high_index],
#     average="macro",
#     zero_division=0
# )

# print("\n" + "=" * 55)
# print("HIGH-RISK CLASS PERFORMANCE")
# print("=" * 55)

# print(f"High-risk Recall: {high_recall:.4f}")
# print(f"High-risk Recall (%): {high_recall * 100:.2f}%")


# # =====================================================
# # 10. DISPLAY PREDICTIONS
# # =====================================================

# print("\nSample Predictions:")

# for actual, predicted in zip(
#     le.inverse_transform(y_test_enc[:10]),
#     y_pred[:10]
# ):
#     print(f"Actual: {actual:<10} | Predicted: {predicted}")

# MODEL BUILDING 7
# Optuna tuning for class-weighted XGBoost (optimize macro-F1)

# ----------------------
# Prepare encoded labels
# ----------------------
le = LabelEncoder()
le.fit(y_train)               # fit encoder on training labels
y_train_enc = le.transform(y_train)
y_test_enc  = le.transform(y_test)
n_classes = len(le.classes_)

# ----------------------
# Objective: maximize macro-F1
# ----------------------
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 400),
        'max_depth': trial.suggest_int('max_depth', 2, 12),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
        # stable args:
        'use_label_encoder': False,
        'eval_metric': 'mlogloss',
        # set n_jobs=1 inside CV to avoid nested parallelism
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in cv.split(X_train, y_train_enc):
        X_t, X_v = X_train[train_idx], X_train[val_idx]
        y_t, y_v = y_train_enc[train_idx], y_train_enc[val_idx]

        # compute class weights from the fold's training data
        classes, counts = np.unique(y_t, return_counts=True)
        # simple inverse frequency scaling
        class_weight_map = {cls: float(max(counts) / cnt) for cls, cnt in zip(classes, counts)}
        sample_weights = np.array([class_weight_map[y] for y in y_t])

        # instantiate classifier with n_jobs=1 for CV
        model = XGBClassifier(**params, n_jobs=1, random_state=42)

        # fit with per-sample weights
        model.fit(X_t, y_t, sample_weight=sample_weights, verbose=False)

        y_pred = model.predict(X_v)
        scores.append(f1_score(y_v, y_pred, average='macro'))

    # return mean macro-F1 across folds
    return float(np.mean(scores))

# ----------------------
# Run Optuna study
# ----------------------
sampler = TPESampler(seed=42)
study = optuna.create_study(direction='maximize', sampler=sampler, study_name='xgb_class_weighted_macroF1')

# fewer trials to start; increase to 50-150 when comfortable
study.optimize(objective, n_trials=40, show_progress_bar=True)

print("Best CV macro-F1:", study.best_value)
print("Best params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# ----------------------
# Train final model on full training set using class weights
# ----------------------
best = study.best_params.copy()
best.update({'use_label_encoder': False, 'eval_metric': 'mlogloss', 'random_state': 42})

# compute final sample weights from full training set
classes, counts = np.unique(y_train_enc, return_counts=True)
final_class_weight_map = {cls: float(max(counts)/cnt) for cls, cnt in zip(classes, counts)}
final_sample_weights = np.array([final_class_weight_map[y] for y in y_train_enc])

final_xgb = XGBClassifier(**best, n_jobs=-1)
final_xgb.fit(X_train, y_train_enc, sample_weight=final_sample_weights, verbose=False)

# Evaluate on untouched test set
y_test_pred_enc = final_xgb.predict(X_test)
y_test_pred = le.inverse_transform(y_test_pred_enc)
y_test_orig = y_test  # already readable labels

print("\nFINAL Test Classification Report (class-weighted XGB):")
print(classification_report(y_test_orig, y_test_pred, labels=le.classes_))
print("Confusion Matrix (rows=true, cols=pred):")
print(confusion_matrix(y_test_orig, y_test_pred, labels=le.classes_))
# DECIDED TO GO WITH THIS MODEL
import joblib
joblib.dump(final_xgb,'model_xgb_new.pkl')

print("Model saved successfully as 'model_xgb_new.pkl'")
