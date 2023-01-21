import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

MODEL_NAME = "classifier_ml_anon"

# loading data
filename_pattern_train = "data/processed_train_ka_lv_ta_ur_eo_lt_sl_hy_hr_sk_eu_et_ms_az_da_bg_sr_ro_el_th_bn_no_hi_ca_hu_ko_fi_vi_uz_sv_cs_he_id_tr_uk_nl_pl_ar_fa_it_zh_ru_es_ja_de_fr_en.csv"
filename_pattern_test = "data/processed_test_ka_lv_ta_ur_eo_lt_sl_hy_hr_sk_eu_et_ms_az_da_bg_sr_ro_el_th_bn_no_hi_ca_hu_ko_fi_vi_uz_sv_cs_he_id_tr_uk_nl_pl_ar_fa_it_zh_ru_es_ja_de_fr_en.csv"
filename_pattern_test_full = "data/processed_test_ka_lv_ta_ur_eo_lt_sl_hy_hr_sk_eu_et_ms_az_da_bg_sr_ro_el_th_bn_no_hi_ca_hu_ko_fi_vi_uz_sv_cs_he_id_tr_uk_nl_pl_ar_fa_it_zh_ru_es_ja_de_fr_en.csv"

train_df = pd.read_csv(filename_pattern_train)
test_df = pd.read_csv(filename_pattern_test)
test_full_df = pd.read_csv(filename_pattern_test_full)

train_df = train_df[train_df["is_text_train"] == 0]

# feature renaming
features_renaming = dict()

old_insert = [f"insert_{c}" for c in ["s_0_max", "s_1_max", "p_0_max", "p_1_max", "s_0_mean", "s_1_mean", "p_0_mean", "p_1_mean"]]
new_insert = [f"bert_insert_{c}" for c in ["s0_max", "s1_max", "p0_max", "p1_max", "s0_mean", "s1_mean", "p0_mean", "p1_mean"]]
features_renaming.update({k:v for k,v in zip(old_insert, new_insert)})

old_insert = [f"remove_{c}" for c in ["s_0_max", "s_1_max", "p_0_max", "p_1_max", "s_0_mean", "s_1_mean", "p_0_mean", "p_1_mean"]]
new_insert = [f"bert_remove_{c}" for c in ["s0_max", "s1_max", "p0_max", "p1_max", "s0_mean", "s1_mean", "p0_mean", "p1_mean"]]
features_renaming.update({k:v for k,v in zip(old_insert, new_insert)})

old_insert = [f"change_{c}" for c in ["s_0_max", "s_1_max", "p_0_max", "p_1_max", "s_0_mean", "s_1_mean", "p_0_mean", "p_1_mean"]]
new_insert = [f"bert_change_{c}" for c in ["s0_max", "s1_max", "p0_max", "p1_max", "s0_mean", "s1_mean", "p0_mean", "p1_mean"]]
features_renaming.update({k:v for k,v in zip(old_insert, new_insert)})

old_insert = [f"comment_{c}" for c in ["s_0", "s_1", "p_0", "p_1"]]
new_insert = [f"bert_comment_{c}" for c in ["s0", "s1", "p0", "p1"]]
features_renaming.update({k:v for k,v in zip(old_insert, new_insert)})

old_insert = [f"title_s_0"]
new_insert = ["bert_title_score"]
features_renaming.update({k:v for k,v in zip(old_insert, new_insert)})

feat = [
    'Argument_change', 'Argument_insert', 'Argument_move', 'Argument_remove', 'Category_change',
    'Category_insert', 'Category_move', 'Category_remove', 'Comment_change', 'Comment_insert',
    'Comment_move', 'Comment_remove', 'ExternalLink_change', 'ExternalLink_insert', 'ExternalLink_move',
    'ExternalLink_remove', 'Gallery_change', 'Gallery_insert', 'Gallery_move', 'Gallery_remove', 'HTMLEntity_change',
    'HTMLEntity_insert', 'HTMLEntity_move', 'HTMLEntity_remove', 'Heading_change', 'Heading_insert', 'Heading_move',
    'Heading_remove', 'List_change', 'List_insert', 'List_move', 'List_remove', 'Media_change', 'Media_insert',
    'Media_move', 'Media_remove', 'Paragraph_change', 'Paragraph_insert', 'Paragraph_move', 'Paragraph_remove',
    'Punctuation_change', 'Punctuation_insert', 'Punctuation_move', 'Punctuation_remove', 'Reference_change',
    'Reference_insert', 'Reference_move', 'Reference_remove', 'Section_change', 'Section_insert', 'Section_move',
    'Section_remove', 'Sentence_change', 'Sentence_insert', 'Sentence_move', 'Sentence_remove', 'Table_change',
    'Table_insert', 'Table_move', 'Table_remove', 'Table Element_change', 'Table Element_insert', 'Table Element_move',
    'Table Element_remove', 'Template_change', 'Template_insert', 'Template_move', 'Template_remove', 'Text_change',
    'Text_insert', 'Text_move', 'Text_remove', 'Text Formatting_change', 'Text Formatting_insert', 'Text Formatting_move',
    'Text Formatting_remove', 'Whitespace_change', 'Whitespace_insert', 'Whitespace_move', 'Whitespace_remove',
    'Wikilink_change', 'Wikilink_insert', 'Wikilink_move', 'Wikilink_remove', 'Word_change', 'Word_insert',
    'Word_move', 'Word_remove'
]

features_renaming.update({old: "_".join([old.split("_")[1], old.split("_")[0]]) for old in feat})

new_columns = [features_renaming.get(x, x) for x in train_df.columns]
train_df.columns = new_columns
new_columns = [features_renaming.get(x, x) for x in test_df.columns]
test_df.columns = new_columns
new_columns = [features_renaming.get(x, x) for x in test_full_df.columns]
test_full_df.columns = new_columns

feat = [
    'Argument_change', 'Argument_insert', 'Argument_move', 'Argument_remove', 'Category_change',
    'Category_insert', 'Category_move', 'Category_remove', 'Comment_change', 'Comment_insert',
    'Comment_move', 'Comment_remove', 'ExternalLink_change', 'ExternalLink_insert', 'ExternalLink_move',
    'ExternalLink_remove', 'Gallery_change', 'Gallery_insert', 'Gallery_move', 'Gallery_remove', 'HTMLEntity_change',
    'HTMLEntity_insert', 'HTMLEntity_move', 'HTMLEntity_remove', 'Heading_change', 'Heading_insert', 'Heading_move',
    'Heading_remove', 'List_change', 'List_insert', 'List_move', 'List_remove', 'Media_change', 'Media_insert',
    'Media_move', 'Media_remove', 'Paragraph_change', 'Paragraph_insert', 'Paragraph_move', 'Paragraph_remove',
    'Punctuation_change', 'Punctuation_insert', 'Punctuation_move', 'Punctuation_remove', 'Reference_change',
    'Reference_insert', 'Reference_move', 'Reference_remove', 'Section_change', 'Section_insert', 'Section_move',
    'Section_remove', 'Sentence_change', 'Sentence_insert', 'Sentence_move', 'Sentence_remove', 'Table_change',
    'Table_insert', 'Table_move', 'Table_remove', 'Table Element_change', 'Table Element_insert', 'Table Element_move',
    'Table Element_remove', 'Template_change', 'Template_insert', 'Template_move', 'Template_remove', 'Text_change',
    'Text_insert', 'Text_move', 'Text_remove', 'Text Formatting_change', 'Text Formatting_insert', 'Text Formatting_move',
    'Text Formatting_remove', 'Whitespace_change', 'Whitespace_insert', 'Whitespace_move', 'Whitespace_remove',
    'Wikilink_change', 'Wikilink_insert', 'Wikilink_move', 'Wikilink_remove', 'Word_change', 'Word_insert',
    'Word_move', 'Word_remove'
]
features_actions = ["_".join([old.split("_")[1], old.split("_")[0]]) for old in feat]

features_list = [
       'wiki_db',
       'revision_text_bytes_diff',
       'is_mobile_edit', 'is_mobile_web_edit',
       'is_visualeditor', 'is_wikieditor',
       'is_ios_app_edit',
] + features_actions \
  + [f"bert_insert_{c}" for c in ["s0_max", "s1_max", "p0_max", "p1_max", "s0_mean", "s1_mean", "p0_mean", "p1_mean"]] \
  + [f"bert_remove_{c}" for c in ["s0_max", "s1_max", "p0_max", "p1_max", "s0_mean", "s1_mean", "p0_mean", "p1_mean"]] \
  + [f"bert_change_{c}" for c in ["s0_max", "s1_max", "p0_max", "p1_max", "s0_mean", "s1_mean", "p0_mean", "p1_mean"]] \
  + ["bert_title_score"]

# Modelling:
BALANCE_DATASET = False
BALANCE_CLASS_MODEL = True

is_text_train_filter = 'is_text_train'
balancing_column = 'is_balanced'
target_column = 'revision_is_identity_reverted'
features = features_list

X_train, X_test, y_train, y_test = train_test_split(train_df[features].fillna(-1),
                                                    train_df[target_column],
                                                    test_size=0.1,
                                                    random_state=42,
                                                    stratify=train_df[target_column])


cat_features = [
    'wiki_db',
    'is_mobile_edit', 'is_mobile_web_edit',
    'is_visualeditor', 'is_wikieditor',
    'is_ios_app_edit',
]

train_data = Pool(
    data=X_train,
    label=y_train,
    cat_features=cat_features
)
test_data = Pool(
    data=X_test,
    label=y_test,
    cat_features=cat_features
)

### class weighting
if BALANCE_CLASS_MODEL:
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
else:
    class_weights = {0: 1, 1: 1}
#####


# Initialize CatBoostClassifier
model = CatBoostClassifier(iterations=5000, metric_period=100, verbose=True, learning_rate=0.01, class_weights=class_weights,
                           # task_type="GPU",
                           custom_metric=['Logloss', 'AUC', 'Precision', 'Recall', 'F1', 'Accuracy'])
# Fit model
model.fit(train_data, eval_set=test_data, plot=False)

print(model.best_score_)
print(pd.DataFrame({
    'feature_importance': model.get_feature_importance(train_data),
    'feature_names': X_test.columns}
).sort_values(by=['feature_importance'], ascending=False).head(30))


hold_out_data = Pool(
    data=test_df[features].fillna(-1),
    label=test_df[target_column],
    cat_features=cat_features
)
res = model.eval_metrics(hold_out_data, metrics=['Logloss', 'AUC', 'Precision', 'Recall', 'F1', 'Accuracy'], eval_period=1000)
metrics_holdout = {c: res[c][-1] for c in ['Logloss', 'AUC', 'Precision', 'Recall', 'F1', 'Accuracy']}
print("Hold-out metrics: ", metrics_holdout)

hold_out_data = Pool(
    data=test_df[test_df.is_balanced == 1][features].fillna(-1),
    label=test_df[test_df.is_balanced == 1][target_column],
    cat_features=cat_features
)
res = model.eval_metrics(
    hold_out_data,
    metrics=['Logloss', 'AUC', 'Precision', 'Recall', 'F1', 'Accuracy'],
    eval_period=1000
)
metrics_holdout = {c: res[c][-1] for c in ['Logloss', 'AUC', 'Precision', 'Recall', 'F1', 'Accuracy']}
print("Balanced hold-out metrics: ", metrics_holdout)

model.save_model(MODEL_NAME)


# Make final prediction file
model = CatBoostClassifier().load_model(MODEL_NAME)
hold_out_data = Pool(
    data=test_full_df[features].fillna(-1),
    label=test_full_df[target_column],
    cat_features=cat_features
)
test_full_df["predict_score"] = model.predict_proba(hold_out_data)[:, 1]
test_full_df[["revision_id", "predict_score"]].to_csv(f"{MODEL_NAME}_scores.csv", index=False)
