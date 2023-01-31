import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm
import pickle


MODEL_NAME = "classifier_ml_all_non_balanced_user_features_mlm"
print(MODEL_NAME)

cc_codes = [
    "ka", "lv", "ta", "ur", "eo", "lt", "sl", "hy", "hr", "sk", "eu", "et", "ms", "az", "da", "bg",
    "sr", "ro", "el", "th", "bn", "no", "hi", "ca", "hu", "ko", "fi", "vi", "uz", "sv", "cs", "he",
    "id", "tr", "uk", "nl", "pl", "ar", "fa", "it", "zh", "ru", "es", "ja", "de", "fr", "en"
]

# loading data
filename_pattern_train = "data/processed_all_users_train_{}.csv".format("_".join(cc_codes))
filename_pattern_test_full = "data/processed_anon_test_full_{}.csv".format("_".join(cc_codes))

train_df = pd.read_csv(filename_pattern_train)
train_df = train_df[train_df["is_text_train"] == 0]

# adding additional user features:
with open('data/additional_features.pickle', 'rb') as handle:
    additional_features_dict = pickle.load(handle)


def get_additional_features(df, dict_):
    additional_features_list = []
    for r, db in tqdm(zip(df.revision_id.values, df.wiki_db.values)):
        additional_features_list.append(dict_.get((db, r), {}))
    additional_features_df = pd.DataFrame(additional_features_list)
    return pd.concat([df.reset_index(drop=True), additional_features_df], axis=1)


train_df = get_additional_features(train_df, additional_features_dict)


user_groups = [
    'editor',
    'trusted',
    'ipblock-exempt',
    'interface-admin',
    'sysop',
    'autoreview',
    'bureaucrat',
    'autopatrolled',
    'patroller',
    'rollbacker',
    'reviewer',
    'extendedconfirmed',
    'templateeditor',
    'uploader',
    'autoreviewer',
    'suppressredirect'
]
user_features = [f"is_{g}" for g in user_groups] + ["is_anonymous"]


# feature renaming
features_renaming = dict()

old_insert = [f"insert_{c}" for c in
              ["s_0_max", "s_1_max", "p_0_max", "p_1_max", "s_0_mean", "s_1_mean", "p_0_mean", "p_1_mean"]]
new_insert = [f"bert_insert_{c}" for c in
              ["s0_max", "s1_max", "p0_max", "p1_max", "s0_mean", "s1_mean", "p0_mean", "p1_mean"]]
features_renaming.update({k: v for k, v in zip(old_insert, new_insert)})

old_insert = [f"remove_{c}" for c in
              ["s_0_max", "s_1_max", "p_0_max", "p_1_max", "s_0_mean", "s_1_mean", "p_0_mean", "p_1_mean"]]
new_insert = [f"bert_remove_{c}" for c in
              ["s0_max", "s1_max", "p0_max", "p1_max", "s0_mean", "s1_mean", "p0_mean", "p1_mean"]]
features_renaming.update({k: v for k, v in zip(old_insert, new_insert)})

old_insert = [f"change_{c}" for c in
              ["s_0_max", "s_1_max", "p_0_max", "p_1_max", "s_0_mean", "s_1_mean", "p_0_mean", "p_1_mean"]]
new_insert = [f"bert_change_{c}" for c in
              ["s0_max", "s1_max", "p0_max", "p1_max", "s0_mean", "s1_mean", "p0_mean", "p1_mean"]]
features_renaming.update({k: v for k, v in zip(old_insert, new_insert)})

old_insert = [f"comment_{c}" for c in ["s_0", "s_1", "p_0", "p_1"]]
new_insert = [f"bert_comment_{c}" for c in ["s0", "s1", "p0", "p1"]]
features_renaming.update({k: v for k, v in zip(old_insert, new_insert)})

old_insert = [f"title_s_0"]
new_insert = ["bert_title_score"]
features_renaming.update({k: v for k, v in zip(old_insert, new_insert)})

new_columns = [features_renaming.get(x, x) for x in train_df.columns]
train_df.columns = new_columns

CONTENT_TYPES = [
    'Argument', 'Category', 'Comment', 'ExternalLink',
    'Gallery', 'HTMLEntity', 'Heading', 'List', 'Media',
    'Paragraph', 'Punctuation', 'Reference', 'Section',
    'Sentence', 'Table', 'Table Element', 'Template',
    'Text', 'Text Formatting', 'Whitespace', 'Wikilink', 'Word'
]
ACTION_TYPES = ['change', 'insert', 'move', 'remove']
features_actions = [f"{c}_{t}" for t in CONTENT_TYPES for c in ACTION_TYPES]

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
+ ["bert_title_score"] \
+ user_features

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
] + user_features

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

# class weighting
if BALANCE_CLASS_MODEL:
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
else:
    class_weights = {0: 1, 1: 1}

# Initialize CatBoostClassifier:
model = CatBoostClassifier(iterations=5000, metric_period=100, verbose=True, learning_rate=0.01,
                           class_weights=class_weights,
                           custom_metric=['Logloss', 'AUC', 'Precision', 'Recall', 'F1', 'Accuracy'])
# Fit model
model.fit(train_data, eval_set=test_data, plot=False)

print(model.best_score_)
print(pd.DataFrame({
    'feature_importance': model.get_feature_importance(train_data),
    'feature_names': X_test.columns}
).sort_values(by=['feature_importance'], ascending=False).head(30))

model.save_model("models/" + MODEL_NAME)

# Make final prediction file
model = CatBoostClassifier().load_model("models/" + MODEL_NAME)

test_full_df = pd.read_csv(filename_pattern_test_full)
test_full_df = get_additional_features(test_full_df, additional_features_dict)
new_columns = [features_renaming.get(x, x) for x in test_full_df.columns]
test_full_df.columns = new_columns

hold_out_data = Pool(
    data=test_full_df[features].fillna(-1),
    label=test_full_df[target_column],
    cat_features=cat_features
)
test_full_df["predict_score"] = model.predict_proba(hold_out_data)[:, 1]
test_full_df[["wiki_db", "revision_id", "predict_score"]].to_csv(f"data/scores/{MODEL_NAME}_scores.csv", index=False)
