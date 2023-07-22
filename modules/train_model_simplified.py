import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

parser = argparse.ArgumentParser(description='Script for final classifier model training')

parser.add_argument('--train', help='Path to train dataset', required=True, default="data/train_all_users.csv")
parser.add_argument('--test', help='Path to test dataset', required=True, default="data/test_all_users.csv")
parser.add_argument('--name', help='Name of model', required=False, default="final_model_all_users")

args = vars(parser.parse_args())
MODEL_NAME = args["name"]
filename_pattern_train = args["train"]
filename_pattern_test_full = args["test"]

train_df = pd.read_csv(filename_pattern_train)
test_full_df = pd.read_csv(filename_pattern_test_full)

# filtering only data for final classifier tuning
train_df = train_df[train_df["is_text_train"] == 0]

print("Training dataset size: ", len(train_df))
print("Hold-out testing dataset size: ", len(test_full_df))

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
user_features = [f"is_{g}" for g in user_groups] + ["user_is_anonymous"]


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
  + ["bert_title_score"] \
+ user_features

# Modelling:
BALANCE_DATASET = False
BALANCE_CLASS_MODEL = True

is_text_train_filter = 'is_text_train'
balancing_column = 'is_balanced'
target_column = 'revision_is_identity_reverted'
features = features_list

X_train, X_test, y_train, y_test = train_test_split(
    train_df[features].fillna(-1),
    train_df[target_column],
    test_size=0.1,
    random_state=42,
    stratify=train_df[target_column]
)

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

# class weighting
if BALANCE_CLASS_MODEL:
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
else:
    class_weights = {0: 1, 1: 1}

# Initialize CatBoostClassifier
model = CatBoostClassifier(
    iterations=5000,
    metric_period=100,
    verbose=True,
    learning_rate=0.01,
    class_weights=class_weights,
    custom_metric=['Logloss', 'AUC', 'Precision', 'Recall', 'F1', 'Accuracy']
)

# Fit model
model.fit(train_data, eval_set=test_data, plot=False)

print(model.best_score_)
print(pd.DataFrame({
    'feature_importance': model.get_feature_importance(train_data),
    'feature_names': X_test.columns}
).sort_values(by=['feature_importance'], ascending=False).head(30))
model.save_model(MODEL_NAME)

# Make final prediction file for further analysis
model = CatBoostClassifier().load_model(MODEL_NAME)
hold_out_data = Pool(
    data=test_full_df[features].fillna(-1),
    label=test_full_df[target_column],
    cat_features=cat_features
)
test_full_df["predict_score"] = model.predict_proba(hold_out_data)[:, 1]
test_full_df[["wiki_db", "revision_id", "predict_score"]].to_csv(f"data/{MODEL_NAME}_scores.csv", index=False)
