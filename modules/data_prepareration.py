import pandas as pd
import numpy as np
from tqdm import tqdm

cc_codes = [
    "ka",
    "lv",
    "ta",
    # "kk",
    "ur",
    "eo",
    "lt",
    "sl",
    "hy",
    "hr",
    "sk",
    "eu",
    "et",
    "ms",
    "az",
    "da",
    "bg",
    "sr",
    "ro",
    "el",
    "th",
    "bn",
    # "simple",
    "no",
    "hi",
    "ca",
    "hu",
    "ko",
    "fi",
    "vi",
    "uz",
    "sv",
    "cs",
    "he",
    "id",
    "tr",
    "uk",
    "nl",
    "pl",
    "ar",
    "fa",
    "it",
    "zh",
    # "pt",
    "ru",
    "es",
    "ja",
    "de",
    "fr",
    "en"
]


class Preprocessor:

    def __init__(self,
                 text_ranking_split=0.6,
                 rw_filter=True,
                 ):
        self.text_ranking_split_rate = text_ranking_split
        self.rw_filter = rw_filter

    def text_ranking_split(self, data, stratify_col="page_title"):

        unique_titles = data[stratify_col].unique()
        n_unique_titles = len(unique_titles)
        np.random.seed(42)
        text_titles = np.random.choice(
            unique_titles,
            size=int(n_unique_titles * self.text_ranking_split_rate),
            replace=False,

        )
        data.loc[data[stratify_col].isin(text_titles), "is_text_train"] = 1
        data.loc[~data[stratify_col].isin(text_titles), "is_text_train"] = 0

        return data

    @staticmethod
    def balance_key(data, stratify_col="revision_is_identity_reverted"):
        part_1 = data[data[stratify_col] == 1]
        part_2 = data[data[stratify_col] == 0]
        part_2 = part_2.sample(np.min([len(part_1), len(part_2)]), random_state=42)
        balanced = pd.concat([part_1, part_2]).sample(len(part_1) + len(part_2))
        balanced_revision_id = set(balanced.revision_id)

        data.loc[data.revision_id.isin(balanced_revision_id), "is_balanced"] = 1
        data.loc[~data.revision_id.isin(balanced_revision_id), "is_balanced"] = 0
        return data

    def filter(self, data):
        if self.rw_filter:
            data = data[data["reverting_revision_is_reverted_revision"] != True]
        return data


filename_pattern = "data/train/{}_anonymous_text_07-2022.csv"
filename_pattern_train = "data/train_{}.csv".format("_".join(cc_codes))
filename_pattern_test = "data/test_{}.csv".format("_".join(cc_codes))
validation_timestamp_split = "2022-07-01"
feature_factory = Preprocessor()

train_dfs = []
test_dfs = []

for cc in tqdm(cc_codes):
    tmp_df = pd.read_csv(filename_pattern.format(cc))

    train_df = tmp_df[tmp_df.event_timestamp < validation_timestamp_split] \
        .reset_index(drop=True)
    train_df = feature_factory.filter(train_df)
    train_df = feature_factory.balance_key(train_df)
    train_df = feature_factory.text_ranking_split(train_df)
    train_dfs.append(train_df)

    test_df = tmp_df[tmp_df.event_timestamp >= validation_timestamp_split] \
        .reset_index(drop=True)
    test_df = feature_factory.filter(test_df)
    test_df = feature_factory.balance_key(test_df)
    test_dfs.append(test_df)

train_df = pd.concat(train_dfs)
test_df = pd.concat(test_dfs)
train_df.to_csv(filename_pattern_train, index=False)
test_df.to_csv(filename_pattern_test, index=False)


# collect test dataset:
test_dfs = []
filename_pattern = "data/test/{}_anonymous_text_07-2022_test.csv"
filename_pattern_test = "data/test_full_{}.csv".format("_".join(cc_codes))
for cc in tqdm(cc_codes):
    test_dfs.append(pd.read_csv(filename_pattern.format(cc)))

test_df_full = pd.concat(test_dfs)


def is_anon(user_name):
    key_1 = len(user_name.split(".")) == 4
    key_2 = len(user_name.split(":")) == 8
    if key_1 | key_2:
        return 1
    else:
        return 0


test_df_full["is_anon"] = test_df_full.event_user_text_historical.apply(lambda x: is_anon(x))
test_df_full.to_csv(filename_pattern_test, index=False)
