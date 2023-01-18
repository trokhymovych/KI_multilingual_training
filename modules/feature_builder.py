import pandas as pd
from tqdm import tqdm
import ast
import numpy as np
import torch
from transformers import AutoTokenizer, BertForSequenceClassification
from transformers import pipeline
import gc

from scipy.special import softmax

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

CONTENT_TYPES = [
    'Argument', 'Category', 'Comment', 'ExternalLink',
    'Gallery', 'HTMLEntity', 'Heading', 'List', 'Media',
    'Paragraph', 'Punctuation', 'Reference', 'Section',
    'Sentence', 'Table', 'Table Element', 'Template',
    'Text', 'Text Formatting', 'Whitespace', 'Wikilink', 'Word'
]

ACTION_TYPES = ['change', 'insert', 'move', 'remove']

insert_model = "inserts_bert-base-multilingual-cased_balanced/checkpoint-150170"
insert_default_values = [-1, -1, -1, -1, -1, -1, -1, -1]
change_model = "changes_bert-base-multilingual-cased_balanced/checkpoint-312715"
change_default_values = [-1, -1, -1, -1, -1, -1, -1, -1]
remove_model = "removes_bert-base-multilingual-cased_balanced/checkpoint-69050"
remove_default_values = [-1, -1, -1, -1, -1, -1, -1, -1]
title_model = "title_bert-base-multilingual-cased-balanced/checkpoint-51945"
title_default_values = [-1, -1]


class FeatureExtractor:

    def __init__(
            self,
            content_types=CONTENT_TYPES,
            action_types=ACTION_TYPES,
            insert_model=insert_model,
            insert_default_values=insert_default_values,
            change_model=change_model,
            change_default_values=change_default_values,
            title_model=title_model,
            title_default_values=title_default_values,
            remove_model=remove_model,
            remove_default_values=remove_default_values,
    ):
        self.content_types = content_types
        self.action_types = action_types
        self.insert_model = insert_model
        self.insert_default_values = insert_default_values
        self.remove_model = remove_model
        self.remove_default_values = remove_default_values
        self.change_model = change_model
        self.change_default_values = change_default_values
        self.title_model = title_model
        self.title_default_values = title_default_values

    def get_features(self, df):
        df = df.reset_index(drop=True)
        df = self._get_actions_features(df)
        df = self._get_insert_text_features(df)
        df = self._get_change_text_features(df)
        df = self._get_title_semantics(df)
        df = self._get_remove_text_features(df)

        return df

    def _get_actions_features(self, df, actions_column="actions"):
        features = []
        feature_names = [f"{t}_{c}" for t in self.content_types for c in self.action_types]
        for actions in tqdm(df[actions_column]):
            actions = ast.literal_eval(actions)
            features_tmp = [actions.get(t, {}).get(c, 0) for t in self.content_types for c in self.action_types]
            features.append(features_tmp)
        features_df = pd.DataFrame(features, columns=feature_names)
        for c in feature_names:
            df[c] = features_df[c].values
        return df

    def _get_insert_text_features(self, df):
        print("Processing inserts....")
        print("Preparing texts: ")
        texts_to_process = self._get_text_to_process(df, "texts_insert")
        print("Preparing features: ")
        texts_mapping = self._get_text_features(texts_to_process, model_type="insert")
        print("Mapping features: ")
        text_features = []
        for texts_raw in tqdm(df.texts_insert):
            texts = ast.literal_eval(texts_raw)
            features_local = [texts_mapping.get(text) for text in texts if not texts_mapping.get(text) is None]
            if len(features_local) == 0:
                text_features.append(self.insert_default_values)
            else:
                text_features.append(np.hstack([np.max(features_local, axis=0), np.mean(features_local, axis=0)]))

        text_features = np.array(text_features)
        p_columns = [f"insert_{c}" for c in
                     ["s_0_max", "s_1_max", "p_0_max", "p_1_max", "s_0_mean", "s_1_mean", "p_0_mean", "p_1_mean"]]
        for i, c in enumerate(p_columns):
            df[c] = text_features[:, i]
        return df

    def _get_change_text_features(self, df):
        print("Processing changes....")
        print("Preparing texts: ")
        texts_to_process = self._get_text_to_process(df, "texts_change")
        print("Preparing features: ")
        texts_mapping = self._get_text_features(texts_to_process, model_type="change")
        print("Mapping features: ")
        text_features = []
        for texts_raw in tqdm(df.texts_change):
            texts = ast.literal_eval(texts_raw)
            features_local = [texts_mapping.get(str({"text": text[0], "text_pair": text[1]})) for text in texts
                              if not texts_mapping.get(str({"text": text[0], "text_pair": text[1]})) is None]
            if len(features_local) == 0:
                text_features.append(self.change_default_values)
            else:
                text_features.append(np.hstack([np.max(features_local, axis=0), np.mean(features_local, axis=0)]))

        text_features = np.array(text_features)
        p_columns = [f"change_{c}" for c in
                     ["s_0_max", "s_1_max", "p_0_max", "p_1_max", "s_0_mean", "s_1_mean", "p_0_mean", "p_1_mean"]]
        for i, c in enumerate(p_columns):
            df[c] = text_features[:, i]
        return df

    def _get_title_semantics(self, df):
        print("Processing title....")
        print("Preparing texts: ")
        texts_to_process = self._get_text_to_process(df, "page_title")
        print("Preparing features: ")
        texts_mapping = self._get_text_features(texts_to_process, model_type="title")
        print("Mapping features: ")
        text_features = []
        for texts_raw in tqdm(df.page_title):
            features_local = texts_mapping.get(texts_raw)
            if features_local is None:
                text_features.append(self.title_default_values)
            else:
                text_features.append(features_local)

        text_features = np.array(text_features)
        p_columns = [f"title_{c}" for c in ["s_0", "p_0"]]
        for i, c in enumerate(p_columns):
            df[c] = text_features[:, i]
        return df

    def _get_remove_text_features(self, df):
        print("Processing removes....")
        print("Preparing texts: ")
        texts_to_process = self._get_text_to_process(df, "texts_removed")
        print("Preparing features: ")
        texts_mapping = self._get_text_features(texts_to_process, model_type="remove")
        print("Mapping features: ")
        text_features = []
        for texts_raw in tqdm(df.texts_removed):
            texts = ast.literal_eval(texts_raw)
            features_local = [texts_mapping.get(text) for text in texts if not texts_mapping.get(text) is None]
            if len(features_local) == 0:
                text_features.append(self.remove_default_values)
            else:
                text_features.append(np.hstack([np.max(features_local, axis=0), np.mean(features_local, axis=0)]))

        text_features = np.array(text_features)
        p_columns = [f"remove_{c}" for c in
                     ["s_0_max", "s_1_max", "p_0_max", "p_1_max", "s_0_mean", "s_1_mean", "p_0_mean", "p_1_mean"]]
        for i, c in enumerate(p_columns):
            df[c] = text_features[:, i]
        return df

    def _preds_processing(self, preds):
        res = []
        for i in preds:
            res.append(i['score'])
        return np.hstack([res, softmax(res)])

    def _get_text_to_process(self, df, field_name):
        texts_to_process = []
        if field_name in ["page_title", "event_comment"]:
            return list(df[field_name].dropna().unique())
        else:
            for texts_raw in df[field_name].unique():
                texts = ast.literal_eval(texts_raw)
                if (field_name == "texts_insert" or field_name == "texts_removed") and len(texts) > 0:
                    texts_to_process += [text for text in texts if len(text) > 0]
                elif field_name == "texts_change" and len(texts) > 0:
                    texts_to_process += [{"text": text[0], "text_pair": text[1]} for text in texts if
                                         text[0] != text[1]]
                else:
                    pass
            return texts_to_process

    def _get_text_features(self, texts, model_type="insert"):

        print("Loading models: ")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tqdm_batch_size = 1000
        print(f"Selected {device}")

        if model_type == "insert":
            tokenizer = AutoTokenizer.from_pretrained(self.insert_model, truncation=True, max_length=512, device=device)
            model = BertForSequenceClassification.from_pretrained(self.insert_model).to(device)
        elif model_type == "remove":
            tokenizer = AutoTokenizer.from_pretrained(self.remove_model, truncation=True, max_length=512, device=device)
            model = BertForSequenceClassification.from_pretrained(self.remove_model).to(device)
        elif model_type == "change":
            tokenizer = AutoTokenizer.from_pretrained(self.change_model, truncation=True, max_length=512, device=device)
            model = BertForSequenceClassification.from_pretrained(self.change_model).to(device)
        elif model_type == "title":
            tokenizer = AutoTokenizer.from_pretrained(self.title_model, truncation=True, max_length=512, device=device)
            model = BertForSequenceClassification.from_pretrained(self.title_model).to(device)
        else:
            raise NotImplementedError

        clf = pipeline(task="text-classification", model=model, tokenizer=tokenizer, device=0, batch_size=124)

        print(f"Predicting {len(texts)} texts: ")
        tokenizer_kwargs = {'truncation': True, 'max_length': 512}

        preds = []
        for i in tqdm(range(0, len(texts), tqdm_batch_size)):
            preds += clf(texts[i:i + tqdm_batch_size], return_all_scores=True, function_to_apply="none",
                         **tokenizer_kwargs, batch_size=64)

        del tokenizer, model
        gc.collect()
        torch.cuda.empty_cache()

        print("Postprocessing: ")
        parser_preds = [self._preds_processing(p) for p in preds]
        preds_dict = {str(k): v for k, v in zip(texts, parser_preds)}

        return preds_dict


filename_pattern_train_input = "data/anon_train_{}.csv".format("_".join(cc_codes))
filename_pattern_test_input = "data/anon_test_{}.csv".format("_".join(cc_codes))
filename_pattern_test_full_input = "data/anon_test_full_{}.csv".format("_".join(cc_codes))

filename_pattern_train = "data/processed_anon_train_{}.csv".format("_".join(cc_codes))
filename_pattern_test = "data/processed_anon_test_{}.csv".format("_".join(cc_codes))
filename_pattern_test_full = "data/processed_anon_test_full_{}.csv".format("_".join(cc_codes))

feature_extractor = FeatureExtractor()

train_df = pd.read_csv(filename_pattern_train_input)
train_df = feature_extractor.get_features(train_df)
train_df.to_csv(filename_pattern_train, index=False)

test_df = pd.read_csv(filename_pattern_test_input)
test_df = feature_extractor.get_features(test_df)
test_df.to_csv(filename_pattern_test, index=False)

test_df_full = pd.read_csv(filename_pattern_test_full_input)
test_df_full = feature_extractor.get_features(test_df_full)
test_df_full.to_csv(filename_pattern_test_full, index=False)
