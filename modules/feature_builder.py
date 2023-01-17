import pandas as pd
import numpy as np
from tqdm import tqdm
import ast

from datasets import load_dataset, load_metric, Dataset, ClassLabel
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

from datasets import load_metric
from sklearn.metrics import mean_squared_error

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

PREFIX = "anonymous"
MODEL = "bert-base-multilingual-cased"


class TextPreparer:
    def __init__(self, mode="changes", balance=True):
        self.mode = mode
        self.balance = balance

    def _build_text_train_title(self, train_df):
        MIN_REV = 5
        titles_stats = train_df.groupby("page_title").agg(
            {"revision_is_identity_reverted": ["mean", "count"], "wiki_db": ["first"]}).reset_index()
        titles_stats.columns = ["page_title", "label", "rev_count", "wikidb"]
        titles_stats = titles_stats[titles_stats.rev_count >= MIN_REV]
        titles_stats.page_title = titles_stats.page_title.apply(lambda x: x.replace("_", " "))
        titles_stats.to_csv(f"data/{PREFIX}_titles_semantic.csv", index=False)

        return f"data/{PREFIX}_titles_semantic.csv"

    def build_text_train(self, train_df) -> str:

        if self.mode == "title":
            return self._build_text_train_title(train_df)

        text_changes = []
        target = []
        revisions = []
        lang = []
        target_column = "revision_is_identity_reverted"

        columns_mapping = {"changes": "texts_change", "inserts": "texts_insert", "removes": "texts_removed"}

        for rev, text, label, lang_ in tqdm(
                zip(train_df.revision_id, train_df[columns_mapping[self.mode]], train_df[target_column],
                    train_df["wiki_db"])):
            texts_to_add = ast.literal_eval(text)
            if len(texts_to_add) != 1:
                continue
            for texts in texts_to_add:
                key = 0
                if self.mode in ["inserts", "removes"]:
                    if (len(texts) > 0):
                        key = 1
                elif self.mode in ["changes"]:
                    if texts[0] != texts[1]:
                        key = 1
                else:
                    pass
                if key == 1:
                    text_changes += [texts]
                    target += [label]
                    revisions += [rev]
                    lang += [lang_]
        print(len(text_changes), len(target), len(revisions), len(lang))

        print(f"Collected {len(target)} records")
        columns = ["sentence1", "sentence2"] if self.mode == "changes" else ["sentence1"]
        print(columns)
        train_2 = pd.DataFrame(text_changes, columns=columns)
        train_2.dropna(inplace=True)
        train_2["label"] = target
        train_2["revision_id"] = revisions
        train_2["lang"] = lang

        train_2.sentence1 = train_2.sentence1.astype(str)
        train_2 = train_2[~train_2.sentence1.isin(['N/A', 'NA', "n/a", "na", "None", "nan", "N/a"])]
        if self.mode == "changes":
            train_2.sentence2 = train_2.sentence2.astype(str)
            train_2 = train_2[~train_2.sentence2.isin(['N/A', 'NA', "n/a", "na", "None", "nan", "N/a"])]
        print(train_2)
        # building balanced dataset
        if self.balance:
            part_1 = train_2[train_2.label == 1]
            part_2 = train_2[train_2.label == 0]

            part_2 = part_2.sample(np.min([len(part_1), len(part_2)]), random_state=42)
            balanced = pd.concat([part_1, part_2]).sample(len(part_1) + len(part_2), random_state=42)
            balanced.to_csv(f"data/{PREFIX}_text_{self.mode}_train_balanced.csv", index=False)
            return f"data/{PREFIX}_text_{self.mode}_train_balanced.csv"
        else:
            train_2.to_csv(f"data/{PREFIX}_text_{self.mode}_train_not-balanced.csv", index=False)
            return f"data/{PREFIX}_text_{self.mode}_train_not-balanced.csv"


class MLMTrainer:
    def __init__(self, train_path, base_model, mode):
        self.training_dataset = Dataset.from_csv(train_path)
        self.mode = mode
        self.base_model = base_model

    def train_model(self):

        # tokenization:
        tokenizer = AutoTokenizer.from_pretrained(self.base_model, use_fast=True)

        if self.mode != "title":
            sentence1_key = "sentence1"
            sentence2_key = "sentence2" if self.mode == "changes" else None

            feat_class = ClassLabel(num_classes=2, names=["not_reverted", "reverted"])
            training_dataset = self.training_dataset.cast_column("label", feat_class)
            training_dataset = training_dataset.train_test_split(
                test_size=0.05, stratify_by_column="label", shuffle=True, seed=42
            )

            def preprocess_function(examples):
                if sentence2_key is None:
                    return tokenizer(examples[sentence1_key], truncation=True)
                return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, max_length=512)

            encoded_dataset = training_dataset.map(preprocess_function, batched=True)

            num_labels = 2
            batch_size = 8
            metric_name = "accuracy"
            model_name = self.base_model.split("/")[-1]

            model = AutoModelForSequenceClassification.from_pretrained(self.base_model, num_labels=num_labels)
        else:
            # tokenization:
            sentence1_key = "page_title"
            sentence2_key = None

            training_dataset = self.training_dataset.train_test_split(test_size=0.05, shuffle=True, seed=42)

            def preprocess_function(examples):
                return tokenizer(examples[sentence1_key], truncation=True, max_length=512)

            encoded_dataset = training_dataset.map(preprocess_function, batched=True)
            num_labels = 1
            metric_name = "rmse"
            model_name = self.base_model.split("/")[-1]
            batch_size = 8
            model = AutoModelForSequenceClassification.from_pretrained(self.base_model, num_labels=num_labels)

        args = TrainingArguments(
            f"models/{self.mode}_{model_name}_balanced",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=5,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model=metric_name,
            push_to_hub=False,
        )

        if self.mode != "title":
            metric = load_metric("glue", "mrpc")

            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                predictions = np.argmax(predictions, axis=1)
                return metric.compute(predictions=predictions, references=labels)
        else:
            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                rmse = mean_squared_error(labels, predictions, squared=False)
                return {"rmse": rmse}

        trainer = Trainer(
            model,
            args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset["test"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
        print("RESULTS")
        print(trainer.evaluate())


data_path = "data/anon_train_ka_lv_ta_ur_eo_lt_sl_hy_hr_sk_eu_et_ms_az_da_bg_sr_ro_el_th_bn_no_hi_ca_hu_ko_fi_vi_uz_sv_cs_he_id_tr_uk_nl_pl_ar_fa_it_zh_ru_es_ja_de_fr_en.csv"
print("Processing path: ", data_path)
train_df = pd.read_csv(data_path)
train_df = train_df[train_df["is_text_train"] == 1]
train_df = train_df
print(train_df.columns)

mode = "title"
PREFIX = "anonymous"
MODEL = "bert-base-multilingual-cased"

for mode in ["title", "changes", "inserts", "removes"]:
    print(mode)
    preparer = TextPreparer(mode)
    prepared_data_path = preparer.build_text_train(train_df)
    print("Data prepared")

    trainer = MLMTrainer(train_path=prepared_data_path, base_model=MODEL, mode=mode)
    trainer.train_model()
    print("Model trained")
