import wmfdata
import argparse

parser = argparse.ArgumentParser(description='Script for training/testing data for KI model collection')

parser.add_argument('-l', '--lang', help='Language code', required=True, default="uk")
parser.add_argument('-m', '--mode', help='Mode', required=True, default="train")
parser.add_argument('-f1', '--filter_1', help='Filter anon users', required=True, default=0)
parser.add_argument('-f2', '--filter_2', help='Filter revision wars', required=True, default=0)
parser.add_argument('-n', '--max_records', help='Maximum amount of records to collect', required=True, default=300000)
parser.add_argument('-uc', '--use_cache', help='Use pre-collected cached data', required=True, default=0)


args = vars(parser.parse_args())
LANG = args["lang"]
MODE = args["mode"]
FILTER_ANON = int(args["filter_1"])
FILTER_RW = int(args["filter_2"])
MAX_COUNT = int(args["max_records"])
USE_CACHE = int(args["use_cache"])

# Collecting data setup (fixed for now)
snapshot = "'2022-07'"
wiki_db = f"'{LANG}wiki'"
min_timestamp = "'2022-01-01'"
max_timestamp = "'2022-08-01'"
dump_name = f"mediawiki_history_{LANG}_07_2022.parquet"

spark = wmfdata.spark.get_session(
    type='yarn-regular',
    app_name=f'mediawiki_history wikitext {LANG} {MODE}',
    ship_python_env=True,
    extra_settings={
        "spark.executor.cores": 2,
        "spark.sql.broadcastTimeout": "3600",
        "spark.sql.autoBroadcastJoinThreshold": 10485760  # 10Mb
    }
)
print(f"Starting {MODE} data collection for {LANG}")

from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.functions import udf
from collections.abc import Iterable

import difflib
import itertools
import re

from typing import Any, List, Set

import catboost as catb  # type: ignore
import joblib  # type: ignore
import mwedittypes  # type: ignore
from fuzzywuzzy import fuzz  # type: ignore

FUZZY_THRESHOLD = 60
ALLOWED_CONTENT_TYPES = [
    "Argument",
    "Category",
    "Comment",
    "ExternalLink",
    "Gallery",
    "HTMLEntity",
    "Heading",
    "List",
    "Media",
    "Paragraph",
    "Punctuation",
    "Reference",
    "Section",
    "Sentence",
    "Table",
    "Table Element",
    "Template",
    "Text",
    "Text Formatting",
    "Whitespace",
    "Wikilink",
    "Word",
]
ALLOWED_ACTION_TYPES = ["change", "insert", "move", "remove"]


def sentence_tokenize(text: str) -> List[str]:
    """
    Basic method used to split text into sentences
    """
    return list(map(str.strip, re.split(r"[.!?](?!$)", text)))


def flatten_list(list_of_items: List[List[Any]]) -> List[Any]:
    """
    Method to flatten the list
    """
    return [item for sublist in list_of_items for item in sublist]


def diff_lines(old: str, new: str):
    """Splits `old` and `new` into lines and calculates the
    `TextDiff` for them.
    """
    old_lines, new_lines = old.splitlines(), new.splitlines()
    matcher = difflib.SequenceMatcher(isjunk=None, a=old_lines, b=new_lines)
    added, deleted = set(), set()
    for tag, old_lo, old_hi, new_lo, new_hi in matcher.get_opcodes():
        if tag != "equal":
            added.update(
                flatten_list([sentence_tokenize(s) for s in new_lines[new_lo:new_hi]])
            )
            deleted.update(
                flatten_list([sentence_tokenize(s) for s in old_lines[old_lo:old_hi]])
            )

    changed = []
    seen: Set[str] = set()
    for deleted_line, added_line in itertools.product(deleted, added):
        if deleted_line not in seen and added_line not in seen:
            similarity = fuzz.ratio(deleted_line, added_line)
            if similarity > FUZZY_THRESHOLD and deleted_line != added_line:
                changed.append((deleted_line, added_line))
                seen.update((deleted_line, added_line))
            elif deleted_line == added_line:
                seen.update((deleted_line, added_line))

    return list(added - seen), list(deleted - seen), changed


def get_changes(curr_wikitext, prev_wikitext, lang=LANG):
    try:
        et = mwedittypes.EditTypes(prev_wikitext, curr_wikitext, lang)
        actions = et.get_diff()
        filtered_actions = {
            f"{at}_{ct}": actions.get(ct, {}).get(at, 0)
            for ct in ALLOWED_CONTENT_TYPES
            for at in ALLOWED_ACTION_TYPES
        }

        lines_added = [
            node["text"] for node in et.tree_diff["insert"] if node["type"] == "Text"
        ]
        lines_deleted = [
            node["text"] for node in et.tree_diff["remove"] if node["type"] == "Text"
        ]
        lines_changed = []
        for node in et.tree_diff["change"]:
            if node["prev"]["type"] == node["curr"]["type"] == "Text":
                change_diff = diff_lines(node["prev"]["text"], node["curr"]["text"])
                lines_added.extend(change_diff[0])
                lines_deleted.extend(change_diff[1])
                lines_changed.extend(change_diff[2])
        return [str(lines_deleted), str(lines_added), str(lines_changed), str(filtered_actions)]
    except:
        return [str(list()), str(list()), str(list()), str(dict())]


features_to_observe = [
    "wiki_db",
    "event_comment",
    "event_user_text_historical",
    "event_user_is_bot_by",
    "event_user_is_anonymous",
    "event_user_groups",
    "event_user_seconds_since_previous_revision",
    "revision_id",
    "page_title",
    "page_revision_count",
    "revision_text_bytes_diff",
    "revision_is_identity_reverted",
    "event_timestamp",
    "revision_tags",
    "revision_parent_id",
    "revision_first_identity_reverting_revision_id"
]

if not USE_CACHE:
    changes = spark.sql(
        f'''
           SELECT {','.join(features_to_observe)}
           FROM wmf.mediawiki_history 
           WHERE event_entity = 'revision' AND wiki_db = {wiki_db} AND snapshot = {snapshot} 
           AND event_timestamp >= {min_timestamp} AND event_timestamp < {max_timestamp}
           AND page_namespace = 0
        '''
    )

    changes.write.parquet(dump_name, mode="overwrite")

changes = spark.read.parquet(dump_name)
print(changes.count())
print(f"Initial file is saved to {dump_name}")

if MODE == "test":
    min_timestamp = '2022-07-01'
    max_timestamp = '2022-07-08'
    changes = changes.where(changes["event_timestamp"] >= F.lit(min_timestamp))
    changes = changes.where(changes["event_timestamp"] < F.lit(max_timestamp))

# change type of target column for stats calculation
target_column = "revision_is_identity_reverted"
changes = changes.withColumn(target_column, changes[target_column].cast(T.IntegerType()))

# adding info regarding reverting revisions
columns_to_select = changes.columns
changes_new = changes.alias("t1") \
    .join(changes.alias("t2"), F.col("t1.revision_first_identity_reverting_revision_id") == F.col("t2.revision_id"),
          "left") \
    .select(
    *[f"t1.{c}" for c in changes.columns],
    F.col("t2.revision_is_identity_reverted").alias("reverting_revision_is_reverted_revision"),
    F.col("t2.event_user_groups").alias("reverting_revision_event_user_groups"),
)

# rate of "bad" reverts (reverts, that was later also reverted) -> we should filter out those
changes_new.where(F.col("revision_is_identity_reverted") == True) \
    .select(F.mean(F.col("reverting_revision_is_reverted_revision").cast(T.IntegerType()))).show()

# filter out only "not_reverted" revisions and reverted revisions that were reverted by "good" revert
changes_new = changes_new.fillna({'reverting_revision_is_reverted_revision': False})

# filter revision wars
if FILTER_RW:
    changes_new = changes_new.where(changes_new["reverting_revision_is_reverted_revision"] != True)


# filter bots
@udf("Integer")
def is_bot(user_groups: List[Any]):
    if isinstance(user_groups, Iterable):
        return 1 if len(user_groups) > 0 else 0
    else:
        return 0


changes_new = changes_new.withColumn("is_bot", is_bot(F.col("event_user_is_bot_by")))
changes_new = changes_new.where(changes_new["is_bot"] == 0)

# filter bots
changes_new = changes_new.where(F.col("revision_parent_id") != 0)

# Leave only is_anonymous users
if FILTER_ANON:
    changes_new = changes_new.where(changes_new["event_user_is_anonymous"] == True)

# Artificial balancing (in case we have more than 300k records)
n_revisions = changes_new.count()
print(f"Number of revisions: {n_revisions}")
if n_revisions > MAX_COUNT:
    changes_new = changes_new.sample(MAX_COUNT / n_revisions, seed=42)
    print(f"Doing random sampling to limit number of revisions per language to {MAX_COUNT}")


# Mark if reverted revision is reverted by experienced user
@udf("Integer")
def is_good_user(user_groups):
    good_users = ['sysop', 'oversight', 'editor', 'rollbacker', 'checkuser', 'abusefilter', 'bureaucrat']
    if isinstance(user_groups, Iterable):
        return 1 if len([i for i in user_groups if i in good_users]) > 0 else 0
    else:
        return 0


changes_new = \
    changes_new.withColumn("is_reverted_by_good_user", is_good_user(F.col("reverting_revision_event_user_groups")))

# Mark specific revision tags:
revision_tags_to_get = [
    "mobile edit",
    "mobile web edit",
    "visualeditor",
    "wikieditor",
    "mobile app edit",
    "android app edit",
    "ios app edit"
]
for tag in revision_tags_to_get:
    @udf("Integer")
    def is_tag(tags_list, tag=tag):
        if isinstance(tags_list, Iterable):
            return 1 if tag in tags_list else 0
        else:
            return 0


    changes_new = changes_new.withColumn(f"is_{tag.replace(' ', '_')}", is_tag(F.col("revision_tags")))

# Make initial dump (without text features)
columns_to_save = [
    'wiki_db',
    'event_comment',
    'event_user_text_historical',
    'event_user_seconds_since_previous_revision',
    'revision_id',
    'page_title',
    'page_revision_count',
    'revision_text_bytes_diff',
    'revision_is_identity_reverted',
    'event_timestamp',
    'revision_parent_id',
    'revision_first_identity_reverting_revision_id',
    'reverting_revision_is_reverted_revision',
    'is_reverted_by_good_user',
    'is_mobile_edit',
    'is_mobile_web_edit',
    'is_visualeditor',
    'is_wikieditor',
    'is_mobile_app_edit',
    'is_android_app_edit',
    'is_ios_app_edit'
]

dump_name = f"{LANG}_mediawiki_history_filtered_07_2022.parquet"
changes_new.select(columns_to_save).write.parquet(dump_name, mode="overwrite")

# collected prepared mediawiki_history
changes_new = spark.read.parquet(dump_name)
print(f"Number of revisions per lang={LANG}, {changes_new.count()}")

# Collecting text differences:
# FIY: Previous snapshots are dropping, so it is needed to update to newer in the future
snapshot = "2022-11"
wikitext_df = spark.sql(f"""SELECT revision_id, wiki_db, revision_text 
    FROM wmf.mediawiki_wikitext_history
    WHERE snapshot = '{snapshot}' and wiki_db = {wiki_db} and page_namespace = 0
    """)

# Merging full wikitexts per revision
history_columns = changes_new.columns
revisions_text_1 = wikitext_df.alias("text") \
    .join(
    changes_new.alias("history"),
    F.col("text.revision_id") == F.col("history.revision_id"),
    "right"
).select(
    *[F.col(f"history.{c}") for c in history_columns],
    F.col("text.revision_text").alias("revision_id_revision_text"),
).alias("t1")
revisions_text_2 = wikitext_df.alias("text") \
    .join(
    changes_new.alias("history"),
    F.col("text.revision_id") == F.col("history.revision_parent_id"),
    "right"
).select(
    F.col(f"history.revision_id"),
    F.col("text.revision_text").alias("revision_parent_id_revision_text")
).alias("t2")
revisions_text_all = revisions_text_1 \
    .join(revisions_text_2, F.col("t1.revision_id") == F.col("t2.revision_id")) \
    .select(
    F.col("t1.revision_id_revision_text").alias("revision_id_revision_text"),
    F.col("t2.revision_parent_id_revision_text").alias("revision_parent_id_revision_text"),
    *[F.col(f"t1.{c}") for c in history_columns],
)
revisions_text_all = revisions_text_all.where(F.col("revision_parent_id_revision_text").isNotNull())
revisions_text_all = revisions_text_all.where(F.col("revision_id_revision_text").isNotNull())

# Developing functions to take a diffs
schema = T.StructType([
    T.StructField("texts_removed", T.StringType(), True),
    T.StructField("texts_insert", T.StringType(), True),
    T.StructField("texts_change", T.StringType(), True),
    T.StructField("actions", T.StringType(), True),
])

udf_get_changes = F.udf(get_changes, schema)

# comment to work with sample only
revisions_text_all_sample = revisions_text_all.repartition(1024, "revision_id")

revisions_text_all_sample = revisions_text_all_sample \
    .withColumn("udf_res", udf_get_changes(revisions_text_all_sample['revision_id_revision_text'],
                                           revisions_text_all_sample['revision_parent_id_revision_text'],
                                           )) \
    .select(
        *history_columns,
        F.col("udf_res.texts_removed"),
        F.col("udf_res.texts_insert"),
        F.col("udf_res.texts_change"),
        F.col("udf_res.actions")
    )
# dump data to parquet:
dump_name = f"{LANG}_anonymous_text_07-2022_{MODE}"
revisions_text_all_sample.write.parquet(dump_name + ".parquet", mode="overwrite")

# dump data to csv:
revisions_text_all_sample = spark.read.parquet(dump_name + ".parquet")
revisions_text_all_sample_df = revisions_text_all_sample.toPandas()
revisions_text_all_sample_df.to_csv(f"data/{dump_name}.csv", index=False)
