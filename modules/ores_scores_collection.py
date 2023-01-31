import wmfdata
spark = wmfdata.spark.get_session(
    type='yarn-regular',
    app_name ='ores scores extraction',
    ship_python_env=True,
    extra_settings={
        "spark.sql.broadcastTimeout": "3600",
        "spark.sql.autoBroadcastJoinThreshold": 10485760 # 10Mb
    }
)
import pandas as pd
from pyspark.sql import functions as F
from tqdm.auto import tqdm

cc_codes = [
    "ka", "lv", "ta", "ur", "eo", "lt", "sl", "hy", "hr", "sk", "eu", "et", "ms", "az", "da", "bg",
    "sr", "ro", "el", "th", "bn", "no", "hi", "ca", "hu", "ko", "fi", "vi", "uz", "sv", "cs", "he",
    "id", "tr", "uk", "nl", "pl", "ar", "fa", "it", "zh", "ru", "es", "ja", "de", "fr", "en"
]

filename_pattern_test = "data/processed_test_full_{}.csv".format("_".join(cc_codes))
test_df = pd.read_csv(filename_pattern_test)
test_df["batch_id"] = [i % 50 for i in range(len(test_df))]
columns = ["wiki_db", "revision_id", "batch_id"]
sparkDF = spark.createDataFrame(test_df[columns])

# ### Collecting data
ores = spark.sql(
    f""" 
    SELECT database as wiki_db, scores, rev_id as revision_id FROM event_sanitized.mediawiki_revision_score
    WHERE year = 2022 AND page_namespace = 0
    """
)

dfs = []
for i in tqdm(range(50)):
    ores_ = ores.join(F.broadcast(sparkDF.where(sparkDF.batch_id==i).select('revision_id','wiki_db')),
                      on=['revision_id','wiki_db'])
    dfs.append(ores_.toPandas())

ores_df = pd.concat(dfs)
ores_df = ores_df.drop_duplicates(["revision_id", "wiki_db"])


def getOresPred(x):
    try:
        if x.get('damaging').prediction[0] == 'false':
            return False
        if x.get('damaging').prediction[0] == 'true':
            return True
    except:
        return None


def getOresProbs(x):
    try:
        return x.get('damaging').probability["true"]
    except:
        return None


ores_df['pred'] = ores_df.scores.apply(getOresPred)
ores_df['probs'] = ores_df.scores.apply(getOresProbs)

ores_dict = {(db, idd): pred for db, idd, pred in zip(ores_df.wiki_db, ores_df.revision_id, ores_df.pred)}
preds = []
for db, idd in zip(test_df.wiki_db, test_df.revision_id):
    preds.append(ores_dict.get((db, idd)))

test_df["ores_preds"] = preds


ores_dict = {(db, idd): pred for db, idd, pred in zip(ores_df.wiki_db, ores_df.revision_id, ores_df.probs)}
preds = []
for db, idd in zip(test_df.wiki_db, test_df.revision_id):
    preds.append(ores_dict.get((db, idd)))

test_df["ores_pred"] = preds
test_df[["wiki_db", "revision_id", "ores_pred"]].to_csv("data/test_ores_scores_full_test.csv", index = False)