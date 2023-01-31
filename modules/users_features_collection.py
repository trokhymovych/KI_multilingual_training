import wmfdata
spark = wmfdata.spark.get_session(
    type='yarn-regular',
    app_name='mediawiki_history wikitext additional features collection',
    ship_python_env=True,
    extra_settings={
        "spark.executor.cores": 4,
        "spark.sql.broadcastTimeout": "3600",
        "spark.sql.autoBroadcastJoinThreshold": 10485760 # 10Mb
    }
)

from pyspark.sql import functions as F
import pandas as pd
from tqdm.auto import tqdm

cc_codes = [
    "ka", "lv", "ta", "ur", "eo", "lt", "sl", "hy", "hr", "sk", "eu", "et", "ms", "az", "da", "bg",
    "sr", "ro", "el", "th", "bn", "no", "hi", "ca", "hu", "ko", "fi", "vi", "uz", "sv", "cs", "he",
    "id", "tr", "uk", "nl", "pl", "ar", "fa", "it", "zh", "ru", "es", "ja", "de", "fr", "en"
]

files = [
    "data/processed_anon_test_full_{}.csv".format("_".join(cc_codes)),
    "data/processed_anon_test_{}.csv".format("_".join(cc_codes)),
    "data/processed_anon_train_{}.csv".format("_".join(cc_codes)),
    "data/all_users_train_{}.csv".format("_".join(cc_codes)),
    "data/all_users_test_{}.csv".format("_".join(cc_codes)),
]

train_dfs = [pd.read_csv(file) for file in tqdm(files)]
ccs = train_dfs[3].wiki_db.unique()

dfs = []

for ccwiki in tqdm(ccs):
    print(ccwiki, " loading dump")
    dump_name = f"mediawiki_history_{ccwiki[:2]}_07_2022.parquet"
    changes = spark.read.parquet(dump_name)

    for train_df in tqdm(train_dfs):
        print(ccwiki, " test_part")
        test_df_cc = train_df[train_df.wiki_db == ccwiki]
        columns = ["revision_id", "wiki_db"]
        sparkDF = spark.createDataFrame(test_df_cc[columns])

        columns = ["revision_id", "event_user_is_anonymous", "event_user_groups"]
        aggregated = changes.select(columns).join(F.broadcast(sparkDF), on=['revision_id'])
        aggregated_df = aggregated.toPandas()
        dfs.append(aggregated_df)

        print(f"Length initial: {len(test_df_cc)}")
        print(f"Length processed: {len(aggregated_df)}")

additional_features_df = pd.concat(dfs)
additional_features_df = additional_features_df.drop_duplicates(["revision_id", "wiki_db"])
additional_features_df.to_csv("data/additional_features.csv", index=False)

# Additional features processing
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

# getting features dict
import ast
additional_features_dict = {}
for r, db, ua, ug in tqdm(zip(additional_features_df.revision_id.values,
                         additional_features_df.wiki_db.values,
                         additional_features_df.event_user_is_anonymous.values,
                         additional_features_df.event_user_groups.fillna('[]').values)):
    features = {}
    features["is_anonymous"] = ua
    user_groups_tmp = ast.literal_eval(ug)
    for g in user_groups:
        if g in user_groups_tmp:
            features[f"is_{g}"] = True
        else:
            features[f"is_{g}"] = False
    additional_features_dict[(db, r)] = features

import pickle
with open('data/additional_features.pickle', 'wb') as handle:
    pickle.dump(additional_features_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)