import pandas as pd


def load_datasets_metrics(metric="rmse"):
    openfield_df = pd.read_hdf("data/openfield_ratios.h5")[metric.upper()]
    openfield_unbalanced_zeroshot = openfield_df.loc["unbalanced_zeroshot", "600000"]
    drop_list = [
        "balanced_memory_replay_threshold_0.0_750000",
        "unbalanced_zeroshot",
        "balanced_zeroshot",
        "balanced_super_remove_head_750000",
        "balanced_memory_replay_threshold_0.8_snapshot_700000",
        "balanced_memory_replay_threshold_0.8_750000",
        "unbalanced_memory_replay_750000",
    ]
    openfield_df.drop(drop_list, inplace=True)
    rename_dict = {
        "baseline": "ImageNet transfer learning",
        "zeroshot": "SA + Zeroshot",
        "unbalanced_super_remove_head_750000": "SA + Randomly Initialized Decoder",
        "unbalanced_memory_replay_threshold_0.8_750000": "SA + Memory Replay",
        "unbalanced_naive_finetune_750000": "SA + Naive Fine-tuning",
    }
    openfield_df.rename(index=rename_dict, inplace=True)

    rodent_df = pd.read_hdf("data/rodent_ratios.h5")[metric.upper()]
    rodent_unbalanced_zeroshot = rodent_df.loc["unbalanced_zeroshot", "700000"]
    drop_list = ["unbalanced_zeroshot"]
    rodent_df.drop(drop_list, inplace=True)
    rename_dict = {
        "baseline": "ImageNet transfer learning",
        "zeroshot": "SA + Zeroshot",
        "super_remove_head": "SA + Randomly Initialized Decoder",
        "unbalanced_memory_replay_threshold_0.8_700000": "SA + Memory Replay",
        "unbalanced_naive_finetune_700000": "SA + Naive Fine-tuning",
    }
    rodent_df.rename(index=rename_dict, inplace=True)

    horse_df = pd.read_hdf("data/horse_ratios.h5")[f"{metric.upper()}_iid"]
    horse_unbalanced_zeroshot = horse_df.loc["unbalanced_zeroshot", "1000"]
    horse_unbalanced_zeroshot.index = horse_unbalanced_zeroshot.index.str.replace(
        "_best", ""
    )
    horse_unbalanced_zeroshot.rename(
        {
            "shuffle1": "shuffle0",
            "shuffle2": "shuffle1",
            "shuffle3": "shuffle2",
        },
        inplace=True,
    )
    drop_list = ["unbalanced_zeroshot", "baseline_dev"]
    horse_df.drop(drop_list, inplace=True)
    rename_dict = {
        "baseline": "ImageNet transfer learning",
        "zeroshot": "SA + Zeroshot",
        "super_remove_head": "SA + Randomly Initialized Decoder",
        "unbalanced_memory_replay_threshold_0.8_700000": "SA + Memory Replay",
        "unbalanced_naive_finetune_700000": "SA + Naive Fine-tuning",
        "shuffle1": "shuffle0",
        "shuffle2": "shuffle1",
        "shuffle3": "shuffle2",
    }
    horse_df.rename(index=rename_dict, inplace=True)

    def _reset_df_index(df_):
        df = df_.reset_index()
        df.columns = ["method", "frac", "shuffle", metric]
        return df

    def _add_zeroshot(df_, vals):
        df = pd.concat([vals.to_frame().reset_index()] * 5)
        df["frac"] = [frac for frac in list(df_["frac"].unique()) for _ in range(3)]
        df["method"] = "zeroshot"
        df.columns = ["shuffle", metric, "frac", "method"]
        return pd.concat((df_, df)).reset_index(drop=True)

    df_openfield = _reset_df_index(openfield_df)
    df_openfield = _add_zeroshot(df_openfield, openfield_unbalanced_zeroshot)
    df_openfield["dataset"] = "openfield"
    df_rodent = _reset_df_index(rodent_df)
    df_rodent = _add_zeroshot(df_rodent, rodent_unbalanced_zeroshot)
    df_rodent["dataset"] = "rodent"
    df_horse = _reset_df_index(horse_df)
    df_horse = _add_zeroshot(df_horse, horse_unbalanced_zeroshot)
    df_horse["dataset"] = "horse"
    return pd.concat((df_openfield, df_rodent, df_horse)).reset_index(drop=True)
