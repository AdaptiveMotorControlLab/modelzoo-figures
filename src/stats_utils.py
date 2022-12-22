import pandas as pd


def load_datasets_rmse():
    openfield_df = pd.read_hdf("data/openfield_ratios.h5")["RMSE"]
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

    rodent_df = pd.read_hdf("data/rodent_ratios.h5")["RMSE"]
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

    horse_df = pd.read_hdf("data/horse_ratios.h5")["RMSE_iid"]
    horse_unbalanced_zeroshot = horse_df.loc["unbalanced_zeroshot", "1000"]
    horse_unbalanced_zeroshot.index = horse_unbalanced_zeroshot.index.str.replace('_best', '')
    drop_list = ["unbalanced_zeroshot", "baseline_dev"]
    horse_df.drop(drop_list, inplace=True)
    rename_dict = {
        "baseline": "ImageNet transfer learning",
        "zeroshot": "SA + Zeroshot",
        "super_remove_head": "SA + Randomly Initialized Decoder",
        "unbalanced_memory_replay_threshold_0.8_700000": "SA + Memory Replay",
        "unbalanced_naive_finetune_700000": "SA + Naive Fine-tuning",
    }
    horse_df.rename(index=rename_dict, inplace=True)

    def _reset_df_index(df_):
        df = df_.reset_index()
        df.columns = ["method", "frac", "shuffle", "rmse"]
        return df

    def _add_zeroshot(df_, vals):
        df = pd.concat([vals.to_frame().reset_index()] * 5)
        df["frac"] = [
            frac for frac in list(df_openfield_rmse["frac"].unique()) for _ in range(3)
        ]
        df["method"] = "zeroshot"
        df.columns = ["shuffle", "rmse", "frac", "method"]
        return pd.concat((df_, df)).reset_index(drop=True)

    df_openfield_rmse = _reset_df_index(openfield_df)
    df_openfield_rmse = _add_zeroshot(df_openfield_rmse, openfield_unbalanced_zeroshot)
    df_openfield_rmse["dataset"] = "openfield"
    df_rodent_rmse = _reset_df_index(rodent_df)
    df_rodent_rmse = _add_zeroshot(df_rodent_rmse, rodent_unbalanced_zeroshot)
    df_rodent_rmse["dataset"] = "rodent"
    df_horse_rmse = _reset_df_index(horse_df)
    df_horse_rmse = _add_zeroshot(df_horse_rmse, horse_unbalanced_zeroshot)
    df_horse_rmse["dataset"] = "horse"

    return pd.concat((df_openfield_rmse, df_rodent_rmse, df_horse_rmse)).reset_index(
        drop=True
    )
