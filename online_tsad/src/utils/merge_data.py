import pandas as pd
import argparse
import os

from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path_1", type=str, default='ucr')
    parser.add_argument("--data_path_2", type=str, default='ucr')
    parser.add_argument("--store_path", type=str,default='ucr')
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--anomaly_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train_df, val_df, test_df, test_metas = [], [], [], []
    for type_idx, name in enumerate([args.data_path_1, args.data_path_2]):
        normal_df = pd.read_parquet(os.path.join('data', name, 'normal.parquet'))
        anomaly_df = pd.read_parquet(os.path.join('data', name, 'generated_tsa.parquet'))
        meta_df = pd.read_parquet(os.path.join('data', name, 'meta_data.parquet'))
        meta_df['type'] = -1

        if type_idx == 0:
            train_idx, test_idx = train_test_split(range(normal_df.shape[0]), train_size=args.train_ratio,
                                                   random_state=args.seed)
            val_idx, test_idx = train_test_split(test_idx, train_size=args.val_ratio / (1 - args.train_ratio),
                                                 random_state=args.seed)

        train_df.append(normal_df.iloc[train_idx])

        # val_idx_0, val_idx_1 = train_test_split(val_idx, test_size=args.anomaly_ratio, random_state=args.seed)
        # val_df.append(pd.concat([normal_df.iloc[val_idx_0], anomaly_df.iloc[val_idx_1]], axis=0))
        val_df.append(normal_df.iloc[val_idx])

        test_idx_0, test_idx_1 = train_test_split(test_idx, test_size=args.anomaly_ratio, random_state=args.seed)
        test_df.append(pd.concat([normal_df.iloc[test_idx_0], anomaly_df.iloc[test_idx_1]], axis=0))

        test_meta_0, test_meta_1 = meta_df.iloc[test_idx_0], meta_df.iloc[test_idx_1]
        test_meta_0.loc[:, ['level_h0', 'level_h1', 'length_h0', 'length_h1', 'level', 'start', 'length']] = 0
        test_meta_1.loc[:, 'type'] = type_idx
        test_meta = pd.concat([test_meta_0, test_meta_1], axis=0)
        test_metas.append(test_meta)

    train_df = pd.concat(train_df, axis=0).reset_index(drop=True)
    val_df = pd.concat(val_df, axis=0).reset_index(drop=True)
    test_df = pd.concat(test_df, axis=0).reset_index(drop=True)
    test_metas = pd.concat(test_metas, axis=0).reset_index(drop=True)

    os.makedirs(f"data/{args.store_path}/", exist_ok=True)
    train_df.to_parquet(f"data/{args.store_path}/train_data.parquet", index=False)
    val_df.to_parquet(f"data/{args.store_path}/val_data.parquet", index=False)
    test_df.to_parquet(f"data/{args.store_path}/test_data.parquet", index=False)
    test_metas.to_parquet(f"data/{args.store_path}/test_meta.parquet", index=False)
    print(f'Successfully merge {args.data_path_1} and {args.data_path_2} to {args.store_path}')
