import json
import os
import random
from argparse import ArgumentParser, ArgumentTypeError
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from rdkit.Chem import Descriptors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from chemprop.v2 import data, featurizers, models
from chemprop.v2.models import modules
from noisefunctions import descriptor_bias, gauss_noise
from splitfunctions import split_by_prop_dict


def main():
    parser = ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    if args.model_type == "single_fidelity" and (args.add_pn_bias_to_make_lf > 0 or args.add_gauss_noise_to_make_lf > 0 or args.add_descriptor_bias_to_make_lf > 0):
        raise ValueError(
            "Cannot add bias to make low fidelity data when model type is single fidelity"
        )

    if args.model_type in [
        "multi_fidelity_weight_sharing_non_diff",
        "trad_delta_ml",
    ]:
        raise NotImplementedError("Not implemented yet")

    # make unique folder for results of each run
    os.makedirs(args.results_dir, exist_ok=True)
    os.chdir(args.results_dir)
    now = datetime.now()
    datetime_string = now.strftime("%Y-%m-%d_%H-%M-%S.%f")
    os.mkdir(datetime_string)
    os.chdir(datetime_string)

    # output args to a file
    args_dict = vars(args)
    with open("args.json", "w") as f:
        json.dump(args_dict, f, indent=4)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Model
    mgf = featurizers.MoleculeFeaturizer()
    mp_block_hf = modules.molecule_block()  # TODO: use aggregation='sum' or 'norm' instead of default 'mean'? (also on line below)
    mp_block_lf = modules.molecule_block()

    model_dict = {
        "single_fidelity": models.RegressionMPNN(mp_block_hf, n_tasks=1),
        "multi_target": models.RegressionMPNN(
            mp_block_hf, n_tasks=2
        ),
        "multi_fidelity": models.MultifidelityRegressionMPNN(
            mp_block_hf, n_tasks=1, mpn_block_low_fidelity=mp_block_lf
        ),
        "multi_fidelity_weight_sharing": models.MultifidelityRegressionMPNN(
            mp_block_hf, n_tasks=1
        ),
        "delta_ml": models.RegressionMPNN(mp_block_hf, n_tasks=1),
        "trad_ delta_ml": models.RegressionMPNN(mp_block_hf, n_tasks=1),
        # "multi_fidelity_weight_sharing_non_diff": ,  # TODO: (!) multi-fidelity non-differentiable feature
    }
    # TODO: add method: multi-fidelity with evidential uncertainty?
    # TODO: add methods: transfer learning, deltaML, Bayesian methods, etc.

    mpnn = model_dict[args.model_type]

    # Data
    data_df = pd.read_csv(args.data_file, index_col="smiles")

    # So multiple noises can be added on top of one another
    if args.model_type != "single_fidelity":
        data_df[args.lf_col_name] = data_df[args.hf_col_name]

    if args.add_pn_bias_to_make_lf > 0:
        # Creating the coefficients for the polynomial function
        coefficients = np.random.uniform(-1, 1, args.add_pn_bias_to_make_lf + 1)  # need to add 1 because the one coefficient is for x^0
        # Adding bias calculated from the polynomial function of HF to data_df LF column
        data_df[args.lf_col_name] = data_df[args.lf_col_name] + np.polyval(coefficients, list(data_df[args.hf_col_name]))

    if args.add_constant_bias_to_make_lf != 0.0:
        # Add the constant bias to HF to make data_df LF column
        data_df[args.lf_col_name] = data_df[args.lf_col_name] + args.add_constant_bias_to_make_lf

    if args.add_gauss_noise_to_make_lf > 0.0:
        data_df[args.lf_col_name] = data_df[args.lf_col_name] + gauss_noise(df_len=len(data_df[args.lf_col_name]), std=args.add_gauss_noise_to_make_lf, seed=args.seed)

    if args.add_descriptor_bias_to_make_lf != 0.0:
        descriptors = [
            Descriptors.qed, Descriptors.MolWt, Descriptors.BalabanJ,
            Descriptors.BertzCT, Descriptors.HallKierAlpha, Descriptors.Ipc,
            Descriptors.Kappa1, Descriptors.Kappa2, Descriptors.Kappa3,
            Descriptors.LabuteASA, Descriptors.TPSA, Descriptors.MolLogP,
            Descriptors.MolMR
        ]
        # Creating the weight descriptor pair
        coefficients = np.random.uniform(-1, 1, len(descriptors)) * args.add_descriptor_bias_to_make_lf
        descriptors_coefficients = list(zip(descriptors, coefficients))
        # Adding bias calculated from normalized descriptors to data_df LF column
        data_df[args.lf_col_name] = data_df[args.lf_col_name] + descriptor_bias(data_df, descriptors_coefficients)

    if args.model_type != "single_fidelity":
        export_and_plot_hf_lf_data(data_df, args)

    if args.model_type == "single_fidelity":
        targets = data_df[[args.hf_col_name]].values.reshape(-1, 1)

        hf_train_index, hf_test_index = split_by_prop_dict[args.split_type](
            df=data_df)

        train_index = hf_train_index
        test_index = hf_test_index

        # Selecting the target values for train and test
        train_t = data_df.loc[train_index][[args.hf_col_name]].values
        test_t = data_df.loc[test_index][[args.hf_col_name]].values

    else:

        if args.lf_superset_of_hf:  # TODO: (!!) are test sets different depending on LF:HF ratio? so compare within but not between?
            hf_frac = 1 / args.lf_hf_size_ratio
            lf_df = data_df.copy()
            hf_df = data_df.copy().sample(frac=hf_frac, random_state=args.seed)
        else:
            hf_frac = 1 / (args.lf_hf_size_ratio + 1)
            hf_df = data_df.sample(frac=hf_frac, random_state=args.seed)
            lf_df = data_df.drop(index=hf_df.index)

        # Creating a list of train and test indexes
        hf_df.drop(args.lf_col_name, inplace=True, axis=1)
        lf_df.drop(args.hf_col_name, inplace=True, axis=1)

        # TODO: (!) scaffold splits appear to not be reproducible???
        hf_train_index, hf_test_index = split_by_prop_dict[args.split_type](
            df=hf_df
        )
        lf_train_index, lf_test_index = split_by_prop_dict[args.split_type](
            df=lf_df
        )

        train_index = lf_train_index + hf_train_index
        test_index = lf_test_index + hf_test_index

        # If it is delta_ml then we don't need to set the LF HF values to nan, pass it in as feature
        if args.model_type == "delta_ml":
            # Oracle is LF data
            train_oracle = data_df.loc[train_index][[args.lf_col_name]].values
            test_oracle = data_df.loc[test_index][[args.lf_col_name]].values
            train_t = data_df.loc[train_index][[args.hf_col_name]].values
            test_t = data_df.loc[test_index][[args.hf_col_name]].values

        else:
            # Setting nan to specify LF and HF
            lf_not_hf_index = list(set(lf_train_index + lf_test_index).difference(set(hf_train_index + hf_test_index)))
            data_df[args.hf_col_name].loc[lf_not_hf_index] = np.nan
            if not args.lf_superset_of_hf:
                data_df[args.lf_col_name].loc[hf_train_index + hf_test_index] = np.nan

            # Selecting the target values for each train and test
            # LF column must be first, HF second to work with expected order in loss function during training
            train_t = data_df.loc[train_index][[args.lf_col_name, args.hf_col_name]].values
            test_t = data_df.loc[test_index][[args.lf_col_name, args.hf_col_name]].values

    # print(data_df)
    # print('---------------------------')
    # print(train_t)
    # print(train_oracle)
    # Initializing the data
    if args.model_type == "delta_ml":
        train_data = [data.MoleculeDatapoint(smi, t, features=o) for smi, t, o in zip(train_index, train_t, train_oracle)]
        test_data = [data.MoleculeDatapoint(smi, t, features=o) for smi, t, o in zip(test_index, test_t,test_oracle)]

    else:
        train_data = [data.MoleculeDatapoint(smi, t) for smi, t in zip(train_index, train_t)]
        test_data = [data.MoleculeDatapoint(smi, t) for smi, t in zip(test_index, test_t)]

    # TODO: (!) also do non-random split between train and val?
    train_data, val_data = train_test_split(train_data, test_size=0.11, random_state=args.seed)

    train_dset = data.MoleculeDataset(train_data, mgf)
    val_dset = data.MoleculeDataset(val_data, mgf)
    test_dset = data.MoleculeDataset(test_data, mgf)

    if args.scale_data:
        train_scaler = train_dset.normalize_targets()
        _ = val_dset.normalize_targets(train_scaler)
        test_scaler = test_dset.normalize_targets()

    train_loader = data.MolGraphDataLoader(train_dset, batch_size=50, num_workers=12)
    val_loader = data.MolGraphDataLoader(
        val_dset, batch_size=50, num_workers=12, shuffle=False
    )
    test_loader = data.MolGraphDataLoader(
        test_dset, batch_size=50, num_workers=12, shuffle=False
    )

    # Print sizes of datasets and splits
    print("Total dataset size:", len(data_df))
    if args.model_type != "single_fidelity":
        print("HF total size:", len(hf_df))
        print("HF train/val size:", len(hf_train_index))
        print("HF test size:", len(hf_test_index))
        print("LF total size:", len(lf_df))
        print("LF train/val size:", len(lf_train_index))
        print("LF test size:", len(lf_test_index))

    # Train
    trainer = pl.Trainer(
        logger=True,
        enable_checkpointing=False,
        enable_progress_bar=True,
        accelerator="gpu",
        devices=1,
        max_epochs=args.num_epochs,
    )
    trainer.fit(mpnn, train_loader, val_loader)

    preds = trainer.predict(mpnn, test_loader)

    test_smis = [x.smi for x in test_data]

    if args.model_type == "single_fidelity":
        preds = [x[0].item() for x in preds]
        targets = [x.targets[0] for x in test_data]

        if args.scale_data:
            preds = test_scaler.inverse_transform(np.array(preds).reshape(-1, 1))
            targets = test_scaler.inverse_transform(np.array(targets).reshape(-1, 1))
        else:
            preds = np.array(preds)
            targets = np.array(targets)

        print("Test set")
        mae, rmse, r2 = eval_metrics(targets, preds)
        metrics_df = pd.DataFrame({"MAE_hf": [mae], "RMSE_hf": [rmse], "R2_hf": [r2],
                                   "MAE_lf": [np.nan], "RMSE_lf": [np.nan], "R2_lf": [np.nan]})
        metrics_df.to_csv("test_metrics.csv", index=False)

        if args.save_test_plot:
            plt.scatter(targets, preds)
            plt.xlabel("Target")
            plt.ylabel("Prediction")
            plt.text(min(targets), max(preds),
                     f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR^2: {r2:.2f}",
                     fontsize=12, ha='left', va='top')
            plt.savefig("mf_test_preds.png")

        test_df = pd.DataFrame(
            {
                "smiles": test_smis,
                args.hf_col_name: targets.flatten(),
                f"{args.hf_col_name}_preds": preds.flatten(),
            }
        )
    else:
        if args.model_type == "multi_target":
            preds = np.array([x[0].numpy()[0] for x in preds])
        elif args.model_type in [
            "multi_fidelity",
            "multi_fidelity_weight_sharing",
        ]:  # TODO: (!) will this also work for multi-fidelity non-differentiable?
            preds = np.array([[[x[0][0].numpy(), x[0][1].numpy()]] for x in preds]).reshape(len(preds), 2)

        # Both HF and LF targets are identical if the only difference in the original HF and LF was a bias term -- this is not a bug -- once normalized, the network should learn both the same way
        targets = np.array([x.targets for x in test_data])

        if args.scale_data:
            preds = test_scaler.inverse_transform(preds)
            targets = test_scaler.inverse_transform(targets)

        targets_lf, targets_hf, preds_lf, preds_hf = [], [], [], []

        for target, pred in zip(targets, preds):
            # LF
            if not np.isnan(target[0]):
                targets_lf.append(target[0])
                preds_lf.append(pred[0])
            # HF
            if not np.isnan(target[1]):
                targets_hf.append(target[1])
                preds_hf.append(pred[1])

        targets_hf = np.array(targets_hf)
        targets_lf = np.array(targets_lf)
        preds_hf = np.array(preds_hf)
        preds_lf = np.array(preds_lf)

        print("High Fidelity - Test set")
        hf_mae, hf_rmse, hf_r2 = eval_metrics(targets_hf, preds_hf)
        print("Low Fidelity - Test set")
        lf_mae, lf_rmse, lf_r2 = eval_metrics(targets_lf, preds_lf)
        metrics_df = pd.DataFrame({"MAE_hf": [hf_mae], "RMSE_hf": [hf_rmse], "R2_hf": [hf_r2],
                                   "MAE_lf": [lf_mae], "RMSE_lf": [lf_rmse], "R2_lf": [lf_r2]})
        metrics_df.to_csv("test_metrics.csv", index=False)

        if args.save_test_plot:
            fig, axes = plt.subplots(figsize=(6, 3), nrows=1, ncols=2)

            axes[0].scatter(targets_hf, preds_hf, alpha=0.3, label="High Fidelity")
            axes[0].set_xlabel("Target")
            axes[0].set_ylabel("Prediction")
            axes[0].text(min(targets_hf), max(preds_hf),
                         f"MAE: {hf_mae:.2f}\nRMSE: {hf_rmse:.2f}\nR^2: {hf_r2:.2f}",
                         fontsize=12, ha='left', va='top')
            axes[0].set_title("High Fidelity")

            axes[1].scatter(targets_lf, preds_lf, alpha=0.3, label="Low Fidelity")
            axes[1].set_xlabel("Target")
            axes[1].set_ylabel("Prediction")
            axes[1].text(min(targets_lf), max(preds_lf),
                         f"MAE: {lf_mae:.2f}\nRMSE: {lf_rmse:.2f}\nR^2: {lf_r2:.2f}",
                         fontsize=12, ha='left', va='top')
            axes[1].set_title("Low Fidelity")

            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

            plt.savefig("mf_test_preds.png", bbox_inches="tight")

        test_df = pd.DataFrame(
            {
                "smiles": test_smis,
                args.hf_col_name: targets[:, 1].flatten(),
                f"{args.hf_col_name}_preds": preds[:, 1].flatten(),
                args.lf_col_name: targets[:, 0].flatten(),
                f"{args.lf_col_name}_preds": preds[:, 0].flatten(),
            }
        )

    test_df.to_csv("mf_test_preds.csv", index=False, float_format='%.6f')

    if args.export_train_and_val:
        export_train_and_val(args, train_data, val_data, train_scaler)


def export_and_plot_hf_lf_data(data_df, args):

    data_df.to_csv("lf_hf_targets.csv", float_format='%.6f')

    lf = data_df[args.lf_col_name].values
    hf = data_df[args.hf_col_name].values

    plt.scatter(hf, lf, alpha=0.3)
    plt.xlabel("High Fidelity")
    plt.ylabel("Low Fidelity")

    title = ""
    if args.add_descriptor_bias_to_make_lf > 0:
        title += f"Descriptor ({args.add_descriptor_bias_to_make_lf}); "
    if args.add_pn_bias_to_make_lf > 0:
        title += f"Poly ({args.add_pn_bias_to_make_lf}); "
    if args.add_constant_bias_to_make_lf > 0:
        title += f"Constant ({args.add_constant_bias_to_make_lf}); "
    if args.add_gauss_noise_to_make_lf > 0:
        title += f"Gaussian ({args.add_gauss_noise_to_make_lf}); "

    if title.endswith("; "):
        title = title[:-2]

    plt.title(title)
    plt.savefig("lf_vs_hf_targets.png")

    return


def export_train_and_val(args, train_data, val_data, train_scaler):
    train_smis = [x.smi for x in train_data]
    val_smis = [x.smi for x in val_data]

    if args.model_type == "single_fidelity":
        train_targets = np.array([x.targets[0] for x in train_data])
        val_targets = np.array([x.targets[0] for x in val_data])
    else:
        train_targets = np.array([x.targets for x in train_data])
        val_targets = np.array([x.targets for x in val_data])

    if args.scale_data:
        if args.model_type == "single_fidelity":
            train_targets = train_scaler.inverse_transform(train_targets.reshape(-1, 1))
            val_targets = train_scaler.inverse_transform(val_targets.reshape(-1, 1))
        else:
            train_targets = train_scaler.inverse_transform(train_targets)
            val_targets = train_scaler.inverse_transform(val_targets)

    train_dict = {"smiles": train_smis}
    val_dict = {"smiles": val_smis}
    if args.model_type == "single_fidelity":
        train_dict[args.hf_col_name] = train_targets.flatten()
        val_dict[args.hf_col_name] = val_targets.flatten()
    else:
        train_dict[args.lf_col_name] = train_targets[:, 0].flatten()
        val_dict[args.lf_col_name] = val_targets[:, 0].flatten()
        train_dict[args.hf_col_name] = train_targets[:, 1].flatten()
        val_dict[args.hf_col_name] = val_targets[:, 1].flatten()

    train_df = pd.DataFrame(train_dict)
    val_df = pd.DataFrame(val_dict)
    train_df.to_csv("mf_train.csv", index=False, float_format='%.6f')
    val_df.to_csv("mf_val.csv", index=False, float_format='%.6f')

    return


def eval_metrics(targets, preds):
    mae = mean_absolute_error(targets, preds)
    rmse = mean_squared_error(targets, preds, squared=False)
    r2 = r2_score(targets, preds)
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"R2: {r2}")
    return mae, rmse, r2


# use this approach instead of action="store_true" to be compatible with LLMapReduce on supercloud
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def add_args(parser: ArgumentParser):
    parser.add_argument(
        "--model_type",
        type=str,
        default="single_fidelity",
        choices=[
            "single_fidelity",
            "multi_target",
            "multi_fidelity",
            "multi_fidelity_weight_sharing",
            "multi_fidelity_non_diff",
            "delta_ml",
            "trad_delta_ml"
        ],
    )
    parser.add_argument("--data_file", type=str, default="/home/temujin/chemprop-mf/tests/data/gdb11_0.001.csv")
    # choices=["multifidelity_joung_stda_tddft.csv", "gdb11_0.0001.csv" (too small), "gdb11_0.0001.csv"]
    parser.add_argument("--hf_col_name", type=str, default="h298")  # choices=["h298", "lambda_maxosc_tddft"]
    parser.add_argument("--lf_col_name", type=str, default="h298_lf", required=False)  # choices=["h298_bias_1", "lambda_maxosc_stda"]
    parser.add_argument("--scale_data", type=str2bool, default=False)
    parser.add_argument("--save_test_plot", type=str2bool, default=False)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--export_train_and_val", type=str2bool, default=False)
    parser.add_argument("--add_pn_bias_to_make_lf", type=int, default=0, help="order of the polynomial in x, for order >= 1")
    parser.add_argument("--add_constant_bias_to_make_lf", type=float, default=0.0, help="use this instead of `--add_pn_bias_to_make_lf` for order = 0")
    parser.add_argument("--add_gauss_noise_to_make_lf", type=float, default=0.0)
    parser.add_argument("--add_descriptor_bias_to_make_lf", type=float, default=0.0, help="descriptor weights range from -N to N")
    # TODO: add atom bias?
    parser.add_argument("--split_type", type=str, default="random", choices=["random", "scaffold", "h298", "molwt", "atom"])
    parser.add_argument("--lf_hf_size_ratio", type=int, default=1)  # <N> : 1 = LF : HF
    parser.add_argument("--lf_superset_of_hf", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results_dir", type=str, default="results")
    return


if __name__ == "__main__":
    main()
