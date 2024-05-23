import json
import os
import random
from argparse import ArgumentParser, ArgumentTypeError
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from chemprop.v2 import data, featurizers, models
from chemprop.v2.models import modules
from noisefunctions import descriptor_bias, gauss_noise
from splitfunctions import split_by_prop_dict
from kernel_helper import kernel_end2end


def main():
    parser = ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    seed_everything(args.seed)
    np.random.seed(args.seed)

    if args.model_type == "single_fidelity" and (
        args.add_pn_bias_to_make_lf > 0
        or args.add_gauss_noise_to_make_lf > 0
        or args.add_descriptor_bias_to_make_lf > 0
    ):
        raise ValueError(
            "Cannot add bias to make low fidelity data when model type is single fidelity"
        )

    if args.model_type == "multi_fidelity_weight_sharing_non_diff":
        raise NotImplementedError("Not implemented yet")

    # make unique folder for results of each run
    os.makedirs(args.results_dir, exist_ok=True)
    os.chdir(args.results_dir)
    now = datetime.now()
    datetime_string = now.strftime("%Y-%m-%d_%H-%M-%S.%f")
    os.mkdir(datetime_string)
    os.chdir(datetime_string)

    # Output args to a file
    args_dict = vars(args)
    with open("args.json", "w") as f:
        json.dump(args_dict, f, indent=4)

    data_df = pd.read_csv(args.data_file).drop_duplicates(subset=["smiles"])
    data_df.set_index("smiles", inplace=True)

    if args.model_type != "kernel":
        mpnn = choose_model(args.model_type, args.loss_mod)

    # Train-Val-Test Splits
    # Val and test are the same for LF and HF; train is separated later
    # TODO: (!) scaffold splits appear to not be reproducible???
    # TODO: (!) add in **kwargs from dicts for non-random split types
    train_val_smiles, test_smiles = split_by_prop_dict[args.split_type](df=data_df, seed=args.seed)
    train_val_df = data_df.loc[train_val_smiles]
    train_smiles, val_smiles = split_by_prop_dict["random"](df=train_val_df, seed=args.seed)

    if data_df.shape[1] == 1:  # This means if no LF column exists
        data_df = create_low_fidelity(data_df, args)

    print("Total dataset size:", len(data_df))

    if args.model_type == "single_fidelity":
        target_col_names = [args.hf_col_name]

        # Print sizes of datasets and splits
        print("Train size:", len(train_smiles))
        print("Val size:", len(val_smiles))
        print("Test size:", len(test_smiles))

    elif args.model_type == "delta_ml":
        # HF is target and LF is oracle
        target_col_names = [args.hf_col_name]
        oracle_col_names = [args.lf_col_name]
        # No need to seperate into HF and LF
        print("HF/Oracle train size:", len(train_smiles))
        print("HF/Oracle val size:", len(val_smiles))
        print("HF/Oracle test size:", len(test_smiles))

    elif args.model_type in ["trad_delta_ml", "kernel"]:
        # HF is target and LF is oracle
        data_df["delta"] = data_df[args.hf_col_name] - data_df[args.lf_col_name]
        #data_df["delta"] = data_df["delta"].apply(lambda x: [x])
        target_col_names = "delta"

        print("HF/Oracle train size:", len(train_smiles))
        print("HF/Oracle val size:", len(val_smiles))
        print("HF/Oracle test size:", len(test_smiles))

    else:

        data_df, lf_train_smiles, hf_train_smiles = separate_hf_and_lf_train(
            data_df, train_smiles, args
        )

        if args.model_type == "transfer":
            # To remove some redundancy, "train_targets, val_targets, test_targets" are "lf_train_targets, lf_val_targets, lf_test_targets"
            target_col_names = [args.lf_col_name]
            # Can seperate using the same function.
            data_df, lf_val_smiles, hf_val_smiles = separate_hf_and_lf_train(
                data_df, val_smiles, args
            )
            data_df, lf_test_smiles, hf_test_smiles = separate_hf_and_lf_train(
                data_df, test_smiles, args
            )

            print("HF train size:", len(hf_train_smiles))
            print("LF train size:", len(lf_train_smiles))
            print("HF val size:", len(hf_val_smiles))
            print("LF val size:", len(lf_val_smiles))
            print("HF test size:", len(hf_test_smiles))
            print("LF test size:", len(lf_test_smiles))

        else:
            # LF column must be first, HF second to work with expected order in loss function during training
            target_col_names = [args.lf_col_name, args.hf_col_name]
            export_and_plot_hf_lf_data(data_df, args)
            train_smiles = list(set(list(lf_train_smiles) + list(hf_train_smiles)))

            # Print sizes of datasets and splits
            print("HF train size:", len(hf_train_smiles))
            print("LF train size:", len(lf_train_smiles))
            print("HF/LF val size:", len(val_smiles))
            print("HF/LF test size:", len(test_smiles))

    # Selecting the target values for train, val, and test
    train_targets = data_df.loc[train_smiles][target_col_names].values
    val_targets = data_df.loc[val_smiles][target_col_names].values
    test_targets = data_df.loc[test_smiles][target_col_names].values

    if args.model_type == "kernel":
        #For Kernel Ridge Regression
        train_lf_targets = data_df.loc[train_smiles][args.lf_col_name].values
        test_hf_targets = data_df.loc[test_smiles][args.hf_col_name].values
        test_lf_targets = data_df.loc[test_smiles][args.lf_col_name].values

    if args.model_type == "transfer":
        # Regular train, val, test targets are the lf versions in the case of transfer learning
        hf_train_targets = data_df.loc[hf_train_smiles][[args.hf_col_name]].values
        hf_test_targets = data_df.loc[hf_test_smiles][[args.hf_col_name]].values
        hf_val_targets = data_df.loc[hf_val_smiles][[args.hf_col_name]].values

    if args.model_type == "delta_ml":
        # Fetching LF as oracles
        train_oracles = data_df.loc[train_smiles][oracle_col_names].values
        val_oracles = data_df.loc[val_smiles][oracle_col_names].values
        test_oracles = data_df.loc[test_smiles][oracle_col_names].values

        # Initializing the data
        train_data = [
            data.MoleculeDatapoint(smi, t, features=o)
            for smi, t, o in zip(train_smiles, train_targets, train_oracles)
        ]
        val_data = [
            data.MoleculeDatapoint(smi, t, features=o)
            for smi, t, o in zip(val_smiles, val_targets, val_oracles)
        ]
        test_data = [
            data.MoleculeDatapoint(smi, t, features=o)
            for smi, t, o in zip(test_smiles, test_targets, test_oracles)
        ]

    else:
        # Initializing the data
        train_data = [data.MoleculeDatapoint(smi, t) for smi, t in zip(train_smiles, train_targets)]
        val_data = [data.MoleculeDatapoint(smi, t) for smi, t in zip(val_smiles, val_targets)]
        test_data = [data.MoleculeDatapoint(smi, t) for smi, t in zip(test_smiles, test_targets)]

        if args.model_type == "transfer":

            # Redundancy removed if hf_train_data = train_data
            hf_train_data = [
                data.MoleculeDatapoint(smi, t) for smi, t in zip(hf_train_smiles, hf_train_targets)
            ]
            hf_val_data = [
                data.MoleculeDatapoint(smi, t) for smi, t in zip(hf_val_smiles, hf_val_targets)
            ]
            hf_test_data = [
                data.MoleculeDatapoint(smi, t) for smi, t in zip(hf_test_smiles, hf_test_targets)
            ]

    mgf = featurizers.MoleculeFeaturizer()

    train_dset = data.MoleculeDataset(train_data, mgf)
    val_dset = data.MoleculeDataset(val_data, mgf)
    test_dset = data.MoleculeDataset(test_data, mgf)

    if args.scale_data:
        train_scaler = train_dset.normalize_targets()
        _ = val_dset.normalize_targets(train_scaler)
        _ = test_dset.normalize_targets(train_scaler)

    train_loader = data.MolGraphDataLoader(train_dset, batch_size=50, num_workers=12)
    val_loader = data.MolGraphDataLoader(val_dset, batch_size=50, num_workers=12, shuffle=False)
    test_loader = data.MolGraphDataLoader(test_dset, batch_size=50, num_workers=12, shuffle=False)

    if args.model_type == "transfer":

        hf_train_dset = data.MoleculeDataset(hf_train_data, mgf)
        hf_val_dset = data.MoleculeDataset(hf_val_data, mgf)
        hf_test_dset = data.MoleculeDataset(hf_test_data, mgf)

        if args.scale_data:
            hf_train_scaler = hf_train_dset.normalize_targets()
            _ = hf_val_dset.normalize_targets(hf_train_scaler)
            _ = hf_test_dset.normalize_targets(hf_train_scaler)

        hf_train_loader = data.MolGraphDataLoader(hf_train_dset, batch_size=50, num_workers=12)
        hf_val_loader = data.MolGraphDataLoader(
            hf_val_dset, batch_size=50, num_workers=12, shuffle=False
        )
        hf_test_loader = data.MolGraphDataLoader(
            hf_test_dset, batch_size=50, num_workers=12, shuffle=False
        )

    # Train
    if args.model_type != "kernel":
        trainer = pl.Trainer(
            logger=True,
            enable_checkpointing=False,
            enable_progress_bar=True,
            accelerator="gpu",
            devices=1,
            max_epochs=args.num_epochs,
        )

        trainer.fit(mpnn, train_loader, val_loader)  # TODO: (!) matmul dimension error for delta_ml
        preds = trainer.predict(mpnn, test_loader)
        test_smis = [x.smi for x in test_data]
    else:
        preds = kernel_end2end(train_smiles, test_smiles, train_lf_targets, train_targets)
        test_smis = test_smiles


    if args.model_type == "transfer":

        trainer.fit(mpnn, hf_train_loader, hf_val_loader)
        hf_preds = trainer.predict(mpnn, hf_test_loader)
        hf_test_smis = [x.smi for x in hf_test_data]

    if args.model_type in ["single_fidelity", "transfer", "delta_ml", "trad_delta_ml", "transfer"]:
        preds = [x[0].item() for x in preds]
        targets = [x.targets[0] for x in test_data]

        if args.scale_data:
            preds = train_scaler.inverse_transform(np.array(preds).reshape(-1, 1))
            targets = train_scaler.inverse_transform(np.array(targets).reshape(-1, 1))
        else:
            preds = np.array(preds)
            targets = np.array(targets)

        if args.model_type == "transfer":

            # only keep targets, smiles, and preds that are not NaN
            preds = preds[~np.isnan(targets)]
            test_smis = [x for x, y in zip(test_smis, targets) if not np.isnan(y)]
            targets = targets[~np.isnan(targets)]

            hf_preds = [x[0].item() for x in hf_preds]
            hf_targets = [x.targets[0] for x in hf_test_data]

            if args.scale_data:
                hf_preds = hf_train_scaler.inverse_transform(np.array(hf_preds).reshape(-1, 1))
                hf_targets = hf_train_scaler.inverse_transform(np.array(hf_targets).reshape(-1, 1))
            else:
                hf_preds = np.array(hf_preds)
                hf_targets = np.array(hf_targets)

        print("Test set")
        mae, rmse, r2 = eval_metrics(targets, preds)

        if args.model_type == "transfer":
            hf_mae, hf_rmse, hf_r2 = eval_metrics(hf_targets, hf_preds)

            metrics_df = pd.DataFrame(
                {
                    "MAE_hf": [hf_mae],
                    "RMSE_hf": [hf_rmse],
                    "R2_hf": [hf_r2],
                    "MAE_lf": [mae],
                    "RMSE_lf": [rmse],
                    "R2_lf": [r2],
                }
            )
        else:
            metrics_df = pd.DataFrame(
                {
                    "MAE_hf": [mae],
                    "RMSE_hf": [rmse],
                    "R2_hf": [r2],
                    "MAE_lf": [np.nan],
                    "RMSE_lf": [np.nan],
                    "R2_lf": [np.nan],
                }
            )

        metrics_df.to_csv("test_metrics.csv", index=False)

        if args.save_test_plot:
            plt.scatter(targets, preds)
            plt.xlabel("Target")
            plt.ylabel("Prediction")
            plt.text(
                min(targets),
                max(preds),
                f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR^2: {r2:.2f}",
                fontsize=12,
                ha="left",
                va="top",
            )
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
        elif (
            args.model_type == "multi_fidelity"
            or args.model_type == "multi_fidelity_weight_sharing"
        ):  # TODO: (!) will this also work for multi-fidelity non-differentiable?
            preds = np.array([[[x[0][0].numpy(), x[0][1].numpy()]] for x in preds]).reshape(
                len(preds), 2
            )
        elif args.model_type in ["evidentialmf", "evidentialdual", "mvemultifidelity"]:
            preds = np.array(
                [[[x[0][0][0][0].numpy(), x[0][1][0][0].numpy()]] for x in preds]
            ).reshape(len(preds), 2)

        # Both HF and LF targets are identical if the only difference in the original HF and LF was a bias term -- this is not a bug -- once normalized, the network should learn both the same way
        if args.model_type != "kernel":
            targets = np.array([x.targets for x in test_data])

            if args.scale_data:
                preds = train_scaler.inverse_transform(preds)
                targets = train_scaler.inverse_transform(targets)

            targets_lf, targets_hf, preds_lf, preds_hf = [], [], [], []

            for target, pred in zip(targets, preds):
                # LF should be hf
                if not np.isnan(target[0]):
                    targets_lf.append(target[0])
                    preds_lf.append(pred[1])
                # HF should be lf
                if not np.isnan(target[1]):
                    targets_hf.append(target[1])
                    preds_hf.append(pred[0])

            targets_hf = np.array(targets_hf)
            targets_lf = np.array(targets_lf)
            preds_hf = np.array(preds_hf)
            preds_lf = np.array(preds_lf)

        else:
            targets_hf = np.array(test_hf_targets)
            targets_lf = np.array(test_lf_targets)
            preds_hf = np.array(preds[0])
            preds_lf = np.array(preds[1])

            #This is so things don't break in line 444-446
            targets = np.array([[i,j] for i,j in zip(targets_lf,targets_hf)])
            preds = np.array([[i,j] for i,j in zip(preds_lf,preds_hf)])

        print("High Fidelity - Test set")
        hf_mae, hf_rmse, hf_r2 = eval_metrics(targets_hf, preds_hf)
        print("Low Fidelity - Test set")
        lf_mae, lf_rmse, lf_r2 = eval_metrics(targets_lf, preds_lf)
        metrics_df = pd.DataFrame(
            {
                "MAE_hf": [hf_mae],
                "RMSE_hf": [hf_rmse],
                "R2_hf": [hf_r2],
                "MAE_lf": [lf_mae],
                "RMSE_lf": [lf_rmse],
                "R2_lf": [lf_r2],
            }
        )
        metrics_df.to_csv("test_metrics.csv", index=False)

        if args.save_test_plot:
            fig, axes = plt.subplots(figsize=(6, 3), nrows=1, ncols=2)

            axes[0].scatter(targets_hf, preds_hf, alpha=0.3, label="High Fidelity")
            axes[0].set_xlabel("Target")
            axes[0].set_ylabel("Prediction")
            axes[0].text(
                min(targets_hf),
                max(preds_hf),
                f"MAE: {hf_mae:.2f}\nRMSE: {hf_rmse:.2f}\nR^2: {hf_r2:.2f}",
                fontsize=12,
                ha="left",
                va="top",
            )
            axes[0].set_title("High Fidelity")

            axes[1].scatter(targets_lf, preds_lf, alpha=0.3, label="Low Fidelity")
            axes[1].set_xlabel("Target")
            axes[1].set_ylabel("Prediction")
            axes[1].text(
                min(targets_lf),
                max(preds_lf),
                f"MAE: {lf_mae:.2f}\nRMSE: {lf_rmse:.2f}\nR^2: {lf_r2:.2f}",
                fontsize=12,
                ha="left",
                va="top",
            )
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

    test_df.to_csv("mf_test_preds.csv", index=False, float_format="%.6f")

    if args.export_train_and_val:
        export_train_and_val(args, train_data, val_data, train_scaler)


def choose_model(model_type, loss_mod=1):
    mp_block_hf = (
        modules.molecule_block()
    )  # TODO: use aggregation='sum' or 'norm' instead of default 'mean'? (also on line below)
    mp_block_lf = modules.molecule_block()

    model_dict = {
        "single_fidelity": models.RegressionMPNN(mp_block_hf, n_tasks=1),
        "multi_target": models.RegressionMPNN(mp_block_hf, n_tasks=2),
        "multi_fidelity": models.MultifidelityRegressionMPNN(
            mp_block_hf, n_tasks=1, mpn_block_low_fidelity=mp_block_lf, loss_mod=loss_mod
        ),
        "multi_fidelity_weight_sharing": models.MultifidelityRegressionMPNN(mp_block_hf, n_tasks=1),
        "delta_ml": models.RegressionMPNN(mp_block_hf, n_tasks=1, n_features=1),
        "trad_delta_ml": models.RegressionMPNN(mp_block_hf, n_tasks=1),
        "transfer": models.RegressionMPNN(mp_block_hf, n_tasks=1),
        "evidentialmf": models.EvidentialMultifidelityMPNN(
            mp_block_hf, n_tasks=1, mpn_block_low_fidelity=mp_block_lf, loss_mod=loss_mod
        ),
        "evidentialdual": models.EvidentialDualMF(
            mp_block_hf, n_tasks=4, mpn_block_low_fidelity=mp_block_lf
        ),
        "mvemultifidelity": models.MveMultifidelity(
            mp_block_hf, n_tasks=2, mpn_block_low_fidelity=mp_block_lf, loss_mod=loss_mod
        ),
        # "multi_fidelity_weight_sharing_non_diff": ,  # TODO: (!) multi-fidelity non-differentiable feature
    }
    # TODO: add method: multi-fidelity with evidential uncertainty?
    # TODO: add methods: transfer learning, deltaML, Bayesian methods, etc.

    return model_dict[model_type]


def create_low_fidelity(data_df, args):
    
    if args.lf_col_name not in data_df.columns:
        data_df[args.lf_col_name] = data_df[args.hf_col_name]

    if args.add_pn_bias_to_make_lf > 0:
        # Creating the coefficients for the polynomial function
        coefficients = np.random.uniform(
            -1, 1, args.add_pn_bias_to_make_lf + 1
        )  # need to add 1 because the one coefficient is for x^0
        # Adding bias calculated from the polynomial function of HF to data_df LF column
        data_df[args.lf_col_name] = data_df[args.lf_col_name] + np.polyval(
            coefficients, list(data_df[args.hf_col_name])
        )

    if args.add_constant_bias_to_make_lf != 0.0:
        # Add the constant bias to HF to make data_df LF column
        data_df[args.lf_col_name] = data_df[args.lf_col_name] + args.add_constant_bias_to_make_lf

    if args.add_gauss_noise_to_make_lf > 0.0:
        data_df[args.lf_col_name] = data_df[args.lf_col_name] + gauss_noise(
            df_len=len(data_df[args.lf_col_name]),
            std=args.add_gauss_noise_to_make_lf,
            seed=args.seed,
        )

    if args.add_descriptor_bias_to_make_lf != 0.0:
        descriptors = [
            Descriptors.qed,
            Descriptors.MolWt,
            Descriptors.BalabanJ,
            Descriptors.BertzCT,
            Descriptors.HallKierAlpha,
            Descriptors.Ipc,
            Descriptors.Kappa1,
            Descriptors.Kappa2,
            Descriptors.Kappa3,
            Descriptors.LabuteASA,
            Descriptors.TPSA,
            Descriptors.MolLogP,
            Descriptors.MolMR,
        ]
        # Creating the weight descriptor pair
        coefficients = (
            np.random.uniform(-1, 1, len(descriptors)) * args.add_descriptor_bias_to_make_lf
        )
        descriptors_coefficients = list(zip(descriptors, coefficients))
        # Adding bias calculated from normalized descriptors to data_df LF column
        data_df[args.lf_col_name] = data_df[args.lf_col_name] + descriptor_bias(
            data_df, descriptors_coefficients
        )

    return data_df


def export_and_plot_hf_lf_data(data_df, args):

    data_df.to_csv("lf_hf_targets.csv", float_format="%.6f")

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


def separate_hf_and_lf_train(data_df, train_smiles, args):
    train_df = data_df.loc[train_smiles]

    if args.lf_superset_of_hf:
        hf_frac = 1 / args.lf_hf_size_ratio
        hf_train_df = train_df.copy().sample(frac=hf_frac, random_state=args.seed)
        lf_train_df = train_df.copy()
    else:
        hf_frac = 1 / (args.lf_hf_size_ratio + 1)
        hf_train_df = train_df.copy().sample(frac=hf_frac, random_state=args.seed)
        lf_train_df = train_df.copy().drop(index=hf_train_df.index)

    # Creating a list of train and test indexes
    hf_train_df.drop(args.lf_col_name, inplace=True, axis=1)
    lf_train_df.drop(args.hf_col_name, inplace=True, axis=1)

    lf_train_smiles = lf_train_df.index
    hf_train_smiles = hf_train_df.index

    # Setting nan to specify LF and HF
    lf_not_hf_train_smiles = list(set(lf_train_smiles).difference(set(hf_train_smiles)))
    data_df[args.hf_col_name].loc[lf_not_hf_train_smiles] = np.nan
    if not args.lf_superset_of_hf:
        data_df[args.lf_col_name].loc[hf_train_df.index] = np.nan

    return data_df, lf_train_smiles, hf_train_smiles


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
    train_df.to_csv("mf_train.csv", index=False, float_format="%.6f")
    val_df.to_csv("mf_val.csv", index=False, float_format="%.6f")

    return

def eval_metrics(targets, preds):
    valid_indices = ~np.isnan(targets)
    targets = targets[valid_indices]
    preds = preds[valid_indices]

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
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")


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
            "trad_delta_ml",
            "transfer",
            "evidentialmf",
            "evidentialdual",
            "mvemultifidelity",
            "kernel"
        ],
    )
    parser.add_argument(
        "--data_file", type=str, default="tests/data/gdb11_0.001.csv"
    )
    # choices=["multifidelity_joung_stda_tddft.csv", "gdb11_0.0001.csv" (too small), "gdb11_0.0001.csv", "dHsolv.csv", "pubchem.csv", "lambda.csv"]
    parser.add_argument(
        "--hf_col_name", type=str, default="h298"
    )  # choices=["h298", "lambda_maxosc_tddft", dHsolv_expt, "energy_max_osc_nm"]
    parser.add_argument(
        "--lf_col_name", type=str, default="h298_bias_1", required=False
    )  # choices=["h298_bias_1", "lambda_maxosc_stda", "dHsolv_cosmo", "peakwavs_max"]
    parser.add_argument("--scale_data", type=str2bool, default=False)
    parser.add_argument("--save_test_plot", type=str2bool, default=False)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--export_train_and_val", type=str2bool, default=False)
    parser.add_argument(
        "--add_pn_bias_to_make_lf",
        type=int,
        default=0,
        help="order of the polynomial in x, for order >= 1",
    )
    parser.add_argument(
        "--add_constant_bias_to_make_lf",
        type=float,
        default=0.0,
        help="use this instead of `--add_pn_bias_to_make_lf` for order = 0",
    )
    parser.add_argument("--add_gauss_noise_to_make_lf", type=float, default=0.0)
    parser.add_argument(
        "--add_descriptor_bias_to_make_lf",
        type=float,
        default=0.0,
        help="descriptor weights range from -N to N",
    )
    # TODO: add atom bias?
    # TODO: (!) add arguments for h298, molwt, atom splits
    parser.add_argument(
        "--split_type",
        type=str,
        default="random",
        choices=["random", "scaffold", "h298", "molwt", "atom"],
    )
    parser.add_argument("--lf_hf_size_ratio", type=int, default=1)  # <N> : 1 = LF : HF
    parser.add_argument("--lf_superset_of_hf", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--loss_mod", type=float, default=1)
    return


if __name__ == "__main__":
    main()
