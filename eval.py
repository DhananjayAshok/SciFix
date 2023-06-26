import numpy as np

import data
import utils
from data import data_root
from prompting import VerificationAlgorithm
from query_models import BARTSeq2Seq, T5Seq2Seq, BARTClassification, T5Classification, PenalizeSimilarity, \
    PenalizeExplanationSimilarity
import os
from tqdm import tqdm
import pandas as pd
import warnings
from evaluate import load

model_path = "models"
results_path = "results"
save_path = "scoring"
turk_path = "turk_output"
turk_processed_path = "results"

utils.safe_mkdir(save_path)
utils.safe_mkdir(turk_processed_path)

val_files = ["bart_claim_correction_bal_val.csv", "bart_claim_correction_split_bal_val.csv",
             "bart_claim_correction_exp_bal_val.csv", "bart_claim_correction_exp_split_bal_val.csv",
             "t5_claim_correction_exp_bal_val.csv", "t5_claim_correction_exp_split_bal_val.csv",
             "correction_gpt_short_bal_test_VENCE.csv",
             "GPT_correction_mode_0_benchmark.csv", "GPT_correction_mode_1_benchmark.csv"
             ]

final_files = ["bart_claim_correction_bal_final.csv", "bart_claim_correction_split_bal_final.csv",
               "bart_claim_correction_exp_bal_final.csv", "bart_claim_correction_exp_split_bal_final.csv",
                "t5_claim_correction_exp_bal_final.csv", "t5_claim_correction_exp_split_bal_final.csv",
               "final_test_formatted_VENCE.csv",
               "GPT_correction_mode_0_benchmark_final.csv", "GPT_correction_mode_1_benchmark_final.csv"
               ]


def generate_classification_preds(model, df=None, test_file=None):
    assert df is not None or test_file is not None
    if df is None:
        df = pd.read_csv(test_file)
    df["pred"] = None
    if "text_column" in df.columns:
        df["label"] = df["summary_column"].apply(lambda x: x[2:]) == "True"  # get just the true or false
    for i in tqdm(df.index, total=len(df)):
        if "text_column" in df.columns:  # Then is seq format
            inp = df.loc[i, "text_column"]
            pred = model.tf_prob(inp) >= 0.5
        else:
            inp = df.loc[i, "text"]
            pred = model(inp)
        df.loc[i, "pred"] = pred
    return df


def generate_seq_preds(model, df=None, test_file=None, mode=None, limit=500):
    assert df is not None or test_file is not None
    if df is None:
        df = pd.read_csv(test_file)
    if limit is not None and limit < len(df):
        df = df.sample(limit)
    mode_str = f"_{mode}" if mode is not None else ""
    if mode == "score":
        mode = PenalizeSimilarity()
    if mode == "exp_score":
        mode = PenalizeExplanationSimilarity()
        mode_str = "score"
    df[f"pred{mode_str}"] = None
    for i in tqdm(df.index, total=len(df)):
        inp = df.loc[i, "text_column"]
        if isinstance(inp, float):
            pred = "NaN"
        else:
            pred = model(inp, mode=mode)
        df.loc[i, f"pred{mode_str}"] = pred
    df[f"pred{mode_str}"] = df[f"pred{mode_str}"].apply(lambda x: x.strip())
    if "summary_column" not in df.columns:
        df["summary_column"] = ""
    df["summary_column"] = df["summary_column"].apply(lambda x: x.strip())
    return df


def get_preds(model_name, test_file):
    pred_file = f"{model_path}/{model_name}/generated_predictions.txt"
    assert os.path.exists(test_file)
    df = pd.read_csv(test_file)
    if not os.path.exists(pred_file):
        warnings.warn("No Predfile: Maybe you have not run the predict_script")
    else:
        with open(pred_file, 'r') as f:
            lines = f.readlines()
        df = df[:len(lines)]
        df.loc[:, "pred"] = lines
        df = df[~df['pred'].isna() & ~df['summary_column'].isna()]
        df["pred"] = df["pred"].apply(lambda x: x.strip())
    df["summary_column"] = df["summary_column"].apply(lambda x: x.strip())
    return df


def equality_score(df, col1="summary_column", col2="pred"):
    df["equality"] = (df[col1].apply(lambda x: x.strip()) == df[col2].apply(lambda x: x.strip())).apply(int)
    return


def inequality_score(df, col1="summary_column", col2="pred"):
    df["inequality"] = (df[col1] != df[col2]).apply(int)
    return


def get_correction(bart=True, explanation=False, balanced=True, split=False):
    model_name = f"{'bart' if bart else 't5'}_claim_correction{'_exp' if explanation else ''}" \
                 f"{'_split' if split else ''}{'_bal' if balanced else ''}"
    test_file = f"{data_root}/correction_gpt_short{'_exp' if explanation else ''}{'_bal' if balanced else ''}_test.csv"
    df = get_preds(model_name=model_name, test_file=test_file)
    if bart:
        model = BARTSeq2Seq(path=f"{model_path}/{model_name}")
    else:
        model = T5Seq2Seq(path=f"{model_path}/{model_name}")
    return df, model, test_file


def get_verification(bart=True, seq=True):
    model_name = f"{'bart' if bart else 't5'}_verification{'_seq' if seq else ''}"
    test_file = f"{data_root}/verification{'_seq' if seq else ''}_test.csv"
    if seq:
        if bart:
            model = BARTSeq2Seq(path=f"{model_path}/{model_name}")
        else:
            model = T5Seq2Seq(path=f"{model_path}/{model_name}")
    else:
        if bart:
            model = BARTClassification(path=f"{model_path}/{model_name}")
        else:
            model = T5Classification(path=f"{model_path}/{model_name}")
    df = generate_classification_preds(model, test_file=test_file)
    accuracy = (df["label"] == df["pred"]).astype(int).mean()
    print(f"Accuracy: {accuracy}")
    return df, model, test_file


def get_difference(bart=True, seq=True):
    model_name = f"{'bart' if bart else 't5'}_difference{'_seq' if seq else ''}"
    test_file = f"{data_root}/difference{'_seq' if seq else ''}_test.csv"
    if seq:
        if bart:
            model = BARTSeq2Seq(path=f"{model_path}/{model_name}")
        else:
            model = T5Seq2Seq(path=f"{model_path}/{model_name}")
    else:
        if bart:
            model = BARTClassification(path=f"{model_path}/{model_name}")
        else:
            model = T5Classification(path=f"{model_path}/{model_name}")
    #df = generate_classification_preds(model, test_file=test_file)
    df = pd.read_csv(test_file)
    #accuracy = (df["label"] == df["pred"]).astype(int).mean()
    #print(f"Accuracy: {accuracy}")
    return df, model, test_file


def save_correction_preds(bart=True, explanation=False, balanced=True, split=False):
    model_name = f"{'bart' if bart else 't5'}_claim_correction{'_exp' if explanation else ''}" \
                 f"{'_split' if split else ''}{'_bal' if balanced else ''}"
    df, model, test_file = get_correction(bart=bart, explanation=explanation, balanced=balanced, split=split)
    model.to(utils.Parameters.devices[0])
    df = generate_seq_preds(model, test_file=test_file, mode="beam")
    df = generate_seq_preds(model, df=df, mode="score" if not explanation else "exp_score")
    df.to_csv(f"{results_path}/{model_name}_val.csv", index=False)

    if split:
        test_file = f"{data_root}/final_test_split_formatted.csv"
    else:
        test_file = f"{data_root}/final_test_formatted.csv"
    df = generate_seq_preds(model, test_file=test_file, mode="beam")
    df.to_csv(f"{results_path}/{model_name}_final.csv", index=False)
    df = generate_seq_preds(model, df=df, mode="score" if not explanation else "exp_score")
    df.to_csv(f"{results_path}/{model_name}_final.csv", index=False)


def save_correction():
    for bart in [True, False]:
        for explanation in [False, True]:
            for split in False, True:
                save_correction_preds(bart=bart, explanation=explanation, split=split)


def scroll(df, exclude=[], scoring=False, score_col="score"):
    iterator = df.index
    if scoring:
        if score_col not in df:
            df[score_col] = None
        na = df[score_col].isna()
    n = len(df)
    for i in iterator:
        if scoring:
            if not na[i]:
                continue
        print(f"Row: {i}/{n}")
        for col in df:
            if col in exclude:
                pass
            else:
                print(f"{col}: {df.loc[i, col]}")
                print(f"{'X'*10}")
        if not scoring:
            cont = input(f"Hit Enter to continue. Any other input breaks out")
            if cont.strip() == "":
                break
        else:
            cont = ""
            while cont.strip() == "":
                cont = input(f"Give a score (1: correct, 0: incorrect, -1: unclear) "
                             f"for the current example. To exit the scoring loop enter any other "
                             f"(non whitespace) input")
            cont = cont.strip()
            if cont == "0":
                df.loc[i, score_col] = 0
            elif cont == "1":
                df.loc[i, score_col] = 1
            elif cont == "-1":
                df.loc[i, score_col] = -1
            else:
                print(f"Breaking out of scoring loop...")
                break
    return


def score_metric(df, metric_class, metric_name, save_name, limit=500):
    metric = metric_class()
    df = df[~df["text_column"].isna()]
    if len(df) > limit:
        df = df[~df["pred"].isna()]
        df = df.sample(n=limit)
    iterator = df.index
    if "predscore" in df.columns:
        df["pred_score"] = df["predscore"]
        df = df.drop("predscore", axis=1)
    if "pred_score" in df.columns:
        df[f"{metric_name}_beam"] = None
        df[f"{metric_name}_score"] = None
    elif "pred_beam" in df.columns:
        df[f"{metric_name}_beam"] = None
    else:
        df[metric_name] = None
    for i in tqdm(iterator):
        row = df.loc[i]
        evidence_claim = row["text_column"]
        evidence, incorrect_claim = evidence_claim.split("Claim:")
        correct_claim = None
        if "summary_column" in df:
            if df["summary_column"].isna()[i]:
                pass
            else:
                correct_claim = df.loc[i, "summary_column"]
        if "pred_score" in df.columns:
            pred = row["pred_score"]
            df.loc[i, f"{metric_name}_score"] = metric(evidence, incorrect_claim, pred, correct_claim)
        if "pred_beam" in df.columns:
            pred = row["pred_beam"]
            df.loc[i, f"{metric_name}_beam"] = metric(evidence, incorrect_claim, pred, correct_claim)
        else:
            pred = row["pred"]
            df.loc[i, f"{metric_name}"] = metric(evidence, incorrect_claim, pred, correct_claim)
    df.to_csv(f"{save_path}/{save_name}", index=False)
    return df


class Metric:
    def __call__(self, evidence, incorrect_claim, pred, correct_claim, separator="|"):
        if separator in pred:
            pred = pred.split(separator)[1]
        if correct_claim is not None:
            if separator in correct_claim:
                correct_claim = correct_claim.split(separator)[1]
        return self.score(evidence, incorrect_claim, pred, correct_claim)


class SariMetric(Metric):
    def __init__(self):
        self.sari = load("sari")

    def score(self, evidence, incorrect_claim, pred, correct_claim):
        assert correct_claim is not None
        sources = [evidence]
        references = [[correct_claim]]
        prediction = [pred]
        return self.sari.compute(sources=sources, references=references, predictions=prediction)['sari']


def score_sari(df, save_name):
    return score_metric(df, metric_class=SariMetric, metric_name="SARI", save_name=save_name)


def score_difference(df, save_name):
    return score_metric(df, metric_class=DifferenceMetric, metric_name="Difference", save_name=save_name)


class DifferenceMetric(Metric):
    def __init__(self):
        self.model = PenalizeSimilarity().model

    def score(self, evidence, incorrect_claim, pred, correct_claim):
        prompt = f"Claim 1: {incorrect_claim}\nClaim 2: {pred}"
        return self.model(prompt).cpu().numpy()[0]


def autoscore(df, save_name, pred_col):
    verifier = VerificationAlgorithm(mode=2)
    df = verifier.mass_predict(dataset=df, pred_col=pred_col, save_name=save_name)
    return df


def handle_scoring():

    for file in val_files:
        df = pd.read_csv(results_path+"/"+file)
        df = score_sari(df, file)
        df = score_difference(df, file)
        if "bart" in file:
            df = autoscore(df, file, pred_col="pred_beam")
            df = autoscore(df, file, pred_col="pred_score")
        else:
            pred_col = "pred"
            if "t5" in file:
                pred_col = "pred_beam"
            df = autoscore(df, file, pred_col=pred_col)
        df.to_csv(f"{save_path}/{file}", index=False)

    for file in final_files:
        print(f"Working on File: {file}")
        df = pd.read_csv(results_path+"/"+file)
        df = score_difference(df, file)
        if "bart" in file:
            df = autoscore(df, file, pred_col="pred_beam")
            df = autoscore(df, file, pred_col="pred_score")
        else:
            df = autoscore(df, file, pred_col="pred")
        df.to_csv(f"{save_path}/{file}", index=False)

    return



if __name__ == "__main__":
    pass