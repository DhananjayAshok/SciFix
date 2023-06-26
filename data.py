import os
import time

import pandas as pd
from datasets import load_dataset
from random import choice, random
from tqdm import tqdm
from dataset_creation import ErrorCreation, TruthExplanation, SimilarCreation
from query_models import OpenAIGPT
from openai.error import APIError

data_root = "correction-dataset/"
scifact_open_data_root = "scifact-open/data"
healthver_root = "HealthVer/data"


def load_scifact_open():
    try:
        claims = pd.read_json(scifact_open_data_root + "/claims.jsonl", lines=True)
        corpus = pd.read_json(scifact_open_data_root + "/corpus.jsonl", lines=True)
    except ValueError:
        raise ValueError(f"Got ValueError: are you sure you have run the scripts/get_data.sh from scifact-open repo?")
    return claims, corpus


def _dataframe_extract(dataset, columns):
    data = []
    for item in dataset:
        row = []
        for column in columns:
            row.append(item[column])
        data.append(row)
    return pd.DataFrame(data=data, columns=columns)


def load_scifact():
    claims = load_dataset("allenai/scifact", "claims")
    corpus = load_dataset("allenai/scifact", "corpus")
    claimlist = []
    corpuslist = []
    print(f"Gets here all loaded")
    for split in ["train", "validation", "test"]:
        print(f"Working on split {split}")
        columns = list(claims[split].features.keys())
        claim_df = _dataframe_extract(claims[split], columns)
        columns = list(corpus["train"].features.keys())
        corpus_df = _dataframe_extract(corpus['train'], columns)
        claimlist.append(claim_df)
        corpuslist.append(corpus_df)
    return pd.concat(claimlist, ignore_index=True), pd.concat(corpuslist, ignore_index=True)


def load_covidfact():
    j = pd.read_json("covidfact/COVIDFACT_dataset.jsonl", lines=True)
    return j


def load_healthver():
    dfs = []
    for split in ["train", "dev", "test"]:
        dfs.append(pd.read_csv(f"{healthver_root}/healthver_{split}.csv"))
    return pd.concat(dfs, ignore_index=True)


class ClaimDataset:
    def __init__(self, df, path=None):
        if path is None:
            self.df = df.reset_index(drop=True)
        else:
            self.df = pd.read_csv(ClaimDataset.handle_path(path))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        return self.df.loc[item]

    @staticmethod
    def handle_path(path):
        if path is None:
            raise ValueError(f"Path Should not be None")
        if "/" not in path or ".csv" not in path:
            return f"{data_root}/{path}.csv"
        elif "/" not in path and ".csv" in path:
            return f"{data_root}/{path}"
        else:
            return path

    def save(self, path):
        self.df.to_csv(ClaimDataset.handle_path(path), index=False)


def create_errors(df, error_creation, dset_checkpoint_name="tmp_err.csv", checkpoint_every=500):
    df = df[df['label'] == "SUPPORT"].reset_index(drop=True)
    df = df[~df['short_evidence'].isna()].reset_index(drop=True)
    df["correct_claim"] = df["claim"]
    df["incorrect_claim"] = None
    pre_df = None
    if os.path.exists(dset_checkpoint_name):
        pre_df = pd.read_csv(dset_checkpoint_name)
        pre_index = pre_df[~pre_df["incorrect_claim"].isna()].index
    idx = df.index
    for i in tqdm(idx, total=len(idx)):
        if pre_df is not None:
            if i in pre_index:
                df.loc[i, "incorrect_claim"] = pre_df.loc[i, "incorrect_claim"]
                continue
        incorrect_claim, success = error_creation(df.loc[i, "claim"])
        if success:
            df.loc[i, "incorrect_claim"] = incorrect_claim
        else:
            df.loc[i, "incorrect_claim"] = "3721OMG[FAIL]OMG" # oddly specific string to avoid issues with fpositive(s)
        if i % checkpoint_every == 0:
            df[["correct_claim", "incorrect_claim", "dataset", "long_evidence", "short_evidence", "explanation"]].to_csv(dset_checkpoint_name, index=False)
    df = df[["correct_claim", "incorrect_claim", "dataset", "long_evidence", "short_evidence", "explanation"]]
    df = df[df["incorrect_claim"] != "3721OMG[FAIL]OMG"]
    return df


class ClaimCorrectionDataset(ClaimDataset):
    def __init__(self, df, error_creation=None, path=None):
        if path is not None:
            super().__init__(df, path=path)
        else:
            self.error_creation = error_creation
            if "label" in df.columns:  # Then this is not a corrected dataset and we must use the ec class
                if error_creation is None:
                    raise ValueError(f"Error Creation cannot be None when dataset doesn't have precomputed errors")
                df = create_errors(df, error_creation)
            super().__init__(df)


def create_similarity_and_errors(df, error_creation, similarity_creation, dataset_limit=None,
                                 dset_checkpoint_name="tmp_sim_err.csv", checkpoint_every=500):
    """
    Label is 1 iff sentences are different in meaning
    """
    df = df[df['label'] == "SUPPORT"].reset_index(drop=True)
    columns = ["text", "label", "dataset"]
    data = []
    checkpointing = dataset_limit is None
    if dataset_limit is None:
        idx = df.index
    else:
        dataset_limit = min(len(df), dataset_limit)
        idx = df.sample(dataset_limit).index
    pre_df = None
    if checkpointing:
        if os.path.exists(dset_checkpoint_name):
            pre_df = pd.read_csv(dset_checkpoint_name)
            print(f"Detecting Checkpoint, restarting from {len(pre_df)}")
    if pre_df is None:
        checkpointing = False
    for i in tqdm(idx, total=len(idx)):
        if checkpointing:
            if i < len(pre_df):
                data.append(pre_df.loc[i].tolist())
                continue
        claim = df.loc[i, "claim"]
        dataset = df.loc[i, "dataset"]
        incorrect_claim, success = error_creation(claim)
        if success:
            data.append([f"Claim 1: {claim}\nClaim 2: {incorrect_claim}", 1, dataset])
        if random() > 0.5:
            data.append([f"Claim 1: {claim}\nClaim 2: {claim}", 0, dataset])
        else:
            correct_claim = similarity_creation(claim)
            data.append([f"Claim 1: {claim}\nClaim 2: {correct_claim}", 0, dataset])
        if checkpointing:
            if i % checkpoint_every == 0:
                tmp_df = pd.DataFrame(data=data, columns=columns)
                tmp_df.to_csv(dset_checkpoint_name, index=False)
    df = pd.DataFrame(data=data, columns=columns)
    return df


class ClaimDifferenceDataset(ClaimDataset):
    def __init__(self, df, error_creation=None, similarity_creation=None, path=None):
        if path is not None:
            super().__init__(df, path=path)
        else:
            self.error_creation = error_creation
            self.similarity_creation = similarity_creation
            if "text" not in df.columns:  # Then this is not a precomputed dataset and we must generate
                if error_creation is None or similarity_creation is None:
                    raise ValueError(f"Error Creation and Similarity Creation cannot be None when dataset doesn't have precomputed results")
                df = create_similarity_and_errors(df, error_creation, similarity_creation)
            super().__init__(df)


def to_common_format(claims, corpus=None, evidence_gap=2):
    """
    Takes in as input, scifact, scifactopen format claims/corpus dataframes
    returns a dataframe with seven columns:
        claim: str, evidence: list[str], evidence_sentences: list[int], label: str, dataset: str, long_evidence: str,
            short_evidence: str
    For covidfact evidence = short_evidence
    """
    working = claims.copy()
    if corpus is None:  # Then its covidfact or healthver
        if all(i in claims.columns for i in ['claim', 'label', 'evidence', 'gold_source', 'flair']):
            working["dataset"] = "covidfact"
            label_map = {"SUPPORTED": "SUPPORT", "REFUTED": "CONTRADICT"}
            working["label"] = claims["label"].map(label_map)
            working["evidence_sentences"] = claims["label"].map(lambda x: [])
            working['long_evidence'] = working['evidence'].apply(lambda x: " ".join(x))
            working['short_evidence'] = working['long_evidence']
            working.drop(["gold_source", "flair"], axis=1, inplace=True)
        elif all(i in claims.columns for i in ['id','evidence','claim','label','topic_ip','question']):
            working["dataset"] = "healthver"
            label_map = {"Supports": "SUPPORT", "Refutes": "CONTRADICT", "Neutral": "NEI"}
            working["label"] = claims["label"].map(label_map)
            working["evidence_sentences"] = claims["label"].map(lambda x: [])
            working['long_evidence'] = working['evidence']
            working['short_evidence'] = working['long_evidence']
            working = working[working["label"].isin(["SUPPORT", "CONTRADICT"])].reset_index(drop=True)

    elif all(i in corpus.columns for i in ['doc_id', 'title', 'abstract', 'metadata', 'scifact_orig']):
        # Then scifact open format. Choose a random document if multiple exist
        working.drop("id", axis=1, inplace=True)
        working["label"] = "NEI"
        working["dataset"] = "scifact-open"
        working["evidence_sentences"] = None
        working["long_evidence"] = ""
        working["short_evidence"] = ""
        iter_index = claims.index
        for i in iter_index:
            claim = claims.loc[i]
            if len(claim["evidence"]) == 0:
                working.at[i, "evidence"] = []
                working.at[i, "evidence_sentences"] = []
            else:
                key = choice(list(claim["evidence"].keys()))
                evidence = claim['evidence'][key]
                label = evidence["label"]
                doc = corpus[corpus['doc_id'] == int(key)]
                idx = doc.index[0]
                doc = doc.loc[idx]
                ev = [doc["title"]] + doc["abstract"]
                working.at[i, "evidence"] = ev
                working.loc[i, "label"] = label
                working.at[i, "evidence_sentences"] = [j+1 for j in evidence["sentences"]]
                working.loc[i, "long_evidence"] = " ".join(ev)
                if len(evidence['sentences']) == 0:
                    working.loc[i, "short_evidence"] = working.loc[i, "long_evidence"]
                else:
                    short_ev_start_ind = min(working.loc[i, 'evidence_sentences']) - evidence_gap
                    short_ev_end_ind = max(working.loc[i, 'evidence_sentences']) + evidence_gap
                    working.loc[i, "short_evidence"] = " ".join(ev[short_ev_start_ind: short_ev_end_ind])

    elif all(i in claims.columns for i in ['id', 'claim', 'evidence_doc_id', 'evidence_label',
       'evidence_sentences', 'cited_doc_ids']):
        working.drop(["id", "cited_doc_ids"], axis=1, inplace=True)
        working["dataset"] = "scifact"
        working["label"] = "NEI"
        working["evidence"] = None
        working["long_evidence"] = ""
        working["short_evidence"] = ""
        for i in claims.index:
            claim = claims.loc[i]
            if claim["evidence_label"] == "":
                pass
            else:
                working.loc[i, "label"] = claim["evidence_label"]
            working.at[i, "evidence_sentences"] = [j + 1 for j in claim["evidence_sentences"]]
            evidence_doc = claim["evidence_doc_id"]
            if evidence_doc in [None, ""]:
                working.at[i, "evidence"] = []
            else:
                doc = corpus[corpus['doc_id'] == int(evidence_doc)]
                idx = doc.index[0]
                doc = doc.loc[idx]
                ev = [doc["title"]] + doc["abstract"]
                working.at[i, "evidence"] = ev
                working.loc[i, "long_evidence"] = " ".join(ev)
                if len(working.loc[i, "evidence_sentences"]) == 0:
                    working.loc[i, "short_evidence"] = working.loc[i, "long_evidence"]
                else:
                    short_ev_start_ind = min(working.loc[i, 'evidence_sentences']) - evidence_gap
                    short_ev_end_ind = max(working.loc[i, 'evidence_sentences']) + evidence_gap
                    working.loc[i, "short_evidence"] = " ".join(ev[short_ev_start_ind: short_ev_end_ind])

        working.drop(["evidence_label"], axis=1, inplace=True)
    else:
        raise ValueError(f"Unrecognized claims dataframe format with columns {claims.columns}")

    return working[['claim', 'evidence_sentences', 'dataset', 'label', 'evidence', 'short_evidence', 'long_evidence']]


def get_joined_df(components=["scifact", "scifact-open", "covidfact", "healthver"]):
    dfs = []
    assert len(components) > 0
    for component in components:
        print(f"Doing Component: {component}")
        if component == "scifact":
            claims, corpus = load_scifact()
            df = to_common_format(claims, corpus)
            dfs.append(df)
        elif component == "scifact-open":
            claims, corpus = load_scifact_open()
            df = to_common_format(claims, corpus)
            dfs.append(df)
        elif component == "covidfact":
            claims = load_covidfact()
            df = to_common_format(claims)
            dfs.append(df)
        elif component == "healthver":
            claims = load_healthver()
            df = to_common_format(claims)
            dfs.append(df)
        else:
            raise ValueError(f"Unrecognized Component {component}")
    df = pd.concat(dfs, ignore_index=True)
    return df


def truth_explain(df, dset_checkpoint_name="tmp_truth_explain.csv", checkpoint_every=2000):
    tmp_df = None
    if os.path.exists(dset_checkpoint_name):
        tmp_df = pd.read_csv(dset_checkpoint_name)
    explainer = TruthExplanation()
    df["explanation"] = None
    if tmp_df is not None:
        print(f"Detected Checkpoint, skipping precomputed explanations")
        skip_indices = tmp_df[~tmp_df.explanation.isna()].index
    iterator = df.index
    has_reached = False
    for i in tqdm(iterator, total=len(iterator)):
        if isinstance(df.loc[i, "short_evidence"], float):
            continue
        if df.loc[i, "label"].strip() == "SUPPORT" and df.loc[i, "short_evidence"].strip() != "":
            if tmp_df is not None:
                if i in skip_indices:
                    df.loc[i, "explanation"] = tmp_df.loc[i, "explanation"]
                    continue
            explanation = explainer(evidence=df.loc[i, "short_evidence"], fact=df.loc[i, "claim"])
            df.loc[i, "explanation"] = explanation
            has_reached = True
        if i % checkpoint_every == 0 and has_reached:
            df.to_csv(dset_checkpoint_name, index=False)
    return


def get_verification_dataset(components=["scifact", "scifact-open", "covidfact", "healthver"], path=None, gen=False):
    if path is not None:
        return ClaimDataset(df=None, path=path)
    elif not gen and os.path.exists(f"{data_root}/all_claims_verification.csv"):
        return ClaimDataset(df=None, path="all_claims_verification")
    else:
        whole_df = get_joined_df(components)
        dataset = ClaimDataset(whole_df)
        return dataset


def get_model_input(dset, short_evidence=True, claim_col="incorrect_claim"):
    evidence_str = "short_evidence" if short_evidence else "long_evidence"
    evidence = dset.df[evidence_str]
    claim = dset.df[claim_col]
    return "Evidence: " + evidence + " \nClaim: " + claim


def get_correction_dataset(gpt=True, short_evidence=True):
    if gpt:
        dset = ClaimCorrectionDataset(df=None, path=f"gpt_dataset")
    else:
        dset = ClaimCorrectionDataset(df=None, path=f"nltk_dataset")
    dset.df["model_input"] = get_model_input(dset, short_evidence=short_evidence)
    return dset


def get_difference_dataset():
    dset = ClaimCorrectionDataset(df=None, path=f"difference_dataset")
    return dset


def generate_full_dataset():
    dset = get_verification_dataset(gen=False)
    truth_explain(dset.df)
    dset.save(f"all_claims_verification")
    df = dset.df[dset.df.label == "CONTRADICT"]
    dataset = ClaimDataset(df=df)
    dataset.save("final_test_full")
    dataset.df = dataset.df[(dataset.df["short_evidence"] != dataset.df["long_evidence"]) |
                            (dataset.df.dataset.isin(["covidfact", "healthver"]))].reset_index(drop=True)
    dataset.save(f"final_test")
    ec = ErrorCreation(mode=1, model=OpenAIGPT())
    dataset = ClaimCorrectionDataset(dset.df, error_creation=ec)
    dataset.save("gpt_dataset")
    sc = SimilarCreation(model=OpenAIGPT())
    dataset = ClaimDifferenceDataset(dset.df, error_creation=ec, similarity_creation=sc)
    dataset.save("difference_dataset")
    ec = ErrorCreation(mode=0)
    dataset = ClaimCorrectionDataset(dset.df, error_creation=ec)
    dataset.save("nltk_dataset")


def generate_verification_splits(train_split=0.8):
    dset = get_verification_dataset()
    dset.df = dset.df.sample(n=len(dset.df))
    dset.df = dset.df[dset.df['label'].isin(["CONTRADICT", "SUPPORT"])].reset_index(drop=True)
    dset.df = dset.df[(dset.df["short_evidence"] != dset.df["long_evidence"]) | (dset.df["dataset"].isin(["covidfact", "healthver"]) )]
    dset.df["label"] = (dset.df['label'] == "SUPPORT").astype(int)  # true claim is label 1
    dset.df["text"] = "Evidence: " + dset.df["short_evidence"] + "\nClaim: " + dset.df["claim"]
    limit_no = 2_300  # Number of sciFact examples is around this much
    covid_index = dset.df[dset.df['dataset'].isin(["covidfact", "healthver"])].index
    index_subset = covid_index[limit_no:]
    dset.df.drop(index_subset, inplace=True)
    dset.df = dset.df[["text", "label", "dataset"]]
    dset.df = dset.df[~dset.df.isna().any(axis=1)].reset_index(drop=True)
    dset.df = dset.df.sample(n=len(dset))
    train_size = int(len(dset) * train_split)
    train_df = dset.df[:train_size]
    save_name = f"verification_train.csv"
    train_df.to_csv(f"{data_root}/{save_name}", index=False)

    test_df = dset.df[train_size:]
    test_df.reset_index(drop=True, inplace=True)
    save_name = f"verification_test.csv"
    test_df.to_csv(f"{data_root}/{save_name}", index=False)

    for df, suffix in [(train_df, "train"), (test_df, "test")]:
        df["text_column"] = df["text"] + "\nQ: Is the following claim true or false? " \
                                         "(Answer with true or false)\nAnswer"
        df["summary_column"] = None
        df["summary_column"][df["label"] == 0] = ": False"
        df["summary_column"][df["label"] == 1] = ": True"
        df = df.drop(["text", "label"], axis=1)
        save_name = f"verification_seq_{suffix}.csv"
        df.to_csv(f"{data_root}/{save_name}", index=False)
    return


def generate_difference_splits(train_split=0.8):
    dset = get_difference_dataset()
    dset.df = dset.df.sample(n=len(dset))
    dset.df = dset.df[~dset.df.isna().any(axis=1)].reset_index(drop=True)
    if "dataset" in dset.df.columns:
        limit_no = 1500  # Number of sciFact examples is around this
        covid_index = dset.df[dset.df['dataset'].isin(["covidfact", "healthver"])].index
        index_subset = covid_index[limit_no:]
        dset.df.drop(index_subset, inplace=True)
    train_size = int(len(dset) * train_split)
    train_df = dset.df[:train_size]
    save_name = f"difference_train.csv"
    train_df.to_csv(f"{data_root}/{save_name}", index=False)

    test_df = dset.df[train_size:]
    test_df.reset_index(drop=True, inplace=True)
    save_name = f"difference_test.csv"
    test_df.to_csv(f"{data_root}/{save_name}", index=False)

    for df, suffix in [(train_df, "train"), (test_df, "test")]:
        df["text_column"] = df["text"] + "\nQ: Do these claims have different meanings?\nAnswer: "
        df["summary_column"] = None
        df["summary_column"][df["label"] == 0] = df["text"][df["label"] == 0] + "\nThese two claims are actually have the same meaning"
        df["summary_column"][df["label"] == 1] = df["text"][df["label"] == 1] + "\nIt is true, these are two different claims with different meanings"
        df = df.drop(["text", "label"], axis=1)
        save_name = f"difference_seq_{suffix}.csv"
        df.to_csv(f"{data_root}/{save_name}", index=False)
    return


def generate_correction_splits(num_val=200, gpt=True, short_evidence=True, explanation=True, split=False, balanced=True):
    dset = get_correction_dataset(gpt=gpt, short_evidence=short_evidence)
    dset.df = dset.df[dset.df.dataset != "healthver"]
    dset.df = dset.df[~dset.df.isna().any(axis=1)].reset_index(drop=True)
    if balanced:
        limit_no = 1000  # 890 Number of sciFact examples
        covid_index = dset.df[dset.df['dataset'].isin(["covidfact", "healthver"])].index
        index_subset = covid_index[limit_no:]
        dset.df.drop(index_subset, inplace=True)
        dset.df.reset_index(drop=True)
    if split:
        dset.df.drop(dset.df[dset.df['dataset'].isin(["covidfact", "healthver"])].index, inplace=True)
        dset.df.reset_index(drop=True)
    dset.df = dset.df.sample(n=len(dset))
    train_size = len(dset) - num_val
    train_df = dset.df[:train_size]
    save_name = f"correction_{'gpt' if gpt else 'nltk'}_{'short' if short_evidence else 'long'}" \
                f"{'_exp' if explanation else ''}{'_split' if split else ''}{'_bal' if balanced else ''}"
    train_df["text_column"] = train_df["model_input"]
    if not explanation:
        train_df["summary_column"] = train_df["correct_claim"]
    else:
        train_df["summary_column"] = train_df["explanation"] + " | " + train_df["correct_claim"]
    train_df = train_df[["text_column", "summary_column", "dataset"]]
    train_df.to_csv(f"{data_root}/{save_name}_train.csv", index=False)

    test_df = dset.df[train_size:]
    test_df.reset_index(drop=True, inplace=True)
    test_df["text_column"] = test_df["model_input"]
    if not explanation:
        test_df["summary_column"] = test_df["correct_claim"]
    else:
        test_df["summary_column"] = test_df["explanation"] + " | " + test_df["correct_claim"]
    test_df = test_df[["text_column", "summary_column", "dataset"]]
    test_df.to_csv(f"{data_root}/{save_name}_test.csv", index=False)


def generate_final_formatted(num_samples=500):
    dset = ClaimDataset(df=None, path=f"final_test")
    dset.df = dset.df[~dset.df.short_evidence.isna()]
    dset.df = dset.df[dset.df.dataset != "healthver"]
    curr_df = dset.df.copy()
    total_df = pd.DataFrame(columns=["text_column", "dataset"])
    dfs = []
    for dataset in dset.df.dataset.unique().tolist():
        subset = dset.df[dset.df.dataset == dataset]
        dset.df = subset.sample(min(len(subset), (num_samples - 84)//(len(dset.df.dataset.unique().tolist())-1) ) ).reset_index(drop=True)  # scifact-open has 90 examples without na in short evidence
        df = total_df.copy()
        df['text_column'] = get_model_input(dset, short_evidence=True, claim_col="claim")
        df['dataset'] = dset.df.dataset
        dfs.append(df)
        dset.df = curr_df
    df = pd.concat(dfs, ignore_index=True).sample(num_samples).reset_index(drop=True)
    df.to_csv(f"{data_root}/final_test_formatted.csv", index=False)

    dset = ClaimDataset(df=None, path=f"final_test")
    dset.df = dset.df[dset.df.dataset != "healthver"]
    df = pd.DataFrame(columns=["text_column", "dataset"])
    dset.df = dset.df[dset.df.dataset.isin(["covidfact", "healthver"])].sample(num_samples).reset_index(drop=True)
    df['text_column'] = get_model_input(dset, short_evidence=True, claim_col="claim")
    df['dataset'] = dset.df.dataset
    df.to_csv(f"{data_root}/final_test_split_formatted.csv", index=False)


def generate_splits(train_split=0.8):
    generate_verification_splits(train_split=train_split)
    generate_difference_splits(train_split=train_split)
    generate_final_formatted()
    for balanced in [False, True]:
        for split in [False, True]:
            for explanation in [False, True]:
                generate_correction_splits(explanation=explanation, balanced=balanced, split=split)


def generate_vence_verifier_training(train_split=0.85):
    df = get_verification_dataset().df
    df["sentence1"] = df["claim"]
    df["sentence2"] = df["short_evidence"]
    df = df[["sentence1", "sentence2", "label"]]
    df = df[df["label"].isin(["SUPPORT", "CONTRADICT"])]
    df = df[~df.isna().any(axis=1)].reset_index(drop=True)
    label_map = {"SUPPORT": "SUPPORTS", "CONTRADICT": "REFUTES"}
    df["label"] = df["label"].map(label_map)
    df = df.sample(n=len(df)).reset_index(drop=True)
    train_size = int(len(df)* train_split)
    train = df[:train_size]
    val = df[train_size:]
    train.to_json(f"{data_root}/VENCE_train.json")
    val.to_json(f"{data_root}/VENCE_val.json")
    train.to_csv(f"{data_root}/VENCE_train.csv")
    val.to_csv(f"{data_root}/VENCE_val.csv")


def generate_zerofec():
    dset1 = ClaimDataset(df=None, path="correction_gpt_short_bal_test")
    dset2 = ClaimDataset(df=None, path="final_test_formatted")
    for dset in dset1, dset2:
        df = dset.df
        df['verdict'] = "REFUTES"
        df['original_id'] = 0
        df['sentence_id'] = 0
        if "summary_colum" in df:
            df["original"] = df["summary_column"]
        else:
            df["original"] = None
        df["mutated"] = df["text_column"].apply(lambda x: x.split("Claim:")[1])
        df["evidence_text"] = df["text_column"].apply(lambda x: x.split("Claim:")[0].split(". "))
        for i in []:
            row = df.loc[i]
            evidence, mutated = row['text_column'].split("Claim:")
            evidence = evidence.split(". ")
            df.loc[i, "mutated"] = mutated
            df.loc[i, "evidence_text"] = evidence
        if "summary_column" in df:
            df.to_json("val_test.jsonl", orient="records", lines=True)
        else:
            df.to_json("final_test.jsonl", orient="records", lines=True)            


if __name__ == "__main__":
    #generate_full_dataset()
    generate_splits()


