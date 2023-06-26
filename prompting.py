import pandas as pd
from data import *
from tqdm import tqdm
import warnings

results_root = "results"
save_root = "scoring"


class CorrectionAlgorithm:
    zero_shot_single_task_description = """
        For the following evidence claim pair, give a corrected claim that fixes the error in the original claim:

        Evidence: ...
        Claim: 'Incorrect Claim'
        Answer: 'Corrected Claim'
        """

    few_shot_single_task_description = """
    For the following evidence claim pair, give a corrected claim that fixes the error in the original claim:
    
    Evidence: 'Policies requiring discontinuation of methadone in 32% of all programs contradict the evidence base 
    for efficacy of long-term replacement therapies and potentially result in relapse of previously stable patients    
    Claim: '48% of liver transplantation programs allowed patients to continue methadone treatment in 2001.'
    Answer: 32% of liver transplantation programs required patients to discontinue methadone treatment in 2001
    
    Evidence: Although disorder severity was correlated with probability of treatment in almost all countries, 
    35.5% to 50.3% of serious cases in developed countries and 76.3% to 85.4% in less-developed countries received 
    no treatment in the 12 months before the interview.
    Claim: '15.5%-60% of people with severe mental disorder receive no treatment in low and middle income countries.'    
    Answer: '76.3-85.4% of people with severe mental disorder receive no treatment in low and middle income countries.'
    """

    cot_single_task_description = """
    For the following evidence claim pair,  give a corrected claim that fixes the error in the original claim along with an explanation:

    Evidence: 'Policies requiring discontinuation of methadone in 32% of all programs contradict the evidence base 
    for efficacy of long-term replacement therapies and potentially result in relapse of previously stable patients    
    Claim: '48% of liver transplantation programs allowed patients to continue methadone treatment in 2001.'
    Answer: The claim incorrectly identifies the percentage number from the evidence | 32% of liver transplantation programs required patients to discontinue methadone treatment in 2001

    Evidence: Although disorder severity was correlated with probability of treatment in almost all countries, 
    35.5% to 50.3% of serious cases in developed countries and 76.3% to 85.4% in less-developed countries received 
    no treatment in the 12 months before the interview.
    Claim: '15.5%-60% of people with severe mental disorder receive no treatment in low and middle income countries.'    
    Answer: The claim incorrectly identifies the percentage number from the evidence | '76.3-85.4% of people with severe mental disorder receive no treatment in low and middle income countries.'
    """

    zero_shot_joint_task_description = """
    For the following evidence claim pair, declare whether the claim is TRUE or FALSE and if false give a corrected claim:

    Evidence: ...
    Claim: Some false claim
    Answer: False | Corrected Claim

    Evidence: ...
    Claim: Some true claim
    Answer: True | NONE
    """

    few_shot_joint_task_description = """
    For the following evidence claim pair, declare whether the claim is TRUE or FALSE and if false give a corrected claim:
    
    Evidence: 'Policies requiring discontinuation of methadone in 32% of all programs contradict the evidence base 
    for efficacy of long-term replacement therapies and potentially result in relapse of previously stable patients    
    Claim: '48% of liver transplantation programs allowed patients to continue methadone treatment in 2001.'
    Answer: False | 32% of liver transplantation programs required patients to discontinue methadone treatment in 2001
    
    Evidence: Although disorder severity was correlated with probability of treatment in almost all countries, 
    35.5% to 50.3% of serious cases in developed countries and 76.3% to 85.4% in less-developed countries received 
    no treatment in the 12 months before the interview.    
    Claim: '76-85% of people with severe mental disorder receive no treatment in low and middle income countries.'
    Answer: True | NONE
    """

    cot_joint_task_description = """
    For the following evidence claim pair, declare whether the claim is TRUE or FALSE and if false give a corrected claim:
    
    Evidence: 'Policies requiring discontinuation of methadone in 32% of all programs contradict the evidence base 
    for efficacy of long-term replacement therapies and potentially result in relapse of previously stable patients    
    Claim: '48% of liver transplantation programs allowed patients to continue methadone treatment in 2001.'
    Answer: False because the percentage number is incorrectly identified | 32% of liver transplantation programs required patients to discontinue methadone treatment in 2001
    
    Evidence: Although disorder severity was correlated with probability of treatment in almost all countries, 
    35.5% to 50.3% of serious cases in developed countries and 76.3% to 85.4% in less-developed countries received 
    no treatment in the 12 months before the interview.    
    Claim: '76-85% of people with severe mental disorder receive no treatment in low and middle income countries.'
    Answer: True because the claim is consistent with the evidence | NONE
    """

    def __init__(self, model=OpenAIGPT(), joint=False, mode=0):
        if joint:
            if mode == 0:
                self.task_description = self.zero_shot_joint_task_description
            elif mode == 1:
                self.task_description = self.few_shot_joint_task_description
            else:
                self.task_description = self.cot_joint_task_description
        else:
            if mode == 0:
                self.task_description = self.zero_shot_single_task_description
            elif mode == 1:
                self.task_description = self.few_shot_single_task_description
            else:
                self.task_description = self.cot_single_task_description
        self.model = model

    def perform(self, evidence, incorrect_claim):
        if isinstance(evidence, float) or isinstance(incorrect_claim, float):
            return False, "NaN"
        single = f"Evidence: {evidence} \n Claim: {incorrect_claim}"
        return self.perform_single(single)

    def perform_single(self, single):
        if isinstance(single, float):
            return False, "NaN"
        task = self.task_description + single + "\nAnswer:"
        ans = self.model.query(task)
        if "|" not in ans:
            status = "false"
            corrected = ans
        else:
            status, corrected = ans.split("|")
        if "True".lower() in status.lower():
            return True, None
        elif "False".lower() in status.lower():
            return False, corrected

    def mass_predict(self, dataset, save_name="GPT3CorrResults", limit=None, checkpoint_every=500):
        if not isinstance(dataset, pd.DataFrame):
            dataset = dataset.df
        summ_format = "text_column" in dataset.columns  # Then its in summarization df format
        if limit is None:
            limit = len(dataset) + 1
        dataset = dataset.reset_index(drop=True)
        checkpoint = False
        checkpoint_file = f"{results_root}/tmp_{save_name}.csv"
        if os.path.exists(checkpoint_file):
            tmp_df = pd.read_csv(checkpoint_file)
            checkpoint = True
            na_series = tmp_df["pred"].isna()
        dataset["pred"] = None
        for i in tqdm(range(len(dataset))):
            if checkpoint:
                if not na_series[i]:
                    dataset.loc[i, "pred"] = tmp_df.loc[i, "pred"]
            row = dataset.loc[i]
            if summ_format:
                single = row["text_column"]
                ans_bool, ans = self.perform_single(single)
            else:
                evidence = row["short_evidence"]
                claim = row["incorrect_claim"]
                ans_bool, ans = self.perform(evidence, claim)
            dataset.loc[i, "pred"] = ans
            if i % checkpoint_every == 0:
                dataset.to_csv(checkpoint_file, index=False)
            if i > limit:
                print(f"Early Exit ...")
                break
        dataset.to_csv(f"{results_root}/{save_name}.csv", index=False)
        return dataset


class VerificationAlgorithm:
    task_description = """
    For the following evidence claim pair, declare whether the claim is TRUE or FALSE. Explain why:

    Evidence: 'Policies requiring discontinuation of methadone in 32% of all programs contradict the evidence base 
    for efficacy of long-term replacement therapies and potentially result in relapse of previously stable patients
    Claim: '48% of liver transplantation programs allowed patients to continue methadone treatment in 2001.'
    Answer: False

    Evidence: Although disorder severity was correlated with probability of treatment in almost all countries, 
    35.5% to 50.3% of serious cases in developed countries and 76.3% to 85.4% in less-developed countries received 
    no treatment in the 12 months before the interview.
    Claim: '76-85% of people with severe mental disorder receive no treatment in low and middle income countries.'
    Answer: True    
    """

    cot_task_description = """
    For the following evidence claim pair, declare whether the claim is TRUE or FALSE and explain why:

    Evidence: 'Policies requiring discontinuation of methadone in 32% of all programs contradict the evidence base 
    for efficacy of long-term replacement therapies and potentially result in relapse of previously stable patients
    Claim: '48% of liver transplantation programs allowed patients to continue methadone treatment in 2001.'
    Answer: The evidence states 32% of all programs had policies requiring discontinuation of methadone, this means 68% of programs allowed patients to continue methadone treatmnent, which is not the same as 48% | False

    Evidence: Although disorder severity was correlated with probability of treatment in almost all countries, 
    35.5% to 50.3% of serious cases in developed countries and 76.3% to 85.4% in less-developed countries received 
    no treatment in the 12 months before the interview.
    Claim: 76-85% of people with severe mental disorder receive no treatment in low and middle income countries.'
    Answer: The evidence states that 76.3% to 85.4% of serious cases of mental disorder received no treatment, this is approximately the same as 76%-85% from the claim | True
    """

    autoscore_description = """
    For the following evidence and claim (fact may or may not be included), declare whether the claim is supported by the evidence and fact (TRUE) or not (FALSE):

    Evidence: 'Policies requiring discontinuation of methadone in 32% of all programs contradict the evidence base 
    for efficacy of long-term replacement therapies and potentially result in relapse of previously stable patients
    Fact: '32% of liver transplantation programs allowed patients to continue methadone treatment in 2001.'
    Claim: '48% of liver transplantation programs allowed patients to continue methadone treatment in 2001.'
    Answer: False
    
    Evidence: Although disorder severity was correlated with probability of treatment in almost all countries, 
    35.5% to 50.3% of serious cases in developed countries and 76.3% to 85.4% in less-developed countries received 
    no treatment in the 12 months before the interview.
    Fact: 76-85% of people with severe mental disorder receive no treatment in low and middle income countries.
    Claim: 76.3-85.4% of people with severe mental disorder receive no treatment in low and middle income countries.
    Answer: True
    
    Evidence: 'Policies requiring discontinuation of methadone in 32% of all programs contradict the evidence base 
    for efficacy of long-term replacement therapies and potentially result in relapse of previously stable patients
    Fact: None
    Claim: Liver transplantation is a risky procedure which has a non trivial chance of fatal complications
    Answer: False

    
    Evidence: Although disorder severity was correlated with probability of treatment in almost all countries, 
    35.5% to 50.3% of serious cases in developed countries and 76.3% to 85.4% in less-developed countries received 
    no treatment in the 12 months before the interview.
    Fact: None
    Claim: '35.5% to 50.3% of people with severe mental disorder receive no treatment in developed countries.
    Answer: True
    """

    def __init__(self, model=OpenAIGPT(), mode=0):
        if mode == 1:
            self.task_description = self.cot_task_description
        elif mode >= 2:
            self.task_description = self.autoscore_description
        self.model = model

    def perform(self, evidence, claim):
        single = f"Evidence: {evidence} \n Claim: {claim}"
        return self.perform_single(single)

    def perform_score(self, evidence, fact, claim):
        single = f"Evidence: {evidence}\nFact: {fact} \n Claim: {claim}"
        return self.perform_single(single)

    def perform_single(self, single):
        task = self.task_description + f"{single} \nAnswer: "
        ans = self.model.query(task)
        if "|" not in ans:
            status, explanation = ans, None
        else:
            explanation, status = ans.split("|")
        if "True".lower() in status.lower() or "Supports".lower() in status.lower():
            return True, explanation
        elif "False".lower() in status.lower() or "does not support".lower() in status.lower():
            return False, explanation
        else:
            warnings.warn(f"GPT Got not true or false: {status}")
            return None, None

    def mass_predict(self, dataset, save_name="GPT3VerResults", limit=None, pred_col=None, checkpoint_every=250):
        if not isinstance(dataset, pd.DataFrame):
            dataset = dataset.df
        pred_format = pred_col is not None
        if limit is None:
            limit = len(dataset) + 1
        dataset = dataset.reset_index(drop=True)
        if not pred_format:
            dec_col = "decision"
        else:
            dec_col = f"{pred_col}_decision"
        checkpoint = False
        checkpoint_file = f"{results_root}/tmp_{save_name}.csv"
        if os.path.exists(checkpoint_file):
            tmp_df = pd.read_csv(checkpoint_file)
            if dec_col in tmp_df:
                checkpoint = True
                na_series = tmp_df[dec_col].isna()
        dataset[dec_col] = None
        for i in tqdm(range(len(dataset))):
            row = dataset.loc[i]
            if checkpoint:
                if not na_series[i]:
                    dataset.loc[i, dec_col] = tmp_df.loc[i, dec_col]
                    continue
            if pred_format:
                evidence = row["text_column"].split("Claim:")[0]
                claim = row[pred_col]
                if "|" in claim:
                    claim = claim.split("|")[1]
                if "summary_column" in row:  # then label exists
                    if row.isna()["summary_column"]:
                        fact = "None"
                    else:
                        fact = row["summary_column"]
                        if "|" in fact:
                            fact = fact.split("|")[1]
                else:
                    fact = "None"
                ans_bool, ans = self.perform_score(evidence, fact, claim)
            else:
                evidence = row["short_evidence"]
                claim = row["claim"]
                ans_bool, ans = self.perform(evidence, claim)
            dataset.loc[i, dec_col] = ans_bool
            if i % checkpoint_every == 0:
                dataset.to_csv(checkpoint_file, index=False)
            if i > limit:
                print(f"Early Exit ...")
                break
        dataset.to_csv(f"{save_root}/{save_name}.csv" if ".csv" not in save_name else f"{save_root}/{save_name}", index=False)
        return dataset


def benchmark_verification(cot=True, limit=200):
    dataset = get_verification_dataset()
    dataset.df = dataset.df[dataset.df["label"].isin(["SUPPORT", "CONTRADICT", "SUPPORTS", "CONTRADICTS"])]
    alg = VerificationAlgorithm(mode=int(cot))
    alg.mass_predict(dataset, save_name=f"GPT_verification{'_CoT' if cot else ''}_benchmark",
                     limit=limit)
    return


def benchmark_correction(limit=200, final=False, mode=0, split=False):
    if not final:
        dataset = ClaimDataset(df=None, path=f"correction_gpt_short_{'split_' if split else ''}bal_test")
    else:
        dataset = ClaimDataset(df=None, path=f"final_test_{'split_' if split else ''}formatted")
    alg = CorrectionAlgorithm(joint=False, mode=mode)
    alg.mass_predict(dataset, save_name=f"GPT_correction_mode_{mode}_benchmark{'_split' if split else ''}{'_final' if final else ''}", limit=limit)
    return


if __name__ == "__main__":
    # benchmark_correction(limit=200)
    # benchmark_verification(limit=200)
    for final in [False, True]:
        for mode in [0, 1]:
            benchmark_correction(final=final, mode=mode, limit=None)
