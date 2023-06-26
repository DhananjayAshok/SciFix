import openai.error
from numpy.random import choice
from numpy.random import randint
from query_models import OpenAIGPT, T5Seq2Seq
import torch


class ErrorCreation:
    opposite_configuration = """
    For the given sentence provide a sentence with a completely different, contradictory or opposite meaning
    Sentence: 0-dimensional biomaterials lack inductive properties.
    Answer: Inductive properties are possessed by 0-dimensional biomaterials
    
    Sentence: Coronavirus vaccine to be tested on mice by june 1
    Answer: The coronavirus vaccine is not going to be tested
    
    Sentence: 10% of sudden infant death syndrome (SIDS) deaths happen in newborns aged less than 6 months.
    Answer: 10% of cholera deaths happen in newborns aged less than 6 months
    
    Sentence:
    """

    def __init__(self, mode=1, model=None):  # model from query_models.py
        self.mode = mode
        if mode == 0:
            raise ValueError(f"mode 0 is depricated and removed from source code, please use only mode=1")
        else:
            if model is None:
                self.model = T5Seq2Seq(size="large")
            else:
                self.model = model

    @staticmethod
    def output_parse(ret):
        return ret

    def perform(self, claim):
        prompt = ErrorCreation.opposite_configuration + claim + "\nAnswer: "
        ret = self.model.query(prompt).strip()
        return ret, ret != claim.strip()

    def __call__(self, claim):
        return self.perform(claim)


class SimilarCreation:
    configuration = """
    For the given sentence provide a slightly different sentence with the same meaning
    Sentence: 0-dimensional biomaterials lack inductive properties.
    Answer: 0-dimensional biomaterials do not have any inductive properties. 

    Sentence: 1-10% of colorectal cancer patients are diagnosed with regional or distant metastases.
    Answer: 1-10% of colorectal cancer patients are diagnosed with metastases.

    Sentence: 1 in 5 million in UK have abnormal PrP positivity.
    Answer: The rate of prevalence of abnormal PrP positivity in the UK is 1 in 5 million

    Sentence:
    """

    def __init__(self, model=OpenAIGPT()):  # model from query_models.py
        self.model = model

    @staticmethod
    def output_parse(ret):
        return ret

    def perform(self, claim):
        prompt = SimilarCreation.configuration + claim + "\nAnswer: "
        ret = self.model.query(prompt).strip()
        return ret

    def __call__(self, claim):
        return self.perform(claim)


class TruthExplanation:
    configuration = """
    For the given evidence, explain why the following fact is true
    Evidence:  Little anecdotal evidence for negative impact of opiate replacement therapy on liver transplantation outcome was found. Policies requiring discontinuation of methadone in 32% of all programs contradict the evidence base for efficacy of long-term replacement therapies and potentially result in relapse of previously stable patients.
    Fact: 32% of liver transplantation programs required patients to discontinue methadone treatment in 2001.
    Explanation: The evidence explicitly mentions that policies required discontuation of methadone in 32% of all programs, hence the claim is true
    
    Evidence: INTERVENTION Participants received a daily capsule containing 40 mg of folic acid, 100 mg of pyridoxine hydrochloride (vitamin B6), and 2 mg of cyanocobalamin (vitamin B12) or a placebo.   \n MAIN OUTCOME MEASURES The primary outcome was all-cause mortality. Secondary outcomes included myocardial infarction (MI), stroke, amputation of all or part of a lower extremity, a composite of these 3 plus all-cause mortality, time to initiation of dialysis, and time to thrombosis of arteriovenous access in hemodialysis patients.   \n RESULTS Mean baseline homocysteine level was 24.0 micromol/L in the vitamin group and 24.2 micromol/L in the placebo group. It was lowered 6.3 micromol/L (25.8%; P < .001) in the vitamin group and 0.4 micromol/L (1.7%; P = .14) in the placebo group at 3 months, but there was no significant effect on mortality (448 vitamin group deaths vs 436 placebo group deaths) (hazard ratio [HR], 1.04; 95% CI, 0.91-1.18). No significant effects were demonstrated for secondary outcomes or adverse events: there were 129 MIs in the vitamin group vs 150 for placebo (HR, 0.86; 95% CI, 0.67-1.08), 37 strokes in the vitamin group vs 41 for placebo (HR, 0.90; 95% CI, 0.58-1.40), and 60 amputations in the vitamin group vs 53 for placebo (HR, 1.14; 95% CI, 0.79-1.64).
    Fact: 40mg/day dosage of folic acid and 2mg/day dosage of vitamin B12 does not affect chronic kidney disease (CKD) progression.
    Explanation: The evidence mentions that the mean baseline homocysteine (potential effect) of the dose of folic acid was the same as the placebo and that no significant effect was observed, hence the dose does not affect chronic kidney disease and the claim is true
    
    Evidence: 
    """

    def __init__(self, model=OpenAIGPT()):  # model from query_models.py
        self.model = model
    @staticmethod
    def output_parse(ret):
        return ret

    def perform(self, evidence, fact):
        prompt = TruthExplanation.configuration + " " + evidence + f"\nFact: {fact}\nExplanation: "
        ret = self.model.query(prompt).strip()
        return ret

    def __call__(self, evidence, fact):
        return self.perform(evidence, fact)


if __name__ == "__main__":
    pass