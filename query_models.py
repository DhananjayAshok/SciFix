import os
import warnings

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, \
    BartForConditionalGeneration, AutoModelForSequenceClassification
from transformers.models.bart.modeling_bart import shift_tokens_right
import numpy as np
import torch
import openai
import time
import utils

openai.api_key = os.getenv("OPENAI_API_KEY")


class OpenAIGPT:
    #model = "gpt-3.5-turbo"
    model = "text-davinci-002"
    #model = "gpt-4"
    #model = "davinci"
    seconds_per_query = (60 / 20) + 0.01
    @staticmethod
    def request_model(prompt):
        return openai.Completion.create(model=OpenAIGPT.model, prompt=prompt, max_tokens=250)

    @staticmethod
    def request_chat_model(msgs):
        messages = []
        for message in msgs:
            content, role = message
            messages.append({"role": role, "content": content})
        return openai.ChatCompletion.create(model=OpenAIGPT.model, messages=messages)

    @staticmethod
    def decode_response(response):
        if OpenAIGPT.is_chat():
            return response["choices"][0]["message"]["content"]
        else:
            return response["choices"][0]["text"]

    @staticmethod
    def query(prompt):
        while True:
            try:
                return OpenAIGPT.decode_response(OpenAIGPT.request_model(prompt))
            except openai.error.APIError:
                print(f"Got APIError. Sleeping 5 Seconds...")
                time.sleep(5)
            except openai.error.RateLimitError:
                print(f"Got RateLimitError. Sleeping 5 Seconds...")
                time.sleep(5)
            else:
                warnings.warn(f"Unrecognized OpenAI Error")
                time.sleep(5)

    @staticmethod
    def chat_query(msgs):
        return OpenAIGPT.decode_response(OpenAIGPT.request_chat_model(msgs))

    @staticmethod
    def is_chat():
        return OpenAIGPT.model in ["gpt-4"]

    @staticmethod
    def __call__(inputs):
        while True:
            try:
                if OpenAIGPT.is_chat():
                    return OpenAIGPT.chat_query(inputs)
                else:
                    return OpenAIGPT.query(inputs)
            except openai.error.APIError:
                time.sleep(5)
            except openai.error.RateLimitError:
                time.sleep(5)
            else:
                warnings.warn(f"Unrecognized OpenAI Error")
                time.sleep(5)


class HuggingFaceModel:

    def __call__(self, prompt, mode=None):
        return self.query(prompt, mode=mode)

    def to(self, device=None):
        if device is None:
            device = utils.Parameters.devices[0]
        self.model = self.model.to(device)


class Seq2SeqHugginFaceModel(HuggingFaceModel):
    def query(self, prompt, mode=None):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(utils.Parameters.devices[0])
        with torch.set_grad_enabled(False):
            if isinstance(mode, Scorer):
                if isinstance(mode, PenalizeExplanationSimilarity):
                    outputs = score_guided_decode(self, prompt, scorer=mode, beam_size=3, expected_separator="|")[0]
                else:
                    outputs = score_guided_decode(self, prompt, scorer=mode, beam_size=3)[0]
            elif mode == "beam":
                outputs = self.model.generate(**inputs, max_new_tokens=1000, num_beams=5)
            elif mode == "score":
                outputs = score_guided_decode(self, prompt, scorer=PenalizeSimilarity(), beam_size=3)[0]
            else:
                outputs = self.model.generate(**inputs, max_new_tokens=1000)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    def tf_prob(self, prompt):
        t_token = self.tokenizer("True").input_ids[0]
        f_token = self.tokenizer("False").input_ids[0]
        inputs = self.tokenizer(prompt, return_tensors="pt").to(utils.Parameters.devices[0])
        logits = torch.zeros(2)
        with torch.set_grad_enabled(False):
            outputs = self.model(**inputs)
        logits[0] = outputs.logits[0, -2][f_token]
        logits[1] = outputs.logits[0, -2][t_token]
        probs = torch.softmax(logits, dim=0)
        return probs[1]


class ClassificationHuggingfaceModel(HuggingFaceModel):
    def query(self, prompt, mode=None):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(utils.Parameters.devices[0])
        with torch.set_grad_enabled(False):
            outputs = self.model(**inputs).logits
        return torch.softmax(outputs, dim=1)[:, 1]


class T5Seq2Seq(Seq2SeqHugginFaceModel):
    def __init__(self, size="base", path=None):
        if path is None:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(f"google/flan-t5-{size}")
            self.tokenizer = AutoTokenizer.from_pretrained(f"google/flan-t5-{size}", model_max_length=800)
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(f"{path}")
            self.tokenizer = AutoTokenizer.from_pretrained(path, model_max_length=800)
        self.model.to(utils.Parameters.devices[0])


class BARTSeq2Seq(Seq2SeqHugginFaceModel):
    def __init__(self, path=None):
        if path is None:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(f"AutonLabTruth/bart-base-mlm-subewordmask-matrix")
            self.tokenizer = AutoTokenizer.from_pretrained(f"AutonLabTruth/bart-base-mlm-subewordmask-matrix",
                                                           model_max_length=800)
        else:
            self.model = BartForConditionalGeneration.from_pretrained(f"{path}")
            self.tokenizer = AutoTokenizer.from_pretrained(f"{path}", model_max_length=800)
        self.model.to(utils.Parameters.devices[0])


class T5Classification(ClassificationHuggingfaceModel):
    def __init__(self, size="base", path=None, num_labels=2):
        if path is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(f"google/flan-t5-{size}",
                                                                            num_labels=num_labels)
            self.tokenizer = AutoTokenizer.from_pretrained(f"google/flan-t5-{size}", model_max_length=800)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(f"{path}", num_labels=num_labels)
            self.tokenizer = AutoTokenizer.from_pretrained(path, model_max_length=800)
        self.model.to(utils.Parameters.devices[0])


class BARTClassification(ClassificationHuggingfaceModel):
    def __init__(self, path=None, num_labels=2):
        if path is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                f"AutonLabTruth/bart-base-mlm-subewordmask-matrix", num_labels=num_labels)
            self.tokenizer = AutoTokenizer.from_pretrained(f"AutonLabTruth/bart-base-mlm-subewordmask-matrix",
                                                           model_max_length=800)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(f"{path}", num_labels=num_labels)
            self.tokenizer = AutoTokenizer.from_pretrained(f"{path}", model_max_length=800)
        self.model.to(utils.Parameters.devices[0])


def get_bart_cc():
    return BARTSeq2Seq(path="models/bart_claim_correction_exp_bal")


def _get_next_token_logits(model, input_ids, beam_ids):
    beam_words = model.tokenizer.decode(beam_ids)
    beam_words = model.tokenizer(beam_words, return_tensors="pt").input_ids.to(utils.Parameters.devices[0])
    if isinstance(model, T5Seq2Seq):
        decoder_inp_ids = model.model._shift_right(beam_words)
    else:
        decoder_inp_ids = shift_tokens_right(beam_words, pad_token_id=model.model.config.pad_token_id,
                                             decoder_start_token_id=model.model.config.decoder_start_token_id)

    decoder_inp_ids.to(utils.Parameters.devices[0])
    with torch.set_grad_enabled(False):
        logits = model.model(input_ids=input_ids, decoder_input_ids=decoder_inp_ids).logits[0, -1]
    return logits


def greedy_decode(model, text, max_length=200, partial_output=[], input_ids=None):
    if input_ids is None:
        input_ids = model.tokenizer(text, return_tensors="pt").input_ids.to(utils.Parameters.devices[0])
    if len(partial_output) > 0:
        if partial_output[-1] == model.model.config.eos_token_id or len(partial_output) >= max_length:
            return partial_output
    for _ in range(max_length - len(partial_output)):
        logits = _get_next_token_logits(model, input_ids, beam_ids=partial_output)
        partial_output = partial_output + [logits.argmax()]
        if partial_output[-1] == model.model.config.eos_token_id or len(partial_output) >= max_length:
            return partial_output
        del logits
    return partial_output


def beam_search_decode(model, text, beam_size=5, max_length=200):
    beams = [[] for _ in range(beam_size)]
    # Initialize the log-likelihoods of the beams with zero
    log_probs = [0.0 for _ in range(beam_size)]
    # Initialize a flag to indicate if all beams are finished
    done = False

    # Encode the source input once
    input_ids = model.tokenizer(text, return_tensors="pt").input_ids.to(utils.Parameters.devices[0])
    final_candidates = []

    # Loop until all beams are finished or the maximum length is reached
    for _ in range(max_length):
        if done:
            break

        # Store the candidates for the next step
        candidates = []
        # Loop over each beam
        for i, beam in enumerate(beams):
            # Check if the beam is already finished
            if len(beam) > 0:
                if beam[-1] == model.model.config.eos_token_id:
                    # If so, add it to the candidates with its log-likelihood and continue
                    if beam in final_candidates:
                        pass
                    else:
                        final_candidates.append((beam, log_probs[i]))
                        beam_size -= 1
                    continue

            logits = _get_next_token_logits(model, input_ids, beam_ids=beam)
            # Get the probabilities from the logits using softmax
            output_probs = torch.softmax(logits, dim=-1)
            # Get the top k probabilities and indices using torch.topk
            topk_probs, topk_indices = torch.topk(output_probs, k=beam_size)
            # Loop over the top k candidates
            for j in range(beam_size):
                # Get the probability and index of the candidate
                prob = topk_probs[j].item()
                index = topk_indices[j].item()
                # Extend the current beam with the candidate index
                extended_beam = beam + [index]
                # Update the log-likelihood of the extended beam by adding the log-probability of the candidate
                extended_log_prob = log_probs[i] + np.log(prob)
                # Add the extended beam and its log-likelihood to the candidates
                candidates.append((extended_beam, extended_log_prob))
        # Sort the candidates by their log-likelihoods in descending order
        candidates.sort(key=lambda x: x[1] / len(x[0]), reverse=True)
        # Keep only the top k candidates as the new beams
        unique_candidates = []
        for cand in candidates:
            if cand in unique_candidates:
                pass
            else:
                unique_candidates.append(cand)
        candidates = unique_candidates
        candidates = candidates[:beam_size]
        beams = [candidate[0] for candidate in candidates]
        # Keep only the log-likelihoods of the new beams
        log_probs = [candidate[1] for candidate in candidates]
        # Check if all new beams are finished
        done = all(beam[-1] == 1 for beam in beams)

    unique_candidates = []
    candidates = candidates + final_candidates
    for cand in candidates:
        if cand in unique_candidates:
            pass
        else:
            unique_candidates.append(cand)
    candidates = unique_candidates
    candidates.sort(key=lambda x: x[1] / len(x[0]), reverse=True)
    beams = [cand[0] for cand in candidates]
    scores = [candidates[i][1]/len(candidates[i][0]) for i in range(len(candidates))]
    return beams, scores


class Scorer:
    def score(self, text, claim):
        raise NotImplementedError

    def score_parse(self, text, beam_text, separator="|"):
        if separator in beam_text:
            claim = beam_text.split(separator)[1]
        else:
            claim = beam_text
        return self.score(text, claim)

    def __call__(self, text, beam_text):
        return self.score_parse(text, beam_text)


class PenalizeIdentity(Scorer):
    def __init__(self):
        Scorer.__init__(self)

    def score(self, text, claim):
        incorrect_claim = text.split("Claim: ")[1].strip().lower()
        claim = claim.strip().lower()
        if incorrect_claim == claim:
            return 0.10
        else:
            return 0.90


class PenalizeSimilarity(Scorer):
    def __init__(self):
        self.model = BARTClassification(path="models/bart_difference")
        self.model.model.eval()
        Scorer.__init__(self)

    def score(self, text, claim):
        incorrect_claim = text.split("Claim: ")[1].strip().lower()
        claim = claim.strip().lower()
        inp = f"Claim 1: {claim}\nClaim 2: {incorrect_claim}"
        with torch.set_grad_enabled(False):
            to_ret = self.model(inp).cpu().numpy()
        return to_ret


class PenalizeExplanationSimilarity(Scorer):
    def __init__(self):
        self.model = BARTClassification(path="models/bart_difference")
        self.model.model.eval()
        Scorer.__init__(self)

    def score(self, text, claim):
        incorrect_claim = text.split("Claim: ")[1].strip().lower()
        claim = claim.strip().lower()
        inp = f"Claim 1: {claim}\nClaim 2: {incorrect_claim}"
        with torch.set_grad_enabled(False):
            to_ret = self.model(inp).cpu().numpy()
        return to_ret


def score_guided_decode(model, text, scorer, beam_size=5, max_length=200, expected_separator=None):
    # scorer takes in an input sentence (in text form) and outputs a score between 0 and 1 with 1 being good.
    do_scoring = True
    beams = [[] for _ in range(beam_size)]
    # Initialize the log-likelihoods of the beams with zero
    log_probs = [0.0 for _ in range(beam_size)]
    # Initialize a flag to indicate if all beams are finished
    done = False
    if expected_separator is not None:
        expected_separator = model.tokenizer.encode(expected_separator)[1]


    # Encode the source input once
    input_ids = model.tokenizer(text, return_tensors="pt").input_ids.to(utils.Parameters.devices[0])
    final_candidates = []

    # Loop until all beams are finished or the maximum length is reached
    for _ in range(max_length):
        do_scoring_next = False
        if done:
            break

        # Store the candidates for the next step
        candidates = []
        # Loop over each beam
        for i, beam in enumerate(beams):
            # Check if the beam is already finished
            if len(beam) > 0:
                if beam[-1] == model.model.config.eos_token_id:
                    # If so, add it to the candidates with its log-likelihood and continue
                    if beam in final_candidates:
                        pass
                    else:
                        final_candidates.append((beam, log_probs[i]))
                        beam_size -= 1
                    continue

            logits = _get_next_token_logits(model, input_ids, beam_ids=beam)
            # Get the probabilities from the logits using softmax
            output_probs = torch.softmax(logits, dim=-1)
            # Get the top k probabilities and indices using torch.topk
            topk_probs, topk_indices = torch.topk(output_probs, k=beam_size)
            # Loop over the top k candidates
            for j in range(beam_size):
                # Get the probability and index of the candidate
                prob = topk_probs[j].item()
                index = topk_indices[j].item()
                # Extend the current beam with the candidate index
                extended_beam = beam + [index]
                # Update the log-likelihood of the extended beam by adding the log-probability of the candidate
                if do_scoring:
                    if expected_separator is None or expected_separator in extended_beam:
                        lookahead_beam = greedy_decode(model, text, max_length=max_length, partial_output=extended_beam, input_ids=input_ids)
                        lookahead_beam_text = model.tokenizer.decode(lookahead_beam, skip_special_tokens=True)
                        lookahead_score = scorer(text, lookahead_beam_text)
                    else:
                        lookahead_score = 0.5
                        do_scoring_next = True
                    #  print(f"On token {_}, beam {j}: lookahead: {lookahead_beam_text}, score: {lookahead_score}")
                    if lookahead_score < 0.85:
                        #  print(f"Bad Option")
                        do_scoring_next = True
                else:
                    lookahead_score = 0.5
                extended_log_prob = log_probs[i] + np.log(prob) + np.log(lookahead_score)
                # Add the extended beam and its log-likelihood to the candidates
                candidates.append((extended_beam, extended_log_prob))
        # Sort the candidates by their log-likelihoods in descending order
        candidates.sort(key=lambda x: x[1] / len(x[0]), reverse=True)
        # Keep only the top k candidates as the new beams
        unique_candidates = []
        for cand in candidates:
            if cand in unique_candidates:
                pass
            else:
                unique_candidates.append(cand)
        candidates = unique_candidates
        candidates = candidates[:beam_size]
        beams = [candidate[0] for candidate in candidates]
        # Keep only the log-likelihoods of the new beams
        log_probs = [candidate[1] for candidate in candidates]
        # Check if all new beams are finished
        done = all(beam[-1] == 1 for beam in beams)
        do_scoring = do_scoring_next

    unique_candidates = []
    candidates = candidates + final_candidates
    for cand in candidates:
        if cand in unique_candidates:
            pass
        else:
            unique_candidates.append(cand)
    candidates = unique_candidates
    candidates.sort(key=lambda x: x[1] / len(x[0]), reverse=True)
    beams = [cand[0] for cand in candidates]
    scores = [candidates[i][1]/len(candidates[i][0]) for i in range(len(candidates))]
    return beams, scores


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("correction-dataset/correction_gpt_short_test.csv")
    model = get_bart_cc()

    text = 'Evidence: Mammalian CLASPs are required for mitotic spindle organization and kinetochore alignment CLASP1 and CLASP2 are homologous mammalian proteins, which associate with the ends of growing microtubules, as well as the cell cortex and the kinetochores of mitotic chromosomes. Previous studies have shown that in interphase cells CLASPs can attach microtubule plus ends to the cortex and stabilize them by repeatedly rescuing them from depolymerization. Here we show that CLASP1 and 2 play similar and redundant roles in organizing the mitotic apparatus in HeLa cells. Simultaneous depletion of both CLASPs causes mitotic spindle defects and a significant metaphase delay, which often results in abnormal exit from mitosis. Metaphase delay is associated with decreased kinetochore tension, increased kinetochore oscillations and more rapid microtubule growth. We show that the association of CLASP2 with the kinetochores relies on its C-terminal domain, but is independent of microtubules or association with CLIP-170. We propose that CLASPs exhibit at the kinetochores an activity similar to that at the cortex, providing apparent stabilization of microtubules by locally reducing the amplitude of growth/shortening episodes at the microtubule ends. This local stabilization of microtubules is essential for the formation of normal metaphase spindle, completion of anaphase and cytokinesis. ' \
           'Claim: Disorganized destabilization of kinetochore-microtubule attachments occurs at the prometaphase to metaphase transition.'
    print(text)
    true = 'Coordinated stabilization of kinetochore-microtubule attachments occurs at the prometaphase to metaphase transition.'
    print(f"Correct Answer: {true}")
    print(f"Greedy Decode Results")
    output = greedy_decode(model, text)
    ans = model.tokenizer.decode(output, skip_special_tokens=True)
    print(ans)
    if ans.lower().strip() == true.lower().strip():
        print(f"Greedy Decoding was correct")
    beams, scores = beam_search_decode(model, text)
    print(f"Beam Search Results")
    ans = model.tokenizer.decode(beams[0], skip_special_tokens=True)
    print(ans)
    if ans.lower().strip() == true.lower().strip():
        print(f"Beam Search Decoding was correct")
    dec = input(f"Do Constrained Beam Search? (Y/N)")
    if dec.lower() == "y":
        beam = model(text, mode=PenalizeExplanationSimilarity())
        print(f"Constrained Beam Search Results")
        print(beam)
    else:
        pass



