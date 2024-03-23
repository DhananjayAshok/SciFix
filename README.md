`SciFix` is the official implementation of [The student becomes the master: Outperforming GPT3 on Scientific Factual Error Correction](https://arxiv.org/abs/2305.14707). We release the dataset setup and code to replicate results and apply our method to a new dataset.


## Paper
For more details, refer to the accompanying paper: 
[The student becomes the master: Outperforming GPT3 on Scientific Factual Error Correction](https://arxiv.org/abs/2305.14707). If you have questions, please feel free to reach us at dhananja@andrew.cmu.edu / dhananjay.ashok99@gmail.com or open an issue.  

If you find this repository useful or use this code in your research, please cite the following paper: 

> Ashok, Dhananjay, et al. "The student becomes the master: Outperforming GPT3 on Scientific Factual Error Correction." The 2023 Conference on Empirical Methods in Natural Language Processing. 2023.
```
@inproceedings{ashok2023student,
  title={The student becomes the master: Outperforming GPT3 on Scientific Factual Error Correction},
  author={Ashok, Dhananjay and Kulkarni, Atharva and Pham, Hai and Poczos, Barnabas},
  booktitle={The 2023 Conference on Empirical Methods in Natural Language Processing},
  year={2023}
}
```

### Installation

Set up the Python (3.10) environment through conda using:
```bash
conda env create -f environment.yml
```

## Datasets 
We take the SciFact-Open dataset and place the jsonl files under SciFix/scifact-open/data, similarly, the CovidFact jsonl is stored as SciFix/covidfact/COVIDFACT_dataset.jsonl. See `data.py` for parsing implementation, if you want to implement a new dataset you will be interested in the function `to_common_format`, if you make sure your dataset is eventually in this format and then passed in a list to `get_joined_df` you should not have to make any other changes to the code. 

## Synthetic Dataset Generation
Once in the common format, we can generate the synthetic datasets using `generate_full_dataset` and split them using `generate_splits`. To change the underlying prompt model see `query_models.py` and `dataset_creation.py`. To modify the prompts used see `prompting.py`

## Creating the Error Correction LM and Semantic Difference Model
We use the files `finetune_as_summarization.py` to generate the Seq2Seq Language Model and `finetune_classification.py` to train the semantic difference model

## Inference
See the `query_models.py` file and refer to `get_bart_cc()` as an example of loading a trained Seq2Seq Error Correction Model, to use score-guided decoding see `score_guided_decode`. The main function has an example of these functions being used. 

## License
This repository is licensed under the terms of the [Apache License](LICENSE).
