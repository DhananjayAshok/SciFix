#python data.py
#bash scripts/run_basic_correction.sh
#bash scripts/run_all_correction.sh
#bash scripts/run_all_difference.sh
#bash scripts/run_all_verification.sh
python eval.py
python prompting.py
cd ../VENCE
python main/main.py --test_file ../TextCorrectionLLMs/correction-dataset/correction_gpt_short_bal_test.csv
python main/main.py --test_file ../TextCorrectionLLMs/correction-dataset/final_test_formatted.csv