# !/bin/bash
# default arguments
train="False"
t5_model="False"
task="correction"
evidence_string="_short"
balanced_string=""
split_string=""
explanation_string=""
task_string="claim_correction"


# parse optional arguments
while [[ $# -gt 0 ]]
do
  key="$1"
  case $key in
    --t5_model)
      t5_model="True"
      shift # past argument
      ;;
    --long)
      evidence_string="_long"
      shift # past argument
      ;;
    --split)
      split_string="_split"
      shift # past argument
      ;;
    --balanced)
      balanced_string="_bal"
      shift # past argument
      ;;
    --explanation)
      explanation_string="_exp"
      shift # past argument
      ;;
    --task)
      task="$2" # get the value of the task argument
      shift # past argument
      shift # past value
      ;;
    --train)
      train="True"
      shift # past argument
      ;;
    -h|--help)
      echo "Usage: bash ft_correction.sh [options]"
      echo "Options:"
      echo "--t5_model : add t5_model flag to use T5 base model, else BART is used. "
      echo "--long : add long flag to use long evidence"
      echo "--split : add split flag to drop covid datasets during training"
      echo "--balanced : add balanced flag to balance covidfact to scifact size during training"
      echo "--explanation : add explanation flag to require explanations in output model"
      echo "--task: specify the task to perform. Task must be one of verification, difference or correction."
      echo "--train: add train flag to train model from base checkpoint, else will only predict on test set"
      echo "-h|--help : show this help message and exit"
      exit 0
      ;;
    *)
      echo "Unknown argument: $key"
      exit 1
      ;;
  esac
done


if [[ "$task" == "verification" || "$task" == "difference" || "$task" == "correction" ]]; then
  if [[ "$task" == "verification" ]]; then
    task_string="verification_seq"
    train_file="correction-dataset/verification_seq_train.csv"
    test_file="correction-dataset/verification_seq_test.csv"
    explanation_string=""
    balanced_string=""
  elif [[ "$task" == "difference" ]]; then
    task_string="difference_seq"
    train_file="correction-dataset/difference_seq_train.csv"
    test_file="correction-dataset/difference_seq_test.csv"
    explanation_string=""
    balanced_string=""
  else
    train_file="correction-dataset/correction_gpt"$evidence_string$explanation_string$split_string$balanced_string"_train.csv"
    test_file="correction-dataset/correction_gpt"$evidence_string$explanation_string$split_string$balanced_string"_test.csv"
  fi
  else
    echo "Invalid task: $task. Task must be one of verification, difference or correction."
    exit 1
fi


if [[ "$t5_model" == "False" ]]; then
  base_model_path="AutonLabTruth/bart-base-mlm-subewordmask-no-line-by-line-matrix"
  output_dir="models/bart_"$task_string$explanation_string$split_string$balanced_string
else
  base_model_path="AutonLabTruth/t5-base-mlm-subwordmask-matrix"
  output_dir="models/t5_"$task_string$explanation_string$split_string$balanced_string
fi



# call python script with arguments
if [[ "$train" == "True" ]]; then
  python finetune_as_summarization.py --model_name_or_path "$base_model_path" --output_dir "$output_dir" --overwrite_output_dir --train_file "$train_file" --test_file "$test_file" --do_train  --per_device_train_batch_size 1 --num_train_epochs 10 --do_predict --predict_with_generate --max_predict_samples 5 --save_total_limit 2
else
  python finetune_as_summarization.py --model_name_or_path "$base_model_path" --output_dir "$output_dir" --overwrite_output_dir --train_file "$train_file" --test_file "$test_file" --do_predict --predict_with_generate --max_predict_samples 10
fi
