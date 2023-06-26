# !/bin/bash
# default arguments
t5_model="False"
task="verification"


# parse optional arguments
while [[ $# -gt 0 ]]
do
  key="$1"
  case $key in
    --t5_model)
      t5_model="True"
      shift # past argument
      ;;
    --difference)
      task="difference"
      shift # past argument
      ;;
    --verification)
      task="verification"
      shift # past argument
      ;;
    -h|--help)
      echo "Usage: bash ft_correction.sh [options]"
      echo "Options:"
      echo "--t5_model : add t5_model flag to use T5 base model, else BART is used. "
      echo "--difference : set task to difference"
      echo "--verification : set task to verification"
      echo "-h|--help : show this help message and exit"
      exit 0
      ;;
    *)
      echo "Unknown argument: $key"
      exit 1
      ;;
  esac
done


if [[ "$t5_model" == "False" ]]; then
  base_model_path="AutonLabTruth/bart-base-mlm-subewordmask-no-line-by-line-matrix"
  output_dir="models/bart_"$task
else
  base_model_path="AutonLabTruth/t5-base-mlm-subwordmask-matrix"
  output_dir="models/t5_"$task
fi


train_file="correction-dataset/"$task"_train.csv"
test_file="correction-dataset/"$task"_test.csv"
val_file="correction-dataset/"$task"_test.csv"



# call python script with arguments
python finetune_classification.py --model_name_or_path "$base_model_path" --output_dir "$output_dir" --overwrite_output_dir --train_file "$train_file" --validation_file "$val_file" --do_train  --per_device_train_batch_size 4 --num_train_epochs 10 --save_total_limit 2