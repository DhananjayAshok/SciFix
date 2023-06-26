#bash scripts/generation.sh --train --task difference
#bash scripts/generation.sh --train --task difference --t5_model

bash scripts/classification.sh --difference
bash scripts/classification.sh --t5_model --difference

