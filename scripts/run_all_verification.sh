#bash scripts/generation.sh --train --task verification
#bash scripts/generation.sh --train --task verification --t5_model

bash scripts/classification.sh --verification
bash scripts/classification.sh --t5_model --verification