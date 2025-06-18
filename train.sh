accelerate launch \
  --num_machines 1 \
  --num_processes 1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  train.py config.yaml