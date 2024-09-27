dataset="zju"
profile="zju_30s"
logbase=${profile}

for seq in  "my_377"
do
    python solver.py --profile ./profiles/zju/${profile}.yaml --dataset $dataset --seq $seq --logbase $logbase --fast --no_eval
done