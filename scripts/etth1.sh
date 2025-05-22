
gpu=0
el=3; pl=6; dm=32; df=128; nh=16; plr=5e-4
for s in 2024; do
    python run.py --task_name pretrain --data ETTh1 --gpu $gpu --seed $s \
                --e_layers $el --n_heads $nh --d_model $dm --d_ff $df --patch_len $pl --stride $pl --pretrain_learning_rate $plr 
    
    for l in 96 192 336 720; do
        python run.py --task_name finetune --data ETTh1 --gpu $gpu --pred_len $l --seed $s --freeze_epochs 10 --finetune_epoochs 20 \
        --e_layers $el --n_heads $nh --d_model $dm --d_ff $df --patch_len $pl --stride $pl --pretrain_learning_rate $plr --finetune_learning_rate 5e-4
    done
done
