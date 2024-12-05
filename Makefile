train:
	mkdir -p runs
	python shared/coronary.py --path '$(DATASET_PATH)' --epochs 7 --shuffle True --lr 0.01 --batch_size 32 --optimizer "[\"adam\"]" --bce_weight 0.05 --loss_type "[\"dice\"]" --halfres
metrics:
	tensorboard --logdir runs