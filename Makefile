ifneq (,$(wildcard .env))
    include .env
    export $(shell sed 's/=.*//' .env)
endif

train:
	mkdir -p runs
	@echo $(DATASET_PATH)
	@echo $(MLFLOW_TRACKING_URI)
	python shared/train.py --path '$(DATASET_PATH)' --epochs "[10]" --shuffle True --lr 0.01 --batch_size [32] --bce_weight "[0.05, 0.1]" --halfres
metrics:
	tensorboard --logdir runs
predict:
	python shared/predict.py --image_path '$(IMAGE_PATH)' --epoch 9
