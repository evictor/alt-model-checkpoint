default:
	pipenv install

test:
	cd alt_model_checkpoint
	pipenv run python3 -m unittest