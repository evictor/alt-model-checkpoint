default:
	pipenv install

test:
	cd submodel_checkpoint
	pipenv run python3 -m unittest