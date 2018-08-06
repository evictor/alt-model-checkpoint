default:
	pipenv install

test:
	cd alt_model_checkpoint
	pipenv run python3 -m unittest

dist:
	pipenv run python3 setup.py sdist bdist_wheel