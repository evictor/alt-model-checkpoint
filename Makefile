default:
	pipenv install

test:
	cd alt_model_checkpoint
	pipenv run python3 -m unittest

dist-build:
	pipenv run python3 setup.py sdist bdist_wheel

dist-upload: dist-build
	twine upload dist/*

dist-test: dist-build
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*
