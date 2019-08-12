default:
	pipenv install

test-build:
	pipenv install --dev

test: test-build
	cd alt_model_checkpoint
	pipenv run python3 -m unittest

dist-build:
	rm -rf dist/*
	pipenv run python3 setup.py sdist bdist_wheel

dist-upload: dist-build
	twine upload dist/*

dist-test: dist-build
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*
