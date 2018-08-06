import setuptools
from submodel_checkpoint import name

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name=name,
    version='1.0.0',
    author='Ezekiel Victor',
    author_email='zekevictor@gmail.com',
    description='An adapter callback for Keras ModelCheckpoint that allows checkpointing an alternate model'
                ' (often submodel of a multi-GPU model).',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/TextpertAi/submodel-checkpoint',
    packages=setuptools.find_packages(),
    classifiers=(
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ),
    python_requires='>=3',
    install_requires=[
        'keras'
    ],
)
