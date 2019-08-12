import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='alt-model-checkpoint',
    version='1.13.0',
    author='Ezekiel Victor',
    author_email='zekevictor@gmail.com',
    description='An adapter callback for Keras ModelCheckpoint that allows checkpointing an alternate model'
                ' (often submodel of a multi-GPU model).',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/TextpertAi/alt-model-checkpoint',
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
