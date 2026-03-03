from setuptools import setup, find_packages

setup(
    name='suiren-models',
    version='0.1.0',
    author='ajy',
    author_email='junyian0827@gmail.com',
    description='A model inference package for Suiren models',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/model-inference-package',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch>=2.0.0',
        'pyyaml>=6.0',
        'numpy>=1.21.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0',
            'pytest-cov>=4.0',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)