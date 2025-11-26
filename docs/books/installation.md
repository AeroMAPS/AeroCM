Installation
------------------

The use of the Python Package Index ([PyPI](https://pypi.org/)) is the simplest method for installing AeroCM.

**Prerequisite**: AeroMAPS needs at least Python 3.10.0.

You can install the latest version with this command:

``` {.bash}
pip install --upgrade aerocm
```

If you also want to run the Jupyter notebooks developed for the reference paper, use the following command:

``` {.bash}
pip install --upgrade aerocm[publications]
```


For developers
------------------

As a developer, the use of poetry is recommended.

You can install the required packages with this command:

``` {.bash}
poetry install
```

If you also want to run the Jupyter notebooks developed for the reference paper, use the following command:

``` {.bash}
poetry install -E publications
```

The use of requirements files is also possible.