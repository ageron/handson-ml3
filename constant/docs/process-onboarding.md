
Welcome to the team!

# 1

Begin by cloning https://github.com/O1SoftwareNetwork/handson-ml3 ,
perhaps within your IDE or at the command line.

# 2

We use a single python
[conda](https://docs.conda.io/projects/miniconda/en/latest)
environment, based on and expanded from what Aurélien Géron uses.

    cd handson-ml3
    conda env update

# 3

This will read `environment.yml`, download some libraries,
and make them available via this command:

    conda activate homl3

Verify that `$ git status` is clean even after starting up
your IDE, or after `vim` wrote a `script.py.swp` swap file.
Adjust `.gitignore` so you see clean status with no files
awaiting a commit.

# 4

Ask git to run some routine lint checks each time you commit:

    pre-commit install

# 5

Try

    make lint

to verify the homl3 environment can access libraries, such as
[isort](https://pypi.org/project/isort).

That's it, you're done!
