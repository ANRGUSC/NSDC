from setuptools import setup, find_packages

setup(
    name='usc.anrg.nsdc',
    version='0.0.1',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        "networkx", # 2.5.1
        "numpy", # 1.20.2
        "pandas",
        "matplotlib",
        "pygraphviz",
        "plotly",
        "kaleido",
        "heft @ git+https://github.com/mackncheesiest/heft",
    ]
)
