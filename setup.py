from setuptools import setup, find_packages

setup(
    name='usc.anrg.bnet',
    version='0.0.1',
    license='MIT',
    url='https://github.com/meejah/python-skeleton',

    install_requires=open('requirements.txt').readlines(),
    extras_require=dict(
        dev=open('requirements-dev.txt').readlines()
    ),

    description='XXX Skeleton python project example.',
    long_description=open('README.rst', 'r').read(),
    packages=find_packages(),
    data_files=[('share/checkout', ['README.rst'])],
    entry_points=dict(
        console_scripts=[
            'bnet=bnet.main:main'
        ]
    ),
)