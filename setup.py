from setuptools import setup

setup(
    name='muvi',
    version='0.1',
    description='Python-based 3D movie viewing software.  Developed by the MUVI center at UC Merced.',
    # url='http://github.com/',
    author='Dustin Kleckner',
    author_email='dkleckner@ucmerced.edu',
    license='Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)',
    packages=['muvi'],
    install_requires=[
        # 'pytables',
        'numpy',
        'PyQt5',
    ],
    zip_safe=False
)
