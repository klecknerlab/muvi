from setuptools import setup

setup(
    name='muvi',
    version='0.5',
    description='Python-based 3D movie viewing software.  Developed by the MUVI center at UC Merced.',
    url='https://github.com/klecknerlab/muvi',
    author='Dustin Kleckner',
    author_email='dkleckner@ucmerced.edu',
    license='Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)',
    packages=['muvi'],
    install_requires=[ #Many of the packages are not in PyPi, so assume the user knows how to isntall them!
        # 'numpy',
        # 'PyQt5',
    ],
    scripts=['bin/muvi_convert'],
    entry_points={
        'gui_scripts': ['muvi=muvi.view.qtview:qt_viewer']
    },
    zip_safe=False
)
