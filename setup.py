import setuptools

__packagename__ = 'cutisplit'

def get_version():
    import os, re
    VERSIONFILE = os.path.join(__packagename__, '__init__.py')
    initfile_lines = open(VERSIONFILE, 'rt').readlines()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in initfile_lines:
        mo = re.search(VSRE, line, re.M)
        if mo:
            return mo.group(1)
    raise RuntimeError('Unable to find version string in %s.' % (VERSIONFILE,))

__version__ = get_version()


setuptools.setup(name = __packagename__,
        packages = setuptools.find_packages(), # this must be the same as the name above
        version=__version__,
        description='Package for parsing and transforming photometer raw data.',
        url='https://jugit.fz-juelich.de/IBG-1/m.osthege/cutisplit',
        download_url = 'https://jugit.fz-juelich.de/IBG-1/m.osthege/cutisplit/tarball/%s' % __version__,
        author='Michael Osthege, Laura Helleckes',
        copyright='(c) 2022 Forschungszentrum Jülich GmbH',
        license='(c) 2022 Forschungszentrum Jülich GmbH',
        classifiers= [
            'Programming Language :: Python',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3.8',
            'Intended Audience :: Developers'
        ],
        install_requires=[
            'pandas', 'numpy', 'python-dateutil', 'pymc'
        ]
)

