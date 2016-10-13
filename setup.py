from setuptools import setup

setup(name='speciesOverlap',
      version='0.0.2',
      description='Compute incidence and abundance based overlaps of species in different ponds in a fast manner using linear algebra. Works with parallel support.',
      url='https://github.com/benmaier/species-overlap',
      author='Benjamin F. Maier',
      author_email='benjaminfrankmaier@gmail.com',
      license='MIT',
      packages=['speciesOverlap'],
      install_requires=[
          'numpy',
          'scipy',
          'bottleneck',
      ],
      dependency_links=[
          ],
      zip_safe=False)
