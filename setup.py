from distutils.core import setup

setup(name='joommftools',
      version='0.1',
      description='Visualisation tools for micromagnetism',
      author='Computational Modelling Group',
      author_email='fangohr@soton.ac.uk',
      url='http://github.com/joommf/tools',
      packages=['joommftools'],
      install_requires=[
          'discretisedfield',
          'holoviews',
          'pandas',
      ],
      classifiers=[
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 3',
      ]
)

