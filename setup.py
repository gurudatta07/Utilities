from setuptools import find_packages, setup

setup(name='data-science-utils',
      version='1.0.0',
      description='Utils for use in python data-science',
      url='https://github.com/osmtech07/data-science-utils',
      author='Gurudatta',
      author_email='gurudatta07@gmail.com',
      install_requires=[
          'numpy','pandas','beautifulsoup4','fastnumbers','more-itertools',
            'dill','stockstats','pytidylib','seaborn','gensim','nltk','fastnumbers',
            'joblib','Pygments','opencv-python',
      ],
      keywords=['Pandas','numpy','data-science','IPython', 'Jupyter','ML','Machine Learning'],
      packages=find_packages(),
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
