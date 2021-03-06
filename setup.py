from distutils.core import setup
from version import get_git_version

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# setup(
#     name='pynacolada',
#     version='0.2.21',
#     author='H. Wouters',
#     author_email='hendrikwout@gmail.com',
#     packages=['pynacolada'],
#     url='http://www.nowebsite.com',
#     license='LICENSE.txt',
#     description='easy and customized Processing huge amounts of gridded Climate Data.',
#     long_description=open('README.md').read(),
# )

setup(
    name = 'pynacolada',
    packages = ['pynacolada'],
    version = get_git_version(), #'version number',  # Ideally should be same as your GitHub release tag varsion
    license='LICENSE.txt',
    description = 'description',
    author = 'hendrikwout',
    author_email = 'hendrikwout@gmail.com',
    # url = 'github package source url',
    # download_url = 'download link you saved',
    keywords = ['xarrays', 'climate data processing'],
    classifiers = [],
    long_description=long_description,
    #long_description_content_type= 'text/markdown',
    #long_description=open('README.md').read(),
)

