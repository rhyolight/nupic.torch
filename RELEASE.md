
# Release 

- Update version available via `nupic.torch.__version__`
- `git tag $VERSION`
- `git push --tags upstream`
- CircleCI does the rest, new update should be available via `pip install nupic.torch`


- Release via [CircleCI](https://circleci.com/blog/continuously-deploying-python-packages-to-pypi-with-circleci/)
- deploy to https://pypi.org/project/nupic.torch/
- `pip install nupic.torch`
- release `0.0.1`
- roll to `0.0.2.dev0`