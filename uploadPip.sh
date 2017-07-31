#!/bin/bash

# First change manually the version
version="$(python setup.py --version)"
echo "Current version $version"
read -p "Enter new version:"  newVersion
sed -i ".backup" "s/$version/$newVersion/g" setup.py
git tag "$newVersion" -m "from $version to $newVersion"
git push --tags origin master

python2 setup.py sdist upload
#twine upload "dist/*$newVersion*"
