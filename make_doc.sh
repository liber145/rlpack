parentdir="$(dirname "$PWD")"
rm -rf doc
sphinx-apidoc  -f -e -M -F -a --ext-autodoc --ext-doctest --ext-intersphinx --ext-todo --ext-coverage --ext-imgmath --ext-viewcode -o  doc ./
sed -i "N;24asys.path.insert\(0, \'$parentdir\'\)" doc/conf.py
sed -i "s/alabaster/sphinx_rtd_theme/g" doc/conf.py
cd doc && make html
