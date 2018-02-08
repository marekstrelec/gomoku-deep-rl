#!/bin/bash
if [[ `python3 --version` != *"Python 3.5"* ]]; then
  echo "Python3.5 is required. 'python3' command should default to python 3.5.x"
else
  echo "Creating and activating virtual environment for Python3"
  python3 -m venv venv
  source venv/bin/activate

  echo "Installing all required python packages"
  pip install --upgrade pip
  pip install numpy matplotlib jupyter tqdm ipdb
  pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.5.0-py3-none-any.whl
fi
