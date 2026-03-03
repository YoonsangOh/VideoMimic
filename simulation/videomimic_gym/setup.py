from setuptools import find_packages
from distutils.core import setup

setup(name='videomimic_gym',
      version='0.0.1',
      author='Arthur Allshire',
      license="MIT",
      packages=find_packages(),
      author_email='arthur@allshire.org',
      description='VideoMimic Gym',
      install_requires=['rsl-rl', 'matplotlib', 'tensorboard', 'mujoco==3.2.3', 'pyyaml', 'plotly', 'wandb', 'trimesh', 'numpy==1.24.4', 'hydra-core', 'warp-lang', 'pyyaml', 'tqdm', 'yourdfpy', 'tensorboard==2.11.0', 'viser', 'pyliblzfse', 'robot_descriptions', 'h5py', 'dm_control', 'rtree'])
      # Note: isaacgym is not available on PyPI and must be installed separately
      # Install it with: cd /path/to/isaacgym/python && pip install -e .
