from setuptools import setup, find_packages
import sys

if sys.platform == "win32":
    install_requires = ['open3d~=0.16.1', 'pymeshlab==2023.12.post1']
elif sys.platform == "darwin":
    install_requires = ['open3d']
else:
    install_requires = ['open3d', 'pymeshlab']
install_requires += ['tqdm', 'trimesh', 'rtree', 'mitsuba', 'pytorch3d', 'pytz']

setup(
    name="virtual_scanner",
    version="0.2.1",
    author="icy",
    author_email="i@icys.top",
    description="A virtual scanner using open3d",
    packages=find_packages(),
    install_requires=install_requires,
    python_requires='>=3.8',
)