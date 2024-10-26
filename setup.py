from setuptools import setup, find_packages
import sys

if sys.platform == "win32":
    install_requires = ['open3d<=0.16']
else:
    install_requires = []

setup(
    name="virtual_scanner",
    version="0.1.2",
    author="icy",
    author_email="i@icys.top",
    description="A virtual scanner using open3d",
    packages=find_packages(),
    classifiers=[],
    install_requires=install_requires,
    python_requires='>=3.8',
)