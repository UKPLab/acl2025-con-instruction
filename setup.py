"""Python setup.py for coninstruction package"""
import io
import os
from setuptools import find_packages, setup


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("acl2025_con_instruction", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


setup(
    name="acl2025_con_instruction",
    url="https://github.com/UKPLab/acl2025-con-instruction",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="jiahuigeng",
    packages=find_packages(exclude=["tests", ".github"]),
    install_requires=read_requirements("requirements.txt"),
    entry_points={
        "console_scripts": ["acl2025_con_instruction = acl2025_con_instruction.__main__:main"]
    },
    extras_require={"test": read_requirements("requirements-dev.txt")},
)
