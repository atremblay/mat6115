from setuptools import find_namespace_packages, setup

setup(
    name="mat6115",
    version="0.0.1",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": ["templating = explorance.core.utils.template_filler:main"]
    },
    install_requires=["poutyne",],
)

