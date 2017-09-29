from setuptools import setup

setup(
        name="lowrankautoml",
        version="0.1",
        author="Chengrun Yang, Dae Won Kim, Yuji Akimoto",
        author_email="cy438@cornell.edu",
        packages=["lowrankautoml"],
        package_dir={"lowrankautoml":"lowrankautoml"},
        url="https://github.com/chengrunyang/AutoML/",
        license="MIT",
        install_requires=[  "numpy >= 1.8",
                            "scipy >= 0.13",
			    "sklearn >= 0.18",
			    "pysmac >= 0.9",
			    "pathos >= 0.2.0"]
)