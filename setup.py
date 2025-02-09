from setuptools import setup, find_packages

setup(
    name="gis_toolbox",
    version="0.1.0",
    author="Francisco Mestres",
    author_email="francisco.mestres@mailbox.tu-dresden.de",
    description="A toolbox for GIS operations including line discretization, distance computation, point classification, and node merging.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/francisco-mestres/gis_toolbox",
    packages=find_packages(),
    install_requires=[
        "geopandas==0.9.0",
        "rasterio==1.2.10",
        "requests==2.32.3",       
        "shapely==1.8.0",
        "tqdm==4.67.1",
        "fiona==1.8.20",
        "numba==0.56.4",
        "cudf==22.06.00",
        "cuspatial==22.06.00",
        "numpy==1.23.5",
        "scikit-learn==1.3.2",
        "scipy==1.10.1",
        "pandas==1.4.4",
        "pyproj==3.2.1",
        "pyarrow==7.0.0",
        "pygeos==0.10.2",
        "pyogrio==0.2.0",
        "folium==0.18.0",
        "matplotlib==3.7.1",     
        "attrs==24.2.0",
        "click==8.1.7",
        "click-plugins==1.1.1",
        "cligj==0.7.2",
        "colorama==0.4.6",
        "contourpy==1.1.1",
        "cycler==0.12.1",
        "fastavro==1.9.5",
        "fastrlock==0.8.2",
        "fonttools==4.53.1",
        "fsspec==2024.10.0",
        "importlib-metadata==8.5.0",
        "importlib-resources==6.4.5",
        "jinja2==3.1.4",
        "joblib==1.4.2",
        "kiwisolver==1.4.5",
        "markupsafe==2.1.5",
        "munch==4.0.0",
        "networkx==3.1",
        "numexpr==2.8.4",
        "packaging==24.2",
        "platformdirs==4.3.6",
        "pooch==1.8.2",
        "protobuf==3.20.3",
        "pytz==2024.2",
        "six==1.16.0",
        "snuggs==1.4.7",
        "threadpoolctl==3.5.0",
        "typing_extensions==4.12.2",
        "zipp==3.21.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
