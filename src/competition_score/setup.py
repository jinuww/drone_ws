from setuptools import find_packages, setup

package_name = 'competition_score'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
    ],
    install_requires=[
        'setuptools',
    ],
    zip_safe=True,
    maintainer='tuab',
    maintainer_email='tuab@example.com',
    description='Offline trajectory scoring package for the drone competition',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'compet_score = competition_score.compet_score:main',
        ],
    },
)
