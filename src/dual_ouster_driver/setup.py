from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'dual_ouster_driver'

setup(
    name=package_name,
    version='0.0.1',

    packages=find_packages(
        exclude=['test']
    ),

    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
        (
            os.path.join(
                'share',
                package_name,
                'launch'
            ),
            glob('launch/*.py')
        ),
        (
            os.path.join(
                'share',
                package_name,
                'config'
            ),
            glob('config/*.yaml')
        ),
        (
            os.path.join(
                'share',
                package_name,
                'rviz'
            ),
            glob('rviz/*.rviz')
        ),
    ],

    install_requires=[
        'setuptools'
    ],

    zip_safe=True,

    maintainer='drone',
    maintainer_email='drone@example.com',

    description='Dual Ouster OS1-128 driver launch',

    license='Apache-2.0',
)
