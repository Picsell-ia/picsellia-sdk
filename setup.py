from distutils.core import setup
setup(
    name='picsellia',
    packages=['picsellia'],
    version='4.4',

    license='MIT',
    description='Python SDK to make your code Picsell.ia compatible !',
    author='Thibaut Lucas CEO @ Picsell.ia',
    author_email='thibaut@picsellia.com',
    url='https://www.picsellia.com',
    download_url='https://github.com/Picsell-ia/picsellia-sdk/archive/v0.3.tar.gz',
    keywords=['SDK', 'Picsell.ia', 'Computer Vision', 'Deep Learning'],
    install_requires=[
        'opencv-python',
        'requests',
        'Pillow',
        'numpy'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'],
)
