from distutils.core import setup
setup(
  name = 'picsellia',         # How you named your package folder (MyLib)
  packages = ['picsellia'],   # Chose the same as "name"
  version = '2.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Python SDK to make your code Picsell.ia compatible !',   # Give a short description about your library
  author = 'Thibaut Lucas CEO @ Picsell.ia',                   # Type in your name
  author_email = 'thibaut@picsellia.com',      # Type in your E-Mail
  url = 'https://www.picsellia.com',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/Picsell-ia/picsellia-sdk/archive/v0.3.tar.gz',    # I explain this later on
  keywords = ['SDK', 'Picsell.ia', 'Computer Vision', 'Deep Learing'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'opencv-python',
          'requests',
          'Pillow',
          'numpy',
          'zipfile36',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
