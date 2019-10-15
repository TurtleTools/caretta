from distutils.core import setup


def main():
      setup(name='caretta',
            version='1.0',
            authors=["Janani Durairaj", "Mehmet Akdel"],
            packages=["caretta"],
            install_requires=["numpy", "numba", "prody", "biopython", "fire"])


if __name__ == '__main__':
      main()
