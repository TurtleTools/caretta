from distutils.core import setup

# from numba.pycc import CC

# numba_cc = CC('caretta')
# numba_cc.verbose = True


# def main():
setup(name='caretta',
      version='1.0',
      authors=["Janani Durairaj", "Mehmet Akdel"],
      packages=["caretta"],
      install_requires=["numpy", "numba", "prody", "biopython", "fire"])
# ext_modules=[numba_cc.distutils_extension()])


# if __name__ == '__main__':
#       main()
