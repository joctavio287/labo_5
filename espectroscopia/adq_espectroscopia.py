import os
from herramientas.config.config_builder import Parser

# Importo los paths
glob_path = os.path.normpath(os.getcwd())
variables = Parser(configuration = 'espectroscopia').config()
#prueba