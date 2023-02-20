import yaml, argparse, os, pickle
def save_dict(fname:str, dic:dict, rewrite: bool = False):
    '''
    Para salvar un diccionario en formato pickle. 
    '''
    isfile = os.path.isfile(fname)
    if isfile:
        if not rewrite:
            print('Este archivo ya existe, para sobreescribirlo usar el argumento rewrite = True.')
            return
    try:
        with open(file = fname, mode = "wb") as archive:
            pickle.dump(file = archive, obj=dic)
        texto = f'Se guardo en: {fname}.'
        if isfile:
            texto += f'Atención: se reescribió el archivo {fname}'
    except:
        print('Algo fallo cuando se guardaba')
    return

def load_dict(fname:str):
    '''
    Para cargar un diccionario en formato pickle. 
    '''
    isfile = os.path.isfile(fname)
    if not isfile:
        print(f'El archivo {fname} no existe')
        return
    try:
        with open(file = fname, mode = "rb") as archive:
            data = pickle.load(file = archive)
        return data
    except:
        print('Algo fallo')
    return

class Parser:
    def __init__(self, configuration:str = None) -> None:

        # Root directory where all yamls parameters are stored #TODO #FIXME 
        self.file_path = 'herramientas/config/parameters/'

        # This is a list with all possible yamls parameters
        self.parameters_list = ', '.join([param.removesuffix('.yaml') for param in os.listdir(self.file_path) if param.endswith('.yaml')])

        # Configuration name and dictionary
        self.configuration = configuration

    def config(self):
        # In case the program is run internally
        if self.configuration:
            # Dump variables from parameter
            dic = Parser.configuration_builder(self.file_path, self.configuration)
            return dic

        # In case the program is run from command prompt
        else:
            # Create a parser from which select a possible yaml parameter
            parser = argparse.ArgumentParser()

            # Create possible arguments
            parser.add_argument(
            "config", 
            help= f"Select parameters for the model: {self.parameters_list}", 
            type = str)

            parser.add_argument("-v", 
            "--verbose", 
            action = "store_true", 
            help = "increase output verbosity")

            args = parser.parse_args()
            
            # If verbose is asked, give more information
            if args.verbose:
                print(f"The selected parameter is {args.config}.")
            else:
                print(args.config)
            
            # Dump variables from parameter
            dic = Parser.configuration_builder(self.file_path, args.config)
            return dic
    
    @staticmethod
    def configuration_builder(filepath, config):
        '''
        Loads yaml file
        '''
        file_descriptor = open(filepath + f'{config}'+ '.yaml', "r") 
        data = yaml.load(file_descriptor, Loader = yaml.Loader)
        file_descriptor.close()
        return data

    @staticmethod
    def yaml_dump(filepath, data):
        '''
        Dumps into yaml file
        '''
        file_descriptor = open(filepath, "r") 
        data = yaml.dump(file_descriptor, Dumper = yaml.Dumper)
        file_descriptor.close()
        return data

if __name__ == '__main__':
    p = Parser('default')
    p.config()