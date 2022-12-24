import yaml, argparse, os, datetime

BASE_CONF_PATH = os.path.dirname(os.path.abspath(__file__))

class Parser:
    def __init__(self, configuration:str = None, prediction_month:str = None) -> None:
        # Root directory where all yamls parameters are stored #TODO #FIXME # add 'portolio/app/'
        self.file_path = 'config/parameters/'

        # This is a list with all possible yamls parameters
        self.parameters_list = ', '.join([param.removesuffix('.yaml') for param in os.listdir(self.file_path) if param.endswith('.yaml')])

        # Configuration name and dictionary
        self.configuration = configuration
        self.prediction_month = prediction_month + '-01'
        
        # Today date
        self.date = datetime.datetime.today().replace(day = 1).strftime('%Y-%m-%d')

    def config(self):

        # In case the program is run internally
        if self.configuration:
            # Dump variables from parameter
            dic = Parser.configuration_builder(self.file_path, self.configuration)
            # dic['path_glob'] = BASE_CONF_PATH = os.path.dirname(os.path.abspath(__file__)) #TODO for the future

            if self.prediction_month:
                dic['prediction_month'] = self.prediction_month
            else:
                dic['prediction_month'] = self.date
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

            parser.add_argument(
            "prediction_month", 
            help= f"Select prediction month with the following format (choose first day of the month): {'2022-02'}, if None is passed then it will predict current month.", 
            type = str)

            parser.add_argument("-v", 
            "--verbose", 
            action = "store_true", 
            help = "increase output verbosity")

            args = parser.parse_args()
            
            # If verbose is asked, give more information
            if args.verbose:
                print(f"The selected parameter is {args.config} and the prediction month is {args.prediction_month}.")
            else:
                print(args.config)
            
            # Dump variables from parameter
            dic = Parser.configuration_builder(self.file_path, args.config)
            # dic['path_glob'] = BASE_CONF_PATH = os.path.dirname(os.path.abspath(__file__)) #TODO for the future

            if args.prediction_month:
                dic['prediction_month'] = args.prediction_month
            else:
                dic['prediction_month'] = self.date
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
    p = Parser('default', '2020-12-12')
    p.config()