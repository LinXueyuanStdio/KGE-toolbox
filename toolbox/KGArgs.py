from argparse import ArgumentParser


class KGEArgParser:
    """The class implements the argument parser for the pykg2vec.

    KGEArgParser defines all the necessary arguments for the global and local
    configuration of all the modules.

    Attributes:
        general_group (object): It parses the general arguements used by most of the modules.
        general_hyper_group (object): It parses the arguments for the hyper-parameter tuning.

    Examples:
        >>> from toolbox.KGArgs import KGEArgParser
        >>> args = KGEArgParser().get_args()
    """

    def __init__(self):
        self.parser = ArgumentParser(description='Knowledge Graph Embedding tunable configs.')

        ''' argument group for hyperparameters '''
        self.general_hyper_group = self.parser.add_argument_group('Generic Hyperparameters')
        self.general_hyper_group.add_argument('-lmda', dest='lmbda', default=0.1, type=float,
                                              help='The lmbda for regularization.')
        self.general_hyper_group.add_argument('-b', dest='batch_size', default=128, type=int,
                                              help='training batch size')
        self.general_hyper_group.add_argument('-mg', dest='margin', default=0.8, type=float,
                                              help='Margin to take')
        self.general_hyper_group.add_argument('-opt', dest='optimizer', default='adam', type=str,
                                              help='optimizer to be used in training.')
        self.general_hyper_group.add_argument('-s', dest='sampling', default='uniform', type=str,
                                              help='strategy to do negative sampling.')
        self.general_hyper_group.add_argument('-ngr', dest='neg_rate', default=1, type=int,
                                              help='The number of negative samples generated per positive one.')
        self.general_hyper_group.add_argument('-l', dest='epochs', default=100, type=int,
                                              help='The total number of Epochs')
        self.general_hyper_group.add_argument('-lr', dest='learning_rate', default=0.01, type=float,
                                              help='learning rate')
        self.general_hyper_group.add_argument('-k', dest='hidden_size', default=50, type=int,
                                              help='Hidden embedding size.')
        self.general_hyper_group.add_argument('-km', dest='ent_hidden_size', default=50, type=int,
                                              help="Hidden embedding size for entities.")
        self.general_hyper_group.add_argument('-kr', dest='rel_hidden_size', default=50, type=int,
                                              help="Hidden embedding size for relations.")
        self.general_hyper_group.add_argument('-k2', dest='hidden_size_1', default=10, type=int,
                                              help="Hidden embedding size for relations.")
        self.general_hyper_group.add_argument('-l1', dest='l1_flag', default=True,
                                              type=lambda x: (str(x).lower() == 'true'),
                                              help='The flag of using L1 or L2 norm.')
        self.general_hyper_group.add_argument('-al', dest='alpha', default=0.1, type=float,
                                              help='The alpha used in self-adversarial negative sampling.')
        self.general_hyper_group.add_argument('-fsize', dest='filter_sizes', default=[1, 2, 3], nargs='+', type=int,
                                              help='Filter sizes to be used in convKB which acts as the widths of the kernals')
        self.general_hyper_group.add_argument('-fnum', dest='num_filters', default=50, type=int,
                                              help='Filter numbers to be used in convKB and InteractE.')
        self.general_hyper_group.add_argument('-fmd', dest='feature_map_dropout', default=0.2, type=float,
                                              help='feature map dropout value used in ConvE and InteractE.')
        self.general_hyper_group.add_argument('-idt', dest='input_dropout', default=0.3, type=float,
                                              help='input dropout value used in ConvE and InteractE.')
        self.general_hyper_group.add_argument('-hdt', dest='hidden_dropout', default=0.3, type=float,
                                              help='hidden dropout value used in ConvE.')
        self.general_hyper_group.add_argument('-hdt1', dest='hidden_dropout1', default=0.4, type=float,
                                              help='hidden dropout value used in TuckER.')
        self.general_hyper_group.add_argument('-hdt2', dest='hidden_dropout2', default=0.5, type=float,
                                              help='hidden dropout value used in TuckER.')
        self.general_hyper_group.add_argument('-lbs', dest='label_smoothing', default=0.1, type=float,
                                              help='The parameter used in label smoothing.')
        self.general_hyper_group.add_argument('-cmax', dest='cmax', default=0.05, type=float,
                                              help='The parameter for clipping values for KG2E.')
        self.general_hyper_group.add_argument('-cmin', dest='cmin', default=5.00, type=float,
                                              help='The parameter for clipping values for KG2E.')
        self.general_hyper_group.add_argument('-fp', dest='feature_permutation', default=1, type=int,
                                              help='The number of feature permutations for InteractE.')
        self.general_hyper_group.add_argument('-rh', dest='reshape_height', default=20, type=int,
                                              help='The height of the reshaped matrix for InteractE.')
        self.general_hyper_group.add_argument('-rw', dest='reshape_width', default=10, type=int,
                                              help='The width of the reshaped matrix for InteractE.')
        self.general_hyper_group.add_argument('-ks', dest='kernel_size', default=9, type=int,
                                              help='The kernel size to use for InteractE.')
        self.general_hyper_group.add_argument('-ic', dest='in_channels', default=9, type=int,
                                              help='The kernel size to use for InteractE.')
        self.general_hyper_group.add_argument('-evd', dest='ent_vec_dim', default=200, type=int, help='.')
        self.general_hyper_group.add_argument('-rvd', dest='rel_vec_dim', default=200, type=int, help='.')

        # basic configs
        self.general_group = self.parser.add_argument_group('Generic')
        self.general_group.add_argument('-mn', dest='model_name', default='TransE', type=str, help='Name of model')
        self.general_group.add_argument('-db', dest='debug', default=False, type=lambda x: (str(x).lower() == 'true'),
                                        help='To use debug mode or not.')
        self.general_group.add_argument('-exp', dest='exp', default=False, type=lambda x: (str(x).lower() == 'true'),
                                        help='Use Experimental setting extracted from original paper. (use Freebase15k by default)')
        self.general_group.add_argument('-ds', dest='dataset_name', default='Freebase15k', type=str,
                                        help='The dataset name (choice: fb15k/wn18/wn18_rr/yago/fb15k_237/ks/nations/umls)')
        self.general_group.add_argument('-dsp', dest='dataset_path', default=None, type=str,
                                        help='The path to custom dataset.')
        self.general_group.add_argument('-ld', dest='load_from_data', default=None, type=str,
                                        help='The path to the pretrained model.')
        self.general_group.add_argument('-sv', dest='save_model', default=True,
                                        type=lambda x: (str(x).lower() == 'true'), help='Save the model!')
        self.general_group.add_argument('-tn', dest='test_num', default=1000, type=int,
                                        help='The total number of test triples')
        self.general_group.add_argument('-ts', dest='test_step', default=10, type=int, help='Test every _ epochs')
        self.general_group.add_argument('-t', dest='tmp', default='../intermediate', type=str,
                                        help='The folder name to store trained parameters.')
        self.general_group.add_argument('-r', dest='result', default='../results', type=str,
                                        help='The folder name to save the results.')
        self.general_group.add_argument('-fig', dest='figures', default='../figures', type=str,
                                        help='The folder name to save the figures.')
        self.general_group.add_argument('-plote', dest='plot_embedding', default=False,
                                        type=lambda x: (str(x).lower() == 'true'), help='Plot the entity only!')
        self.general_group.add_argument('-plot', dest='plot_entity_only', default=False,
                                        type=lambda x: (str(x).lower() == 'true'), help='Plot the entity only!')
        self.general_group.add_argument('-device', dest='device', default='cpu', type=str, choices=['cpu', 'cuda'],
                                        help='Device to run pykg2vec (cpu or cuda).')
        self.general_group.add_argument('-npg', dest='num_process_gen', default=2, type=int,
                                        help='number of processes used in the Generator.')
        self.general_group.add_argument('-hpf', dest='hp_abs_file', default=None, type=str,
                                        help='The path to the hyperparameter configuration YAML file.')
        self.general_group.add_argument('-ssf', dest='ss_abs_file', default=None, type=str,
                                        help='The path to the search space configuration YAML file.')
        self.general_group.add_argument('-mt', dest='max_number_trials', default=100, type=int,
                                        help='The maximum times of trials for bayesian optimizer.')

    def get_args(self, args):
        """This function parses the necessary arguments.

        This function is called to parse all the necessary arguments.

        Returns:
          object: ArgumentParser object.
        """
        return self.parser.parse_args(args)
