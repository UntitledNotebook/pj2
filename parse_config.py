import os
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
from logger import setup_logging
from utils import read_json, write_json
from typing import Optional, Dict, Any
import torchvision.transforms as T

class ConfigParser:
    def __init__(self, 
                 config: Dict[str, Any], 
                 resume: Optional[str] = None, 
                 modification: Optional[Dict[str, Any]] = None, 
                 run_id: Optional[str] = None):
        """
        Class to parse configuration JSON file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.
        
        :param config: Dict containing configurations, hyperparameters for training. Contents of `config.json` file for example.
        :param resume: Optional string, path to the checkpoint being loaded.
        :param modification: Optional dict keychain:value, specifying position values to be replaced from config dict.
        :param run_id: Optional unique identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default.
        """
        # load config file and apply modification
        self._config = _update_config(config, modification)
        self.resume = resume

        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config['trainer']['save_dir'])

        exper_name = self.config['name']
        if run_id is None:  # use timestamp as default run-id
            run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        self._save_dir = save_dir / 'models' / exper_name / run_id
        self._log_dir = save_dir / 'log' / exper_name / run_id

        # make directory for saving checkpoints and logs.
        exist_ok = run_id == ''
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

        # save updated config file to the checkpoint dir
        write_json(self.config, self.save_dir / 'config.json')

        # configure logging module
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    @classmethod
    def from_args(cls, args, options: str = '') -> 'ConfigParser':
        """
        Initialize this class from some CLI arguments. Used in train, test.
        
        :param args: Command-line arguments.
        :param options: Additional options for parsing.
        :return: ConfigParser instance.
        """
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()

        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / 'config.json'
        else:
            msg_no_cfg = "Configuration file needs to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_fname = Path(args.config)
        
        config = read_json(cfg_fname)
        if args.config and resume:
            # update new config for fine-tuning
            config.update(read_json(args.config))

        # parse custom CLI options into dictionary
        modification = {opt.target : getattr(args, _get_opt_name(opt.flags)) for opt in options}
        return cls(config, resume, modification)
    
    def parse_transforms(self, transform_list: list) -> T.Compose:
        """
        Parse a list of transformations from the config and return a Compose object.

        :param transform_list: List of dictionaries defining the transformations.
        :return: A Compose object containing the parsed transformations.
        """
        transform_ops = []
        
        for transform in transform_list:
            transform_type = transform['type']
            transform_args = dict(transform['args'])
            
            # Dynamically create the transform operation
            if transform_type == 'RandomHorizontalFlip':
                transform_ops.append(T.RandomHorizontalFlip(**transform_args))
            elif transform_type == 'RandomCrop':
                transform_ops.append(T.RandomCrop(**transform_args))
            elif transform_type == 'ToTensor':
                transform_ops.append(T.ToTensor())
            elif transform_type == 'Normalize':
                transform_ops.append(T.Normalize(**transform_args))
            else:
                raise ValueError(f"Unsupported transform type: {transform_type}")
        
        return T.Compose(transform_ops)

    def init_obj(self, 
                 name: str, 
                 module: Any, 
                 *args: Any, 
                 **kwargs: Any) -> Any:
        """
        Finds a function handle with the name given as 'type' in config, and returns the instance initialized with corresponding arguments.

        `object = config.init_obj('name', module, a, b=1)` is equivalent to `object = module.name(a, b=1)`.
        
        :param name: Name of the module or object in config.
        :param module: The module to be used for object creation.
        :param args: Positional arguments for the object.
        :param kwargs: Keyword arguments for the object.
        :return: Initialized object.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name: str, module: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Finds a function handle with the name given as 'type' in config, and returns the function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)` is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.

        :param name: Name of the function in config.
        :param module: The module where the function is located.
        :param args: Positional arguments.
        :param kwargs: Keyword arguments.
        :return: Partial function.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name: str) -> Any:
        """Access items like an ordinary dict."""
        return self.config[name]

    def get_logger(self, name: str, verbosity: int = 2) -> logging.Logger:
        """
        Get a logger with the specified verbosity level.
        
        :param name: The name of the logger.
        :param verbosity: Logging verbosity level (0: WARNING, 1: INFO, 2: DEBUG).
        :return: Logger instance.
        """
        msg_verbosity = f'Verbosity option {verbosity} is invalid. Valid options are {list(self.log_levels.keys())}.'
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # Read-only attributes
    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    @property
    def save_dir(self) -> Path:
        return self._save_dir

    @property
    def log_dir(self) -> Path:
        return self._log_dir

# Helper functions to update the config dict with custom CLI options
def _update_config(config: Dict[str, Any], modification: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config

def _get_opt_name(flags: list) -> str:
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')

def _set_by_path(tree: dict, keys: str, value: Any):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value

def _get_by_path(tree: dict, keys: list) -> Any:
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
