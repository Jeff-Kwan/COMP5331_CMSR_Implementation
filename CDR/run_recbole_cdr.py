# @Time   : 2022/3/11
# @Author : Zihan Lin
# @Email  : zhlin@ruc.edu.cn

import argparse
from logging import getLogger

from recbole.utils import init_logger, init_seed, set_color

from recbole_cdr.config import CDRConfig
from recbole_cdr.data import create_dataset, data_preparation
from recbole_cdr.utils import get_model, get_trainer

def run_recbole_cdr(model=None, config_file_list=None, config_dict=None, saved=True):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    config = CDRConfig(model=model, config_file_list=config_file_list, config_dict=config_dict)

    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)
    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)
    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='DCDCSR', help='name of models', choices=['DCDCSR', 'NATR', 'SSCDR'])
    parser.add_argument('--source-data', type=str, default='us', help='source data')
    parser.add_argument('--target-data', type=str, default='de', help='target data')

    args, _ = parser.parse_known_args()

    config_file_list = ["properties/sample.yaml"]
    config_file_list = [f'properties/{args.model}.yaml'] + config_file_list
    
    config_dict = {}
    config_dict['source_domain'] = {'dataset': args.source_data + '_raw'}
    config_dict['target_domain'] = {'dataset': args.target_data + '_raw'}

    run_recbole_cdr(model=args.model, config_file_list=config_file_list, config_dict=config_dict)
