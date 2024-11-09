import argparse
import torch
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer.trainer import Trainer
from recbole.utils import init_seed, init_logger
from CATSR import CATSR
from recbole.trainer import HyperTuning
from recbole.utils import init_seed, init_logger, get_model, get_trainer
from recbole.data import create_dataset, data_preparation
from recbole.config import Config
import functools



def objective_function(config_dict=None, config_file_list=None, fixed_config_dict=None):
    # 合并固定参数和超参数
    if fixed_config_dict is not None:
        if config_dict is None:
            config_dict = fixed_config_dict
        else:
            config_dict.update(fixed_config_dict)

    # 创建配置
    config = Config(model=CATSR, config_dict=config_dict, config_file_list=config_file_list)

    # 初始化随机种子和日志
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info('Starting a new trial with parameters: {}'.format(config_dict))

    # 加载预训练权重
    weight_list = ["query", "key"]
    weight_path = config['weight_path']
    weight_dict = load_param(weight_path, weight_list)

    # 创建数据集
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # 初始化模型
    model = CATSR(config, train_data.dataset, weight_dict).to(config['device'])
    logger.info(model)
    # 记录训练开始
    logger.info('Training started.')

    # 记录验证结果


    # 记录测试结果


    # 训练模型
    trainer = Trainer(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, show_progress=False)
    logger.info('Validation result: {}'.format(best_valid_result))

    # 测试模型
    test_result = trainer.evaluate(test_data)
    logger.info('Test result: {}'.format(test_result))
    # 返回结果
    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }

def load_param(path, weight_list):
    checkpoint = torch.load(path)
    weight_dict = dict()
    try:
        q_name = 'trm_encoder.layer.multi_head_attention.query.weight'
        k_name = 'trm_encoder.layer.multi_head_attention.key.weight'
        v_name = 'trm_encoder.layer.multi_head_attention.value.weight'
        for weight in weight_list:
            if weight == 'query':
                weight_dict[weight] = checkpoint['state_dict'][q_name].detach().cpu()
            if weight == 'key':
                weight_dict[weight] = checkpoint['state_dict'][k_name].detach().cpu()
            if weight == 'value':
                weight_dict[weight] = checkpoint['state_dict'][v_name].detach().cpu()
    except KeyError:
        key_list = list(checkpoint['state_dict'].keys())
        max_layer_num = 0
        for k in key_list:
            res = k.split('trm_encoder.layer.')
            if len(res) > 1:
                if int(res[1][0]) > max_layer_num:
                    max_layer_num = int(res[1][0])
        for weight in weight_list:
            name = 'trm_encoder.layer.{}.multi_head_attention.{}.weight'.format(max_layer_num, weight)
            weight_dict[weight] = checkpoint['state_dict'][name].detach().cpu()
    return weight_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', type=str, default='properties/CATSR.yaml properties/market.yaml',
                        help='固定配置文件列表')
    parser.add_argument('--params_file', type=str, default='model.hyper', help='参数文件')
    parser.add_argument('--output_file', type=str, default='hyper_example.result', help='输出结果文件')
    parser.add_argument('--weight_path', type=str, default='saved/CATSR-us-200.pth', help='预训练权重路径')
    parser.add_argument('--tool', type=str, default='Hyperopt', choices=['Hyperopt', 'Ray'], help='选择调优工具')
    args, _ = parser.parse_known_args()

    # 定义固定的参数字典
    fixed_config_dict = {'weight_path': args.weight_path}

    # 固定配置文件列表
    config_file_list = args.config_files.strip().split(' ')

    # 创建带有固定参数的部分函数
    objective_function = functools.partial(objective_function, fixed_config_dict=fixed_config_dict)

    # 初始化 HyperTuning 实例（移除 config_dict 参数）
    hp = HyperTuning(objective_function=objective_function, algo='exhaustive',
                     params_file=args.params_file, fixed_config_file_list=config_file_list)

    # 运行超参数调优
    hp.run()

    # 导出结果
    hp.export_result(output_file=args.output_file)

    # 打印最佳参数和结果
    print('best params: ', hp.best_params)
    print('best result: ')
    print(hp.params2result[hp.params2str(hp.best_params)])
