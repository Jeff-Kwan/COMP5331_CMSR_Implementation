import argparse
import torch
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer.trainer import Trainer
from recbole.utils import init_seed, init_logger, get_trainer, set_color
from CATSR import CATSR

from UniSRec_model.UnisRec import UniSRec
from UniSRec_model.dataset import UniSRecDataset


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
    parser.add_argument('--alpha', default=1.2, type=float, help='alpha in WCE')
    parser.add_argument('--beta', default=0.99, type=float, help='beta in WCE')
    parser.add_argument('--loss_type', default='CE', type=str, help='loss type: CE, WCE')
    parser.add_argument('--dataset', default='ca', type=str, help='the source market to pretrain, default is us')
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--model_name', type=str, help='Name of the model to initialize (SASRec, S3Rec, CATSR)')
    args = parser.parse_args()
    
    
    config_dict = {}
    config_dict['alpha'] = args.alpha # 1.2 1.4 1.6
    config_dict['beta'] = args.beta # 0.5 1.0 1.5
    config_dict['loss_type'] = args.loss_type
    config_dict['dataset'] = args.dataset
    config_dict['gpu_id'] = args.gpu_id
    config_dict['checkpoint_dir'] = "saved/"
    config_dict['with_adapter'] = True # pretrian do not need adapter
    
    # configurations initialization
    model_name=args.model_name
    if model_name=='UniSRec' or model_name=='S3Rec':
        from UniSRec_model.UnisRec import UniSRec
        config_file_list = [f'properties/{model_name}.yaml'] + ['properties/UniSRec_finetune.yaml']
        config = Config(model=UniSRec, config_dict=config_dict, config_file_list=config_file_list)
    else:
        config_file_list = [f'properties/{model_name}.yaml'] + ['properties/market.yaml']
        config = Config(model=model_name, config_dict=config_dict, config_file_list=config_file_list)

    weight_list = ["query", "key"]
    
    # if config['use_bert']:
    #     weight_path = f'saved/{model_name}-bert-us-200.pth'
    # else:
    weight_path = f'saved/{model_name}-us-200.pth'
    # init random seed
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    # write config info into log
    logger.info(config)

    # dataset creating and filtering
    if config['model_name'] == "UniSRec" or config['model_name']=='S3Rec' :
        # dataset filtering
        dataset = UniSRecDataset(config)
    else:
        dataset = create_dataset(config)
    logger.info(dataset)
    
    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    if config['model_name'] == "CATSR":
        from CATSR import CATSR
        weight_dict = load_param(weight_path, weight_list)
        model = CATSR(config, train_data.dataset, weight_dict).to(config['device'])
    elif config['model_name'] == "SASRec":
        from SASRec import SASRec
        weight_dict = load_param(weight_path, weight_list)
        model = SASRec(config, train_data.dataset,weight_list).to(config['device'])
    elif config['model_name'] == "S3Rec":
        from S3Rec import S3Rec
        model = S3Rec(config, train_data.dataset).to(config['device'])
    elif config['model_name'] == "UniSRec":
        from UniSRec_model.UnisRec import UniSRec
        model = UniSRec(config, train_data.dataset).to(config['device'])
    else:
        raise ValueError("Unknown model name. Available models: SASRec, S3Rec, CATSR, UnisRec")
    
    logger.info(model)
    if config['model_name'] =="UniSRec":
        fix_enc=True
        checkpoint = torch.load(weight_path, map_location='cpu')
        logger.info(f'Loading from {weight_path}')
        logger.info(f'Transfer [{checkpoint["config"]["dataset"]}] -> [{dataset}]')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        if fix_enc:
            logger.info(f'Fix encoder parameters.')
            for _ in model.position_embedding.parameters():
                _.requires_grad = False
            for _ in model.trm_encoder.parameters():
                _.requires_grad = False
    logger.info(model)
    
    # trainer loading and initialization
    if config['model_name'] =="CATSR":
        trainer = Trainer(config, model)
    else:
        # trainer loading and initialization
        trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, saved=False, show_progress=False)
    
    # model evaluation
    test_result = trainer.evaluate(test_data,load_best_model=False)
    res_str = str()
    for k, v in test_result.items():
        res_str += str(v) + ' '
    print('CSV_easy_copy_format:\n', res_str)
