import argparse
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer.trainer import PretrainTrainer
from recbole.utils import init_seed, init_logger
from UniSRec_model.dataset import PretrainUniSRecDataset
from UniSRec_model.dataloader import CustomizedTrainDataLoader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', default=1.2, type=float, help='alpha in WCE')
    parser.add_argument('--beta', default=0.99, type=float)
    parser.add_argument('--loss_type', default='CE', type=str, help='loss type: CE, WCE')
    parser.add_argument('--dataset', default='us', type=str, help='the source market to pretrain, default is us')
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--pretrain_epochs', default=200, type=int, help='pretrain epochs')
    parser.add_argument('--save_step', default=50, type=int, help='save step')
    parser.add_argument('--model_name', type=str, help='Name of the model to initialize (SASRec, S3Rec, CATSR)')
    args = parser.parse_args()
    
    
    config_dict = {}
    config_dict['alpha'] = args.alpha # 1.2 1.4 1.6
    config_dict['beta'] = args.beta # 0.5 1.0 1.5
    config_dict['loss_type'] = args.loss_type
    config_dict['dataset'] = args.dataset
    config_dict['gpu_id'] = args.gpu_id
    config_dict['save_step'] = args.save_step
    config_dict['pretrain_epochs'] = args.pretrain_epochs
    config_dict['checkpoint_dir'] = "saved/"
    config_dict['with_adapter'] = False # pretrian do not need adapter
    
    model_name=args.model_name
    if model_name=='UniSRec':
        from UniSRec_model.UnisRec import UniSRec
        config_file_list = [f'properties/{model_name}.yaml'] + ['properties/UniSRec_pretrain.yaml']
        config = Config(model=UniSRec, config_dict=config_dict, config_file_list=config_file_list)
    else:
        config_file_list = [f'properties/{model_name}.yaml'] + ['properties/market.yaml']
        config = Config(model=model_name, config_dict=config_dict, config_file_list=config_file_list)
    # model_name=args.model_name
    # config_file_list = [f'properties/{model_name}.yaml'] + ['properties/market.yaml']
    init_seed(config['seed'], config['reproducibility'])


    # logger initialization
    init_logger(config)
    logger = getLogger()

    # write config info into log
    logger.info(config)

    # dataset creating and filtering
    
    if config['model_name'] == "UniSRec":
        # dataset filtering
        dataset = PretrainUniSRecDataset(config)
    else:
        dataset = create_dataset(config)
    logger.info(dataset)
    
    # dataset splitting
    if config['model_name'] == "UniSRec":
        pretrain_dataset = dataset.build()[0]
        train_data = CustomizedTrainDataLoader(config, pretrain_dataset, None, shuffle=True)
    else:
        train_data, valid_data, test_data = data_preparation(config, dataset)
    # for batch in train_data:
    #     first_sample = batch[0]  # 假设 batch 是一个元组 (inputs, targets)
    #     print(f'-----------The example of train data :{first_sample}')
    #     break  # 打印完第一个批次后退出循环
    
    # print(config['model_name'])
    if config['model_name'] == "CATSR":
        from CATSR import CATSR
        model = CATSR(config, train_data.dataset).to(config['device'])
    elif config['model_name'] == "SASRec":
        from SASRec import SASRec
        model = SASRec(config, train_data.dataset).to(config['device'])
    elif config['model_name'] == "S3Rec":
        from S3Rec import S3Rec
        model = S3Rec(config, train_data.dataset).to(config['device'])
    elif config['model_name'] == "UniSRec":
        from UniSRec_model.UnisRec import UniSRec
        model = UniSRec(config, train_data.dataset).to(config['device'])
    else:
        raise ValueError("Unknown model name. Available models: SASRec, S3Rec, CATSR, UnisRec")
    
    # trainer loading and initialization
    trainer = PretrainTrainer(config, model)
    trainer.pretrain(train_data, show_progress=False)
    print("pretrain done")