import argparse
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer.trainer import PretrainTrainer
from recbole.utils import init_seed, init_logger
import torch



from UniSRec_model.dataset import PretrainUniSRecDataset
from UniSRec_model.dataloader import CustomizedTrainDataLoader
import torch.multiprocessing as mp

def run_pretrain(rank, *args):
    # init random seed
    model_name = args[0]
    config_dict = args[1]
    config_file_list = args[2]
    config_dict['local_rank'] = rank
    
    if model_name == "UniSRec":
        from UniSRec_model.UnisRec import UniSRec
        config = Config(model=UniSRec, config_dict=config_dict, config_file_list=config_file_list)
    else:
        config = Config(model=model_name, config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    if rank == 0:
        logger.info(config)
    
    if config['model_name'] == "UniSRec":
        # dataset filtering
        dataset = PretrainUniSRecDataset(config)
    else:
        dataset = create_dataset(config)
    if rank == 0:
        logger.info(dataset)
    
    if config['model_name'] == "UniSRec":
        pretrain_dataset = dataset.build()[0]
        train_data = CustomizedTrainDataLoader(config, pretrain_dataset, None, shuffle=True)
    else:
        # dataset splitting
        train_data, valid_data, test_data = data_preparation(config, dataset)

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
    logger.info(model)
    if config['model_name'] == "UniSRec":
        weight_path = f'saved/{model_name}-start-us-100.pth'
        checkpoint = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    # trainer loading and initialization
    trainer = PretrainTrainer(config, model)

    trainer.pretrain(train_data, show_progress=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', default=1.2, type=float, help='alpha in WCE')
    parser.add_argument('--beta', default=0.99, type=float)
    parser.add_argument('--loss_type', default='CE', type=str, help='loss type: CE, WCE')
    parser.add_argument('--dataset', default='us', type=str, help='the source market to pretrain, default is us')
    parser.add_argument('--gpu_id', default='4,5,6,7', type=str)
    parser.add_argument('--pretrain_epochs', default=200, type=int, help='pretrain epochs')
    parser.add_argument('--save_step', default=50, type=int, help='save step')
    parser.add_argument('--model_name', type=str, help='Name of the model to initialize (SASRec, S3Rec, CATSR)')

    parser.add_argument('--ip', default='127.0.0.1', type=str)
    parser.add_argument('--port', default=6006, type=int)
    parser.add_argument('--world_size', default=4, type=int)
    parser.add_argument('--nproc', default=4, type=int)
    parser.add_argument('--group_offset', default=0, type=int)
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

    config_dict['ip'] = args.ip
    config_dict['port'] = args.port
    config_dict['world_size'] = args.world_size
    config_dict['nproc'] = args.nproc
    config_dict['offset'] = args.group_offset
    
    model_name=args.model_name
    if model_name=='UniSRec':
        config_file_list = [f'properties/{model_name}.yaml'] + ['properties/UniSRec_pretrain.yaml']
    else:
        config_file_list = [f'properties/{model_name}.yaml'] + ['properties/market.yaml']
    mp.spawn(
        run_pretrain,
        args=(args.model_name, config_dict, config_file_list),
        nprocs=args.nproc,
        join=True,
    )

    # dataset creating and filtering
    print("pretrain done")