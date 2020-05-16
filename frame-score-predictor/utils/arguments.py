import argparse
from datetime import datetime


def parse_arguments():

    parser = argparse.ArgumentParser(description='Configurations.')

    parser.add_argument(
        '--optimizer',
        '-o',
        default='adam',
        choices=['adam', 'SGD'],
        help='The optimizer for training.')
    parser.add_argument(
        '--lr',
        default=0.0001,
        type=float,
        help='Leaning rate.')
    parser.add_argument(
        '--epochs',
        '-e',
        default=100,
        type=int,
        help='Number of training epochs.')
    parser.add_argument(
        '--batch_size',
        '-b',
        default=100,
        type=int,
        help='Batch size.')
    parser.add_argument(
        '--hospitals',
        nargs='+',
        default=['Brescia', 'Gemelli - Roma', 'Lucca', 'No Covid Data', 'Pavia', 'Germania'],
        help='Name of the hospital to be used.')
    parser.add_argument(
        '--sensors',
        nargs='+',
        default=['linear', 'convex', 'unknown'],
        help='Sensors to be used.')
    parser.add_argument(
        '--num_workers',
        '-w',
        default=5,
        type=int,
        help='Number of workers in data loader')
    parser.add_argument(
        '--shuffle',
        default=False,
        help='Whether to shuffle the training set or not')
    parser.add_argument(
        '--model_name',
        '-n',
        default='covid19',
        type=str,
        help='Name of the model you want to use.')
    parser.add_argument(
        '--raw_dataset_root',
        default='./dataset/covid19',
        type=str,
        help='Root folder for the raw datasets.')
    parser.add_argument(
        '--dataset_root',
        default='./preprocessed',
        type=str,
        help='Root folder for the datasets.')
    parser.add_argument(
        '--device',
        default='cuda',
        type=str,
        help='Device to be used. Default: cuda')
    parser.add_argument(
        '--comment',
        '-c',
        default=datetime.now().strftime('%b%d_%H-%M-%S'),
        type=str,
        help='Comment to be appended to the model name to identify the run')
    parser.add_argument(
        '--keep_last',
        dest='drop_last',
        default=True,
        action='store_false',
        help='Whether to drop the last batch if incomplete or not')
    parser.add_argument(
        '--seed',
        default=None,
        type=int,
        help='Random seed.')
    parser.add_argument(
        '--test_size',
        default=0.3,
        type=float,
        help='Relative size of the test set.')
    parser.add_argument(
        '--split',
        default='patient_hash',
        type=str,
        help='The split strategy.')
    parser.add_argument(
        '--stratify',
        default=None,
        type=str,
        help='The field to stratify by.')
    parser.add_argument(
        '--use_stn',
        default=False,
        action='store_true',
        help='Whether to use a spatial transformer network.')
    parser.add_argument(
        '--log_interval',
        default=10,
        type=int,
        help='Interval for printing')
    parser.add_argument(
        '--pretrained',
        default=False,
        action='store_true',
        help='Whether to use pre-trained network.')
    parser.add_argument(
        '--img_size',
        default=512,
        type=int,
        help='image size.')
    parser.add_argument('--sigma',
        type=float,
        default=0.1,
        help='sigma for affine transformation')
    parser.add_argument('--smoothing',
                        type=float,
                        default=0.,
                        help='co-efficient for label smoothing')
    parser.add_argument(
        '--wrn_depth',
        default=16,
        type=int,
        help='wide resnet depth.')
    parser.add_argument(
        '--wrn_width',
        default=8,
        type=int,
        help='wide resnet width.')
    parser.add_argument('--dropout_rate',
                        type=float,
                        default=0.3,
                        help='dropout rate for wide resnet')
    parser.add_argument(
        '--arch',
        default='CNN2D',
        type=str,
        help='Name of the model you want to use (e.g. CNN2D, SimpleCNN, WideResnet, ResNet50, etc.).')
    parser.add_argument(
        '--mode',
        default='train',
        choices=['train', 'visualize', 'generate_csv', 'generate_dataset', 'bbox_video'])
    parser.add_argument(
        '--wandb_name',
        required=True,
        type=str,
        help='Name of the experiment.')
    parser.add_argument(
        '--whiten',
        default=False,
        action='store_true',
        help='Whether to the network input.')
    parser.add_argument('--lambda_cons',
                        type=float,
                        default=1.,
                        help='weight for consistency reg loss')
    parser.add_argument('--lambda_stn_params',
                        type=float,
                        default=1.,
                        help='weight for stn params')
    parser.add_argument('--weights_path', type=str,
                        help='path to the weights of the model')
    parser.add_argument(
        '--multiplier',
        default=2,
        type=int,
        help='multiplier for sord.')
    parser.add_argument(
        '--img_path',
        type=str)
    parser.add_argument(
        '--constant',
        default=2,
        type=int,
        help='some constant.')

    args = parser.parse_args()
    args.run_name = '-'.join([args.model_name, args.comment])

    return args