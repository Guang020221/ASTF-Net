import argparse

parser = argparse.ArgumentParser(description='ASTF-NET')

parser.add_argument('--debug', action='store_true', help='Enables debug mode')

parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--validation', '-v', action='store_true',
                    help='set this option to validate after training')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--train_test', action='store_true',
                    help='test after training each epoch')

# Data specifications
parser.add_argument('--epochs', type=int, default=200,
                    help='training epochs')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--ep', type=int, default=None,
                    help='test epoch')
parser.add_argument('--log_dir', type=str, default='./train_log',
                    help='output event file path')

parser.add_argument('--dir_train', type=str, default='./dataset/CW-MEF/train_data', help='training dataset directory')

parser.add_argument('--loss_log', type=str, default='./output/Loss_log.txt', help='loss_log')
parser.add_argument('--figure_path', type=str, default='./output/train_loss_curve.png', help='figure_path')

# 优化配置
parser.add_argument('--Window_size', type=int, default=4,
                    help='Window_size')
parser.add_argument('--wd', type=float, default=1e-5,
                    help='weight_decay')
parser.add_argument('--dir_val', type=str, default='./dataset/CW-MEF/val_data', help='validation dataset directory')

parser.add_argument('--dir_test', type=str, default='./dataset/CW-MEF/test_data', help='test dataset directory')

parser.add_argument('--model_path', type=str, default='',
                    help='trained model directory')
parser.add_argument('--model', type=str, default='49.pth',
                    help='model name')

parser.add_argument('--GC_model', type=str, default='./model/GCM.pth', help='GC_model name')
parser.add_argument('--Denoise_model', type=str, default='./model/ffdnet_gray.pth', help='Denoise_model name')

parser.add_argument('--ext', type=str, default='.png',
                    help='extension of image files')
parser.add_argument('--scale', type=int, default=4,
                    help='super resolution scale')
parser.add_argument('--batch_size', type=int, default=2,  # 保持 2
                    help='number of batches each time')
parser.add_argument('--patch_size', type=int, default=192,
                    help='input patch size')

parser.add_argument('--save_dir', type=str, default='./output/test_result/', help='test results directory')

# Model specifications
parser.add_argument('--in_channels', type=int, default=1,
                    help='number of input channels')
parser.add_argument('--out_channels', type=int, default=1,
                    help='number of output channels')
parser.add_argument('--num_features', type=int, default=16,
                    help='number of features')
parser.add_argument('--growth', type=int, default=16,
                    help='channel growth in dense blocks')
parser.add_argument('--num_layers', type=int, default=5,
                    help='number of dense layers')
parser.add_argument('--act_type', type=str, default='prelu',
                    help='type of activation function')

parser.add_argument('--eval', action='store_true',
                    help='evaluate the test results')

parser.add_argument('--resume', type=str, default='',
                    help='Path to checkpoint to resume training from. e.g.: --resume ./checkpoints/epoch_100.pth')
args = parser.parse_args()