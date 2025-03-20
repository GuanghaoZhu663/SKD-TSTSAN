import argparse
from distutils.util import strtobool
from train_classify_SKD_TSTSAN_functions import main_SKD_TSTSAN_with_Aug_with_SKD


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=strtobool, default=True)
    parser.add_argument('--pre_trained', type=strtobool, default=True)
    parser.add_argument('--Aug_COCO_pre_trained', type=strtobool, default=True)
    parser.add_argument('--save_model', type=strtobool, default=True)
    parser.add_argument('--pre_trained_model_path', type=str, default="", help="path to the model weights pre-trained on macro-expression dataset")
    parser.add_argument('--main_path', type=str, default="", help="path to the dataset directory")
    parser.add_argument('--exp_name', type=str, default="", help="name of the folder to save experimental results")
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--max_iter', type=int, default=20000)
    parser.add_argument('--model', type=str, default="SKD_TSTSAN")
    parser.add_argument('--loss_function', type=str, default="FocalLoss_weighted")
    parser.add_argument('--class_num', type=int, default=5)

    parser.add_argument('--temperature', default=3, type=int, help='temperature to smooth the logits')
    parser.add_argument('--alpha', default=0.1, type=float, help='weight of kd loss')
    parser.add_argument('--beta', default=1e-6, type=float, help='weight of feature loss')

    parser.add_argument('--Aug_alpha', type=float, default=2)

    config = parser.parse_args()

    main_SKD_TSTSAN_with_Aug_with_SKD(config)


