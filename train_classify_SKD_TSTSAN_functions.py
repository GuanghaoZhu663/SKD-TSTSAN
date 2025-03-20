from os import path
import os
import numpy as np
import cv2
import time
from sklearn.metrics import confusion_matrix
from torch.utils.data import TensorDataset, DataLoader
import random
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from collections import OrderedDict
import shutil
import sys
from all_model import *


def reset_weights(m):  # Reset the weights for network to avoid weight leakage
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def confusionMatrix(gt, pred, show=False):
    TN, FP, FN, TP = confusion_matrix(gt, pred).ravel()
    f1_score = (2 * TP) / (2 * TP + FP + FN)
    num_samples = len([x for x in gt if x == 1])
    average_recall = TP / num_samples

    return f1_score, average_recall


def normalize_gray(images):
    images = cv2.normalize(images, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return images


def recognition_evaluation(dataset, final_gt, final_pred, show=False):
    if dataset == "CASME2":
        label_dict = {'happiness': 0, 'surprise': 1, 'disgust': 2, 'repression': 3, 'others': 4}

    f1_list = []
    ar_list = []
    try:
        for emotion, emotion_index in label_dict.items():
            gt_recog = [1 if x == emotion_index else 0 for x in final_gt]
            pred_recog = [1 if x == emotion_index else 0 for x in final_pred]
            try:
                f1_recog, ar_recog = confusionMatrix(gt_recog, pred_recog)
                f1_list.append(f1_recog)
                ar_list.append(ar_recog)
            except Exception as e:
                pass
        UF1 = np.mean(f1_list)
        UAR = np.mean(ar_list)
        return UF1, UAR
    except:
        return '', ''


def extract_prefix(file_name):
    prefixes = ["_1_u", "_2_u", "_1_v", "_2_v", "_apex"]
    for prefix in prefixes:
        if prefix in file_name:
            return file_name.split(prefix)[0]
    return None


def get_folder_all_cases(folder_path):
    unique_prefixes = set()
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".jpg"):
            prefix = extract_prefix(file_name)
            if prefix is not None:
                unique_prefixes.add(prefix)
    unique_prefixes = list(unique_prefixes)
    unique_prefixes.sort()

    return unique_prefixes


class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''

    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss


def get_loss_function(loss_name, weight=None):
    if loss_name == "CELoss":
        return nn.CrossEntropyLoss()
    elif loss_name == "FocalLoss":
        return FocalLoss()
    elif loss_name == "FocalLoss_weighted":
        return FocalLoss(weight=weight)


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def new_kd_loss_function(output, target_output, temperature):
    """Compute kd loss"""
    """
    para: output: middle ouptput logits.
    para: target_output: final output has divided by temperature and softmax.
    """

    output = output / temperature
    output_log_softmax = torch.log_softmax(output, dim=1)
    loss_kd = nn.KLDivLoss(reduction="batchmean")(output_log_softmax, target_output)
    return loss_kd


def feature_loss_function(fea, target_fea):
    loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
    return torch.abs(loss).sum()


CASME2_numbers = [32, 25, 61, 27, 99]

def main_SKD_TSTSAN_with_Aug_with_SKD(config):
    learning_rate = config.learning_rate
    batch_size = config.batch_size

    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if config.loss_function == "FocalLoss_weighted":
        if config.main_path.split("/")[1].split("_")[0] == "CASME2":
            numbers = CASME2_numbers

        sum_reciprocal = sum(1 / num for num in numbers)
        weights = [(1 / num) / sum_reciprocal for num in numbers]

        loss_fn = get_loss_function(config.loss_function, torch.tensor(weights).to(device))
    else:
        loss_fn = get_loss_function(config.loss_function)

    if (config.train):
        if not path.exists('./Experiment_for_recognize/' + config.exp_name):
            os.makedirs('./Experiment_for_recognize/' + config.exp_name)

    current_file = os.path.abspath(__file__)
    shutil.copy(current_file, './Experiment_for_recognize/' + config.exp_name)
    shutil.copy("./all_model.py", './Experiment_for_recognize/' + config.exp_name)

    log_file_path = './Experiment_for_recognize/' + config.exp_name + "/log.txt"
    sys.stdout = Logger(log_file_path)

    total_gt = []
    total_pred = []
    best_total_pred = []
    all_accuracy_dict = {}

    t = time.time()

    main_path = config.main_path
    subName = os.listdir(main_path)


    for n_subName in subName:
        print('Subject:', n_subName)

        X_train = []
        y_train = []

        X_test = []
        y_test = []

        expression = os.listdir(main_path + '/' + n_subName + '/train')
        for n_expression in expression:
            case_list = get_folder_all_cases(main_path + '/' + n_subName + '/train/' + n_expression)

            for case in case_list:
                y_train.append(int(n_expression))

                end_input = []
                large_S = normalize_gray(
                    cv2.imread(main_path + '/' + n_subName + '/train/' + n_expression + '/' + case + "_apex.jpg", 0))
                large_S_onset = normalize_gray(
                    cv2.imread(main_path + '/' + n_subName + '/train/' + n_expression + '/' + case + "_onset.jpg", 0))
                small_S = cv2.resize(large_S, (48, 48))
                small_S_onset = cv2.resize(large_S_onset, (48, 48))
                end_input.append(small_S)
                end_input.append(small_S_onset)

                grid_sizes = [4]
                for grid_size in grid_sizes:
                    height, width = large_S.shape
                    block_height, block_width = height // grid_size, width // grid_size

                    for i in range(grid_size):
                        for j in range(grid_size):
                            block = large_S[i * block_height: (i + 1) * block_height,
                                    j * block_width: (j + 1) * block_width]

                            scaled_block = cv2.resize(block, (48, 48))

                            end_input.append(scaled_block)

                for grid_size in grid_sizes:
                    height, width = large_S.shape
                    block_height, block_width = height // grid_size, width // grid_size

                    for i in range(grid_size):
                        for j in range(grid_size):
                            block = large_S_onset[i * block_height: (i + 1) * block_height,
                                    j * block_width: (j + 1) * block_width]

                            scaled_block = cv2.resize(block, (48, 48))

                            end_input.append(scaled_block)

                end_input.append(normalize_gray(
                    cv2.imread(main_path + '/' + n_subName + '/train/' + n_expression + '/' + case + "_1_u.jpg", 0)))
                end_input.append(normalize_gray(
                    cv2.imread(main_path + '/' + n_subName + '/train/' + n_expression + '/' + case + "_1_v.jpg", 0)))
                end_input.append(normalize_gray(
                    cv2.imread(main_path + '/' + n_subName + '/train/' + n_expression + '/' + case + "_2_u.jpg", 0)))
                end_input.append(normalize_gray(
                    cv2.imread(main_path + '/' + n_subName + '/train/' + n_expression + '/' + case + "_2_v.jpg", 0)))

                end_input = np.stack(end_input, axis=-1)
                X_train.append(end_input)

        expression = os.listdir(main_path + '/' + n_subName + '/test')
        for n_expression in expression:
            case_list = get_folder_all_cases(main_path + '/' + n_subName + '/test/' + n_expression)

            for case in case_list:
                y_test.append(int(n_expression))

                end_input = []
                large_S = normalize_gray(
                    cv2.imread(main_path + '/' + n_subName + '/test/' + n_expression + '/' + case + "_apex.jpg", 0))
                large_S_onset = normalize_gray(
                    cv2.imread(main_path + '/' + n_subName + '/test/' + n_expression + '/' + case + "_onset.jpg", 0))
                small_S = cv2.resize(large_S, (48, 48))
                small_S_onset = cv2.resize(large_S_onset, (48, 48))
                end_input.append(small_S)
                end_input.append(small_S_onset)

                grid_sizes = [4]
                for grid_size in grid_sizes:
                    height, width = large_S.shape
                    block_height, block_width = height // grid_size, width // grid_size

                    for i in range(grid_size):
                        for j in range(grid_size):
                            block = large_S[i * block_height: (i + 1) * block_height,
                                    j * block_width: (j + 1) * block_width]

                            scaled_block = cv2.resize(block, (48, 48))

                            end_input.append(scaled_block)

                for grid_size in grid_sizes:
                    height, width = large_S.shape
                    block_height, block_width = height // grid_size, width // grid_size

                    for i in range(grid_size):
                        for j in range(grid_size):
                            block = large_S_onset[i * block_height: (i + 1) * block_height,
                                    j * block_width: (j + 1) * block_width]

                            scaled_block = cv2.resize(block, (48, 48))

                            end_input.append(scaled_block)

                end_input.append(normalize_gray(
                    cv2.imread(main_path + '/' + n_subName + '/test/' + n_expression + '/' + case + "_1_u.jpg", 0)))
                end_input.append(normalize_gray(
                    cv2.imread(main_path + '/' + n_subName + '/test/' + n_expression + '/' + case + "_1_v.jpg", 0)))
                end_input.append(normalize_gray(
                    cv2.imread(main_path + '/' + n_subName + '/test/' + n_expression + '/' + case + "_2_u.jpg", 0)))
                end_input.append(normalize_gray(
                    cv2.imread(main_path + '/' + n_subName + '/test/' + n_expression + '/' + case + "_2_v.jpg", 0)))

                end_input = np.stack(end_input, axis=-1)
                X_test.append(end_input)

        weight_path = './Experiment_for_recognize/' + config.exp_name + '/' + n_subName + '/' + n_subName + '.pth'
        log_path = './Experiment_for_recognize/' + config.exp_name + '/' + n_subName + '/' + "logs"

        writer = SummaryWriter(log_path)

        model = get_model(config.model, config.class_num, config.Aug_alpha).to(device)


        if (config.train):
            if (config.pre_trained):
                model.apply(reset_weights)
                pre_trained_model = torch.load(config.pre_trained_model_path)
                filtered_dict = OrderedDict((k, v) for k, v in pre_trained_model.items() if (not "fc" in k))
                model.load_state_dict(filtered_dict, strict=False)
            elif (config.Aug_COCO_pre_trained):
                model.apply(reset_weights)
                Aug_weight_path = r"motion_magnification_learning_based_master/magnet.pth"
                Aug_state_dict = gen_state_dict(Aug_weight_path)
                model.Aug_Encoder_L.load_state_dict(Aug_state_dict, strict=False)
                model.Aug_Encoder_S.load_state_dict(Aug_state_dict, strict=False)
                model.Aug_Encoder_T.load_state_dict(Aug_state_dict, strict=False)
                model.Aug_Manipulator_L.load_state_dict(Aug_state_dict, strict=False)
                model.Aug_Manipulator_S.load_state_dict(Aug_state_dict, strict=False)
                model.Aug_Manipulator_T.load_state_dict(Aug_state_dict, strict=False)
            else:
                model.apply(reset_weights)

        else:
            model.load_state_dict(torch.load(weight_path))

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), weight_decay=0.0005)
        X_train = torch.Tensor(X_train).permute(0, 3, 1, 2)
        y_train = torch.Tensor(y_train).to(dtype=torch.long)
        dataset_train = TensorDataset(X_train, y_train)

        def worker_init_fn(worker_id):
            random.seed(seed + worker_id)
            np.random.seed(seed + worker_id)

        train_dl = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                              worker_init_fn=worker_init_fn)

        X_test = torch.Tensor(X_test).permute(0, 3, 1, 2)
        y_test = torch.Tensor(y_test).to(dtype=torch.long)
        dataset_test = TensorDataset(X_test, y_test)
        test_dl = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

        best_accuracy_for_each_subject = 0
        best_each_subject_pred = []

        max_iter = config.max_iter
        iter_num = 0
        epochs = max_iter // len(train_dl) + 1

        for epoch in range(1, epochs + 1):
            if (config.train):
                model.train()
                train_ce_loss = 0.0
                middle_loss1 = 0.0
                middle_loss2 = 0.0
                KL_loss1 = 0.0
                KL_loss2 = 0.0
                L2_loss1 = 0.0
                L2_loss2 = 0.0
                loss_sum = 0.0

                num_train_correct = 0
                num_train_examples = 0

                middle1_num_train_correct = 0
                middle2_num_train_correct = 0

                for batch in train_dl:
                    optimizer.zero_grad()
                    x = batch[0].to(device)
                    y = batch[1].to(device)
                    yhat, AC1_out, AC2_out, final_feature, AC1_feature, AC2_feature = model(x)
                    loss = loss_fn(yhat, y)
                    AC1_loss = loss_fn(AC1_out, y)
                    AC2_loss = loss_fn(AC2_out, y)
                    temperature = config.temperature
                    temp4 = yhat / temperature
                    temp4 = torch.softmax(temp4, dim=1)
                    loss1by4 = new_kd_loss_function(AC1_out, temp4.detach(), temperature) * (temperature ** 2)
                    loss2by4 = new_kd_loss_function(AC2_out, temp4.detach(), temperature) * (temperature ** 2)
                    feature_loss_1 = feature_loss_function(AC1_feature, final_feature.detach())
                    feature_loss_2 = feature_loss_function(AC2_feature, final_feature.detach())

                    total_losses = loss + (1 - config.alpha) * (AC1_loss + AC2_loss) + \
                                   config.alpha * (loss1by4 + loss2by4) + \
                                   config.beta * (feature_loss_1 + feature_loss_2)

                    total_losses.backward()
                    optimizer.step()

                    train_ce_loss += loss.data.item() * x.size(0)
                    middle_loss1 += AC1_loss.data.item() * x.size(0)
                    middle_loss2 += AC2_loss.data.item() * x.size(0)
                    KL_loss1 += loss1by4.data.item() * x.size(0)
                    KL_loss2 += loss2by4.data.item() * x.size(0)
                    L2_loss1 += feature_loss_1.data.item() * x.size(0)
                    L2_loss2 += feature_loss_2.data.item() * x.size(0)
                    loss_sum += total_losses * x.size(0)

                    num_train_correct += (torch.max(yhat, 1)[1] == y).sum().item()
                    num_train_examples += x.shape[0]

                    middle1_num_train_correct += (torch.max(AC1_out, 1)[1] == y).sum().item()
                    middle2_num_train_correct += (torch.max(AC2_out, 1)[1] == y).sum().item()

                    iter_num += 1
                    if iter_num >= max_iter:
                        break

                train_acc = num_train_correct / num_train_examples
                middle1_acc = middle1_num_train_correct / num_train_examples
                middle2_acc = middle2_num_train_correct / num_train_examples

                train_ce_loss = train_ce_loss / len(train_dl.dataset)
                middle_loss1 = middle_loss1 / len(train_dl.dataset)
                middle_loss2 = middle_loss2 / len(train_dl.dataset)
                KL_loss1 = KL_loss1 / len(train_dl.dataset)
                KL_loss2 = KL_loss2 / len(train_dl.dataset)
                L2_loss1 = L2_loss1 / len(train_dl.dataset)
                L2_loss2 = L2_loss2 / len(train_dl.dataset)
                loss_sum = loss_sum / len(train_dl.dataset)

                writer.add_scalar("Train_Acc", train_acc, epoch)
                writer.add_scalar("Middle1_Train_Acc", middle1_acc, epoch)
                writer.add_scalar("Middle2_Train_Acc", middle2_acc, epoch)
                writer.add_scalar("train_ce_loss", train_ce_loss, epoch)
                writer.add_scalar("middle_loss1", middle_loss1, epoch)
                writer.add_scalar("middle_loss2", middle_loss2, epoch)
                writer.add_scalar("KL_loss1", KL_loss1, epoch)
                writer.add_scalar("KL_loss2", KL_loss2, epoch)
                writer.add_scalar("L2_loss1", L2_loss1, epoch)
                writer.add_scalar("L2_loss2", L2_loss2, epoch)
                writer.add_scalar("loss_sum", loss_sum, epoch)

                writer.add_scalar("Aug Factor", model.amp_factor, epoch)

            model.eval()
            num_val_correct = 0

            middle1_num_val_correct = 0
            middle2_num_val_correct = 0

            num_val_examples = 0
            temp_best_each_subject_pred = []
            temp_y = []
            for batch in test_dl:
                x = batch[0].to(device)
                y = batch[1].to(device)
                yhat, AC1_out, AC2_out, final_feature, AC1_feature, AC2_feature = model(x)

                num_val_correct += (torch.max(yhat, 1)[1] == y).sum().item()

                middle1_num_val_correct += (torch.max(AC1_out, 1)[1] == y).sum().item()
                middle2_num_val_correct += (torch.max(AC2_out, 1)[1] == y).sum().item()

                num_val_examples += y.shape[0]
                temp_best_each_subject_pred.extend(torch.max(yhat, 1)[1].tolist())
                temp_y.extend(y.tolist())

            val_acc = num_val_correct / num_val_examples
            middle1_val_acc = middle1_num_val_correct / num_val_examples
            middle2_val_acc = middle2_num_val_correct / num_val_examples

            writer.add_scalar("Val_Acc", val_acc, epoch)
            writer.add_scalar("Middle1_Val_Acc", middle1_val_acc, epoch)
            writer.add_scalar("Middle2_Val_Acc", middle2_val_acc, epoch)
            if best_accuracy_for_each_subject <= val_acc:
                best_accuracy_for_each_subject = val_acc
                best_each_subject_pred = temp_best_each_subject_pred
                if (config.train) and (config.save_model):
                    torch.save(model.state_dict(), weight_path)

            if val_acc == 1:
                break

            if not (config.train):
                break

        print('Best Predicted    :', best_each_subject_pred)
        accuracydict = {}
        accuracydict['pred'] = best_each_subject_pred
        accuracydict['truth'] = temp_y
        all_accuracy_dict[n_subName] = accuracydict

        print('Ground Truth :', temp_y)
        print('Evaluation until this subject: ')
        total_pred.extend(temp_best_each_subject_pred)
        total_gt.extend(temp_y)
        best_total_pred.extend(best_each_subject_pred)
        UF1, UAR = recognition_evaluation(config.main_path.split("/")[1].split("_")[0], total_gt, total_pred, show=True)
        best_UF1, best_UAR = recognition_evaluation(config.main_path.split("/")[1].split("_")[0], total_gt,
                                                    best_total_pred, show=True)
        print('UF1:', round(UF1, 4), '| UAR:', round(UAR, 4))
        print('best UF1:', round(best_UF1, 4), '| best UAR:', round(best_UAR, 4))

    writer.close()
    print('Final Evaluation: ')
    UF1, UAR = recognition_evaluation(config.main_path.split("/")[1].split("_")[0], total_gt, total_pred)
    print('Total Time Taken:', time.time() - t)
    print(all_accuracy_dict)

    sys.stdout.log.close()




