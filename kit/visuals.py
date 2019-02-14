import matplotlib.pyplot as plt
import numpy as np
import re, os, pickle


def plot_multiple_models(model_name_list):
    la = [item for item in os.listdir('models/') if 'pkl' in item]
    list_of_models = list(
        set([
            laitem for item in model_name_list for laitem in la
            if item in laitem
        ]))

    fig = plt.figure(figsize=(15, 8))

    for st_ in enumerate(['val_loss', 'val_mcc_k', 'loss', 'mcc_k']):
        plt.subplot(2, 2, st_[0] + 1)
        ############
        for model_ in list_of_models:
            with open('models/' + model_, 'rb') as fp:
                model_data = pickle.load(fp)

            quan_av = []
            for key, split_data in model_data['training'].items():
                quan = []
                for k, v in split_data.items():
                    quan += v['results'][st_[1]]
                quan_av.append([quan])
            thing = np.mean(np.squeeze(np.array(quan_av)), axis=0)
            plt.plot(
                thing,
                label=re.findall("[a-zA-Z0-9]*", model_)[0],
                linewidth=3)
        ##############

        plt.legend()
        plt.title(st_[1])
    plt.show()

    for model_ in list_of_models:
        with open('models/' + model_, 'rb') as fp:
            model_data = pickle.load(fp)

            for k, v in model_data['training']['split_1'].items():
                print(re.findall("[a-zA-Z0-9]*", model_)[0] +
                      ': {}epochs/{} batch size'.format(
                          v['epochs'], v['batch_size']))

            for k, v in model_data['metrics'].items():
                if k == 'roc_curve':
                    pass
                else:
                    print(k, ':', v, end=' ')
            print('\n' + '-' * 100)


def plot_model_trainingdata(model_name):
    la = [item for item in os.listdir('models/') if 'pkl' in item]
    list_of_models = list(
        set([
            laitem for item in [model_name] for laitem in la if item in laitem
        ]))[0]

    with open('models/' + list_of_models, 'rb') as fp:
        model_data = pickle.load(fp)

    fig = plt.figure(figsize=(15, 9))

    for st_ in enumerate(['val_loss', 'val_mcc_k', 'loss', 'mcc_k']):
        plt.subplot(2, 2, st_[0] + 1)
        quan_av = []
        for key, split_data in model_data['training'].items():
            quan = []
            for k, v in split_data.items():
                quan += v['results'][st_[1]]
            quan_av.append([quan])
            plt.plot(quan, '--', label=key, linewidth=2, alpha=0.6)
        thing = np.mean(np.squeeze(np.array(quan_av)), axis=0)
        plt.plot(thing, label='avg', linewidth=2, color='black')
        plt.title(st_[1])
        plt.legend()
    plt.show()

    fig = plt.figure(figsize=(15, 4))

    for st_ in enumerate(['val_loss', 'val_mcc_k', 'loss', 'mcc_k']):
        quan_av = []
        for key, split_data in model_data['training'].items():
            quan = []
            for k, v in split_data.items():
                quan += v['results'][st_[1]]
            quan_av.append([quan])
        thing = np.mean(np.squeeze(np.array(quan_av)), axis=0)
        if 'val' in st_[1]:
            plt.plot(thing, label=st_[1], linewidth=3)
        else:
            plt.plot(thing, '--', label=st_[1], linewidth=3)
        plt.legend()
        plt.title('avgs')
    plt.show()

    ii = 1
    for k, v in model_data['training']['split_1'].items():
        print('Stage ' + str(ii) + ': {} epochs at {} batch size'.format(
            v['epochs'], v['batch_size']))
        ii += 1

    for k, v in model_data['metrics'].items():
        if k == 'roc_curve':
            pass
        else:
            print(k, ':', v, end=' ')
