import datetime
import os

def save_result(args, accuracy, save_path, total_time, read_time=0.0, latency_times=[], training_time=0.0, opt_times=[], aggr_times=[],\
                append_time=[], read_consume=0.0, opt_consume=[], weights_consume=[]):

    if args.rule == "Drichlet":
        path = f'{save_path}/Output'
    else:
        raise ValueError("Unknown rule")

    if not os.path.exists(path):
        os.makedirs(path)

    accuracy_file = 'accuracy_{}_{}.txt'.format(args.algorithm,
                                                        datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    print('save result to: ', os.path.join(path, accuracy_file))

    with open(os.path.join(path, accuracy_file), 'a') as f1:

        f1.write('accuracy: ')
        for element in accuracy:
            f1.write(str(element))
            f1.write(' ')
        f1.write('\n')

        f1.write('total_time(s): ')
        f1.write(str(total_time))
        f1.write('\n')

        f1.write('read times(s): ')
        f1.write(str(read_time))
        f1.write(' ')
        f1.write('\n')

        f1.write('latency times(s): ')
        for element in latency_times:
            f1.write(str(element))
            f1.write(' ')
        f1.write('\n')

        f1.write('training times(s): ')
        f1.write(str(training_time))
        f1.write(' ')
        f1.write('\n')

        f1.write('optimization problem times(s): ')
        for element in opt_times:
            f1.write(str(element))
            f1.write(' ')
        f1.write('\n')

        f1.write('aggregation times(s): ')
        for element in aggr_times:
            f1.write(str(element))
            f1.write(' ')
        f1.write('\n')

        f1.write('append times(s): ')
        f1.write(str(append_time))
        f1.write(' ')
        f1.write('\n')

        f1.write('consume_read(Wh): ')
        f1.write(str(read_consume))
        f1.write(' ')
        f1.write('\n')

        f1.write('consume_opt(Wh): ')
        for element in opt_consume:
            f1.write(str(element))
            f1.write(' ')
        f1.write('\n')

        f1.write('consume_weights(Wh): ')
        for element in weights_consume:
            f1.write(str(element))
            f1.write(' ')
        f1.write('\n')


    print('save finished')
    f1.close()



