import numpy as np
import scipy.stats
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error


def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    # 4-parameter logistic function
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def compute_metrics(y_pred, y):
    '''
    compute metrics btw predictions & labels
    '''
    # compute SRCC & KRCC
    SRCC = scipy.stats.spearmanr(y, y_pred)[0]
    try:
        KRCC = scipy.stats.kendalltau(y, y_pred)[0]
    except:
        KRCC = scipy.stats.kendalltau(y, y_pred, method='asymptotic')[0]

    # logistic regression btw y_pred & y
    beta_init = [np.max(y), np.min(y), np.mean(y_pred), 0.5]
    popt, _ = curve_fit(logistic_func, y_pred, y, p0=beta_init, maxfev=int(1e8))
    y_pred_logistic = logistic_func(y_pred, *popt)

    # compute  PLCC RMSE
    PLCC = scipy.stats.pearsonr(y, y_pred_logistic)[0]
    RMSE = np.sqrt(mean_squared_error(y, y_pred_logistic))
    return [SRCC, KRCC, PLCC, RMSE]

def formated_print_results(metrics: dict, state, epoch):
    print('==========={}============='.format(state))
    print("Best Val Epoch {} : \n{} results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(
                                                                        epoch, state,
                                                                        metrics['SROCC'], metrics['KROCC'],
                                                                        metrics['PLCC'], metrics['RMSE']))

def formated_print_avg_results(avg_res: dict, num_iters: int):
    print('===========number of iters {}============='.format(num_iters))
    avg_metrics = avg_res['avg']
    for k, v in avg_metrics.items():
        print(k, '\n', '\tmean: ', v['mean'], '\n\tstd: ', v['std'])
        print('--------------')