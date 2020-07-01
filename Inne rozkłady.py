import pandas as pd
import numpy as np
from TestyMVN import Mardia_test_skewness, Mardia_test_kurtosis, HZtest, Roystest, optimal_beta
from openpyxl import load_workbook


# t-Student

def tStudent10(wymiar):
    Mardia_skew = []
    Mardia_kurt = []
    BHEP = []
    Royston = []
    for i in range(1, 10001):
        tst = np.array(pd.read_csv(r"E:\Documents\Aartykuły PRACA MAGISTERSKA\Python\tStudent%s\tstudent_%s.txt" % (wymiar, i)))
        Mardia_skew.append(Mardia_test_skewness(tst, True))
        Mardia_kurt.append(Mardia_test_kurtosis(tst, True))
        BHEP.append(HZtest(tst, True, beta=optimal_beta(100, wymiar)))
        Royston.append(Roystest(tst))

    Mardia_skew = pd.DataFrame(Mardia_skew)
    Mardia_kurt = pd.DataFrame(Mardia_kurt)
    BHEP = pd.DataFrame(BHEP)
    Royston = pd.DataFrame(Royston)

    path = r"E:\Documents\Aartykuły PRACA MAGISTERSKA\Python\tstudent_wyniki%s.xlsx" % wymiar

    writer = pd.ExcelWriter(path, engine = 'openpyxl')
    Mardia_skew.to_excel(writer, index=False, sheet_name = 'Mardia_skew')
    Mardia_kurt.to_excel(writer, index=False, sheet_name = 'Mardia_kurt')
    BHEP.to_excel(writer, index=False, sheet_name = 'Henze_Zirkler')
    Royston.to_excel(writer, index=False, sheet_name = 'Royston')
    writer.save()
    writer.close()
    return 'ok'


def tStudent(df, wymiar):
    Mardia_skew = []
    Mardia_kurt = []
    BHEP = []
    Royston = []
    for i in range(1, 10001):
        tst = np.array(pd.read_csv(r"E:\Documents\Aartykuły PRACA MAGISTERSKA\Python\tStudent_df%s\%s\tstudent_%s.txt" %
                                   (df, wymiar, i)))
        Mardia_skew.append(Mardia_test_skewness(tst, True))
        Mardia_kurt.append(Mardia_test_kurtosis(tst, True))
        BHEP.append(HZtest(tst, True, beta=optimal_beta(100, wymiar)))
        Royston.append(Roystest(tst))

    Mardia_skew = pd.DataFrame(Mardia_skew)
    Mardia_kurt = pd.DataFrame(Mardia_kurt)
    BHEP = pd.DataFrame(BHEP)
    Royston = pd.DataFrame(Royston)

    path = r"E:\Documents\Aartykuły PRACA MAGISTERSKA\Python\tStudent_df%s\tstudent_wyniki%s.xlsx" % (df, wymiar)

    writer = pd.ExcelWriter(path, engine = 'openpyxl')
    Mardia_skew.to_excel(writer, index=False, sheet_name = 'Mardia_skew')
    Mardia_kurt.to_excel(writer, index=False, sheet_name = 'Mardia_kurt')
    BHEP.to_excel(writer, index=False, sheet_name = 'Henze_Zirkler')
    Royston.to_excel(writer, index=False, sheet_name = 'Royston')
    writer.save()
    writer.close()
    return 'ok'


tStudent(15, 2)
tStudent(15, 3)
tStudent(15, 4)
tStudent(15, 5)

tStudent(30, 2)
tStudent(30, 3)
tStudent(30, 4)
tStudent(30, 5)

# Normalny


def losuj_norm(wymiar):
    Mardia_skew = []
    Mardia_kurt = []
    BHEP = []
    Royston = []
    mean1 = np.ones(wymiar) * 2
    cov1 = np.identity(wymiar)
    for i in range(1, 10001):
        tst = np.random.multivariate_normal(mean1, cov1, 100)
        Mardia_skew.append(Mardia_test_skewness(tst, True))
        Mardia_kurt.append(Mardia_test_kurtosis(tst, True))
        BHEP.append(HZtest(tst, True, beta=optimal_beta(100, wymiar)))
        Royston.append(Roystest(tst))

    Mardia_skew = pd.DataFrame(Mardia_skew)
    Mardia_kurt = pd.DataFrame(Mardia_kurt)
    BHEP = pd.DataFrame(BHEP)
    Royston = pd.DataFrame(Royston)

    path = r"E:\Documents\Aartykuły PRACA MAGISTERSKA\Python\Normal\inny\normal_wyniki%s.xlsx" % wymiar

    writer = pd.ExcelWriter(path, engine='openpyxl')
    Mardia_skew.to_excel(writer, index=False, sheet_name='Mardia_skew')
    Mardia_kurt.to_excel(writer, index=False, sheet_name='Mardia_kurt')
    BHEP.to_excel(writer, index=False, sheet_name='Henze_Zirkler')
    Royston.to_excel(writer, index=False, sheet_name='Royston')
    writer.save()
    writer.close()
    return str(wymiar) + 'ok'


losuj_norm(2)
losuj_norm(3)
losuj_norm(4)
losuj_norm(5)


def losuj_mix(n, d, p):
    X = np.random.multivariate_normal(np.zeros(d), np.identity(d), n)
    X2 = np.random.multivariate_normal(np.ones(d)*2, np.identity(d), n)

    for i in range(n):
        test = np.random.choice([0, 1], 1, p=[p, 1-p])
        if test == 1:
            X[i, :] = X2[i, :]
    return X


# test = losuj_mix(100, 3, 0.9)
# print(np.mean(test, axis=0))


def licz_mix(n, d, p):
    Mardia_skew = []
    Mardia_kurt = []
    BHEP = []
    Royston = []
    for i in range(1, 10001):
        tst = losuj_mix(n, d, p)
        Mardia_skew.append(Mardia_test_skewness(tst, True))
        Mardia_kurt.append(Mardia_test_kurtosis(tst, True))
        BHEP.append(HZtest(tst, True, beta=optimal_beta(n, d)))
        Royston.append(Roystest(tst))

    Mardia_skew = pd.DataFrame(Mardia_skew)
    Mardia_kurt = pd.DataFrame(Mardia_kurt)
    BHEP = pd.DataFrame(BHEP)
    Royston = pd.DataFrame(Royston)

    path = r"E:\Documents\Aartykuły PRACA MAGISTERSKA\Python\mix_normal\078\mix_wyniki%s.xlsx" % d

    writer = pd.ExcelWriter(path, engine='openpyxl')
    Mardia_skew.to_excel(writer, index=False, sheet_name='Mardia_skew')
    Mardia_kurt.to_excel(writer, index=False, sheet_name='Mardia_kurt')
    BHEP.to_excel(writer, index=False, sheet_name='Henze_Zirkler')
    Royston.to_excel(writer, index=False, sheet_name='Royston')
    writer.save()
    writer.close()
    return str(d) + 'ok'


licz_mix(100, 2, 0.9)
licz_mix(100, 3, 0.9)
licz_mix(100, 4, 0.9)
licz_mix(100, 5, 0.9)


def Pearson_wektor(d, m, k):
    X1 = np.sqrt(np.random.beta(1 / 2, m + d / 2 + 1 / 2, 1))
    test = np.random.choice([0, 1], 1, p=[0.5, 0.5])
    if test == 0:
        X1 = (-1) * X1
    if k ==1:
        return [X1, X1**2]
    V = np.random.beta(1 / 2, m + d / 2 + 1 / 2 - (k - 1) / 2, 1)
    poprzednik = Pearson_wektor(d, m, k-1)
    X = np.sqrt(V*(1-poprzednik[1]))
    suma_kw = poprzednik[1] + X**2
    test = np.random.choice([0, 1], 1, p=[0.5, 0.5])
    if test == 0:
        X = (-1) * X
    wektor = np.append(poprzednik[0], X)
    return wektor, suma_kw


def Pearson_typeII(d, m, n):
    Xn = np.zeros([1, d])
    for i in range(n):
        X = Pearson_wektor(d, m, d)[0].reshape([1, d])
        Xn = np.append(Xn, X, axis=0)
    Xn = np.delete(Xn, [0], 0)
    return Xn


def losuj_Pearson(wymiar, m, sciezka):
    Mardia_skew = []
    Mardia_kurt = []
    BHEP = []
    Royston = []
    for i in range(1, 10001):
        tst = Pearson_typeII(wymiar, m, 100)
        Mardia_skew.append(Mardia_test_skewness(tst, True))
        Mardia_kurt.append(Mardia_test_kurtosis(tst, True))
        BHEP.append(HZtest(tst, True, beta=optimal_beta(100, wymiar)))
        Royston.append(Roystest(tst))

    Mardia_skew = pd.DataFrame(Mardia_skew)
    Mardia_kurt = pd.DataFrame(Mardia_kurt)
    BHEP = pd.DataFrame(BHEP)
    Royston = pd.DataFrame(Royston)

    path = sciezka % wymiar

    writer = pd.ExcelWriter(path, engine='openpyxl')
    Mardia_skew.to_excel(writer, index=False, sheet_name='Mardia_skew')
    Mardia_kurt.to_excel(writer, index=False, sheet_name='Mardia_kurt')
    BHEP.to_excel(writer, index=False, sheet_name='Henze_Zirkler')
    Royston.to_excel(writer, index=False, sheet_name='Royston')
    writer.save()
    writer.close()
    return str(wymiar) + 'ok'


losuj_Pearson(2, 2, r"E:\Documents\Aartykuły PRACA MAGISTERSKA\Python\Pearson\Pearson_m2\pearson2_wyniki%s.xlsx")
losuj_Pearson(3, 2, r"E:\Documents\Aartykuły PRACA MAGISTERSKA\Python\Pearson\Pearson_m2\pearson2_wyniki%s.xlsx")
losuj_Pearson(4, 2, r"E:\Documents\Aartykuły PRACA MAGISTERSKA\Python\Pearson\Pearson_m2\pearson2_wyniki%s.xlsx")
losuj_Pearson(5, 2, r"E:\Documents\Aartykuły PRACA MAGISTERSKA\Python\Pearson\Pearson_m2\pearson2_wyniki%s.xlsx")

losuj_Pearson(2, 7, r"E:\Documents\Aartykuły PRACA MAGISTERSKA\Python\Pearson\Pearson_m7\pearson7_wyniki%s.xlsx")
losuj_Pearson(3, 7, r"E:\Documents\Aartykuły PRACA MAGISTERSKA\Python\Pearson\Pearson_m7\pearson7_wyniki%s.xlsx")
losuj_Pearson(4, 7, r"E:\Documents\Aartykuły PRACA MAGISTERSKA\Python\Pearson\Pearson_m7\pearson7_wyniki%s.xlsx")
losuj_Pearson(5, 7, r"E:\Documents\Aartykuły PRACA MAGISTERSKA\Python\Pearson\Pearson_m7\pearson7_wyniki%s.xlsx")