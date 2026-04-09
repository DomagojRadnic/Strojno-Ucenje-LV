import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

def non_func(x):
    y = 1.6345 - 0.6235*np.cos(0.6067*x) - 1.3501*np.sin(0.6067*x) - 1.1622 * np.cos(2*x*0.6067) - 0.9443*np.sin(2*x*0.6067)
    return y

def add_noise(y):
    np.random.seed(14)
    varNoise = np.max(y) - np.min(y)
    y_noisy = y + 0.1*varNoise*np.random.normal(0,1,len(y))
    return y_noisy

def compute_mse(degree, x, y_measured, indeksi_train, indeksi_test):
    poly = PolynomialFeatures(degree=degree)
    xnew = poly.fit_transform(x)
    
    xtrain = xnew[indeksi_train]
    ytrain = y_measured[indeksi_train]
    xtest = xnew[indeksi_test]
    ytest = y_measured[indeksi_test]
    
    linearModel = lm.LinearRegression()
    linearModel.fit(xtrain, ytrain)
    
    ytest_p = linearModel.predict(xtest)
    MSE_test = mean_squared_error(ytest, ytest_p)
    
    ytrain_p = linearModel.predict(xtrain)
    MSE_train = mean_squared_error(ytrain, ytrain_p)
    
    return MSE_train, MSE_test, linearModel, xnew, ytest_p

x = np.linspace(1, 10, 50)
y_true = non_func(x)
y_measured = add_noise(y_true)

x = x[:, np.newaxis]
y_measured = y_measured[:, np.newaxis]

np.random.seed(12)
indeksi = np.random.permutation(len(x))
indeksi_train = indeksi[0:int(np.floor(0.7*len(x)))]
indeksi_test = indeksi[int(np.floor(0.7*len(x)))+1:len(x)]

degrees = [2, 6, 15]
MSEtrain = []
MSEtest = []

for degree in degrees:
    mse_train, mse_test, model, xnew, ytest_p = compute_mse(degree, x, y_measured, indeksi_train, indeksi_test)
    MSEtrain.append(mse_train)
    MSEtest.append(mse_test)

print("MSE on training data: ", MSEtrain)
print("MSE on testing data: ", MSEtest)

plt.figure(figsize=(10, 6))
for i, degree in enumerate(degrees):
    poly = PolynomialFeatures(degree=degree)
    xnew = poly.fit_transform(x)
    _, _, model, _, _ = compute_mse(degree, x, y_measured, indeksi_train, indeksi_test)
    plt.plot(x, model.predict(xnew), label=f'Degree {degree}')
    
plt.plot(x, y_true, 'k-', label='True function')
plt.legend(loc='upper right')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Model Comparison for Different Polynomial Degrees')
plt.show()

sample_sizes = [50, 100, 200]
for sample_size in sample_sizes:
    x = np.linspace(1, 10, sample_size)
    y_true = non_func(x)
    y_measured = add_noise(y_true)
    x = x[:, np.newaxis]
    y_measured = y_measured[:, np.newaxis]
    
    np.random.seed(12)
    indeksi = np.random.permutation(len(x))
    indeksi_train = indeksi[0:int(np.floor(0.7*len(x)))]
    indeksi_test = indeksi[int(np.floor(0.7*len(x)))+1:len(x)]
    
    MSEtrain = []
    MSEtest = []
    
    for degree in degrees:
        mse_train, mse_test, _, _, _ = compute_mse(degree, x, y_measured, indeksi_train, indeksi_test)
        MSEtrain.append(mse_train)
        MSEtest.append(mse_test)
    
    print(f"Sample size: {sample_size}")
    print(f"MSE train: {MSEtrain}")
    print(f"MSE test: {MSEtest}")