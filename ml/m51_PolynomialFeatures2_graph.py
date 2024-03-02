import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
plt.rcParams['font.family']='Malgun Gothic' #한글

# 1. Data
np.random.seed(123)

x= 2*np.random.rand(100,1)-1    # => -1 부터 1까지의 랜덤한 100개 난수
y= 3*x**2+2*x+1+np.random.randn(100,1)  # y = 3x^2 + 2x +1 + 노이즈[np.random.randn(100,1)]
#np.random.randn(100,1) - 0부터 1 까지 난수

print(x.shape)  #(100, 1)
print(y.shape)  #(100, 1)

pf = PolynomialFeatures(degree=2, include_bias=False)
x_poly = pf.fit_transform(x)

print(x_poly.shape) #(100, 2)

# 2. Model
model = LinearRegression()
model2 = LinearRegression()

# 3. Compile Fit
model.fit(x,y)
model2.fit(x_poly,y)

plt.scatter(x,y,color='blue', label='원데이터')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression Example')

x_plot = np.linspace(-1,1,100).reshape(-1,1)
x_plot_poly = pf.transform(x_plot)
y_plot = model.predict(x_plot)
y_plot2 = model2.predict(x_plot_poly)
plt.plot(x_plot,y_plot,color='red', label='Polynomial Regression')
plt.plot(x_plot,y_plot2,color='green',label='Polynomial Regression2')
plt.legend()
plt.show()
