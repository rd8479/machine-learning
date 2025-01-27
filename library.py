import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import library as lib
import time
import matplotlib
from tkinter import * 
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


def evaluation_metric(y,y_test,y_pred):
    mean=y.mean()
    MAE=mean_absolute_error(y_test,y_pred)
    MSE=mean_squared_error(y_test,y_pred)
    RMSE=np.sqrt(mean_squared_error(y_test,y_pred))
    return{'MAE':(MAE,MAE/mean*100),'MSE':(MSE,MSE/mean*100),'RMSE':(RMSE,RMSE/mean*100)}


def Residual_Plots(y_test,y_pred):
    residual=y_test-y_pred
    sns.scatterplot(x=y_test,y=residual)
    sns.displot(residual,bins=50,kde=True)


def polynomial_Features_new_data(x,y):
    train_rmse_error=[]
    test_rmse_error=[]
    for d in range(1,10):
        poly_converter=PolynomialFeatures(degree=d,include_bias=False)
        poly_feature=poly_converter.fit_transform(x)
        x_train,x_test,y_train,y_test=train_test_split(poly_feature,y,test_size=0.33,random_state=42)
        model=LinearRegression()
        model.fit(x_train,y_train)
        train_pred=model.predict(x_train)
        test_pred=model.predict(x_test)
        train_rsme=np.sqrt(mean_squared_error(y_train,train_pred))
        test_rsme=np.sqrt(mean_squared_error(y_test,test_pred))
        train_rmse_error.append(train_rsme)
        test_rmse_error.append(test_rsme)
    tk=Tk()
    label = Label(text="Graph Page!")
    label.pack(pady=10,padx=10)
    Button(tk, text="Quit", command=tk.destroy).pack()
    f = Figure(figsize=(5,5))
    a = f.add_subplot(111)
    a.plot(range(1,6),train_rmse_error[:5],label='Train RMSE')
    a.plot(range(1,6),test_rmse_error[:5],label='Test RMSE')
    canvas = FigureCanvasTkAgg(f)
    canvas.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=True)
    toolbar = NavigationToolbar2Tk(canvas)
    toolbar.update()
    canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=True)
    deg_var=StringVar()
    def Polynomial_Features():
        deg=int(deg_var.get())
        poly_converter=PolynomialFeatures(degree=deg,include_bias=False)
        global poly_features
        poly_features=poly_converter.fit_transform(x)
        tk.destroy()
    name_label = Label(tk, text = 'Username', font=('calibre',10, 'bold'))
    name_entry = Entry(tk,textvariable = deg_var, font=('calibre',10,'normal')) 
    sub_btn=Button(tk,text = 'Polynomial_Features', command = Polynomial_Features)
    name_label.pack()
    name_entry.pack()
    sub_btn.pack()
    tk.mainloop()
    return poly_features





