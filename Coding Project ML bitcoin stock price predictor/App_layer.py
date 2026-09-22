import customtkinter 
from CTkTable import * 
import numpy as np 
from matplotlib.backend_bases import key_press_handler 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg) 
from matplotlib.figure import Figure
import yfinance as yf
import pandas as pd
import os
import Neural_Network as NN
import Data_Pipeline as NSA

 
 
customtkinter.set_appearance_mode("light") 
customtkinter.set_default_color_theme("green") 
 
app = customtkinter.CTk() 
app.title("Bitcoin Stock Predictor") 
 
app.geometry("1600x1000") 

def Table(i, j, location, table_Values): 
    table = CTkTable(master=location, row=i, column=j, values=table_Values)
    table.pack(expand=True, fill="both", padx=20, pady=20) 
    table.place(x=200, y=650)
    return table
 

predictions = []

Tabs = customtkinter.CTkTabview(app, 
    corner_radius=40, 
    width = 1500, 
    height = 900 
                                 
                                 
                                ) 
Tabs.pack(padx=20, pady=10, fill="both", expand=True) 

 
 
Tab_2 = Tabs.add("Predictions") 

 
 
Crypto_currency = ["5 Day", "1 Month", "3 Month", "6 Month", "1 Year"] 
 
def Clean_up(): 
    data = yf.download("BTC-USD", period="250d", interval="1d")
    data.columns = data.columns.get_level_values(0)
    data = data.reset_index(drop=True)
    data = data[["Open", "High", "Low", "Close"]]
    return data



def initialise_weights(n):
    print("This model,", n)
    path = f"{n} Model"
    layers = [NN.Input_Layer, NN.Hidden_Layer_1, NN.Hidden_Layer_2, NN.Output_Layer]
    mean_train_Y = np.load(f"mean_train_Y {n}.npy")
    std_train_Y = np.load(f"std_train_Y {n}.npy")
    for i, layer in enumerate(layers, start=1):
        layer.weights = np.load(os.path.join(path, f"layer{i}_weights.npy"))
        layer.biases = np.load(os.path.join(path, f"layer{i}_biases.npy"))
    Price = NSA.add_Indicators(Clean_up(), n)
    NN.Input_Layer.forward(Price) 
    NN.RELU.forward(NN.Input_Layer.outputs) 
    NN.Hidden_Layer_1.forward(NN.RELU.output) 
    NN.RELU2.forward(NN.Hidden_Layer_1.outputs) 
    NN.Hidden_Layer_2.forward(NN.RELU2.output) 
    NN.RELU3.forward(NN.Hidden_Layer_2.outputs) 
    NN.Output_Layer.forward(NN.RELU3.output)
    prediction_norm = NN.Output_Layer.outputs
    prediction = (prediction_norm * std_train_Y) + mean_train_Y
    if len(predictions) < 5:
        predictions.append(round(float(prediction[0][0])))
    return prediction
    
def menu_pressed(choice):
    Prediction = initialise_weights(choice)
    Prediction = str(Prediction[0][0])
    share = yf.Ticker('BTC-USD').info
    market_price = share['regularMarketPrice' ]
    percentage_change = share['regularMarketChangePercent']
    lst2[1][1] = market_price
    lst2[1][2] = percentage_change
    if choice == "5 Day":
        lst2[1][3] = Prediction
        
    elif choice == "1 Month":
        lst2[1][4] = Prediction
        
    elif choice == "3 Month":
        lst2[1][5] = Prediction
        
    elif choice == "6 Month":
        lst2[1][6] = Prediction
        
    elif choice == "1 Year":
        lst2[1][7] = Prediction
        
    
    table_2 = Table(2, 8, Tab_2, lst2) 
    

     
def menu(Tab): 
    Menu = customtkinter.CTkOptionMenu(Tab, values=Crypto_currency, command=menu_pressed) 
    Menu.place(x=30, y=20)
    return Menu
 
 
def Button(Tab, Command): 
    button = customtkinter.CTkButton(Tab, text = "Plot", fg_color="blue", command=Command) 
    button.place(x=1300, y=20) 

def plot_predictions():
    predict = predictions
    time = [5, 30, 90, 182, 365]
    x_plot = time
    fig = Figure(figsize=(7, 4), dpi=150)
    plot1 = fig.add_subplot(111)
    plot1.plot(x_plot, predictions, marker='o',linestyle='--', color='orange', label='Predicted Prices')
    plot1.set_xticks(x_plot)
    plot1.set_xticklabels(time)
    plot1.set_xlabel("Timeframe")
    plot1.set_ylabel("Predicted Price ($)")
    plot1.legend()
    plot1.grid(True )
    canvas = FigureCanvasTkAgg(fig, master=Tab_2)
    canvas.draw()
    canvas.get_tk_widget().place(x=200, y=20)



Menu2 = menu(Tab_2) 
 
Button2 = Button(Tab_2, plot_predictions) 

lst2 = [['Crypto-currency', 'Live price', '24 hour % change', '5d predicton', '1 month prediction', '3 month prediction', '6  month prediction', '1 year prediction'], 
       ['Bitcoin', '', '', '', '', '', '', '']] 

app.mainloop() 