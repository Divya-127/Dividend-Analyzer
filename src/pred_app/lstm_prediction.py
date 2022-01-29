import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn 
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Dropout,Bidirectional
from tensorflow.keras.optimizers import RMSprop
import json
import datetime


def lstm_prediction(se, stock_symbol):
	import pandas as pd
	import numpy as np

	def fetch_stock_data(se, stock_symbol,sdate,edate):
		"""fetch stock data"""
		from pandas_datareader import data as pdr
		import yfinance as yf
		yf.pdr_override()
		if se == 'NSE': stock_symbol += ".NS" 
		return yf.download(stock_symbol, start=sdate, end=edate)

	df = fetch_stock_data(se, stock_symbol,'2013-01-01','2018-12-31')
	print(df.head())
	df2 = fetch_stock_data(se, stock_symbol,'2019-01-01','2019-12-31')
	df = df.dropna()
	df = df.reset_index(drop=True)
	df2 = df2.dropna()
	df2 = df2.reset_index(drop=True)

	print(df.head())

	ema_26_dataset1 = []
	ema_12_dataset1 = []
	macd = []
	smoothing = 2

	for num in range(len(df)):
		if num == 0:
			temp1 = df['Adj Close'][num] * (smoothing/(1+26))
			temp2 = df['Adj Close'][num] * (smoothing/(1+12))
			ema_26_dataset1.append(temp1)
			ema_12_dataset1.append(temp2)
			macd.append(ema_12_dataset1[num] - ema_26_dataset1[num])
		else:
			temp1 = (df['Adj Close'][num] * (smoothing/(1+26))) - (ema_26_dataset1[num-1] * (1-(smoothing/(1+26))))
			temp2 = (df['Adj Close'][num] * (smoothing/(1+12))) - (ema_12_dataset1[num-1] * (1-(smoothing/(1+12))))
			ema_26_dataset1.append(temp1)
			ema_12_dataset1.append(temp2)
			macd.append(ema_12_dataset1[num] - ema_26_dataset1[num])
		
	df['MACD'] = macd
	ema_26_dataset2 = []
	ema_12_dataset2 = []
	macd2 = []
	smoothing = 2

	print("2"*50)

	for num in range(len(df2)):
		if num == 0:
			temp1 = df2['Adj Close'][num] * (smoothing/(1+26))
			temp2 = df2['Adj Close'][num] * (smoothing/(1+12))
			ema_26_dataset2.append(temp1)
			ema_12_dataset2.append(temp2)
			macd2.append(ema_12_dataset2[num] - ema_26_dataset2[num])
		else:
			temp1 = (df2['Adj Close'][num] * (smoothing/(1+26))) - (ema_26_dataset2[num-1] * (1-(smoothing/(1+26))))
			temp2 = (df2['Adj Close'][num] * (smoothing/(1+12))) - (ema_12_dataset2[num-1] * (1-(smoothing/(1+12))))
			ema_26_dataset2.append(temp1)
			ema_12_dataset2.append(temp2)
			macd2.append(ema_12_dataset2[num] - ema_26_dataset2[num])

	df2['MACD'] = macd2
	mfi_dataset1 = []

	print("3"*50)

	for num in range(14,len(df)):
		typical_price = (df['High'][num] + df['Low'][num] + df['Close'][num])/3
		raw_money_flow = typical_price * df['Volume'][num]
		positive_money_flow = 0
		negative_money_flow = 0
		curr_money_flow = raw_money_flow
		num1 = num-1
		while num1 >= num-14:
			typical_price1 = (df['Close'][num1] + df['Low'][num1] + df['High'][num1])/3
			raw_money_flow1 = typical_price1 * df['Volume'][num1]
			if raw_money_flow1 < curr_money_flow:
				positive_money_flow += curr_money_flow
			else:
				negative_money_flow += curr_money_flow
			curr_money_flow = raw_money_flow1
			num1-=1
		money_flow_ratio = positive_money_flow/negative_money_flow
		mfi = 100 - 100/(1+money_flow_ratio)
		mfi_dataset1.append(mfi)

	mfi_dataset2 = []

	for num in range(14,len(df2)):
		typical_price = (df2['High'][num] + df2['Low'][num] + df2['Close'][num])/3
		raw_money_flow = typical_price * df2['Volume'][num]
		positive_money_flow = 0
		negative_money_flow = 0
		curr_money_flow = raw_money_flow
		num1 = num-1
		while num1 >= num-14:
			typical_price1 = (df2['Close'][num1] + df2['Low'][num1] + df2['High'][num1])/3
			raw_money_flow1 = typical_price1 * df['Volume'][num1]
			if raw_money_flow1 < curr_money_flow:
				positive_money_flow += curr_money_flow
			else:
				negative_money_flow += curr_money_flow
			curr_money_flow = raw_money_flow1
			num1-=1
		money_flow_ratio = positive_money_flow/negative_money_flow
		mfi = 100 - 100/(1+money_flow_ratio)
		mfi_dataset2.append(mfi)

	df = df[:][14:]
	df2 = df2[:][14:]
	df['MFI'] = mfi_dataset1
	df2['MFI'] = mfi_dataset2

	print(df)

	dataset=df.drop(['Close'],axis=1)
	dataset2=df2.drop(['Close'],axis=1)

	dataset.rename(columns={'Adj Close':'Prediction Price'}, inplace=True)
	dataset2.rename(columns={'Adj Close':'Prediction Price'}, inplace=True)

	print("1"*50)

	from sklearn.preprocessing import MinMaxScaler
	min_max_scaler = MinMaxScaler()

	dataset[["Open","High","Low","Prediction Price","Volume","MACD","MFI"]] = min_max_scaler.fit_transform(dataset[["Open","High","Low","Prediction Price","Volume","MACD", "MFI"]])

	dataset2[["Open","High","Low","Prediction Price","Volume","MACD", "MFI"]] = min_max_scaler.fit_transform(dataset2[["Open","High","Low","Prediction Price","Volume","MACD", "MFI"]])

	new_dataset=dataset.copy()
	new_dataset2=dataset2.copy()
	n_train = len(new_dataset)
	n_test = len(new_dataset2)
	train_data = new_dataset.iloc[:n_train,:].values
	test_data = new_dataset2.iloc[:n_test,:].values

	import numpy
	def create_dataset(dataset,time_step=1):
		dataX,dataY=[],[]
		for i in range(len(dataset)-time_step-1):
			a = dataset[i:(i+time_step),[0,1,2,4,5,6]]
			dataX.append(a)
			dataY.append(dataset[i+time_step,3])
			# print(dataX,dataY)
		return numpy.array(dataX),numpy.array(dataY)
	
	time_step=100
	X_train,y_train = create_dataset(train_data,time_step)
	X_test,y_test = create_dataset(test_data,time_step)

	model = Sequential()
	model.add(Bidirectional(LSTM(50,return_sequences=False,input_shape=(100,6)),input_shape=(100,6)))
	model.add(Dense(100))
	model.add(Dropout(0.25))

	model.add(Dense(1,activation='tanh'))
	optimizer = RMSprop(lr=0.0001)
	model.compile(loss='mean_squared_error',optimizer=optimizer)
	print("#"*50)

	#!Change Epoch
	model.fit(X_train,y_train,epochs=50,batch_size=50, verbose=1)

	print("&"*50)

	training_error = model.evaluate(X_train,y_train,verbose=0)
	testing_error = model.evaluate(X_test,y_test,verbose=0)

	train_predict = model.predict(X_train)
	test_predict = model.predict(X_test)
	
	#print(train_predict.shape,test_predict.shape, X_test.shape)
	
	# create empty table with 6 fields
	trainPredict_dataset_like = np.zeros(shape=(len(train_predict), 7) )
	# put the predicted values in the right field
	trainPredict_dataset_like[:,0] = train_predict[:,0]
	# inverse transform and then select the right field
	train_predict = min_max_scaler.inverse_transform(trainPredict_dataset_like)[:,0]
	
	#train_predict = min_max_scaler.inverse_transform(train_predict)
	
	# create empty table with 6 fields
	testPredict_dataset_like = np.zeros(shape=(len(test_predict), 7) )
	# put the predicted values in the right field
	testPredict_dataset_like[:,0] = test_predict[:,0]
	# inverse transform and then select the right field
	test_predict = min_max_scaler.inverse_transform(testPredict_dataset_like)[:,0]
	#print(train_predict[0],test_predict[0])
	
	#test_predict = min_max_scaler.inverse_transform(test_predict)
	
	# create empty table with 6 fields
	ogiPredict_dataset_like = np.zeros(shape=(len(test_predict), 7) )
	# y_test.reshape(-1,1)
	# print(y_test.shape)
	# put the predicted values in the right field
	ogiPredict_dataset_like[:,0] = y_test
	# inverse transform and then select the right field
	y_test = min_max_scaler.inverse_transform(ogiPredict_dataset_like)[:,0]
	
	#print(train_predict.shape,test_predict.shape,y_test.shape)
	#print(test_predict[100],y_test[100])

	# y train ke liye

	# create empty table with 6 fields
	ogiPredict_dataset_like = np.zeros(shape=(len(y_train), 7) )
	# y_test.reshape(-1,1)
	# print(y_test.shape)
	# put the predicted values in the right field
	ogiPredict_dataset_like[:,0] = y_train
	# inverse transform and then select the right field
	y_train = min_max_scaler.inverse_transform(ogiPredict_dataset_like)[:,0]
	
	#print(train_predict.shape,test_predict.shape,y_test.shape, y_train.shape)
	#print(train_predict[100],y_train[100])

	import matplotlib.pyplot as plt
	plt.plot(y_test,color = 'b')
	plt.plot(test_predict,color='r')
	plt.title('Predicted Vs Actual for LSTM')
	plt.legend(['Actual','Predicted'])
	plt.xlabel('Time Period')
	plt.ylabel('Price ($)')
	min1 = min(test_predict)//2
	max1 = max(test_predict)*3
	plt.ylim(min1, max1)

	lst_output=[]
	n_steps=100
	nextNumberOfDays = 30
	i=0

	while(i<nextNumberOfDays):
		x_input = X_test[i:]        

		
		if(len(x_input)>100):
			yhat = model.predict(x_input)
			#print("{} day output".format(i))
			#print(float(yhat[100]))
			
			# print(yhat[i+100].shape)
			
			lst_output.append(float(yhat[100]))        
		# else:
		#     x_input = X_test[i:] 
		#     yhat = model.predict(x_input, verbose=0)
		#     # lst_output.append(int(yhat[i+100]))


		i=i+1

	test_predict1 = (model.predict(X_test))
	#print(test_predict1)
	#print(lst_output)
	predicted_price = []
	final_price = []

	for d in test_predict1:
		predicted_price.append(d[0])

	for d in lst_output:
		predicted_price.append(d)

	for d in predicted_price:
		trainPredict_dataset_like[:,0] = float(d)
		train_predict = min_max_scaler.inverse_transform(trainPredict_dataset_like)[:,0]
		final_price.append(train_predict[0])
	#print(train_predict[0])

	#print(len(y_test),len(final_price))

	import matplotlib.pyplot as plt

	plt.plot(y_test,color = 'b')
	plt.plot(final_price,color='r')
	plt.title('Predicted Vs Actual for LSTM')
	plt.legend(['Actual','Predicted'])
	plt.xlabel('Time Period')
	plt.ylabel('Price ($)')
	min1 = min(final_price)//2
	max1 = max(final_price)*3
	plt.ylim(min1, max1)
	plt.savefig('plot.png')

	#Combining og and predicted dataset for end result.
	result_df = fetch_stock_data(se,stock_symbol,'2013-01-01','2021-12-31')
	result_df.reset_index(inplace=True)
	print(lst_output)
	print(result_df.head())
	print(result_df.columns)
	result_df.drop('Open',axis=1,inplace=True)
	result_df.drop('High',axis=1,inplace=True)
	result_df.drop('Low',axis=1,inplace=True)
	result_df.drop('Adj Close',axis=1,inplace=True)
	result_df.drop('Volume',axis=1,inplace=True)
	print(result_df.columns)
	

	print("HIHIHIHIHI")

	#to print the info of the END RESULT dataset
	print("\n<----------------------Info of the RESULT dataset---------------------->")
	print(result_df.info())
	print("<------------------------------------------------------------------------>\n")

	def get_json(df):
		def convert_timestamp(item_date_object):
			if isinstance(item_date_object, (datetime.date, datetime.datetime)):
				return item_date_object.strftime("%Y-%m-%d")
		dict_ = df.to_dict(orient='records')
		return json.dumps(dict_, default=convert_timestamp)

	return get_json(result_df)