#Linear Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# For regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from mpl_toolkits.mplot3d import axes3d

from sklearn.preprocessing import scale
import sklearn.linear_model as skl_lm
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf

# %matplotlib inline
# #plt.style.use('seaborn-white')
tv_data = pd.read_csv(r'C:\Users\shanc\my-AI\scripts\TVmarketing.csv')
print(tv_data)
tv_data.describe()
print(tv_data.head(2))
print(tv_data.tail(6))
print(tv_data.shape)
##################################################################################
# Visualise the relationship between the features and the response using scatterplots
tv_data.plot(x='TV',y='Sales',kind='scatter')
X_trn = tv_data['TV'].values
y_trn = tv_data['Sales'].values
print(X_trn)
print(y_trn)
###################################################################################
#random_state is the seed used by the random number generator, it can be any integer.
from sklearn.model_selection import train_test_split
##This imports the train_test_split function from sklearn.model_selection. 
# It's used to split datasets into training and testing subsets.
X_trainnew, X_testnew, y_trainnew, y_testnew = train_test_split(X_trn, y_trn,  test_size=0.3, random_state = 1)
print(X_trainnew.shape)
#X_trn: The feature matrix for the training data (input variables).
#y_trn: The target vector for the training data (output variable).
#test_size=0.3: This specifies that 30% of the data should be used for testing, and the remaining 70% will be used for training.
#random_state=1: This ensures reproducibility of the split. By fixing the random seed, you will get the same split each time you run the code.

# random_state एक पैरामीटर है जो डेटा को विभाजित (split) करते समय "रैंडमनेस" (randomness) को नियंत्रित करता है। 
# इसका मुख्य उद्देश्य पुनरावृत्ति (reproducibility) को सुनिश्चित करना है, यानी यह सुनिश्चित करना कि जब आप कोड को बार-बार चलाते हैं, तो डेटा का विभाजन हमेशा एक जैसा रहे।
# डेटा का विभाजन: जब आप train_test_split का इस्तेमाल करते हैं, तो यह आपके डेटा को रैंडम तरीके से ट्रेनिंग और टेस्टिंग सेट्स में बाँटता है। 
# लेकिन कभी-कभी आप चाहते हैं कि यह विभाजन हर बार एक जैसा हो, ताकि आप आसानी से परिणामों की तुलना कर सकें।
#जब आप random_state को एक फिक्स (fix) मान देते हैं (जैसे random_state=1), तो हर बार जब आप कोड चलाएंगे, 
# तो डेटा का विभाजन वही रहेगा। इससे यह सुनिश्चित होता है कि परिणामों में कोई बदलाव नहीं होगा।

class new_fun_LR:
    
    def __init__(self):
        #crearting an instance of the class and declaring and initialising all the variables.
        self.m = None
        self.b = None
        
    def fit(self,X_train,y_train):
        
        num = 0
        den = 0
        # printing the TV column mean value
        print("xmean",X_train.mean())
        # printing the Sales column mean value
        print("ymean",y_train.mean())
        
        for i in range(X_train.shape[0]):
            print(i)
            print("x=",X_train[i])
            print("y=",y_train[i])
            print("c",((X_train[i] - X_train.mean())))
            print("d",(y_train[i] - y_train.mean()))
            print("num",(X_train[i] - X_train.mean())*(y_train[i] - y_train.mean()))
            print("den",((X_train[i] - X_train.mean())*(X_train[i] - X_train.mean())))
            
            num = num + ((X_train[i] - X_train.mean())*(y_train[i] - y_train.mean()))
            den = den + ((X_train[i] - X_train.mean())*(X_train[i] - X_train.mean()))
            
            print("updated_numerator=",num)
            print("updated_denominator=",den)
            print('*******************************')
            
        print("final num=",num)
        print("final den=",den)
        self.m = num/den
        self.b = y_train.mean() - (self.m * X_train.mean())
        print("slope",self.m)
        print("Intercept",self.b)       
    
    def predict(self,X_test):
        
        print("The input x values")
        print(X_test)
        
        #mx+b
        return self.m * X_test + self.b

# Representing LinearRegression as lr(Creating LinearRegression Object)
lr = new_fun_LR()

# Fit the model using lr.fit()
lr.fit(X_trainnew, y_trainnew)
print(X_testnew)          
y_pred1 = lr.predict(X_testnew)
print("Predicted values \n",y_pred1)

# Actual vs Predicted
import matplotlib.pyplot as plt
c = [i for i in range(1,61,1)]         # generating index 
fig = plt.figure()
plt.plot(c,y_testnew, color="navy", linewidth=2, linestyle="-")
plt.plot(c,y_pred1, color="grey",  linewidth=2, linestyle="-")

fig.suptitle('Actual and Predicted', fontsize=20)   #This adds a title to the plot and labels to the x and y axes.
# Plot heading 
plt.xlabel('Index', fontsize=18)                               # X-label
plt.ylabel('Sales', fontsize=16)    
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_testnew, y_pred1)
print(mse)
r_squared = r2_score(y_testnew, y_pred1)
print(r_squared)