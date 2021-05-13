# Prediction-of-Covid-19
-Used a SEIR model to describe the spread of disease. 

-In the case we continue implementing the current policy, predicted the number of cases. 

-Did all analyses using Python Numpy package and make data visualization with matplotlib.

# Data-visualization
Compare data with three methods
![image](https://github.com/Hans0524/Prediction-of-Covid-19/blob/main/screenshot/All%20methods%20plot.png)

# Run rt_err_h()
![image](https://github.com/Hans0524/Prediction-of-Covid-19/blob/main/screenshot/error%20vs%20h.png)
![image](https://github.com/Hans0524/Prediction-of-Covid-19/blob/main/screenshot/runtime.png)
![image](https://github.com/Hans0524/Prediction-of-Covid-19/blob/main/screenshot/time%20vs%20h.png)

I conv order 0.9980598074464063

II conv order 1.0019451676685105

III conv order 1.1857646689998234

Comment:For the relationship of error and h,when the h become larger the methodII rises linearly which is similar with method I and method II error is thehighest. Method III has a speedy rise but the final error is much lower.

# Run rt_err()
![image](https://github.com/Hans0524/Prediction-of-Covid-19/blob/main/screenshot/runtime%20vs%20error.png)

Comment: From the plot, with the iteration increases the relative error of method II
has a steep rise and method I has same trend as method II but steadier. Method III
almost has no relative error even the iteration is very large.

# Run pop()
![image](https://github.com/Hans0524/Prediction-of-Covid-19/blob/main/screenshot/Population.png)

Comment: From the plot the number of inflected people become lower after getting a peak andthe recovered population always rises until almost all people be recovered.
