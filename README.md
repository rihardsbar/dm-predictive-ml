# This is a masters course project for producing a predictive data mining model. 
Based on https://www.kaggle.com/deepmatrix/imdb-5000-movie-dataset


## Running the regressor_solver.py script
### Description 
What it is: </br>
  This script runs the GridSearchCV (http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) module in order to find the best possible pipeline for the model. Pipeline is configured of ||preprocesing||transformation||dimmension reduction||modelling||. The first three steps have been preconfigured and can break if configuration is changed. You should configure the moddeling part for your models.</br>
What is is not:</br>
  Even though it has been tested to be reliable, it is not a production level script and it is quite likely you are going to run into some issues. Script handles most of the errors which can be occur due to misconfguration or wrong values and is going to print those out. However might be exceptions happening which it is not going to handle, hence it is a good idea to try to understand the full flow of how it works.</br>
What is produces:</br>
  It produces a log automatically for the each run of the script. The amount of runs are defined within tuples_of_data list as described below. The end of the log will contain all the errors caught during the run and best results for each model you configure. An example log can be seen in */logistic_regression/regressRes_2017_04_08_173536_case0.log*

### Prerequisites 
*Setup your VM and run the script there. Running script for a single set of samples for your list of models is probably going to take 1-2 days. If you run those locally you can easily break the run, hence you'll need to start all over again.
*Set up a session managment tool like **tmux** and run everything within it. If you simply start running your computations on a ssh bare connection your simulation will be lost.
*Make sure you have enough disk space on your VM if have one set up already. Log files per one run can easily go over hundreds of megs. If you are running a computation for couple of sets of samples it may require couple of gigs of memory. If you run out of mememory big part your computational results are going to be lost. VM has got 50GB of mem per instance.
*Setup the python dependecies on your VM from the requirements.txt file. Requiremnts file is simply a dump of all pip packages that were present on the VM were script was written, some of them might required linux (apt-get) dependecies, please try to resolve those if you run into any issues. Script was written in python3, hence use pip3 to install the dependecies.  
> sudo pip3 install -r requirements.txt

### Setting up the script
*Put all your assigned models into the  `models = [Model1(), Model2(), Model3()]` list. Models list requires the object to be passed and no the class, hence class has be constructed via the call operator `()`. For now scipt contains only the linear regression instance  as `models = [LinearRegression()]`. 
*For each of your models prepopulate the `models_cfg = {}` parameter grid that is passed into the GridSearchCV. Param grid is dictonary of parameters with a list literals of all values that GridSearchCV is going to go through. Each parameter name in the model param grid should star with `model__`. For now it is configured only for the linear regression to have one parameter `models_cfg[LinearRegression.__name__] = dict( model__fit_intercept = [True])`. Look into the *logistic_regression/classifier_solver.py* for some examples. Do not simply copy the parameters over, for most methods they will not simply map. To find what parameters have to tuned look into the [API doc](http://scikit-learn.org/stable/modules/classes.html). As rule of thumb don't try to tune more than 4 params with more than 3-4 variations each per model, as otherwise your simulation will take forever.
*Provide the sets of samples that computations have to done for into the `tuples_of_data = [(X,y, "all samples"), (X_1,y_1, "samples class1") , (X_2,y_2", "samples class2")]` list. It consists of tuples with your X and y data as well as the description of the sample for the logging purposes. One tuple represents data for a single, make sure you run your simulation for all the samples and then for samples divided into 3 classes. The ranges of division can be seen in the *label_gross_3* method in  *logistic_regression/classifier_solver.py*

### Running the script
To run the script simply use the command below. Script will produce an indivudal log for each sample set with the timestamp of the time the script was started.
> python3 regressor_solver.py

### Monitoring the health of the run
Most likely the script is going to take 1-2 days to finish a run per one set of samples, so it might about take a week on total. Monitoring the health of the run can be simply done via the **htop** tool. GridSearchCV is configured to use all the CPUs on the VM, hence it should be shown in the CPU use as seen in the image. If your CPU is stalling it means simulation is not running. Moreover, Htop shows the state of the process: if it is running <strong><font color="green">R</font></strong> or stalling <strong><font color="gray">R</font></strong> and for how long. If a single solver sub-process has been running for an over hour and the log file has not been updated, it means your simulation is most likey stuck.
![Htop Example](http://i.imgur.com/Dyiwgor.png)

### Strategy for checking the sanity of the script before running the full simulation
Before running the full pipeline simulation ensure that your model parameter grid is correct. Fill the preprocessors, transfomers, reducers lists to have only the Dummy transfomers in them as `transfomers = [DummyTransformer], preprocessors = [DummyTransformer], reducers = [DummyTransformer]`. As well as configure the tuples_of_data of list to have only one sample item as `tuples_of_data = [(X,y, "all samples")]`. Then run the script and check the end of the log to see if you have any errors, if there are, those are most likey due to misconfiguration. This sanity run could take couple of hours to finish, but is a good idea to check first if your grid is correctly configured before commiting to running the full simulation. Once you are confident in your configuration return the *transfomers*, *preprocessors*, *reducers* and *tuples_of_data* lists to their previous state and run the full simulation.
