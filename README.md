#This is a masters course project for producing a predictive data mining model. Based on https://www.kaggle.com/deepmatrix/imdb-5000-movie-dataset


## Running the regressor_solver.py script
### Description 
What it is:
What is is not:
What is produces:
*/logistic_regression/regressRes_2017_04_08_173536_case0.log*
### Prerequisites 
*Setup your VM and run the script there. Running script for a single set of samples for your list of models is probably going to take 1-2 days. If you run those locally you can easily break the run, hence you'll need to start all over again.
*Set up a session managment tool like **tmux** and run everything within it. If you simply start running your computations on a ssh bare connection your simulation will be lost.
*Make sure you have enough disk space on your VM if have one set up already. Log files per one run can easily go over hundreds of megs. If you are running a computation for couple of sets of samples it may require couple of gigs of memory. If you run out of mememory big part your computational results are going to be lost. VM has got 50GB of mem per instance.
*Setup the python dependecies on your VM from the requirements.txt file. Requiremnts file is simply a dump of all pip packages that were present on the VM were script was written, some of them might required linux (apt-get) dependecies, please try to resolve those if you run into any issues. Script was written in python3, hence use pip3 to install the dependecies.  
> sudo pip3 install -r requirements.txt
### Setting up the script
*Put all your assigned models into the  `models = [Model1(), Model2(), Model3()]` array. Models array requires the object to be passed and no the class, hence class has be constructed via the call operator `()`. For now scipt contains only the linear regression instance  as `models = [LinearRegression()]`. 
*For each of your models prepopulate the `models_cfg = {}` parameter grid that is passed into the GridSearchCV. Param grid is dictonary of parameters with an array literals of all values that GridSearchCV is going to go through. Each parameter name in the model param grid should star with `model__`. For now it is configured only for the linear regression to have one parameter `models_cfg[LinearRegression.__name__] = dict( model__fit_intercept = [True])`. Look into the *logistic_regression/classifier_solver.py* for some examples. Do not simply copy the parameters over, for most methods they will not simply map. To find what parameters have to tuned look into the [API doc](http://scikit-learn.org/stable/modules/classes.html). As rule of thumb don't try to tune more than 4 params with more than 3-4 variations each, as otherwise your simulation will take forever.
*Provide the sets of samples that computations have to done for into the `tuples_of_data = []` array. 
### Running the script
To run the script simply use the command below. Script will produce an indivudal log 
> python3 regressor_solver.pyMost likely the script is going to take 1-2 days to finish a run per one set of samples, so it might about take a week on total. Monitoring the health of the run can be simply done via the **htop** tool. GridSearchCV is configured to use all the CPUs on the VM, hence it should be shown in the CPU use as seen in the image. If your CPU is stalling it means simulation is not running. Moreover, Htop shows the state of the process if it is running <strong><font color="green">R</font></strong> and for how long. 

![Htop Example](http://i.imgur.com/Dyiwgor.png)

### Monitoring the health of the run
### Strategy for checking the sanity of the script
