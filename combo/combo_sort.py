import logging
from argparse import ArgumentParser
import sys
import os
from time import time
import re
import numpy


# parse commandline arguments
parser = ArgumentParser()
parser.add_argument("file", help="input log file for sorting results", metavar="FILE")
parser.add_argument("mode", help="Output mode: all, model, transformer, reducer, preprocessor ", metavar="MODE")
#parser.add_argument("score", help="Score: class_train_sc,class_valid_sc,class_test_sc,reg_train_sc,reg_valid_sc,reg_test_sc,reg_cl_train_sc,reg_cl_valid_sc,reg_cl_test_sc", metavar="score")
parser.add_argument("score", help="Score: class_train_sc,class_test_sc,reg_cl_train_sc,reg_cl_valid_sc,reg_cl_test_sc", metavar="score")
args = parser.parse_args()
results_all = []
with open(args.file) as f:
    for line in f:
      	if "Precomp class" in line and "print (" not in line and "print(" not in line:
            try:
                ###clasify
                class_pipeline =  line
                class_pipeline_cfg = next(f)
                _ = next(f)
                class_cfg = next(f)
                _ = next(f)
                class_train_sc = next(f)
                _ = next(f)
                class_valid_sc = next(f)
                _ = next(f)
                class_test_sc = next(f)
                class_model = re.search(r'(?:\[)(.*)(?:\])', class_pipeline).group(1)
                class_preprocessor = re.search(r'(?:preprocessor\:)(.*? |)', class_pipeline).group(1)
                class_transfomer = re.search(r'(?:transfomer\: )(.*? |)', class_pipeline).group(1)
                class_reducer = re.search(r'(?:reducer\: )(.*? )', class_pipeline).group(1)
                class_model_cfg = [tupl[0] for tupl in re.findall(r'(?:model__)(.*?)(,|})', class_cfg)]
                class_preprocessor_cfg = [tupl[0] for tupl in re.findall(r'(?:preprocessor__)(.*?)(,|})', class_pipeline_cfg)]
                class_transfomer_cfg = [tupl[0] for tupl in re.findall(r'(?:transfomer__)(.*?)(,|})', class_pipeline_cfg)]
                class_reducer_cfg = [tupl[0] for tupl in re.findall(r'(?:reducer__)(.*?)(,|})', class_pipeline_cfg)]

                '''
                ##regress
                reg_pipeline =  next(f)
                reg_pipeline_cfg = next(f)
                _ = next(f)
                reg_cfg = next(f)
                _ = next(f)
                reg_train_sc = next(f)
                _ = next(f)
                reg_valid_sc = next(f)
                _ = next(f)
                reg_test_sc = next(f)
                reg_model = re.search(r'(?:\[)(.*)(?:\])', reg_pipeline).group(1)
                reg_preprocessor = re.search(r'(?:preprocessor\:)(.*? |)', reg_pipeline).group(1)
                reg_transfomer = re.search(r'(?:transfomer\: )(.*? |)', reg_pipeline).group(1)
                reg_reducer = re.search(r'(?:reducer\: )(.*? )', reg_pipeline).group(1)
                reg_model_cfg = [tupl[0] for tupl in re.findall(r'(?:model__)(.*?)(,|})', reg_cfg)]
                reg_preprocessor_cfg = [tupl[0] for tupl in re.findall(r'(?:preprocessor__)(.*?)(,|})', reg_pipeline_cfg)]
                reg_transfomer_cfg = [tupl[0] for tupl in re.findall(r'(?:transfomer__)(.*?)(,|})', reg_pipeline_cfg)]
                reg_reducer_cfg = [tupl[0] for tupl in re.findall(r'(?:reducer__)(.*?)(,|})', reg_pipeline_cfg)]
               '''
                
                ##regress with clasess
                _ =  next(f)
                _ =  next(f)
                reg_cl_train_sc = next(f)
                _ = next(f)
                reg_cl_valid_sc = next(f)
                _ = next(f)
                reg_cl_test_sc = next(f)
                
                results_all.append({
                        "class_train_sc":	float(class_train_sc),
                        "class_valid_sc":	float(class_valid_sc),
                        #"class_valid_sc":	'n.nnnnnnnnnnn',
                        "class_test_sc":	float(class_test_sc),
                        "class_model":	class_model,
                        "class_preprocessor":	class_preprocessor,
                        "class_transfomer":	class_transfomer,
                        "class_reducer":	class_reducer,
                        "class_model_cfg":	class_model_cfg,
                        "class_preprocessor_cfg":	class_preprocessor_cfg,
                        "class_transfomer_cfg":	class_transfomer_cfg,
                        "class_reducer_cfg":	class_reducer_cfg #,
                        
                        
                        #"reg_train_sc":	float(reg_train_sc),
                        #"reg_valid_sc":	float(reg_valid_sc),
                        #"reg_test_sc":	float(reg_test_sc),
                        #"reg_train_sc":	'n.nnnnnnnnnnn',
                        #"reg_valid_sc":	'n.nnnnnnnnnnn',
                        #"reg_test_sc": 'n.nnnnnnnnnnn',
                        #"reg_model":	reg_model,
                        #"reg_preprocessor":	reg_preprocessor,
                        #"reg_transfomer":	reg_transfomer,
                        #"reg_reducer":	reg_reducer,
                        #"reg_model_cfg":	reg_model_cfg,
                        #"reg_preprocessor_cfg":	reg_preprocessor_cfg,
                        #"reg_transfomer_cfg":	reg_transfomer_cfg,
                        #"reg_reducer_cfg":	reg_reducer_cfg,
                    
                        #"reg_cl_train_sc":	float(reg_cl_train_sc),
                        #"reg_cl_valid_sc":	float(reg_cl_valid_sc),
                        #"reg_cl_test_sc":	float(reg_cl_test_sc)
                        
                })
            except:
                print("Error while processing:")
                print(sys.exc_info()[0])
                print("Line:")
                print(line)
                raise
                

##helper 
def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

def median(lst):
    return numpy.median(numpy.array(lst))

def print_results(itt):
    for item in itt:
        
        str_class = str(item['class_train_sc']) + " " + str(item['class_valid_sc']) + " " + str(item['class_test_sc']) +  " (" + item['class_model'] + "):" + str(item['class_model_cfg']) + " | " + item['class_preprocessor'] + ": " + str(item['class_preprocessor_cfg']) + " " + item['class_transfomer'] + ": " + str(item['class_transfomer_cfg']) + " " + item['class_reducer'] + ": " + str(item['class_reducer_cfg'])
        
        #str_reg = str(item['reg_train_sc']) + " " + str(item['reg_valid_sc']) + " " + str(item['reg_test_sc']) +  " (" + item['reg_model'] + "):" + str(item['reg_model_cfg']) + " | " + item['reg_preprocessor'] + ": " + str(item['reg_preprocessor_cfg']) + " " + item['reg_transfomer'] + ": " + str(item['reg_transfomer_cfg']) + " " + item['reg_reducer'] + ": " + str(item['reg_reducer_cfg'])
        
        #str_class = str(item['class_train_sc']) + " " + str(item['class_test_sc']) +  " (" + item['class_model'] + "):" + str(item['class_model_cfg']) + " | " + item['class_preprocessor'] + ": " + str(item['class_preprocessor_cfg']) + " " + item['class_transfomer'] + ": " + str(item['class_transfomer_cfg']) + " " + item['class_reducer'] + ": " + str(item['class_reducer_cfg'])
        
        #str_reg = " (" + item['reg_model'] + "):" + str(item['reg_model_cfg']) + " | " + item['reg_preprocessor'] + ": " + str(item['reg_preprocessor_cfg']) + " " + item['reg_transfomer'] + ": " + str(item['reg_transfomer_cfg']) + " " + item['reg_reducer'] + ": " + str(item['reg_reducer_cfg'])
        
        #str_reg_cl = str(item['reg_cl_train_sc']) + " " + str(item['reg_cl_valid_sc']) + " " + str(item['reg_cl_test_sc'])
        
        
        print(str_class)
        #print(str_reg)
        #print(str_reg_cl)
        print("")
        
_sorted_all = sorted(results_all, key=lambda k: k[args.score])        

if args.mode != "all":
    print("Other modes not supported")
    '''
    amount = 10
    model_list = [{item[args.mode]: removekey(item, args.mode)} for item in _sorted_all]
    model_dict = {item[args.mode]: [] for item in _sorted_all}
    for lst in model_list:
        model_dict[lst.keys()[0]].append(lst.values())

    info_lst = []
    for el in model_dict:
        scores = [lst[0]['score'] for lst in model_dict[el]]
        _sorted =  sorted(model_dict[el], key=lambda k: k[0]['score'])
        info_lst.append([{args.mode: el, "median":median(scores), "high": max(scores), "low":min(scores)}, _sorted[-amount:]])

    _sorted_models = sorted(info_lst, key=lambda k: k[0][args.measure])

    for items in _sorted_models:
        _str = args.mode + ": " + items[0][args.mode] + " median: " + str(items[0]["median"]) + " high: " + str(items[0]["high"]) + " low: " + str(items[0]["low"]) 
        print(_str)
        print("top " + str(amount) + "scores are:")
        for elem in items[1]:
            item = elem[0]
            item[args.mode] = items[0][args.mode]
            _str = str(item['score']) + " (" + item['model'] + "):" + str(item['model_cfg']) + " | " + item['preprocessor'] + ": " + str(item['preprocessor_cfg']) + " " + item['transformer'] + ": " + str(item['transfomer_cfg']) + " " + item['reducer'] + ": " + str(item['reducer_cfg'])
            print(_str)
        print("")
        '''
        
else:
    print_results(_sorted_all)
    print("Total results found: " + str(len(_sorted_all)))
    #filtered1 = filter(lambda x: x['reg_cl_test_sc'] - x['reg_cl_valid_sc'] < 0.03 , _sorted_all)
    #print_results(filtered1)
    #print("Filter results found: " + str(len(filtered1)))