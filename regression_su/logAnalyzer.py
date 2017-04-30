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
parser.add_argument("measure", help="Measue mode: high, median ", metavar="measure")

args = parser.parse_args()
results_all = []
with open(args.file) as f:
    for line in f:
        if "Best param" in line and "print (" not in line:
            try:
                pipeline = line
                cfg = next(f)
                _ = next(f)
                score = next(f)
                model = re.search(r'(?:\[)(.*)(?:\])', pipeline).group(1)
                preprocessor = re.search(r'(?:preprocessor\:)(.*? |)', pipeline).group(1)
                transfomer = re.search(r'(?:transfomer\: )(.*? |)', pipeline).group(1)
                reducer = re.search(r'(?:reducer\: )(.*? )', pipeline).group(1)
                model_cfg = [tupl[0] for tupl in re.findall(r'(?:model__)(.*?)(,|})', cfg)]
                preprocessor_cfg = [tupl[0] for tupl in re.findall(r'(?:preprocessor__)(.*?)(,|})', cfg)]
                transfomer_cfg = [tupl[0] for tupl in re.findall(r'(?:transfomer__)(.*?)(,|})', cfg)]
                reducer_cfg = [tupl[0] for tupl in re.findall(r'(?:reducer__)(.*?)(,|})', cfg)]
                #_str = "|" + str(score) + " (" + model + "):" + str(model_cfg) + " | " + preprocessor + ": " + str(preprocessor_cfg) + " " + transfomer + ": " + str(transfomer_cfg) + " " + reducer + ": " + str(reducer_cfg)
                results_all.append({
                                "score":float(score),
                                "model":model,
                                "preprocessor":preprocessor,
                                "transformer":transfomer,
                                "reducer":reducer,
                                "model_cfg":model_cfg,
                                "preprocessor_cfg":preprocessor_cfg,
                                "transfomer_cfg":transfomer_cfg,
                                "reducer_cfg":reducer_cfg
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
        _str = str(item['score']) + " (" + item['model'] + "):" + str(item['model_cfg']) + " | " + item['preprocessor'] + ": " + str(item['preprocessor_cfg']) + " " + item['transformer'] + ": " + str(item['transfomer_cfg']) + " " + item['reducer'] + ": " + str(item['reducer_cfg'])
        print(_str)
_sorted_all = sorted(results_all, key=lambda k: k['score'])        

if args.mode != "all":
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
        
else:
    print_results(_sorted_all)
    
