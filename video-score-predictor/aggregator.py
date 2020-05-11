import torch
import torch.nn as nn
from aggregator.data import patientDataset
from aggregator.trainer import train, test, lopo, lovo, kfolds
import argparse
from datetime import datetime
from os import path, mkdir

parser = argparse.ArgumentParser()
parser.add_argument("datafile")
parser.add_argument("labelfile")
parser.add_argument("mapfile")
parser.add_argument("outputdir")
parser.add_argument("--testfile", default=None)
parser.add_argument("--expname", default="trial")
parser.add_argument("--tnorm", default="product", choices=['lukasiewicz', 'product'])
parser.add_argument("--off_diagonal", default="min", choices=['min', 'mean', 'max'])
parser.add_argument("--loss", default="ce", choices=['ce', 'kl'])
parser.add_argument("--earlystop", default="last", choices=['last', 'train_loss', 'train_acc'])
parser.add_argument('--setting', default='traintest', choices=['traintest', 'lopo', 'lovo', 'kfolds'])
parser.add_argument("--epochs", default=20, type=int)
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--normalize_neutral", action='store_true', default=False)
parser.add_argument("--use_majority_label", action='store_true', default=False)
parser.add_argument("--use_binary_labels", action='store_true', default=False)
parser.add_argument("--use_score_hierarchy", action='store_true', default=False)
parser.add_argument("--rebalance_scores", action='store_true', default=False)
parser.add_argument("--use_sord", action='store_true', default=False)
parser.add_argument("--multithread", action='store_true', default=False)
parser.add_argument("--stratified", action='store_true', default=False)
parser.add_argument("--zero_score_gap", default=0.5, type=float)
parser.add_argument("--init_neutral", default=0., type=float)
parser.add_argument("--lr_gamma", default=1/3, type=float)
parser.add_argument("--numfolds", default=5, type=int)
parser.add_argument("--activate_linear", default=0, type=int, help="Activate linear layer after <val> iterations (0=no activation)")
args = parser.parse_args()

dataset = patientDataset(args.datafile, args.labelfile, args.mapfile, use_majority_label=args.use_majority_label, use_binary_labels=args.use_binary_labels)

workdir = path.join(args.outputdir, datetime.now().isoformat())
mkdir(workdir)
outprefix = path.join(workdir, args.expname)

# TERRIBLE PATCH BEFORE IMPLEMENTING PROPER LOGGING
import sys
sys.stdout = open(outprefix + ".log", 'w')


if args.testfile:
	testset = patientDataset(args.testfile, args.labelfile, args.mapfile, use_majority_label=args.use_majority_label, use_binary_labels=args.use_binary_labels)
else:
	testset = dataset

score_range = dataset.get_score_range()

if args.setting == 'lopo':
	lopo(dataset, outprefix, score_range, args)
elif args.setting == 'lovo':
	lovo(dataset, outprefix, score_range, args)
elif args.setting == 'kfolds':
	kfolds(dataset, outprefix, score_range, args)
else:
	net = train(dataset, outprefix + "_model", score_range, args)
	test(net, testset, outprefix + "_preds", score_range)
