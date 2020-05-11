import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.functional import softmax
from aggregator.nn import UninormAggregator, ScoreHierarchyNet, CovidNoCovidNet
from aggregator.util import *
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from threading import Thread
import pickle

class TrainThread(Thread):

	def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}):
		super(TrainThread, self).__init__(group, target, name, args, kwargs)
		self.trainset = args[0]
		self.testset = args[1]
		self.modelfile = args[2]
		self.outputfile = args[3]
		self.score_range = args[4]
		self.args = args[5]
		self._return = 0

	def run(self):
		net = train(self.trainset, self.modelfile, self.score_range, self.args)
		self._return = test(net, self.testset, self.outputfile, self.score_range) 

	def join(self, *args):
		Thread.join(self, *args)
		return self._return

def train(dataset, modelfile, score_range, args):
	
	if args.use_binary_labels:
		net = CovidNoCovidNet(score_range, tnorm=args.tnorm, normalize_neutral=args.normalize_neutral, init_neutral=args.init_neutral)
	elif args.use_score_hierarchy:
		net = ScoreHierarchyNet(score_range, tnorm=args.tnorm, normalize_neutral=args.normalize_neutral, init_neutral=args.init_neutral)
	else:
		net = UninormAggregator(score_range, tnorm=args.tnorm, normalize_neutral=args.normalize_neutral, init_neutral=args.init_neutral)

	optimizer = optim.Adam(net.parameters(), lr=args.lr)#, weight_decay=0.01)
	lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [15, 20, 25], gamma=args.lr_gamma)

	if args.loss == "kl":
		criterion = kl_div_loss
	else:
		criterion = cross_entropy_loss

	if args.rebalance_scores:
		score_weight = dataset.compute_score_weights()
	else:
		score_weight = None

	max_accuracy = 0.0
	max_loss = 0.0
	
	for epoch in range(args.epochs):
		running_loss = 0.0		
		running_accuracy = 0.0					
		num = 0
		if args.activate_linear and args.activate_linear == epoch: 
			net.activate_linear()
			optimizer.add_param_group({"params": net.fc.parameters()})
		optimizer.zero_grad()	
		for x, label in dataset:			
			y = net(x).view(1,-1)			
			loss = criterion(y, label, use_sord=args.use_sord, zero_score_gap=args.zero_score_gap, weight=score_weight)
			loss.backward()			
			#torch.nn.utils.clip_grad_norm_(net.parameters(), 0.005)						
			running_loss += loss.item()					
			running_accuracy += label == torch.argmax(y)
			num +=1			
		running_accuracy /= num
		running_loss /= num		
		print("parameters")
		net.print_parameters()
		print("gradient")
		net.print_gradient()
		print('[%d] accuracy: %.3f' % (epoch + 1, running_accuracy))
		print('[%d] loss: %.3f' % (epoch + 1, running_loss))		
		max_accuracy, max_loss = save_checkpoint(net, modelfile, args.earlystop, running_accuracy, running_loss, max_accuracy, max_loss)
		optimizer.step()
		if not args.normalize_neutral:
			net.clamp_params()
		lr_scheduler.step()
	
	if args.earlystop != "last":
		net.load_state_dict(torch.load(modelfile))
	
	print("learned parameters")
	net.print_parameters()
	
	return net

def save_checkpoint(net, modelfile, earlystop, running_accuracy, running_loss, max_accuracy, max_loss):
	if running_accuracy > max_accuracy:
		max_accuracy = running_accuracy
		if earlystop == 'train_acc':
			torch.save(net.state_dict(), modelfile)
	if running_loss > max_loss:
		max_loss = running_loss
		if earlystop == 'train_loss':
			torch.save(net.state_dict(), modelfile)			
	return max_accuracy, max_loss

def test(net, testset, outputfile, score_range):

	inputs=[]
	labels=[]		
	outputs=[]

	with torch.no_grad():
		for x, label in testset:
			inputs.append(x)
			labels.append(label)	
			outputs.append(net(x))		

		print("Predictor results")
		preds=[torch.argmax(o) for o in outputs]
		print_results(labels,preds,score_range)
		save_predictions(labels,preds,outputfile)

	baselines = (argmax_mean, max_argmax) #, argmax_count_argmax, max_thres_count_argmax_5, max_thres_count_argmax_10, max_thres_count_argmax_15)

	for baseline in baselines:
		baseline_preds=[]
		for x, label in testset:
			baseline_preds.append(baseline(x))
		print("Baseline <%s> results" %baseline.__name__)
		print_results(labels,baseline_preds,score_range)	

	return inputs,labels,outputs,preds

def evaluate(inputs, labels, outputs, preds, outprefix, score_range, kfolds=False):

	baselines = (argmax_mean, max_argmax) #, argmax_count_argmax, max_thres_count_argmax_5, max_thres_count_argmax_10, max_thres_count_argmax_15)

	if kfolds:	
		print("Predictor average results")	
		print_average_results(labels, preds)
		
		for baseline in baselines:
			baseline_preds=[]
			for fold_inputs in inputs:
				baseline_preds.append([baseline(x) for x in fold_inputs])
			print("Baseline <%s> average results" %baseline.__name__)
			print_average_results(labels,baseline_preds)

		inputs = flatten(inputs)
		labels = flatten(labels)
		outputs = flatten(outputs)
		preds = flatten(preds)

	print("Predictor overall results")
	print_results(labels,preds,score_range)
	save_predictions(labels,preds,outprefix + "_preds")
	
	for baseline in baselines:
		baseline_preds=[]
		for x, label in zip(inputs, labels):
			baseline_preds.append(baseline(x))
		print("Baseline <%s> overall results" %baseline.__name__)
		print_results(labels,baseline_preds,score_range)
	
	compute_roc_curve(inputs, labels, outputs, preds, outprefix, score_range)

def lopo(dataset, outprefix, score_range, args):

	inputs=[]
	labels=[]
	outputs=[]	
	preds=[]

	for patient in dataset.get_patient_indices():
		print("LOPO computation for patient ", patient)
		trainset = dataset.exclude_patient(patient)
		testset = dataset.get_patient(patient)
		net = train(trainset, "%s_model.%s" %(outprefix,str(patient)), score_range, args)
		with torch.no_grad():
			for x, label in testset:				
				inputs.append(x)
				labels.append(label)
				outputs.append(net(x))
				preds.append(torch.argmax(outputs[-1]))

	evaluate(inputs, labels, outputs, preds, outprefix, score_range)

def lovo(dataset, outprefix, score_range, args):

	inputs=[]
	labels=[]
	outputs=[]	
	preds=[]

	for video in dataset.get_indices():
		print("LOVO computation for video ", video[2])
		trainset = dataset.exclude_video(video)
		testset = dataset.get_video(video)
		net = train(trainset, "%s_model.%s" %(outprefix,video[2]), score_range, args)
		with torch.no_grad():
			for x, label in testset:				
				inputs.append(x)
				labels.append(label)
				outputs.append(net(x))
				preds.append(torch.argmax(outputs[-1]))

	evaluate(inputs, labels, outputs, preds, outprefix, score_range)


def kfolds(dataset, outprefix, score_range, args):

	allinputs=[]
	alllabels=[]
	alloutputs=[]
	allpreds=[]	
	numfolds = args.numfolds
	if args.stratified:
		splits = dataset.get_stratified_kfold_splits(numfolds, score_range)
	else:
		splits = dataset.get_kfold_splits(numfolds)
	all = dataset.get_patient_indices()
	train_threads = []

	print("Splits")
	print(splits)
	for i,split in enumerate(splits,0):		
		trainset = dataset.get_patients(all.difference(split))
		testset = dataset.get_patients(split)
		if args.multithread:
			print("Starting thread ", i)
			train_thread = TrainThread(args=(trainset, testset, 
											 "%s_model.%d" %(outprefix,i), 
											 "%s_preds.%d" %(outprefix,i), 
											 score_range, args))
			train_thread.start()
			train_threads.append(train_thread)
		else:
			print("Running fold ", i)
			net = train(trainset, "%s_model.%d" %(outprefix,i), score_range, args)		
			inputs,labels,outputs,preds = test(net, testset, "%s_preds.%d" %(outprefix,i), score_range)
			allinputs.append(inputs)
			alllabels.append(labels)
			alloutputs.append(outputs)
			allpreds.append(preds)

	if args.multithread:
		for train_thread in train_threads:				
			inputs,labels,outputs,preds = train_thread.join()
			allinputs.append(inputs)
			alllabels.append(labels)
			alloutputs.append(outputs)
			allpreds.append(preds)

	evaluate(allinputs, alllabels, alloutputs, allpreds, outprefix, score_range, kfolds=True)

def print_results(labels,preds,score_range):
	print(confusion_matrix(labels,preds))
	print("\nweighted f1 = %.3f" %f1_score(labels,preds, average='weighted'))
	print("weighted pre = %.3f" %precision_score(labels,preds, average='weighted'))
	print("weighted rec = %.3f" %recall_score(labels,preds, average='weighted'))
	print("accuracy = %.3f" %accuracy_score(labels,preds))
	print("covid/nocovid accuracy = %.3f\n" %covid_nocovid_accuracy_score(labels, preds))
	# target_names = ['score %d' %i for i in range(score_range)]
	# print(classification_report(labels,preds, target_names = ['score 0', 'score 1', 'score 2', 'score 3']))


def print_average_results(labels, preds):
	nfolds=len(labels)
	weighted_f1 = np.zeros(nfolds)
	weighted_pre = np.zeros(nfolds)
	weighted_rec = np.zeros(nfolds)
	accuracy = np.zeros(nfolds)
	covid_nocovid_accuracy = np.zeros(nfolds)
	for i,(l,p) in enumerate(zip(labels, preds),0):			
		weighted_f1[i]=f1_score(l,p, average='weighted')
		weighted_pre[i]=precision_score(l,p, average='weighted')
		weighted_rec[i]=recall_score(l,p, average='weighted')
		accuracy[i]=accuracy_score(l,p)
		covid_nocovid_accuracy[i]=covid_nocovid_accuracy_score(l,p)
	print("weighted f1 = %.3f += %.3f" %(weighted_f1.mean(),weighted_f1.std())) 
	print("weighted pre = %.3f += %.3f" %(weighted_pre.mean(),weighted_pre.std()))
	print("weighted rec = %.3f += %.3f" %(weighted_rec.mean(),weighted_rec.std()))
	print("accuracy = %.3f += %.3f" %(accuracy.mean(),accuracy.std()))
	print("covid/nocovid accuracy = %.3f += %.3f" %(covid_nocovid_accuracy.mean(),covid_nocovid_accuracy.std()))

def save_predictions(labels,preds,outputfile):	
	with open(outputfile, "w") as f:
		np.savetxt(f, torch.tensor([labels,preds], dtype=torch.int8).t(), fmt='%d') 

def covid_nocovid_accuracy_score(labels, preds):
	binary_labels = [min(l,1) for l in labels]
	binary_preds = [min(p,1) for p in preds]
	return accuracy_score(binary_labels,binary_preds) 		

def compute_roc_curve(inputs, labels, outputs, preds, outprefix, score_range, curve_type='scorewise'):

	if curve_type=='micro_average':
		onehot_labels = label_binarize(labels, classes=list(range(score_range))).ravel()
		aggregator_outputs = flatten([softmax(o, dim=0).detach().numpy() for o in outputs])		
		baseline_outputs = flatten([softmax(torch.mean(x, dim=1), dim=0).numpy() for x in inputs])
		plot_roc_curve(onehot_labels, aggregator_outputs, baseline_outputs,
						outprefix + "_roc_micro_average", 
						"Micro-averaged ROC curve")
	elif curve_type=='binary_classification':
		binary_labels = [min(l,1) for l in labels]
		aggregator_outputs = [torch.sum(softmax(o, dim=0)[1:]).detach().numpy() for o in outputs]		
		baseline_outputs = [torch.sum(softmax(torch.mean(x,dim=1), dim=0)[1:]) for x in inputs]
		plot_roc_curve(binary_labels, aggregator_outputs, baseline_outputs,
						outprefix + "_roc_binary_classification", 
						"Covid/NoCovid ROC curve")
	if curve_type=='scorewise':
		for s in range(score_range):
			score_labels = [l == s and 1 or 0 for l in labels]
			aggregator_outputs = [softmax(o, dim=0)[s].detach().numpy() for o in outputs]		
			baseline_outputs = [softmax(torch.mean(x,dim=1), dim=0)[s] for x in inputs]
			plot_roc_curve(score_labels, aggregator_outputs, baseline_outputs,
							"%s_roc_score_%d" %(outprefix,s), 
							"ROC curve for score %d" %s)
	else:
		raise Exception("Unknown curve_type <%s> for roc_curve computation" %curve_type)

def plot_roc_curve(labels, aggregator_outputs, baseline_outputs, fileprefix, title):
		
	aggregator_fpr, aggregator_tpr, _ = roc_curve(labels, aggregator_outputs)
	aggregator_roc_auc = roc_auc_score(labels, aggregator_outputs)
	baseline_fpr, baseline_tpr, _ = roc_curve(labels, baseline_outputs)
	baseline_roc_auc = roc_auc_score(labels, baseline_outputs)
	curve_file = fileprefix + ".pdf"
	data_file = fileprefix + ".pkl"	
	with open(data_file, "wb") as f:
		pickle.dump((aggregator_fpr, aggregator_tpr, aggregator_roc_auc), f)
		pickle.dump((baseline_fpr, baseline_tpr, baseline_roc_auc), f)
	print("Saving ROC curve to file: " + curve_file)
	plt.clf()
	plt.plot(aggregator_fpr, aggregator_tpr, color='blue', label='video-based predictor (AUC = %.2f)' %aggregator_roc_auc)
	plt.plot(baseline_fpr, baseline_tpr, color='green', label='mean baseline (AUC = %.2f)' %baseline_roc_auc)
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title(title)
	plt.legend(loc="lower right")
	plt.savefig(curve_file)


