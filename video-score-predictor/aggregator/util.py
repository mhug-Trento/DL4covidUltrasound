import torch
import torch.nn.functional as F
import numpy as np

def flatten(l):
	return [item for sublist in l for item in sublist]

def argmax_mean(x):
	return torch.argmax(torch.mean(x,dim=1))

def max_argmax(x):
	return torch.max(torch.argmax(x,dim=0))

def argmax_count_argmax(x):
	unique,counts = torch.argmax(x,dim=0).unique(return_counts=True)
	return unique[torch.argmax(counts)]

def max_thres_count_argmax(x, thres):
	unique,counts = torch.argmax(x,dim=0).unique(return_counts=True)
	return torch.max(unique[counts >= x.shape[1] * thres])

def max_thres_count_argmax_5(x):
	return max_thres_count_argmax(x, 0.05)

def max_thres_count_argmax_10(x):
	return max_thres_count_argmax(x, 0.1)

def max_thres_count_argmax_15(x):
	return max_thres_count_argmax(x, 0.15)

def sord_labels(label, num_classes, zero_score_gap=0.5):
	batch_size = label.shape[0]
	labels_sord = np.zeros((batch_size, num_classes))
	for element_idx in range(batch_size):
		current_label = label[element_idx]
		for class_idx in range(num_classes):
			current_label_weighted = current_label
			class_idx_weighted = class_idx
			if zero_score_gap:
				if current_label == 0:
					current_label_weighted = -zero_score_gap
				if class_idx == 0:
					class_idx_weighted = -zero_score_gap
			labels_sord[element_idx][class_idx] = 2 * abs(current_label_weighted - class_idx_weighted) ** 2
	labels_sord = torch.from_numpy(labels_sord)#.cuda(non_blocking=True)
	return F.softmax(-labels_sord, dim=1)

def cross_entropy_loss(y, label, use_sord=False, zero_score_gap=0.5, weight=None):
	if use_sord:
		labels = sord_labels(label, y.shape[1], zero_score_gap)
	else:
		labels = torch.zeros(y.shape[1])
		labels[label] = 1
	log_predictions = F.log_softmax(y, 1)
	if weight != None:
		return (-weight * labels * log_predictions).sum(dim=1).mean()
	else:
		return (-labels * log_predictions).sum(dim=1).mean()	

def kl_div_loss(y, label, use_sord=True, zero_score_gap=0.5, weight=None):
	assert(use_sord)
	assert(weight == None)
	label = sord_labels(label, y.shape[1], zero_score_gap).float()
	log_predictions = F.log_softmax(y, 1)
	loss = torch.nn.KLDivLoss(reduction='batchmean')
	return loss(y, label)
