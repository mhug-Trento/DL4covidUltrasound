from torch.utils.data import Dataset
import torch.tensor
import pandas as pd
import numpy as np
from scipy.stats import describe

def majority_label(labeldir):
    from os import walk
    from os import path
    from functools import reduce
    labels4annotator = []
    for root,_,filenames in walk(labeldir):
        for filename in filenames:            
            if not 'Score' in filename:
                continue          
            print('Processing file ', filename)  
            labels4annotator.append(pd.read_excel(path.join(root,filename)))
    merged_labels = reduce(lambda left,right: pd.merge(left, right, on='Video'), labels4annotator)
    majority_labels = merged_labels.iloc[:,1:].mode(axis=1).max(axis=1)
    majority_labels = pd.concat([merged_labels['Video'], majority_labels], axis=1)
    majority_labels.columns = ['Video','Score']
    return majority_labels

class patientDataset(Dataset):
    
    def __init__(self, predfile=None, labelfile=None, mapfile=None, data=None, use_majority_label=False, use_binary_labels=False):
        if predfile:
            preds = pd.read_pickle(predfile).rename(columns={'filenames' : 'filename'})
            if use_majority_label:
                labels = majority_label(labelfile)
            else:
                labels = pd.read_excel(labelfile)
            mapping = pd.read_excel(mapfile)
            labelmap = mapping.merge(labels,on='Video')
            preds['filename']=preds['filename'].map(lambda s : s.replace(".mat",""))
            #self.data = pd.merge(preds,labelmap, on=['hospital', 'patient', 'filename'])
            self.data = pd.merge(preds,labelmap, on=['filename'])
        else:  
            self.data = data
        if use_binary_labels:
            self.data['Score'] = self.data['Score'].apply(lambda x: min(x,1))
        #self.videos = self.data.groupby(['hospital','patient','filename'])
        self.videos = self.data.groupby(['hospital_x','patient_x','filename'])

    def __getitem__(self, index):            
        return self.get_input(index), self.get_target(index)

    def __len__(self):
        return len(self.videos)

    def __iter__(self):
        return (self.__getitem__(i) for i in self.get_indices())            

    def get_indices(self):
        return list(self.videos.groups.keys())

    def get_input(self, index):
        data = self.videos.get_group(index)     
        return torch.tensor(np.array(data['scores'].values.tolist()).transpose(), dtype=torch.float32) 
        
    def get_target(self, index):
        data = self.videos.get_group(index)     
        return torch.tensor([data['Score'].iloc[0]])

    def get_patient_indices(self):
        return set([(hospital,patient) for (hospital,patient,filename) in self.videos.groups.keys()])

    def get_hospitals(self):
        return set([(hospital) for (hospital,patient,filename) in self.videos.groups.keys()])        

    def get_hospital_patients(self, hospital):
        return set([(hosp,patient) for (hosp,patient,filename) in self.videos.groups.keys() if hosp == hospital])

    def get_patient(self, patient):
        patient_data = self.data.query('hospital_x == "%s" and patient_x == "%s"' %(patient[0],patient[1]))
        return patientDataset(data=patient_data)

    def exclude_patient(self, patient):
        other_patients_data = self.data.query('hospital_x != "%s" or patient_x != "%s"' %(patient[0],patient[1]))        
        return patientDataset(data=other_patients_data)

    def get_video(self, video):
        video_data = self.data.query('hospital_x == "%s" and patient_x == "%s" and filename == "%s"' %(video[0],video[1],video[2]))
        return patientDataset(data=video_data)

    def exclude_video(self, video):
        other_videos_data = self.data.query('hospital_x != "%s" or patient_x != "%s" or filename != "%s"' %(video[0],video[1],video[2]))
        return patientDataset(data=other_videos_data)

    def get_score_range(self):
        return len(self.data['scores'][0])

    def get_patients(self, patients):
        selection = pd.DataFrame(patients, columns=['hospital_x','patient_x'])
        patients_data = pd.merge(self.data, selection, on=['hospital_x','patient_x'])
        return patientDataset(data=patients_data)

    def get_target_stats(self):
        return self.videos.head(1)['Score'].value_counts(sort=False)  

    def get_kfold_splits(self, k):
        splits = [set() for i in range(k)]
        i = 0
        for hospital in self.get_hospitals():
            for patient in self.get_hospital_patients(hospital):
                splits[i % k].add(patient)
                i+=1
        return splits

    def get_stratified_kfold_splits(self, k, score_range):
        splits = [set() for i in range(k)] 
        splits_stats = pd.DataFrame(data=np.zeros((score_range,k)), index=range(score_range), columns=range(k))
        hospitals = list(self.get_hospitals())
        hospitals.sort()
        for hospital in hospitals:
            hospital_patients = list(self.get_hospital_patients(hospital))
            hospital_patients.sort()
            for patient in hospital_patients:
                patient_stats = self.get_patient(patient).get_target_stats()
                split = splits_stats.sub(patient_stats, axis=0).sum(axis=0).argmin()
                splits[split].add(patient)
                splits_stats[split] = splits_stats[split].add(patient_stats, axis=0, fill_value=0)
        print("Splits label distribution")
        print(splits_stats)
        return splits

    def print_stats(self):
        data = np.array(self.data['scores'].values.tolist())
        print(describe(data))
        print(describe(data.max(1)))

    def compute_score_weights(self):
        counts = self.videos.head(1)['Score'].value_counts(sort=False).to_numpy(dtype=float)
        counts[0] = counts[1:].sum()/counts[0]
        counts[1:] = counts[1:].max()/counts[1:]
        return torch.tensor(counts, dtype=torch.float32)


