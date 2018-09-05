import pandas as pd
import numpy as np

payment = pd.read_excel("FPS_examples_5.xlsx",'Payment Data')
query = pd.read_excel("FPS_examples_5.xlsx",'Query Data')

combined = pd.merge(payment, query, how = 'left', left_on = 'TRANSACTION_REFERENCE_NUM', right_on = 'FPS Ref')

#combined.to_excel("Payment+Query_Merged.xlsx")

final = combined.loc[:,['TRANSACTION_REFERENCE_NUM', 'FPS_IDR', 'SETTLEMENT_AMT','BENEFICIARY_CR_INST_SRT_CDE', 'BENEFICIARY_CUS_ACC_NUM',
       'BENEFICIARY_CUS_ACC_NME', 'ORIGINATING_CR_INST_SRT_CDE',
       'ORIGINATING_CUS_ACC_NUM', 'ORIGINATING_CUS_ACC_NME','Reason for Claim']]

#print(final.dtypes)

# Changing the Dtype of Sort Code and Account Number to Strings for easy processing
final['BENEFICIARY_CR_INST_SRT_CDE'] = final['BENEFICIARY_CR_INST_SRT_CDE'].astype(str)
final['BENEFICIARY_CUS_ACC_NUM'] = final['BENEFICIARY_CUS_ACC_NUM'].astype(str)
final['ORIGINATING_CR_INST_SRT_CDE'] = final['ORIGINATING_CR_INST_SRT_CDE'].astype(str)
final['ORIGINATING_CUS_ACC_NUM'] =final['ORIGINATING_CUS_ACC_NUM'].astype(str)


final['ORIGINATING_SCAN'] = final['ORIGINATING_CR_INST_SRT_CDE'] + final['ORIGINATING_CUS_ACC_NUM']
final['BENEFICIARY_SCAN'] = final['BENEFICIARY_CR_INST_SRT_CDE'] + final['BENEFICIARY_CUS_ACC_NUM']

#SCORE CALCULATION

# S1 : Completeness
def CheckForCompleteness(sortcode, accnum, name) :
	if len(sortcode) == 6 and sortcode.isdigit() and len(accnum) == 8 and accnum.isdigit() and len(name) > 0 and name.isalpha():      # check for length, blank, format
		return 1
	else :
		return 0
final['S1'] = final.apply(lambda x: CheckForCompleteness(x['BENEFICIARY_CR_INST_SRT_CDE'], x['BENEFICIARY_CUS_ACC_NUM'], x['BENEFICIARY_CUS_ACC_NME']), axis=1)

# S2 : Modulo Check
from ukmodulus import validate_number
def ModuloCheck(sortcode, accnum) :
	try :
		if validate_number(sortcode, accnum) == True :
			return 1
	except :
		print('Account Number should be in range(6, 11) and Sort Code should be 6')
		return 0
final['S2'] = final.apply(lambda x: ModuloCheck(x['BENEFICIARY_CR_INST_SRT_CDE'], x['BENEFICIARY_CUS_ACC_NUM']), axis=1)

# S3 : Showstopper failure
previous_failed_SCAN = list(final['BENEFICIARY_SCAN'])
def CheckForPreviousFailures(scan) :
	if scan in previous_failed_SCAN :
		return 1
	else :
		return 0
final['S3'] = final.apply(lambda x: CheckForPreviousFailures(x['BENEFICIARY_CR_INST_SRT_CDE'] + x['BENEFICIARY_CUS_ACC_NUM']), axis=1)

# S6 : STring Match (Similarity)
from pyjarowinkler import distance
import jellyfish
from jaccard_index.jaccard import jaccard_index
import difflib
from fuzzywuzzy import fuzz

#get the list of successful acc-sort-name sets  (from incoming)
SortCode = ['564321','564321','876434','678453']
AccNumber = ['12389559','12389559', '99330022','43781267']
CustName = ["Bhavana", 'Bhavna',"Nimmy",'Richard']

Bag_of_words = pd.DataFrame(list(zip(SortCode,AccNumber,CustName)), columns = ['SortCode','AccNumber','CustName'])
Bag_of_words['SCAN'] = Bag_of_words.apply(lambda x : x['SortCode'] + x['AccNumber'], axis = 1)

def NameMatch(scan, name) :
	similarity_score = []	
	names_with_same_scan = list(Bag_of_words[Bag_of_words['SCAN'] == scan]['CustName'])
	if names_with_same_scan :
		for item in names_with_same_scan :
			similarity_score.append(distance.get_jaro_distance(item, name , winkler=True, scaling=0.1))
			similarity_score.append(jellyfish.jaro_winkler(item, name))
			similarity_score.append(jaccard_index(item,name))
			similarity_score.append(difflib.SequenceMatcher(None,item,name).ratio())	
			similarity_score.append(fuzz.ratio(item,name)/100)
			#similarity_score.append(fuzz.token_sort_ratio(item,name))
		print('The valid names for the given SCAN : ', ', '.join(names_with_same_scan))
		print('The name given for checking the match : ', name)
		print('Printing the similarity_scores : ', similarity_score)
		#print('The scores list for the valid names '+ ','.join(names_with_same_scan)+' with the given name( provided SCAN is matching) ' + name+' : ', similarity_score )
		if max(similarity_score) > 0.8 :
			return 1
		else :
			return 0
	else :
		return 'No previous record of the given SCAN'    # FIND another retur value
print(NameMatch('56432112389559', 'Bhuvana'))
final['S6'] = final.apply(lambda x: NameMatch(x['BENEFICIARY_CR_INST_SRT_CDE'] + x['BENEFICIARY_CUS_ACC_NUM'], x['BENEFICIARY_CUS_ACC_NME']), axis=1)

#S7 : Keyboard Proximity
from keyboard_proximity import *
def KeyboardProximity(scan, name) :
	names_with_same_scan = list(Bag_of_words[Bag_of_words['SCAN'] == scan]['CustName'])
	typo_score = []
	if names_with_same_scan :
		for item in names_with_same_scan :
			typo_score.append(typoDistance(item, name))
		print('The valid names for the given SCAN : ', ', '.join(names_with_same_scan))
		print('The name given for checking the match : ', name)
		print('Printing the typo score (less the better) : ', typo_score)
		if min(typo_score) <= 2 :
			return 1
		else :
			return 0
	else :
		return 'No previous record of the given SCAN'
KeyboardProximity('56432112389559', 'Bhqvana')
final['S7'] = final.apply(lambda x: KeyboardProximity(x['BENEFICIARY_CR_INST_SRT_CDE'] + x['BENEFICIARY_CUS_ACC_NUM'], x['BENEFICIARY_CUS_ACC_NME']), axis=1)

#S8 : Phonetics
import phonetics
def PhoneticsScore(scan,name):
	names_with_same_scan = list(Bag_of_words[Bag_of_words['SCAN'] == scan]['CustName'])
	ph_input_sound = phonetics.metaphone(name)
	ph_bag_words_sound =[]
	score=[]
	for item in names_with_same_scan:
		print('name', item)
		item_sound = phonetics.metaphone(item)
		print('itemsound', item_sound)
		ph_bag_words_sound.append(item_sound)
	print('Input sound', ph_input_sound)
	print('Bag_of_words sound', ph_bag_words_sound)
	for i in ph_bag_words_sound:
	 	fuzz_score = fuzz.ratio(i,ph_input_sound)/100
	 	print(fuzz_score)
	 	score.append(fuzz_score)
	if max(score) > 0.8:
		return 1
	else:
		return 0
print(PhoneticsScore('67845343781267','Rick'))
final['S8'] = final.apply(lambda x: PhoneticsScore(x['BENEFICIARY_CR_INST_SRT_CDE'] + x['BENEFICIARY_CUS_ACC_NUM'], x['BENEFICIARY_CUS_ACC_NME']), axis=1)

#S9 : Nick Name Match
df = pd.read_csv("names.csv")
d = df.set_index('Official').T.to_dict('list')
for i in d.keys() :
    d[i] = [elem.lower() for elem in d[i] if isinstance(elem,str)]

def GetOfficialName(scan, name) :
	names_with_same_scan = list(Bag_of_words[Bag_of_words['SCAN'] == scan]['CustName'])   # possible names for the given SCAN
	official_list = []
	if names_with_same_scan : 
		for i in d.keys() :
		    if name in d[i] :
		        official_list.append(i)
		names_with_same_scan = list(map(lambda x:x.lower(),names_with_same_scan))
		official_list = list(map(lambda x:x.lower(),official_list))
		print('names with scan', names_with_same_scan)
		print('Offiaci', official_list)
		if list(set(names_with_same_scan).intersection(official_list)) :
			return 1
		else :
			return 0
	else :
		return 'The SCAN given doesnt match the previous successful records'

print(GetOfficialName('56432112389558','bhavu'.lower()))
print(GetOfficialName('67845343781267','Rick'.lower()))
final['S9'] = final.apply(lambda x: GetOfficialName(x['BENEFICIARY_CR_INST_SRT_CDE'] + x['BENEFICIARY_CUS_ACC_NUM'], x['BENEFICIARY_CUS_ACC_NME']), axis=1)

#final.to_excel("RequiredColumns.xlsx")