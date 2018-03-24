import pandas as pd
import pickle


def buildDataSet():
	#Import Ingredients DF
	print('Loaded Products...')
	ewg_ing_df = pd.read_json('ingredients_products_keys_fixed/ewg_ingredients.json', orient = 'index')

	#Build mapping between Ingredient ID and ingredient Name
	ing_map = {}
	for i in range(len(ewg_ing_df)):
	    ID = ewg_ing_df.iloc[i]['ingredient_id']
	    name = ewg_ing_df.iloc[i]['ingredient_name']
	    ing_map[ID] = name


	#Read in Product Data and Initialize Acne Score
	ewg_prd_df = pd.read_json('ingredients_products_keys_fixed/ewg_products.json', orient = 'index')
	ewg_prd_df['Acne_Score'] = 0
	print('Loaded ingredients')

	#Build Lists of ingredients to modify original DataFrame and Initialize Dataset for Model
	from collections import Counter
	n = len(ewg_prd_df)
	ing_lists = []
	ing_cnts = Counter()
	string_lists = []
	for i in range(n):
	    try:
	        new_list = []
	        strings = ''
	        ing_list = ewg_prd_df.iloc[i]['ingredient_list']
	        for ID in ing_list:
	            new_list.append(ing_map[ID])
	            ing_cnts[ing_map[ID]] += 1
	            #strings = strings + ' ' + ing_map[ID]
	            
	        #print(new_list)
	        ing_lists.append(new_list)
	        string_lists.append(str(new_list))
	    except:
	        ing_lists.append([''])
	        string_lists.append('')
	        print('Failed on',i, 'no ingredient list.')
	print('Finished matching ingredients to keys.')



	ewg_prd_df['New_List'] = ing_lists

	#Build Synonym Dictionary
	synonym_dict = {}
	for i in range(ewg_ing_df.shape[0]):
	    row = ewg_ing_df.iloc[i]
	    syns = row['synonym_list']
	    if type(syns) == list:
	        for syn in syns:
	            synonym_dict[syn.strip()] = row['ingredient_name']
	        synonym_dict[row['ingredient_name']] = row['ingredient_name']
	    else:
	        synonym_dict[row['ingredient_name']] = row['ingredient_name']
	print('Build Synonyms')   

	#Initialize Ingredient Score
	ewg_ing_df['Acne_Score'] = 0.0

	#Extract Comodegenic Scores

	comodegenic = []

	with open('comodegenic.csv','r') as f:
	    for line in f:
	        if line[0] != ',':
	            words = line.strip().split(',')
	            if words[1] != '':
	                comodegenic.append(( words[0], words[1], words[2]))
	cd_df = pd.DataFrame(comodegenic)


	#Match Comodegeic Ingredients to EWG
	from fuzzywuzzy import fuzz
	from fuzzywuzzy import process
	matches = []
	print('Matching Comodegenic to EWG...')
	for i in range(cd_df.shape[0]):
	    cur_ingredient = cd_df.iloc[i][0].upper()
	    matches.append(process.extract(cur_ingredient, synonym_dict.keys(),limit=1, scorer=fuzz.token_sort_ratio))


	#Match Comodegenic Ingredients to EWG
	cd_ranks = []
	stop
	for i in range(cd_df.shape[0]):
	    match_score = int(matches[i][0][1])
	    match_name = matches[i][0][0]
	    cd_name = cd_df.iloc[i][0].upper()
	    cd_ranks.append(match_score)
	    
	    if match_score >= 90:
	        ewg_name = synonym_dict[match_name]
	        #print(temp_score, '\t', match_name, '\t', cd_name, '\t', synonym_dict[match_name])
	        #print(cd_df.iloc[i][1],cd_df.iloc[i][0])
	        row= ewg_ing_df[ewg_ing_df['ingredient_name']==ewg_name].index
	        ewg_ing_df.loc[row,'Acne_Score'] = cd_df.iloc[i][1]
	        #print(ewg_ing_df.loc[row]['ingredient_name'], ewg_ing_df.loc[row]['Acne_Score'])
	        #print(ewg_ing_df[ewg_ing_df['ingredient_name']==ewg_name])
	print('Updated EWG with Acne Scores')
	#Update Product Acne Score
	acne_score_list = []
	for i in range(ewg_prd_df.shape[0]):
	    row = ewg_prd_df.iloc[i]
	    total_acne = 0
	    for ing in row['New_List']:
	        try:
	            acne_score = float(ewg_ing_df[ewg_ing_df['ingredient_name']==ing]['Acne_Score'])
	            #print(ing, acne_score)
	            total_acne += acne_score
	        except:
	            None
	    acne_score_list.append(total_acne)
	#print(acne_score_list)
	ewg_prd_df['Acne_Score'] = acne_score_list


	#Save Final Acne Matrix
	pickle_out = open("ewg_prd_df.pickle","wb")
	pickle.dump(ewg_prd_df, pickle_out)
	pickle_out.close()
	print('Saved dataset to "ewg_prd_df.pickle"')


try:
	
	pickle.load(open("ewg_prd_df.pickle","rb"))
	print('Loaded from Pickle')
	ewg_prd_df = pickle.load(open("ewg_prd_df.pickle","rb"))
except:
	print("Building Dataset from Files...")
	buildDataSet()
	ewg_prd_df = pickle.load(open("ewg_prd_df.pickle","rb"))

#try:	
#	X = pickle.load(open("X.pickle","rb"))
#except:
#Need to change to a real function...code block simple
print('Building Dataset...')
#print(ewg_prd_df)
from collections import Counter
n = ewg_prd_df.shape[0]

print(n)
ing_lists = []
ing_cnts = Counter()
string_lists = []

for i in range(n):
	ings = ewg_prd_df.iloc[i]['New_List']
	str_list = ''
	if type(ings) == list:
		#print(type(ings), i)
		for ing in ings:
			if type(ing) == str:
				str_list = str_list + '|' + ing
		string_lists.append(str_list)
	else:
		print('Failed',i)
		string_lists.append('')



#Build TD-IDF Matrix
from sklearn.feature_extraction.text import TfidfVectorizer
def ing_tokenizer(word):
    return word.split('|')

#print(ewg_prd_df['New_List'].tolist())
vectorizer = TfidfVectorizer(tokenizer = ing_tokenizer, lowercase = False, stop_words = ['WATER','GLYCERIN','',
	'TITANIUM DIOXIDE', 'IRON OXIDES','BEESWAX','METHYLPARABEN', 'PROPYLPARABEN', 'PROPYLENE GLYCOL', 'PANTHENOL', 'MICA'] )
X = vectorizer.fit_transform(string_lists)
#print(vectorizer.vocabulary_)
pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()	


#print(X)

print('Running Optimization...')

from sklearn.metrics import confusion_matrix
for thresh in [0]:
	

	for test_size in [.001,.05,.01,.1]:

		for alph in [.001]:
			best_alpha = 0
			best_test_size = 0
			best_thresh_hold = 0
			best_test_score = 0
			best_train_score = 0
			best_model = None

		

			#Initialize Acne Score by Product
			Y = []
			for i in ewg_prd_df['Acne_Score']:
			    if i > 0 and i < 3:
			        Y.append(1)
			    elif i > 2:
			    	Y.append(2)
			    else:
			        Y.append(0)

			#Split Training and Test Data by 1/3 to 2/3
			from sklearn.model_selection import train_test_split
			X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)


			#Build NB Model
			from sklearn.naive_bayes import  MultinomialNB
			gnb =  MultinomialNB(alpha = alph)
			gnb_fit = gnb.fit(X_train,y_train)

			y_pred = gnb_fit.predict(X_test)
			#y_pred_tr = gnb_fit.predict(X_train)
			test_score = confusion_matrix(y_test, y_pred)
			#train_score = confusion_matrix(y_train, y_pred_tr)

			#if test_score:
			best_test_score = test_score
			best_alpha = alph
			best_test_size = test_size
			best_thresh_hold = thresh
			best_model = gnb_fit


			print('Best Test Score:',gnb_fit.score(X_test,y_test), '\n', test_score) #,'\t', train_score)
			print('Alpha:\t', best_alpha)
			print('Test_size:\t',test_size)
			print('Thresh:\t', thresh,'\n')


			#print('Thresh:',thresh, 'TestSize\t',test_size,'\n' ,'\tTraining Error:', )
			#print('\tTesting Error', )

pickle_out = open("nb.pickle","wb")
pickle.dump(gnb_fit, pickle_out)
pickle_out.close()	

ingredient_weights = {}
i = 0
print(len(gnb.coef_), best_model.coef_, type(best_model.coef_[0]))
for i in range(gnb_fit.coef_[0].shape[0]):
	#print( gnb.coef_[0][i], vectorizer.get_feature_names()[i])
    ingredient_weights[vectorizer.get_feature_names()[i]] =(gnb.coef_[0][i])
    #print(, gnb.coef_[i])



import operator
sorted_weights = sorted(ingredient_weights.items(), key=operator.itemgetter(1))
for i in range(1,20):
    print(sorted_weights[-i])

score = best_model.predict_proba(X_train)
pred = best_model.predict(X_train)

for i in range(100):
	print(ewg_prd_df.iloc[i]['Acne_Score'], score[i], pred[i])

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#%matplotlib inline
ewg_prd_df['Acne_Score'].hist(bins=40)
plt.show()
#for i in range(gnb_fit.coef_
#print(gnb_fit.coef_)
#out = gnb_fit.predict_proba(X_test)
#for i in range(len(out)):
#	print(out[i])
#print(gnb_fit.class_log_prior_)
#print(gnb_fit.feature_count_)
#print(gnb_fit.class_count_)
#print(gnb_fit.get_params())