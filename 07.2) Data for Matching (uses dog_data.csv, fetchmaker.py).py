import numpy as np
import fetchmaker
from scipy.stats import binom_test
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

#1  Get the tail lengths of all of the "rottweiler"s in the system.
rottweiler_tl = fetchmaker.get_tail_length("rottweiler")

#2 Print out the mean of rottweiler_tl and the standard deviation of rottweiler_tl.
mean_rottweiler_tl = np.mean(rottweiler_tl)
print(mean_rottweiler_tl)
std_rottweiler_tl = np.std(rottweiler_tl)
print(std_rottweiler_tl)

#3 Over the years, we have seen that we expect 8% of dogs in the FetchMaker system to be rescues. We want to know if whippets are significantly more or less likely to be a rescue.
whippet_rescue = fetchmaker.get_is_rescue("whippet")

#4 Rescued whippet
num_whippet_rescues = np.count_nonzero(whippet_rescue)
print(num_whippet_rescues)

#5 Get the number of samples in the whippet set.
num_whippets = np.size(whippet_rescue)
print(num_whippets)

#6 Binomial test
pval_binomial_test = binom_test(num_whippet_rescues, num_whippets, 0.08)

#7 Print out the p-value. Is your result significant?
print(pval_binomial_test)

#8 Three of our most popular mid-sized dog breeds are whippets, terriers, and pitbulls. Is there a significant difference in the average weights of these three dog breeds? Perform a comparative numerical test to determine if there is a significant difference.
w = fetchmaker.get_weight("whippet")
t = fetchmaker.get_weight("terrier")
p = fetchmaker.get_weight("pitbull")

f_oneway(w,t,p)
print(f_oneway(w,t,p).pvalue)

#9 Now, perform another test to determine which of the pairs of these dog breeds differ from each other.
values = np.concatenate([w, t, p])       # one long list from lists w, t, p
labels = ['whippet']*len(w)   +   ['terrier']*len(t)   +   ['pitbull']*len(p)  # 
#print(labels)
print(pairwise_tukeyhsd(values, labels, 0.05))

#10 Test if "poodle"s and "shihtzu"s have significantly different color breakdowns.
poodle_colors = fetchmaker.get_color("poodle")
shihtzu_colors = fetchmaker.get_color("shihtzu")

#11 color_table

color_table = [
    [np.count_nonzero(poodle_colors == "black"), np.count_nonzero(shihtzu_colors == "black")],
    [np.count_nonzero(poodle_colors == "brown"), np.count_nonzero(shihtzu_colors == "brown")],
    [np.count_nonzero(poodle_colors == "gold"), np.count_nonzero(shihtzu_colors == "gold")],
    [np.count_nonzero(poodle_colors == "grey"), np.count_nonzero(shihtzu_colors == "grey")],
    [np.count_nonzero(poodle_colors == "white"), np.count_nonzero(shihtzu_colors == "white")]
    ]

#12 chi-sq test
_,color_pval,_,_ = chi2_contingency(color_table)
print(color_pval)
#######################################################################################################











