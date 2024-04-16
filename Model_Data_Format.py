import re
import pandas as pd
import numpy as np
import matplotlib as mpl

class Model_Data:
    model_data = { #Raw Data in String Format
"Decision Tree" : """Chi-Square Feature Selection with 1 Features: AUC = 0.4913 (+/-) 0.0
Chi-Square Feature Selection with 2 Features: AUC = 0.4587 (+/-) 0.0306
Chi-Square Feature Selection with 3 Features: AUC = 0.5433 (+/-) 0.0396
Chi-Square Feature Selection with 4 Features: AUC = 0.4862 (+/-) 0.0402
Chi-Square Feature Selection with 5 Features: AUC = 0.5024 (+/-) 0.0418
Chi-Square Feature Selection with 6 Features: AUC = 0.6343 (+/-) 0.0359
Chi-Square Feature Selection with 7 Features: AUC = 0.6444 (+/-) 0.057
Chi-Square Feature Selection with 8 Features: AUC = 0.6444 (+/-) 0.0559
Chi-Square Feature Selection with 9 Features: AUC = 0.6515 (+/-) 0.0537
############################################################################################################################################################################################################################
Mutual Information Feature Selection with 1 Features: AUC = 0.6098 (+/-) 0.0
Mutual Information Feature Selection with 2 Features: AUC = 0.6888 (+/-) 0.0255
Mutual Information Feature Selection with 3 Features: AUC = 0.6708 (+/-) 0.0413
Mutual Information Feature Selection with 4 Features: AUC = 0.663 (+/-) 0.0467
Mutual Information Feature Selection with 5 Features: AUC = 0.6917 (+/-) 0.0323
Mutual Information Feature Selection with 6 Features: AUC = 0.6673 (+/-) 0.0317
############################################################################################################################################################################################################################
ANOVA F-Value Feature Selection with 1 Features: AUC = 0.5791 (+/-) 0.0
ANOVA F-Value Feature Selection with 2 Features: AUC = 0.5676 (+/-) 0.0261
ANOVA F-Value Feature Selection with 3 Features: AUC = 0.5591 (+/-) 0.0297
ANOVA F-Value Feature Selection with 4 Features: AUC = 0.51 (+/-) 0.0299
ANOVA F-Value Feature Selection with 5 Features: AUC = 0.6385 (+/-) 0.0355
ANOVA F-Value Feature Selection with 6 Features: AUC = 0.6368 (+/-) 0.0352
ANOVA F-Value Feature Selection with 7 Features: AUC = 0.6453 (+/-) 0.0576
ANOVA F-Value Feature Selection with 8 Features: AUC = 0.6437 (+/-) 0.0545
ANOVA F-Value Feature Selection with 9 Features: AUC = 0.6526 (+/-) 0.0565
############################################################################################################################################################################################################################""", 
"SVM": """Chi-Square Feature Selection with 1 Features: AUC = 0.5251 (+/-) 0.0726440981782589
Chi-Square Feature Selection with 2 Features: AUC = 0.604 (+/-) 0.04608269315128046
Chi-Square Feature Selection with 3 Features: AUC = 0.7405 (+/-) 0.03934390043984398
Chi-Square Feature Selection with 4 Features: AUC = 0.7692 (+/-) 0.04351014725153104
Chi-Square Feature Selection with 5 Features: AUC = 0.7594 (+/-) 0.04871035787537267
Chi-Square Feature Selection with 6 Features: AUC = 0.8028 (+/-) 0.037442448378054376
Chi-Square Feature Selection with 7 Features: AUC = 0.7972 (+/-) 0.0389180788156184
Chi-Square Feature Selection with 8 Features: AUC = 0.7897 (+/-) 0.04024857606328336
Chi-Square Feature Selection with 9 Features: AUC = 0.7833 (+/-) 0.04279785019009795
############################################################################################################################################################################################################################
Mutual Information Feature Selection with 1 Features: AUC = 0.4625 (+/-) 0.11412459126411578
Mutual Information Feature Selection with 2 Features: AUC = 0.7492 (+/-) 0.03258268022564585
Mutual Information Feature Selection with 3 Features: AUC = 0.7744 (+/-) 0.039188631448347756
Mutual Information Feature Selection with 4 Features: AUC = 0.7699 (+/-) 0.04044657693312865
Mutual Information Feature Selection with 5 Features: AUC = 0.79 (+/-) 0.03994859705197367
Mutual Information Feature Selection with 6 Features: AUC = 0.7885 (+/-) 0.0422742467265027
############################################################################################################################################################################################################################
ANOVA F-Value Feature Selection with 1 Features: AUC = 0.7644 (+/-) 0.025906642450511132
ANOVA F-Value Feature Selection with 2 Features: AUC = 0.7582 (+/-) 0.029716513750322412
ANOVA F-Value Feature Selection with 3 Features: AUC = 0.7509 (+/-) 0.03782960413139725
ANOVA F-Value Feature Selection with 4 Features: AUC = 0.7767 (+/-) 0.04448146174211578
ANOVA F-Value Feature Selection with 5 Features: AUC = 0.815 (+/-) 0.0330007659374516
ANOVA F-Value Feature Selection with 6 Features: AUC = 0.8029 (+/-) 0.03738457045008192
ANOVA F-Value Feature Selection with 7 Features: AUC = 0.7973 (+/-) 0.039063151267415185
ANOVA F-Value Feature Selection with 8 Features: AUC = 0.7896 (+/-) 0.04011110184828466
ANOVA F-Value Feature Selection with 9 Features: AUC = 0.7833 (+/-) 0.04300803309243881
############################################################################################################################################################################################################################""",
"Random Forests": """Chi-Square Feature Selection with 1 Features: AUC = 0.469 (+/-) 0.021717
Chi-Square Feature Selection with 2 Features: AUC = 0.4904 (+/-) 0.023432
Chi-Square Feature Selection with 3 Features: AUC = 0.637 (+/-) 0.023636
Chi-Square Feature Selection with 4 Features: AUC = 0.6682 (+/-) 0.024649
Chi-Square Feature Selection with 5 Features: AUC = 0.6659 (+/-) 0.023221
Chi-Square Feature Selection with 6 Features: AUC = 0.7291 (+/-) 0.027987
Chi-Square Feature Selection with 7 Features: AUC = 0.7713 (+/-) 0.026163
Chi-Square Feature Selection with 8 Features: AUC = 0.7492 (+/-) 0.028489
Chi-Square Feature Selection with 9 Features: AUC = 0.7555 (+/-) 0.025703
############################################################################################################################################################################################################################
Mutual Information Feature Selection with 1 Features: AUC = 0.6195 (+/-) 0.014304
Mutual Information Feature Selection with 2 Features: AUC = 0.7645 (+/-) 0.017464
Mutual Information Feature Selection with 3 Features: AUC = 0.8415 (+/-) 0.017768
Mutual Information Feature Selection with 4 Features: AUC = 0.7873 (+/-) 0.020817
Mutual Information Feature Selection with 5 Features: AUC = 0.8011 (+/-) 0.022807
Mutual Information Feature Selection with 6 Features: AUC = 0.7638 (+/-) 0.023125
############################################################################################################################################################################################################################
ANOVA F-Value Feature Selection with 1 Features: AUC = 0.604 (+/-) 0.01961
ANOVA F-Value Feature Selection with 2 Features: AUC = 0.6243 (+/-) 0.020625
ANOVA F-Value Feature Selection with 3 Features: AUC = 0.6185 (+/-) 0.018784
ANOVA F-Value Feature Selection with 4 Features: AUC = 0.6601 (+/-) 0.021039
ANOVA F-Value Feature Selection with 5 Features: AUC = 0.7284 (+/-) 0.02572
ANOVA F-Value Feature Selection with 6 Features: AUC = 0.7284 (+/-) 0.02663
ANOVA F-Value Feature Selection with 7 Features: AUC = 0.7722 (+/-) 0.025699
ANOVA F-Value Feature Selection with 8 Features: AUC = 0.7497 (+/-) 0.027845
ANOVA F-Value Feature Selection with 9 Features: AUC = 0.756 (+/-) 0.02673
############################################################################################################################################################################################################################""",
"Logistic Regression" : """Chi-Square Feature Selection with 1 Features: AUC = 0.5243 (+/-) 0.074142
Chi-Square Feature Selection with 2 Features: AUC = 0.6057 (+/-) 0.04381
Chi-Square Feature Selection with 3 Features: AUC = 0.7498 (+/-) 0.03539
Chi-Square Feature Selection with 4 Features: AUC = 0.7698 (+/-) 0.041305
Chi-Square Feature Selection with 5 Features: AUC = 0.7678 (+/-) 0.041966
Chi-Square Feature Selection with 6 Features: AUC = 0.8098 (+/-) 0.037219
Chi-Square Feature Selection with 7 Features: AUC = 0.804 (+/-) 0.038901
Chi-Square Feature Selection with 8 Features: AUC = 0.7945 (+/-) 0.03994
Chi-Square Feature Selection with 9 Features: AUC = 0.7884 (+/-) 0.042269
############################################################################################################################################################################################################################
Mutual Information Feature Selection with 1 Features: AUC = 0.4675 (+/-) 0.113352
Mutual Information Feature Selection with 2 Features: AUC = 0.7636 (+/-) 0.03169
Mutual Information Feature Selection with 3 Features: AUC = 0.7818 (+/-) 0.038207
Mutual Information Feature Selection with 4 Features: AUC = 0.7776 (+/-) 0.03982
Mutual Information Feature Selection with 5 Features: AUC = 0.7987 (+/-) 0.038493
Mutual Information Feature Selection with 6 Features: AUC = 0.7953 (+/-) 0.039578
############################################################################################################################################################################################################################
ANOVA F-Value Feature Selection with 1 Features: AUC = 0.7644 (+/-) 0.025907
ANOVA F-Value Feature Selection with 2 Features: AUC = 0.7629 (+/-) 0.03072
ANOVA F-Value Feature Selection with 3 Features: AUC = 0.7609 (+/-) 0.031268
ANOVA F-Value Feature Selection with 4 Features: AUC = 0.7817 (+/-) 0.03828
ANOVA F-Value Feature Selection with 5 Features: AUC = 0.8242 (+/-) 0.033808
ANOVA F-Value Feature Selection with 6 Features: AUC = 0.8098 (+/-) 0.037219
ANOVA F-Value Feature Selection with 7 Features: AUC = 0.804 (+/-) 0.038898
ANOVA F-Value Feature Selection with 8 Features: AUC = 0.7945 (+/-) 0.039974
ANOVA F-Value Feature Selection with 9 Features: AUC = 0.7884 (+/-) 0.042299
############################################################################################################################################################################################################################""",
"Naive Bayes" : """Chi-Square Feature Selection with 1 Features: AUC = 0.4788 (+/-) 0.060066
Chi-Square Feature Selection with 2 Features: AUC = 0.6338 (+/-) 0.043401
Chi-Square Feature Selection with 3 Features: AUC = 0.7356 (+/-) 0.033373
Chi-Square Feature Selection with 4 Features: AUC = 0.7479 (+/-) 0.034343
Chi-Square Feature Selection with 5 Features: AUC = 0.7461 (+/-) 0.034438
Chi-Square Feature Selection with 6 Features: AUC = 0.7572 (+/-) 0.035569
Chi-Square Feature Selection with 7 Features: AUC = 0.7855 (+/-) 0.034539
Chi-Square Feature Selection with 8 Features: AUC = 0.7675 (+/-) 0.039871
Chi-Square Feature Selection with 9 Features: AUC = 0.7626 (+/-) 0.040905
############################################################################################################################################################################################################################
Mutual Information Feature Selection with 1 Features: AUC = 0.6764 (+/-) 0.050945
Mutual Information Feature Selection with 2 Features: AUC = 0.783 (+/-) 0.03236
Mutual Information Feature Selection with 3 Features: AUC = 0.8003 (+/-) 0.037185
Mutual Information Feature Selection with 4 Features: AUC = 0.8045 (+/-) 0.034385
Mutual Information Feature Selection with 5 Features: AUC = 0.7963 (+/-) 0.037287
Mutual Information Feature Selection with 6 Features: AUC = 0.791 (+/-) 0.036335
############################################################################################################################################################################################################################
ANOVA F-Value Feature Selection with 1 Features: AUC = 0.7304 (+/-) 0.027358
ANOVA F-Value Feature Selection with 2 Features: AUC = 0.7689 (+/-) 0.02794
ANOVA F-Value Feature Selection with 3 Features: AUC = 0.7682 (+/-) 0.027643
ANOVA F-Value Feature Selection with 4 Features: AUC = 0.7697 (+/-) 0.033275
ANOVA F-Value Feature Selection with 5 Features: AUC = 0.7816 (+/-) 0.033872
ANOVA F-Value Feature Selection with 6 Features: AUC = 0.7572 (+/-) 0.035569
ANOVA F-Value Feature Selection with 7 Features: AUC = 0.7855 (+/-) 0.034539
ANOVA F-Value Feature Selection with 8 Features: AUC = 0.7675 (+/-) 0.039871
ANOVA F-Value Feature Selection with 9 Features: AUC = 0.7626 (+/-) 0.040905
############################################################################################################################################################################################################################""",
"Multi Layer Perceptron" : """Chi-Square Feature Selection with 1 Features: AUC = 0.4471 (+/-) 0.08180664783675999
Chi-Square Feature Selection with 2 Features: AUC = 0.522 (+/-) 0.06503869054852361
Chi-Square Feature Selection with 3 Features: AUC = 0.6654 (+/-) 0.05492868697088987
Chi-Square Feature Selection with 4 Features: AUC = 0.6841 (+/-) 0.05302143463990964
Chi-Square Feature Selection with 5 Features: AUC = 0.6827 (+/-) 0.05772075814507834
Chi-Square Feature Selection with 6 Features: AUC = 0.7587 (+/-) 0.04819019017769524
Chi-Square Feature Selection with 7 Features: AUC = 0.7968 (+/-) 0.0593487958999709
Chi-Square Feature Selection with 8 Features: AUC = 0.8036 (+/-) 0.05291239581066659
Chi-Square Feature Selection with 9 Features: AUC = 0.7854 (+/-) 0.052783403095352656
############################################################################################################################################################################################################################
Mutual Information Feature Selection with 1 Features: AUC = 0.7042 (+/-) 0.043329483001546085
Mutual Information Feature Selection with 2 Features: AUC = 0.825 (+/-) 0.029667211024336653
Mutual Information Feature Selection with 3 Features: AUC = 0.8156 (+/-) 0.05430753190556917
Mutual Information Feature Selection with 4 Features: AUC = 0.7803 (+/-) 0.04923069734265625
Mutual Information Feature Selection with 5 Features: AUC = 0.7954 (+/-) 0.04466522414582904
Mutual Information Feature Selection with 6 Features: AUC = 0.791 (+/-) 0.046320183181764805
############################################################################################################################################################################################################################
ANOVA F-Value Feature Selection with 1 Features: AUC = 0.7596 (+/-) 0.029236105728931115
ANOVA F-Value Feature Selection with 2 Features: AUC = 0.7585 (+/-) 0.03248622880835128
ANOVA F-Value Feature Selection with 3 Features: AUC = 0.7465 (+/-) 0.04394619997581685
ANOVA F-Value Feature Selection with 4 Features: AUC = 0.7339 (+/-) 0.06152061182783121
ANOVA F-Value Feature Selection with 5 Features: AUC = 0.7884 (+/-) 0.05188700669508809
ANOVA F-Value Feature Selection with 6 Features: AUC = 0.7591 (+/-) 0.049240029770614216
ANOVA F-Value Feature Selection with 7 Features: AUC = 0.7959 (+/-) 0.05286673114414231
ANOVA F-Value Feature Selection with 8 Features: AUC = 0.8045 (+/-) 0.04960017005095709
ANOVA F-Value Feature Selection with 9 Features: AUC = 0.7861 (+/-) 0.05359462262577239
############################################################################################################################################################################################################################""",
"kNN" : """Chi-Square Feature Selection with 1 Features: AUC = 0.4525 (+/-) 0.079206
Chi-Square Feature Selection with 2 Features: AUC = 0.4537 (+/-) 0.082151
Chi-Square Feature Selection with 3 Features: AUC = 0.4765 (+/-) 0.084809
Chi-Square Feature Selection with 4 Features: AUC = 0.4747 (+/-) 0.084661
Chi-Square Feature Selection with 5 Features: AUC = 0.4764 (+/-) 0.084945
Chi-Square Feature Selection with 6 Features: AUC = 0.4813 (+/-) 0.084748
Chi-Square Feature Selection with 7 Features: AUC = 0.4966 (+/-) 0.079066
Chi-Square Feature Selection with 8 Features: AUC = 0.481 (+/-) 0.08111
Chi-Square Feature Selection with 9 Features: AUC = 0.4837 (+/-) 0.082461
############################################################################################################################################################################################################################
Mutual Information Feature Selection with 1 Features: AUC = 0.6769 (+/-) 0.055156
Mutual Information Feature Selection with 2 Features: AUC = 0.7564 (+/-) 0.044432
Mutual Information Feature Selection with 3 Features: AUC = 0.8278 (+/-) 0.043755
Mutual Information Feature Selection with 4 Features: AUC = 0.8282 (+/-) 0.043015
Mutual Information Feature Selection with 5 Features: AUC = 0.8053 (+/-) 0.043401
Mutual Information Feature Selection with 6 Features: AUC = 0.795 (+/-) 0.042161
############################################################################################################################################################################################################################
ANOVA F-Value Feature Selection with 1 Features: AUC = 0.6742 (+/-) 0.05706
ANOVA F-Value Feature Selection with 2 Features: AUC = 0.682 (+/-) 0.053075
ANOVA F-Value Feature Selection with 3 Features: AUC = 0.714 (+/-) 0.050442
ANOVA F-Value Feature Selection with 4 Features: AUC = 0.759 (+/-) 0.042551
ANOVA F-Value Feature Selection with 5 Features: AUC = 0.7742 (+/-) 0.040087
ANOVA F-Value Feature Selection with 6 Features: AUC = 0.4813 (+/-) 0.084748
ANOVA F-Value Feature Selection with 7 Features: AUC = 0.4966 (+/-) 0.079066
ANOVA F-Value Feature Selection with 8 Features: AUC = 0.481 (+/-) 0.08111
ANOVA F-Value Feature Selection with 9 Features: AUC = 0.4837 (+/-) 0.082461
############################################################################################################################################################################################################################""",
"AdaBoost" : """Chi-Square Feature Selection with 1 Features: AUC = 0.448 (+/-) 0.0
Chi-Square Feature Selection with 2 Features: AUC = 0.5734 (+/-) 0.0
Chi-Square Feature Selection with 3 Features: AUC = 0.6747 (+/-) 0.0
Chi-Square Feature Selection with 4 Features: AUC = 0.7347 (+/-) 0.006818
Chi-Square Feature Selection with 5 Features: AUC = 0.7363 (+/-) 0.008332
Chi-Square Feature Selection with 6 Features: AUC = 0.7393 (+/-) 0.023473
Chi-Square Feature Selection with 7 Features: AUC = 0.7909 (+/-) 0.00303
Chi-Square Feature Selection with 8 Features: AUC = 0.7614 (+/-) 0.003785
Chi-Square Feature Selection with 9 Features: AUC = 0.7633 (+/-) 0.007575
############################################################################################################################################################################################################################
Mutual Information Feature Selection with 1 Features: AUC = 0.6877 (+/-) 0.0
Mutual Information Feature Selection with 2 Features: AUC = 0.7753 (+/-) 0.0
Mutual Information Feature Selection with 3 Features: AUC = 0.8296 (+/-) 0.0
Mutual Information Feature Selection with 4 Features: AUC = 0.8198 (+/-) 0.01439
Mutual Information Feature Selection with 5 Features: AUC = 0.789 (+/-) 0.013628
Mutual Information Feature Selection with 6 Features: AUC = 0.8071 (+/-) 0.018161
############################################################################################################################################################################################################################
ANOVA F-Value Feature Selection with 1 Features: AUC = 0.6998 (+/-) 0.0
ANOVA F-Value Feature Selection with 2 Features: AUC = 0.673 (+/-) 0.0
ANOVA F-Value Feature Selection with 3 Features: AUC = 0.6876 (+/-) 0.0
ANOVA F-Value Feature Selection with 4 Features: AUC = 0.7486 (+/-) 0.009841
ANOVA F-Value Feature Selection with 5 Features: AUC = 0.7358 (+/-) 0.008332
ANOVA F-Value Feature Selection with 6 Features: AUC = 0.7399 (+/-) 0.02348
ANOVA F-Value Feature Selection with 7 Features: AUC = 0.7909 (+/-) 0.003029
ANOVA F-Value Feature Selection with 8 Features: AUC = 0.7613 (+/-) 0.003787
ANOVA F-Value Feature Selection with 9 Features: AUC = 0.7636 (+/-) 0.007567
############################################################################################################################################################################################################################"""
}
    
    #Transforming Data for Visualization
    model_lists = {}

    for models in list(model_data.keys()):
        model_lists[models] = model_data[models].split("############################################################################################################################################################################################################################") 
    #Chi Square Feature Selection Results:
    ChiSquare = {}
    for models in list(model_lists.keys()):
        ChiSquare[models] = re.sub(r'^.*= ', '', model_lists[models][0]) #^.*=  
        ChiSquare[models] = re.sub(r'\n.*= ', ';', ChiSquare[models]) 
        ChiSquare[models] = re.sub(r'\n', '', ChiSquare[models]) 
        ChiSquare[models] = ChiSquare[models].split(';')
    ChiSquare = pd.DataFrame(ChiSquare)

    #Mutual Information Feature Selection Results:
    MutualInfo = {}
    for models in list(model_lists.keys()):
        MutualInfo[models] = re.sub(r'^.*= ', '', model_lists[models][1]) #^.*=  
        MutualInfo[models] = re.sub(r'\n.*= ', ';', MutualInfo[models]) 
        MutualInfo[models] = re.sub(r'\n', '', MutualInfo[models]) 
        MutualInfo[models] = MutualInfo[models].split(';')[1:]
    MutualInfo = pd.DataFrame(MutualInfo)
    
    #ANOVA F-Value Feature Selection Results:
    ANOVA = {}
    for models in list(model_lists.keys()):
        ANOVA[models] = re.sub(r'^.*= ', '', model_lists[models][2]) #^.*=  
        ANOVA[models] = re.sub(r'\n.*= ', ';', ANOVA[models]) 
        ANOVA[models] = re.sub(r'\n', '', ANOVA[models]) 
        ANOVA[models] = ANOVA[models].split(';')[1:]
    ANOVA = pd.DataFrame(ANOVA)
# print(model_data)
