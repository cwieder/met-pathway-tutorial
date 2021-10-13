import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats

def t_tests(matrix, classes, multiple_correction_method, return_stat=False):
    """
    Function for two tailed t-tests
    :param Matrix: Metabolite abundance matrix with columns representing metabolites and rows representing samples
    :param classes: Column of the abundance matrix containing the class names 
    :param multiple_correction_method: multiple testing correction method - use any available on scipy
    :return: DataFrame of t-test results
    """
    metabolites = matrix.columns.tolist()
    matrix['Target'] = pd.factorize(classes)[0]
    
    disease = matrix[matrix["Target"] == 0].copy()
    disease.drop(['Target'], axis=1, inplace=True)
    ctrl = matrix[matrix["Target"] != 0].copy()
    ctrl.drop(['Target'], axis=1, inplace=True)

    pvalues = stats.ttest_ind(disease, ctrl)[1]
    tstat = stats.ttest_ind(disease, ctrl)[0]
    padj = sm.stats.multipletests(pvalues, 0.05, method=multiple_correction_method)
    
    if return_stat:
        results = pd.DataFrame(zip(metabolites, pvalues, padj[1], tstat),
                               columns=["Metabolite", "P-value", "P-adjust", "t-statistic"])
    else:
        results = pd.DataFrame(zip(metabolites, pvalues, padj[1]),
                               columns=["Metabolite", "P-value", "P-adjust"])
    return results

def over_representation_analysis(DA_list, background_list, pathways_df):
    """
    Function for over representation analysis using Fisher exact test (right tailed)
    :param DA_list: List of differentially abundant metabolite IDENTIFIERS
    :param background_list: background list of IDENTIFIERS
    :param pathways_df: pathway dataframe containing compound identifiers
    :return: DataFrame of ORA results for each pathway, p-value, q-value, hits ratio
    """

    KEGG_pathways = pathways_df.dropna(axis=0, how='all', subset=pathways_df.columns.tolist()[1:])
    pathways = KEGG_pathways.index.tolist()
    pathway_names = KEGG_pathways["Pathway_name"].tolist()
    pathway_dict = dict(zip(pathways, pathway_names))
    KEGG_pathways = KEGG_pathways.drop('Pathway_name', axis=1)

    pathways_with_compounds = []
    pathway_names_with_compounds = []
    pvalues = []
    pathway_ratio = []
    pathway_count = 0
    pathway_coverage = []

    for pathway in pathways:
        # perform ORA for each pathway
        pathway_compounds = list(set(KEGG_pathways.loc[pathway, :].tolist()))
        pathway_compounds = [i for i in pathway_compounds if str(i) != "nan"]
        if not pathway_compounds or len(pathway_compounds) < 2:
            # ignore pathway if contains no compounds or has less than 3 compounds
            continue
        else:

            DA_in_pathway = len(set(DA_list) & set(pathway_compounds))
            # k: compounds in DA list AND pathway
            DA_not_in_pathway = len(np.setdiff1d(DA_list, pathway_compounds))
            # K: compounds in DA list not in pathway
            compound_in_pathway_not_DA = len(set(pathway_compounds) & set(np.setdiff1d(background_list, DA_list)))
            # not DEM compounds present in pathway
            compound_not_in_pathway_not_DA = len(
                np.setdiff1d(np.setdiff1d(background_list, DA_list), pathway_compounds))
            # compounds in background list not present in pathway
            if DA_in_pathway == 0 or (compound_in_pathway_not_DA + DA_in_pathway) < 2:
                # ignore pathway if there are no DEM compounds in that pathway
                continue
            else:
                pathway_count += 1
                # Create 2 by 2 contingency table
                pathway_ratio.append(str(DA_in_pathway) + "/" + str(compound_in_pathway_not_DA + DA_in_pathway))
                pathway_coverage.append(
                    str(compound_in_pathway_not_DA + DA_in_pathway) + "/" + str(len(pathway_compounds)))
                pathways_with_compounds.append(pathway)
                pathway_names_with_compounds.append(pathway_dict[pathway])
                contingency_table = np.array([[DA_in_pathway, compound_in_pathway_not_DA],
                                              [DA_not_in_pathway, compound_not_in_pathway_not_DA]])
                # Run right tailed Fisher's exact test
                oddsratio, pvalue = stats.fisher_exact(contingency_table, alternative="greater")
                pvalues.append(pvalue)
    try:
        padj = sm.stats.multipletests(pvalues, 0.05, method="fdr_bh")
        results = pd.DataFrame(
            zip(pathways_with_compounds, pathway_names_with_compounds, pathway_ratio, pathway_coverage, pvalues,
                padj[1]),
            columns=["Pathway_ID", "Pathway_name", "Hits", "Coverage", "P-value", "P-adjust"])
    except ZeroDivisionError:
        padj = [1] * len(pvalues)
        results = pd.DataFrame(zip(pathways_with_compounds, pathway_names_with_compounds, pathway_ratio, pvalues, padj),
                               columns=["Pathway_ID", "Pathway_name", "Hits", "Coverage", "P-value", "P-adjust"])
    return results