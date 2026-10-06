# Reference audit (reference-checker skill, exhaustive pass)

Source: `paper/references.bib` — 30 entries; every entry was checked individually. Routes: Crossref DOI metadata, Crossref bibliographic query (restricted to the journal/publisher), PubMed esummary, DBLP, OpenAlex, arXiv API. Raw matched metadata: `review/reference_audit_v2.json`.

| Ref | Submitted title | Submitted authors | Source | Year | Identifier / route | Match quality | Status | Confidence | Main issue / suggested fix |
|---|---|---|---|---|---|---|---|---|---|
| `belloni2011l1` | $ell_1$-penalized quantile regression in high-dimensional sparse models | Belloni, A. and Chernozhukov, V | The Annals of Statistics | 2011 | 10.1214/10-AOS827 — Crossref DOI metadata | Exact | Verified | High | none |
| `bickel2008` | Regularized estimation of large covariance matrices | Bickel, P. J. and Levina, E | The Annals of Statistics | 2008 | 10.1214/009053607000000758 — Crossref DOI metadata | Exact | Verified | High | none |
| `bickel2009simultaneous` | Simultaneous analysis of Lasso and Dantzig selector | Bickel, P. J. and Ritov, Y. and Tsybakov, A. B | The Annals of Statistics | 2009 | 10.1214/08-AOS620 — Crossref DOI metadata | Exact | Verified | High | none |
| `bradic2017uniform` | Uniform inference for high-dimensional quantile regression: Linear functionals a | Bradic, J. and Kolar, M |  | 2017 | https://arxiv.org/abs/1702.06209 — arXiv API id_list | Exact | Verified | Medium | none |
| `buhlmann2011statistics` | Statistics for High-Dimensional Data: Methods, Theory and Applications | B\"uhlmann, P. and van de Geer, S | Springer | 2011 | 10.1007/978-3-642-20192-9 — Crossref DOI metadata | Exact | Verified | High | none |
| `carroll2006measurement` | Measurement Error in Nonlinear Models: A Modern Perspective | Carroll, R. J. and Ruppert, D. and Stefanski, L. A. and | Chapman \& Hall/CRC | 2006 | 10.1201/9781420010138 — Crossref DOI metadata | Exact | Verified | High | none |
| `datta2017cocolasso` | CoCoLasso for high-dimensional error-in-variables regression | Datta, A. and Zou, H | The Annals of Statistics | 2017 | 10.1214/16-AOS1527 — Crossref DOI metadata | Exact | Verified | High | none |
| `fernandes2021smoothing` | Smoothing quantile regressions | Fernandes, M. and Guerre, E. and Horta, E | Journal of Business \& Economic Statistics | 2021 | 10.1080/07350015.2019.1660177 — Crossref DOI metadata | Exact | Verified | High | year 2021 = the print/issue year; the identifier's `issued' date is 2019 (online first) |
| `fuller1987` | Measurement Error Models | Fuller, W. A | Wiley | 1987 | 10.1002/9780470316665 — Crossref DOI metadata | Exact | Verified | High | none |
| `gryparis2009measurement` | Measurement error caused by spatial misalignment in environmental epidemiology | Gryparis, A. and Paciorek, C. J. and Zeka, A. and Schwa | Biostatistics | 2009 | 10.1093/biostatistics/kxn033 — Crossref DOI metadata | Exact | Verified | High | year 2009 = the print/issue year; the identifier's `issued' date is 2008 (online first); PubMed 18927119: Biostatistics (Oxford, England) 2009;10(2):258-74 |
| `he2000quantile` | Quantile regression estimates for a class of linear and partially linear errors- | He, X. and Liang, H | Statistica Sinica | 2000 | — — publisher PDF (Statistica Sinica archive) | Exact | Verified | Medium | none |
| `he2023smoothed` | Smoothed quantile regression with large-scale inference | He, X. and Pan, X. and Tan, K. M. and Zhou, W.-X | Journal of Econometrics | 2023 | 10.1016/j.jeconom.2021.07.010 — Crossref DOI metadata | Exact | Verified | High | PubMed 36776480: Journal of econometrics 2023;232(2):367-388 |
| `javanmard2014hypothesis` | Hypothesis testing in high-dimensional regression under the Gaussian random desi | Javanmard, A. and Montanari, A | IEEE Transactions on Information Theory | 2014 | 10.1109/TIT.2014.2343629 — Crossref DOI metadata | Exact | Verified | High | none |
| `koenker1978` | Regression quantiles | Koenker, R. and Bassett, G | Econometrica | 1978 | 10.2307/1913643 — Crossref DOI metadata | Exact | Verified | High | page 33-50 refines 33 |
| `ledoit2004` | A well-conditioned estimator for large-dimensional covariance matrices | Ledoit, O. and Wolf, M | Journal of Multivariate Analysis | 2004 | 10.1016/S0047-259X(03)00096-4 — Crossref DOI metadata | Exact | Verified | High | none |
| `loh2012highdim` | High-dimensional regression with noisy and missing data: Provable guarantees wit | Loh, P.-L. and Wainwright, M. J | The Annals of Statistics | 2012 | 10.1214/12-AOS1018 — Crossref DOI metadata | Exact | Verified | High | none |
| `nakamura1990` | Corrected score function for errors-in-variables models: Methodology and applica | Nakamura, T | Biometrika | 1990 | 10.1093/biomet/77.1.127 — Crossref DOI metadata | Exact | Verified | High | none |
| `negahban2012unified` | A unified framework for high-dimensional analysis of $M$-estimators with decompo | Negahban, S. N. and Ravikumar, P. and Wainwright, M. J. | Statistical Science | 2012 | 10.1214/12-STS400 — Crossref DOI metadata | Exact | Verified | High | none |
| `noble2017` | Influences on the test-retest reliability of functional connectivity MRI and its | Noble, S. and Spann, M. N. and Tokoglu, F. and Shen, X. | Cerebral Cortex | 2017 | 10.1093/cercor/bhx230 — Crossref DOI metadata | Exact | Verified | High | PubMed 28968754: Cerebral cortex (New York, N.Y. : 1991) 2017;27(11):5415-5429 |
| `politis1999subsampling` | Subsampling | Politis, D. N. and Romano, J. P. and Wolf, M | Springer | 1999 | 10.1007/978-1-4612-1554-7 — Crossref DOI metadata | Exact | Verified | High | none |
| `pollard1991` | Asymptotics for least absolute deviation regression estimators | Pollard, D | Econometric Theory | 1991 | 10.1017/S0266466600004394 — Crossref DOI metadata | Exact | Verified | High | none |
| `romano2019conformalized` | Conformalized quantile regression | Romano, Y. and Patterson, E. and Cand\`es, E | Advances in Neural Information Processing  | 2019 | — — NeurIPS 2019 proceedings index (papers.nips.cc) | Exact | Verified | Medium | none |
| `rosenbaum2010` | Sparse recovery under matrix uncertainty | Rosenbaum, M. and Tsybakov, A. B | The Annals of Statistics | 2010 | 10.1214/10-AOS793 — Crossref DOI metadata | Exact | Verified | High | none |
| `stefanski1989` | Unbiased estimation of a nonlinear function of a normal mean with application to | Stefanski, L. A | Communications in Statistics - Theory and  | 1989 | 10.1080/03610928908830159 — Crossref DOI metadata | Exact | Verified | High | none |
| `stefanski1995simulation` | Simulation-extrapolation: The measurement error jackknife | Stefanski, L. A. and Cook, J. R | Journal of the American Statistical Associ | 1995 | 10.1080/01621459.1995.10476629 — Crossref DOI metadata | Exact | Verified | High | none |
| `vandegeer2014asymptotically` | On asymptotically optimal confidence regions and tests for high-dimensional mode | van de Geer, S. and B\"uhlmann, P. and Ritov, Y. and De | The Annals of Statistics | 2014 | 10.1214/14-AOS1221 — Crossref DOI metadata | Exact | Verified | High | none |
| `wang2012` | Quantile regression for analyzing heterogeneity in ultra-high dimension | Wang, L. and Wu, Y. and Li, R | Journal of the American Statistical Associ | 2012 | 10.1080/01621459.2012.656014 — Crossref DOI metadata | Exact | Verified | High | PubMed 23082036: Journal of the American Statistical Association 2012;107(497):214-222 |
| `wang2012correctedloss` | Corrected-loss estimation for quantile regression with covariate measurement err | Wang, H. J. and Stefanski, L. A. and Zhu, Z | Biometrika | 2012 | 10.1093/biomet/ass005 — Crossref DOI metadata | Exact | Verified | High | PubMed 23843665: Biometrika 2012;99(2):405-421 |
| `wei2009quantile` | Quantile regression with measurement error | Wei, Y. and Carroll, R. J | Journal of the American Statistical Associ | 2009 | 10.1198/jasa.2009.tm08420 — Crossref DOI metadata | Exact | Verified | High | PubMed 20305802: Journal of the American Statistical Association 2009;104(487):1129-1143 |
| `zhang2014confidence` | Confidence intervals for low dimensional parameters in high dimensional linear m | Zhang, C.-H. and Zhang, S. S | Journal of the Royal Statistical Society:  | 2014 | 10.1111/rssb.12026 — Crossref DOI metadata | Exact | Verified | High | year 2014 = the print/issue year; the identifier's `issued' date is 2013 (online first) |

## Whole-list checks

* duplicates: none
* citation keys used in the text but absent from the .bib: none
* entries never cited: none
* retraction/withdrawal signals: none
* DOIs located by this audit that the .bib does not yet carry: none
