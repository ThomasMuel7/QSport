> SMART Project Proposal

**Titre:** Q-Sport:PipelineHybrideQuantum-Classiquepour
l'AnalysePrédictiveetl'Interaction Conversationnelle

**InfoProjet:** C’estunprojetderecherchequenous(ThomasMuel,
CarlHabsieger etAlexisBusillet) voulonsmener.C’estunprojetindépendantqui
n’estpasliéà AGIRnià desentreprisesetnous
voulonsexplorerlemondecomplexedela
prédictiondecompétitionssportives(coupedumonde
football/tennis/NFL…)pasencoredéfinienutilisantlestechnologiesquantiquesémergentes.

**Partenaire/Client:** Indépendant

**Dimensions(dimensions):**TECHNIQUE / SCIENTIFIQUE

**1.** **Justification** **Technique**

Leprojet
reposesurunearchitecturemoderneenmicroservicespourgarantirlascalabilitéet
l'interopérabilitédescomposants:

> • Moteur ML :UtilisationdePythonaveclesbibliothèquesPyTorchetPennyLane
> etpeutêtre Qiskitpour l'intégrationdescouchesquantiques.
>
> • Backend& Données:UtilisationdeSupabasepour
> unegestionlégèreetperformantedes
> basesdedonnéessportives(statistiquesdematchs,classementsATP/FIFA)etdes
> prédictions.NousutiliseronssurementPOSTGRESQL etlepluginpgvectorpour
> faireduRAG.
>
> • InterfaceIA : Intégrationd'unLLMlocal(via Ollamaouunframework
> similaire) pour transformer
> lesprédictionscomplexeseninsightscompréhensiblespar l'utilisateur via
> un simplechat.
>
> • Conteneurisation:L'ensemblesera orchestrévia Docker pour faciliter
> ledéploiement.

**2.** **Justification** **Scientifique**

L'objectifestd'explorerl'AvantageQuantiquedansl'analysededonnéessportives
multidimensionnelles(météo,fatiguedesjoueurs,historiquesdeconfrontations…).

> • QuantumFeatureMaps:ProjectiondesdonnéesdansunespacedeHilbertdehaute
> dimensionpourfaciliterla
> séparationlinéairedesclasses(VictoirevsDéfaite).
>
> • CircuitsQuantiquesVariationnels(VQC)
> :Recherchedemotifscomplexesvial'intricationet la
> superposition,quelesmodèlesclassiques(XGBoost,Random Forest)
> pourraientnepas détecter.
>
> • Réductiondedimension:UtilisationdePCA oudeRandom ForestoudeXGBOOST
> puisd’un NN classiquepouroptimiser
> le"budgetdequbits"nécessaireaumodèlequantique.

**Résumé** **ycomprislesobjectifsetleslivrables:**

**Résumé** **:** LeprojetQ-Sportviseà révolutionner
laprédictiondesissuesd’unerencontresportiveen dépassantleslimitesdela
régressionclassique.Encombinantl'apprentissageautomatique
classiquedehauteperformance(XGBoost,RandomForest)
etunmodèleclassiquederéseaude
neuronepourl'ingénieriedescaractéristiquesavecdesréseauxdeneuronesquantiques(QNN)pour
la classification,cetteinitiativeanalysera
devastesensemblesdedonnéessportivespour prédire

l'issuedesrencontres.Lesystèmesera déployésousla
formed'uneapplicationwebcomplètedotée
d'uneinterfacesimplifiée,utilisantunconseillerLLMlocaletunpipelineRAG
pour traduireles donnéescomplexesenanalysesconversationnellesfluides.

**3.** **ObjectifsSpécifiques**

> 1\.
> IngénieriedesDonnéesSportives:Constructiond'unebasededonnées(TennisouFootball
> ouautre)intégrantlesvariablescritiques(surfacesdejeu,météo,blessures).
>
> 2\. SélectiondeFeaturesHybride:Optimisationdesentréespour
> leclassifieur quantiqueen utilisantdesméthodesclassiques.
>
> 3\. DéveloppementduModèleQML :Conceptionetentraînementdeclassifieurs
> hybride(réseau deneuroneclassiquepuisquantique) via PennyLanesur
> dumatériel IBMQuantum (ordinateur quantique) ousimulateurs.
>
> 4\.
> InterfaceConversationnelle:Miseenplaced'unfrontendminimalistepermettantà
> l'utilisateur deposerdesquestionssur unmatchà veniretd'obtenir
> uneanalysebaséesur lesrésultatsdumodèlequantique.

**4.** **LivrablesAttendus**

> • D1 : RépertoireGitHub:Codesourcecompletincluantlemoteur decalcul
> etl'interface.
>
> • D2 : EnvironnementDocker
> :Configurationdocker-composeorchestrantlebackend,le moteur IA et
> lefrontend.
>
> • D3 : PlateformeQ-Sport:Uneapplicationwebsimplecentréesur
> unefenêtredechatpour interagir avecl'IAd'analyse.

**References(références):**

\[1\] Horvat,T.andJob,J.(2020) The
useofmachinelearninginsportoutcomeprediction:A review.
WileyInterdisciplinaryReviews:Data
MiningandKnowledgeDiscovery.\[Accessed15 Dec.2025\].

\[2\] Bosch,P.(2018)Predictingthewinner
ofNFL-gamesusingMachineandDeepLearning.Thesis,
VrijeUniversiteitAmsterdam.\[Accessed15 Dec.2025\].

\[3\] Jolliffe,I. T.andCadima, J.(2016)Principal
componentanalysis:areviewandrecent developments.Philosophical
TransactionsoftheRoyal SocietyA.\[Accessed15 Dec.2025\].

\[4\] Hamadani,B.(2006) PredictingtheoutcomeofNFL
gamesusingmachinelearning.CS229Project
Report,StanfordUniversity.\[Accessed15 Dec.2025\].

\[5\] Schuld,M.andKilloran,N. (2019)
QuantumMachineLearninginFeatureHilbertSpaces.Physical
ReviewLetters.\[Accessed15 Dec.2025\].

\[6\] Farhi,E.andNeven,H.(2018) ClassificationwithQuantumNeural
NetworksonNearTerm Processors.arXivpreprint.\[Accessed15 Dec.2025\].

\[7\] Arthur,D.andDate,P.(2022) A HybridQuantum-Classical Neural
NetworkArchitectureforBinary Classification.arXivpreprint.\[Accessed15
Dec.2025\].

\[8\] Bunker,R. P.andThabtah,F.(2019) Amachinelearningframeworkfor
sportresultprediction. AppliedComputingandInformatics.\[Accessed15
Dec.2025\].

\[9\] Bergholm,V.,etal. (2018)
PennyLane:Automaticdifferentiationofhybridquantum-classical
computations.arXivpreprint.

**Groupe-projet:**

• **ThomasMuel:ProjectManager**

• **Carl** **Habsieger**

• **AlexisBusillet**

• *(Targetinga* *totalteamof4to* *handlethe*
*complexityoftheQuantumandLLMintegrations).*
