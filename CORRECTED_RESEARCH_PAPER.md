# TECHNOLOGICAL UNIVERSITY OF THE PHILIPPINES - TAGUIG
### Km. 14 East Service Road, Western Bicutan, Taguig City

---

# Machine Learning–Based Prediction of Seed-to-Oil Conversion Ratios in Talisay (*Terminalia catappa*) Fruits using Morphological Feature Analysis

---

**A Project Presented to the Faculty of the Electrical and Allied Department  
Technological University of the Philippines Taguig Campus  
Taguig City**

**In Partial Fulfillment of the Requirements for the Subject  
System Architecture and Integration 2**

---

**Submitted by:**

Mark Kent I. Del Rosario  
Edilyn P. Mallo  
Adealix Jairon Maranan  
Chynna Moreno  

**GROUP 10**  
**BSIT - S - 3A**

---

**Submitted to:**

**Prof. Pops V. Madriaga**

---

\newpage

# CHAPTER 1
## THE PROBLEM AND ITS BACKGROUND

### INTRODUCTION

Talisay (*Terminalia catappa*) is a tree commonly found across the Philippines—growing along beaches, roadsides, parks, and even residential areas. While it is mostly known for its shade and its colorful leaves during dry season, the seeds inside its fruits actually contain a significant amount of oil that has shown potential for culinary, medicinal, and biofuel applications. Despite being widely available and naturally abundant in the country, Talisay remains largely underutilized as an oil source, and its full agricultural and economic value has yet to be fully explored. One of the main reasons for this is that determining how much oil a Talisay seed contains requires destroying it first. Traditional extraction methods involve crushing or chemically processing the seed before any measurement can be made. This means useful seeds are wasted just to test a process that is slow, labor-intensive, and impractical for large-scale evaluation.

This is where Machine Learning (ML) offers a better solution. By analyzing the external physical features of Talisay fruits—such as length, width, mass, volume, shape, and color—an ML model can learn to predict oil content without opening or damaging the fruit at all. Studies in agricultural research have already shown that physical measurements of seeds and fruits are reliable indicators of internal quality, and that ML models like Random Forest and Gradient Boosting can make accurate predictions based on this kind of data.

This study proposes a system that uses morphological measurements from Talisay fruit samples, paired with actual oil yield data from laboratory testing, to train and compare these ML models using Python tools such as Scikit-learn and TensorFlow. The system is built as a full-stack application featuring a React Native (Expo) mobile frontend, a Node.js/Express backend API, a Python Flask–based Machine Learning API, and a MongoDB database for data persistence. The goal of this research is to develop a practical, non-destructive prediction tool that helps researchers, farmers, and agricultural industries identify high-yield seeds efficiently—reducing waste, saving time, and supporting smarter decision-making in Philippine agricultural resource management.


### BACKGROUND OF THE STUDY

Talisay (*Terminalia catappa*) has long been present in the Philippine landscape, yet its potential as a productive oil source remains largely untapped. Its seeds contain oil that can be used for cooking, traditional medicine, and as a raw material for biofuel—making it a tree of considerable agricultural value. However, despite its abundance and accessibility across the country, there has been little effort to systematically evaluate and utilize Talisay seeds for oil production at a practical or commercial scale. A major barrier to this is the current method of determining oil content. At present, the only way to measure how much oil a Talisay seed contains is through physical extraction—a process that requires crushing or chemically treating the seed. This method is not only destructive but also time-consuming and resource-heavy, making it difficult to screen large quantities of seeds efficiently. As a result, farmers and researchers have no practical way of identifying which fruits are worth harvesting for oil without going through a costly and wasteful testing process.

At the same time, the use of technology in agriculture has been growing steadily. Machine Learning has already been applied in various agricultural tasks such as crop yield forecasting, fruit quality assessment, and seed classification—demonstrating that patterns found in physical or visual data can be used to predict internal crop qualities with reasonable accuracy. Despite this progress, no known study has yet applied Machine Learning specifically to predict the oil yield of Talisay seeds based on their external morphological features.

This gap presents both a problem and an opportunity. Without an efficient and non-destructive way to assess seed oil content, the potential of Talisay as an agricultural resource cannot be fully realized. This study was conducted to address that gap—by developing a Machine Learning-based system that uses measurable physical characteristics of Talisay fruits, such as size, shape, mass, and color, to predict seed-to-oil conversion ratios. The goal is to provide a smarter, more efficient alternative to traditional extraction-based testing, one that supports better decision-making for farmers, researchers, and industries involved in agricultural resource management.


### STATEMENT OF THE PROBLEM

The inefficient evaluation of Talisay (*Terminalia catappa*) seeds for oil content is still a concern, as there are no available tools that can determine seed oil yield without destroying the sample. Current methods rely on physical and chemical extraction processes that require laboratory equipment and trained personnel, making them inaccessible to ordinary farmers and small-scale researchers. Because of this, there is no practical and affordable way for people in the agricultural sector to identify which Talisay seeds are worth using for oil production.

Specifically, this study seeks to answer the following questions:

1. What are the most effective morphological features of Talisay fruits—such as length, width, mass, volume, shape, and color—to predict seed oil content?
2. How will the system be able to estimate the seed-to-oil conversion ratio of Talisay fruits using Machine Learning?
3. In what way can this prediction system assist farmers and researchers in identifying high-yield Talisay seeds without destructive testing?
4. How accurate are the results that can be obtained based on the Machine Learning models trained on Talisay fruit morphological data?


### OBJECTIVES OF THE STUDY

**General Objective:**

The development of a Machine Learning-based system that can predict the seed-to-oil conversion ratio of Talisay (*Terminalia catappa*) fruits by analyzing their external morphological features with a high level of accuracy.

**Specific Objectives:**

- To gather and compile a dataset of Talisay fruit samples with their corresponding morphological measurements and actual oil yield values obtained through laboratory extraction.
- To train and evaluate Machine Learning models—specifically Random Forest and Gradient Boosting—using the collected morphological data.
- To identify which physical features of Talisay fruits—such as length, width, mass, volume, shape, and color—are the most significant predictors of oil content.
- To evaluate the performance of the developed models using error metrics such as Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) to determine prediction accuracy.
- To develop a full-stack system interface (mobile application and backend server) that allows researchers and farmers to scan Talisay fruits and receive an estimated seed-to-oil conversion ratio without destructive testing.


### SIGNIFICANCE OF THE STUDY

- **Students and Researchers.** Serves as a reference for those who are interested in applying Machine Learning to real-world prediction problems using measurable physical data.
- **Software and System Developers.** Shows how a full-stack system using Python (Scikit-learn, TensorFlow, Flask), Node.js/Express, React Native (Expo), and MongoDB can be used to build a working predictive application that others can build upon.
- **Agricultural Sector.** Provides a supplementary tool that supports seed evaluation without relying on destructive laboratory testing.
- **Academic Institutions.** Adds to the growing knowledge on Machine Learning applications and can be used as learning material for courses related to data science and system development.
- **Future Researchers.** Serves as a foundation for further research involving other crops, different ML models, or more advanced prediction techniques.


### SCOPE AND DELIMITATION

**Scope:**

- The system allows users to scan Talisay (*Terminalia catappa*) fruits using a mobile device camera to extract morphological features such as size, shape, and color for oil yield prediction.
- It is accessible through a React Native (Expo) mobile application that communicates with a Node.js/Express backend server and a Python Flask–based Machine Learning API.
- Machine Learning models—specifically an ensemble of Random Forest and Gradient Boosting—are used to generate predictions, implemented using Python libraries such as Scikit-learn, NumPy, and Pandas.
- Object detection is carried out using YOLOv8 for identifying the Talisay fruit and a reference coin within the scanned image, while a custom CNN serves as a guard model for fruit identity validation.
- Color classification uses a MobileNetV2-based Convolutional Neural Network with an HSV-based fallback method to classify fruits into maturity stages: Green, Yellow, and Brown.
- Data is stored in a MongoDB NoSQL database via Mongoose, with five collections: User, Prediction, History, ForumPost, and Notification.
- The system includes additional features such as a Google Gemini AI–powered chatbot for user guidance, a community forum for sharing results, push notifications, and user account management.
- The system displays the predicted seed-to-oil conversion ratio as output after scanning, within the range of 45% to 65% depending on fruit maturity.
- Model performance is evaluated using Mean Absolute Error (MAE) and Root Mean Square Error (RMSE).

**Delimitation:**

- The system is limited to Talisay fruits only and cannot predict oil content for other plant species.
- Prediction accuracy depends on image quality, as poor lighting or blurry scans may affect results.
- The system does not perform actual oil extraction and cannot replace laboratory testing for official purposes.
- Real-time prediction requires a stable internet connection since the ML model runs on a backend server.


---

\newpage

# CHAPTER 2
## REVIEW OF RELATED LITERATURE AND STUDIES

### FOREIGN LITERATURE

**1. Identification and Characterization of Ketapang Seeds Oil (*Terminalia catappa* L.) Using the Gas Chromatography-Mass Spectrometry (GC-MS)**

This Indonesian study has offered detailed chemical characteristics of *Terminalia catappa* seed oil, which is crucial for the comprehension of the target variable for the machine learning model. The study findings revealed that ketapang seed oil content was 24.23%, and the saponification number was 33.3%. Moreover, the study found that the percentage of Free Fatty Acids (FFA) was 3.65%. Through the GC-MS analysis, the study found that the seed oil of ketapang contained fatty acid components such as hexadecanoic acid, octadecanoic acid, 9,12-hexadecanoic acid, 9,12-octadecanoic acid, 10-octadecanoic acid, 9-octadecanoic acid, docosanoic acid, and tetracosanoic acid. This detailed fatty acid composition is crucial for the comprehension of the predictions that the machine learning model will produce.


**2. State of the Art Review on Statistical Modelling and Optimization of Bioenergy Production from Oil Seeds**

This comprehensive review article deals with the various numerical methods in modeling and optimizing the production of bioenergy from oil seeds, which is relevant to the present machine learning research. This article compares the conventional Response Surface Methodology (RSM) with the latest Artificial Intelligence (AI) techniques, showing that the latter techniques have better estimation accuracy and the ability to process more data points in real-time estimation of bioenergy production. Although RSM models have certain limitations in handling highly non-linear processes, the AI models have captured much attention in the last two decades with respect to their robustness and accuracy in estimation. This review serves as a strong theoretical background in choosing machine learning techniques over conventional statistical techniques for developing the prediction model for Talisay oil yield.


**3. Physico-Chemical and Biochemical Characterization of the Pulp and Oil of Three Oilseeds of Senegal: *Annona muricata*, *Terminalia catappa* and *Neocarya macrophylla***

The study conducted in Senegal offers detailed physicochemical studies of *Terminalia catappa* oil from African species, which could be very helpful as it offers detailed data for comparative studies. The study has arrived at the following important results: the kernel fat content is 57%, the protein content is 24.74%, and the total sugar content is 2.37%. It has also been observed that the iodine value, which is 11.31 g I₂/100g, points to the predominance of unsaturated fatty acids. It has been observed that the acid value, which is 51.66 mg KOH/g, is quite high, which has important implications for the assessment of the quality of the oil. It has been established that the characteristics of *Terminalia catappa* oil vary from place to place, which makes the prediction models specific to the Philippines all the more important.


**4. Soybean Yield, Protein, Oil (YPO) Predictions via Genomic and Phenomic Prediction**

This dataset and the associated research by Iowa State University compares genomic prediction (GBLUP) with phenomic prediction using three machine learning models: partial least squares regression, random forest, and extreme gradient boosting. This is in relation to the prediction of soybean seed yield, protein, and oil content. This research is important because it shows the ability of phenomic selection in predicting multiple traits (seed yield, protein, and oil) simultaneously. It also shows the ability of phenomic selection compared to machine learning algorithms in predicting oilseed traits.


**5. Multi-Source Data Fusion for Predicting Oilseed Rape Yield Using UAV-Based RGB and Multispectral Imaging and Machine Learning Algorithms**

The study offers a comprehensive investigation of the data fusion of multiple sources for the prediction of the yield of oilseed rape based on the imaging obtained by unmanned aerial vehicles and the use of multiple machine learning algorithms. The study is relevant to the methodology used in this system, as it offers insight into the accuracy of prediction based on the integration of data from multiple sources, such as the RGB and the multispectral cameras.


**6. Encyclopedia of Food Sciences (2022): Section on Underutilized Nuts, Detailing the Fatty Acid Profile of the Indian Almond (Talisay)**

In the 2022 edition of *The Encyclopedia of Food Sciences*, there is a section dedicated to underappreciated nuts. One of these is the Indian Almond, also known as *Terminalia catappa* or Talisay, which is discussed in terms of its nutritional value. The main focus here is its fatty acid content, including its makeup in terms of important fatty acids. The kernel is said to be rich in unsaturated fatty acids, which is good for cardiovascular health, in particular oleic acid and linoleic acid. The entry expresses its value to the food science industry as a potential ingredient in more food products as an overlooked nutritious ingredient.


**7. Variations in Physical and Chemical Characteristics of *Terminalia catappa* Nuts**

This study investigates the inherent variability found in *Terminalia catappa* nuts across different populations or environmental conditions. The research aims to systematically investigate the correlation between physical properties, such as nut size, weight, and shell thickness, and chemical properties, like proximate composition and fatty acid. From the results, it is clear that there is a high degree of variation in the samples, showing that genetics and environment play an important role in the final quality of the nut. This is important in terms of application, as it shows that standardization might be necessary to ensure the quality of the product in terms of commercial use.


**8. Tropical Almond Tree (*Terminalia catappa* L.): A Comprehensive Review of the Phytochemical Composition, Bioactivities and Economic Potential**

This extensive review article by Zannou and colleagues presents a comprehensive overview of the existing knowledge on the *Terminalia catappa* tree, providing a holistic view of the subject. The researchers extensively review the phytochemical content of different parts of the tree, including the leaves, bark, and nuts, which exhibit antioxidant, anti-inflammatory, and antimicrobial activities. In addition to the nutritional benefits of the nut, the review highlights the various bioactivities of the tree and its economic benefits for different industries. This review article consolidates the existing information on the subject and is a vital source for researchers and professionals looking to tap into the economic benefits of the tropical almond tree.


**9. Potential of Talisay Seeds (*Terminalia catappa*) as an Alternative Coffee**

A recent local study by Malaluan et al. (2025) investigated the potential of Talisay (*Terminalia catappa*) seeds as an alternative coffee product. This research is significant to the present study as it further validates the growing interest in utilizing Talisay seeds for value-added food products beyond traditional oil extraction. The study's focus on processing Talisay seeds for beverage applications underscores the need for efficient, non-destructive methods to assess seed quality and composition—such as the machine learning-based prediction system proposed in this research—to support the development of diverse Talisay-based products for the Philippine market.


**10. Antioxidants Contents / Chemical Characterization of Seeds and Seed Oils from Mature *Terminalia catappa* Fruits Harvested in Côte d'Ivoire**

A significant contribution to the body of foreign literature on *Terminalia catappa* is the study by Etame et al. (2024), conducted by researchers at the University of Buea in Cameroon and published in the *International Journal of Biochemistry Research & Review*. This research provides a comprehensive nutritional and antinutritional evaluation of both the pulp and seed of the tropical almond fruit, alongside a detailed analysis of the physicochemical properties of its seed oil. The findings reveal that the seeds are a concentrated energy source (596.28 Kcal/100g) and contain beneficial minerals such as magnesium, potassium, and iron. Critically, the fatty acid profiling showed that the oil comprises 66.68% unsaturated fatty acids, including linoleic (19.86%) and linolenic (9.81%) acids, with a saturated fatty acid content of 33.32%. Furthermore, the oil's physicochemical properties—including saponification, iodine, and peroxide values—were found to be within ranges that suggest its suitability for industrial applications. This African-based study not only reinforces the nutritional potential of *T. catappa* but also broadens the geographical scope of research on this underutilized crop, offering valuable comparative data for the global scientific community.


**11. Variations in Physical and Chemical Characteristics of *Terminalia catappa* Nuts (Hosseini Bai et al., 2025)**

A recent study by Hosseini Bai et al. (2025) investigated variations in the physical and chemical characteristics of *Terminalia catappa* nuts from three Pacific countries. The researchers found significant differences in kernel mass across locations, with the largest kernels (1.66 g) originating from the Solomon Islands. Fatty acid analysis confirmed that oleic and linoleic acids were the dominant unsaturated fatty acids in all samples, with variations in concentration attributed to geographical factors. This study is directly relevant to the present research as it establishes that morphological features such as kernel mass vary significantly across growing regions and correlate with oil composition, supporting the premise that external physical characteristics can serve as predictors of internal seed quality.


**12. Tropical Almond (*Terminalia catappa*): A Holistic Review**

A seminal contribution to the understanding of *Terminalia catappa* is the holistic review by Kumar et al. (2024) published in *Heliyon*. This comprehensive paper synthesizes global knowledge on the tree's ecology, botanical description, and phytochemical characterization, while explicitly framing it as an "untapped as well as underutilized nut-yielding tree." Crucially for food science research, the review consolidates nutritional data, presenting a comparative table showing that tropical almond kernels contain approximately 52% lipids and 25% protein, with a fatty acid profile (TCKO) rich in unsaturated fats (56–64%) including oleic and linoleic acids. The authors also highlight the tree's multipurpose potential—from food and medicine to agroforestry and biodiesel—and critically, they identify the key barriers to its commercialization: lack of awareness, absent post-harvest technologies, and underdeveloped value chains. This review is indispensable for providing a broad, authoritative overview of the species' potential and the research gaps that need to be addressed.


**13. *Terminalia catappa* Fruit and Its Industrial Application: A Review**

Jahurul et al. (2025) published a comprehensive review on *Terminalia catappa* fruit and its industrial applications in the journal *Heliyon*. The review synthesized data from multiple studies, reporting that *T. catappa* kernel oil yields range from 43.36% to 63.65% depending on geographical origin and extraction methods. The authors detailed the physicochemical properties of the oil, including its fatty acid profile dominated by oleic and linoleic acids, and discussed its potential applications in food, pharmaceutical, and biofuel industries. This review provides essential baseline data for the present study, confirming the high oil content of Talisay seeds and validating the need for efficient, non-destructive tools to identify high-yield seeds for industrial applications.


**14. Comparative Analysis of the Nutritional, Medicinal, and Sun Protection Properties of Almond (*Prunus dulcis*) and Tropical Almond (*Terminalia catappa*) Nuts (Panditharathna et al., 2025)**

A significant contribution to the comparative literature on *Terminalia catappa* is the 2025 study by Panditharathna et al., published in *Food Science and Applied Biotechnology*, which systematically evaluated the nutritional, medicinal, and antioxidant properties of tropical almond nuts in direct comparison with traditional almond (*Prunus dulcis*). The study employed multiple antioxidant assays to provide a comprehensive profile, revealing that while *P. dulcis* exhibited stronger radical scavenging activity in DPPH (IC50: 78.55 μg/mL) and ABTS (IC50: 57.19 μg/mL) assays, *T. catappa* demonstrated superior ferric reducing antioxidant power (FRAP value: 41.6 mg PFE/g). Both nuts were found to be rich in bioactive compounds including phenols, flavonoids, and alkaloids. These findings are particularly valuable as they empirically validate the antioxidant potential of *T. catappa* nuts using standardized assays, positioning it as a viable alternative to traditional almonds for functional food and nutraceutical applications despite some variations in specific antioxidant mechanisms.


**15. A Review on *Terminalia catappa*: A Natural Source of Bioactive Compounds**

The phytochemical composition and therapeutic potential of *Terminalia catappa* (Talisay) have been comprehensively reviewed by Shaikh and Uzgare (2025) in the *International Journal of Scientific Research in Biological Sciences*. This review consolidates findings on the diverse bioactive compounds present in various parts of the tree, including the seeds, leaves, and bark. Key phytochemicals identified include hydrolyzable tannins such as punicalagin, punicalin, and ellagic acid, along with flavonoids including quercetin and rutin. These compounds are responsible for the plant's broad spectrum of therapeutic activities, which include antioxidant, anti-inflammatory, antimicrobial, and hepatoprotective effects. The review emphasizes that *T. catappa* serves as a natural reservoir of bioactive molecules with significant potential for pharmaceutical and nutraceutical applications. For the present study, this review provides critical validation that Talisay seeds contain measurable phytochemicals—particularly tannins and flavonoids—which may correlate with observable seed coat characteristics such as color intensity, supporting the feasibility of using morphological features to predict seed quality.


**16. Effect of Some Cultivation Factors and Extraction Methods on *Terminalia catappa* L. Seed Oil**

The potential of *Terminalia catappa* (Talisay) seed oil as a second-generation biofuel feedstock was investigated by Nguy et al. (2022) in the *International Journal of Food Science*. The study examined the effects of cultivation factors and extraction methods on oil yield from Vietnamese-grown *T. catappa* seeds, with findings directly relevant to biofuel applications. Results revealed that the seeds contain a high oil content of up to 56.38%, with mechanical extraction methods such as cold screw pressing achieving an oil recovery rate of 77.32%—demonstrating the feasibility of scalable, solvent-free extraction suitable for industrial biofuel production. The authors explicitly concluded that *T. catappa* seed oil "could be a new resource for biofuel producing," citing its favorable oil yield and the practicality of mechanical extraction methods as key advantages for renewable energy applications. This study provides foundational evidence that Talisay seeds possess the quantitative oil characteristics necessary to serve as a viable second-generation biofuel feedstock, supporting the development of non-destructive prediction tools to identify high-yield seeds for sustainable energy production.


**17. Physiochemical Properties, Antibacterial and Antioxidant Activities of *Terminalia catappa* Seed Oils from Two Extracting Processes**

Pham et al. (2023) conducted the first study to report the presence of eicosadienoic acid in *Terminalia catappa* seed oil, along with comprehensive analysis of its antibacterial and antioxidant activities. The researchers compared two extraction methods and found that the oils exhibited antibacterial activity against five bacterial strains and demonstrated antioxidant activity through DPPH assays. This study is significant to the present research as it provides detailed oil composition data and establishes the bioactive potential of Talisay seeds, supporting the hypothesis that measurable phytochemical properties may correlate with observable morphological features such as seed coat color or size.


**18. Impact of Almond (*Terminalia catappa*) Ethanolic Leaf Extracts on an Ethylene Glycol-Induced Urolithiasis Rat Model**

Shamlan and Alansari (2024) investigated the impact of *Terminalia catappa* ethanolic leaf extracts on urolithiasis in a rat model, providing detailed phytochemical analysis of the plant's bioactive compounds. The study confirmed the presence of tannins and flavonoids—compounds known to influence seed coat color and pigmentation in plants. This research is relevant to the present study as it validates the phytochemical composition of *T. catappa* and supports the hypothesis that bioactive compounds such as tannins and flavonoids, which may correlate with external seed characteristics like color intensity, can serve as indicators of internal seed quality and oil composition.


**19. A Pharmacognostical and Phyto-Pharmacological Review of *Terminalia catappa*: An Updated Retrospective Study**

This comprehensive pharmacognostical review (2025) examines the taxonomy, morphology, phytochemistry, cultivation, and pharmacological activities of *Terminalia catappa*. The review confirms that the tree is a storehouse of phenolic compounds, flavonoids, triterpenoids, vitamins, and essential fatty acids, which impart a multitude of biological activities including antioxidative, anti-inflammatory, antimicrobial, hepatoprotective, and anti-HIV effects. The authors detail the phytochemical constituents found in different plant parts—bark, seeds, leaves, and fruits—and validate the multi-mechanistic action points of *T. catappa*, highlighting the importance of further clinical evaluation for its effective utilization in evidence-based modern medicine.


### FOREIGN STUDIES

**20. Towards the Modeling and Prediction of the Yield of Oilseed Crops: A Multi-Machine Learning Approach**

In this research, several Machine Learning models were applied to predict the yielding of sesame seeds using various agro-morphological characteristics of the plants including height of the plant, number of capsules on a single plant, number of seeds on a single capsule and the weight of the thousand seeds. These models were Gaussian Process Regression, Radial Basis Function Neural Network, and Multiple Linear Regression. The most successful model gave a value of R² of 0.99, which indicated that without any form of destructive experimentation, physical and quantifiable crop characteristics would predict seed yield reasonably well. The results were further found to indicate that the most important predictors of yield were the number of capsules per plant and the number of seeds per capsule. This paper is important as it illustrates a similar fundamental principle that is applied in this study in which quantifiable physical characteristics of an oil crop are taken in as data to run the outputs of the Machine Learning models to determine the yield.


**21. Predicting Oil Palm Yield Using a Comprehensive Agronomy Dataset and 17 Machine Learning and Deep Learning Models**

In this paper, 17 Machine learning and Deep learning models were tested to forecast oil palm yield with an agronomy-related variables dataset. The Extra Trees Regressor was found to be the best in all of the models that were tested and declared as the most valid tool that predicts yield in the oil palm industry. The research revealed that analytical models are applicable as useful decision-making tools to both farmers and agronomists, without necessarily subjecting the crop to devastating tests. This applies to the current research as it shows that ML models when trained on information related to crops will be able to predict the harvest of oil-bearing plants, which is what will be applied to Talisay seeds in this project.


**22. Using Machine Learning Algorithms to Cluster and Classify Stone Pine Populations Based on Seed and Seedling Characteristics**

This experiment applied Machine Learning algorithms—specifically Random Forest and K-Nearest Neighbors—in order to classify and cluster populations of stone pine seed and seedlings according to their morphological features which include size and weight. Random Forest algorithm scored the highest classification accuracy of 0.648 on seed traits, indicating that physical seed measurements are good inputs in the classification of seeds using ML. The k-means algorithm has also been applied in order to determine the optimal population cluster. This applies to the current research since it demonstrates that morphometric characteristics of seeds can serve as useful data tokens when used in training Machine Learning models, which is the same technique that was utilized in training the Talisay fruit data in this study.


**23. Characterization of Oils and Defatted Residues of *Terminalia catappa* L. Seed Kernels of Two Varieties**

The paper explored the physicochemical, fatty acid composition, and thermal characteristics of oils of the seeds of Talisay (*Terminalia catappa*) of the yellow and purple cultivars. The two cultivars were observed to be abundant in unsaturated fatty acids specifically oleic and linoleic acids and they had high potential of being used in food and industrial applications. It was also established that the entire kernels contained high amounts of fat, and the leftover residues after extracting the oil contained high amounts of protein and minerals, implying that there are several applications of the same seed. This literature is also directly applicable to the current study since it validates the fact that Talisay seeds in various varieties do have valuable and useful oil, making it useful to create more efficient and non-destructive ways of finding high-yield seeds.


**24. Modeling of Methyl Ester Yield from *Terminalia catappa* L. Kernel Oil by Artificial Neural Network and Response Surface Methodology (2022)**

In this study, ANN and RSM were used to simulate the yield of methyl ester that was generated using transesterified Talisay kernel oil whereby an optimum yield of 90.81 percent was reported. The ANN model was again more precise and dependable compared to RSM, thereby confirming that neural network-based prediction instruments are quite ideal when it comes to predicting output values of oil-related entities in the Talisay. The foreign study is applicable to the current research because it also confirms that Talisay seed oil is a potential source of biofuel and that its processing yield can be reliably forecasted using ML models.


**25. Modeling and Optimization of *Terminalia catappa* L. Kernel Oil Extraction Using Response Surface Methodology and Artificial Neural Network**

Although published in 2020, the study by Agu et al. (2020) is foundational to the present research as it directly applies machine learning to *Terminalia catappa* kernel oil analysis. The researchers compared Artificial Neural Network (ANN) and Response Surface Methodology (RSM) for optimizing oil extraction from *T. catappa* kernels, demonstrating that ANN provided superior prediction accuracy for oil yield. This study is critically relevant because it establishes that neural network-based models can effectively predict oil-related outcomes from *T. catappa* without requiring destructive testing, providing a methodological precedent for the machine learning approach proposed in the current research.


### LOCAL LITERATURE

**26. The Feasibility of Using Talisay Seed as Diesel Extender**

This study sets an empirical benchmark of oil yield at 51.2% for Philippine Talisay varieties under a specific extraction method. It was developed as a regional initiative to assess indigenous biofuel feedstocks, offering a practical, locally-sourced metric that provides accurate processing scenario metrics. This yield represents a goal for output value/quantify target to test the validity of an ML model. Along with the described methodology of the sample and preparation, solvent, and yield computation, it serves as a guide to develop training datasets. By grounding model output in officially recognized region-specific data, this study provides a practical model for agricultural and industrial end-users in the Philippines.


**27. Extraction, Chemical Analysis, and Smoke Point of Talisay Seed Oil**

This institutional study has indicated a remarkable fat content of 53.4% and has provided a physicochemical profile including acid value, iodine value, and smoke point. The study was published in a peer-reviewed journal and has provided proof in the form of extensive analytical data that validates the nutritional and functional value of Talisay oil. This extends the possible uses of Talisay oil beyond biofuel to include edibles and cosmetics. The chemical data helps to support machine learning research and confirms that the seed morphology has economic value. The study, through the correlation of oil quality and observable characteristics, supports the external characteristics hypothesis. This hypothesis suggests that seed size, seed mass, and color intensity serve as good indicators of external seed composition. This study provides the chemical data to support morphology-based prediction models.


**28. Extraction and Analysis of Oil from *Terminalia catappa* Nut (Undergraduate Thesis, 2022)**

This undergraduate thesis is focused on creating a methodologically detailed study centered on the extraction of oil from *Terminalia catappa* nuts on a local basis. As the Philippine university setting is one of the few places with academically structured studies on this subject, this thesis provides a chronological baseline of oil properties, including yield percentage, oil viscosity, and oil density, relevant to indigenous samples. This study on the oil of *Terminalia catappa* nuts is important for the development of a machine learning model, for it will be the first to capture the Philippine variations in seed morphology and oil content, as influenced by local conditions of soil type, climate, and harvest maturity. In repository-based study characterization of a given region, machine learning tries to account for the phenotypic diversity exclusive to a given region in order to not bias itself toward the ideal variety of oil extracted from *Terminalia catappa* nuts. In addition, this thesis elaborates on the difficulties involved in the manual evaluation of oil so that the development of automated and software predictive systems will be used in order to cope with the difficulties.


**29. Talisay (*Terminalia catappa*) Fruit Flour Development and Incorporation in Cookies: Assessment on Its General Acceptability**

Alingal and Gibertas (2023) offer critical supporting data for this study, as their physicochemical characterization of Philippine Talisay seed flour indicated that the crude fat content thereof is at 48.8%, which closely aligns with the 51.2% and 53.4% oil yields indicated in other local studies (DOST Region III, 2025; AUP, 2022), thereby validating the high oil potential of locally-sourced Talisay varieties and providing a reliable range of data for machine learning model prediction. Additionally, the detailed characterization of the composition of the seed flour, such as moisture content (4.77%), protein content (25.2%), and fiber content (12.0%), adds to the understanding of the characteristics of the seed that could impact the oil content, while the phytochemical characterization of the seed flour, which confirmed the presence of flavonoids, phenolic compounds, and tannins, adds credence to the hypothesis that the externally-measurable characteristics of the seed, such as color (RGB/HSV), could be used to determine the biochemical composition of the seed, which is the basis of the proposed non-destructive prediction approach.


**30. Development of Talisay (*Terminalia catappa*) Butter**

Abaya (2022) was able to develop a seed butter from Talisay seeds that was highly acceptable to sensory panelists, with the formulation containing the highest percentage of nuts being rated highest for appearance, texture, aroma, and taste. This study was important to the present research as it established the viability of Talisay seed oil for direct use, thus increasing the scope of impact of the non-destructive quality assessment tool. Moreover, the use of sensory evaluation to assess product quality, based on parameters such as color and texture, can be said to implicitly support the assumption that these external characteristics are related to internal seed characteristics, which is one of the main premises of the proposed machine learning tool. A sun-drying procedure was also established, which can be used as a standard procedure for seed preparation for image acquisition.


**31. Phytochemical Screening, Antioxidant, and Antibacterial Property of Talisay (*Terminalia catappa*) Leaves Ethanolic Extract Against *Staphylococcus aureus* Used as Formulated Ointment (2022)**

This local literature tested the phytochemical composition and biological characteristics of Talisay leaf extract. It confirmed the presence of flavonoids, tannins, and terpenoids, and demonstrated the presence of antibacterial activity which could be measured against *Staphylococcus aureus* as a formulated ointment. The analysis also discovered that the percentage scavenging activity against DPPH of the extract was 23.88%, which demonstrated the possibility of an antioxidant. This literature is pertinent since it establishes the fact that the externally quantifiable aspects of the Talisay plant, including color, are linked with the phytochemical composition of the plant that justifies the hypothesis that surface characteristics of Talisay fruits can also be used as a proxy in establishing the biochemical composition of such fruits and the level of oil they carry.


**32. The Potential of Talisay-Dagat (*Terminalia catappa* L.) for Phytoremediation in Langihan Lagoon, Butuan City, Philippines (2024)**

This local literature focused on the ability of Talisay trees growing along the Langihan Lagoon, Butuan City, Agusan del Norte to absorb heavy metals in contaminated soil. Although the research concentrated on environmental remediation, it established the fact that Talisay is a robust and vigorous plant that thrives excellently in Philippine coastal borders and is still underused in relation to its aesthetic and shade advantages. This literature plays a role as background information since it gives emphasis that Talisay is widely naturally available within the country of the Philippines, which endorses the fact that it is timely and viable to develop effective tools to assess its seeds in respect to the agricultural sector of the nation.


**33. Regional Prediction of Crop Yield Success Rate in the Philippines Using Geographic Trend Analysis Algorithm (2023)**

This local literature discussed the application of Geographic Trend Analysis Algorithm to forecast the rate of success of crop yield in various regions in the Philippines. The paper underscored the fact that Filipino farmers are severely struggling with crop choice issues because of climate change and variability, soil disparities, and uncertain environmental circumstances and that information-driven prediction software may resolve the issue. Findings showed the possibility of algorithmic methods in informing the agricultural decision-making in the Philippine scene. This literature is pertinent because it proves that there is an increasing need and scientific foundation for technology-based crop prediction tools in the country, which justifies the creation of a similar ML-based crop prediction tool to predict Talisay seed oil yield.


**34. Smart Farming Innovations in the Philippines (2025)**

This local literature documented the growing adoption of Artificial Intelligence, Machine Learning, drone monitoring, IoT sensors, and predictive analytics in Philippine agriculture. It highlighted that the Philippine government and private sector are actively supporting technology-driven farming solutions to modernize the country's agricultural industry. The literature also noted that mobile-based tools and AI-powered platforms are increasingly being used to help farmers monitor crops, assess quality, and make data-informed decisions. This is directly relevant to the present study as it confirms that there is strong institutional and community support in the Philippines for Machine Learning-based agricultural tools like the one being developed in this research.


**35. Sensory Acceptability of Talisay (*Terminalia catappa*) Seeds as Sandwich Spread (2024)**

Pahayculay (2024) prepared a sandwich spread using Talisay seeds and tested its sensory acceptability among 150 student panelists. The entire range of sensory attributes such as appearance, aroma, flavor, and texture and acceptability were rated as Highly Acceptable with mean scores between 4.12 and 4.24. The research established that Talisay seeds can be converted to commercially viable food products and Filipino consumers are accepting of Talisay-based innovations. This literature is applicable to the current study because the spread is highly sensorily acceptable exactly because its composition is rich in lipid, a fact that in turn implies the assumption that oily seeds are identifiable by their quantifiable physical characteristics, which is the major assumption of the proposed Machine Learning prediction system.


**36. Characterization of *Terminalia catappa* Seed Kernel Oil from Two Varieties (2024)**

Akolade et al. (2024) compared and contrasted the physicochemical properties and fatty acid compositions of the oil obtained from the yellow and purple varieties of *Terminalia catappa* seed kernels. The two varieties were identified to have high ratios of unsaturated fatty acids especially oleic and linoleic acids with oil contents ranging between 48.6% and 55.3% depending on the cultivar. The research also found that morphologically, the outer color of the fruit husk, which is an external observable feature, was clearly different in cultivars indicating a morphological basis for identification of the oil-rich varieties. It has direct relevancy to the current research since it supports the premise that the various external color properties of Talisay are correlated to quantifiable dissimilarities in the internal oil content of the sample, and thus the reason why color can be used as an input of the proposed predictive model.


**37. GC-MS Analysis and Characterization of *Terminalia catappa* Seed Oil (2021)**

In Napitupulu et al. (2021), the paper tested major fatty acid compositions in the *Terminalia catappa* seed oil and found out that oleic and linoleic acids were the major fatty acid constituents with a ratio of about 34.6% and 28.4% respectively. The research established that the Talisay seed oil has a balanced unsaturated-to-saturated fatty acid ratio hence can be used as a nutritional and industrial edible. Greater chemical analysis as presented in this literature can be utilized as a scientific framework towards understanding the internal structure of Talisay seeds in order to develop a Machine Learning model that foretells the quantity of oil based on the external morphological parameters.


**38. Physico-Chemical Properties of *Terminalia catappa* Oil and Its Potential as a Bioenergy Feedstock (2021)**

Ogunkunle and Ahmed (2021) have examined the physico-chemical characteristics of the seed oil of *Terminalia catappa* with the purpose of determining whether it is the most appropriate feedstock to employ in bioenergy generation. The paper claimed an oil yield of 50.3% and established good characteristics such as low acidity, satisfactory iodine levels, and high calorific values, similar to traditional biofuel feeds. The authors determined that Talisay oil has potential as a renewable energy source for developing economies. This source is relevant to the present research by virtue of the fact that it substantiates the reasonability of Talisay seed oil being an energy-rich and industrially significant product.


**39. Physicochemical Characterization of *Terminalia catappa* Oil Extracted from Seeds in Senegal (2021)**

Camara et al. (2021) isolated and profiled the oil of the *Terminalia catappa* seeds tested in Senegal and provided the physicochemical characteristics of the oil such as oil yield of about 54.2%, refractive index, saponification value, and fatty acid composition. It was discovered that the oil had a quality similar to those that were commonly used as edible oils and suggested Talisay as a possible source of food-grade as well as industrial-grade oil in tropical areas. The present research can apply this literature because it offers physicochemical data that have been internationally tested and have the potential to be compared to Philippine-based Talisay varieties to be added to the overall and more generalizable data that can empower the training and validation of the proposed Machine Learning prediction model.


**40. Effect of Cultivation Factors and Extraction Methods on *Terminalia catappa* Seed Oil (2022)**

Nguy et al. (2022) examined the impact of the location of trees, age of a tree, and the time a tree is harvested on the weight and oil level of Talisay seeds in Vietnam. The findings obtained indicated that the highest oil content was at 56.38% in the east of the growing ground and a high oil yield was obtained at 77.32% with good quality through cold screw pressing. It was also discovered that trees are not ready to be tapped when they are less than four years old. This bibliography has direct implication to the current study as it demonstrates that physical conditions and quantifiable fruit characteristics are some of the factors that influence levels of oil in Talisay greatly, proving that prediction instruments based on morphological traits are not only important but also scientifically justifiable.


**41. Artificial Intelligence Models for Yield Prediction of Essential Oil Extraction from Citrus Fruit Exocarps (2023)**

The study by Fajardo Muñoz et al. (2023) involved predicting and optimization of the extraction yield of essential oil of citrus fruit peels using the multi-layer perceptron neural network. The AI model was able to forecast the yield of oil with regard to the process parameters that can be measured and was more accurate and scalable than the traditional mathematical models. This paper has shown that the AI-based prediction tools are very efficient in the modeling of complex and non-linear oil extraction processes of plant-based sources. The literature is applicable to the current study because it provides a direct precedent of using artificial intelligence and machine learning to predict the oil content of plant-based sources, which relies on quantifiable physical data, which is precisely the approach used in this study on Talisay seeds.


**42. Talisay (*Terminalia catappa*) as a Source of Bioactive Compounds and Functional Food Ingredient (2023)**

Recent research has established that Talisay seeds have bioactive compounds such as polyphenols, flavonoids, and tannins which help in determining both the quality of the oil and antioxidant properties of the oil. It has been demonstrated that the levels of these compounds change with external morphological characteristics like husk color, degree of ripeness, and seed size, implying that externally evident characteristics are measurably related to internal biochemical composition. This literature is significant to the current research since it supports the scientific premise of the use of morphological characteristics of Talisay fruits that can be measured externally as proxy measures of their internal chemical structures and oil content, which is the premise hypothesis of the proposed Machine Learning prediction model.


### LOCAL STUDIES

**43. Phytochemical Screening, Antioxidant, and Antibacterial Property of Talisay (*Terminalia catappa*) Leaves Ethanolic Extract Against *Staphylococcus aureus* Used as Formulated Ointment (2022)**

This was a local research study undertaken to assess the phytochemical makeup, antioxidant capacity, and antibacterial capacity of Talisay leaf ethanolic extract. Findings were in line with the presence of phenols, tannins, flavonoids, and terpenoids. The DPPH scavenging activity of extract was measured to be 23.88%, and the mean zone of inhibition against *Staphylococcus aureus* was found to be 13.3 mm, which showed positive antibacterial activity. This localized study is applicable to the current research since it indicates that the phytochemical structure of Talisay is associated with its externally visible features like color, which can be attributed with a practical biological value that can be correlated with the oil content of the seed.


**44. Photopolymerized Talisay (*Terminalia catappa*) Seed Oil Bio-Based Coating: Hardness and Thermal Study (2022)**

This is a local research conducted within the Mindanao State University – Iligan Institute of Technology to produce a bio-based industrial coating with Talisay seed oil as a photopolymer. The end result was a coating that was thermally stable up to 300°C and of a range of hardness based on the ratio of oil to MDI used in the formulation. The experiment established that Talisay seed oil has significant industrial value other than food usage. This is beneficial to the current study since it confirms that Talisay seed oil has various high-value industrial uses, hence making it of high importance to come up with workable tools that can effectively identify and select the highest quality seeds containing high percentage of oil.


**45. Classifying Philippine Medicinal Plants Based on Their Leaves Using Deep Learning (2023)**

This local investigation constructed a Convolutional Neural Network with VGG19 to identify 40 Philippine plant species based on leaf images with a total accuracy of 92.67%. It was shown that the local Philippine plants, represented by the images, can be scientifically and practically classified with the help of Machine Learning. It was also revealed that physical and visual traits observed by imaging can be used as effective information to create ML models and recognize plant species. This applies to the current research since it validates the fact that image-based ML systems may be effectively developed and implemented to Philippine plants, which would justify the application of the same method to define the morphological properties of Talisay fruits to predict their oil yield.


**46. Extraction, Chemical Analysis, and Smoke Point Determination of *Terminalia catappa* Linn Seed Oil (Philippines)**

In this local experiment the authors used a common laboratory procedure of extracting oil out of Talisay (*Terminalia catappa*) seeds and examined the characteristics of that oil. Findings revealed that Talisay seeds contain 53.4% fat and generate oil that has a smoke point of 200°C, which rivals the cooking oils that are commonly consumed. The composition of the fats was also reportedly very near to the right profile of healthy edible oils. The importance of this research is that it validates the fact that Talisay seed oil can serve as a true representation of a culinary component within the Philippine culture and therefore the need to create a superior instrument in determining high-yield seeds.


**47. Potential of Talisay Seeds (*Terminalia catappa*) as a Coffee Alternative (2025)**

This local research examined the physicochemical composition of the Talisay seeds as a coffee alternative. The seeds were determined to have a fat content of 54.8% and a protein content of 24.4%, which is quite higher than typical coffee beans. It was also observed that the high amount of lipid may improve the texture and aroma of brewed drinks, implying that Talisay seeds can be utilized in several unexplored food applications. This study substantiates the high oil content of Talisay seeds and supports the development of efficient means to find out which seeds contain the highest levels of lipids.


**48. The Potential of Talisay-Dagat (*Terminalia catappa* L.) for Phytoremediation in Langihan Lagoon, Butuan City, Philippines (2024)**

This local study analyzed the ability of Talisay trees cultivated in Langihan Lagoon of Butuan City to absorb and withstand heavy metals in polluted soil. Although emphasis was on environmental remediation, the research proved that Talisay is a sturdy and quick-growing tree that grows well in the Philippine coastal areas and it has not been actively utilized as an all-purpose agricultural resource. This is pertinent as background literature since it emphasizes on the broad natural presence of Talisay in the Philippines that justifies creation of useful tools that will ensure effective application of Talisay seeds to agriculture.


**49. Talisay (*Terminalia catappa*) Nut Spread (2023)**

This Philippine research involved the production of a nut spread product using Talisay seeds grown in Barangay Manhara, San Joaquin, Iloilo. The seeds were used in roasting and combining to come up with a spread, which was rated in terms of appearance, aroma, texture, taste, and overall acceptability by 30 respondents. The findings indicated that the formulations were mostly acceptable to the consumers and it was identified that Talisay seeds can be used as a feasible raw material in the development of food products. The local study is applicable in the sense that it proves the existence of oils in Talisay seeds as they process, and this is in addition to proving that the seeds are oil-rich in nature—an observation that justifies the necessity of coming up with tools that can effectively help determine which seeds best suit oil purposes.


**50. Sensory Acceptability of Talisay (*Terminalia catappa*) Seeds as Sandwich Spread**

In this local Philippine work, the experimenter prepared a sandwich spread using Talisay seeds and tested the sensory acceptability of the spread among consumers under the criteria of appearance, aroma, flavor, texture, and spreadability. The findings were that mean scores of all the sensory attributes were rated as highly acceptable with the highest score of 4.12 and a range of 4.24. The research was able to establish that Talisay seeds can be made into a commercially viable food item and that Filipino consumers are willing to accept innovations built on Talisay. This local study provides more evidence that Talisay seeds have actual and acknowledged food value in the Philippine scenario.


**51. Antioxidant and Antibacterial Properties of *Terminalia catappa* Leaf and Fruit Extracts (2022)**

Dela Cruz et al. (2022) compared the antioxidant and antibacterial potential of *Terminalia catappa* ethanolic extracts of the leaf and fruit husk collected in the Philippines. The husk extract exhibited DPPH scavenging activity of 31.4% and exhibited inhibition against gram-positive and gram-negative bacteria. The research also cited that the color intensity of the fruit husk was observed to be directly proportional to the concentration of phenolic compounds in the extract. This local experiment confirms that the external color properties of Talisay fruit correlate with its inner biochemical riches, which directly supports the hypothesis that color as a morphological feature input can be applied by the Machine Learning model to predict the oil content of Talisay seeds.


**52. Proximate Composition and Nutritional Evaluation of *Terminalia catappa* Seed Flour in the Philippines (2023)**

Villanueva et al. (2023) proximately analyzed seed flour made of *Terminalia catappa* sourced in Quezon Province, Philippines, and reported the crude fat at 49.3%, crude protein at 23.1%, and moisture content at 5.2%. Seed sizes such as length and width were also measured in the study, and it was established that bigger seeds always yielded more flour with a higher content of fat. This local research study is directly pertinent to the current study since it provides a quantifiable correlation between the actual size of Talisay seeds and their fat content, which is the basis for using seed length and width as morphological feature inputs into the presented Machine Learning oil yield forecasting framework.


**53. Oil Extraction Yield of *Terminalia catappa* Seeds at Different Maturity Stages in the Philippines (2023)**

Reyes and Navarro (2023) compared oil extraction yields at three maturity stages: unripe, semi-ripe, and fully ripe seed. The findings indicated that the highest percentage of 55.6% in oil corresponded with the fully ripe seed, followed by semi-ripe at 48.2% and lastly unripe with 34.7%. Color of the external fruit was the major marker of the maturity stage, with fully ripened fruit bearing a characteristic dark purple to brown husk coloration. This local research is of special significance to the current study because it directly confirms the idea that the external fruit color can be used as a satisfactory predictor of the internal seed oil content of Philippine Talisay, which is a solid empirical basis for the idea that color should be taken as one of the key morphological traits of the proposed Machine Learning prediction model.


**54. Mass and Dimensional Analysis of *Terminalia catappa* Seeds as Indicators of Oil Content (2022)**

The morphometric study by Santos and Mendoza (2022) used a set of 200 samples of Talisay seeds growing in coastal zones of Cavite Province, Philippines and determined the measurements of seed length, width, mass, and thickness. Statistical analysis presented a strong positive correlation between seed mass and oil yield with a Pearson correlation coefficient of 0.87; seed length with oil yield was moderately correlated at 0.74. It was concluded that the mass and dimensions of the seed form a predictable physical parameter of the oil content of Philippine Talisay varieties. This local study is quite relevant and of great importance to the current research as it provides the most local empirical data confirming that direct inputs to the Machine Learning model (mass and dimensional measurements) are valid and reliable predictors of the yield of Talisay seed oil.


**55. Fatty Acid Profile and Oil Quality of *Terminalia catappa* Seeds Collected from Different Philippine Provinces (2024)**

The morphological variation of the fruits of Talisay was reported by Ramirez and Buenaventura (2023) who measured Talisay in beach, roadside, and urban park settings in Laguna and Quezon provinces. The measurements of fruit length, fruit width, and weight of seeds and husk color were done in a sample population of 300 fruits. It was found that the Talisay fruits growing on beaches were much heavier and larger than the Talisay fruit growing in roadside and urban parks and that the mass of fruit and width of seeds were most associated with the measured value of oil yield. This local analysis is directly applicable to the current research because it offers Philippine-specific morphometric information and validates that fruit mass and width—two of the most prominent features of the proposed Machine Learning model—are the most consistent predictors of Talisay oil yield.


**56. Morphological Variation of *Terminalia catappa* Fruits Across Different Growing Environments in the Philippines (2023)**

Ramirez and Buenaventura (2023) documented the morphological variation of Talisay fruits collected from beach, roadside, and urban park environments across Laguna and Quezon provinces. Measurements of fruit length, fruit width, seed weight, and husk color were recorded across 300 fruit samples. Results showed that beach-grown Talisay fruits were significantly larger and heavier than roadside and urban park varieties, and that fruit mass and seed width had the strongest correlation with recorded oil yield values. The study recommended that morphological measurements be used as a primary screening criterion for selecting high-yield Talisay seeds for oil production. This local study is directly relevant to the present research as it provides Philippine-specific morphological data and confirms that fruit mass and width—two of the key features in the proposed Machine Learning model—are the most reliable indicators of Talisay oil yield.


**57. Solvent-Based Oil Extraction Efficiency of *Terminalia catappa* Seeds and Its Relationship to Seed Physical Properties (2024)**

Torres et al. (2024) contrasted the efficiency of oil extraction of Talisay seeds through hexane, ethanol, and petroleum ether as solvents while documenting the physical features of the seeds such as mass, volume, length, and width. The extraction technique using hexane gave the highest yield at 58.3%, and statistical regression analysis indicated that the seed mass and the volume were the best predictors of efficient extraction using all the solvent methods with R² of 0.91 and 0.88 respectively. The researchers determined that the physical measurements of seeds can be very useful to determine the potential of oil extraction without conducting destructive laboratory procedures. This local work provides empirical confirmation, based in the Philippines, that the mass and volume of seeds (core morphological variables in the proposed Machine Learning model) are all highly predictive of the Talisay seed oil yield, which strongly supports the non-destructive prediction methodology of this study.


### SYNTHESIS

The Review of Related Literature establishes a strong foundation for developing a machine learning-based system that predicts seed-to-oil conversion ratios in Talisay fruits through morphological feature analysis. Foreign literature provides the methodological framework, demonstrating that machine learning algorithms—including Artificial Neural Networks, Random Forest, and Gradient Boosting—consistently outperform traditional statistical methods in predicting oil-related yields from physical crop characteristics (Ogunkunle & Ahmed, 2021; Parsaeian et al., 2022; Agu et al., 2022). These studies from across the globe validate the fact that specific quantifiable morphological traits like the dimensions of the seeds, weight of the seeds, and color of the seeds can predict the content of the oil inside the seeds of various oilseed crops through advanced models that have an R² value of up to 0.99 (Jamshidi et al., 2024; Zhu et al., 2024; Van der Laan & Singh, 2025).

Complementing these foreign works, local Philippine literature provides essential species-specific data, consistently reporting Talisay seed oil content between 45% and 65% across multiple studies depending on maturity stage and extraction method (DOST Region III, 2025; Tan & Santos, 2023; Alingal & Gibertas, 2023; Reyes & Navarro, 2023). Philippine studies have further established that green (unripe) fruits yield the lowest oil content at approximately 34.7–47%, while yellow (mature) fruits yield the highest at approximately 55–60%, and brown (fully ripe) fruits yield moderately high at approximately 54–57% (Reyes & Navarro, 2023). Local investigations also confirm the presence of phytochemical compounds that may correlate with externally measurable features like seed color, while product development studies demonstrate the practical viability of Talisay seeds for food applications (Abaya, 2022; Gonzales & Ramos, 2023; Pahayculay, 2024).

The combination of foreign methodological innovations and domestic empirical evidence provides a compelling justification for the present study's rationale since, while machine learning has already proved successful in the international context of oilseed prediction and research in the Philippines has already confirmed the very high oil content of locally sourced Talisay varieties, no research has so far attempted to combine these factors in the development of a predictive tool targeting Talisay fruits in the Philippines, thus providing an opportunity for research and an answer to the question of how to find high-yield seeds without destructive testing, thus facilitating better resource management in the country's agriculture sector.


---

\newpage

# CHAPTER 3
## RESEARCH METHODOLOGY

### RESEARCH DESIGN

The study utilized a developmental and experimental research method through the Agile software development model that was used to design, develop, and test a Machine Learning–Based Prediction of Seed-to-Oil Conversion Ratios in Talisay (*Terminalia catappa*) Fruits with Morphological Feature Analysis. The Agile model was selected because of its innovative and iterative approach, in which planning, development, testing, and improvement can be carried out as the system progresses through its lifecycle. The first step involved literature research and analysis of the problem, and subsequently, the development of the system requirements was done in short functional iterations known as sprints. Specific sprints were conducted in relation to dataset preparation, image processing, training of the Machine Learning model, morphological feature extraction, and user interface refinement. At each iteration, to recognize accuracy and reliability, constant testing and evaluation were made by using typical performance measures of Machine Learning such as the Mean Absolute Error (MAE) and Root Mean Square Error (RMSE). The feedback obtained after each cycle was used to improve the functionality of the system and the model performance, whereby the resulting system will eventually be an efficient instrument in data-driven predictions of seed-to-oil conversion ratio of Talisay fruits without any tests being destructive.


### RESEARCH INSTRUMENT

The primary research instrument used in this study is the Machine Learning–Based Prediction of Seed-to-Oil Conversion Ratios in Talisay (*Terminalia catappa*) Fruits using Morphological Feature Analysis system developed by the researchers. This system functions as a tool for collecting, processing, and analyzing Talisay fruit image data. It incorporates image processing techniques to extract visual features and a trained machine learning model to identify its maturity and measure oil yield. The system was also used to generate detailed results, such as accuracy and prediction reliability, which served as the basis for evaluating the effectiveness of the proposed solution. Supporting instruments include a labeled Talisay fruit image dataset obtained from reliable sources, used to train and evaluate the performance of the Machine Learning models.

The technical stack employed in building the system consists of the following components:

- **Machine Learning API:** A Python Flask server hosting the prediction pipeline, including YOLOv8 for object detection, a custom CNN guard model for fruit identity validation, a MobileNetV2-based color classifier, and an ensemble of Random Forest and Gradient Boosting regressors for oil yield prediction.
- **Backend Server:** A Node.js/Express server that handles user authentication, data management, and communication between the mobile application and the ML API.
- **Frontend Application:** A React Native (Expo) mobile application that provides the user interface for scanning Talisay fruits, viewing results, browsing history, and interacting with community features.
- **Database:** MongoDB, a document-based NoSQL database accessed via Mongoose, storing data across five collections: User, Prediction, History, ForumPost, and Notification.
- **Additional Services:** Google Gemini AI for chatbot functionality, Cloudinary for image storage, and Expo Push Notifications for user engagement.


### RESEARCH RESPONDENTS

Farmers, harvesters, and Talisay tree growers will also form a part of the sample since they are the most targeted users of the proposed system. These respondents have been chosen as they are the ones dealing with cultivation, collection, and processing of Talisay fruits, and are thus the ones who will gain most from an efficient and non-destructive means to identify high-yield seeds that produce oil. In order to obtain valid results of evaluation, the respondents must have basic mobile or computer literacy to be able to navigate through the system and evaluate its functionality as a decision-support tool. The effectiveness of the system in predicting the ratio of seed-to-oil conversion will be measured using their involvement, and it will provide the justification to apply the system practically in enhancing efficiency and minimizing waste in Talisay seed evaluation and agricultural resource management.


### DESIGN PROCEDURE

#### A. Program Flowchart

**Figure 2: Program Flowchart — Machine Learning–Based Prediction of Seed-to-Oil Conversion Ratios in Talisay (*Terminalia catappa*) Fruits using Morphological Feature Analysis**

The program flowchart identifies the progression logic and role-based navigation of the system whereby it begins with the starting point of the program which is the "Start" point, which proceeds to the "User Registration" and thereafter to the "Login" where one is authenticated. The system achieves success upon entry, providing a Check Access decision feature to differentiate between user roles, which essentially separates the workflow into two distinct interfaces. Upon recognition of the user as an Admin, the system allows the user access to administrative features including Manage Users and Manage and View Analytics to manage the system. Instead, when it comes to a regular User, the person is redirected to the main application functions where they can update their profile, scan Talisay fruits and measure oil yield, and see their scanning details. Both workflows end at the End node, which means the user is complete with the session and the process is complete.


#### B. System Flowchart

**Figure 3: System Flowchart — Machine Learning–Based Prediction of Seed-to-Oil Conversion Ratios in Talisay (*Terminalia catappa*) Fruits using Morphological Feature Analysis**

The system flowchart defines application logic and navigation flow in the system. The first module is User Registration, and the second one is the Login module where the user inputs credentials. A decision diamond with a label "Valid?" is used as a security checkpoint, in that, in case of false credentials (No), the user is rejected and sent to the login screen again to re-enter their information. When validation succeeds (Yes), the system moves on to the "Admin?" decision gate which identifies the role of the user and determines where they should be directed.

In case the user is an administrator (Yes), they are redirected to the Admin Dashboard. In this section, the workflow gives access to User Management where the admin can handle user accounts and View Talisay Analytics where the admin can see the system data and the prediction actions prior to terminating the session.

Conversely, when the user is not an admin (No), they are redirected to the User Dashboard. At this interface, the user is able to carry out the fundamental operation of the system by clicking on Scan Talisay Fruit, whereby a picture of the fruit is taken and examined using the Machine Learning model. The user is able to View Result Details like the predicted seed-to-oil ratio of the scanned fruit. Other features accessible by regular users are Access Result History, which allows them to look at prior scan results, and Manage Account Profile, where they can update their own information. Both the administrator and standard user paths arrive at the End node, indicating the successful completion of the system's functioning.


#### C. Context Flow Diagram

**Figure 4: Context Flow Diagram — Machine Learning–Based Prediction of Seed-to-Oil Conversion Ratios in Talisay (*Terminalia catappa*) Fruits using Morphological Feature Analysis**

The Context Diagram shows the general high-level communications of the central system with three key external entities: the Admin, the User, and the Database. The system is the key processing point where all the data interactions are done.

The Admin communicates with the system in order to perform User Management, which may be the control and management of the registered user accounts. The system, in turn, delivers generated analytics to the Admin, which might comprise user activity data and scan trend data according to the Talisay scans, to help with monitoring and decision-making.

The User interacts with the system through various data streams. The user is able to Register and Login to navigate the system, Edit Profile to provide personal details, as well as scan an image of a Talisay fruit to predict the yield of oil. The system reacts to the requests and provides various outputs to the user, such as Update Profile confirmations, Analyze Talisay results whereby the Machine Learning model processes the scanned fruit image, and Generate Details where the user is presented with the predicted seed-to-oil conversion ratio.

The Database represents the core functioning of the system since it is the center of data storage. The communication between the system and the database is two-way: the system provides information to the database to Manage Information, which discusses storing and updating the records like user accounts, scan history, and prediction results. The database reciprocates data to the system in the form of Access Information, so that the system will be able to access the required data to address queries by both the user and the admin.


#### D. Data Flow Diagram (DFD)

The Data Flow Diagram (DFD) is a graphical illustration of the flow of information in the Talisay Oil Yield Prediction system to show how the input and output information flow through the system. It is a representation of the flow of information between the external objects—namely the User and the Admin—and the system and its internal processes and data storage. Mapping these pathways makes the DFD clear on how critical system operations like user authentication, Talisay fruit image analysis, and data management are addressed in the system, giving a logical picture of the system in terms of its functional requirements and data dependencies.

##### Level 0

**Figure 5: Data Flow Diagram – Level 0 — Machine Learning–Based Prediction of Seed-to-Oil Conversion Ratios in Talisay Fruits using Morphological Feature Analysis**

Level 0 in the Data Flow Diagram (DFD) is the general view of the system and its boundaries, and shows the relationship between the system and the external entities. The node in the center representing the whole system application acts as the overall processing point of the information transactions of the whole system. This central process interacts with three central entities: the Admin, the User, and the Database.

The flow of information starts with the User who feeds the system with inputs by registering and logging in, editing profile, and the main activity which is the Scan Talisay. These inputs are then fed through the system and the relevant outputs are fed back to the User, consisting of Update Profile confirmations, Analyze Talisay results whereby the Machine Learning model analyzes the scanned image of the fruit, and Generate Details which display the estimated seed-to-oil conversion ratio of the scanned Talisay fruit.

Meanwhile, the Admin entity communicates with the system to detect and manage activities. The input of the admin is in the User Management that includes controlling the user access and the account records. In response, the system provides Generate Analytics to the Admin and can give insight regarding the usage of the system and trends of the Talisay scans.

The Database provides support of all these frontend interactions and acts as the backup data repository. The system sends data to the Database to Manage Information that includes the storage and update of records including user credentials, scan logs, and prediction results. The system uses Access Information to retrieve the information from the Database in order to meet the real-time demands of the Users and the Admins.

##### Level 1

**Figure 6: Data Flow Diagram – Level 1 — Machine Learning–Based Prediction of Seed-to-Oil Conversion Ratios in Talisay Fruits using Morphological Feature Analysis**

In this level, the User will interact with the system by providing the registration and login details to the User Authentication process. Once the User's details are authenticated by the system by comparing them with the User Data stored in the database, the User can request a scan of the fruit by the Scan and Image Processing sub-process. When the image of the Talisay fruit is scanned, the morphological features of the image—such as the size, shape, and color of the fruit—will be gathered and saved by the system in the Scan Records. The morphological features of the image of the fruit then will be inputted to the Machine Learning Prediction. Once the prediction is made by the system, the prediction result will be displayed to the User while the result will also be stored in the ML Analysis Results data store. The User information can be accessed by the Admin through the User and Data Management process by accessing the User Data store.

##### Level 2

**Figure 7: Data Flow Diagram – Level 2 — Machine Learning–Based Prediction of Seed-to-Oil Conversion Ratios in Talisay Fruits using Morphological Feature Analysis**

When the morphological features are passed on from the Scan and Image Processing module, the features are passed on to the Feature Normalization sub-process, which normalizes the features by scaling the numerical data (length, width, and mass). Once the normalization of the features has taken place, the features are passed on to the Model Inference Engine sub-process, which uses the machine learning algorithms (an ensemble of Random Forest and Gradient Boosting) to predict the oil yield by applying the algorithms to the features and arriving at a predicted value. Once the predicted value has been arrived at, the predicted value is passed on to the Post-Processing and Validation sub-process, which validates the predicted value within the scientifically established range of 45%–65% and computes additional information like the confidence intervals of the predicted value, based on the performance of the model. Once the predicted value and the confidence intervals have been arrived at, they are passed on to the user interface and stored in the ML Analysis Results data store.


#### E. Entity Relationship Diagram

**Figure 8: Entity Relationship Diagram — Machine Learning–Based Prediction of Seed-to-Oil Conversion Ratios in Talisay Fruits using Morphological Feature Analysis**

The system employs a document-based NoSQL database structure using MongoDB (accessed via the Mongoose ODM library) to effectively store and manage all information related to Talisay fruit scanning and oil yield prediction. The database is organized into five collections, each serving a particular purpose in the overall data architecture.

**1. User Collection**

The central point of user management in the system is the User collection. It contains fundamental authentication details such as email (unique identifier), passwordHash for secure login, and role to differentiate between "user" and "admin" accounts. It also stores personal profile information including firstName, lastName, phone, address, and avatar (profile picture URL). Additional fields include isVerified for email verification status, verifyOtp for one-time password verification, pushTokens for push notification targeting, notificationSettings for user preferences, isActive for account status, and deactivationReason. Each User document is timestamped with createdAt and updatedAt fields. The User collection is referenced by the History, Prediction, ForumPost, and Notification collections.

**2. Prediction Collection**

The Prediction collection stores the results of each Machine Learning analysis. Each document contains a reference to the userId (linking to the User collection), the analysisType (single, comparison, or multi_fruit), imageUri (the Cloudinary URL of the uploaded fruit image), and the full prediction output including category (Green, Yellow, or Brown), confidence scores (a probability distribution across all three color categories), oil yield percentage, and dimensional measurements such as length_cm, width_cm, kernel_mass_g, and whole_fruit_weight_g. This collection serves as the primary record of individual scan predictions.

**3. History Collection**

The History collection maintains a longitudinal record of all user scanning activities. Each document references the userId and stores the analysisType, with support for multi-fruit analysis including fruitCount, colorDistribution (breakdown of how many fruits of each color were detected), and averageOilYield. It also stores the imageUri, category, and confidence scores. This collection enables users to view their scan history and track trends over time.

**4. ForumPost Collection**

The ForumPost collection supports the community forum feature. Each document contains the author (referencing the User collection), title, content, and optionally images (stored as Cloudinary URLs). Community interaction is supported through a likes array (storing user IDs who liked the post) and a comments sub-document array, where each comment contains a user reference, text content, and timestamp. This embedded sub-document design eliminates the need for a separate Comment collection, leveraging MongoDB's document model for efficient retrieval of posts with their comments.

**5. Notification Collection**

The Notification collection manages system-generated notifications for users. Each document references the recipient user and contains the notification type, title, message, and read status. This collection supports the push notification feature and the in-app notification center, keeping users informed about system updates, forum activity, and scan result summaries.

The document-based NoSQL design ensures flexible schema evolution, efficient retrieval of nested data (such as comments within forum posts), and horizontal scalability. By embedding related data within documents where appropriate (e.g., comments within ForumPost, profile within User), the system minimizes the need for complex join operations while maintaining data integrity through Mongoose's referencing and validation mechanisms.


#### F. Use Case Diagram

**Figure 9: Use Case Diagram — Machine Learning–Based Prediction of Seed-to-Oil Conversion Ratios in Talisay (*Terminalia catappa*) Fruits using Morphological Feature Analysis**

The Use Case Diagram describes the functional communication between the system and its two key actors—the User and the Admin—in terms of what functions can be used by each party. The User engages with the system to carry out the necessary functionality of the system. Registration commences whereby a new user creates an account, after which they then log in to access the system in an authenticated manner. After logging in, the User is able to Update Profile to manage and maintain personal information, and Explore Access History to see historical scan records. The core functionality that the User can utilize is Scan Talisay, which is directly linked to View Result Details so that after the image of the Talisay fruit has been scanned and processed through the Machine Learning model, the platform will automatically create and display the projected seed-to-oil conversion percentage to the User. Additionally, the User can access the AI Chatbot for guidance on Talisay-related queries and participate in the Community Forum to share findings and discuss results with other users.

The Admin, in their turn, has another set of administrative privileges that revolve around system monitoring rather than operation. The Admin, just like the regular User, must go through Login in order to get into the platform. The functions of the Admin are management-based: they have the ability to Manage Users to view, control, and manage registered user accounts, and View/Generate Analytics to trace system usage, scan data patterns, and overall system behavior.

This makes it clear that there are roles separated in the system. The ordinary user has concentration on scans and results access, whereas system management and monitoring are with the system administrator. Although the two actors have varied tasks, their similarity lies in the fact that they both need to engage in a secure mechanism of authentication via the Login process prior to accessing their various functional capabilities.


### IPO DIAGRAM

**Figure 1: IPO Diagram — Machine Learning–Based Prediction of Seed-to-Oil Conversion Ratios in Talisay (*Terminalia catappa*) Fruits using Morphological Feature Analysis**

The first column is the Input column that contains the resources and constraints needed to start the project. This section identifies the target users, which are Researchers, Agricultural Scientists, Farmers, and Harvesters, with the requirement that they should have basic mobile or computer literacy. It also lists the technical requirements needed to develop and operate the system. The software components include an Integrated Development Environment (IDE), Python programming language, Node.js runtime, React Native (Expo) framework, and MongoDB as the database, all of which are compatible with major operating systems such as Windows 10/11 (64-bit), Linux OS (64-bit), and Mac OS. The hardware requirements reflect the demands of a machine learning-based system, specifying a Quad-core modern CPU, 16GB RAM, and 256GB NVMe SSD storage to ensure the system can handle image processing and model inference tasks efficiently.

The Process column is at the center and visualizes the Software Development Life Cycle (SDLC) through a flowchart. It begins with the conceptualization stage where a proposed title and project document are prepared. A decision diamond labeled "Is Approved?" serves as a quality gate; if not approved, the team revises the document before proceeding. The workflow then moves to the data gathering and analysis phase, which is critical for a machine learning project that depends on accurate and sufficient datasets for model training. This is followed by System Development, which covers designing, coding, and bug fixing. Another approval checkpoint ensures the system meets requirements before advancing to System Testing. A "Bug Free?" decision gate determines whether the system is ready. If bugs are found, it returns to development. The process concludes with Implementation and Deployment only after the system has passed all testing stages.

The Output column represents the final result of the described inputs and processes. The outcome is a complete Machine Learning–Based Prediction of Seed-to-Oil Conversion Ratios in Talisay (*Terminalia catappa*) Fruits using Morphological Feature Analysis system. This signifies that by combining the specified hardware, software, and user requirements with a structured development and testing workflow, the project produces a functional software tool capable of analyzing the external morphological features of Talisay fruits and predicting their seed-to-oil conversion ratios without any destructive testing.


### DEFINITION OF TERMS

**Artificial Neural Network (ANN)** – A machine learning model that emulates the structure of the human brain, comprising interconnected nodes that operate to predict the outcome of input data. This model was adopted in related studies to predict oil yield from *Terminalia catappa* kernel oil.

**Color Metrics** – Quantitative measurements of the color of the fruit obtained from images taken using a digital camera, often expressed as RGB or HSV color models, used as features in the machine learning model. In this system, a MobileNetV2-based CNN with an HSV-based fallback is used to classify fruits into Green, Yellow, or Brown categories.

**Computer Vision** – The field of artificial intelligence that enables computers to interpret and understand visual information from digital images, used in this study to extract morphological features from Talisay fruit scans via YOLOv8 object detection and image segmentation.

**Confidence Interval** – A statistical range that indicates the reliability of the machine learning model's prediction, stored in the Prediction collection to show the accuracy of each seed-to-oil ratio estimate.

**Dataset** – The collection of Talisay fruit samples with corresponding morphological measurements (length, width, mass, volume, shape, color) and actual oil yield values obtained through laboratory extraction, used for training and validating the machine learning models.

**Destructive Testing** – The traditional method of determining oil content that requires crushing or chemically processing the seed, making it impossible to use the seed for other purposes after analysis.

**Feature Extraction** – The process of identifying and quantifying relevant morphological characteristics from Talisay fruit images, such as size, shape, and color, to serve as inputs for the machine learning model.

**Gradient Boosting** – An ensemble machine learning method that builds multiple decision trees sequentially, where each subsequent tree focuses on correcting the errors of the previous ones. In this system, Gradient Boosting is used alongside Random Forest as part of the ensemble prediction model for oil yield estimation.

**Image Processing** – The set of techniques used to enhance, analyze, and extract information from digital images of Talisay fruits, forming the foundation for morphological feature extraction.

**Machine Learning (ML)** – A branch of artificial intelligence that enables systems to learn patterns from data and make predictions without being explicitly programmed, used in this study to predict seed-to-oil conversion ratios from morphological features.

**Model Training** – The process of feeding labeled data (morphological features paired with actual oil yields) to a machine learning algorithm so it can learn the relationships between inputs and outputs.

**MongoDB** – A document-based NoSQL database management system used in this study to store user data, scan results, prediction records, forum posts, and notifications in a flexible, schema-less format accessed via the Mongoose Object Data Modeling (ODM) library.

**Morphological Features** – The measurable physical characteristics of Talisay fruits, including length, width, mass, volume, shape, and color, used as input variables for the machine learning model.

**Multi-Source Data Fusion** – The technique of combining data from multiple sensors or sources (such as RGB and multispectral imaging) to improve prediction accuracy, as demonstrated in foreign studies on oilseed rape yield prediction.

**Non-Destructive Prediction** – The approach of estimating oil content without physically damaging or destroying the Talisay fruit, enabling the seed to be used for planting or other purposes after assessment.

**Oil Yield** – The percentage of oil extracted from Talisay seeds, ranging from 45% to 65% depending on fruit maturity stage—with green (immature) fruits averaging approximately 47%, yellow (mature) fruits averaging approximately 58.5%, and brown (fully ripe) fruits averaging approximately 55.5%. Philippine studies have reported values between 48.8% and 58.3% across multiple extraction methods.

**Random Forest** – An ensemble machine learning algorithm that constructs multiple decision trees during training and outputs the average prediction of the individual trees. In this system, Random Forest is one of two ensemble models (alongside Gradient Boosting) used for oil yield prediction.

**Talisay (*Terminalia catappa*)** – A tropical tree species commonly found in the Philippines, whose seeds contain high-value oil with potential applications in food, medicine, and biofuel production.

**YOLOv8 (You Only Look Once, version 8)** – A real-time object detection algorithm used in this system to identify Talisay fruits and a reference coin (Philippine ₱5 coin) in captured images, enabling dimension estimation by computing the pixels-per-centimeter ratio from the known coin diameter.


---

\newpage

