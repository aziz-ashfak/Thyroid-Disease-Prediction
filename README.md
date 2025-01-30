# Thyroid-Disease-Prediction

# Project Overview
## Directory Structure
```
├── .github
   ├── workflows
       ├── main.yml
├── Dataset
   ├── Clean_Dataset.csv 
├── HyperparameterTuning
    ├── __init__.py
    ├── parameter_trainner.py        # create hyperparameter tuning pipeline  
    ├── trainning                    # train models
    ├── tuningMethod.py              # hyperparameter runner
├── Result 
    ├── Analysis.ipynb              # full analysis
    ├── Details.csv                 # accuracy details
    ├── matrix.npy                  # cofusion matrix
    ├── report_details.csv          # classfication report
    ├── test.py                     # test single prediction 
├── TuningArtifacts
    ├── model_accuracy.json        # after tuning best accuracy
    ├── tuning_model.pkl           # store best tuning model
├── kaggleFullNoteBook 
    ├── thyroid-disease-prediction-Full-NoteBook.ipynb  # my kaggle notebook
├── artifacts 
    ├── model.pkl      # save the model
    ├──preprocessor.pkl  # save preprocessor
    ├──raw_data.csv    # main dataset
    ├──test_data.csv   # 20% of main dataset
    ├──train_data.csv  # 80% of main dataset       
├── src  
    ├── component
      ├── __init__.py
      ├── data_ingestion.py                    # split data,transform and train 
      ├── data_transformation.py               # transform all columns
      ├── model_trainner.py                    # model trainner class
   ├── pipeline
      ├── _init__.py
      ├── predict_pipeline.py                   # prediction function
 ├── _init__.py
 ├── exception.py                               # custom exeption
 ├── logger.py                                  # logging info store method
 ├── utilts.py                                  # save all common function
   
├── static
    ├── styles.css                              # style.css file include     
├── templates
    ├── home.html                               # html file include           
├── .gitignore             
├── LICENSE                                    
├── README.md                                   # reame file
├── app.py                                      # application 
├── requirements.txt                            # requirements file
└── setup.py                                    # setup 
```

## Dataset link 

```bash
https://www.kaggle.com/datasets/jainaru/thyroid-disease-data
```
# Describe the project
## Dataset deatils:
### About Dataset
This data set contains 13 clinicopathologic features aiming to predict recurrence of well differentiated thyroid cancer. The data set was collected in duration of 15 years and each patient was followed for at least 10 years.

### Source
The data was procured from thyroid disease datasets provided by the UCI Machine Learning Repository.

### Content
The size for the file featured within this Kaggle dataset is shown below — along with a list of attributes, and their description summaries:

**Age**: The age of the patient at the time of diagnosis or treatment.

**Gender**: The gender of the patient (male or female).

**Smoking**: Whether the patient is a smoker or not.

**Hx Smoking**: Smoking history of the patient (e.g., whether they have ever smoked).

**Hx Radiotherapy**: History of radiotherapy treatment for any condition.

**Thyroid Function**: The status of thyroid function, possibly indicating if there are any abnormalities.

**Physical Examination**: Findings from a physical examination of the patient, which may include palpation of the thyroid gland and surrounding structures.

**Adenopathy**: Presence or absence of enlarged lymph nodes (adenopathy) in the neck region.

**Pathology**: Specific types of thyroid cancer as determined by pathology examination of biopsy samples.

**Focality**: Whether the cancer is unifocal (limited to one location) or multifocal (present in multiple locations).

**Risk**: The risk category of the cancer based on various factors, such as tumor size, extent of spread, and histological type.

**T**: Tumor classification based on its size and extent of invasion into nearby structures.

**N**: Nodal classification indicating the involvement of lymph nodes.

**M**: Metastasis classification indicating the presence or absence of distant metastases.

**Stage**: The overall stage of the cancer, typically determined by combining T, N, and M classifications.

**Response**: Response to treatment, indicating whether the cancer responded positively, negatively, or remained stable after treatment.

**Recurred**: Indicates whether the cancer has recurred after initial treatment.

## Describe the preprocessesor 
In this project we use 80% data as trainset and  20% data as testset.In this project we use  different preprocessing method:
      **MinMaxScaler**: use for numeric columns
      **OneHotEncoder**: use for categorical columns
      **SimpleImputer**: use to detect nan value both for numeric and categorical cols
## Models we use 
    LogisticRegression
    SGDClassifier
    RandomForestClassifier
    AdaBoostClassifier
    GradientBoostingClassifier
    BaggingClassifier
    CatBoostClassifier

In our analysis our best model is RandomForestClassifier with accuracy 98.70% before tuning parameter. After tuning our best model accuracy is 98.70%.So we simply choose first one.

## kaggle notebook link
```bash 
https://www.kaggle.com/code/azizashfak/thyroid-disease-prediction-accuracy-98-7
```

## Installation
1. Clone the repository:
   ```bash
   git clone <https://github.com/AzizAshfak/Thyroid-Disease-Prediction.git>
   ```
2. Navigate to the project directory:
   ```bash
   cd <Thyroid-Disease-Prediction>
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
4. Usage
Run the main application script:
```bash
python app.py
   ```
## Github action pipeline 
``` bash 
  ├── .github
   ├── workflows
       ├── main.yml
```
## use cloud servce 
```bash 
link : www.render.com
```
## Use Thyroid-Disease-Prediction web-service link

```bash
 https://thyroid-disease-prediction-yker.onrender.com/
```
## Contributing
If you would like to contribute, please fork the repository and submit a pull request.

## License
This project is licensed under the [LICENSE] file included in the repository.

## Author

👤 **Aziz Ashfak**  
📧 Email: [azizashfak@gmail.com](mailto:azizashfak@gmail.com)  
🔗 LinkedIn: [linkedin.com/in/aziz-ashfak](https://www.linkedin.com/in/aziz-ashfak-27353b262/)  
🐙 GitHub: [github.com/AzizAshfak](https://github.com/AzizAshfak/)  
