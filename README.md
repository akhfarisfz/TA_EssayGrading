If you'd like to run the Flask app using `python app.py`, here’s how you can adjust the README for that method.

---

# TA\_EssayGrading

This project provides a web-based essay grading system that leverages natural language processing (NLP) and machine learning models to assess student responses to essay questions.

## Requirements

To set up the environment, install the following dependencies:

```bash
Flask==2.0.1
pandas==1.3.3
numpy==1.21.2
scikit-learn==0.24.2
transformers==4.9.2
torch>=1.10.0
nlp-id>=0.6.0
gunicorn
```

## Installation

1. Clone this repository:

```bash
git clone https://github.com/your-username/TA_EssayGrading.git
cd TA_EssayGrading
```

2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate the virtual environment:

   * On Windows:

     ```bash
     venv\Scripts\activate
     ```
   * On Mac/Linux:

     ```bash
     source venv/bin/activate
     ```

4. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Prepare the Template

* Download the provided essay template file.
* The template will have fields for student answers to one specific essay question.

### Step 2: Fill in Student Answer

* Open the template and input the student's essay answer into the corresponding field.

### Step 3: Run the Application

1. Run the Flask app locally:

```bash
python app.py
```

2. Visit `http://127.0.0.1:5000` in your web browser.

3. On the web page, enter the correct answer key for the essay question.

4. Upload the student’s essay answer in the required format.

5. The system will grade the essay based on the provided answer key.

### Step 4: Deploy (Optional)

To deploy the app in production using Gunicorn:

```bash
gunicorn app:app
```

This will start the application in production mode.

---

With this setup, the app will be run via `python app.py` instead of `flask run`. Let me know if this works or if you need any more adjustments!
