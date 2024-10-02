from flask import Flask, request, render_template, redirect, session
from flask_sqlalchemy import SQLAlchemy
import bcrypt
from flask import Flask, render_template, request, redirect, url_for, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__,static_folder='static')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.secret_key = 'secret_key'






history = []
generated = ["Hello! Ask me anything about 🤗"]
past = ["Hey! 👋"]

def get_pdf_text_buddy(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks_buddy(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store_buddy(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("spunky-2\\faiss_index")

def get_conversational_chain_buddy():
    prompt_template = """
    consider context as syllabus and answer question asked on given syllabus, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input_chain_buddy(user_question, history):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("spunky-2\\faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)

    chain = get_conversational_chain_buddy()

    response = chain.invoke(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    history.append((user_question, response["output_text"]))
    return response["output_text"]

@app.route('/upload_files_buddy', methods=['POST'])
def upload_files_buddy():
    if request.method == 'POST':
        uploaded_files = request.files.getlist("pdfs")
        if uploaded_files:
            raw_text = get_pdf_text_buddy(uploaded_files)
            text_chunks = get_text_chunks_buddy(raw_text)
            get_vector_store_buddy(text_chunks)
    return 'Done'


@app.route('/ask_question_buddy', methods=['POST'])
def ask_question_buddy():
    user_question = request.form.get('question')
    if user_question:
        response = user_input_chain_buddy(user_question, history)
        past.append(user_question)
        generated.append(response)
    return redirect(url_for('index_buddy'))

@app.route('/clear_history_buddy', methods=['POST'])
def clear_history():
    global history, generated, past
    history.clear()
    generated = ["Hello! Ask me anything about 🤗"]
    past = ["Hey! 👋"]
    return redirect(url_for('index_buddy'))







def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("spunky-1\\faiss_index")

def get_conversational_chain():
    prompt_template = """
    consider context as syllabus and generate questions and subtopic based on inputs, provide it in python dictionary form , if pdf not provided just say"PDF unavailable", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    inputs: 
    {question}
    
    output should be python dictionary containing question no, question, mark, subtopic for each section
    

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input_chain(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("spunky-1\\faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain.invoke(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    
    return response["output_text"]

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        uploaded_files = request.files.getlist("pdfs")
        if uploaded_files:
            raw_text = get_pdf_text(uploaded_files)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
    return 'Done'

@app.route('/result', methods=['GET','POST'])
def result():
    user_question = ""
    total_sections = 0
    
    if request.method == 'POST':
        total_sections = int(request.form.get('total_sections', 0))
        sections = []
        
        for i in range(total_sections):
            section_type = request.form.get(f'section_type_{i}')
            section_questions = request.form.get(f'section_questions_{i}', '0')
            section_marks = request.form.get(f'section_marks_{i}', '0')
            sections.append({
                'type': section_type,
                'questions': section_questions,
                'marks': section_marks
            })

        user_question = ""
        for i, section in enumerate(sections):
            user_question += f"""
            Section no: {i + 1};
            Question Type: {section['type']};
            Number of Questions: {section['questions']};
            Marks per Question: {section['marks']};
            """
    
    if user_question:
        response = user_input_chain(user_question)
    try:
        result_dict = json.loads(response)  # Parse the JSON string into a dictionary
    except json.JSONDecodeError:
        result_dict = {}

    return render_template('result.html', result=result_dict)  # Pass the parsed result to the template








# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    university = db.Column(db.String(100), nullable=False)  # New field
    branch = db.Column(db.String(50), nullable=False)       # New field
    year = db.Column(db.String(10), nullable=False)         # New field

    def __init__(self, email, password, name, university, branch, year):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        self.university = university
        self.branch = branch
        self.year = year

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))

# Create the database tables
with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Welcome to Exagram</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.2/dist/css/bootstrap.min.css">
        <style>
            /* Full-width GIF background */
            .gif-background {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                z-index: -1; /* Set behind other content */
                object-fit: cover; /* Cover entire area */
            }

            /* Center button over the GIF */
            .content {
                position: relative;
                z-index: 1; /* Bring content in front */
                text-align: center;
                color: white; /* White text for contrast */
                margin-top: 50%; /* Center the button vertically */
                transform: translateY(-50%); /* Adjust for perfect centering */
            }

            /* Button Styles */
            .get-started-btn {
                background-color: rgba(0, 123, 255, 0.8); /* Slightly transparent blue */
                border: none;
                padding: 15px 30px; /* Increased padding for a larger button */
                font-size: 18px; /* Larger font size */
                color: #E6E6FA;
            }
        </style>
    </head>
    <body>
        <!-- GIF Background -->
        <img src="https://wallpapers.com/images/featured/study-pictures-6ovzato85a4dtczb.jpg" alt="EXAGRAM Background" class="gif-background">

        <!-- Content Overlay -->
        <div class="container content">
            <a href="/register" class="btn get-started-btn">Get Started</a>
        </div>
    </body>
    </html>
    '''

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Get form data
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        university = request.form['university']
        branch = request.form['branch']
        year = request.form['year']

        # Check if the email already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return render_template('register.html', error="Email already registered. Please use a different email.")

        # If email is not registered, create a new user
        new_user = User(name=name, email=email, password=password, university=university, branch=branch, year=year)
        db.session.add(new_user)
        db.session.commit()

        return redirect('/login')

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Get form data
        email = request.form['email']
        password = request.form['password']

        # Find user by email
        user = User.query.filter_by(email=email).first()

        # Check if user exists and password is correct
        if user and user.check_password(password):
            session['name'] = user.name
            session['email'] = user.email
            session['university'] = user.university  # Store university in session
            session['branch'] = user.branch          # Store branch in session
            session['year'] = user.year              # Store year in session
            return redirect('/dashboard')
        else:
            return render_template('login.html', error='Invalid email or password')

    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    # Check if user is logged in
    if 'name' in session:
        return render_template('dashboard.html', 
                               name=session['name'],
                               university=session['university'],
                               branch=session['branch'],
                               year=session['year'])
    return redirect('/login')

@app.route('/logout')
def logout():
    session.clear()  # Clear session data
    return redirect('/login')  # Redirect to the login page after logout


@app.route('/index_buddy')
def index_buddy():
    session.clear()  # Clear session data
    return render_template('index_buddy.html', generated=generated, past=past, zip=zip)


@app.route('/index2')
def index2():
    return render_template('index2.html')

if __name__ == '__main__':
    app.run(debug=True)
