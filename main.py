import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import re   
from flask import Flask, request, jsonify
# import pandas as pd
import PyPDF2
import io
# import os
import requests
# from urllib.parse import urlparse
from multiprocessing import Pool, cpu_count

app = Flask(__name__)

def extract_text_from_pdf(pdf_bytes):
    # Create PDF reader object from bytes
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))

    # Extract text from all pages
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    return text

def preprocess_text(text):
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())
    return text.lower()

def get_embeddings(text, model_name='sentence-transformers/all-mpnet-base-v2'):
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Split text into chunks of max_length tokens
    chunks = [text[i:i+512] for i in range(0, len(text), 512)]
    all_embeddings = []

    for chunk in chunks:
        # Tokenize and get embeddings for each chunk
        inputs = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            outputs = model(**inputs)
            chunk_embedding = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(chunk_embedding)

    # Average embeddings across all chunks
    if all_embeddings:
        final_embedding = torch.mean(torch.cat(all_embeddings, dim=0), dim=0, keepdim=True)
        return final_embedding
    return None

def calculate_similarity(resume_text, job_description):
    # Preprocess texts
    resume_text = preprocess_text(resume_text)
    job_description = preprocess_text(job_description)

    # Extract key skills from job description
    skills_pattern = r'skills.*?:(.+?)(?=education|experience|preferred|how to apply|$)'
    skills_match = re.search(skills_pattern, job_description, re.IGNORECASE | re.DOTALL)
    skills_text = skills_match.group(1) if skills_match else job_description

    # Get embeddings
    resume_embedding = get_embeddings(resume_text)
    skills_embedding = get_embeddings(skills_text)
    full_jd_embedding = get_embeddings(job_description)

    if resume_embedding is not None and skills_embedding is not None and full_jd_embedding is not None:
        # Calculate similarities
        skills_similarity = cosine_similarity(
            resume_embedding.numpy(),
            skills_embedding.numpy()
        )[0][0]

        full_similarity = cosine_similarity(
            resume_embedding.numpy(),
            full_jd_embedding.numpy()
        )[0][0]

        # Weight skills similarity more heavily
        final_score = (skills_similarity * 0.7) + (full_similarity * 0.3)

        return {
            'similarity_score': float(final_score)
        }
    return {
        'similarity_score': 0.0
    }

def download_file(url):
    try:
        # Check if it's a Google Drive URL
        if 'drive.google.com' in url:
            # Extract file ID from Google Drive URL
            file_id = url.split('/d/')[1].split('/view')[0]
            # Convert to direct download link
            url = f'https://drive.google.com/uc?export=download&id={file_id}'

        response = requests.get(url)
        if response.status_code == 200:
            # Get filename from URL or Content-Disposition header
            if 'Content-Disposition' in response.headers:
                filename = re.findall("filename=(.+)", response.headers['Content-Disposition'])[0]
            else:
                filename = url.split('/')[-1]
                if '?' in filename:  # Remove query parameters
                    filename = filename.split('?')[0]
            return response.content, filename
        return None, None
    except Exception as e:
        print(f'Error downloading file: {e}')
        return None, None

def process_single_resume(args):
    url, job_description = args
    # Download and extract text from PDF resume
    pdf_bytes, filename = download_file(url)
    if pdf_bytes:
        resume_text = extract_text_from_pdf(pdf_bytes)

        # Calculate similarity
        similarity = calculate_similarity(resume_text, job_description)
        score = similarity['similarity_score']

        # Determine status based on score with more granular thresholds
        if score > 0.7:
            status = 'highly_recommended'
        elif score > 0.6:
            status = 'shortlisted'
        else:
            status = 'not_recommended'

        return {
            'Resume': filename,
            'Similarity Score': score,
            'Status': status,
            'URL': url
        }
    return None

def process_resumes(resume_urls, job_description):
    # Create a pool of workers
    num_workers = min(cpu_count(), len(resume_urls))  # Don't create more workers than URLs
    pool = Pool(processes=num_workers)

    # Prepare arguments for parallel processing
    args = [(url, job_description) for url in resume_urls]

    # Process resumes in parallel
    results = pool.map(process_single_resume, args)

    # Close the pool
    pool.close()
    pool.join()

    # Filter out None results and return
    return [result for result in results if result is not None]

@app.route('/match-resumes', methods=['POST'])
def match_resumes():
    """
    Match resumes from URLs (including Google Drive links) with a job description.

    Example usage:
    ```python
    import requests
    import json

    url = "http://localhost:5000/match-resumes"

    payload = {
        "resume_urls": [
            "https://drive.google.com/file/d/1Llv77xZuwlwQe5AnC2j7WbIqfbpqgwEX/view?usp=drive_link",
            "http://example.com/resume2.pdf"
        ],
        "job_description": "We are seeking a Full Stack Developer..."
    }

    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.post(url, headers=headers, json=payload)
    results = response.json()

    # Example response:
    # {
    #     "results": [
    #         {
    #             "Resume": "resume1.pdf",
    #             "Similarity Score": 0.85,
    #             "Status": "highly_recommended",
    #             "URL": "https://drive.google.com/file/d/1Llv77xZuwlwQe5AnC2j7WbIqfbpqgwEX/view?usp=drive_link"
    #         },
    #         {
    #             "Resume": "resume2.pdf",
    #             "Similarity Score": 0.65,
    #             "Status": "shortlisted", 
    #             "URL": "http://example.com/resume2.pdf"
    #         }
    #     ]
    # }
    ```
    """
    try:
        data = request.get_json()

        if not data or 'resume_urls' not in data or 'job_description' not in data:
            return jsonify({'error': 'Missing required parameters'}), 400

        resume_urls = data['resume_urls']
        job_description = data['job_description']

        # Process resumes and get results
        results = process_resumes(resume_urls, job_description)

        # Sort results by similarity score
        results = sorted(results, key=lambda x: x['Similarity Score'], reverse=True)

        return jsonify({
            'results': results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
