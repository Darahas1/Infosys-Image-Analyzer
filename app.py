from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from PIL import Image
import openai
from transformers import BlipProcessor, BlipForConditionalGeneration
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
from werkzeug.utils import secure_filename
import imghdr
from gtts import gTTS
import tempfile

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image(stream):
    header = stream.read(512)
    stream.seek(0)
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')

# Configure OpenAI
openai.api_key = os.environ.get('OPENAI_API_KEY')

# Initialize BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_alt_text(image):
    """Generates alt text for an image using BLIP model."""
    try:
        # Preprocess the image for BLIP model
        inputs = processor(images=image, return_tensors="pt")
        
        # Generate alt text
        out = model.generate(**inputs)
        
        # Decode the generated caption
        alt_text = processor.decode(out[0], skip_special_tokens=True)
        
        return alt_text
    except Exception as e:
        return f"Error generating alt text: {e}"

def generate_context(alt_text):
    """Generates context from alt text using OpenAI."""
    prompt = f"Generate a brief context (maximum 70 words) for this image description:\n\n{alt_text}"
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise context for images. Keep responses under 50 words."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.7
        )
        context = response.choices[0].message.content.strip()
        # Truncate to roughly 70 words if needed
        words = context.split()
        if len(words) > 70:
            context = ' '.join(words[:70]) + '...'
        return context
    except Exception as e:
        return f"Error generating context: {e}"

def enhance_context(context):
    """Enhances the alt text using OpenAI 4 model."""
    prompt = f"Enhance this context of a image into a brief, creative caption (maximum 50 words):\n\n{context}"
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Keep responses under 30 words."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=75,
            temperature=0.7
        )
        enhanced = response.choices[0].message.content.strip()
        # Truncate to roughly 50 words if needed
        words = enhanced.split()
        if len(words) > 50:
            enhanced = ' '.join(words[:50]) + '...'
        return enhanced
    except Exception as e:
        return f"Error generating enhanced caption: {e}"
    
def social_media_caption(context):
    """Enhances the generated context into a social media caption using OpenAI 4 model."""
    prompt = f"Enhance this context of a image into a brief, creative caption along with related hashtags for my social media platform:\n\n{context}"
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a great social media manager. Keep responses accurate and relevant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=75,
            temperature=0.7
        )
        caption = response.choices[0].message.content.strip()
        return caption
    except Exception as e:
        return f"Error generating Social Media caption: {e}"

def analyze_sentiment(text):
    """Analyzes the sentiment of the text using VADER Sentiment Analyzer."""
    try:
        # Initialize VADER
        sid = SentimentIntensityAnalyzer()
        
        # Get sentiment scores
        scores = sid.polarity_scores(text)
        compound_score = scores['compound']
        
        # Convert compound score (-1 to 1) to percentage (0 to 100)
        sentiment_percent = (compound_score + 1) * 50
        
        # Determine sentiment category based on compound score
        if compound_score >= 0.05:
            category = "Positive"
        elif compound_score <= -0.05:
            category = "Negative"
        else:
            category = "Neutral"
            
        return {
            'category': category,
            'score': sentiment_percent,
            'raw_score': compound_score,
            'detailed_scores': scores
        }
    except Exception as e:
        return f"Error analyzing sentiment: {e}"

def generate_seo_description(context, alt_text):
    """
    Generates a detailed product description and SEO title with improved formatting.
    
    Args:
        context (str): Context of the image
        alt_text (str): Generated alt text of the image
        
    Returns:
        dict: Contains formatted description and SEO title
    """
    description_prompt = f"""Based on this image context and alt text, generate a structured product description:

Context: {context}
Alt Text: {alt_text}

Please provide the description in the following exact format:

About this item:
• [First key feature]
• [Second key feature]
• [Third key feature]

Technical Specifications:
• [First specification]
• [Second specification]
• [Third specification]

Additional Features:
• [First additional feature]
• [Second additional feature]
• [Third additional feature]

Requirements:
- Use bullet points with the • character (not hyphens or asterisks)
- Each bullet point should be a complete, informative sentence
- Include specific technical details and measurements where applicable
- Maintain consistent grammatical structure across bullet points
- Ensure each section has at least 3 bullet points
- Total description should be minimum 80 words
"""

    try:
        # Generate detailed description
        description_response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """You are a technical product description writer. Create detailed, accurate product descriptions with the following rules:
                    - Always use bullet points with the • character
                    - Maintain consistent formatting
                    - Each bullet point must be a complete sentence
                    - Focus on technical specifications and measurable features
                    - Use parallel structure in bullet points
                    - Separate content into distinct sections as specified in the prompt"""
                },
                {"role": "user", "content": description_prompt}
            ],
            max_tokens=400,
            temperature=0.7
        )
        detailed_description = description_response.choices[0].message.content.strip()

        # Generate SEO title with improved prompt
        title_prompt = f"""Based on this detailed product description, create an SEO-optimized product title:

Description:
{detailed_description}

Follow these exact requirements:
1. Maximum 150 characters
2. Format: [Brand] + [Model] + [Key Specs] + [Product Type]
3. Include following elements if applicable:
   - Size/Dimensions
   - Key Technical Specification
   - Color/Material
   - Unique Selling Point
4. Use proper capitalization (capitalize important words)
5. Include numbers and measurements where relevant"""

        title_response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are an SEO expert specializing in e-commerce product titles. Create compelling, keyword-rich titles that follow SEO best practices and exact formatting requirements."
                },
                {"role": "user", "content": title_prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        seo_title = title_response.choices[0].message.content.strip()

        # Post-process the description to ensure proper formatting
        sections = detailed_description.split('\n\n')
        formatted_sections = []
        
        for section in sections:
            if ':' in section:
                title, content = section.split(':', 1)
                # Ensure bullet points are properly formatted
                bullet_points = [point.strip() for point in content.split('\n') if point.strip()]
                formatted_points = [f"• {point.lstrip('•').strip()}" for point in bullet_points]
                formatted_section = f"{title}:\n" + '\n'.join(formatted_points)
                formatted_sections.append(formatted_section)

        formatted_description = '\n\n'.join(formatted_sections)

        return {
            'description': formatted_description,
            'title': seo_title,
            'sections': {
                'about': formatted_sections[0] if len(formatted_sections) > 0 else "",
                'technical': formatted_sections[1] if len(formatted_sections) > 1 else "",
                'additional': formatted_sections[2] if len(formatted_sections) > 2 else ""
            }
        }
        
    except Exception as e:
        return {
            'error': f"Error generating SEO content: {str(e)}",
            'description': "",
            'title': "",
            'sections': {}
        }

# Routes
@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/social-media')
def social_media():
    return render_template('social_media.html')

@app.route('/seo')
def seo():
    return render_template('seo.html')

@app.route('/general')
def general():
    return render_template('general.html')

@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    text = request.json.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Create a temporary file for the audio
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    tts = gTTS(text=text, lang='en')
    tts.save(temp_file.name)
    
    return send_file(
        temp_file.name,
        mimetype='audio/mp3',
        as_attachment=True,
        download_name='speech.mp3'
    )

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a PNG, JPG, JPEG, or GIF'}), 400
    
    try:
        # Validate image
        if not validate_image(file.stream):
            return jsonify({'error': 'Invalid image file'}), 400
            
        # Save and process image
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            image = Image.open(filepath)
            alt_text = generate_alt_text(image)
            context = generate_context(alt_text)
            enhanced = enhance_context(context)
            caption = social_media_caption(context)
            seo_description = generate_seo_description(context, alt_text)
            sentiment = analyze_sentiment(seo_description['description'])

            
            return jsonify({
                'alt_text': alt_text,
                'context': context,
                'enhanced': enhanced,
                'caption': caption,
                'sentiment': sentiment,
                'seo_description': seo_description
            })
            
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
        
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
    
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)