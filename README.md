# INFOSYS Image Analysis Suite

<p align="center">
  <!-- Add your project logo here if you have one -->
  <h3 align="center">A Comprehensive AI-Powered Image Analysis Platform</h3>
</p>

## 🌟 Overview
INFOSYS Image Analysis Suite is a sophisticated web application that leverages cutting-edge AI technologies to provide comprehensive image analysis tools. The platform is designed to serve multiple use cases, from SEO optimization to social media content generation, making it an invaluable tool for digital marketers, content creators, and businesses.

## ✨ Key Features

### 🔍 General Analysis Tool
- **Smart Alt Text Generation**: Creates detailed, context-aware alt text
- **Contextual Analysis**: Provides in-depth image context analysis
- **Enhanced Descriptions**: Generates comprehensive image descriptions
- **Accessibility Features**: Built-in text-to-speech functionality
- **Quick Actions**: One-click copy functionality for all generated content

### 📊 SEO Description Generator
- **Product Descriptions**: Creates SEO-optimized product descriptions
- **Title Generation**: Generates keyword-rich product titles
- **Structured Content**:
  - Detailed "About" sections
  - Technical specifications
  - Feature highlights
- **Real-time Monitoring**: Character count tracking
- **Easy Export**: Quick copy functionality

### 📱 Social Media Tool
- **Platform-Optimized Content**: Generates platform-specific captions
- **Hashtag Generation**: Creates relevant, trending hashtags
- **Social Alt Text**: Specialized alt text for social platforms
- **Engagement Focus**: Creates engagement-optimized descriptions
- **Accessibility**: Text-to-speech and copy functionality

## 🛠️ Technology Stack

### Backend
- **Core**: Python with Flask framework
- **AI Models**:
  - OpenAI GPT-3.5 & GPT-4 for text generation
  - BLIP Image Captioning for visual analysis
  - NLTK for sentiment analysis
- **Additional Features**:
  - gTTS for text-to-speech conversion
  - CORS support for cross-origin requests
  - Secure file handling implementation

### Frontend
- **Core**: HTML5, CSS3, JavaScript
- **Styling**: Custom CSS with responsive design
- **Icons**: Font Awesome integration
- **UI/UX**: Modern, intuitive interface

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager
- OpenAI API key

### Setup Steps

1. **Clone the Repository**
    git clone https://github.com/yourusername/infosys-image-analysis.git

    ##### cd infosys-image-analysis

2. **Create and activate virtual environment**
    python -m venv venv
    #### On Windows:
       - venv\Scripts\activate
    #### On macOS/Linux:
       - source venv/bin/activate

3. **Install dependencies**
    pip install -r requirements.txt

4. **Set your OpenAI API key**
    - Create a .env file in the root directory and add your OpenAI API key:
      OPENAI_API_KEY=your_api_key_here

5. **Run the application**
    python app.py

6. **Access the application**
    Open your browser and navigate to http://localhost:5000

## 📁 Project Structure

#### ├── app.py # Main Flask application
#### ├── templates/ # HTML templates
#### │ ├── base.html # Base template
#### │ ├── general.html # General analysis page
#### │ ├── landing.html # Home page
#### │ ├── seo.html # SEO tool page
#### │ └── social_media.html # Social media tool page
#### │ └── medical_analysis.html # Medical Image Analysis tool page(to be added)
#### ├── static/ # Static assets
#### ├── uploads/ # Temporary image storage
#### ├── requirements.txt # Python dependencies
#### └── README.md # Project documentation

## 🔒 Security Features
- Secure file upload handling
- File type validation
- Size restrictions
- Temporary file cleanup
- CORS protection

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License
[To be added]

## 🙏 Acknowledgments
- OpenAI for their powerful GPT models
- Hugging Face for the innovative BLIP model
- NLTK team for their comprehensive sentiment analysis tools
- Flask team for the robust web framework
- Pillow team for efficient image processing
- gTTS (Google Text-to-Speech) for seamless audio generation
- CORS team for cross-origin resource sharing.


