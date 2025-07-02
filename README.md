Creative Writing Prompt Generator 📝
A sophisticated Streamlit web application that uses a fine-tuned GPT-2 model to generate unique and inspiring creative writing prompts. Perfect for writers, students, and anyone looking to spark their creativity!

✨ Features
Customizable Content: Select from various genres, themes, and tones
Advanced AI Controls: Fine-tune creativity with temperature, top-k, top-p, and length parameters
Random Generation: Get surprise prompts for unexpected inspiration
User-Friendly Interface: Clean, intuitive design with helpful tooltips
Copy-Friendly Output: Easy copying of generated prompts
Real-Time Generation: Fast prompt creation with loading indicators
🚀 Quick Start
Prerequisites
Python 3.7+
CUDA-compatible GPU (recommended) or CPU
Fine-tuned GPT-2 model
Installation
Clone the repository:
bash
git clone <your-repository-url>
cd creative-writing-prompt-generator
Install dependencies:
bash
pip install streamlit torch transformers
Prepare your model:
Train your GPT-2 model using train_model.py (if available)
Ensure your fine-tuned model is saved in: ./fine_tuned_writing_model/
The directory should contain:
config.json
pytorch_model.bin
tokenizer.json
vocab.json
merges.txt
Run the application:
bash
streamlit run app.py
Open your browser and navigate to http://localhost:8501
🎯 How to Use
Basic Usage
Select Genre: Choose from Fantasy, Sci-Fi, Mystery, Horror, Romance, and more
Pick a Theme: Select themes like Love, Betrayal, Redemption, Identity, etc.
Choose Tone (Optional): Set the mood with Mysterious, Humorous, Dark, Uplifting, etc.
Generate: Click "GENERATE PROMPT" to create your writing prompt
Copy: Use the copy button to easily transfer your prompt
Advanced Usage
Expand the "🔧 Advanced Options" section to fine-tune generation:

Temperature (0.1-1.5):
Low (0.1-0.5): More focused, coherent output
Medium (0.6-0.9): Balanced creativity and coherence
High (1.0-1.5): Very creative but potentially less coherent
Top-K Sampling (10-100):
Lower values: More focused vocabulary
Higher values: More diverse word choices
Top-P Sampling (0.1-1.0):
Lower values: More predictable output
Higher values: More varied and creative output
Max Length (50-200):
Controls the length of generated text
Random Generation
Click "🎲 Generate Random Prompt" for surprise combinations that use your current advanced settings.


🛠️ Technical Details
Model Architecture
Base Model: GPT-2 (Generative Pre-trained Transformer 2)
Framework: Hugging Face Transformers
Fine-tuning: Custom dataset of creative writing prompts
Device Support: CUDA GPU (preferred) or CPU fallback
Key Components
PromptGenerator Class:
Handles model loading and initialization
Manages prompt construction and generation
Supports both basic and advanced parameter control
Streamlit Interface:
Interactive parameter selection
Real-time generation with progress indicators
Session state management for consistent experience
Advanced Parameters:
Temperature control for creativity balance
Top-k and top-p sampling for output diversity
Configurable output length
🎨 Customization
Adding New Genres/Themes/Tones
Edit the lists in app.py:

python
genres = [
    "Any Genre", "Fantasy", "Sci-Fi", "Mystery", 
    "Your New Genre"  # Add here
]

themes = [
    "Any Theme", "Love", "Betrayal", 
    "Your New Theme"  # Add here
]

tones = [
    "Any Tone", "Mysterious", "Humorous", 
    "Your New Tone"  # Add here
]
Styling
Modify the CSS in the st.markdown() sections to customize appearance:

python
st.markdown(
    f'<div style="background-color: #your-color; ...">'
    f'<p style="font-size: 16px; ...">{prompt}</p>'
    f'</div>', 
    unsafe_allow_html=True
)
🔧 Troubleshooting
Common Issues
Model Not Found Error:
Ensure your model is in ./fine_tuned_writing_model/
Check that all required files are present
Verify file permissions
CUDA Out of Memory:
Reduce max_length parameter
Use CPU instead: Set device = torch.device("cpu")
Slow Generation:
Reduce max_length
Lower top_k value
Use GPU if available
Poor Quality Output:
Adjust temperature (try 0.7-0.9)
Fine-tune top_p (try 0.9-0.95)
Ensure model is properly trained
Performance Tips
GPU Usage: Enable CUDA for faster generation
Memory Management: Close other applications if running out of memory
Parameter Tuning: Start with default values and adjust gradually
📋 Requirements
txt
streamlit>=1.28.0
torch>=2.0.0
transformers>=4.30.0
🤝 Contributing
Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request
Development Setup
bash
# Clone and setup development environment
git clone <your-repo>
cd creative-writing-prompt-generator
pip install -r requirements.txt

# Make your changes
# Test thoroughly
streamlit run app.py

# Submit PR
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Hugging Face for the Transformers library
Streamlit for the excellent web framework
OpenAI for the GPT-2 architecture
PyTorch for the deep learning framework
📞 Support
If you encounter any issues or have questions:

Check the troubleshooting section above
Search existing issues in the repository
Create a new issue with detailed information:
Error messages
System specifications
Steps to reproduce
🔮 Future Enhancements
 Support for other model architectures (GPT-3, T5, etc.)
 Batch prompt generation
 Export functionality (PDF, Word, etc.)
 Prompt rating and favoriting system
 Multi-language support
 Integration with writing platforms
 Mobile-responsive design improvements
Happy Writing! 🖋️

Made with ❤️ using Streamlit and Transformers

