import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import random
import os

# Page configuration
st.set_page_config(
    page_title="Creative Writing Prompt Generator",
    page_icon="📝",
    layout="centered"
)

class PromptGenerator:
    def __init__(self, model_id):
        """Initialize the Prompt Generator."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self.tokenizer = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the tokenizer and model."""
        try:
            # Load directly from Hugging Face using the model_id
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_id)
            self.model = GPT2LMHeadModel.from_pretrained(self.model_id).to(self.device)
            self.model.eval()
            return True
            
        except Exception as e:
            # Provide a more specific error message if loading from HF fails
            st.error(f"Error loading model from Hugging Face '{self.model_id}': {str(e)}")
            st.warning("Please ensure the model ID is correct and accessible on Hugging Face.")
            return False
    
    def build_prompt(self, genre="", theme="", tone=""):
        """Construct the base prompt string based on user input."""
        prompt = "Create a"
        if tone:
            prompt += f" {tone}"
        if genre:
            prompt += f" {genre}"
        prompt += " story"
        if theme:
            prompt += f" about {theme}"
        prompt += "."
        return prompt
    
    def generate_prompt(self, genre="", theme="", tone=""):
        """Generate a creative writing prompt using the model with default parameters."""
        return self.generate_prompt_advanced(genre, theme, tone)
    
    def generate_prompt_advanced(self, genre="", theme="", tone="", temperature=0.8, top_k=50, top_p=0.95, max_length=80):
        """Generate a creative writing prompt with advanced parameters."""
        if not self.model or not self.tokenizer:
            return "Model not loaded properly."
        
        base_prompt = self.build_prompt(genre, theme, tone)
        
        try:
            input_ids = self.tokenizer.encode(base_prompt, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    num_return_sequences=1
                )
            
            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            result = generated_text[len(base_prompt):].strip()
            
            if result and not result.endswith(('.', '!', '?')):
                result += '.'
            
            final_prompt = base_prompt + " " + result.capitalize() if result else base_prompt
            return final_prompt
            
        except Exception as e:
            return f"Error generating prompt: {str(e)}"

# Initialize session state
if 'generator' not in st.session_state:
    st.session_state.generator = None
if 'current_prompt' not in st.session_state:
    st.session_state.current_prompt = "Click the button to get a new prompt!"

# Title and header
st.title("Creative Writing Prompt Generator")
st.markdown("---")

# Automatically load model on app start
#MODEL_PATH = "./fine_tuned_writing_model"
HF_MODEL_ID = "Anish6648/Creative-Writing-Prompt-Generator/fine_tuned_writing_model"
# Automatically load model on app start
if 'generator' not in st.session_state: # This condition might prevent re-initialization
    st.session_state.generator = None # Initialize to None if not present
if st.session_state.generator is None:
    with st.spinner(f"Loading model from Hugging Face ({HF_MODEL_ID})..."):
        # Ensure HF_MODEL_ID is passed directly here
        st.session_state.generator = PromptGenerator(HF_MODEL_ID)
        if st.session_state.generator.model is not None:
            st.success("Model loaded successfully from Hugging Face!")

# Check if model is loaded
if st.session_state.generator and st.session_state.generator.model is not None:
    
    # Genre selection
    st.subheader("Select Genre:")
    genres = [
        "Any Genre", "Fantasy", "Sci-Fi", "Mystery", "Horror", "Romance", 
        "Thriller", "Adventure", "Historical Fiction", "Coming-of-Age"
    ]
    selected_genre = st.selectbox("", genres, key="genre_select")
    
    # Theme selection
    st.subheader("Select Theme:")
    themes = [
        "Any Theme", "Love", "Betrayal", "Redemption", "Identity", "Survival", 
        "Hope", "Sacrifice", "Family", "Friendship", "Power", "Justice"
    ]
    selected_theme = st.selectbox("", themes, key="theme_select")
    
    # Tone selection (optional)
    st.subheader("Select Tone (Optional):")
    tones = [
        "Any Tone", "Mysterious", "Humorous", "Dark", "Uplifting", 
        "Suspenseful", "Whimsical", "Melancholic", "Inspiring"
    ]
    selected_tone = st.selectbox("", tones, key="tone_select")
    
    # Add an advanced options expander
    with st.expander("🔧 Advanced Options", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            temperature = st.slider(
                "Creativity (Temperature)", 
                min_value=0.1, 
                max_value=1.5, 
                value=0.8, 
                step=0.1,
                help="Higher values make output more creative but less coherent"
            )
            
            top_k = st.slider(
                "Top-K Sampling", 
                min_value=10, 
                max_value=100, 
                value=50, 
                step=10,
                help="Consider only top K most likely next words"
            )
        
        with col2:
            top_p = st.slider(
                "Top-P Sampling", 
                min_value=0.1, 
                max_value=1.0, 
                value=0.95, 
                step=0.05,
                help="Consider words that make up top P probability mass"
            )
            
            max_length = st.slider(
                "Max Length", 
                min_value=50, 
                max_value=200, 
                value=80, 
                step=10,
                help="Maximum length of generated text"
            )
    
    # Generate prompt button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("GENERATE PROMPT", type="primary", use_container_width=True):
            with st.spinner("Generating your creative prompt..."):
                # Prepare parameters
                genre = selected_genre if selected_genre != "Any Genre" else ""
                theme = selected_theme if selected_theme != "Any Theme" else ""
                tone = selected_tone if selected_tone != "Any Tone" else ""
                
                # Generate prompt with advanced parameters
                st.session_state.current_prompt = st.session_state.generator.generate_prompt_advanced(
                    genre.lower(), theme.lower(), tone.lower(),
                    temperature=temperature, top_k=top_k, top_p=top_p, max_length=max_length
                )
    
    # Display current prompt
    st.markdown("---")
    st.subheader("Your Creative Writing Prompt:")
    st.markdown(
        f'<div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4;">'
        f'<p style="font-size: 16px; margin: 0;">{st.session_state.current_prompt}</p>'
        f'</div>', 
        unsafe_allow_html=True
    )
    
    # Random prompt button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎲 Generate Random Prompt", use_container_width=True):
            with st.spinner("Creating a surprise prompt..."):
                # Random selections
                random_genre = random.choice([g.lower() for g in genres[1:]])  # Skip "Any Genre"
                random_theme = random.choice([t.lower() for t in themes[1:]])  # Skip "Any Theme"
                random_tone = random.choice([t.lower() for t in tones[1:]])    # Skip "Any Tone"
                
                # Use current advanced settings for random prompt too
                st.session_state.current_prompt = st.session_state.generator.generate_prompt_advanced(
                    random_genre, random_theme, random_tone,
                    temperature=temperature, top_k=top_k, top_p=top_p, max_length=max_length
                )
                st.rerun()
    
    # Copy to clipboard functionality
    st.markdown("---")
    st.markdown(
        """
        <style>
        .copy-button {
            background-color: #f0f2f6;
            border: 1px solid #d1d5db;
            border-radius: 5px;
            padding: 5px 10px;
            font-size: 12px;
            cursor: pointer;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    
    if st.button("📋 Copy Prompt to Clipboard"):
        st.code(st.session_state.current_prompt, language=None)
        st.info("Prompt displayed above - you can select and copy it!")
    
else:
    st.error("Failed to load model. Please ensure the model exists at: ./fine_tuned_writing_model")
    
    # Instructions
    st.markdown("---")
    st.subheader("Instructions:")
    st.markdown("""
    1. First, train your model using the `train_model.py` script
    2. Ensure your trained model is saved in the default directory: `./fine_tuned_writing_model`
    3. The app will automatically load the model on startup
    4. Select your preferred genre, theme, and tone
    5. Adjust advanced parameters if needed (temperature, top-k, top-p, max length)
    6. Click "GENERATE PROMPT" to create your writing prompt
    7. Use the random prompt generator for surprise inspiration!
    """)
    
    st.markdown("---")
    st.markdown("*Built with Streamlit and Transformers*")

# Sidebar with additional info
with st.sidebar:
    st.header("About")
    st.markdown("""
    This Creative Writing Prompt Generator uses a fine-tuned GPT-2 model 
    to create unique and inspiring writing prompts based on your preferences.
    
    **Features:**
    - Customizable genre, theme, and tone
    - Advanced parameter controls
    - Random prompt generation
    - Easy-to-use interface
    - Copy-friendly output
    """)
    
    st.header("Advanced Parameters")
    st.markdown("""
    - **Temperature**: Controls creativity vs coherence
    - **Top-K**: Limits vocabulary to top K likely words
    - **Top-P**: Uses nucleus sampling for variety
    - **Max Length**: Controls output length
    """)
    
    st.header("Tips")
    st.markdown("""
    - Try different combinations for varied results
    - Use the random generator for unexpected inspiration
    - Higher temperature = more creative but less coherent
    - Lower top-k/top-p = more focused output
    - Experiment with different tones to change the mood
    """)
