#!/usr/bin/env python3
"""
GUI Application for LegalEase Project
Tkinter-based desktop application for legal text simplification
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import json
from pathlib import Path
import threading
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "data" / "models"

class LegalEaseApp:
    """Main GUI application class"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.setup_window()
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cpu")
        
        # Status variables
        self.model_loaded = False
        self.processing = False
        
        self.create_widgets()
        self.setup_layout()
        self.load_model_async()
    
    def setup_window(self):
        """Setup main window properties"""
        self.root.title("LegalEase - Legal Text Simplifier")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)
        
        # Set icon (if available)
        try:
            # You can add an icon file here
            pass
        except Exception:
            pass
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
    
    def create_widgets(self):
        """Create all GUI widgets"""
        
        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        
        # Header
        self.header_frame = ttk.Frame(self.main_frame)
        self.title_label = ttk.Label(
            self.header_frame,
            text="LegalEase - Legal Text Simplifier",
            font=("Arial", 18, "bold")
        )
        self.subtitle_label = ttk.Label(
            self.header_frame,
            text="Convert complex Indian legal texts into understandable English",
            font=("Arial", 10)
        )
        
        # Status bar
        self.status_frame = ttk.Frame(self.header_frame)
        self.status_label = ttk.Label(self.status_frame, text="Loading model...")
        self.progress_var = tk.StringVar(value="●")
        self.progress_label = ttk.Label(self.status_frame, textvariable=self.progress_var, foreground="orange")
        
        # Input section
        self.input_frame = ttk.LabelFrame(self.main_frame, text="Legal Text Input", padding="10")
        self.input_text = scrolledtext.ScrolledText(
            self.input_frame,
            height=12,
            width=80,
            wrap=tk.WORD,
            font=("Consolas", 11)
        )
        
        # Buttons frame
        self.buttons_frame = ttk.Frame(self.main_frame)
        
        self.simplify_button = ttk.Button(
            self.buttons_frame,
            text="Simplify Text",
            command=self.simplify_text_async,
            state=tk.DISABLED
        )
        
        self.clear_button = ttk.Button(
            self.buttons_frame,
            text="Clear All",
            command=self.clear_all
        )
        
        self.load_file_button = ttk.Button(
            self.buttons_frame,
            text="Load File",
            command=self.load_file
        )
        
        self.save_button = ttk.Button(
            self.buttons_frame,
            text="Save Result",
            command=self.save_result
        )
        
        # Output section
        self.output_frame = ttk.LabelFrame(self.main_frame, text="Simplified Text Output", padding="10")
        self.output_text = scrolledtext.ScrolledText(
            self.output_frame,
            height=12,
            width=80,
            wrap=tk.WORD,
            font=("Arial", 11),
            state=tk.DISABLED
        )
        
        # Examples section
        self.examples_frame = ttk.LabelFrame(self.main_frame, text="Quick Examples", padding="5")
        self.examples_combo = ttk.Combobox(
            self.examples_frame,
            values=self.get_example_texts(),
            state="readonly",
            width=70
        )
        self.load_example_button = ttk.Button(
            self.examples_frame,
            text="Load Example",
            command=self.load_example
        )
        
        # Processing indicator
        self.processing_label = ttk.Label(
            self.main_frame,
            text="",
            font=("Arial", 10),
            foreground="blue"
        )
    
    def setup_layout(self):
        """Setup widget layout"""
        # Main frame
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Header
        self.header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        self.title_label.grid(row=0, column=0, sticky="w")
        self.subtitle_label.grid(row=1, column=0, sticky="w")
        
        # Status
        self.status_frame.grid(row=2, column=0, sticky="w", pady=(5, 0))
        self.status_label.grid(row=0, column=0, sticky="w")
        self.progress_label.grid(row=0, column=1, sticky="w", padx=(10, 0))
        
        # Input section
        self.input_frame.grid(row=1, column=0, sticky="nsew", pady=5)
        self.input_text.grid(row=0, column=0, sticky="nsew")
        
        # Buttons
        self.buttons_frame.grid(row=2, column=0, sticky="ew", pady=5)
        self.simplify_button.grid(row=0, column=0, padx=(0, 5))
        self.clear_button.grid(row=0, column=1, padx=5)
        self.load_file_button.grid(row=0, column=2, padx=5)
        self.save_button.grid(row=0, column=3, padx=5)
        
        # Output section
        self.output_frame.grid(row=3, column=0, sticky="nsew", pady=5)
        self.output_text.grid(row=0, column=0, sticky="nsew")
        
        # Examples
        self.examples_frame.grid(row=4, column=0, sticky="ew", pady=5)
        self.examples_combo.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.load_example_button.grid(row=0, column=1)
        
        # Processing indicator
        self.processing_label.grid(row=5, column=0, sticky="ew", pady=5)
        
        # Configure weights for resizing
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(3, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        self.input_frame.grid_rowconfigure(0, weight=1)
        self.input_frame.grid_columnconfigure(0, weight=1)
        
        self.output_frame.grid_rowconfigure(0, weight=1)
        self.output_frame.grid_columnconfigure(0, weight=1)
        
        self.examples_frame.grid_columnconfigure(0, weight=1)
    
    def get_example_texts(self):
        """Get example legal texts"""
        return [
            "The plaintiff has filed a writ petition under Article 32 of the Constitution seeking mandamus against the respondent.",
            "The court granted an interim injunction restraining the defendant from proceeding with the construction.",
            "The appellant contends that the lower court erred in not considering the precedent established in the landmark judgment.",
            "The tribunal held that the petitioner had locus standi to challenge the notification issued by the authority.",
            "The judgment was delivered ex-parte as the respondent failed to appear despite proper service of notice.",
            "The court observed that the contract was ultra vires and hence null and void ab initio.",
            "The High Court issued a writ of certiorari quashing the order passed by the subordinate court.",
            "The petitioner seeks compensation for the illegal detention and violation of fundamental rights under habeas corpus."
        ]
    
    def animate_progress(self):
        """Animate loading progress"""
        if self.processing or not self.model_loaded:
            current = self.progress_var.get()
            if current == "●":
                self.progress_var.set("●●")
            elif current == "●●":
                self.progress_var.set("●●●")
            elif current == "●●●":
                self.progress_var.set("●")
            
            self.root.after(500, self.animate_progress)
    
    def load_model_async(self):
        """Load model asynchronously"""
        def load_model():
            try:
                self.animate_progress()
                
                # Try to load trained model first
                trained_model_dir = MODELS_DIR / "t5_legal_simplification_trained"
                if trained_model_dir.exists():
                    logger.info("Loading trained model...")
                    self.status_label.config(text="Loading trained model...")
                    self.tokenizer = T5Tokenizer.from_pretrained(str(trained_model_dir))
                    self.model = T5ForConditionalGeneration.from_pretrained(str(trained_model_dir))
                else:
                    # Load base T5 model
                    logger.info("Loading base T5 model...")
                    self.status_label.config(text="Loading base T5 model...")
                    model_dir = MODELS_DIR / "t5_simplification"
                    if model_dir.exists():
                        self.tokenizer = T5Tokenizer.from_pretrained(str(model_dir))
                        self.model = T5ForConditionalGeneration.from_pretrained(str(model_dir))
                    else:
                        # Download T5-small
                        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
                        self.model = T5ForConditionalGeneration.from_pretrained("t5-small")
                
                # Move to CPU and set eval mode
                self.model.to(self.device)
                self.model.eval()
                
                # Update UI
                self.root.after(0, self.on_model_loaded)
                
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.root.after(0, lambda: self.on_model_load_failed(str(e)))
        
        # Start loading in background thread
        thread = threading.Thread(target=load_model)
        thread.daemon = True
        thread.start()
    
    def on_model_loaded(self):
        """Called when model is successfully loaded"""
        self.model_loaded = True
        self.status_label.config(text="Model loaded successfully!")
        self.progress_label.config(text="✓", foreground="green")
        self.simplify_button.config(state=tk.NORMAL)
        messagebox.showinfo("Success", "Model loaded successfully! You can now simplify legal texts.")
    
    def on_model_load_failed(self, error_msg):
        """Called when model loading fails"""
        self.status_label.config(text="Failed to load model")
        self.progress_label.config(text="✗", foreground="red")
        messagebox.showerror("Error", f"Failed to load model:\\n{error_msg}\\n\\nPlease ensure the model is properly downloaded.")
    
    def simplify_text_async(self):
        """Simplify text asynchronously"""
        input_text = self.input_text.get("1.0", tk.END).strip()
        
        if not input_text:
            messagebox.showwarning("Warning", "Please enter some legal text to simplify.")
            return
        
        if not self.model_loaded:
            messagebox.showwarning("Warning", "Model is not loaded yet. Please wait.")
            return
        
        # Disable button and show processing
        self.simplify_button.config(state=tk.DISABLED)
        self.processing = True
        self.processing_label.config(text="Processing... Please wait.")
        self.animate_progress()
        
        def simplify():
            try:
                # Add T5 task prefix if not present
                if not input_text.startswith("simplify"):
                    formatted_text = f"simplify legal text: {input_text}"
                else:
                    formatted_text = input_text
                
                # Tokenize
                inputs = self.tokenizer(
                    formatted_text,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True
                ).to(self.device)
                
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs.input_ids,
                        max_length=200,
                        num_beams=3,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        early_stopping=True
                    )
                
                # Decode
                simplified_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Update UI
                self.root.after(0, lambda: self.on_simplification_complete(simplified_text))
                
            except Exception as e:
                logger.error(f"Simplification failed: {e}")
                self.root.after(0, lambda: self.on_simplification_failed(str(e)))
        
        # Start processing in background thread
        thread = threading.Thread(target=simplify)
        thread.daemon = True
        thread.start()
    
    def on_simplification_complete(self, simplified_text):
        """Called when simplification is complete"""
        self.processing = False
        self.processing_label.config(text="")
        self.progress_label.config(text="✓", foreground="green")
        
        # Update output
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", simplified_text)
        self.output_text.config(state=tk.DISABLED)
        
        # Re-enable button
        self.simplify_button.config(state=tk.NORMAL)
    
    def on_simplification_failed(self, error_msg):
        """Called when simplification fails"""
        self.processing = False
        self.processing_label.config(text="")
        self.progress_label.config(text="✗", foreground="red")
        
        # Re-enable button
        self.simplify_button.config(state=tk.NORMAL)
        
        messagebox.showerror("Error", f"Simplification failed:\\n{error_msg}")
    
    def clear_all(self):
        """Clear all text areas"""
        self.input_text.delete("1.0", tk.END)
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.config(state=tk.DISABLED)
        self.processing_label.config(text="")
    
    def load_example(self):
        """Load selected example"""
        selected = self.examples_combo.get()
        if selected:
            self.input_text.delete("1.0", tk.END)
            self.input_text.insert("1.0", selected)
    
    def load_file(self):
        """Load text from file"""
        file_path = filedialog.askopenfilename(
            title="Load Legal Text File",
            filetypes=[
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                self.input_text.delete("1.0", tk.END)
                self.input_text.insert("1.0", content)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file:\\n{e}")
    
    def save_result(self):
        """Save simplified result to file"""
        output_content = self.output_text.get("1.0", tk.END).strip()
        
        if not output_content:
            messagebox.showwarning("Warning", "No simplified text to save.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Simplified Text",
            defaultextension=".txt",
            filetypes=[
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"Original Legal Text:\\n")
                    f.write("="*50 + "\\n")
                    f.write(self.input_text.get("1.0", tk.END).strip() + "\\n\\n")
                    f.write("Simplified Text:\\n")
                    f.write("="*50 + "\\n")
                    f.write(output_content)
                
                messagebox.showinfo("Success", f"Result saved to: {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file:\\n{e}")
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

def main():
    """Main function"""
    print("🚀 Starting LegalEase GUI Application")
    
    # Check if models directory exists
    if not MODELS_DIR.exists():
        print("❌ Models directory not found.")
        print("Please run the setup scripts first:")
        print("   1. python scripts/download_datasets.py")
        print("   2. python src/model_setup.py")
        return False
    
    try:
        app = LegalEaseApp()
        app.run()
        return True
        
    except Exception as e:
        print(f"❌ Failed to start GUI application: {e}")
        return False

if __name__ == "__main__":
    main()
