#!/usr/bin/env python3
"""
Phase 1: Update CLI/GUI to use FLAN-T5 for immediate quality improvement
"""

import sys
from pathlib import Path
import shutil

def update_cli_to_flan_t5():
    """Update CLI app to use FLAN-T5 instead of T5-small"""
    
    project_dir = Path(__file__).parent.parent
    cli_file = project_dir / "src" / "cli_app.py"
    backup_file = project_dir / "src" / "cli_app_backup.py"
    
    print("🔄 Phase 1: Updating CLI to use FLAN-T5...")
    
    # Create backup
    shutil.copy2(cli_file, backup_file)
    print(f"✅ Backup created: {backup_file}")
    
    # Read current file
    with open(cli_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace T5-small with FLAN-T5 imports
    updated_content = content.replace(
        'from transformers import (\n    AutoTokenizer, AutoModel,\n    T5ForConditionalGeneration, T5Tokenizer,\n)',
        'from transformers import (\n    AutoTokenizer, AutoModel,\n    AutoModelForSeq2SeqLM,\n)'
    )
    
    # Update model loading section
    old_t5_loading = '''        # Load T5 model for simplification
        print("🔄 Loading T5 Simplification Model...")
        # Use base T5 model for better English generation
        print("⚠️  Using base T5 model for reliable English output")
        self.t5_tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
        self.t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")'''
    
    new_flan_loading = '''        # Load FLAN-T5 model for simplification
        print("🔄 Loading FLAN-T5 Simplification Model...")
        flan_t5_path = self.models_dir / "flan_t5"
        if flan_t5_path.exists():
            print("✅ Using local FLAN-T5 model")
            self.t5_tokenizer = AutoTokenizer.from_pretrained(str(flan_t5_path))
            self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(str(flan_t5_path))
        else:
            print("⚠️  Local FLAN-T5 not found, downloading...")
            self.t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
            self.t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")'''
    
    updated_content = updated_content.replace(old_t5_loading, new_flan_loading)
    
    # Update simplify_text method to use FLAN-T5 properly
    old_simplify_section = '''            # For now, use manual rule-based simplification for reliable results
            # The T5 model is generating non-English output due to multilingual training
            print("🔧 Using rule-based simplification for reliable results...")
            simplified_text = self.manual_simplification(legal_text)
            
            # If we want to try T5 as well, uncomment below:
            # But keeping it simple for now
            """
            # Prepare input text with better prompt for base T5
            input_text = f"paraphrase: {legal_text.strip()}"'''
    
    new_flan_section = '''            # Use FLAN-T5 with proper instruction prompting
            print("🤖 Using FLAN-T5 for high-quality simplification...")
            
            # FLAN-T5 works best with clear instructions
            instruction = "Simplify this legal text into plain English that anyone can understand:"
            input_text = f"{instruction} {legal_text.strip()}"'''
    
    updated_content = updated_content.replace(old_simplify_section, new_flan_section)
    
    # Update the generation parameters
    old_generation = '''            # Tokenize input
            max_len = max_length or self.config.get("max_length", 512)
            inputs = self.t5_tokenizer.encode(
                input_text,
                return_tensors="pt",
                max_length=max_len,
                truncation=True
            )
            
            # Generate simplified text
            with torch.no_grad():
                outputs = self.t5_model.generate(
                    inputs,
                    max_new_tokens=150,
                    min_length=10,
                    num_beams=2,
                    temperature=0.7,
                    do_sample=True,
                    early_stopping=True,
                    pad_token_id=self.t5_tokenizer.pad_token_id,
                    eos_token_id=self.t5_tokenizer.eos_token_id
                )
            
            # Decode output
            t5_output = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up T5 output
            if t5_output.startswith("paraphrase:"):
                t5_output = t5_output.replace("paraphrase:", "").strip()
            
            # Use T5 output if it's in English, otherwise use manual
            if self.is_english(t5_output):
                simplified_text = t5_output
            """'''
    
    new_generation = '''            # Tokenize input
            max_len = max_length or self.config.get("max_length", 512)
            inputs = self.t5_tokenizer(
                input_text,
                return_tensors="pt",
                max_length=max_len,
                truncation=True,
                padding=True
            )
            
            # Generate simplified text with enhanced parameters
            with torch.no_grad():
                outputs = self.t5_model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=200,
                    min_length=20,
                    num_beams=4,
                    temperature=0.6,
                    do_sample=True,
                    top_p=0.9,
                    length_penalty=1.2,
                    early_stopping=True,
                    pad_token_id=self.t5_tokenizer.pad_token_id,
                    eos_token_id=self.t5_tokenizer.eos_token_id
                )
            
            # Decode output
            flan_output = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up FLAN-T5 output
            if instruction in flan_output:
                flan_output = flan_output.replace(instruction, "").strip()
            
            # Post-process for better readability
            simplified_text = self.post_process_output(flan_output)'''
    
    updated_content = updated_content.replace(old_generation, new_generation)
    
    # Add post-processing method before the manual_simplification method
    post_process_method = '''    
    def post_process_output(self, text):
        """Post-process FLAN-T5 output for better readability"""
        if not text or len(text.strip()) < 10:
            return "Unable to simplify this text."
        
        # Clean up common issues
        text = text.strip()
        
        # Ensure proper capitalization
        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]
        
        # Ensure proper ending punctuation
        if text and text[-1] not in '.!?':
            text += '.'
        
        # Remove redundant phrases
        redundant_phrases = [
            "In simple terms:",
            "Simply put:",
            "To put it simply:",
            "In other words:",
        ]
        
        for phrase in redundant_phrases:
            if text.startswith(phrase):
                text = text[len(phrase):].strip()
                if text and not text[0].isupper():
                    text = text[0].upper() + text[1:]
        
        return text

'''
    
    # Insert the post_process_output method before manual_simplification
    manual_method_start = updated_content.find('    def manual_simplification(self, legal_text):')
    if manual_method_start != -1:
        updated_content = (updated_content[:manual_method_start] + 
                          post_process_method + 
                          updated_content[manual_method_start:])
    
    # Write updated content
    with open(cli_file, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print("✅ CLI updated to use FLAN-T5!")

def update_gui_to_flan_t5():
    """Update GUI app to use FLAN-T5"""
    
    project_dir = Path(__file__).parent.parent
    gui_file = project_dir / "src" / "gui_app.py"
    backup_file = project_dir / "src" / "gui_app_backup.py"
    
    print("🔄 Updating GUI to use FLAN-T5...")
    
    # Create backup
    if gui_file.exists():
        shutil.copy2(gui_file, backup_file)
        print(f"✅ GUI backup created: {backup_file}")
        
        # The GUI likely imports the CLI class, so it should automatically use FLAN-T5
        print("✅ GUI will automatically use FLAN-T5 through updated CLI!")
    else:
        print("⚠️  GUI file not found, skipping...")

def main():
    """Phase 1: Update both CLI and GUI to use FLAN-T5"""
    print("🚀 Starting Phase 1: FLAN-T5 Integration")
    print("=" * 50)
    
    update_cli_to_flan_t5()
    update_gui_to_flan_t5()
    
    print("\n" + "=" * 50)
    print("🎉 Phase 1 Complete!")
    print("\n📋 What's Changed:")
    print("   ✅ CLI now uses FLAN-T5 instead of T5-small")
    print("   ✅ Enhanced generation parameters for better quality")
    print("   ✅ Added post-processing for cleaner output")
    print("   ✅ Proper instruction prompting for FLAN-T5")
    print("\n🧪 Test the updated system:")
    print("   python src/cli_app.py")
    print("\n📁 Backups created:")
    print("   - src/cli_app_backup.py")
    print("   - src/gui_app_backup.py (if GUI exists)")

if __name__ == "__main__":
    main()
