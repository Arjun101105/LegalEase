#!/usr/bin/env python3
"""
Enhanced Training Data for Better Simplification Quality
Creates high-quality training pairs that match ChatGPT-level simplification
"""

ENHANCED_LEGAL_SIMPLIFICATION_PAIRS = [
    # ChatGPT-style quality examples
    {
        "legal": "The plaintiff filed a writ petition under Article 32 of the Constitution seeking mandamus against the respondent for non-compliance with statutory obligations.",
        "simplified": "The plaintiff filed a petition under Article 32 of the Constitution, asking the court to issue an order (mandamus) directing the respondent to carry out their legal duties, which they had failed to do.",
        "category": "constitutional_law"
    },
    {
        "legal": "The appellant contends that the lower court erred in not considering the precedent established in the landmark judgment.",
        "simplified": "The person appealing the decision argues that the lower court made a mistake by not considering the precedent (previous similar case ruling) from an important judgment.",
        "category": "appellate_procedure"
    },
    {
        "legal": "The tribunal held that the petitioner had locus standi to challenge the notification issued by the authority.",
        "simplified": "The tribunal decided that the petitioner had the legal right (locus standi) to challenge the official announcement made by the authority.",
        "category": "administrative_law"
    },
    {
        "legal": "The court granted an interim injunction restraining the defendant from proceeding with the construction.",
        "simplified": "The court issued a temporary order (interim injunction) stopping the defendant from continuing with the construction work.",
        "category": "civil_procedure"
    },
    {
        "legal": "The judgment was delivered ex-parte as the respondent failed to appear despite proper service of notice.",
        "simplified": "The court gave its judgment without the respondent present (ex-parte) because they failed to appear in court even though they were properly notified.",
        "category": "court_procedure"
    },
    {
        "legal": "The High Court issued a writ of certiorari quashing the order passed by the subordinate court.",
        "simplified": "The High Court issued an order (writ of certiorari) canceling the decision made by the lower court.",
        "category": "judicial_review"
    },
    {
        "legal": "The party of the first part hereby covenants and agrees to indemnify and hold harmless the party of the second part.",
        "simplified": "The first party promises to protect and compensate the second party for any losses or damages.",
        "category": "contract_law"
    },
    {
        "legal": "Take notice that my client is constrained to initiate appropriate legal proceedings for recovery of the aforesaid amount.",
        "simplified": "Please be aware that my client will be forced to start legal action to recover the money mentioned earlier.",
        "category": "legal_notice"
    },
    {
        "legal": "The court observed that the contract was ultra vires and hence null and void ab initio.",
        "simplified": "The court noted that the contract was beyond legal authority (ultra vires) and therefore invalid from the beginning (ab initio).",
        "category": "contract_validity"
    },
    {
        "legal": "The matter is sub judice and any discussion thereof would amount to contempt of court.",
        "simplified": "The case is currently under court consideration (sub judice), and discussing it could be considered contempt of court.",
        "category": "court_etiquette"
    }
]

# Enhanced prompt templates for better T5 training
ENHANCED_PROMPT_TEMPLATES = [
    "Simplify this legal text while keeping key terms with explanations: {legal_text}",
    "Rewrite in plain English with legal terms explained in parentheses: {legal_text}",
    "Make this legal text understandable while preserving important legal concepts: {legal_text}",
    "Convert to simple language with brief explanations of legal terms: {legal_text}",
    "Explain this legal text in everyday language: {legal_text}"
]

def create_enhanced_training_dataset():
    """Create enhanced training dataset with better quality pairs"""
    enhanced_pairs = []
    
    for pair in ENHANCED_LEGAL_SIMPLIFICATION_PAIRS:
        # Add multiple variations with different prompts
        for template in ENHANCED_PROMPT_TEMPLATES:
            enhanced_pairs.append({
                "input": template.format(legal_text=pair["legal"]),
                "target": pair["simplified"],
                "category": pair["category"]
            })
    
    return enhanced_pairs

if __name__ == "__main__":
    pairs = create_enhanced_training_dataset()
    print(f"Created {len(pairs)} enhanced training pairs")
    for i, pair in enumerate(pairs[:3]):
        print(f"\nPair {i+1}:")
        print(f"Input: {pair['input']}")
        print(f"Target: {pair['target']}")
