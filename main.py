if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)# requirements.txt
"""
fastapi==0.104.1
anthropic==0.7.8
uvicorn==0.24.0
python-multipart==0.0.6
pydantic==2.5.0
numpy==1.24.3
scikit-learn==1.3.0
python-dotenv==1.0.0
cors==0.2.1
"""

# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import anthropic
import os
from dotenv import load_dotenv
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

app = FastAPI(title="Babushka - Wisdom-Guided Relationship Advisor")

# Enable CORS for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Claude client
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

class AdviceRequest(BaseModel):
    situation: str
    selected_philosophies: List[str]

class AdviceResponse(BaseModel):
    advice: str
    sources: List[str]
    philosophy_weights: Dict[str, float]
    dominant_philosophy: str

class WisdomDatabase:
    def __init__(self):
        self.wisdom_texts = {
            "christianity": {
                "forgiveness": [
                    "Be kind to one another, tenderhearted, forgiving one another, as God in Christ forgave you. (Ephesians 4:32)",
                    "Then Peter came up and said to him, 'Lord, how often will my brother sin against me, and I forgive him? As many as seven times?' Jesus said to him, 'I do not say to you seven times, but seventy-seven times.' (Matthew 18:21-22)"
                ],
                "love": [
                    "Love is patient and kind; love does not envy or boast; it is not arrogant or rude. (1 Corinthians 13:4)",
                    "Above all, keep loving one another earnestly, since love covers a multitude of sins. (1 Peter 4:8)"
                ],
                "conflict": [
                    "If your brother sins against you, go and tell him his fault, between you and him alone. (Matthew 18:15)",
                    "A soft answer turns away wrath, but a harsh word stirs up anger. (Proverbs 15:1)"
                ]
            },
            "nichiren_buddhism": {
                "transformation": [
                    "The purpose of practicing Buddhism is to become happy and to make others happy through our human revolution.",
                    "When we change our inner determination, our entire environment changes."
                ],
                "compassion": [
                    "Buddhism teaches that all people possess the Buddha nature - inherent dignity and unlimited potential.",
                    "Chanting Nam-myoho-renge-kyo reveals our compassionate nature and that of others."
                ],
                "relationships": [
                    "Our relationships are mirrors reflecting our inner life condition.",
                    "By transforming ourselves, we create harmony in our relationships and environment."
                ]
            },
            "jungian_psychology": {
                "projection": [
                    "Everything that irritates us about others can lead us to an understanding of ourselves.",
                    "The meeting of two personalities is like the contact of two chemical substances: if there is any reaction, both are transformed."
                ],
                "individuation": [
                    "The privilege of a lifetime is being who you are.",
                    "Your vision becomes clear when you look into your heart. Who looks outside, dreams. Who looks inside, awakens."
                ],
                "shadow": [
                    "One does not become enlightened by imagining figures of light, but by making the darkness conscious.",
                    "The most terrifying thing is to accept oneself completely."
                ]
            }
        }
        
        # Create vectorizer for semantic matching
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self._build_search_index()
    
    def _build_search_index(self):
        """Build search index for semantic matching"""
        all_texts = []
        self.text_metadata = []
        
        for philosophy, categories in self.wisdom_texts.items():
            for category, texts in categories.items():
                for text in texts:
                    all_texts.append(text.lower())
                    self.text_metadata.append({
                        'philosophy': philosophy,
                        'category': category,
                        'text': text
                    })
        
        self.text_vectors = self.vectorizer.fit_transform(all_texts)
    
    def search_relevant_wisdom(self, situation: str, selected_philosophies: List[str], top_k: int = 5):
        """Find most relevant wisdom texts for the situation"""
        situation_vector = self.vectorizer.transform([situation.lower()])
        similarities = cosine_similarity(situation_vector, self.text_vectors).flatten()
        
        # Get top matches from selected philosophies
        relevant_texts = []
        for idx in np.argsort(similarities)[::-1]:
            metadata = self.text_metadata[idx]
            if metadata['philosophy'] in selected_philosophies and len(relevant_texts) < top_k:
                relevant_texts.append({
                    'text': metadata['text'],
                    'philosophy': metadata['philosophy'],
                    'category': metadata['category'],
                    'relevance': similarities[idx]
                })
        
        return relevant_texts

class BabushkaAdvisor:
    def __init__(self):
        self.wisdom_db = WisdomDatabase()
        
    def _calculate_philosophy_weights(self, situation: str, selected_philosophies: List[str], relevant_texts: List[Dict]):
        """Calculate weighted percentages for each philosophy"""
        weights = {phil: 0.0 for phil in selected_philosophies}
        
        # Base weights from relevant texts
        for text in relevant_texts:
            philosophy = text['philosophy']
            if philosophy in weights:
                weights[philosophy] += text['relevance']
        
        # Keyword-based adjustments
        situation_lower = situation.lower()
        keyword_bonuses = {
            'forgive': {'christianity': 0.3, 'nichiren_buddhism': 0.2},
            'angry': {'jungian_psychology': 0.3, 'nichiren_buddhism': 0.2},
            'conflict': {'christianity': 0.2, 'jungian_psychology': 0.3},
            'growth': {'nichiren_buddhism': 0.3, 'jungian_psychology': 0.2},
            'trust': {'christianity': 0.3, 'jungian_psychology': 0.2}
        }
        
        for keyword, bonuses in keyword_bonuses.items():
            if keyword in situation_lower:
                for phil, bonus in bonuses.items():
                    if phil in weights:
                        weights[phil] += bonus
        
        # Normalize to percentages
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: (v / total_weight) * 100 for k, v in weights.items()}
        else:
            # Equal distribution if no matches
            equal_weight = 100 / len(selected_philosophies)
            weights = {k: equal_weight for k in selected_philosophies}
        
        return weights
    
    def _create_advice_prompt(self, situation: str, relevant_texts: List[Dict], philosophy_weights: Dict[str, float]):
        """Create the prompt for Claude"""
        
        philosophy_descriptions = {
            'christianity': 'Christian wisdom emphasizing love, forgiveness, service, and Christ-like compassion',
            'nichiren_buddhism': 'Nichiren Buddhist principles of human revolution, compassionate action, and revealing Buddha nature',
            'jungian_psychology': 'Jungian psychology focusing on individuation, shadow work, and psychological integration'
        }
        
        prompt = f"""You are Babushka, a wise relationship advisor drawing from multiple philosophical traditions. 

SITUATION: {situation}

RELEVANT WISDOM TEXTS:
"""
        
        for text in relevant_texts:
            prompt += f"- [{text['philosophy'].upper()}] {text['text']}\n"
        
        prompt += f"""

PHILOSOPHY WEIGHTS (base your advice on these proportions):
"""
        for phil, weight in philosophy_weights.items():
            if weight > 0:
                prompt += f"- {philosophy_descriptions.get(phil, phil)}: {weight:.1f}%\n"
        
        prompt += """
Please provide compassionate, practical relationship advice that:
1. Integrates wisdom from the weighted philosophical traditions
2. Addresses the specific situation with empathy and understanding
3. Offers concrete, actionable guidance
4. Maintains hope and possibility for positive change

Keep the response focused, warm, and genuinely helpful. Avoid being preachy or overly academic.
"""
        
        return prompt
    
    async def get_advice(self, request: AdviceRequest) -> AdviceResponse:
        """Generate philosophical relationship advice"""
        
        # Find relevant wisdom texts
        relevant_texts = self.wisdom_db.search_relevant_wisdom(
            request.situation, 
            request.selected_philosophies
        )
        
        # Calculate philosophy weights
        philosophy_weights = self._calculate_philosophy_weights(
            request.situation, 
            request.selected_philosophies, 
            relevant_texts
        )
        
        # Create prompt for Claude
        prompt = self._create_advice_prompt(request.situation, relevant_texts, philosophy_weights)
        
        # Get advice from Claude
        try:
            message = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            advice_text = message.content[0].text
            
            # Extract sources from relevant texts
            sources = [text['text'] for text in relevant_texts[:3]]  # Top 3 sources
            
            # Find dominant philosophy
            dominant_philosophy = max(philosophy_weights.items(), key=lambda x: x[1])[0]
            
            return AdviceResponse(
                advice=advice_text,
                sources=sources,
                philosophy_weights=philosophy_weights,
                dominant_philosophy=dominant_philosophy
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating advice: {str(e)}")

# Initialize Babushka advisor
advisor = BabushkaAdvisor()

@app.post("/get-advice", response_model=AdviceResponse)
async def get_relationship_advice(request: AdviceRequest):
    """Get wisdom-guided relationship advice from Babushka"""
    return await advisor.get_advice(request)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/philosophies")
async def get_available_philosophies():
    """Get list of available philosophical traditions"""
    return {
        "philosophies": [
            {"id": "christianity", "name": "Christianity"},
            {"id": "nichiren_buddhism", "name": "Nichiren Buddhism"},
            {"id": "jungian_psychology", "name": "Jungian Psychology"}
        ]
    }

# Add endpoint to get philosophy distribution
@app.get("/philosophy-mix")
async def get_philosophy_mix():
    """Return the current philosophy distribution for frontend display"""
    philosophy_weights = {
        'christian': 0.15,
        'buddhist': 0.15,
        'jewish': 0.12,
        'islamic': 0.10,
        'hindu': 0.10,
        'confucian': 0.08,
        'indigenous': 0.08,
        'sikh': 0.06,
        'taoist': 0.06,
        'african_traditional': 0.05,
        'secular_humanist': 0.03,
        'stoic': 0.02
    }
    
    # Convert to percentages and add display names
    philosophy_display = {}
    for philosophy, weight in philosophy_weights.items():
        display_names = {
            'christian': 'Christian',
            'buddhist': 'Buddhist',
            'jewish': 'Jewish',
            'islamic': 'Islamic',
            'hindu': 'Hindu',
            'confucian': 'Confucian',
            'indigenous': 'Indigenous',
            'sikh': 'Sikh',
            'taoist': 'Taoist',
            'african_traditional': 'African Traditional',
            'secular_humanist': 'Secular Humanist',
            'stoic': 'Stoic'
        }
        
        philosophy_display[philosophy] = {
            'name': display_names[philosophy],
            'percentage': round(weight * 100, 1),
            'weight': weight
        }
    
    return {
        'philosophy_mix': philosophy_display,
        'total_traditions': len(philosophy_weights),
        'description': 'Babushka draws wisdom from diverse global traditions to provide balanced, culturally-aware relationship advice.'
    }
