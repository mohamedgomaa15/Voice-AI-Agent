"""
Advanced Arabic text correction for Whisper STT errors
Fixes common misrecognitions in Egyptian Arabic voice commands
"""

import re
from difflib import SequenceMatcher
import unicodedata

class ArabicTextCorrector:
    def __init__(self):
        # Common Whisper misrecognitions for Egyptian Arabic commands
        self.word_corrections = {
            # Command verbs - CRITICAL
            'أبت': 'افتح',
            'ابت': 'افتح',
            'أفت': 'افتح',
            'افت': 'افتح',
            'أفتح': 'افتح',
            'إفتح': 'افتح',
            'افطح': 'افتح',
            'افتاح': 'افتح',
            'افتيح': 'افتح',
            
            'شغل': 'شغل',
            'شقل': 'شغل',
            'شكل': 'شغل',
            'شغال': 'شغل',
            
            'ابحث': 'ابحث',
            'إبحث': 'ابحث',
            'ابحت': 'ابحث',
            'ابحس': 'ابحث',
            
            'دور': 'دور',
            'دوار': 'دور',
            'دوور': 'دور',
            
            # Common words
            'تحكوقل': 'جوجل',
            'كوقل': 'جوجل',
            'قوقل': 'جوجل',
            'جوقل': 'جوجل',
            'كوكل': 'جوجل',
            'تحكوكل': 'جوجل',
            'غوغل': 'جوجل',
            
            'تنبيقه': 'تطبيق',
            'تنبيق': 'تطبيق',
            'تبيق': 'تطبيق',
            'طبيق': 'تطبيق',
            'تطبيقه': 'تطبيق',
            'التطبيق': 'تطبيق',
            'الطبيق': 'تطبيق',
            
            'يوتويب': 'يوتيوب',
            'يوتوب': 'يوتيوب',
            'يوتيب': 'يوتيوب',
            'يوطيوب': 'يوتيوب',
            'يوتوبي': 'يوتيوب',
            'اليوتيوب': 'يوتيوب',
            
            'نتفليكس': 'نتفليكس',
            'نتفلكس': 'نتفليكس',
            'نتفليك': 'نتفليكس',
            'نيتفليكس': 'نتفليكس',
            'النتفليكس': 'نتفليكس',
            
            'سبوتيفاي': 'سبوتيفاي',
            'سبوتفاي': 'سبوتيفاي',
            'سبوتيفي': 'سبوتيفاي',
            'اسبوتيفاي': 'سبوتيفاي',
            
            'انستغرام': 'انستقرام',
            'انستجرام': 'انستقرام',
            'انستكرام': 'انستقرام',
            'انستاغرام': 'انستقرام',
            'الانستقرام': 'انستقرام',
            
            'فيسبوك': 'فيسبوك',
            'فيس بوك': 'فيسبوك',
            'فيسبك': 'فيسبوك',
            'الفيسبوك': 'فيسبوك',
            
            'تويتر': 'تويتر',
            'توتر': 'تويتر',
            'تويتير': 'تويتر',
            'التويتر': 'تويتر',
            
            'واتساب': 'واتساب',
            'وتساب': 'واتساب',
            'واتس اب': 'واتساب',
            'واتسآب': 'واتساب',
            'الواتساب': 'واتساب',
            
            'تيك توك': 'تيك توك',
            'تيكتوك': 'تيك توك',
            'تك توك': 'تيك توك',
            'التيك توك': 'تيك توك',
        }
        
        # Phrase-level corrections (before word-level)
        self.phrase_corrections = {
            'أبت تحكوقل': 'افتح جوجل',
            'ابت تحكوقل': 'افتح جوجل',
            'أفتح جوقل': 'افتح جوجل',
            'افتح تنبيقه يوتيوب': 'افتح تطبيق يوتيوب',
            'افتح تنبيق يوتيوب': 'افتح تطبيق يوتيوب',
            'شغل نتفلكس': 'شغل نتفليكس',
            'دور على': 'دور على',
            'ابحث عن': 'ابحث عن',
        }
        
        # Fuzzy match database for app names
        self.app_names = [
            'جوجل', 'يوتيوب', 'نتفليكس', 'سبوتيفاي', 'انستقرام',
            'فيسبوك', 'تويتر', 'واتساب', 'تيك توك', 'تطبيق'
        ]
        
        # Command verbs
        self.command_verbs = ['افتح', 'شغل', 'ابحث', 'دور', 'العب', 'شاهد']
    
    def normalize_arabic(self, text):
        """Normalize Arabic text"""
        if not text:
            return text
        
        # Remove diacritics
        text = ''.join(c for c in text if not unicodedata.category(c).startswith('M'))
        
        # Normalize variations
        replacements = {
            'أ': 'ا', 'إ': 'ا', 'آ': 'ا',
            'ة': 'ه',
            'ى': 'ي',
            'ئ': 'ي',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def fuzzy_match(self, word, candidates, threshold=0.6):
        """Find closest match from candidates using fuzzy matching"""
        word_norm = self.normalize_arabic(word)
        best_match = None
        best_score = 0
        
        for candidate in candidates:
            candidate_norm = self.normalize_arabic(candidate)
            score = SequenceMatcher(None, word_norm, candidate_norm).ratio()
            
            if score > best_score and score >= threshold:
                best_score = score
                best_match = candidate
        
        return best_match, best_score
    
    def correct_word(self, word):
        """Correct a single word"""
        # Direct match
        if word in self.word_corrections:
            return self.word_corrections[word]
        
        # Normalized match
        word_norm = self.normalize_arabic(word)
        for wrong, correct in self.word_corrections.items():
            if self.normalize_arabic(wrong) == word_norm:
                return correct
        
        # Fuzzy match for app names
        match, score = self.fuzzy_match(word, self.app_names, threshold=0.6)
        if match and score > 0.6:
            return match
        
        # Fuzzy match for command verbs
        match, score = self.fuzzy_match(word, self.command_verbs, threshold=0.65)
        if match and score > 0.65:
            return match
        
        return word
    
    def correct_text(self, text, debug=False):
        """
        Correct Arabic text with multiple strategies
        """
        if not text or not isinstance(text, str):
            return text
        
        original = text
        
        # Step 1: Phrase-level corrections (most specific)
        for wrong_phrase, correct_phrase in self.phrase_corrections.items():
            if wrong_phrase in text:
                text = text.replace(wrong_phrase, correct_phrase)
                if debug:
                    print(f"  Phrase correction: '{wrong_phrase}' → '{correct_phrase}'")
        
        # Step 2: Word-level corrections
        words = text.split()
        corrected_words = []
        
        for word in words:
            corrected = self.correct_word(word)
            corrected_words.append(corrected)
            
            if debug and corrected != word:
                print(f"  Word correction: '{word}' → '{corrected}'")
        
        text = ' '.join(corrected_words)
        
        # Step 3: Pattern-based corrections
        # Fix common patterns like "افتح ال..." → "افتح ..."
        text = re.sub(r'(افتح|شغل|ابحث|دور)\s+ال(\w+)', r'\1 \2', text)
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        if debug and text != original:
            print(f"\n  Original: {original}")
            print(f"  Corrected: {text}")
        
        return text
    
    def add_correction(self, wrong, correct):
        """Add a new correction to the database"""
        self.word_corrections[wrong] = correct
        print(f"✓ Added correction: '{wrong}' → '{correct}'")
    
    def test_corrections(self):
        """Test the correction system"""
        test_cases = [
            ("أبت تحكوقل", "افتح جوجل"),
            ("افتح تنبيقه يوتويب", "افتح تطبيق يوتيوب"),
            ("شقل نتفلكس", "شغل نتفليكس"),
            ("ابحت عن فيديوهات", "ابحث عن فيديوهات"),
            ("افتح انستجرام", "افتح انستقرام"),
            ("دور على اغاني", "دور على اغاني"),
            ("شغل يوتوب", "شغل يوتيوب"),
            ("افتح الفيسبوك", "افتح فيسبوك"),
        ]
        
        print("\n" + "="*70)
        print("TESTING ARABIC TEXT CORRECTION")
        print("="*70)
        
        correct_count = 0
        for wrong, expected in test_cases:
            corrected = self.correct_text(wrong)
            is_correct = corrected == expected
            
            if is_correct:
                correct_count += 1
                status = "✓"
            else:
                status = "✗"
            
            print(f"\n{status} Input:    '{wrong}'")
            print(f"  Expected: '{expected}'")
            print(f"  Got:      '{corrected}'")
            if not is_correct:
                print(f"  ❌ MISMATCH")
        
        accuracy = correct_count / len(test_cases)
        print("\n" + "="*70)
        print(f"CORRECTION ACCURACY: {correct_count}/{len(test_cases)} = {accuracy:.1%}")
        print("="*70)


# Singleton instance
_corrector = None

def get_corrector():
    """Get or create the corrector instance"""
    global _corrector
    if _corrector is None:
        _corrector = ArabicTextCorrector()
    return _corrector

def correct_arabic_text(text, debug=False):
    """Convenience function to correct Arabic text"""
    corrector = get_corrector()
    return corrector.correct_text(text, debug=debug)


if __name__ == "__main__":
    corrector = ArabicTextCorrector()
    
    # Run tests
    corrector.test_corrections()
    
    # Interactive mode
    print("\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print("Enter Arabic text to correct (or 'exit' to quit):\n")
    
    while True:
        try:
            text = input("Input: ").strip()
            if text.lower() == 'exit':
                break
            
            if text:
                corrected = corrector.correct_text(text, debug=True)
                print(f"✓ Result: {corrected}\n")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")