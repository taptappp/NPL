import json
import os
from .utils import normalize_list, create_inverted_vocab, ref_data as clean_text_helper

try:
    from underthesea import word_tokenize, sent_tokenize
except ImportError:
    # [FIX] Trả về list thay vì string để tránh lỗi logic vòng lặp bên dưới
    def word_tokenize(text, format=None): return text.split() 
    def sent_tokenize(text): return [text]


class SentimentEngine:
    _instance = None 

    def __new__(cls, *args, **kwargs):
        # [NOTE]Giữ nguyên để tối ưu bộ nhớ 
        if not cls._instance:
            cls._instance = super(SentimentEngine, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self, config_dir=None):
        if self.initialized:
            return

        if config_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_dir = os.path.join(base_dir, "config")

        self.load_resources(config_dir)
        self.initialized = True
        print(" [SentimentEngine] Resources loaded successfully!")

    def load_resources(self, path):
        emo_path = os.path.join(path, "emotions.json")
        cfg_path = os.path.join(path, "config.json")

        if not os.path.exists(emo_path): raise FileNotFoundError(f"Missing {emo_path}")
        if not os.path.exists(cfg_path): raise FileNotFoundError(f"Missing {cfg_path}")

        with open(emo_path, "r", encoding="utf-8") as f: raw_emotions = json.load(f)
        with open(cfg_path, "r", encoding="utf-8") as f: config = json.load(f)

        self.raw_emotions_keys = list(raw_emotions.keys())
        self.emotion_lookup = create_inverted_vocab(raw_emotions)
        
        # Load các từ nối, phủ định
        self.contrast_words = normalize_list(config["contrast"])
        self.negation_words = normalize_list(config["negation"])
        self.intensifiers = {
            "pre": normalize_list(config["intensifiers"]["pre"]),
            "post": normalize_list(config["intensifiers"]["post"]),
            "decrease": normalize_list(config["intensifiers"]["decrease"])
        }
        self.emoji_dict = config["emoji_dict"]
        self.inverse_map = config["inverse_map"]

    
    def _normalize_compound_tokens(self, tokens):
        normalized = []
        for t in tokens:
            if t in self.emotion_lookup:
                normalized.append(t)
            elif "_" in t:             
                parts = t.split("_")
                has_meaning = any(
                    p in self.emotion_lookup
                    or p in self.intensifiers["pre"]
                    or p in self.intensifiers["post"]
                    for p in parts
                )
                normalized.extend(parts if has_meaning else [t])
            else:
                normalized.append(t)
        return normalized

   
    def _calculate_segment_score(self, tokens, weight=1.0):
        local_score = {k: 0.0 for k in self.raw_emotions_keys}

        for i, token in enumerate(tokens):
            if token not in self.emotion_lookup:
                continue

            base_emo = self.emotion_lookup[token]
            score = 1.0
            multiplier = 1.0

            
            if i > 0:
                prev_token = tokens[i - 1]
                if prev_token in self.intensifiers["pre"]:
                    multiplier *= 2.0
                elif prev_token in self.intensifiers["decrease"]:
                    multiplier *= 0.5

            
            if i < len(tokens) - 1 and tokens[i + 1] in self.intensifiers["post"]:
                multiplier *= 2.0
           
            is_negated = False       
            for j in range(1, 3): 
                if i - j < 0: break
                prev = tokens[i - j]
                if prev in [",", ".", ";", "!", "?"]: break # Gặp dấu câu thì dừng
                if prev in self.negation_words:
                    is_negated = True
                    break
         
            final_emo = base_emo
            if is_negated and base_emo in self.inverse_map:
                final_emo = self.inverse_map[base_emo]
                score = 0.8 

            local_score[final_emo] += score * multiplier * weight

        return local_score
  
    def analyze_paragraph_rules(self, paragraph):
        if not isinstance(paragraph, str) or not paragraph.strip():
            return "TRUNG TÍNH", {k: 0.0 for k in self.raw_emotions_keys}

        sentences = sent_tokenize(paragraph)
        total_scores = {k: 0.0 for k in self.raw_emotions_keys}

        for sent in sentences:
            clean_sent = clean_text_helper(sent)

            try:              
                processed = word_tokenize(clean_sent, format="text").lower()
            except Exception:
                processed = clean_sent.lower()

            tokens = self._normalize_compound_tokens(processed.split())

            for emo, icons in self.emoji_dict.items():
                for icon in icons:
                    if icon in sent:
                        total_scores[emo] += 1.5

            
            contrast_index = next((i for i, t in enumerate(tokens) if t in self.contrast_words), -1)

            if contrast_index >= 0:
                segment_before = tokens[:contrast_index]
                segment_after = tokens[contrast_index + 1:]
                
                score_before = self._calculate_segment_score(segment_before, weight=0.5)
                score_after = self._calculate_segment_score(segment_after, weight=2.0)
                
                for k in total_scores:
                    total_scores[k] += score_before.get(k, 0) + score_after.get(k, 0)
            else:
                # Câu đơn
                score_segment = self._calculate_segment_score(tokens)
                for k in total_scores:
                    total_scores[k] += score_segment.get(k, 0)

     
        total_weight = sum(total_scores.values())
        if total_weight == 0:
            return "TRUNG TÍNH", {k: 0.0 for k in total_scores}

        percent_scores = {k: round(v / total_weight * 100, 2) for k, v in total_scores.items()}
        
        
        best_emotion = max(percent_scores, key=percent_scores.get).upper().replace("_", " ")

        return best_emotion, percent_scores