import unicodedata
import re
def num_word(char):
    word_vn = "áàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ"
    if '0' <= char <= '9': return True
    if 'a' <= char <= 'z': return True
    if char in word_vn: return True
    if char == ' ': return True
    return False

def ref_data(txt: str):
    """Hàm làm sạch dữ liệu: Xóa ký tự lạ, giữ lại dấu câu cảm xúc"""
    if not isinstance(txt, str): return ""
    
    txt = unicodedata.normalize('NFC', txt)
    txt = txt.lower()
    clean_chars = []

    for char in txt:
        if num_word(char):
            clean_chars.append(char)
        elif char in ['!', '?','.',';',',']: 
            clean_chars.append(f' {char} ') 
        else:
            clean_chars.append(' ') 
            

    return " ".join("".join(clean_chars).split())

def normalize_list(raw_list):
    """Input: ['Tuy Nhiên'] -> Output: ['tuy_nhiên']"""
    return list(set([w.lower().replace(" ", "_") for w in raw_list]))

def create_inverted_vocab(raw_dict):
    """Đảo ngược từ điển để tra cứu nhanh"""
    inverted = {}
    for emo, words in raw_dict.items():
        for w in words:
            clean_w = w.lower().replace(" ", "_")
            inverted[clean_w] = emo
    return inverted

def normalize_vietnamese(text):
    """
    Hàm chuẩn hóa tiếng Việt:
    - Chuyển về chuẩn Unicode NFC.
    - Chuyển về chữ thường.
    - Xóa khoảng trắng thừa.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFC', text)

    text = text.lower()

    text = re.sub(r'[^\w\s?!.,]', ' ', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text