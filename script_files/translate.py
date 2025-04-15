from googletrans import Translator

translator = Translator()

def translate_text(text, target_lang="ar"):
    try:
        translated = translator.translate(text, dest=target_lang)
        return translated.text
    except Exception as e:
        return f"[translation error: {str(e)}]"
