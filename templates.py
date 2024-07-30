# Template
from dataclasses import dataclass


@dataclass
class Template:
    """
    We are interested in the few-/zero-shot setup in order to perform a task such as mapping an
    input x to an output y. In machine translation for example, x is a sentence written in a source
    language (say english) and y is the corresponding sentence in the language of interest (say french)

    For this specific example, an example template could be
    English: {x} French: {y}

    Believe that a template is defined by 3 components
    - The `prefix` : That is, everything that comes before the input (i.e. x)
    - The `middle` : Everything that comes between the input (i.e. x) and the output (i.e. y)
    - The `suffix`: That is, the part after the output y_{k} and the next input x_{k+1} in k-shot learning
    """

    header: str = ""
    prefix: str = "[src]: "
    middle: str = "\n[tgt]: "
    suffix: str = "\n\n"

    def get_prompt(self, demonstrations, example, start="", end=""):
        """
        Takes as input a list of demonstrations (i.e. few-shot examples) and an input in order
        to build the prompt to be fed to a LLM.
        Example :

        demonstrations  = [
            (
                "What is the capital of Cameroon?",
                "The capital of Cameroon is Yaoundé"
            ),
            (
                "What is the capital of France?",
                "The capital of France is Paris."
            )
        ]
        example = "What is the capital of Japan?"

        """
        prompt = self.header
        if demonstrations:
            for x, y in demonstrations:
                prompt += f"{self.prefix}"
                prompt += f"{start}{x}{end}"
                prompt += f"{self.middle}"
                prompt += f"{start}{y}{end}"
                prompt += f"{self.suffix}"
        prompt += f"{self.prefix}"
        prompt += f"{start}{example}{end}"
        prompt += f"{self.middle}"
        return prompt

    def copy(self):
        return Template(
            header=self.header,
            prefix=self.prefix,
            middle=self.middle,
            suffix=self.suffix,
        )


MAPPING_LANG_TO_TRANSLATION = {
    "English": "English",
    "French": "Français",
    "Swahili": "Kiswahili",
    "German": "Deutsch",
    "Wolof": "Wolof",
}

LEFT = {
    "English": "English sentence",
    "French": "Phrase en français",
    "German": "Satz auf Deutsch",
    "Swahili": "Sentensi ya Kiswahili",
    "Wolof": "Mbindum wolof",
}

RIGHT = {
    "English": "French translation",
    "French": "Traduction en français",
    "German": "Deutsche Übersetzung",
    "Swahili": "Tafsiri ya Kiswahili",
    "Wolof": "Mbinde buñu sirri si wolof",
}


def get_template(key: int, src: str, tgt: str) -> Template:
    header = ""
    prefix, middle, suffix = None, None, "\n\n"
    if key == 1:
        prefix = "Given the following source text: "
        middle = f"a good {tgt} translation is: "
    elif key == 2:
        prefix = f"Given the following source text in {src}: "
        middle = f"a good {tgt} translation is: "
    elif key == 3:
        prefix = "If the original version says "
        middle = f"then the {tgt} version should say:"
    elif key == 4:
        prefix = f"What is the {tgt} translation of the sentence: "
        middle = "?\n"
    elif key == 5:
        prefix = ""
        middle = f"= {tgt}: "
    elif key == 6:
        prefix = f"{src}: "
        middle = f"= {tgt}: "
    elif key == 7:
        prefix = f"{src}\n"
        middle = f"\ntranslates into\n{tgt}\n"
        suffix = "\n###\n"
    elif key == 8:
        prefix = f"{MAPPING_LANG_TO_TRANSLATION[src]}\n"
        middle = f"\ntranslates into\n{MAPPING_LANG_TO_TRANSLATION[tgt]}\n"
        suffix = "\n###\n"
    elif key == 9:
        prefix = f"{src}: "
        middle = f"\n{tgt}: "
        suffix = "\n###\n"
    elif key == 10:
        prefix = f"{MAPPING_LANG_TO_TRANSLATION[src]}: "
        middle = f"\n{MAPPING_LANG_TO_TRANSLATION[tgt]}: "
        suffix = "\n###\n"
    elif key == 11:
        prefix = f"{src} sentence\n"
        middle = f"\n{tgt} translation\n"
        suffix = "\n###\n"
    elif key == 12:
        prefix = f"{LEFT[src]}\n"
        middle = f"\n{RIGHT[tgt]}\n"
        suffix = "\n###\n"
    else:
        raise KeyError(
            f"The key {key} does not describe one of the ICL format that we support!"
        )

    return Template(header=header, prefix=prefix, middle=middle, suffix=suffix)
